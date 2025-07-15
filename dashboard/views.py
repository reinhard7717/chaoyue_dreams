# dashboard/views.py
import asyncio
import json
from asgiref.sync import async_to_sync
from django.db.models import Max, F, Q, OuterRef, Subquery
from collections import OrderedDict # 导入 OrderedDict
from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger
from dao_manager.tushare_daos.strategies_dao import StrategiesDAO
from dao_manager.tushare_daos.user_dao import UserDAO
from django.shortcuts import render
from django.contrib.auth.decorators import login_required
from rest_framework import generics, viewsets
from rest_framework.permissions import IsAuthenticated

from django.core.serializers.json import DjangoJSONEncoder
from stock_models.stock_analytics import TrendFollowStrategySignalLog, TrendFollowStrategyState
from stock_models.stock_basic import StockInfo
from users.models import FavoriteStock
from utils.websockets import send_update_to_user_sync
from .serializers import StockInfoSerializer, FavoriteStockSerializer
from itertools import chain
import logging # 导入 logging

logger = logging.getLogger('dashboard') # 获取 logger 实例
target_queue = 'dashboard'

# 内部辅助函数，用于处理所有策略列表页的通用逻辑
def _render_strategy_list_page(request, base_queryset, page_title, template_name):
    """
    一个通用的辅助函数，用于渲染策略列表页面。
    """
    print(f"--- [View] 开始渲染页面: {page_title} ---")
    
    ordered_queryset = base_queryset
    print("--- [View] DAO已提供完整数据，跳过视图层注解 ---")

    paginator = Paginator(ordered_queryset, 25)  # 每页显示25条
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)

    context = {
        'page_obj': page_obj,
        'total_count': paginator.count,
        'page_title': page_title,
    }
    return render(request, template_name, context)

# --- 页面视图 ---
@login_required
def dashboard_view(request):
    user_dao = UserDAO()
    """渲染主控台页面"""
    get_favorites_async = user_dao.get_user_favorites
    user_favorites = async_to_sync(get_favorites_async)(request.user.id)
    initial_favorites_data = [
        {
            'id': fav.id,
            'code': fav.stock.stock_code,
            'name': fav.stock.stock_name,
            "current_price": None,
            "high_price": None,
            "low_price": None,
            "open_price": None,
            "prev_close_price": None,
            "trade_time": None,
            'volume': None,
            'change_percent': None,
            'signal': None,
        } 
        for fav in user_favorites if fav.stock
    ]
    initial_favorites_json_string = json.dumps(initial_favorites_data, cls=DjangoJSONEncoder)
    context = {
        'initial_favorites_json': initial_favorites_json_string,
        'user_favorites': user_favorites, # 新增此行
    }
    return render(request, 'dashboard/home.html', context)

def get_playbook_priority(playbook_name):
    """
    【V2.0 修正版】根据剧本名称中的关键字为其分配优先级。
    此函数的判断逻辑现在与模板中的样式逻辑完全一致，确保排序正确。
    返回值越小，优先级越高。
    """
    # 安全检查，防止 playbook_name 不是字符串类型导致错误
    if not isinstance(playbook_name, str):
        return 99 # 返回一个很大的值，使其排在最后

    playbook_name_upper = playbook_name.upper()
    
    # ▼▼▼【代码修改】: 统一判断逻辑 ▼▼▼
    # 优先级 0: 火箭信号 (与模板逻辑一致)
    if 'ROCKET' in playbook_name_upper or '火箭' in playbook_name:
        return 0
    
    # 优先级 1: 王牌信号 (修正为与模板逻辑一致，检查'王牌')
    if 'BREAKOUT_TRIGGER_SCORE' in playbook_name_upper or '王牌' in playbook_name:
        return 1
        
    # 优先级 2: 专家信号 (修正为与模板逻辑一致，检查'专家')
    if '专家' in playbook_name:
        return 2
        
    # 优先级 4: 加分项 (逻辑保持不变)
    if '【加分】' in playbook_name or '资金流入' in playbook_name:
        return 4
        
    # 优先级 5: 基础形态 (逻辑保持不变)
    if 'MA20' in playbook_name_upper or '均线' in playbook_name:
        return 5
        
    # 默认优先级
    return 3
    # ▲▲▲【代码修改结束】▲▲▲

@login_required
def trend_following_list(request):
    """
    【V117.22 终极修正版】
    - 核心修正: 在准备“剧本筛选器”数据时，复用已经过 .annotate() 处理的 QuerySet，
                确保了 'latest_sell_time' 虚拟字段的存在。
    - 收益: 解决了 FieldError 崩溃问题，使页面能够正常加载，同时保持了V117.21版本
            带来的巨大性能优势。
    """
    print("--- [View] 开始渲染策略状态监控中心 (trend_following_list) V117.22 ---")

    # 步骤1: 定义基础数据集
    latest_buy_signals = TrendFollowStrategySignalLog.objects.filter(
        entry_signal=True
    ).values('stock_id').annotate(
        latest_buy_id=Max('id')
    )
    latest_buy_ids = [item['latest_buy_id'] for item in latest_buy_signals]
    base_queryset = TrendFollowStrategySignalLog.objects.filter(id__in=latest_buy_ids)

    # 步骤2: 定义子查询
    latest_sell_time_subquery = TrendFollowStrategySignalLog.objects.filter(
        stock_id=OuterRef('stock_id'),
        exit_signal_code__gt=0
    ).order_by('-trade_time').values('trade_time')[:1]

    # 步骤3: 注解QuerySet
    annotated_queryset = base_queryset.annotate(
        latest_sell_time=Subquery(latest_sell_time_subquery)
    )

    # 步骤4: 定义持仓条件
    holding_condition = Q(latest_sell_time__isnull=True) | Q(latest_sell_time__lt=F('trade_time'))
    
    # 步骤5: 筛选出持仓中的QuerySet
    held_queryset = annotated_queryset.filter(holding_condition)

    # 步骤6: (可选) 剧本筛选
    selected_playbooks = request.GET.getlist('playbooks')
    final_queryset = held_queryset # 先定义一个最终的QuerySet
    if selected_playbooks:
        for playbook in selected_playbooks:
            final_queryset = final_queryset.filter(triggered_playbooks__contains=playbook)
    
    # 步骤7: 排序
    final_queryset = final_queryset.select_related('stock').order_by('-trade_time', '-entry_score')

    # 步骤8: 分页
    paginator = Paginator(final_queryset, 25)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)

    # 步骤9: 格式化分页后的数据
    final_list_for_template = []
    for log in page_obj.object_list:
        final_list_for_template.append({
            'stock': log.stock,
            'latest_trade_time': log.trade_time,
            'latest_score': log.entry_score,
            'last_buy_time': log.trade_time,
            'last_sell_time': log.latest_sell_time,
            'active_playbooks': sorted(log.triggered_playbooks, key=get_playbook_priority),
            'strategy_names': [log.strategy_name],
            'stable_platform_price': log.stable_platform_price,
        })

    # ▼▼▼【代码修改 V117.22】: 复用已注解并过滤的 held_queryset ▼▼▼
    # 准备筛选器中的剧本列表
    # 我们直接在已经筛选出“持仓中”股票的 held_queryset 上获取剧本列表
    # 这样既能保证 'latest_sell_time' 字段存在，又能减少不必要的数据查询
    all_playbook_lists = held_queryset.values_list('triggered_playbooks', flat=True)
    # ▲▲▲【代码修改 V117.22】▲▲▲
    
    unique_playbooks = sorted(
        list(set(chain.from_iterable(p for p in all_playbook_lists if p))), 
        key=get_playbook_priority
    )

    context = {
        'page_title': '策略状态监控中心',
        'items_for_display': final_list_for_template,
        'page_obj': page_obj,
        'total_count': paginator.count,
        'all_playbooks': unique_playbooks,
        'selected_playbooks': selected_playbooks,
    }

    return render(request, 'dashboard/trend_following_list.html', context)

@login_required
def fav_trend_following_list(request):
    """
    【V135.0 战情显示一致性版】
    - 核心革命: 修正了后端视图的数据准备逻辑。
    - 新逻辑:
      1. 当一个头寸不再被持有时，无论原因(告警或解除)，都强制填充 item['sell_info']。
      2. 为“警报解除”的状态提供了更明确的文本 '空仓等待 (风险已解除)'。
    - 收益: 为前端模板提供了完整、无歧义的数据，从根源上解决了显示矛盾的问题。
    """
    print("--- [View] 开始渲染自选股持仓监控 (fav_trend_following_list) V135.0 ---")
    user_dao = UserDAO()
    
    user_favorites = async_to_sync(user_dao.get_user_favorites)(request.user.id)
    fav_codes = [fav.stock.stock_code for fav in user_favorites if fav.stock]
    fav_id_map = {fav.stock.stock_code: fav.id for fav in user_favorites if fav.stock}

    if not fav_codes:
        return render(request, 'dashboard/fav_trend_following_list.html', {
            'page_title': '自选股持仓监控',
            'page_obj': None,
            'total_count': 0,
        })

    latest_buy_signals = TrendFollowStrategySignalLog.objects.filter(
        stock__stock_code__in=fav_codes,
        entry_signal=True
    ).values('stock_id').annotate(
        latest_buy_id=Max('id')
    )
    latest_buy_ids = [item['latest_buy_id'] for item in latest_buy_signals]
    buy_logs = TrendFollowStrategySignalLog.objects.filter(id__in=latest_buy_ids).select_related('stock')
    buy_logs_map = {log.stock_id: log for log in buy_logs}

    latest_sell_signals = TrendFollowStrategySignalLog.objects.filter(
        stock__stock_code__in=fav_codes,
        entry_signal=False
    ).values('stock_id').annotate(
        latest_sell_id=Max('id')
    )
    latest_sell_ids = [item['latest_sell_id'] for item in latest_sell_signals]
    sell_logs = TrendFollowStrategySignalLog.objects.filter(id__in=latest_sell_ids).select_related('stock')
    sell_logs_map = {log.stock_id: log for log in sell_logs}

    processed_list = []
    for fav_stock_code in fav_codes:
        stock_obj = next((fav.stock for fav in user_favorites if fav.stock.stock_code == fav_stock_code), None)
        if not stock_obj: continue

        buy_log = buy_logs_map.get(stock_obj.id)
        sell_log = sell_logs_map.get(stock_obj.id)

        item = {
            'stock': stock_obj,
            'favorite_id': fav_id_map.get(fav_stock_code),
            'buy_info': None,
            'sell_info': None,
            'latest_trade_time': None,
            'latest_score': 0,
            'active_playbooks': [],
            'swing_status': '等待建仓',
            'status_class': 'status-wait',
            'sort_priority': 3,
            
        }

        if buy_log:
            is_holding = sell_log is None or sell_log.trade_time < buy_log.trade_time
            
            item['latest_trade_time'] = buy_log.trade_time
            item['latest_score'] = buy_log.entry_score
            item['active_playbooks'] = sorted(buy_log.triggered_playbooks_cn or buy_log.triggered_playbooks, key=get_playbook_priority)
            item['buy_info'] = {
                'time': buy_log.trade_time,
                'strategy_name': buy_log.strategy_name,
                'time_level': buy_log.timeframe,
                'stable_platform_price': buy_log.stable_platform_price,
            }

            if is_holding:
                item['swing_status'] = '持仓观察'
                item['status_class'] = 'status-holding'
                item['sort_priority'] = 2
            else: # 已平仓 (无论是告警还是解除)
                # 【核心修正】无论如何，只要不是持仓状态，就填充sell_info
                item['sell_info'] = {
                    'time': sell_log.trade_time,
                    'strategy_name': sell_log.strategy_name,
                    'time_level': sell_log.timeframe,
                    'severity_level': sell_log.exit_severity_level,
                    'reason': sell_log.exit_signal_reason or "风险预警",
                    'code': sell_log.exit_signal_code,
                }
                
                if sell_log.exit_signal_code > 0:
                    level = sell_log.exit_severity_level
                    if level == 1:
                        item['swing_status'] = '一级预警'
                        item['status_class'] = 'status-alert-level-1'
                    elif level == 3:
                        item['swing_status'] = '三级警报'
                        item['status_class'] = 'status-alert-level-3'
                    else:
                        item['swing_status'] = '二级警报'
                        item['status_class'] = 'status-alert-level-2'
                    item['sort_priority'] = 1
                else:
                    # 【核心修正】为解除状态提供更明确的文本
                    item['swing_status'] = '空仓等待 (风险已解除)'
                    item['status_class'] = 'status-wait'
                    item['sort_priority'] = 3
        
        elif sell_log and sell_log.exit_signal_code > 0:
            item['latest_trade_time'] = sell_log.trade_time
            item['sell_info'] = {
                'time': sell_log.trade_time,
                'strategy_name': sell_log.strategy_name,
                'time_level': sell_log.timeframe,
                'severity_level': sell_log.exit_severity_level,
                'reason': sell_log.exit_signal_reason or "风险预警",
                'code': sell_log.exit_signal_code,
            }
            item['swing_status'] = '空仓预警'
            item['status_class'] = 'status-alert-level-2'
            item['sort_priority'] = 4

        processed_list.append(item)

    processed_list.sort(key=lambda x: (
        x['sort_priority'],
        -(x['latest_trade_time'].timestamp() if x['latest_trade_time'] else 0)
    ))

    paginator = Paginator(processed_list, 25)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)

    context = {
        'page_title': '自选股持仓监控',
        'page_obj': page_obj,
        'total_count': len(processed_list),
    }
    return render(request, 'dashboard/fav_trend_following_list.html', context)

# --- DRF API 视图 ---

class StockSearchView(generics.ListAPIView):
    serializer_class = StockInfoSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        query = self.request.query_params.get('q', None)
        if query:
            return StockInfo.objects.filter(
                Q(stock_code__icontains=query) | Q(stock_name__icontains=query)
            )[:10]
        return StockInfo.objects.none()

class FavoriteStockViewSet(viewsets.ModelViewSet):
    serializer_class = FavoriteStockSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        return FavoriteStock.objects.filter(user=self.request.user).select_related('stock')

    def perform_create(self, serializer):
        favorite = serializer.save(user=self.request.user)
        payload = {
            'id': favorite.id,
            'code': favorite.stock.stock_code,
            'name': favorite.stock.stock_name,
            "current_price": None,
            "high_price": None,
            "low_price": None,
            "open_price": None,
            "prev_close_price": None,
            "trade_time": None,
            'volume': None,
            'change_percent': None,
            'signal': None,
        }
        send_update_to_user_sync(
            user_id=self.request.user.id,
            sub_type='favorite_added_with_data',
            payload=payload
        )
        try:
            from tasks.tushare.stock_tasks import fetch_data_for_new_favorite
            task_args = (
                self.request.user.id,
                favorite.stock.stock_code,
                favorite.id
            )
            
            fetch_data_for_new_favorite.apply_async(
                args=task_args,
                queue=target_queue,
            )
            logger.info(f"已将后台任务发送到队列 '{target_queue}' 为用户 {self.request.user.id} 的新自选股 {favorite.stock.stock_code} 获取数据")
        except Exception as task_error:
            logger.error(f"触发后台任务 fetch_data_for_new_favorite 时出错: {task_error}", exc_info=True)
    
    def perform_destroy(self, instance):
        user_id = instance.user.id
        instance.delete()
        updated_list = self._get_formatted_favorites(user_id)
        send_update_to_user_sync(
            user_id=user_id,
            sub_type='favorites_update',
            payload=updated_list
        )
    
    def _get_formatted_favorites(self, user_or_id):
        if isinstance(user_or_id, int):
            favorites = FavoriteStock.objects.filter(user_id=user_or_id).select_related('stock').order_by('added_at')
        else:
             favorites = FavoriteStock.objects.filter(user=user_or_id).select_related('stock').order_by('added_at')
        return [
            {
                'id': fav.id,
                'code': fav.stock.stock_code,
                'name': fav.stock.stock_name,
                "current_price": None,
                "high_price": None,
                "low_price": None,
                "open_price": None,
                "prev_close_price": None,
                "trade_time": None,
                'volume': None,
                'change_percent': None,
                'signal': None,
            } 
            for fav in favorites
        ]
