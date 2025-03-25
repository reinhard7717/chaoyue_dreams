"""
任务管理视图
提供任务管理的Web界面
"""
import logging
from django.shortcuts import render, redirect
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.utils.translation import gettext_lazy as _
from users.models import FavoriteStock

from .index_tasks import manual_refresh_all_index_data
from .fund_flow_tasks import manual_refresh_all_fund_flow_data
from .stock_tasks import manual_refresh_stock_data, manual_refresh_all_favorites_data
from .datacenter_tasks import manual_refresh_all_datacenter_data
from .stock_pool_tasks import manual_refresh_all_stock_pools
from .strategy_tasks import manual_calculate_stock_strategy, manual_calculate_all_favorites_strategy

logger = logging.getLogger(__name__)

@login_required
def task_dashboard(request):
    """
    任务管理仪表盘
    显示所有可执行的任务
    """
    # 获取自选股列表
    favorite_stocks = FavoriteStock.objects.filter(user=request.user).order_by('stock_name')
    
    context = {
        'favorite_stocks': favorite_stocks,
    }
    
    return render(request, 'tasks/dashboard.html', context)

@login_required
def refresh_stock_data(request, stock_code):
    """
    刷新单个股票数据
    """
    logger.info(f"用户 {request.user.username} 手动触发刷新股票 {stock_code} 数据")
    
    # 异步执行任务
    manual_refresh_stock_data.delay(stock_code)
    
    messages.success(request, _(f'已成功触发刷新股票 {stock_code} 数据的任务，请等待执行完成'))
    return redirect('task_dashboard')

@login_required
def refresh_all_favorites_data(request):
    """
    刷新所有自选股数据
    """
    logger.info(f"用户 {request.user.username} 手动触发刷新所有自选股数据")
    
    # 异步执行任务
    manual_refresh_all_favorites_data.delay()
    
    messages.success(request, _('已成功触发刷新所有自选股数据的任务，请等待执行完成'))
    return redirect('task_dashboard')

@login_required
def refresh_all_index_data(request):
    """
    刷新所有指数数据
    """
    logger.info(f"用户 {request.user.username} 手动触发刷新所有指数数据")
    
    # 异步执行任务
    manual_refresh_all_index_data.delay()
    
    messages.success(request, _('已成功触发刷新所有指数数据的任务，请等待执行完成'))
    return redirect('task_dashboard')

@login_required
def refresh_all_fund_flow_data(request):
    """
    刷新所有资金流向数据
    """
    logger.info(f"用户 {request.user.username} 手动触发刷新所有资金流向数据")
    
    # 异步执行任务
    manual_refresh_all_fund_flow_data.delay()
    
    messages.success(request, _('已成功触发刷新所有资金流向数据的任务，请等待执行完成'))
    return redirect('task_dashboard')

@login_required
def refresh_all_datacenter_data(request):
    """
    刷新所有数据中心数据
    """
    logger.info(f"用户 {request.user.username} 手动触发刷新所有数据中心数据")
    
    # 异步执行任务
    manual_refresh_all_datacenter_data.delay()
    
    messages.success(request, _('已成功触发刷新所有数据中心数据的任务，请等待执行完成'))
    return redirect('task_dashboard')

@login_required
def refresh_all_stock_pools(request):
    """
    刷新所有股票池
    """
    logger.info(f"用户 {request.user.username} 手动触发刷新所有股票池")
    
    # 异步执行任务
    manual_refresh_all_stock_pools.delay()
    
    messages.success(request, _('已成功触发刷新所有股票池的任务，请等待执行完成'))
    return redirect('task_dashboard')

@login_required
def calculate_stock_strategy(request, stock_code):
    """
    计算单个股票策略
    """
    logger.info(f"用户 {request.user.username} 手动触发计算股票 {stock_code} 策略")
    
    # 异步执行任务
    manual_calculate_stock_strategy.delay(stock_code)
    
    messages.success(request, _(f'已成功触发计算股票 {stock_code} 策略的任务，请等待执行完成'))
    return redirect('task_dashboard')

@login_required
def calculate_all_favorites_strategy(request):
    """
    计算所有自选股策略
    """
    logger.info(f"用户 {request.user.username} 手动触发计算所有自选股策略")
    
    # 异步执行任务
    manual_calculate_all_favorites_strategy.delay()
    
    messages.success(request, _('已成功触发计算所有自选股策略的任务，请等待执行完成'))
    return redirect('task_dashboard') 