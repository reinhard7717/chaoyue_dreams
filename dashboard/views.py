# dashboard/views.py
import json
from asgiref.sync import async_to_sync
from django.shortcuts import render
from django.contrib.auth.decorators import login_required
from rest_framework import generics, viewsets, status
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView
from django.db.models import Q # 用于复杂查询
from django.core.serializers.json import DjangoJSONEncoder
from dao_manager.daos.user_dao import UserDAO
from stock_models.stock_basic import StockInfo
from users.models import FavoriteStock
from utils.websockets import send_update_to_user_sync
from .serializers import StockInfoSerializer, FavoriteStockSerializer
from tasks.stock_tasks import fetch_data_for_new_favorite # 导入新任务
import logging # 导入 logging

logger = logging.getLogger('dashboard') # 获取 logger 实例
# --- 页面视图 ---
@login_required
def dashboard_view(request):
    user_dao = UserDAO()
    """渲染主控台页面"""
    # 1. 获取当前用户的初始自选股列表
    # --- 使用 async_to_sync 调用异步 DAO 方法 ---
    get_favorites_async = user_dao.get_user_favorites # 获取异步方法本身
    user_favorites = async_to_sync(get_favorites_async)(request.user.id) # 调用包装后的同步版本
    # 2. 准备传递给模板的数据结构
    initial_favorites_data = [
        {
            'id': fav.id,
            'code': fav.stock.stock_code,
            'name': fav.stock.stock_name,
            # 初始加载时，实时数据为空，等待 WebSocket 推送
            'latest_price': None,
            'change_percent': None,
            'volume': None,
            'signal': None,
        } 
        for fav in user_favorites
    ]
    initial_favorites_json_string = json.dumps(initial_favorites_data, cls=DjangoJSONEncoder)
    # 3. 定义上下文
    context = {
        # 将初始数据转换为 JSON 字符串，以便在模板中安全地嵌入到 JavaScript 变量中
        'initial_favorites_json': initial_favorites_json_string,
    }
    # 4. 渲染模板
    return render(request, 'dashboard/home.html', context)

# --- DRF API 视图 ---

class StockSearchView(generics.ListAPIView):
    """
    股票搜索 API (GET /api/dashboard/search/?q=...)
    """
    serializer_class = StockInfoSerializer # <--- 使用对应的 Serializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        """根据 stock_code 或 stock_name 搜索 StockInfo"""
        query = self.request.query_params.get('q', None)
        if query:
            # --- 使用正确的字段名进行过滤 ---
            return StockInfo.objects.filter(
                Q(stock_code__icontains=query) | Q(stock_name__icontains=query)
            )[:10] # 限制返回结果数量
            # --- 修改结束 ---
        return StockInfo.objects.none() # 如果没有查询参数，返回空

class FavoriteStockViewSet(viewsets.ModelViewSet):
    """
    自选股 API (GET, POST, DELETE /api/dashboard/favorites/)
    """
    serializer_class = FavoriteStockSerializer
    permission_classes = [IsAuthenticated] # 必须登录

    def get_queryset(self):
        """只返回当前用户的自选股"""
        return FavoriteStock.objects.filter(user=self.request.user).select_related('stock')

    def perform_create(self, serializer):
        """
        处理添加自选股请求：保存到数据库，并触发后台任务获取详细数据后推送。
        """
        # 1. 保存 FavoriteStock 实例，并获取创建的对象
        favorite = serializer.save(user=self.request.user)

        # 2. 触发后台任务以获取新添加股票的详细数据并推送
        #    传递 user_id, stock_id, 和新创建的 favorite_id
        try:
             # 准备任务参数
            task_args = (
                self.request.user.id,
                favorite.stock.stock_code,
                favorite.id
            )
            target_queue = 'dashboard'
            fetch_data_for_new_favorite.apply_async(
                args=task_args,
                queue=target_queue,
                # routing_key=target_queue # (可选) 通常可以省略，如果 queue 定义了 routing_key
            )
            logger.info(f"已将后台任务发送到队列 '{target_queue}' 为用户 {self.request.user.id} 的新自选股 {favorite.stock.stock_code} 获取数据")
        except Exception as task_error:
            # 记录触发任务时的错误，但这不应阻止 API 返回成功 (因为收藏本身已成功)
            logger.error(f"触发后台任务 fetch_data_for_new_favorite 时出错: {task_error}", exc_info=True)

        # 3. (隐式) DRF 会自动返回 HTTP 201 Created 响应给前端
        #    我们不再需要手动推送整个列表 ('favorites_update')

    
    def perform_destroy(self, instance):
        user_id = instance.user.id # 在删除前获取 user_id
        instance.delete()
        # --- 推送更新 ---
        updated_list = self._get_formatted_favorites(user_id) # 需要根据 user_id 获取
        send_update_to_user_sync(
            user_id=user_id,
            sub_type='favorites_update',
            payload=updated_list
        )
    
    def _get_formatted_favorites(self, user_or_id):
        """辅助方法获取格式化的自选股列表"""
        if isinstance(user_or_id, int):
            favorites = FavoriteStock.objects.filter(user_id=user_or_id).select_related('stock').order_by('added_at')
        else: # 假设是 User 对象
             favorites = FavoriteStock.objects.filter(user=user_or_id).select_related('stock').order_by('added_at')
        return [
            {
                'id': fav.id,
                'code': fav.stock.stock_code,
                'name': fav.stock.stock_name,
                'latest_price': None, # 实时数据由其他推送处理
                'change_percent': None,
                'volume': None,
                'signal': None,
            } 
            for fav in favorites
        ]
    # ModelViewSet 自动处理 list (GET), create (POST), retrieve (GET /id/),
    # update (PUT/PATCH /id/), destroy (DELETE /id/)
    # 我们主要用 list, create, destroy
