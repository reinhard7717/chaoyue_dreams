# dashboard/views.py
import json
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
from .serializers import StockInfoSerializer, FavoriteStockSerializer
# --- 页面视图 ---
@login_required
def dashboard_view(request):
    user_dao = UserDAO()
    """渲染主控台页面"""
    # 1. 获取当前用户的初始自选股列表
    user_favorites = user_dao.get_user_favorites(request.user.id)
    # 2. 准备传递给模板的数据结构
    initial_favorites_data = [
        {
            'code': fav.stock.code,
            'name': fav.stock.name,
            # 初始加载时，实时数据为空，等待 WebSocket 推送
            'latest_price': None,
            'change_percent': None,
            'volume': None,
            'signal': None,
        } 
        for fav in user_favorites
    ]
    # 3. 定义上下文
    context = {
        # 将初始数据转换为 JSON 字符串，以便在模板中安全地嵌入到 JavaScript 变量中
        'initial_favorites_json': json.dumps(initial_favorites_data, cls=DjangoJSONEncoder),
    }
    # 4. 渲染模板
    return render(request, 'dashboard/home.html', context)

# --- DRF API 视图 ---

class StockSearchView(generics.ListAPIView):
    """
    股票搜索 API (GET /api/dashboard/search/?q=...)
    """
    serializer_class = StockInfoSerializer
    permission_classes = [IsAuthenticated] # 需要登录才能搜索

    def get_queryset(self):
        query = self.request.query_params.get('q', None)
        if query:
            # 搜索代码或名称包含查询字符串的股票
            return StockInfo.objects.filter(
                Q(code__icontains=query) | Q(name__icontains=query)
            )[:10] # 限制返回结果数量
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
        """创建时自动关联当前用户 (虽然序列化器里也处理了)"""
        # 序列化器需要 request context 来获取 user
        serializer.save(user=self.request.user)

    # ModelViewSet 自动处理 list (GET), create (POST), retrieve (GET /id/),
    # update (PUT/PATCH /id/), destroy (DELETE /id/)
    # 我们主要用 list, create, destroy
