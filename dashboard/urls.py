# dashboard/urls.py
from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import FavoriteStockViewSet, TransactionViewSet
from . import views

app_name = 'dashboard'

# DRF Router 用于 ViewSet
router = DefaultRouter()
router.register(r'favorites', FavoriteStockViewSet, basename='favorite')
router.register(r'transactions', TransactionViewSet, basename='transaction')
urlpatterns = [
    # 页面 URL
    path('', views.dashboard_view, name='home'), # 主控台页面
    path('trend_following_list/', views.trend_following_list, name='trend_following_list'),
    path('stock/<str:stock_code>/', views.stock_detail_view, name='stock_detail'),
    path('prophet-signals/', views.prophet_signal_list, name='prophet_signal_list'),
    path('fav_trend_following_list/', views.fav_trend_following_list, name='fav_trend_following_list'),
    path('realtime_engine/', views.realtime_engine_view, name='realtime_engine'),
    # API URLs
    path('api/search/', views.StockSearchView.as_view(), name='stock-search'),
    path('api/', include(router.urls)), # 包含 ViewSet 的 URLs (/api/favorites/)
]
