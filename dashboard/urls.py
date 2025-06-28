# dashboard/urls.py
from django.urls import path, include
from rest_framework.routers import DefaultRouter
from . import views

app_name = 'dashboard'

# DRF Router 用于 ViewSet
router = DefaultRouter()
router.register(r'favorites', views.FavoriteStockViewSet, basename='favorite')

urlpatterns = [
    # 页面 URL
    path('', views.dashboard_view, name='home'), # 主控台页面
    path('trend_following_list/', views.trend_following_list, name='trend_following_list'),
    path('fav_trend_following_list/', views.fav_trend_following_list, name='fav_trend_following_list'),
    path('monthly_trend_following/', views.monthly_trend_strategy_list, name='monthly_trend_following'),
    path('fav_monthly_trend_list/', views.fav_monthly_trend_list, name='fav_monthly_trend_list'),

    # API URLs
    path('api/search/', views.StockSearchView.as_view(), name='stock-search'),
    path('api/', include(router.urls)), # 包含 ViewSet 的 URLs (/api/favorites/)
]
