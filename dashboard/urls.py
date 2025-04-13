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

    # API URLs
    path('api/search/', views.StockSearchView.as_view(), name='stock-search'),
    path('api/', include(router.urls)), # 包含 ViewSet 的 URLs (/api/favorites/)
]
