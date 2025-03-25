from django.urls import path
from . import views

urlpatterns = [
    # 主页
    path('', views.task_dashboard, name='task_dashboard'),
    
    # 数据刷新任务
    path('refresh/stock/<str:stock_code>/', views.refresh_stock_data, name='refresh_stock_data'),
    path('refresh/favorites/', views.refresh_all_favorites_data, name='refresh_all_favorites_data'),
    path('refresh/indexes/', views.refresh_all_index_data, name='refresh_all_index_data'),
    path('refresh/fund_flow/', views.refresh_all_fund_flow_data, name='refresh_all_fund_flow_data'),
    path('refresh/datacenter/', views.refresh_all_datacenter_data, name='refresh_all_datacenter_data'),
    path('refresh/stock_pools/', views.refresh_all_stock_pools, name='refresh_all_stock_pools'),
    path('refresh/all/', views.refresh_all_system_data, name='refresh_all_system_data'),
    
    # 策略计算任务
    path('calculate/stock/<str:stock_code>/', views.calculate_stock_strategy, name='calculate_stock_strategy'),
    path('calculate/favorites/', views.calculate_all_favorites_strategy, name='calculate_all_favorites_strategy'),
] 