from django.urls import path
from django.contrib.auth.views import LogoutView
from . import views

urlpatterns = [
    # 用户认证
    path('login/', views.CustomLoginView.as_view(), name='login'),
    path('logout/', LogoutView.as_view(), name='logout'),
    
    # 个人资料
    path('profile/', views.profile_view, name='profile'),
    
    # 自选股管理
    path('favorites/', views.favorite_stock_list, name='favorite_stock_list'),
    path('favorites/add/', views.add_favorite_stock, name='add_favorite_stock'),
    path('favorites/edit/<int:pk>/', views.edit_favorite_stock, name='edit_favorite_stock'),
    path('favorites/delete/<int:pk>/', views.delete_favorite_stock, name='delete_favorite_stock'),
] 