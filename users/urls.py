# users\urls.py
from django.urls import path
from django.contrib.auth.views import LogoutView
from django.contrib.auth import views as auth_views
from .forms import UserLoginForm # 引入你的自定义表单
from . import views
from dashboard.views import dashboard_view  # 导入dashboard_view


app_name = 'users' # 定义 app 命名空间

urlpatterns = [
    # 用户认证
    path('login/', auth_views.LoginView.as_view(
        template_name='users/login.html', # 指定登录模板
        authentication_form=UserLoginForm  # 使用你的自定义表单
    ), name='login'),
    path('logout/', auth_views.LogoutView.as_view(next_page='/'), name='logout'), # 登出后重定向到首页
    path('home/', dashboard_view, name='home'), # 主控台 URL
    path('favorites/', views.favorite_list_view, name='favorite_list'), # 自选股列表页 URL (示例)
    path('profile/', views.profile_view, name='profile'), # 个人设置页 URL (示例)
    # 注册账户
    path('register/', views.register, name='register'),

    # 个人资料
    path('profile/', views.profile_view, name='profile'),
    # 自选股管理
    path('favorites/', views.favorite_stock_list, name='favorite_stock_list'),
    path('favorites/add/', views.add_favorite_stock, name='add_favorite_stock'),
    path('favorites/edit/<int:pk>/', views.edit_favorite_stock, name='edit_favorite_stock'),
    path('favorites/delete/<int:pk>/', views.delete_favorite_stock, name='delete_favorite_stock'),
] 