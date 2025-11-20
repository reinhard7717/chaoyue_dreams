# chaoyue_dreams\urls.py
from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from django.views.generic import RedirectView

urlpatterns = [
    # path('admin/', admin.site.urls),
    # path('users/', include('users.urls', namespace='users')),  # 用户相关URL
    # path('dashboard/', include('dashboard.urls', namespace='dashboard')), # 包含 dashboard 的 URLs
    # path('', include('django.contrib.auth.urls')), # 添加这行可以快速使用内置的登录/登出等URL name
    # # 可以设置一个根路径重定向到 dashboard
    # path('', RedirectView.as_view(url='/dashboard/', permanent=False), name='go-to-dashboard'),
    path('reinhard7717_admin/', admin.site.urls),
    path('users/', include('users.urls', namespace='users')),  # 用户相关URL
    path('dashboard/', include('dashboard.urls', namespace='dashboard')), # dashboard URLs
    path('accounts/', include('django.contrib.auth.urls')), # 登录/登出等URL
    path('', RedirectView.as_view(url='/dashboard/', permanent=False), name='go-to-dashboard'),
    # path('tasks/', include('tasks.urls')),  # 任务管理URL
    # path('', RedirectView.as_view(url='/users/', permanent=False)),  # 将根路径重定向到用户主页
]

# 在开发环境中添加媒体文件的URL
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
    # urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
