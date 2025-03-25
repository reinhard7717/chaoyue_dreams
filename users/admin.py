from django.contrib import admin
from django.utils.translation import gettext_lazy as _
from .models import UserProfile, FavoriteStock

@admin.register(UserProfile)
class UserProfileAdmin(admin.ModelAdmin):
    """
    用户资料管理
    """
    list_display = ('user', 'phone', 'last_login_ip', 'email_notification', 'created_at')
    list_filter = ('email_notification', 'created_at')
    search_fields = ('user__username', 'user__email', 'phone', 'bio')
    readonly_fields = ('created_at', 'updated_at')
    
    fieldsets = (
        (None, {'fields': ('user',)}),
        (_('个人信息'), {'fields': ('phone', 'avatar', 'bio')}),
        (_('通知设置'), {'fields': ('email_notification',)}),
        (_('登录信息'), {'fields': ('last_login_ip', 'created_at', 'updated_at')}),
    )

@admin.register(FavoriteStock)
class FavoriteStockAdmin(admin.ModelAdmin):
    """
    自选股管理
    """
    list_display = ('user', 'stock_code', 'stock_name', 'is_pinned', 'added_at')
    list_filter = ('is_pinned', 'user')
    search_fields = ('stock_code', 'stock_name', 'user__username')
    readonly_fields = ('added_at',)
    
    fieldsets = (
        (None, {'fields': ('user', 'stock_code', 'stock_name')}),
        (_('自选股设置'), {'fields': ('is_pinned', 'note', 'tags')}),
        (_('添加时间'), {'fields': ('added_at',)}),
    )
