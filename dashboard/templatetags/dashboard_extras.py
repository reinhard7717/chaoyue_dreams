# 文件: dashboard/templatetags/dashboard_extras.py
# 版本: V2.0 - 统一中文显示过滤器
# 描述: 提供一个统一的模板过滤器，用于将所有内部英文ID转换为中文显示名称。

import datetime
from django import template
from django.utils import timezone
from utils.display_maps import DISPLAY_MAP

register = template.Library()

@register.filter(name='playbook_display')
def playbook_display(value):
    """
    一个统一的翻译过滤器。
    接收一个英文ID (可能是策略名或剧本名)，
    在DISPLAY_MAP中查找对应的中文名。
    如果找不到，则返回原始值，以确保显示不会中断。
    """
    # 解释: 使用.get(key, default)方法可以安全地获取值。
    # 如果value在字典中不存在，它会返回默认值，这里我们让默认值就是value本身。
    return DISPLAY_MAP.get(value, value)

@register.filter(name='make_utc_aware')
def make_utc_aware(value):
    """
    这个过滤器接收一个从数据库取出的朴素型(naive)datetime对象,
    我们业务上明确知道它是UTC时间, 此过滤器会将其转换为带UTC时区信息的
    感知型(aware)datetime对象, 以便后续的localtime过滤器能正确工作。
    """
    # 检查传入的是否为datetime对象，并且是朴素型的
    if isinstance(value, datetime.datetime) and timezone.is_naive(value):
        # 使用Django的make_aware函数，将朴素时间标记为UTC时区
        # 这是最关键的一步
        print(f"DEBUG: Naive time '{value}' is being made UTC aware.") # 调试信息
        aware_time = timezone.make_aware(value, timezone.utc)
        print(f"DEBUG: Aware time is now '{aware_time}'.") # 调试信息
        return aware_time
    
    # 如果值已经是感知型或者不是datetime对象，则直接返回原值
    return value
