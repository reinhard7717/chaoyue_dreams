# 文件: dashboard/templatetags/dashboard_extras.py
# 版本: V2.0 - 统一中文显示过滤器
# 描述: 提供一个统一的模板过滤器，用于将所有内部英文ID转换为中文显示名称。

import datetime
from django import template
from django.utils import timezone
from django.utils.dateparse import parse_datetime
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
    【V2.1 健壮版】
    这个过滤器接收一个从数据库取出的时间值。
    - 新功能: 它现在可以处理字符串格式的时间。
    - 核心逻辑:
      1. 如果输入是字符串，先用 `parse_datetime` 将其转换为一个朴素的datetime对象。
      2. 无论输入是datetime对象还是解析后的对象，只要它是朴素的(naive)，
         就将其标记为UTC时区，以便后续的 `localtime` 过滤器能正确转换为本地时间。
    """
    # 步骤0: 如果值为空，直接返回
    if not value:
        return value

    parsed_time = None
    # 步骤1: 检查传入的是否为字符串，如果是，则尝试解析
    if isinstance(value, str):
        parsed_time = parse_datetime(value)
        # 如果解析失败，直接返回原始字符串，避免崩溃
        if not parsed_time:
            return value
    # 如果传入的已经是datetime对象，则直接使用
    elif isinstance(value, datetime.datetime):
        parsed_time = value
    
    # 步骤2: 如果我们有一个有效的datetime对象，并且它是朴素的
    if parsed_time and timezone.is_naive(parsed_time):
        # 使用Django的make_aware函数，将朴素时间标记为UTC时区
        print(f"DEBUG: Naive time '{parsed_time}' is being made UTC aware.") # 调试信息
        aware_time = timezone.make_aware(parsed_time, timezone.utc)
        print(f"DEBUG: Aware time is now '{aware_time}'.") # 调试信息
        return aware_time
    
    # 如果值已经是感知型，或者是无法处理的类型，则直接返回原值
    return parsed_time or value