# 文件: dashboard/templatetags/dashboard_extras.py
# 版本: V2.0 - 统一中文显示过滤器
# 描述: 提供一个统一的模板过滤器，用于将所有内部英文ID转换为中文显示名称。

import datetime
from django import template
from django.utils import timezone
from django.utils.http import urlencode
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
    【V2.2 终极版】
    接收一个可能是朴素的datetime对象（我们假定它代表UTC时间），
    并将其转换为一个带UTC时区的、可感知的datetime对象。
    这是在 USE_TZ=True 但ORM未自动附加时区时的关键修复步骤。
    """
    # 如果输入值为空或不是datetime对象，直接返回，不做处理
    if not isinstance(value, datetime.datetime):
        return value

    # ▼▼▼【核心逻辑】▼▼▼
    # 检查时间是否是朴素的 (naive)
    if timezone.is_naive(value):
        # 如果是朴素的，我们必须假定它代表的是UTC时间。
        # 使用 timezone.make_aware 将其“激活”为带UTC时区的对象。
        print(f"DEBUG: Naive time '{value}' is being made UTC aware.") # 调试信息
        return timezone.make_aware(value, timezone.utc)
    
    # 如果时间已经是感知的 (aware)，则直接返回原值，避免重复处理
    return value

@register.simple_tag(takes_context=True)
def query_builder(context, **kwargs):
    """
    一个强大的URL查询参数构造器。
    它能根据当前请求的GET参数，智能地添加、修改或删除指定的参数。
    特别为多值参数（如 'playbooks'）做了优化。

    用法:
    - 添加/修改单个值: {% query_builder key1='value1' key2='value2' %}
    - 清空所有参数: {% query_builder clear_all=True %}
    - 添加一个多值参数: {% query_builder add_playbook='some_playbook' %}
    - 移除一个多值参数: {% query_builder remove_playbook='some_playbook' %}
    """
    # 复制当前请求的GET参数字典，以便修改
    query_dict = context['request'].GET.copy()

    # 处理特殊指令
    if kwargs.get('clear_all'):
        return '' # 返回空字符串，即清空所有查询参数

    add_playbook = kwargs.pop('add_playbook', None)
    remove_playbook = kwargs.pop('remove_playbook', None)

    # 处理 'playbooks' 多值参数
    playbooks = query_dict.getlist('playbooks', [])
    if add_playbook and add_playbook not in playbooks:
        playbooks.append(add_playbook)
    if remove_playbook and remove_playbook in playbooks:
        playbooks.remove(remove_playbook)
    
    # 更新字典中的 'playbooks' 列表
    query_dict.setlist('playbooks', playbooks)
    # 如果列表为空，则从字典中移除该键，保持URL整洁
    if not query_dict.getlist('playbooks'):
        query_dict.pop('playbooks', None)

    # 处理其他普通的键值对参数
    for key, value in kwargs.items():
        query_dict[key] = value

    # 将最终的字典编码为URL查询字符串
    return f"?{urlencode(query_dict)}" if query_dict else '?'