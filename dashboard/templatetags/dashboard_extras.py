# 文件: dashboard/templatetags/dashboard_extras.py
# 版本: V2.0 - 统一中文显示过滤器
# 描述: 提供一个统一的模板过滤器，用于将所有内部英文ID转换为中文显示名称。

import datetime
from django import template
from django.utils import timezone
from django.template.defaultfilters import stringfilter
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

@register.filter
@stringfilter
def replace(value, args):
    """
    自定义模板过滤器，用于替换字符串中的子串。
    用法: {{ some_string|replace:"old|new" }}
    """
    # --- 【核心修改】---
    # 使用竖线 '|' 作为新的分隔符
    if '|' not in args:
        return value # 如果参数格式不正确，直接返回原值

    old, new = args.split('|', 1) # 使用 split('|', 1) 只分割一次，更健壮
    return value.replace(old, new)

@register.simple_tag(takes_context=True)
def query_builder(context, **kwargs):
    """
    【V2.1 类型修复版】
    一个强大的URL查询参数构造器。
    - 核心修复: 统一将所有 playbook ID 转换为字符串进行比较和操作，解决因类型不匹配导致的筛选链接生成失败问题。
    """
    query_dict = context['request'].GET.copy()

    if kwargs.get('clear_all'):
        return '?' # 清空时返回带问号的空URL

    # 步骤1: 将所有传入的 playbook ID 统一转换为字符串
    add_playbook_str = str(kwargs.pop('add_playbook', '')) if 'add_playbook' in kwargs else None
    remove_playbook_str = str(kwargs.pop('remove_playbook', '')) if 'remove_playbook' in kwargs else None

    # 步骤2: 从 GET 参数获取已有的 playbooks 列表 (它们已经是字符串)
    playbooks_list = query_dict.getlist('playbooks', [])

    # 步骤3: 使用字符串进行添加和移除操作
    if add_playbook_str and add_playbook_str not in playbooks_list:
        playbooks_list.append(add_playbook_str)

    if remove_playbook_str and remove_playbook_str in playbooks_list:
        playbooks_list.remove(remove_playbook_str)

    # 步骤4: 更新字典中的 'playbooks' 列表
    if playbooks_list:
        query_dict.setlist('playbooks', playbooks_list)
    else:
        # 如果列表为空，则从字典中移除该键，保持URL整洁
        query_dict.pop('playbooks', None)

    # 步骤5: 处理其他普通的键值对参数 (例如 'page')
    for key, value in kwargs.items():
        # 确保所有值都是字符串
        query_dict[key] = str(value)

    # 步骤6: 生成最终的URL查询字符串
    if query_dict:
        # doseq=True 确保列表被正确编码为 a=1&a=2
        return f"?{urlencode(query_dict, doseq=True)}"
    else:
        # 如果最终字典为空，返回一个干净的问号
        return '?'

@register.filter
def multiply(value, arg):
    """
    Multiplies the value with the argument.
    Usage: {{ value|multiply:arg }}
    """
    try:
        return float(value) * float(arg)
    except (ValueError, TypeError):
        print(f"DEBUG: multiply filter error - value: {value}, arg: {arg}") # 调试信息
        return 0.0 # 发生错误时返回0.0，或根据业务需求处理

# 新增代码行：定义 subtract 过滤器
@register.filter
def subtract(value, arg):
    """
    Subtracts the argument from the value.
    Usage: {{ value|subtract:arg }}
    """
    try:
        return float(value) - float(arg)
    except (ValueError, TypeError):
        print(f"DEBUG: subtract filter error - value: {value}, arg: {arg}") # 调试信息
        return 0.0 # 发生错误时返回0.0，或根据业务需求处理

@register.filter(name='get_cn_name')
def get_cn_name(signal_key, metadata_dict):
    """
    一个安全的字典查找过滤器。
    从 metadata_dict 中获取 signal_key 对应的中文名，如果找不到，则返回原始的 key。
    """
    if not isinstance(metadata_dict, dict):
        return signal_key
    return metadata_dict.get(signal_key, signal_key)

@register.filter(name='get_signal_status')
def get_signal_status(signal_key, metadata_dict):
    """
    一个安全的字典查找过滤器。
    从 metadata_dict 中获取 signal_key 对应的状态，如果找不到，则返回原始的 key。
    """
    if not isinstance(metadata_dict, dict):
        return signal_key
    return metadata_dict.get(signal_key, signal_key)


