# 文件: dashboard/templatetags/dashboard_extras.py
# 版本: V2.0 - 统一中文显示过滤器
# 描述: 提供一个统一的模板过滤器，用于将所有内部英文ID转换为中文显示名称。

import datetime
from decimal import Decimal
from django import template
from django.utils import timezone
from django.template.defaultfilters import stringfilter
from django.utils.http import urlencode

register = template.Library()

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
    # --- ---
    # 使用竖线 '|' 作为新的分隔符
    if '|' not in args:
        return value # 如果参数格式不正确，直接返回原值
    old, new = args.split('|', 1) # 使用 split('|', 1) 只分割一次，更健壮
    return value.replace(old, new)

@register.simple_tag(takes_context=True)
def query_builder(context, **kwargs):
    """
    【V2.3 列表参数修复版】
    - 核心修复: 修正了处理传入的列表类型参数（如 playbooks）的逻辑。
                现在可以正确地将模板中传递的 playbooks=selected_playbooks
                设置到 query_dict 中，确保分页和筛选可以同时工作。
    """
    request = context.get('request')
    if not request:
        return "?"
    # 复制当前请求的GET参数字典
    query_dict = request.GET.copy()
    # 优先处理 kwargs 中明确传入的参数，这会覆盖掉 request.GET 中的同名参数
    for key, value in kwargs.items():
        # 如果值是一个列表（比如 selected_playbooks），使用 setlist
        if isinstance(value, list):
            query_dict.setlist(key, value)
        # 否则，直接设置
        else:
            query_dict[key] = str(value)
    # 特殊处理 playbook 的添加/移除逻辑 (如果需要的话)
    add_playbook_val = kwargs.get('add_playbook')
    remove_playbook_val = kwargs.get('remove_playbook')
    if add_playbook_val is not None or remove_playbook_val is not None:
        playbooks_list = query_dict.getlist('playbooks', [])
        if add_playbook_val is not None:
            add_playbook_str = str(add_playbook_val)
            if add_playbook_str not in playbooks_list:
                playbooks_list.append(add_playbook_str)
        if remove_playbook_val is not None:
            remove_playbook_str = str(remove_playbook_val)
            if remove_playbook_str in playbooks_list:
                playbooks_list.remove(remove_playbook_str)
        if playbooks_list:
            query_dict.setlist('playbooks', playbooks_list)
        else:
            query_dict.pop('playbooks', None)
    # 生成最终URL
    if query_dict:
        # doseq=True 确保列表参数被正确编码为 a=1&a=2 的形式
        final_url = f"?{urlencode(query_dict, doseq=True)}"
    else:
        final_url = "?" 
    return final_url

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

#代码行：定义 subtract 过滤器
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

@register.filter(name='show_sign')
def show_sign(value):
    """
    为一个数字添加正负号，并格式化为整数。
    例如: 25 -> '+25', -10 -> '-10'
    """
    try:
        value = float(value)
        # 使用 f-string 格式化，自动处理负号
        return f"{value:+.0f}"
    except (ValueError, TypeError, AttributeError):
        return "" # 如果值无效，返回空字符串

@register.filter
def subtract(value, arg):
    """模板中的减法过滤器"""
    try:
        return Decimal(str(value)) - Decimal(str(arg))
    except (ValueError, TypeError):
        return ''

@register.filter
def multiply(value, arg):
    """模板中的乘法过滤器"""
    try:
        return Decimal(str(value)) * Decimal(str(arg))
    except (ValueError, TypeError):
        return ''

@register.filter
def divide(value, arg):
    """模板中的除法过滤器"""
    try:
        return Decimal(str(value)) / Decimal(str(arg))
    except (ValueError, TypeError):
        return ''

@register.filter
def power(value, arg):
    """模板中的指数过滤器"""
    try:
        return Decimal(str(value)) ** Decimal(str(arg))
    except (ValueError, TypeError):
        return ''

@register.filter
def round(value, arg):
    """模板中的四舍五入过滤器"""
    try:
        return Decimal(str(value)).quantize(Decimal(str(arg)), rounding=ROUND_HALF_UP)
    except (ValueError, TypeError):
        return ''

@register.filter
def abs(value):
    """模板中的绝对值过滤器"""
    try:
        return abs(Decimal(str(value)))
    except (ValueError, TypeError):
        return ''

@register.filter
def floor(value):
    """模板中的向下取整过滤器"""
    try:
        return math.floor(Decimal(str(value)))
    except (ValueError, TypeError):
        return ''

@register.filter
def ceil(value):
    """模板中的向上取整过滤器"""
    try:
        return math.ceil(Decimal(str(value)))
    except (ValueError, TypeError):
        return ''













