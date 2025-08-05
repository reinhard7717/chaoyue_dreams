# 文件: dashboard/templatetags/dashboard_extras.py
# 版本: V2.0 - 统一中文显示过滤器
# 描述: 提供一个统一的模板过滤器，用于将所有内部英文ID转换为中文显示名称。

import datetime
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
    # --- 【核心修改】---
    # 使用竖线 '|' 作为新的分隔符
    if '|' not in args:
        return value # 如果参数格式不正确，直接返回原值

    old, new = args.split('|', 1) # 使用 split('|', 1) 只分割一次，更健壮
    return value.replace(old, new)

@register.simple_tag(takes_context=True)
def query_builder(context, **kwargs):
    """
    【V2.2 终极诊断版】
    - 核心诊断: 加入大量打印语句，跟踪从模板接收到的上下文、参数以及每一步处理后的字典状态。
    """
    print("\n--- [Template Tag Debug] 'query_builder' 被调用 ---")
    
    # 诊断1: 检查上下文和请求对象是否存在
    request = context.get('request')
    if not request:
        print("  [Debug-ERROR] 上下文中没有找到 'request' 对象！标签无法工作。")
        return "?"
    
    print(f"  [Debug] 原始请求GET参数: {request.GET}")
    print(f"  [Debug] 从模板接收到的kwargs: {kwargs}")

    # 复制当前请求的GET参数字典
    query_dict = request.GET.copy()

    # 诊断2: 检查传入的 playbook ID
    add_playbook_val = kwargs.get('add_playbook')
    remove_playbook_val = kwargs.get('remove_playbook')
    print(f"  [Debug] 解析到 add_playbook: {add_playbook_val} (类型: {type(add_playbook_val)})")
    print(f"  [Debug] 解析到 remove_playbook: {remove_playbook_val} (类型: {type(remove_playbook_val)})")

    # --- 核心逻辑，统一为字符串处理 ---
    playbooks_list = query_dict.getlist('playbooks', [])
    print(f"  [Debug] 当前URL中的playbooks列表 (字符串): {playbooks_list}")

    if add_playbook_val is not None:
        add_playbook_str = str(add_playbook_val)
        if add_playbook_str not in playbooks_list:
            playbooks_list.append(add_playbook_str)
            print(f"  [Debug] 添加后，playbooks列表变为: {playbooks_list}")

    if remove_playbook_val is not None:
        remove_playbook_str = str(remove_playbook_val)
        if remove_playbook_str in playbooks_list:
            playbooks_list.remove(remove_playbook_str)
            print(f"  [Debug] 移除后，playbooks列表变为: {playbooks_list}")

    # 更新或清理字典中的 'playbooks'
    if playbooks_list:
        query_dict.setlist('playbooks', playbooks_list)
    else:
        query_dict.pop('playbooks', None)
    
    print(f"  [Debug] 处理完playbooks后，query_dict: {query_dict}")

    # 处理其他参数，如 'page'
    # 我们从 kwargs 中移除已经处理过的 playbook 参数
    kwargs.pop('add_playbook', None)
    kwargs.pop('remove_playbook', None)
    for key, value in kwargs.items():
        query_dict[key] = str(value)
    
    print(f"  [Debug] 处理完所有参数后，最终的query_dict: {query_dict}")

    # 生成最终URL
    if query_dict:
        final_url = f"?{urlencode(query_dict, doseq=True)}"
    else:
        # 如果字典为空，返回一个干净的URL（指向当前路径，无参数）
        # 在模板中，这通常意味着清除所有筛选
        final_url = "?" 
    
    print(f"  [Debug] 最终生成的URL后缀: {final_url}")
    print("--- [Template Tag Debug] 'query_builder' 调用结束 ---\n")
    
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


