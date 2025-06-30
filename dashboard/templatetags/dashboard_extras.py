# dashboard/templatetags/dashboard_extras.py

from django import template
# 从我们刚刚创建的映射模块中导入辅助函数
from dashboard.playbook_mapping import get_playbook_display_name

# 实例化一个 template.Library，用于注册我们的自定义标签和过滤器
register = template.Library()

@register.filter(name='playbook_display')
def playbook_display(playbook_key):
    """
    一个自定义模板过滤器。
    用法: {{ playbook_key|playbook_display }}
    它会接收一个剧本的英文key，并返回其对应的中文名称。
    """
    # 如果传入的 key 是空的或 None，直接返回空字符串
    if not playbook_key:
        return ""
    # 调用我们之前写好的函数来获取显示名称
    return get_playbook_display_name(playbook_key)
