"""
工具包
包含各种通用工具类和方法
"""

# 使用延迟导入来避免Django应用注册表未准备好的问题

def get_model_json_encoder():
    from .models import ModelJSONEncoder
    return ModelJSONEncoder

def get_custom_serializer():
    from .custom_serializer import CustomJSONSerializer
    return CustomJSONSerializer

def get_common():
    from .common import (
        parse_datetime,
        parse_number,
        format_datetime,
        format_number
    )
    return {
        'parse_datetime': parse_datetime,
        'parse_number': parse_number,
        'format_datetime': format_datetime,
        'format_number': format_number
    }

__all__ = [
    'get_cache_manager',
    'get_model_json_encoder',
    'get_custom_serializer',
    'get_common',
]
