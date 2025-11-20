# 文件: utils/config_loader.py
import json
from typing import Any, Dict, List, Union

def _strip_comments_recursive(data: Any) -> Any:
    """
    递归地从字典或字典列表中移除所有说明性键。
    说明性键定义为 '说明' 或以 '说明_' 开头的键。
    这个函数确保业务逻辑代码接收到的是纯净的参数字典。
    Args:
        data: 输入的数据，可以是字典、列表或任何其他类型。
    Returns:
        移除了说明性键后的纯净数据。
    """
    # 如果是字典，则遍历其键值对
    if isinstance(data, dict):
        # 使用字典推导式创建一个新字典，只包含非说明性的键
        # 对于值，我们递归调用本函数
        return {
            key: _strip_comments_recursive(value)
            for key, value in data.items()
            if not key.startswith('说明')
        }
    # 如果是列表，则遍历其每个元素
    elif isinstance(data, list):
        # 使用列表推导式创建一个新列表，对每个元素递归调用本函数
        return [_strip_comments_recursive(item) for item in data]
    # 如果是其他类型（如字符串、数字等），直接返回
    else:
        return data

def load_strategy_config(file_path: str) -> Dict:
    """
    加载策略JSON配置文件，并自动移除所有说明性键。
    Args:
        file_path: JSON配置文件的路径。
    Returns:
        一个不包含任何说明性键的纯净配置字典。
    Raises:
        FileNotFoundError: 如果文件路径不存在。
        json.JSONDecodeError: 如果文件不是有效的JSON。
    """
    # print(f"正在从 {file_path} 加载配置...")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_config = json.load(f)
        # 调用递归函数剥离所有说明性键
        clean_config = _strip_comments_recursive(raw_config)
        # print("配置加载并清理完成。")
        return clean_config
    except FileNotFoundError:
        print(f"错误: 配置文件未找到 at '{file_path}'")
        raise
    except json.JSONDecodeError as e:
        print(f"错误: 解析JSON配置文件 '{file_path}' 失败: {e}")
        raise

