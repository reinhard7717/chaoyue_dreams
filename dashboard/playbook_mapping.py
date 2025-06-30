# dashboard/playbook_mapping.py

import json
import os
from django.conf import settings

# 定义一个全局变量来缓存映射表，避免重复读取文件
PLAYBOOK_MAP = {}

def load_playbook_map():
    """
    【V3.0 重构版】从多个 JSON 配置文件中加载剧本 Key -> 中文名称的映射。
    - 适配新的JSON结构: "KEY": { "name": "中文名", "score": 100, "description": "..." }
    - 使用递归函数深度遍历整个JSON结构，专门查找 "points" 字典。
    - 这个函数只会在 Django 启动时被调用一次。
    """
    # 声明我们要修改的是全局变量
    global PLAYBOOK_MAP
    # 如果映射表已经加载，则直接返回，防止重复加载
    if PLAYBOOK_MAP:
        return PLAYBOOK_MAP

    print("正在加载剧本中文映射配置 (V3.0 重构版)...")

    # 定义一个包含所有可能含有 "points" 字段的策略配置文件列表。
    config_files = [
        'trend_follow_strategy.json',
    ]

    temp_map = {}

    # ▼▼▼ 修改区域开始 ▼▼▼
    # 解释: 递归函数被重构，以解析新的、更清晰的JSON结构。
    def _recursive_find_mappings(node):
        """
        递归地在JSON节点（字典或列表）中查找 "points" 字典并解析。
        """
        # 如果当前节点是字典
        if isinstance(node, dict):
            # 遍历字典的键值对
            for key, value in node.items():
                # 1. 核心目标：找到名为 "points" 的字典
                if key == "points" and isinstance(value, dict):
                    # 2. 遍历 "points" 字典中的所有剧本
                    for playbook_key, playbook_data in value.items():
                        # 忽略分类标记
                        if playbook_key.startswith("__"):
                            continue
                        
                        # 3. 检查剧本数据是否为字典，并且包含 "name" 字段
                        if isinstance(playbook_data, dict) and "name" in playbook_data:
                            # 存储映射关系：KEY -> 中文名
                            temp_map[playbook_key] = playbook_data["name"]
                        else:
                            # 4. 备用逻辑：如果结构不符合预期，则使用key自身作为名称
                            temp_map.setdefault(playbook_key, playbook_key)
                
                # 5. 对子节点继续递归查找
                _recursive_find_mappings(value)
        
        # 如果当前节点是列表，则遍历列表中的每个元素
        elif isinstance(node, list):
            for item in node:
                _recursive_find_mappings(item)
    # ▲▲▲ 修改区域结束 ▲▲▲

    # 遍历所有配置文件
    for config_file in config_files:
        config_path = os.path.join(settings.BASE_DIR, 'config', config_file)
        print(f"  -> 正在读取文件: {config_file}")
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # 对加载的JSON数据启动递归查找
                _recursive_find_mappings(data)
        except FileNotFoundError:
            print(f"  -> 警告：剧本配置文件未找到，已跳过。路径: {config_path}")
        except json.JSONDecodeError:
            print(f"  -> 错误：剧本配置文件格式错误，路径: {config_path}")

    PLAYBOOK_MAP = temp_map
    print(f"剧本映射加载完成，共聚合 {len(PLAYBOOK_MAP)} 条映射关系。")
    return PLAYBOOK_MAP

# 在模块加载时，自动执行加载函数，完成初始化
load_playbook_map()

def get_playbook_display_name(playbook_key):
    """
    一个辅助函数，根据剧本key获取显示名称。
    如果在映射表中找不到，则返回原始key，保证总有输出。
    """
    return PLAYBOOK_MAP.get(playbook_key, playbook_key)
