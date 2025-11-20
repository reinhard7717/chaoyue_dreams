# 文件: strategies/management/commands/populate_playbooks.py
import json
from pathlib import Path
from django.conf import settings
from django.core.management.base import BaseCommand
from django.db import transaction
from stock_models.stock_analytics import Playbook

class Command(BaseCommand):
    help = '【V4.0 · 权威配置驱动版】解析唯一的信号字典(signal_dictionary.json)，填充或更新Playbook模型。'
    def handle(self, *args, **options):
        """
        【V4.0 · 权威配置驱动版】
        - 核心重构: 彻底废除对策略配置文件的解析。本脚本现在只读取 signal_dictionary.json
                      作为唯一的、权威的数据源，确保数据库与“单一法源”完全同步。
        - 逻辑简化: 整个流程简化为单一循环，遍历信号字典，智能解析每个信号的元数据
                      （中文名、类型、分数），并更新数据库。
        """
        self.stdout.write(self.style.SUCCESS(f'🚀 启动Playbook填充/更新流程 ({self.help})...'))
        # --- 步骤 1: 加载唯一的权威数据源 ---
        file_path = Path(settings.BASE_DIR) / 'config' / 'signal_dictionary.json'
        self.stdout.write(f"  -> 正在从唯一的权威源 '{file_path}' 加载信号字典...")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            score_type_map = data.get('score_type_map', {})
            if not score_type_map:
                self.stderr.write(self.style.ERROR("错误: 信号字典为空或格式不正确。"))
                return
        except FileNotFoundError:
            self.stderr.write(self.style.ERROR(f"错误: 无法在 '{file_path}' 找到信号字典文件。"))
            return
        except json.JSONDecodeError:
            self.stderr.write(self.style.ERROR(f"错误: 解析 '{file_path}' 文件失败，请检查JSON格式。"))
            return
        # --- 步骤 2: 获取现有数据并准备处理列表 ---
        self.stdout.write('  -> 正在从数据库获取现有的playbooks...')
        existing_playbooks_map = {p.name: p for p in Playbook.objects.all()}
        self.stdout.write(f'     已找到 {len(existing_playbooks_map)} 个现有的playbooks。')
        to_create = []
        to_update = []
        # 定义从JSON类型字符串到模型枚举的映射
        TYPE_MAP = {
            "positional": Playbook.PlaybookType.OFFENSIVE,
            "dynamic": Playbook.PlaybookType.OFFENSIVE,
            "composite": Playbook.PlaybookType.OFFENSIVE,
            "playbook": Playbook.PlaybookType.OFFENSIVE,
            "risk": Playbook.PlaybookType.RISK,
            "trigger": Playbook.PlaybookType.TRIGGER,
            "context": Playbook.PlaybookType.CONTEXT,
            "subtotal": Playbook.PlaybookType.CONTEXT, # 将subtotal也视为上下文类型
            "unknown": Playbook.PlaybookType.UNKNOWN,
        }
        # --- 步骤 3: 遍历权威字典，生成创建/更新列表 ---
        self.stdout.write('  -> 正在根据权威字典解析所有信号...')
        for name, meta in score_type_map.items():
            # 过滤掉所有说明性/注释性条目
            if name.startswith('说明_') or not isinstance(meta, dict):
                continue
            # 智能获取分数：优先取 'score'，其次取 'penalty_weight'，都没有则为 0
            score = meta.get('score', meta.get('penalty_weight', 0))
            # 获取中文名，如果不存在则使用信号名本身
            cn_name = meta.get('cn_name', name)
            # 获取类型
            type_str = meta.get('type', 'unknown')
            playbook_type = TYPE_MAP.get(type_str, Playbook.PlaybookType.UNKNOWN)
            existing_obj = existing_playbooks_map.get(name)
            if existing_obj:
                # 检查是否有字段需要更新
                if (existing_obj.cn_name != cn_name or
                    float(existing_obj.default_score) != float(score) or
                    existing_obj.playbook_type != playbook_type):
                    existing_obj.cn_name = cn_name
                    existing_obj.default_score = score
                    existing_obj.playbook_type = playbook_type
                    to_update.append(existing_obj)
            else:
                # 如果对象不存在，则创建一个新的
                to_create.append(Playbook(
                    name=name,
                    cn_name=cn_name,
                    playbook_type=playbook_type,
                    default_score=score
                ))
        # --- 步骤 4: 执行数据库写入 ---
        if to_create:
            with transaction.atomic():
                Playbook.objects.bulk_create(to_create)
            self.stdout.write(self.style.SUCCESS(f'✅ 成功创建 {len(to_create)} 个新的playbooks。'))
        if to_update:
            with transaction.atomic():
                fields_to_update = ['cn_name', 'default_score', 'playbook_type']
                Playbook.objects.bulk_update(to_update, fields_to_update)
            self.stdout.write(self.style.SUCCESS(f'✅ 成功更新 {len(to_update)} 个现有的playbooks。'))
        if not to_create and not to_update:
            self.stdout.write(self.style.WARNING('ℹ️ 未发现新的或需要更新的playbooks。数据库已是最新状态。'))

