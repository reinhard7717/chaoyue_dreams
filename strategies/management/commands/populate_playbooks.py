# 文件: strategies/management/commands/populate_playbooks.py
import json
from pathlib import Path
from django.conf import settings
from django.core.management.base import BaseCommand
from django.db import transaction
from stock_models.stock_analytics import Playbook
# 保持使用 utils.config_loader 来加载策略配置，因为它可能包含环境特定的逻辑
from utils.config_loader import load_strategy_config

class Command(BaseCommand):
   
    help = '【V3.3 Bugfix】解析策略配置文件和信号字典，填充或更新Playbook模型。'

    def _load_signal_dictionary(self):
        """
        【V3.0 逻辑不变】独立加载信号字典文件。
        """
        # 从独立的 signal_dictionary.json 文件加载信号元数据
        file_path = Path(settings.BASE_DIR) / 'config' / 'signal_dictionary.json'
        self.stdout.write(f"  -> 正在从 '{file_path}' 加载信号字典...")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            # 字典的核心内容在 'score_type_map' 键下
            return data.get('score_type_map', {})
        except FileNotFoundError:
            self.stderr.write(self.style.ERROR(f"错误: 无法在 '{file_path}' 找到信号字典文件。"))
            return None
        except json.JSONDecodeError:
            self.stderr.write(self.style.ERROR(f"错误: 解析 '{file_path}' 文件失败，请检查JSON格式。"))
            return None

    def _process_playbook(self, name, score, playbook_type, score_type_map, existing_map, to_create, to_update, processed_signals):
        """
        【V3.2 逻辑不变】处理单个playbook的创建或更新逻辑。
        """
        if not name:
            return

        # 增加健壮性检查，确保传入的score是数字类型。
        if not isinstance(score, (int, float)):
            # 使用 print 输出调试信息
            print(f"调试信息: 信号 '{name}' 的分数不是数字，而是 '{type(score)}' 类型。将使用默认值 0。值为: {score}")
            self.stdout.write(self.style.WARNING(f"警告: 信号 '{name}' 的分数不是数字，而是 '{type(score)}' 类型。将使用默认值 0。"))
            score = 0

        # 从权威的信号字典中获取元数据
        signal_meta = score_type_map.get(name, {})
        # 如果字典中没有中文名，则默认使用信号名本身
        cn_name = signal_meta.get('cn_name', name)

        playbook_obj = existing_map.get(name)

        if playbook_obj:
            # 如果对象已存在，检查是否有字段需要更新
            has_changed = False
            if playbook_obj.cn_name != cn_name:
                playbook_obj.cn_name = cn_name
                has_changed = True
            # 确保比较时类型一致，避免浮点数精度问题
            if float(playbook_obj.default_score) != float(score):
                playbook_obj.default_score = score
                has_changed = True
            if playbook_obj.playbook_type != playbook_type:
                playbook_obj.playbook_type = playbook_type
                has_changed = True
            
            if has_changed:
                to_update.append(playbook_obj)
        else:
            # 如果对象不存在，则创建一个新的
            to_create.append(Playbook(
                name=name,
                cn_name=cn_name,
                playbook_type=playbook_type,
                default_score=score
            ))
        
        # 无论创建还是更新，都将此信号标记为已处理，防止在最终检查时重复处理
        processed_signals.add(name)

    def handle(self, *args, **options):
        """
        【V3.3 Bugfix】
        - 核心修复: 在所有解析循环中，调用 `_process_playbook` 之前，都增加了对 `processed_signals` 集合的检查。
                    这可以防止因同一信号出现在配置文件不同位置而导致的重复处理，从而解决了 `IntegrityError` (主键冲突) 的问题。
        """
        self.stdout.write(self.style.SUCCESS('🚀 启动Playbook填充/更新流程 (V3.3 Bugfix)...'))

        # --- 步骤1: 读取和解析数据 (在事务之外执行) ---
        strategy_config = load_strategy_config('config/trend_follow_strategy.json')
        if not strategy_config:
            self.stderr.write(self.style.ERROR("错误: 无法加载 'config/trend_follow_strategy.json'。进程终止。"))
            return

        # 从独立的JSON文件加载信号字典
        score_type_map = self._load_signal_dictionary()
        if not score_type_map:
            self.stderr.write(self.style.ERROR("错误: 无法加载信号字典。进程终止。"))
            return
        
        self.stdout.write('  -> 正在从数据库获取现有的playbooks...')
        existing_playbooks_map = {p.name: p for p in Playbook.objects.all()}
        self.stdout.write(f'     已找到 {len(existing_playbooks_map)} 个现有的playbooks。')

        playbooks_to_create = []
        playbooks_to_update = []
        # 初始化一个集合，用于存储所有在计分模块中处理过的信号。
        processed_signals = set()

        # 定义从JSON类型字符串到模型枚举的映射
        TYPE_MAP = {
            "positional": Playbook.PlaybookType.OFFENSIVE,
            "dynamic": Playbook.PlaybookType.OFFENSIVE,
            "composite": Playbook.PlaybookType.OFFENSIVE,
            "playbook": Playbook.PlaybookType.OFFENSIVE,
            "risk": Playbook.PlaybookType.RISK,
            "trigger": Playbook.PlaybookType.TRIGGER,
            "context": Playbook.PlaybookType.CONTEXT,
            "unknown": Playbook.PlaybookType.UNKNOWN,
        }

        def get_playbook_type(signal_name):
            """根据信号名字从字典中查找并返回正确的Playbook类型枚举。"""
            meta = score_type_map.get(signal_name, {})
            type_str = meta.get('type', 'unknown')
            return TYPE_MAP.get(type_str, Playbook.PlaybookType.UNKNOWN)

        # --- 步骤2: 解析策略配置文件中的计分项 ---
        trend_follow_params = strategy_config.get('strategy_params', {}).get('trend_follow', {})
        scoring_params = trend_follow_params.get('four_layer_scoring_params', {})

        # 解析新的四层计分结构中的字典部分
        self.stdout.write('  -> 正在解析进攻与触发器信号...')
        dict_based_sections = {
            'contextual_setup_scoring': 'positive_signals',
            'playbook_synergy_scoring': 'positive_signals',
            'dynamic_scoring': 'positive_signals',
            'trigger_event_scoring': 'positive_signals',
        }
        for section_key, data_key in dict_based_sections.items():
            section_data = scoring_params.get(section_key, {})
            rules = section_data.get(data_key, {})
            if isinstance(rules, dict):
                for name, score in rules.items():
                    # 修改: 增加对 processed_signals 的检查，防止重复处理
                    if name.startswith('说明_') or name in processed_signals:
                        continue
                    self._process_playbook(
                        name=name, score=score, playbook_type=get_playbook_type(name),
                        score_type_map=score_type_map, existing_map=existing_playbooks_map,
                        to_create=playbooks_to_create, to_update=playbooks_to_update, processed_signals=processed_signals
                    )

        # 单独解析 composite_scoring (列表结构)
        composite_data = scoring_params.get('composite_scoring', {})
        rules = composite_data.get('rules', [])
        if isinstance(rules, list):
            for rule in rules:
                name = rule.get('name')
                # 修改: 增加对 processed_signals 的检查，防止重复处理
                if not name or name in processed_signals:
                    continue
                score = rule.get('score', 0)
                self._process_playbook(
                    name=name, score=score, playbook_type=get_playbook_type(name),
                    score_type_map=score_type_map, existing_map=existing_playbooks_map,
                    to_create=playbooks_to_create, to_update=playbooks_to_update, processed_signals=processed_signals
                )

        # 解析新的 fused_risk_scoring 结构
        self.stdout.write('  -> 正在解析融合风险信号...')
        fused_risk_data = scoring_params.get('fused_risk_scoring', {})
        risk_categories = fused_risk_data.get('risk_categories', {})
        for risk_dimension, signal_config in risk_categories.items():
            if risk_dimension.startswith('说明_') or not isinstance(signal_config, dict):
                continue
            # 遍历风险维度下的信号配置
            for name, config_value in signal_config.items():
                # 修改: 增加对 processed_signals 的检查，防止重复处理
                if name.startswith('说明_') or name in processed_signals:
                    continue
                # 此处 config_value 是一个类似 {'weight': 1.0, ...} 的字典，而不是分数。
                # 我们只需要确保信号名存在于数据库，因此为其分配默认分数 0。
                self._process_playbook(
                    name=name,
                    score=0,  # 关键修复：将分数硬编码为0
                    playbook_type=Playbook.PlaybookType.RISK,
                    score_type_map=score_type_map,
                    existing_map=existing_playbooks_map,
                    to_create=playbooks_to_create,
                    to_update=playbooks_to_update,
                    processed_signals=processed_signals
                )

        # 解析 critical_exit_params (致命离场信号)
        self.stdout.write('  -> 正在解析致命离场信号...')
        exit_rules = trend_follow_params.get('critical_exit_params', {}).get('signals', {})
        for name, score in exit_rules.items():
            # 修改: 增加对 processed_signals 的检查，防止重复处理
            if not name or name.startswith('说明_') or name in processed_signals:
                continue
            # 对于离场信号，我们明确其类型为EXIT
            self._process_playbook(
                name=name, score=score, playbook_type=Playbook.PlaybookType.EXIT,
                score_type_map=score_type_map, existing_map=existing_playbooks_map,
                to_create=playbooks_to_create, to_update=playbooks_to_update, processed_signals=processed_signals
            )

        # --- 步骤3: 最终检查，确保所有在字典中定义的信号都已同步 ---
        self.stdout.write('  -> 最终检查: 确保所有已定义的信号都存在于数据库中...')
        unprocessed_count = 0
        for name, meta in score_type_map.items():
            # 增加对值类型的检查，确保只处理值为字典的条目，过滤掉所有说明性/注释性条目。
            if name.startswith('说明_') or not isinstance(meta, dict):
                continue
            
            # 如果信号在计分模块中未被处理过，则在此处补上
            if name not in processed_signals:
                unprocessed_count += 1
                # 这些信号没有直接分数，因此默认分数为0
                self._process_playbook(
                    name=name, score=0, playbook_type=get_playbook_type(name),
                    score_type_map=score_type_map, existing_map=existing_playbooks_map,
                    to_create=playbooks_to_create, to_update=playbooks_to_update, processed_signals=processed_signals
                )
        if unprocessed_count > 0:
            self.stdout.write(f'     发现并处理了 {unprocessed_count} 个仅在字典中定义 (无直接分数) 的信号。')

        # --- 步骤4: 执行数据库写入 (在独立的、短小的事务中执行) ---
        if playbooks_to_create:
            with transaction.atomic():
                Playbook.objects.bulk_create(playbooks_to_create)
            self.stdout.write(self.style.SUCCESS(f'✅ 成功创建 {len(playbooks_to_create)} 个新的playbooks。'))

        if playbooks_to_update:
            with transaction.atomic():
                fields_to_update = ['cn_name', 'default_score', 'playbook_type']
                Playbook.objects.bulk_update(playbooks_to_update, fields_to_update)
            self.stdout.write(self.style.SUCCESS(f'✅ 成功更新 {len(playbooks_to_update)} 个现有的playbooks。'))

        if not playbooks_to_create and not playbooks_to_update:
            self.stdout.write(self.style.WARNING('ℹ️ 未发现新的或需要更新的playbooks。数据库已是最新状态。'))
