# 文件: strategies/management/commands/populate_playbooks.py
import json
from django.core.management.base import BaseCommand
from django.db import transaction
from stock_models.stock_analytics import Playbook
from utils.config_loader import load_strategy_config

class Command(BaseCommand):
    help = 'Parses the strategy config and populates/updates the Playbook model.'

    def _process_playbook(self, name, score, playbook_type, score_type_map, existing_map, to_create, to_update, processed_signals):
        """
        【V2.2 查漏补缺版】处理单个playbook的创建或更新逻辑。
        - 新增: 记录已处理的信号名，为后续的“查漏补缺”做准备。
        """
        if not name:
            return

        signal_meta = score_type_map.get(name, {})
        cn_name = signal_meta.get('cn_name', name)

        playbook_obj = existing_map.get(name)

        if playbook_obj:
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
            to_create.append(Playbook(
                name=name,
                cn_name=cn_name,
                playbook_type=playbook_type,
                default_score=score
            ))
        
        # [修改原因] 无论创建还是更新，都将此信号标记为已处理。
        processed_signals.add(name)

    def handle(self, *args, **options):
        """
        【V2.3 定义驱动版】
        - 核心升级: 增加了最终的“查漏补缺”循环。在处理完所有计分信号后，
                    会再次遍历 `score_type_map`，确保那些没有直接分数但有定义的
                    信号（如上下文、中间状态信号）也能被同步到数据库中。
                    这确保了 Playbook 表的完整性。
        """
        self.stdout.write(self.style.SUCCESS('🚀 Starting Playbook population process (V2.3 Definition-Driven)...'))

        try:
            config = load_strategy_config('config/trend_follow_strategy.json')
        except FileNotFoundError:
            self.stderr.write(self.style.ERROR("错误: 无法在 'config/trend_follow_strategy.json' 找到配置文件。请确认文件是否存在。"))
            return
        
        # --- 步骤1: 读取和解析数据 (在事务之外执行) ---
        scoring_params = config.get('strategy_params', {}).get('trend_follow', {}).get('four_layer_scoring_params', {})
        score_type_map = scoring_params.get('score_type_map', {})

        self.stdout.write('  -> Fetching existing playbooks from database...')
        existing_playbooks_map = {p.name: p for p in Playbook.objects.all()}
        self.stdout.write(f'     Found {len(existing_playbooks_map)} existing playbooks.')

        playbooks_to_create = []
        playbooks_to_update = []
        # [修改原因] 初始化一个集合，用于存储所有在计分模块中处理过的信号。
        processed_signals = set()

        self.stdout.write('  -> Parsing offensive playbooks...')
        offensive_sections = {
            'composite_scoring': 'rules',
            'positional_scoring': 'positive_signals',
            'dynamic_scoring': 'positive_signals',
            'trigger_events': 'scoring'
        }
        for section_key, data_key in offensive_sections.items():
            section_data = scoring_params.get(section_key, {})
            rules = section_data.get(data_key, [])
            
            if isinstance(rules, list):
                for rule in rules:
                    name = rule.get('name')
                    score = rule.get('score', 0)
                    self._process_playbook(
                        name=name, score=score, playbook_type=Playbook.PlaybookType.OFFENSIVE,
                        score_type_map=score_type_map, existing_map=existing_playbooks_map,
                        to_create=playbooks_to_create, to_update=playbooks_to_update, processed_signals=processed_signals
                    )
            elif isinstance(rules, dict):
                for name, score in rules.items():
                    if name.startswith('说明_'): continue
                    # [修改原因] 触发器信号在score_type_map中的key不带'trg_'前缀，但在数据库中需要带上以作区分。
                    #           这里的逻辑保持不变，但在后续的查漏补缺中要注意这一点。
                    is_trigger = section_key == 'trigger_events'
                    db_name = name # 默认数据库名与配置文件名一致
                    playbook_type = Playbook.PlaybookType.OFFENSIVE
                    
                    # 从 score_type_map 获取权威类型
                    signal_type_from_map = score_type_map.get(name, {}).get('type')
                    if signal_type_from_map == 'trigger':
                        playbook_type = Playbook.PlaybookType.TRIGGER
                    
                    self._process_playbook(
                        name=db_name, score=score, playbook_type=playbook_type,
                        score_type_map=score_type_map, existing_map=existing_playbooks_map,
                        to_create=playbooks_to_create, to_update=playbooks_to_update, processed_signals=processed_signals
                    )
        
        self.stdout.write('  -> Parsing playbook-specific scores...')
        playbook_scores = scoring_params.get('playbook_scoring', {})
        for name, score in playbook_scores.items():
            if name.startswith('说明_'): continue
            self._process_playbook(
                name=name, score=score, playbook_type=Playbook.PlaybookType.OFFENSIVE,
                score_type_map=score_type_map, existing_map=existing_playbooks_map,
                to_create=playbooks_to_create, to_update=playbooks_to_update, processed_signals=processed_signals
            )

        self.stdout.write('  -> Parsing risk playbooks...')
        warning_rules = scoring_params.get('holding_warning_params', {}).get('signals', {})
        for name, score in warning_rules.items():
            if name and not name.startswith('说明_'):
                self._process_playbook(
                    name=name, score=score, playbook_type=Playbook.PlaybookType.RISK,
                    score_type_map=score_type_map, existing_map=existing_playbooks_map,
                    to_create=playbooks_to_create, to_update=playbooks_to_update, processed_signals=processed_signals
                )

        self.stdout.write('  -> Parsing exit strategies...')
        exit_rules = scoring_params.get('critical_exit_params', {}).get('signals', {})
        for name, score in exit_rules.items():
            if name and not name.startswith('说明_'):
                self._process_playbook(
                    name=name, score=score, playbook_type=Playbook.PlaybookType.EXIT,
                    score_type_map=score_type_map, existing_map=existing_playbooks_map,
                    to_create=playbooks_to_create, to_update=playbooks_to_update, processed_signals=processed_signals
                )

        # [修改原因] 这是本次升级的核心。遍历 `score_type_map` 以确保所有已定义的信号都被同步。
        self.stdout.write('  -> Final check: Ensuring all defined signals exist in database...')
        unprocessed_count = 0
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
        for name, meta in score_type_map.items():
            if name.startswith('说明_'): continue
            
            # 如果信号在计分模块中未被处理过，则在此处补上
            if name not in processed_signals:
                unprocessed_count += 1
                signal_type_str = meta.get('type', 'unknown')
                playbook_type = TYPE_MAP.get(signal_type_str, Playbook.PlaybookType.UNKNOWN)
                
                # 这些信号没有直接分数，因此分数为0
                self._process_playbook(
                    name=name, score=0, playbook_type=playbook_type,
                    score_type_map=score_type_map, existing_map=existing_playbooks_map,
                    to_create=playbooks_to_create, to_update=playbooks_to_update, processed_signals=processed_signals
                )
        if unprocessed_count > 0:
            self.stdout.write(f'     Found and processed {unprocessed_count} definition-only signals (without direct scores).')

        # --- 步骤2: 执行数据库写入 (在独立的、短小的事务中执行) ---
        if playbooks_to_create:
            with transaction.atomic():
                Playbook.objects.bulk_create(playbooks_to_create)
            self.stdout.write(self.style.SUCCESS(f'✅ Successfully CREATED {len(playbooks_to_create)} new playbooks.'))

        if playbooks_to_update:
            with transaction.atomic():
                fields_to_update = ['cn_name', 'default_score', 'playbook_type']
                Playbook.objects.bulk_update(playbooks_to_update, fields_to_update)
            self.stdout.write(self.style.SUCCESS(f'✅ Successfully UPDATED {len(playbooks_to_update)} existing playbooks.'))

        if not playbooks_to_create and not playbooks_to_update:
            self.stdout.write(self.style.WARNING('ℹ️ No new or updated playbooks found. Database is up to date.'))

