# 文件: strategies/management/commands/populate_playbooks.py
import json
from django.core.management.base import BaseCommand
from django.db import transaction
from stock_models.stock_analytics import Playbook
from utils.config_loader import load_strategy_config

class Command(BaseCommand):
    help = 'Parses the strategy config and populates/updates the Playbook model.'

    # --- 代码修改开始 ---
    # [修改原因] V2.1 适配：适配新的 score_type_map 数据结构。
    def _process_playbook(self, name, score, playbook_type, score_type_map, existing_map, to_create, to_update):
        """
        【V2.1 适配版】处理单个playbook的创建或更新逻辑。
        """
        if not name:
            return

        # 从新的信号字典中获取元数据
        signal_meta = score_type_map.get(name, {})
        cn_name = signal_meta.get('cn_name', name) # 如果找不到，则用自身名称作为回退

        playbook_obj = existing_map.get(name)

        if playbook_obj:
            has_changed = False
            if playbook_obj.cn_name != cn_name:
                playbook_obj.cn_name = cn_name
                has_changed = True
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
    # --- 代码修改结束 ---

    @transaction.atomic
    def handle(self, *args, **options):
        """
        【V2.1 信号字典适配版】
        - 核心重构: 从新的 score_type_map 读取元数据。
        """
        self.stdout.write(self.style.SUCCESS('🚀 Starting Playbook population process (V2.1 Signal Dictionary Adapter)...'))

        try:
            config = load_strategy_config('config/trend_follow_strategy.json')
        except FileNotFoundError:
            self.stderr.write(self.style.ERROR("错误: 无法在 'config/trend_follow_strategy.json' 找到配置文件。请确认文件是否存在。"))
            return
        scoring_params = config.get('strategy_params', {}).get('trend_follow', {}).get('four_layer_scoring_params', {})

        score_type_map = scoring_params.get('score_type_map', {})

        self.stdout.write('  -> Fetching existing playbooks from database...')
        existing_playbooks_map = {p.name: p for p in Playbook.objects.all()}
        self.stdout.write(f'     Found {len(existing_playbooks_map)} existing playbooks.')

        playbooks_to_create = []
        playbooks_to_update = []

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
                    # --- 代码修改开始 ---
                    # [修改原因] V2.1 适配：传递 score_type_map
                    self._process_playbook(
                        name=name, score=score, playbook_type=Playbook.PlaybookType.OFFENSIVE,
                        score_type_map=score_type_map, existing_map=existing_playbooks_map,
                        to_create=playbooks_to_create, to_update=playbooks_to_update
                    )
                    # --- 代码修改结束 ---
            elif isinstance(rules, dict):
                for name, score in rules.items():
                    is_trigger = section_key == 'trigger_events'
                    db_name = f'trg_{name}' if is_trigger else name
                    playbook_type = Playbook.PlaybookType.TRIGGER if is_trigger else Playbook.PlaybookType.OFFENSIVE
                    # --- 代码修改开始 ---
                    # [修改原因] V2.1 适配：传递 score_type_map
                    self._process_playbook(
                        name=db_name, score=score, playbook_type=playbook_type,
                        score_type_map=score_type_map, existing_map=existing_playbooks_map,
                        to_create=playbooks_to_create, to_update=playbooks_to_update
                    )
                    # --- 代码修改结束 ---

        self.stdout.write('  -> Parsing risk playbooks...')
        warning_rules = scoring_params.get('holding_warning_params', {}).get('signals', {})
        for name, score in warning_rules.items():
            if name and not name.startswith('说明_'):
                # --- 代码修改开始 ---
                # [修改原因] V2.1 适配：传递 score_type_map
                self._process_playbook(
                    name=name, score=score, playbook_type=Playbook.PlaybookType.RISK,
                    score_type_map=score_type_map, existing_map=existing_playbooks_map,
                    to_create=playbooks_to_create, to_update=playbooks_to_update
                )
                # --- 代码修改结束 ---

        self.stdout.write('  -> Parsing exit strategies...')
        exit_rules = scoring_params.get('critical_exit_params', {}).get('signals', {})
        for name, score in exit_rules.items():
            if name and not name.startswith('说明_'):
                # --- 代码修改开始 ---
                # [修改原因] V2.1 适配：传递 score_type_map
                self._process_playbook(
                    name=name, score=score, playbook_type=Playbook.PlaybookType.EXIT,
                    score_type_map=score_type_map, existing_map=existing_playbooks_map,
                    to_create=playbooks_to_create, to_update=playbooks_to_update
                )
                # --- 代码修改结束 ---

        if playbooks_to_create:
            Playbook.objects.bulk_create(playbooks_to_create)
            self.stdout.write(self.style.SUCCESS(f'✅ Successfully CREATED {len(playbooks_to_create)} new playbooks.'))

        if playbooks_to_update:
            fields_to_update = ['cn_name', 'default_score', 'playbook_type']
            Playbook.objects.bulk_update(playbooks_to_update, fields_to_update)
            self.stdout.write(self.style.SUCCESS(f'✅ Successfully UPDATED {len(playbooks_to_update)} existing playbooks.'))

        if not playbooks_to_create and not playbooks_to_update:
            self.stdout.write(self.style.WARNING('ℹ️ No new or updated playbooks found. Database is up to date.'))


