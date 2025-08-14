# 文件: strategies/management/commands/populate_playbooks.py
import json
from django.core.management.base import BaseCommand
from django.db import transaction
from stock_models.stock_analytics import Playbook
from utils.config_loader import load_strategy_config

class Command(BaseCommand):
    help = 'Parses the strategy config and populates/updates the Playbook model.'

    def _process_playbook(self, name, score, playbook_type, score_type_map, existing_map, to_create, to_update):
        """
        【V2.1 适配版】处理单个playbook的创建或更新逻辑。
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

    def handle(self, *args, **options):
        """
        【V2.2 死锁修复版】
        - 核心重构: 移除了全局的 @transaction.atomic 装饰器，仅在数据库写入时开启短事务，
                    从根本上解决了数据库死锁 (Deadlock) 问题。
        """
        self.stdout.write(self.style.SUCCESS('🚀 Starting Playbook population process (V2.2 Deadlock Fix)...'))

        try:
            config = load_strategy_config('config/trend_follow_strategy.json')
        except FileNotFoundError:
            self.stderr.write(self.style.ERROR("错误: 无法在 'config/trend_follow_strategy.json' 找到配置文件。请确认文件是否存在。"))
            return
        
        # --- 步骤1: 读取和解析数据 (在事务之外执行) ---
        # 这部分操作不涉及数据库写入，不需要在事务中，可以安全地在外面执行。
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
                    self._process_playbook(
                        name=name, score=score, playbook_type=Playbook.PlaybookType.OFFENSIVE,
                        score_type_map=score_type_map, existing_map=existing_playbooks_map,
                        to_create=playbooks_to_create, to_update=playbooks_to_update
                    )
            elif isinstance(rules, dict):
                for name, score in rules.items():
                    # 过滤掉说明性的键
                    if name.startswith('说明_'): continue
                    is_trigger = section_key == 'trigger_events'
                    db_name = f'trg_{name}' if is_trigger else name
                    playbook_type = Playbook.PlaybookType.TRIGGER if is_trigger else Playbook.PlaybookType.OFFENSIVE
                    self._process_playbook(
                        name=db_name, score=score, playbook_type=playbook_type,
                        score_type_map=score_type_map, existing_map=existing_playbooks_map,
                        to_create=playbooks_to_create, to_update=playbooks_to_update
                    )
        # [修改原因] 新增对 'playbook_scoring' 配置块的解析，确保新剧本能被同步。
        self.stdout.write('  -> Parsing playbook-specific scores...')
        playbook_scores = scoring_params.get('playbook_scoring', {})
        for name, score in playbook_scores.items():
            if name.startswith('说明_'): continue
            # 所有 playbook_scoring 下的信号都属于进攻战法
            self._process_playbook(
                name=name,
                score=score,
                playbook_type=Playbook.PlaybookType.OFFENSIVE,
                score_type_map=score_type_map,
                existing_map=existing_playbooks_map,
                to_create=playbooks_to_create,
                to_update=playbooks_to_update
            )
        self.stdout.write('  -> Parsing risk playbooks...')
        warning_rules = scoring_params.get('holding_warning_params', {}).get('signals', {})
        for name, score in warning_rules.items():
            if name and not name.startswith('说明_'):
                self._process_playbook(
                    name=name, score=score, playbook_type=Playbook.PlaybookType.RISK,
                    score_type_map=score_type_map, existing_map=existing_playbooks_map,
                    to_create=playbooks_to_create, to_update=playbooks_to_update
                )

        self.stdout.write('  -> Parsing exit strategies...')
        exit_rules = scoring_params.get('critical_exit_params', {}).get('signals', {})
        for name, score in exit_rules.items():
            if name and not name.startswith('说明_'):
                self._process_playbook(
                    name=name, score=score, playbook_type=Playbook.PlaybookType.EXIT,
                    score_type_map=score_type_map, existing_map=existing_playbooks_map,
                    to_create=playbooks_to_create, to_update=playbooks_to_update
                )

        # --- 步骤2: 执行数据库写入 (在独立的、短小的事务中执行) ---
        if playbooks_to_create:
            # 仅在创建时开启事务
            with transaction.atomic():
                Playbook.objects.bulk_create(playbooks_to_create)
            self.stdout.write(self.style.SUCCESS(f'✅ Successfully CREATED {len(playbooks_to_create)} new playbooks.'))

        if playbooks_to_update:
            # 仅在更新时开启事务
            with transaction.atomic():
                fields_to_update = ['cn_name', 'default_score', 'playbook_type']
                Playbook.objects.bulk_update(playbooks_to_update, fields_to_update)
            self.stdout.write(self.style.SUCCESS(f'✅ Successfully UPDATED {len(playbooks_to_update)} existing playbooks.'))

        if not playbooks_to_create and not playbooks_to_update:
            self.stdout.write(self.style.WARNING('ℹ️ No new or updated playbooks found. Database is up to date.'))
