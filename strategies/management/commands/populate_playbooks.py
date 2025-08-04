# 文件: strategies/management/commands/populate_playbooks.py
import json
from django.core.management.base import BaseCommand
from django.db import transaction
from stock_models.stock_analytics import Playbook
from utils.config_loader import load_strategy_config

class Command(BaseCommand):
    help = 'Parses the strategy config and populates/updates the Playbook model.'

    def _process_playbook(self, name, cn_name, score, playbook_type, existing_map, to_create, to_update):
        """
        【V2.0 新增】处理单个playbook的创建或更新逻辑。
        """
        if not name:
            return

        # 从内存中的字典获取现有对象
        playbook_obj = existing_map.get(name)

        if playbook_obj:
            # --- 更新逻辑 ---
            has_changed = False
            # 比较中文名
            if playbook_obj.cn_name != cn_name:
                playbook_obj.cn_name = cn_name
                has_changed = True
            # 比较默认分 (转换为浮点数以避免类型问题)
            if float(playbook_obj.default_score) != float(score):
                playbook_obj.default_score = score
                has_changed = True
            # 比较类型
            if playbook_obj.playbook_type != playbook_type:
                playbook_obj.playbook_type = playbook_type
                has_changed = True
            
            if has_changed:
                to_update.append(playbook_obj)
        else:
            # --- 创建逻辑 ---
            to_create.append(Playbook(
                name=name,
                cn_name=cn_name,
                playbook_type=playbook_type,
                default_score=score
            ))

    @transaction.atomic
    def handle(self, *args, **options):
        """
        【V2.0 Create or Update 版】
        - 核心重构: 实现 playbook 的创建与更新双重功能。
        - 性能优化: 采用“一次查询，内存比对，批量操作”模式，效率极高。
        """
        self.stdout.write(self.style.SUCCESS('🚀 Starting Playbook population process (V2.0 Create/Update)...'))

        # 加载配置文件
        config = load_strategy_config('config/trend_follow_strategy.json')
        scoring_params = config.get('strategy_params', {}).get('trend_follow', {}).get('four_layer_scoring_params', {})
        metadata = scoring_params.get('metadata', {})

        # 1. 一次性获取所有现有playbook到内存字典
        self.stdout.write('  -> Fetching existing playbooks from database...')
        existing_playbooks_map = {p.name: p for p in Playbook.objects.all()}
        self.stdout.write(f'     Found {len(existing_playbooks_map)} existing playbooks.')

        playbooks_to_create = []
        playbooks_to_update = []

        # 2. 解析进攻战法 (Composite, Positional, Dynamic, Trigger)
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
            
            if isinstance(rules, list): # for composite_scoring
                for rule in rules:
                    name = rule.get('name')
                    score = rule.get('score', 0)
                    self._process_playbook(
                        name=name,
                        cn_name=metadata.get(name, name),
                        score=score,
                        playbook_type=Playbook.PlaybookType.OFFENSIVE,
                        existing_map=existing_playbooks_map,
                        to_create=playbooks_to_create,
                        to_update=playbooks_to_update
                    )
            elif isinstance(rules, dict): # for other sections
                for name, score in rules.items():
                    is_trigger = section_key == 'trigger_events'
                    db_name = f'trg_{name}' if is_trigger else name
                    playbook_type = Playbook.PlaybookType.TRIGGER if is_trigger else Playbook.PlaybookType.OFFENSIVE
                    self._process_playbook(
                        name=db_name,
                        cn_name=metadata.get(db_name, metadata.get(name, name)),
                        score=score,
                        playbook_type=playbook_type,
                        existing_map=existing_playbooks_map,
                        to_create=playbooks_to_create,
                        to_update=playbooks_to_update
                    )

        # 3. 解析风险剧本 (WarningLayer)
        self.stdout.write('  -> Parsing risk playbooks...')
        warning_rules = scoring_params.get('holding_warning_params', {}).get('signals', {})
        for name, score in warning_rules.items():
            if name and not name.startswith('说明_'):
                self._process_playbook(
                    name=name,
                    cn_name=metadata.get(name, name),
                    score=score,
                    playbook_type=Playbook.PlaybookType.RISK,
                    existing_map=existing_playbooks_map,
                    to_create=playbooks_to_create,
                    to_update=playbooks_to_update
                )

        # 4. 解析离场策略 (ExitLayer)
        self.stdout.write('  -> Parsing exit strategies...')
        exit_rules = scoring_params.get('critical_exit_params', {}).get('signals', {})
        for name, score in exit_rules.items():
            if name and not name.startswith('说明_'):
                self._process_playbook(
                    name=name,
                    cn_name=metadata.get(name, name),
                    score=score,
                    playbook_type=Playbook.PlaybookType.EXIT,
                    existing_map=existing_playbooks_map,
                    to_create=playbooks_to_create,
                    to_update=playbooks_to_update
                )

        # 5. 执行批量数据库操作
        if playbooks_to_create:
            Playbook.objects.bulk_create(playbooks_to_create)
            self.stdout.write(self.style.SUCCESS(f'✅ Successfully CREATED {len(playbooks_to_create)} new playbooks.'))

        if playbooks_to_update:
            # 指定要更新的字段
            fields_to_update = ['cn_name', 'default_score', 'playbook_type']
            Playbook.objects.bulk_update(playbooks_to_update, fields_to_update)
            self.stdout.write(self.style.SUCCESS(f'✅ Successfully UPDATED {len(playbooks_to_update)} existing playbooks.'))

        if not playbooks_to_create and not playbooks_to_update:
            self.stdout.write(self.style.WARNING('ℹ️ No new or updated playbooks found. Database is up to date.'))
