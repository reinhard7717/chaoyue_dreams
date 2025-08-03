# 文件: strategies/management/commands/populate_playbooks.py
import json
from django.core.management.base import BaseCommand
from django.db import transaction
from strategies.trend_following.offensive_layer import OffensiveLayer
from strategies.trend_following.warning_layer import WarningLayer
from strategies.trend_following.exit_layer import ExitLayer
from signals.models import Playbook # 假设您的新模型在 signals app 中
from utils.config_loader import load_strategy_config

class Command(BaseCommand):
    help = 'Parses the strategy config file and populates the Playbook model.'

    @transaction.atomic
    def handle(self, *args, **options):
        """
        主处理函数，执行填充逻辑。
        """
        self.stdout.write(self.style.SUCCESS('🚀 Starting Playbook population process...'))

        # 加载配置文件
        config = load_strategy_config('config/trend_follow_strategy.json')
        scoring_params = config.get('strategy_params', {}).get('trend_follow', {}).get('four_layer_scoring_params', {})
        metadata = scoring_params.get('metadata', {})

        playbooks_to_create = []
        existing_playbooks = set(Playbook.objects.values_list('name', flat=True))

        # 1. 解析进攻战法 (Composite, Positional, Dynamic, Trigger)
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
                    if name and name not in existing_playbooks:
                        playbooks_to_create.append(Playbook(
                            name=name,
                            cn_name=metadata.get(name, name),
                            playbook_type=Playbook.PlaybookType.OFFENSIVE,
                            default_score=score
                        ))
                        existing_playbooks.add(name)
            elif isinstance(rules, dict): # for other sections
                for name, score in rules.items():
                    # 触发器有特殊前缀，需要处理
                    db_name = f'trg_{name}' if section_key == 'trigger_events' else name
                    if db_name and db_name not in existing_playbooks:
                        playbooks_to_create.append(Playbook(
                            name=db_name,
                            cn_name=metadata.get(db_name, metadata.get(name, name)),
                            playbook_type=Playbook.PlaybookType.TRIGGER if section_key == 'trigger_events' else Playbook.PlaybookType.OFFENSIVE,
                            default_score=score
                        ))
                        existing_playbooks.add(db_name)

        # 2. 解析风险剧本 (WarningLayer)
        self.stdout.write('  -> Parsing risk playbooks...')
        warning_rules = scoring_params.get('holding_warning_params', {}).get('signals', {})
        for name, score in warning_rules.items():
            if name and not name.startswith('说明_') and name not in existing_playbooks:
                playbooks_to_create.append(Playbook(
                    name=name,
                    cn_name=metadata.get(name, name),
                    playbook_type=Playbook.PlaybookType.RISK,
                    default_score=score
                ))
                existing_playbooks.add(name)

        # 3. 解析离场策略 (ExitLayer)
        self.stdout.write('  -> Parsing exit strategies...')
        exit_rules = scoring_params.get('critical_exit_params', {}).get('signals', {})
        for name, score in exit_rules.items():
            if name and not name.startswith('说明_') and name not in existing_playbooks:
                playbooks_to_create.append(Playbook(
                    name=name,
                    cn_name=metadata.get(name, name),
                    playbook_type=Playbook.PlaybookType.EXIT,
                    default_score=score
                ))
                existing_playbooks.add(name)

        if playbooks_to_create:
            Playbook.objects.bulk_create(playbooks_to_create)
            self.stdout.write(self.style.SUCCESS(f'✅ Successfully created {len(playbooks_to_create)} new playbooks.'))
        else:
            self.stdout.write(self.style.WARNING('ℹ️ No new playbooks found to create. Database is up to date.'))

