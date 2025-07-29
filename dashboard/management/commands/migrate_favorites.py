# dashboard/management/commands/migrate_favorites.py
from django.core.management.base import BaseCommand
from django.db.models import Max, OuterRef, Subquery
from users.models import FavoriteStock
from stock_models.stock_analytics import FavoriteStockTracker
from stock_models.stock_analytics import TrendFollowStrategySignalLog
from django.contrib.auth import get_user_model

class Command(BaseCommand):
    help = 'Migrates data from old FavoriteStock to new FavoriteStockTracker model.'

    def handle(self, *args, **options):
        self.stdout.write(self.style.SUCCESS('开始从 FavoriteStock 迁移数据到 FavoriteStockTracker...'))
        
        User = get_user_model()
        all_users = User.objects.all()
        
        total_migrated = 0
        
        for user in all_users:
            # 获取该用户的所有旧自选股
            old_favorites = FavoriteStock.objects.filter(user=user).select_related('stock')
            
            if not old_favorites.exists():
                continue
                
            self.stdout.write(f"  正在处理用户: {user.username} ({old_favorites.count()} 个自选股)")
            
            for fav in old_favorites:
                stock = fav.stock
                
                # 为这只股票找到最新的一个“买入”信号作为建仓依据
                latest_buy_log = TrendFollowStrategySignalLog.objects.filter(
                    stock=stock,
                    entry_signal=True
                ).order_by('-trade_time').first()
                
                if not latest_buy_log:
                    self.stdout.write(self.style.WARNING(f"    -> 股票 {stock.stock_code} 找不到任何买入信号，跳过迁移。"))
                    continue
                    
                # 创建或更新追踪器
                tracker, created = FavoriteStockTracker.objects.update_or_create(
                    user=user,
                    stock=stock,
                    entry_log=latest_buy_log,
                    defaults={
                        'status': 'HOLDING',
                        'entry_price': latest_buy_log.close_price,
                        'entry_date': latest_buy_log.trade_time,
                        'entry_score': latest_buy_log.entry_score or 0.0,
                        'latest_log': latest_buy_log,
                        'latest_price': latest_buy_log.close_price,
                        'latest_date': latest_buy_log.trade_time,
                    }
                )
                
                if created:
                    total_migrated += 1
                    self.stdout.write(self.style.SUCCESS(f"    -> 成功为 {stock.stock_code} 创建了追踪器。"))

        self.stdout.write(self.style.SUCCESS(f'\n迁移完成！共创建了 {total_migrated} 条新的追踪记录。'))

