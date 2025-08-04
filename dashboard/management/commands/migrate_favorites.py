# dashboard/management/commands/migrate_favorites.py
from django.core.management.base import BaseCommand
from django.contrib.auth import get_user_model
from users.models import FavoriteStock
from stock_models.stock_analytics import PositionTracker, TradingSignal

class Command(BaseCommand):
    help = '从旧的 FavoriteStock 模型迁移数据到新的 PositionTracker 模型。'

    def handle(self, *args, **options):
        self.stdout.write(self.style.SUCCESS('开始从 FavoriteStock 迁移数据到 PositionTracker...'))
        
        User = get_user_model()
        all_users = User.objects.all()
        
        total_migrated = 0
        total_skipped_no_signal = 0
        
        for user in all_users:
            # 1. 获取该用户的所有旧自选股
            old_favorites = FavoriteStock.objects.filter(user=user).select_related('stock')
            
            if not old_favorites.exists():
                continue
                
            self.stdout.write(f"  正在处理用户: {user.username} ({old_favorites.count()} 个自选股)")
            
            for fav in old_favorites:
                stock = fav.stock
                
                # --- 代码修改开始 ---
                # [修改原因] 适配新的 TradingSignal 模型
                # 2. 为这只股票找到最新的一个“买入”信号作为建仓依据
                latest_buy_signal = TradingSignal.objects.filter(
                    stock=stock,
                    signal_type=TradingSignal.SignalType.BUY # 使用新的信号类型字段
                ).order_by('-trade_time').first()
                
                if not latest_buy_signal:
                    self.stdout.write(self.style.WARNING(f"    -> 股票 {stock.stock_code} 在 TradingSignal 中找不到任何买入信号，跳过迁移。"))
                    total_skipped_no_signal += 1
                    continue
                    
                # 3. 创建或更新 PositionTracker 记录
                #    使用 update_or_create 避免重复迁移
                tracker, created = PositionTracker.objects.update_or_create(
                    user=user,
                    stock=stock,
                    entry_signal=latest_buy_signal, # 确保每个建仓信号只创建一个追踪器
                    defaults={
                        'status': PositionTracker.Status.HOLDING,
                        'entry_price': latest_buy_signal.close_price, # 从新信号模型获取收盘价
                        'entry_date': latest_buy_signal.trade_time,   # 从新信号模型获取交易时间
                    }
                )
                # --- 代码修改结束 ---
                
                if created:
                    total_migrated += 1
                    self.stdout.write(self.style.SUCCESS(f"    -> 成功为 {stock.stock_code} 创建了持仓追踪器 (PositionTracker)。"))
                else:
                    self.stdout.write(self.style.NOTICE(f"    -> 股票 {stock.stock_code} 的追踪器已存在，已使用最新数据更新。"))

        self.stdout.write(self.style.SUCCESS(f'\n迁移完成！'))
        self.stdout.write(self.style.SUCCESS(f'  共创建了 {total_migrated} 条新的追踪记录。'))
        self.stdout.write(self.style.WARNING(f'  共跳过了 {total_skipped_no_signal} 条因无买入信号而无法迁移的记录。'))

