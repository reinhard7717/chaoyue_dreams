# dashboard/management/commands/migrate_favorites.py
from django.core.management.base import BaseCommand
from django.contrib.auth import get_user_model
from users.models import FavoriteStock
from stock_models.stock_analytics import PositionTracker, TradingSignal

class Command(BaseCommand):
    help = '【V2.1 兼容V4.0模型版】从 FavoriteStock 迁移数据到 PositionTracker，为所有自选股创建追踪器。' # 更新帮助信息
    def handle(self, *args, **options):
        self.stdout.write(self.style.SUCCESS('开始从 FavoriteStock 迁移数据到 PositionTracker (V2.1)...')) # 更新版本号
        User = get_user_model()
        all_users = User.objects.all()
        total_created_holding = 0
        total_created_watching = 0
        total_skipped_exists = 0
        for user in all_users:
            old_favorites = FavoriteStock.objects.filter(user=user).select_related('stock')
            if not old_favorites.exists():
                continue
            self.stdout.write(f"  正在处理用户: {user.username} ({old_favorites.count()} 个自选股)")
            for fav in old_favorites:
                stock = fav.stock
                # 步骤1: 检查是否已存在该用户对该股票的追踪器，避免重复创建
                if PositionTracker.objects.filter(user=user, stock=stock).exists():
                    self.stdout.write(self.style.NOTICE(f"    -> 股票 {stock.stock_code} 的追踪器已存在，跳过。"))
                    total_skipped_exists += 1
                    continue
                # 步骤2: 尝试寻找最新的买入信号，这是“理想情况”
                latest_buy_signal = TradingSignal.objects.filter(
                    stock=stock,
                    signal_type=TradingSignal.SignalType.BUY
                ).order_by('-trade_time').first()
                # 步骤3: 根据是否找到买入信号，决定创建哪种状态的追踪器
                if latest_buy_signal:
                    # 情况A: 找到了买入信号，创建“持仓中”的追踪器
                    PositionTracker.objects.create(
                        user=user,
                        stock=stock,
                        status=PositionTracker.Status.HOLDING,
                        # 移除: 'entry_signal' 字段已在V4.0模型中移除
                        average_cost=latest_buy_signal.close_price, # 'entry_price' 字段已更新为 'average_cost'
                        # 移除: 'entry_date' 字段已在V4.0模型中移除
                        current_quantity=100,  # 'quantity' 字段已更名为 'current_quantity'
                    )
                    total_created_holding += 1
                    self.stdout.write(self.style.SUCCESS(f"    -> 成功为 {stock.stock_code} 创建了 [持仓中] 追踪器。"))
                else:
                    # 情况B: 未找到买入信号，创建“观察中”的追踪器
                    PositionTracker.objects.create(
                        user=user,
                        stock=stock,
                        status=PositionTracker.Status.WATCHING,
                        current_quantity=0,  # 观察中的记录，持仓量应为0
                        average_cost=0,      # 观察中的记录，成本应为0
                    )
                    total_created_watching += 1
                    self.stdout.write(self.style.SUCCESS(f"    -> 为 {stock.stock_code} 创建了 [观察中] 追踪器。"))
        self.stdout.write(self.style.SUCCESS(f'\n迁移完成！'))
        self.stdout.write(self.style.SUCCESS(f'  共创建了 {total_created_holding} 条 [持仓中] 追踪记录。'))
        self.stdout.write(self.style.SUCCESS(f'  共创建了 {total_created_watching} 条 [观察中] 追踪记录。'))
        self.stdout.write(self.style.WARNING(f'  共跳过了 {total_skipped_exists} 条已存在的记录。'))
