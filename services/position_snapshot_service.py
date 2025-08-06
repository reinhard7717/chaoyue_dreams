# 文件: services/position_snapshot_service.py

import asyncio
from datetime import date, timedelta
from decimal import Decimal
import logging
from asgiref.sync import sync_to_async

from dao_manager.tushare_daos.strategies_dao import StrategiesDAO
from stock_models.stock_analytics import PositionTracker, Transaction, DailyPositionSnapshot, StrategyDailyScore
from utils.cache_manager import CacheManager

logger = logging.getLogger(__name__)

class PositionSnapshotService:
    """
    【V1.0】持仓快照核心服务引擎
    - 职责: 为单个 PositionTracker 计算并生成其完整的历史快照。
    """
    def __init__(self, cache_manager: CacheManager):
        self.strategies_dao = StrategiesDAO(cache_manager)

    async def rebuild_snapshots_for_tracker(self, tracker_id: int):
        """
        为指定的 PositionTracker 重建所有历史快照。
        这是本服务的核心入口方法。
        """
        try:
            # 1. 获取基础数据
            tracker = await self._get_tracker(tracker_id)
            if not tracker:
                logger.warning(f"Tracker ID {tracker_id} 不存在，无法重建快照。")
                return 0
            
            transactions = await self._get_transactions(tracker_id)
            if not transactions:
                logger.info(f"Tracker ID {tracker_id} ({tracker.stock.stock_code}) 没有任何交易流水，无需生成快照。")
                # 清理可能存在的旧快照
                await self._delete_existing_snapshots(tracker)
                return 0

            # 2. 准备数据
            start_date = transactions[-1].transaction_date.date() # 最早的交易
            end_date = date.today()
            
            # 批量获取行情和分数数据
            price_map = await self._get_price_map(tracker.stock.stock_code, start_date, end_date)
            score_map = await self._get_score_map(tracker.stock.stock_code, start_date, end_date)

            # 3. 核心计算：逐日生成快照
            snapshots_to_create = []
            current_date = start_date
            tx_idx = len(transactions) - 1 # 从最早的交易开始

            # 在循环外初始化，它们代表了“截止到当天开始时”的状态
            current_quantity = Decimal(0)
            total_cost = Decimal(0)

            while current_date <= end_date:
                # a. 处理当天的所有交易，更新持仓状态
                while tx_idx >= 0 and transactions[tx_idx].transaction_date.date() == current_date:
                    tx = transactions[tx_idx]
                    if tx.transaction_type == Transaction.TransactionType.BUY:
                        current_quantity += tx.quantity
                        total_cost += tx.quantity * tx.price
                    elif tx.transaction_type == Transaction.TransactionType.SELL:
                        current_quantity -= tx.quantity
                    tx_idx -= 1
                
                average_cost = (total_cost / current_quantity) if current_quantity > 0 else Decimal(0)

                # b. 如果当天结束后仍在持仓，则生成快照
                if current_quantity > 0 and current_date in price_map:
                    close_price = Decimal(str(price_map[current_date]))
                    daily_score = score_map.get(current_date)
                    
                    profit_loss = (close_price - average_cost) * current_quantity
                    profit_loss_pct = ((close_price / average_cost) - 1) * 100 if average_cost > 0 else Decimal(0)

                    snapshots_to_create.append(DailyPositionSnapshot(
                        tracker=tracker,
                        snapshot_date=current_date,
                        close_price=close_price,
                        quantity_at_snapshot=current_quantity,
                        profit_loss=profit_loss,
                        profit_loss_pct=profit_loss_pct,
                        daily_score=daily_score
                    ))
                
                current_date += timedelta(days=1)

            # 4. 写入数据库
            if snapshots_to_create:
                await self._delete_existing_snapshots(tracker)
                await self._bulk_create_snapshots(snapshots_to_create)
                logger.info(f"成功为 Tracker ID {tracker_id} ({tracker.stock.stock_code}) 创建/更新了 {len(snapshots_to_create)} 条快照。")
                return len(snapshots_to_create)
            
            return 0

        except Exception as e:
            logger.error(f"为 Tracker ID {tracker_id} 重建快照时发生严重错误: {e}", exc_info=True)
            return 0

    # --- 异步辅助方法 ---
    @sync_to_async(thread_sensitive=True)
    def _get_tracker(self, tracker_id):
        try:
            return PositionTracker.objects.select_related('stock').get(id=tracker_id)
        except PositionTracker.DoesNotExist:
            return None

    @sync_to_async(thread_sensitive=True)
    def _get_transactions(self, tracker_id):
        # 按日期升序排列，方便从头开始计算
        return list(Transaction.objects.filter(tracker_id=tracker_id).order_by('transaction_date'))

    async def _get_price_map(self, stock_code, start_date, end_date):
        df = await self.strategies_dao.get_daily_data(stock_code, start_date.strftime('%Y%m%d'), end_date.strftime('%Y%m%d'))
        return {row.name.date(): row['close'] for _, row in df.iterrows()} if not df.empty else {}

    @sync_to_async(thread_sensitive=True)
    def _get_score_map(self, stock_code, start_date, end_date):
        scores = StrategyDailyScore.objects.filter(
            stock__stock_code=stock_code,
            trade_date__gte=start_date,
            trade_date__lte=end_date
        )
        return {s.trade_date: s for s in scores}

    @sync_to_async(thread_sensitive=True)
    def _delete_existing_snapshots(self, tracker):
        DailyPositionSnapshot.objects.filter(tracker=tracker).delete()

    @sync_to_async(thread_sensitive=True)
    def _bulk_create_snapshots(self, snapshots):
        DailyPositionSnapshot.objects.bulk_create(snapshots, batch_size=500)

