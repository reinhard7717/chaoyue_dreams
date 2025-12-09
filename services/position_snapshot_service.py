# 文件: services/position_snapshot_service.py

import asyncio
from datetime import date, timedelta
from decimal import Decimal
import logging
from asgiref.sync import sync_to_async
from stock_models.index import TradeCalendar
from dao_manager.tushare_daos.strategies_dao import StrategiesDAO
from dao_manager.tushare_daos.stock_time_trade_dao import StockTimeTradeDAO
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
        self.stock_time_trade_dao = StockTimeTradeDAO(cache_manager)
    async def rebuild_snapshots_for_tracker(self, tracker_id: int):
        try:
            tracker = await self._get_tracker(tracker_id)
            if not tracker:
                logger.warning(f"Tracker ID {tracker_id} 不存在，无法重建快照。")
                return 0
            transactions = await self._get_transactions(tracker_id)
            if not transactions:
                await self._delete_existing_snapshots(tracker)
                logger.info(f"Tracker ID {tracker_id} ({tracker.stock.stock_code}) 没有任何交易流水，已清理旧快照。")
                return 0
            start_date = transactions[0].transaction_date.date()
            latest_trade_date = await sync_to_async(TradeCalendar.get_latest_trade_date)()
            end_date = latest_trade_date if latest_trade_date else date.today()
            logger.info(f"Tracker {tracker_id}: 准备重建快照，日期范围 {start_date} 到 {end_date}")
            price_map = await self._get_price_map(tracker.stock.stock_code, start_date, end_date)
            if not price_map:
                logger.warning(f"Tracker {tracker_id}: 未能获取到股票 {tracker.stock.stock_code} 在 {start_date} 到 {end_date} 期间的任何行情数据。")
                return 0
            score_map = await self._get_score_map(tracker.stock.stock_code, start_date, end_date)
            snapshots_to_create = []
            current_date = start_date
            tx_idx = len(transactions) - 1
            current_quantity = Decimal(0)
            total_cost = Decimal(0)
            average_cost = Decimal(0)
            while current_date <= end_date:
                while tx_idx >= 0 and transactions[tx_idx].transaction_date.date() == current_date:
                    tx = transactions[tx_idx]
                    if tx.transaction_type == Transaction.TransactionType.BUY:
                        # 这里的成本计算逻辑在我的上一个回复中是有误的，现已修正
                        # 正确的移动平均成本计算
                        total_value = average_cost * current_quantity
                        new_total_value = total_value + (tx.quantity * tx.price)
                        current_quantity += tx.quantity
                        average_cost = new_total_value / current_quantity if current_quantity > 0 else Decimal(0)
                    elif tx.transaction_type == Transaction.TransactionType.SELL:
                        current_quantity -= tx.quantity
                    tx_idx -= 1
                if current_quantity > 0:
                    close_price_obj = price_map.get(current_date)
                    if close_price_obj:
                        close_price = Decimal(str(close_price_obj))
                        daily_score = score_map.get(current_date)
                        profit_loss = (close_price - average_cost) * current_quantity
                        profit_loss_pct = ((close_price / average_cost) - 1) if average_cost > 0 else Decimal(0)
                        snapshots_to_create.append(DailyPositionSnapshot(
                            tracker=tracker,
                            snapshot_date=current_date,
                            close_price=close_price,
                            quantity_at_snapshot=current_quantity,
                            profit_loss=profit_loss,
                            profit_loss_pct=profit_loss_pct * 100, # 转换为百分比
                            daily_score=daily_score
                        ))
                    else:
                        # 如果当天是交易日但没有行情（例如停牌），则跳过快照生成
                        is_trade_day = await sync_to_async(TradeCalendar.objects.filter(cal_date=current_date, is_open=True).exists)()
                        if is_trade_day:
                            logger.info(f"Tracker {tracker_id}: {current_date} 是交易日，但未找到收盘价，可能停牌。跳过快照。")
                current_date += timedelta(days=1)
            if snapshots_to_create:
                await self._delete_existing_snapshots(tracker)
                await self._bulk_create_snapshots(snapshots_to_create)
                logger.info(f"成功为 Tracker ID {tracker_id} ({tracker.stock.stock_code}) 创建/更新了 {len(snapshots_to_create)} 条快照。")
                return len(snapshots_to_create)
            else:
                logger.warning(f"Tracker {tracker_id}: 计算完成，但没有生成任何快照记录。")
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
        # 按日期升序排列，这是正确的
        return list(Transaction.objects.filter(tracker_id=tracker_id).order_by('transaction_date'))
    async def _get_price_map(self, stock_code, start_date, end_date):
        df = await self.stock_time_trade_dao.get_daily_data(stock_code, start_date.strftime('%Y%m%d'), end_date.strftime('%Y%m%d'))
        return {row.name.date(): row['close'] for _, row in df.iterrows()} if not df.empty else {}
    @sync_to_async(thread_sensitive=True)
    def _get_score_map(self, stock_code, start_date, end_date):
        scores = StrategyDailyScore.objects.filter(
            stock__stock_code=stock_code,
            trade_date__gte=start_date,
            trade_date__lte=end_date
        ).select_related('stock') # 优化
        return {s.trade_date: s for s in scores}
    @sync_to_async(thread_sensitive=True)
    def _delete_existing_snapshots(self, tracker):
        DailyPositionSnapshot.objects.filter(tracker=tracker).delete()
    @sync_to_async(thread_sensitive=True)
    def _bulk_create_snapshots(self, snapshots):
        DailyPositionSnapshot.objects.bulk_create(snapshots, batch_size=500)

