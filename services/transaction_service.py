# 文件: services/transaction_service.py

from decimal import Decimal
from django.db import transaction as db_transaction
from stock_models.stock_analytics import PositionTracker, Transaction
from tasks.stock_analysis_tasks import rebuild_snapshots_for_tracker_task
import logging

logger = logging.getLogger(__name__)

class TransactionService:
    """
    【V1.0】交易流水核心服务
    - 职责: 处理交易流水的增删改，并确保关联的 PositionTracker 状态正确更新，
            最后触发快照重建。
    """
    @staticmethod
    def recalculate_tracker_state_and_rebuild_snapshots(tracker_id: int):
        """
        核心方法：重新计算持仓状态并触发快照重建。
        这是所有交易变更后都必须调用的方法。
        """
        try:
            with db_transaction.atomic():
                tracker = PositionTracker.objects.select_for_update().get(id=tracker_id)
                transactions = tracker.transactions.order_by('transaction_date')

                current_quantity = Decimal(0)
                total_cost = Decimal(0)
                
                for tx in transactions:
                    if tx.transaction_type == Transaction.TransactionType.BUY:
                        # 重新计算平均成本的核心逻辑
                        total_cost = (current_quantity * tracker.average_cost) + (tx.quantity * tx.price)
                        current_quantity += tx.quantity
                        tracker.average_cost = total_cost / current_quantity if current_quantity > 0 else Decimal(0)
                    elif tx.transaction_type == Transaction.TransactionType.SELL:
                        current_quantity -= tx.quantity
                
                tracker.current_quantity = current_quantity
                
                # 根据最终数量更新状态
                if tracker.current_quantity > 0:
                    tracker.status = PositionTracker.Status.HOLDING
                else:
                    tracker.status = PositionTracker.Status.WATCHING
                    tracker.average_cost = Decimal(0) # 清仓后成本归零
                
                tracker.save()
            
            # 事务成功后，异步触发快照重建
            rebuild_snapshots_for_tracker_task.delay(tracker.id)
            logger.info(f"成功重新计算 Tracker {tracker_id} 的状态并触发快照重建。")
            return True

        except PositionTracker.DoesNotExist:
            logger.error(f"尝试重新计算时未找到 Tracker ID: {tracker_id}")
            return False
        except Exception as e:
            logger.error(f"重新计算 Tracker {tracker_id} 状态时出错: {e}", exc_info=True)
            # 事务会自动回滚
            return False

