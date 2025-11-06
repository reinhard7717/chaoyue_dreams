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
        【V1.1 修正版】核心方法：重新计算持仓状态并触发快照重建。
        这是所有交易变更后都必须调用的方法。
        - 修正了平均成本的计算逻辑，确保在循环中正确迭代，而不是使用旧的 tracker.average_cost。
        - 增加了按创建时间排序，保证同一天内交易的顺序正确性。
        """
        try:
            with db_transaction.atomic():
                tracker = PositionTracker.objects.select_for_update().get(id=tracker_id)
                # 增加按 created_at 排序，确保同一天内录入的交易顺序正确
                transactions = tracker.transactions.order_by('transaction_date', 'created_at')
                current_quantity = Decimal(0)
                total_cost = Decimal(0)
                # 使用一个在循环中迭代的平均成本变量
                running_avg_cost = Decimal(0)
                print(f"开始为 Tracker ID: {tracker_id} 重新计算状态...") # 调试信息
                for tx in transactions:
                    if tx.transaction_type == Transaction.TransactionType.BUY:
                        # 修正了加权平均成本的计算逻辑
                        total_cost = (current_quantity * running_avg_cost) + (tx.quantity * tx.price)
                        current_quantity += tx.quantity
                        if current_quantity > 0:
                            running_avg_cost = total_cost / current_quantity
                        else:
                            running_avg_cost = Decimal(0)
                        print(f"  [买入] 日期: {tx.transaction_date.date()}, 数量: {tx.quantity}, 价格: {tx.price}, 新数量: {current_quantity}, 新成本: {running_avg_cost:.2f}") # 调试信息
                    elif tx.transaction_type == Transaction.TransactionType.SELL:
                        current_quantity -= tx.quantity
                        print(f"  [卖出] 日期: {tx.transaction_date.date()}, 数量: {tx.quantity}, 新数量: {current_quantity}") # 调试信息
                        # 如果持仓被卖光，则将成本和数量都归零
                        if current_quantity <= 0:
                            current_quantity = Decimal(0)
                            total_cost = Decimal(0)
                            running_avg_cost = Decimal(0)
                tracker.current_quantity = current_quantity
                # 使用循环计算出的最终成本
                tracker.average_cost = running_avg_cost
                # 根据最终数量更新状态
                if tracker.current_quantity > 0:
                    tracker.status = PositionTracker.Status.HOLDING
                else:
                    tracker.status = PositionTracker.Status.WATCHING
                    tracker.average_cost = Decimal(0) # 清仓后成本归零
                print(f"计算完成。最终状态: 数量={tracker.current_quantity}, 成本={tracker.average_cost:.2f}, 状态={tracker.status}") # 调试信息
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
