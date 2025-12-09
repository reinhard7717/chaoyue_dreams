# dashboard/tasks.py
import asyncio
from asgiref.sync import async_to_sync
import logging
from django.contrib.auth import get_user_model
from channels.layers import get_channel_layer
from dao_manager.tushare_daos.strategies_dao import StrategiesDAO
from utils.cache_manager import CacheManager
from utils.websockets import send_update_to_user_sync, broadcast_public_message_sync # 导入同步发送函数
from chaoyue_dreams.celery import app as celery_app
import time
from django.db.models import OuterRef, Subquery
from users.models import FavoriteStock
from dao_manager.tushare_daos.stock_basic_info_dao import StockBasicInfoDao
from dao_manager.tushare_daos.realtime_data_dao import StockRealtimeDAO


logger = logging.getLogger(__name__)
User = get_user_model()

@celery_app.task(bind=True, name='dashboard.tasks.send_update_to_user_task_celery', queue='dashboard')
def send_update_to_user_task_celery(self, user_id: int, sub_type: str, payload: dict):
    """
    Celery异步任务，调用WebSocketSender类的send_update_to_user_task方法推送消息。
    """
    send_update_to_user_sync(user_id, sub_type, payload)

@celery_app.task(bind=True, name='dashboard.tasks.push_realtime_updates_for_stocks')
def push_realtime_updates_for_stocks(updated_stock_codes: list):
    """
    Celery任务：根据更新的股票代码列表，找到相关用户，获取其自选股最新数据并通过WebSocket推送。
    """
    from users.models import FavoriteStock
    from dashboard.tasks import send_update_to_user_task_celery
    if not updated_stock_codes:
        logger.info("没有更新的股票代码，跳过推送任务。")
        return
    channel_layer = get_channel_layer()
    if channel_layer is None:
        logger.error("无法获取 Channel Layer，WebSocket推送失败。")
        return
    for stock_code in updated_stock_codes:
        async def main():
            # 1. 在异步上下文中创建顶层的 CacheManager
            cache_manager_instance = CacheManager()
            # 2. 创建所有需要的 DAO 实例，并注入 cache_manager
            stock_basic_dao = StockBasicInfoDao(cache_manager_instance)
            stock_realtime_dao = StockRealtimeDAO(cache_manager_instance)
            strategy_dao = StrategiesDAO(cache_manager_instance)
            # 3. 使用 asyncio.gather 并发执行所有数据获取任务
            #    这比串行调用 async_to_sync 快得多！
            results = await asyncio.gather(
                stock_basic_dao.get_stock_by_code(stock_code),
                stock_realtime_dao.get_latest_tick_data(stock_code),
                strategy_dao.get_latest_strategy_result(stock_code),
                return_exceptions=True # 捕获异常，防止一个失败导致全部失败
            )
            # 4. 解包结果
            stock_obj, latest_tick, latest_strategy_result = results
            # 检查是否有异常发生
            if isinstance(stock_obj, Exception):
                logger.error(f"获取股票基本信息失败 for {stock_code}: {stock_obj}")
                return # 无法继续，直接返回
            if isinstance(latest_tick, Exception):
                logger.error(f"获取最新Tick失败 for {stock_code}: {latest_tick}")
                # 即使tick失败，可能也想推送其他信息，这里先不返回
            if isinstance(latest_strategy_result, Exception):
                logger.error(f"获取最新策略结果失败 for {stock_code}: {latest_strategy_result}")
            # 5. 获取关注该股票的用户 (这是同步DB操作，放在main之外)
            user_ids = list(FavoriteStock.objects.filter(stock=stock_obj).values_list('user_id', flat=True))
            if not user_ids:
                return
            if not latest_tick or isinstance(latest_tick, Exception):
                logger.warning(f"未获取到股票 {stock_code} 的最新tick数据，跳过推送")
                return
            # 6. 构造 payload
            signal_score = getattr(latest_strategy_result, 'score', None) if latest_strategy_result and not isinstance(latest_strategy_result, Exception) else None
            signal = signal_score if isinstance(signal_score, dict) else {'type': 'hold', 'text': signal_score or 'N/A'}
            payload = {
                'code': stock_code,
                'current_price': latest_tick.get('current_price'),
                'high_price': latest_tick.get('high_price'),
                'low_price': latest_tick.get('low_price'),
                'open_price': latest_tick.get('open_price'),
                'prev_close_price': latest_tick.get('prev_close_price'),
                'trade_time': latest_tick.get('trade_time'),
                'turnover_value': latest_tick.get('turnover_value'),
                'volume': latest_tick.get('volume'),
                'change_percent': latest_tick.get("change_percent"),
                'signal': signal,
            }
            # 7. 推送给所有关注该股票的用户
            for uid in user_ids:
                send_update_to_user_task_celery.apply_async(
                    args=[uid, 'realtime_tick_update', payload],
                    queue='dashboard'  # 指定队列为dashboard
                )
                # print(f"已推送{code}最新tick数据到用户{uid}")
        # 使用 async_to_sync 运行这个总的 main 函数
        try:
            async_to_sync(main)()
        except Exception as e:
            logger.error(f"执行推送任务 for {stock_code} 时发生顶层错误: {e}", exc_info=True)
            

@celery_app.task(bind=True, name='dashboard.tasks.update_favorite_stock_trackers')
def update_favorite_stock_trackers(self):
    """
    【V3.0 快照生成版】
    每日运行，为所有活跃的 PositionTracker (状态为'持仓中'或'观察中') 创建当日的状态快照。
    这个任务的核心职责是从“更新”转变为“创建历史记录”。
    """
    try:
        # 步骤 1: 确定需要生成快照的日期（最新的一个交易日）
        latest_trade_day = TradeCalendar.get_latest_trade_date()
        if not latest_trade_day:
            logger.warning("无法获取最新的交易日，任务终止。")
            return "无法获取最新的交易日，任务终止。"
        logger.info(f"开始为 {latest_trade_day} 生成每日持仓快照...")
        # 步骤 2: 筛选出所有需要创建快照的活跃追踪器
        trackers_to_snapshot = list(PositionTracker.objects.filter(
            Q(status=PositionTracker.Status.HOLDING) | Q(status=PositionTracker.Status.WATCHING)
        ).select_related('stock'))
        if not trackers_to_snapshot:
            logger.info("没有找到需要创建快照的活跃追踪器。")
            return "没有需要更新的活跃追踪器。"
        stock_ids = [tracker.stock_id for tracker in trackers_to_snapshot]
        # 步骤 3: 一次性获取所有相关股票在快照日期的最新信号
        # 注意：这里我们假设每个股票在每个交易日最多只有一个主策略信号
        latest_signals = TradingSignal.objects.filter(
            stock_id__in=stock_ids,
            trade_time__date=latest_trade_day
        )
        # 将信号放入字典中，以便快速查找
        latest_signals_map = {signal.stock_id: signal for signal in latest_signals}
        logger.info(f"为 {len(stock_ids)} 个追踪器获取了 {len(latest_signals_map)} 条对应的当日信号。")
        # 步骤 4: 准备批量创建 DailyPositionSnapshot
        snapshots_to_create = []
        for tracker in trackers_to_snapshot:
            latest_signal_for_stock = latest_signals_map.get(tracker.stock_id)
            if latest_signal_for_stock:
                # 如果找到了当天的信号，就创建一个快照实例
                snapshots_to_create.append(
                    DailyPositionSnapshot(
                        position=tracker,
                        signal=latest_signal_for_stock,
                        snapshot_date=latest_trade_day,
                        close_price=latest_signal_for_stock.close_price,
                        # 其他快照字段可以根据需要从 signal 中获取
                    )
                )
            else:
                logger.warning(f"股票 {tracker.stock.stock_code} ({tracker.stock_id}) 在 {latest_trade_day} 没有找到对应的交易信号，无法创建快照。")
        # 步骤 5: 批量执行创建操作
        if snapshots_to_create:
            created_snapshots = DailyPositionSnapshot.objects.bulk_create(
                snapshots_to_create, 
                ignore_conflicts=True # 如果任务意外重跑，忽略已存在的记录，避免报错
            )
            logger.info(f"成功为 {len(created_snapshots)} / {len(trackers_to_snapshot)} 个活跃追踪器创建了每日快照。")
            return f"成功创建 {len(created_snapshots)} 条每日持仓快照。"
        else:
            logger.warning("没有可供创建的快照数据。")
            return "没有可供创建的快照数据。"
    except Exception as e:
        logger.error(f"更新持仓追踪器（创建快照）时发生严重错误: {e}", exc_info=True)
        # 如果任务失败，可以发起重试
        self.retry(exc=e, countdown=60)
        return f"任务失败: {e}"

# 你还需要配置 Celery Beat 来定时运行这些任务
# 例如，在 settings.py 中配置:
# CELERY_BEAT_SCHEDULE = {
#     'generate-strategy-message-every-10-seconds': {
#         'task': 'dashboard.tasks.generate_random_strategy_message',
#         'schedule': 10.0, # 每 10 秒
#     },
#      'update-favorites-every-5-seconds': {
#         'task': 'dashboard.tasks.update_favorite_stock_data',
#         'schedule': 5.0, # 每 5 秒
#     },
# }
