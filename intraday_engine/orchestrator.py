# 文件: intraday_engine/orchestrator.py
import asyncio
import json
import logging
from datetime import datetime, date
from typing import Dict, List, Set
from channels.layers import get_channel_layer
from stock_models.index import TradeCalendar
from dao_manager.tushare_daos.user_dao import UserDAO
from dao_manager.tushare_daos.strategies_dao import StrategiesDAO
from services.realtime_services import RealtimeServices
from strategies.realtime_strategy import RealtimeStrategy
from utils.cache_manager import CacheManager
from utils.cash_key import IntradayEngineCashKey

logger = logging.getLogger("intraday_engine")

class IntradayEngineOrchestrator:
    """
    【盘中引擎 - 总指挥 V2.0 - Redis状态持久化版】
    - 核心升级: 将监控池等状态信息持久化到Redis，解决了Celery任务的无状态问题。
    """
    def __init__(self, params: Dict):
        self.params = params
        self.user_dao = UserDAO()
        self.strategies_dao = StrategiesDAO()
        self.services = RealtimeServices()
        self.strategy = RealtimeStrategy(params)
        self.cache_manager = CacheManager()
        self.cache_key = IntradayEngineCashKey()
        self.today_str = date.today().strftime('%Y-%m-%d')

    async def initialize_pools(self):
        """
        【盘前准备 V2.0 - 交易日历适配版】
        在交易日开始前，构建两个核心的监控池。
        """
        logger.info("盘中引擎开始盘前准备，正在构建监控池并存入Redis...")
        
        # --- 【核心修改】使用交易日历获取上一个交易日 ---
        # 1. 获取今天的日期
        today = date.today()
        
        # 2. 调用 TradeCalendar 的类方法获取上一个交易日
        #    注意：由于 get_latest_trade_date 是同步方法，我们需要用 sync_to_async 包装
        get_prev_trade_date_async = sync_to_async(TradeCalendar.get_latest_trade_date, thread_sensitive=True)
        previous_trade_date = await get_prev_trade_date_async(reference_date=today)

        if not previous_trade_date:
            logger.error(f"无法从交易日历中找到 {today} 的上一个交易日，盘前准备任务终止。")
            return

        logger.info(f"根据交易日历，确定需要查询的信号日期为: {previous_trade_date}")

        # 3. 构建“待买入池” (Watchlist)
        #    使用获取到的上一个交易日进行查询
        daily_buy_signals = await self.strategies_dao.get_daily_buy_signals(trade_date=previous_trade_date)
        watchlist = {signal.stock.stock_code for signal in daily_buy_signals}
        
        # 4. 构建“持仓监控池” (Position List)
        favorite_stocks = await self.user_dao.get_all_favorite_stocks()
        position_list = {}
        for fav_stock in favorite_stocks:
            stock_code = fav_stock.stock.stock_code
            if stock_code not in position_list:
                position_list[stock_code] = {
                    "stock_code": stock_code,
                    "cost_price": float(fav_stock.stock.latest_quote.get('close', 10.0)),
                    "user_id": fav_stock.user.id
                }

        # 5. 将监控池写入Redis (逻辑不变)
        self.today_str = today.strftime('%Y-%m-%d') # 确保 self.today_str 是当天的日期
        watchlist_key = self.cache_key.watchlist_key(self.today_str)
        position_list_key = self.cache_key.position_list_key(self.today_str)
        
        await self.cache_manager.initialize()
        async with self.cache_manager.redis_client.pipeline() as pipe:
            pipe.delete(watchlist_key)
            pipe.delete(position_list_key)
            if watchlist:
                pipe.sadd(watchlist_key, *watchlist)
            if position_list:
                pipe.hset(position_list_key, mapping={
                    code: json.dumps(info) for code, info in position_list.items()
                })
            await pipe.execute()
            
        logger.info(f"待买入池 ({len(watchlist)}只) 和持仓池 ({len(position_list)}只) 已成功写入Redis。")

    async def run_single_cycle(self, time_level: str = '1'):
        """
        【盘中循环】从Redis读取状态，执行分析，并将结果写回Redis。
        """
        await self.cache_manager.initialize()
        watchlist_key = self.cache_key.watchlist_key(self.today_str)
        position_list_key = self.cache_key.position_list_key(self.today_str)

        # 1. 从Redis读取监控池
        watchlist = await self.cache_manager.redis_client.smembers(watchlist_key)
        position_list_raw = await self.cache_manager.redis_client.hgetall(position_list_key)
        
        # 反序列化持仓信息
        position_list = {
            code.decode(): json.loads(info.decode()) for code, info in position_list_raw.items()
        }

        all_stocks_to_analyze = set(watchlist) | set(position_list.keys())
        if not all_stocks_to_analyze:
            logger.info("监控池为空，本轮循环跳过。")
            return []

        # 2. 并发获取所有需要分析的股票的盘中数据
        tasks = [self.services.prepare_intraday_data(code, time_level, self.today_str) for code in all_stocks_to_analyze]
        results = await asyncio.gather(*tasks)
        
        intraday_data_map = {code: df for code, df in zip(all_stocks_to_analyze, results) if df is not None}

        # 3. 循环决策并准备信号
        all_signals = []
        # 分析待买入池
        for stock_code in watchlist:
            df = intraday_data_map.get(stock_code)
            if df is None: continue
            buy_signal = self.strategy.run_strategy(df, {"stock_code": stock_code})
            if buy_signal:
                all_signals.append(buy_signal)
                # 标记为待移除
                await self.cache_manager.redis_client.srem(watchlist_key, stock_code)

        # 分析持仓池
        for stock_code, pos_info in position_list.items():
            df = intraday_data_map.get(stock_code)
            if df is None: continue
            t_signal = self.strategy.run_t_and_risk_control(df, pos_info)
            if t_signal:
                # 附加用户信息，用于前端展示
                t_signal['user_id'] = pos_info.get('user_id')
                all_signals.append(t_signal)

        # 4. 将信号写入Redis，供Dashboard使用
        if all_signals:
            await self._save_signals_to_cache(all_signals)
            
        return all_signals

    async def _save_signals_to_cache(self, signals: List[Dict]):
        """将产生的信号按用户ID分组，写入Redis List"""
        user_signals_map = {}
        for signal in signals:
            user_id = signal.get('user_id')
            if user_id:
                if user_id not in user_signals_map:
                    user_signals_map[user_id] = []
                user_signals_map[user_id].append(signal)

        if not user_signals_map: return

        async with self.cache_manager.redis_client.pipeline() as pipe:
            for user_id, signal_list in user_signals_map.items():
                key = self.cache_key.user_signals_key(user_id, self.today_str)
                # 使用 lpush 将最新信号推到列表头部
                pipe.lpush(key, *[json.dumps(s) for s in signal_list])
                # 可以设置一个过期时间，例如24小时
                pipe.expire(key, 86400)
            await pipe.execute()
        logger.info(f"已将 {len(signals)} 条信号写入Redis，供Dashboard展示。")
        
        # --- 【新增】触发WebSocket推送 ---
        channel_layer = get_channel_layer()
        if channel_layer:
            for signal in signals:
                user_id = signal.get('user_id')
                if user_id:
                    # 异步地向该用户的group发送消息
                    await channel_layer.group_send(
                        f"user_{user_id}",
                        {
                            "type": "intraday_signal_update", # <--- 必须与 Consumer 中的方法名匹配
                            "payload": signal
                        }
                    )
            logger.info(f"已通过Channel Layer触发 {len(signals)} 条信号的WebSocket推送。")
        else:
            logger.warning("Channel Layer未初始化，无法触发WebSocket推送。")
