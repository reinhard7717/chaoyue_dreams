# 文件: intraday_engine/orchestrator.py
import asyncio
import json
import logging
from asgiref.sync import sync_to_async # 异步转换工具
from datetime import datetime, date, time
from typing import Dict, List, Set
from channels.layers import get_channel_layer
from dao_manager.tushare_daos.stock_basic_info_dao import StockBasicInfoDao
from dao_manager.tushare_daos.stock_time_trade_dao import StockTimeTradeDAO
from stock_models.index import TradeCalendar
from dao_manager.tushare_daos.strategies_dao import StrategiesDAO
from services.realtime_services import RealtimeServices
from stock_models.stock_analytics import TrendFollowStrategySignalLog
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
        # MODIFIED: 直接调用 CacheManager() 获取单例实例
        self.cache_manager = CacheManager()
        self.stock_dao = StockBasicInfoDao(self.cache_manager)
        self.stock_time_trade_dao = StockTimeTradeDAO(self.cache_manager)
        self.strategies_dao = StrategiesDAO(self.cache_manager)
        self.services = RealtimeServices(self.cache_manager)
        self.strategy = RealtimeStrategy(params)
        self.cache_key = IntradayEngineCashKey()
        self.today_str = date.today().strftime('%Y-%m-%d')

    async def initialize_pools(self):
        """
        【盘前准备 V2.2 - 数据库时区修复版】
        - 核心修复: 放弃使用DAO，直接在方法内构建一个明确的、时区感知的
                    范围查询，以解决因数据库存储与Django时区设置不匹配
                    导致的信号查询失败问题。
        """
        print("盘中引擎开始盘前准备，正在构建监控池并存入Redis...")
        
        # --- 1. 获取上一个交易日 ---
        today = date.today()
        get_prev_trade_date_async = sync_to_async(TradeCalendar.get_latest_trade_date, thread_sensitive=True)
        previous_trade_date = await get_prev_trade_date_async(reference_date=today)
        if not previous_trade_date:
            logger.error(f"无法从交易日历中找到 {today} 的上一个交易日，盘前准备任务终止。")
            return
        print(f"根据交易日历，确定需要查询的信号日期为: {previous_trade_date}")

        # --- 2. 构建“待买入池” (Watchlist) ---
        # 构建一个从当天 00:00:00 到 23:59:59 的 naive datetime 范围
        # 这将精确匹配数据库中存储的无时区信息的时间
        start_of_day = datetime.combine(previous_trade_date, time.min)
        end_of_day = datetime.combine(previous_trade_date, time.max)

        # 使用 __range 查询，并传入 naive datetime 对象
        # Django 在处理 naive datetime 时，会根据 settings.TIME_ZONE (通常是'UTC')
        # 来解释它们，但对于 __range 查询，它通常会直接使用你提供的值生成SQL，
        # 这恰好能匹配数据库中的存储方式。
        buy_signals_qs = TrendFollowStrategySignalLog.objects.filter(
            entry_signal=True,
            timeframe='D',
            trade_time__range=(start_of_day, end_of_day)
        ).select_related('stock')
        
        # 使用 sync_to_async 将同步的数据库查询转换为异步操作
        @sync_to_async
        def get_signals_list(qs):
            return list(qs)

        daily_buy_signals = await get_signals_list(buy_signals_qs)
        
        print(f"查询完成，共找到 {len(daily_buy_signals)} 条日线买入信号。")
        
        watchlist = {signal.stock.stock_code for signal in daily_buy_signals}
        
        # --- 3. 构建“持仓监控池” (Position List) ---
        # 3.1 获取基础的自选股信息（字典列表）
        favorite_stocks_list = await self.stock_dao.get_all_favorite_stocks()
        position_list = {}
        if favorite_stocks_list:
            # 3.2 提取所有不重复的股票代码
            stock_codes_to_fetch = sorted(list({fav.get("stock_code") for fav in favorite_stocks_list if fav.get("stock_code")}))
            print(f"调试: 需要为持仓池获取 {len(stock_codes_to_fetch)} 只股票的最新日线行情。")
            quotes_map = {}
            if stock_codes_to_fetch:
                # 3.3 创建一个包含所有异步查询任务的列表
                print(f"调试: 准备并发执行 {len(stock_codes_to_fetch)} 个数据库查询任务...")
                tasks = [self.stock_time_trade_dao.get_latest_daily_quote(code) for code in stock_codes_to_fetch]
                # 3.4 使用 asyncio.gather 并发执行所有任务
                # results 将是一个列表，其顺序与 tasks 列表的顺序完全对应
                results = await asyncio.gather(*tasks)
                print(f"调试: 并发查询完成，获取到 {len([r for r in results if r])} 条有效的行情数据。")
                # 3.5 将返回的报价列表转换成 "代码 -> 报价字典" 的映射，便于查找
                for code, quote_result in zip(stock_codes_to_fetch, results):
                    if quote_result:
                        quotes_map[code] = quote_result
            # 3.6 遍历自选股列表，结合报价数据，构建最终的 position_list
            for fav_stock_dict in favorite_stocks_list:
                stock_code = fav_stock_dict.get("stock_code")
                if not stock_code:
                    continue
                if stock_code not in position_list:
                    # 从刚刚构建的 quotes_map 中查找报价
                    stock_quote = quotes_map.get(stock_code, {}) # 如果没找到报价，返回空字典
                    position_list[stock_code] = {
                        "stock_code": stock_code,
                        # 从报价字典中获取收盘价，如果不存在则使用默认值10.0
                        "cost_price": float(stock_quote.get('close', 10.0)),
                        "user_id": fav_stock_dict.get("user_id")
                    }
        # --- 4. 将监控池写入Redis (逻辑不变) ---
        self.today_str = today.strftime('%Y-%m-%d')
        watchlist_key = self.cache_key.watchlist_key(self.today_str)
        position_list_key = self.cache_key.position_list_key(self.today_str)
        
        await self.cache_manager.initialize()
        redis_client = await self.cache_manager._ensure_client()
        async with redis_client.pipeline() as pipe:
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

        return True

    async def run_single_cycle(self, time_level: str = '1'):
        """
        【盘中循环】从Redis读取状态，执行分析，并将结果写回Redis。
        """
        await self.cache_manager.initialize()
        watchlist_key = self.cache_key.watchlist_key(self.today_str)
        position_list_key = self.cache_key.position_list_key(self.today_str)

        # 1. 从Redis读取监控池
        redis_client = await self.cache_manager._ensure_client()
        watchlist_bytes = await redis_client.smembers(watchlist_key)
        position_list_raw = await redis_client.hgetall(position_list_key)
        
        # 在这里进行解码，统一数据类型
        watchlist = {code.decode('utf-8') for code in watchlist_bytes}
        
        # 反序列化持仓信息
        position_list = {
            code.decode('utf-8'): json.loads(info.decode('utf-8')) 
            for code, info in position_list_raw.items()
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
                await redis_client.srem(watchlist_key, stock_code)

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

        redis_client = await self.cache_manager._ensure_client()
        async with redis_client.pipeline() as pipe:
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
