# 文件: intraday_engine/orchestrator.py
import asyncio
import json
import logging
from asgiref.sync import sync_to_async # 异步转换工具
from datetime import datetime, date, time
from typing import Dict, List, Set, Union # 导入 Union
from channels.layers import get_channel_layer
from dao_manager.tushare_daos.stock_basic_info_dao import StockBasicInfoDao
from dao_manager.tushare_daos.stock_time_trade_dao import StockTimeTradeDAO
from stock_models.index import TradeCalendar
from dao_manager.tushare_daos.strategies_dao import StrategiesDAO
from services.realtime_services import RealtimeServices
from strategies.realtime_strategy import RealtimeStrategy
from utils.cache_manager import CacheManager
from utils.cash_key import IntradayEngineCashKey
import pandas as pd # 导入 pandas

logger = logging.getLogger("intraday_engine")

class IntradayEngineOrchestrator:
    # ... __init__ 和 initialize_pools 方法保持不变 ...
    def __init__(self, params: Dict):
        self.params = params
        # 直接调用 CacheManager() 获取单例实例
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
        # --- 4. 将监控池写入Redis  ---
        self.today_str = today.strftime('%Y-%m-%d')
        watchlist_key = self.cache_key.watchlist_key(self.today_str)
        position_list_key = self.cache_key.position_list_key(self.today_str)
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
    # 重写此方法以增加健壮性
    async def run_single_cycle(self, time_level: str = '1'):
        """
        【盘中循环 V2.1 - 健壮版】从Redis读取状态，执行分析，并将结果写回Redis。
        - 使用 asyncio.gather(..., return_exceptions=True) 来防止单个股票分析失败导致整个循环崩溃。
        """
        watchlist_key = self.cache_key.watchlist_key(self.today_str)
        position_list_key = self.cache_key.position_list_key(self.today_str)
        # 1. 从Redis读取监控池
        redis_client = await self.cache_manager._ensure_client()
        watchlist_bytes = await redis_client.smembers(watchlist_key)
        position_list_raw = await redis_client.hgetall(position_list_key)
        watchlist = {code.decode('utf-8') for code in watchlist_bytes}
        position_list = {
            code.decode('utf-8'): json.loads(info.decode('utf-8')) 
            for code, info in position_list_raw.items()
        }
        all_stocks_to_analyze = sorted(list(set(watchlist) | set(position_list.keys())))
        if not all_stocks_to_analyze:
            logger.info("监控池为空，本轮循环跳过。")
            return []
        print(f"本轮循环准备分析 {len(all_stocks_to_analyze)} 只股票: {all_stocks_to_analyze[:5]}...")
        # 2. 并发获取所有需要分析的股票的盘中数据
        tasks = [self.services.prepare_intraday_data(code, time_level, self.today_str) for code in all_stocks_to_analyze]
        # 添加 return_exceptions=True，这是解决问题的关键！
        results: List[Union[pd.DataFrame, Exception]] = await asyncio.gather(*tasks, return_exceptions=True)
        intraday_data_map = {}
        # 循环检查结果，分离成功和失败的任务
        for stock_code, result in zip(all_stocks_to_analyze, results):
            if isinstance(result, Exception):
                # 如果是异常，打印详细错误日志，而不是让程序崩溃
                logger.error(f"为股票 {stock_code} 准备盘中数据时发生异常: {result}", exc_info=False) # exc_info=False避免打印冗长的堆栈
                print(f"错误: 股票 {stock_code} 数据准备失败，已跳过。异常: {result}")
            elif result is not None and not result.empty:
                # 只有成功返回了非空DataFrame才加入到待分析映射中
                intraday_data_map[stock_code] = result
            else:
                # 数据为空或None，也记录一下
                logger.warning(f"股票 {stock_code} 未能获取到有效的盘中数据，已跳过。")
        # 3. 循环决策并准备信号 (此部分逻辑不变)
        all_signals = []
        # 分析待买入池
        for stock_code in watchlist:
            df = intraday_data_map.get(stock_code)
            if df is None: continue
            print(f"-> 正在为待买入池中的 {stock_code} 执行策略分析...") #调试信息
            buy_signal = self.strategy.run_strategy(df, {"stock_code": stock_code})
            if buy_signal:
                all_signals.append(buy_signal)
                await redis_client.srem(watchlist_key, stock_code)
        # 分析持仓池
        for stock_code, pos_info in position_list.items():
            df = intraday_data_map.get(stock_code)
            if df is None: continue
            print(f"-> 正在为持仓池中的 {stock_code} 执行策略分析...") #调试信息
            t_signal = self.strategy.run_t_and_risk_control(df, pos_info)
            if t_signal:
                t_signal['user_id'] = pos_info.get('user_id')
                all_signals.append(t_signal)
        # 4. 将信号写入Redis，供Dashboard使用 (此部分逻辑不变)
        if all_signals:
            await self._save_signals_to_cache(all_signals)
        print(f"本轮循环分析完成。产出 {len(all_signals)} 条信号。")
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
        # --- 触发WebSocket推送 ---
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
