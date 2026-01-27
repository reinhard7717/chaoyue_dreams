# services/chip_holding_service.py
import asyncio
from asgiref.sync import sync_to_async # 异步转换工具
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import logging
from scipy.optimize import minimize
from django.db.models import Q
from stock_models.time_trade import StockDailyBasic
from stock_models.index import TradeCalendar
from scipy.interpolate import interp1d
from scipy.stats import gaussian_kde
from sklearn.neighbors import KernelDensity
from dao_manager.tushare_daos.stock_time_trade_dao import StockTimeTradeDAO
from utils.cache_manager import CacheManager
from utils.model_helpers import get_chip_holding_matrix_model_by_code
from utils.model_helpers import (
    get_minute_data_model_by_code_and_timelevel,
    get_stock_tick_data_model_by_code,
    get_cyq_chips_model_by_code,
    get_daily_data_model_by_code
)

logger = logging.getLogger(__name__)

class ChipHoldingService:
    """
    基于1分钟数据和逐笔数据的精确筹码持有时间计算服务
    """
    def __init__(self, use_tick_data: bool = True, cache_manager_instance: CacheManager = None):
        """
        初始化计算服务
        Args:
            use_tick_data: 是否使用逐笔数据增强计算
        """
        self.use_tick_data = use_tick_data
        self.price_grid_size = 200  # 价格网格数量
        self.max_holding_days = 250  # 最大追踪持有天数
        self.stock_trade_dao = StockTimeTradeDAO(cache_manager_instance)
        # 从model_helpers导入必要的函数
        self.get_minute_data_model = get_minute_data_model_by_code_and_timelevel
        self.get_tick_data_model = get_stock_tick_data_model_by_code
        self.get_chips_model = get_cyq_chips_model_by_code
        self.get_daily_data_model = get_daily_data_model_by_code

    async def calculate_holding_matrix_daily_async(self, stock_code: str, trade_date: str, lookback_days: int = 60) -> Dict[str, any]:
        """计算单日筹码持有时间矩阵（异步版本）- 修复方法调用问题"""
        try:
            logger.info(f"开始计算 {stock_code} {trade_date} 的筹码持有时间矩阵")
            print(f"🔴 [主流程开始] 股票: {stock_code}, 日期: {trade_date}")
            # 1. 获取基础数据（异步）
            data_dict = await self._fetch_required_data(stock_code, trade_date, lookback_days)
            if not data_dict:
                return self._get_default_result()
            print(f"📊 [主流程] 获取到筹码历史数据: {len(data_dict.get('chip_dists', []))}条")
            # 2. 建立价格网格和筹码矩阵
            price_grid, chip_dist_matrix = await sync_to_async(self._build_price_grid_and_chip_matrix)(data_dict['chip_dists'], data_dict['price_range'])
            print(f"📊 [主流程] 价格网格形状: {price_grid.shape}, 筹码矩阵形状: {chip_dist_matrix.shape}")
            if chip_dist_matrix.size == 0:
                return self._get_default_result(stock_code, trade_date)
            # 3. 计算分钟级成交量分布
            minute_volume_dist = await sync_to_async(self._calculate_minute_volume_distribution)(data_dict['minute_data'], price_grid)
            print(f"📊 [主流程] 分钟成交量分布计算完成")
            # 4. 如果使用逐笔数据，进行增强计算
            if self.use_tick_data and data_dict['tick_data'] is not None:
                enhanced_dist = await sync_to_async(self._enhance_with_tick_data)(minute_volume_dist, data_dict['tick_data'], price_grid)
            else:
                enhanced_dist = minute_volume_dist
            # 5. 计算换手率矩阵
            turnover_matrix = await sync_to_async(self._calculate_turnover_matrix)(enhanced_dist, chip_dist_matrix, data_dict['float_shares'])
            print(f"📊 [主流程] 换手率矩阵形状: {turnover_matrix.shape}")
            # 6. 优化换手概率参数
            optimal_params = await sync_to_async(self._optimize_turnover_parameters)(turnover_matrix, chip_dist_matrix, data_dict['daily_turnover'])
            # 7. 计算持有时间矩阵 - 使用正确的_holding_matrix方法（去掉v2后缀）
            holding_matrix = await sync_to_async(self._calculate_holding_matrix)(chip_dist_matrix, turnover_matrix, optimal_params)
            print(f"📊 [主流程] 持有时间矩阵计算完成: 形状={holding_matrix.shape}")
            # 8. 计算衍生因子 - 使用正确的_calculate_holding_factors方法（去掉v2后缀）
            factors = await sync_to_async(self._calculate_holding_factors)(holding_matrix, chip_dist_matrix, data_dict)
            print(f"📊 [主流程] 衍生因子计算完成: 短线={factors.get('short_term_ratio', 0):.2%}, 长线={factors.get('long_term_ratio', 0):.2%}")
            # 9. 验证结果
            validation = await sync_to_async(self._validate_results)(holding_matrix, factors, data_dict)
            print(f"📊 [主流程] 验证完成: 有效性={validation.get('is_valid', False)}")
            # 构建结果
            result = {
                'stock_code': stock_code,
                'trade_date': trade_date,
                'holding_matrix': holding_matrix,
                'price_grid': price_grid,
                'factors': factors,
                'validation': validation,
                'calc_status': 'success',
                'calc_time': datetime.now()
            }
            print(f"✅ [主流程完成] {stock_code} {trade_date} 计算成功")
            return result
        except Exception as e:
            logger.error(f"计算筹码持有矩阵失败 {stock_code} {trade_date}: {e}", exc_info=True)
            print(f"❌ [主流程异常] {stock_code} {trade_date}: {e}")
            return self._get_default_result(stock_code, trade_date)

    def calculate_holding_matrix_daily(self,stock_code: str,trade_date: str,lookback_days: int = 60) -> Dict[str, any]:
        """
        计算单日筹码持有时间矩阵（主入口函数，同步包装）
        """
        try:
            # 创建事件循环
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(
                    self.calculate_holding_matrix_daily_async(stock_code, trade_date, lookback_days)
                )
            finally:
                loop.close()
            return result
        except Exception as e:
            logger.error(f"计算筹码持有矩阵失败 {stock_code} {trade_date}: {e}")
            return self._get_default_result(stock_code, trade_date)

    def calculate_batch_holding_matrices(self,stock_codes: List[str],trade_date: str,max_workers: int = 4) -> Dict[str, Dict]:
        """
        批量计算多只股票的持有时间矩阵
        Args:
            stock_codes: 股票代码列表
            trade_date: 交易日期
            max_workers: 最大并行工作数
        Returns:
            Dict: 股票代码到结果的映射
        """
        import concurrent.futures
        from tqdm import tqdm
        results = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_code = {
                executor.submit(self.calculate_holding_matrix_daily, code, trade_date): code
                for code in stock_codes
            }
            # 处理结果
            for future in tqdm(concurrent.futures.as_completed(future_to_code), 
                              total=len(stock_codes), desc="计算持有矩阵"):
                code = future_to_code[future]
                try:
                    result = future.result()
                    results[code] = result
                except Exception as e:
                    logger.error(f"批量计算失败 {code}: {e}")
                    results[code] = self._get_default_result(code, trade_date)
        return results
    
    async def _fetch_required_data(self, stock_code: str, trade_date: str, lookback_days: int) -> Dict[str, any]:
        """获取计算所需的所有数据（异步版本，使用交易日历） - 修复分钟数据获取问题V3"""
        try:
            # 转换日期
            trade_date_dt = datetime.strptime(trade_date, "%Y-%m-%d").date()
            print(f"🟢 [数据获取开始_v3] 股票: {stock_code}, 日期: {trade_date}, 回溯交易日: {lookback_days}")
            # 使用交易日历获取回溯的起始交易日（异步调用）
            get_offset_func = sync_to_async(TradeCalendar.get_trade_date_offset, thread_sensitive=True)
            try:
                start_date = await get_offset_func(trade_date_dt, -lookback_days)
                if not start_date:
                    print(f"⚠️ [数据获取_v3] 无法获取 {lookback_days} 个交易日前的日期，使用自然日计算")
                    start_date = trade_date_dt - timedelta(days=lookback_days * 2)
            except Exception as e:
                print(f"⚠️ [数据获取_v3] 获取交易日偏移失败: {e}, 使用自然日计算")
                start_date = trade_date_dt - timedelta(days=lookback_days * 2)
            data = {}
            print(f"📅 [交易日历_v3] 计算日期范围: {start_date} 到 {trade_date_dt}")
            # 1. 获取1分钟数据 - 使用现有的DAO方法
            print(f"🔍 [分钟数据_v3] 开始获取股票 {stock_code} 的分钟数据，日期: {trade_date_dt}")
            try:
                minute_df = await self.stock_trade_dao.get_1_min_kline_time_by_day(stock_code, trade_date_dt)
                if minute_df is not None and not minute_df.empty:
                    # 重置索引并将trade_time作为列
                    minute_df_reset = minute_df.reset_index()
                    # 重命名列以匹配原有代码的期望
                    if 'volume' in minute_df_reset.columns:
                        minute_df_reset = minute_df_reset.rename(columns={'volume': 'vol'})
                    data['minute_data'] = minute_df_reset
                    print(f"✅ [分钟数据_v3] 成功获取分钟数据: {len(minute_df_reset)}条记录")
                    print(f"📊 [分钟数据_v3] 列名: {list(minute_df_reset.columns)}")
                    print(f"📊 [分钟数据_v3] 时间范围: {minute_df_reset['trade_time'].min()} 到 {minute_df_reset['trade_time'].max()}")
                else:
                    print(f"⚠️ [分钟数据_v3] 通过DAO获取分钟数据失败或数据为空，尝试其他方法")
                    # 尝试备用方法
                    minute_model = self.get_minute_data_model(stock_code, '1')
                    if minute_model:
                        try:
                            from stock_models.stock_basic import StockInfo
                            stock = await sync_to_async(StockInfo.objects.get)(stock_code=stock_code)
                            # 使用范围查询
                            next_day = trade_date_dt + timedelta(days=1)
                            minute_qs = minute_model.objects.filter(
                                stock=stock,
                                trade_time__gte=datetime.combine(trade_date_dt, datetime.min.time()),
                                trade_time__lt=datetime.combine(next_day, datetime.min.time())
                            ).order_by('trade_time')
                            minute_records = await sync_to_async(list)(minute_qs.values('trade_time', 'open', 'high', 'low', 'close', 'vol', 'amount'))
                            if minute_records:
                                data['minute_data'] = pd.DataFrame(minute_records)
                                print(f"✅ [分钟数据_v3] 备用方法成功: {len(minute_records)}条记录")
                            else:
                                print(f"⚠️ [分钟数据_v3] 备用方法也返回0条记录")
                                data['minute_data'] = None
                        except Exception as e:
                            print(f"❌ [分钟数据_v3] 备用方法异常: {e}")
                            data['minute_data'] = None
                    else:
                        print(f"⚠️ [分钟数据_v3] 分钟数据模型不存在")
                        data['minute_data'] = None
            except Exception as e:
                print(f"❌ [分钟数据_v3] 获取分钟数据异常: {e}")
                import traceback
                traceback.print_exc()
                data['minute_data'] = None
            # 如果分钟数据为空，尝试从日线数据估算
            if data.get('minute_data') is None or data['minute_data'].empty:
                print(f"⚠️ [分钟数据_v3] 无1分钟数据，尝试从日线数据估算")
                # 从日线数据获取当日总成交量
                daily_model = self.get_daily_data_model(stock_code)
                if daily_model:
                    try:
                        from stock_models.stock_basic import StockInfo
                        stock = await sync_to_async(StockInfo.objects.get)(stock_code=stock_code)
                        daily_qs = daily_model.objects.filter(stock=stock, trade_time=trade_date_dt)
                        daily_record = await sync_to_async(daily_qs.first)()
                        if daily_record and daily_record.vol:
                            # 使用日成交量作为总成交量
                            day_volume = daily_record.vol * 100  # 手转股
                            print(f"📊 [分钟数据_v3] 从日线数据获取成交量: {day_volume}股")
                            # 创建模拟的分钟数据（均匀分布）
                            data['minute_data'] = pd.DataFrame({
                                'trade_time': [trade_date_dt],
                                'open': [daily_record.open_qfq or 0],
                                'high': [daily_record.high_qfq or 0],
                                'low': [daily_record.low_qfq or 0],
                                'close': [daily_record.close_qfq or 0],
                                'vol': [daily_record.vol or 0],  # 成交量（手）
                                'amount': [daily_record.amount or 0]
                            })
                        else:
                            print(f"⚠️ [分钟数据_v3] 无日线成交量数据")
                            data['minute_data'] = pd.DataFrame()
                    except Exception as e:
                        print(f"❌ [分钟数据_v3] 获取日线数据异常: {e}")
                        data['minute_data'] = pd.DataFrame()
                else:
                    print(f"⚠️ [分钟数据_v3] 日线数据模型不存在")
                    data['minute_data'] = pd.DataFrame()
            else:
                print(f"✅ [分钟数据_v3] 分钟数据获取完成，记录数: {len(data['minute_data'])}")
            # 2. 获取逐笔数据（如果可用）- 简化处理
            if self.use_tick_data:
                print(f"🔍 [逐笔数据_v3] 开始获取逐笔数据")
                tick_model = self.get_tick_data_model(stock_code)
                if tick_model:
                    try:
                        from stock_models.stock_basic import StockInfo
                        stock = await sync_to_async(StockInfo.objects.get)(stock_code=stock_code)
                        next_day = trade_date_dt + timedelta(days=1)
                        tick_qs = tick_model.objects.filter(
                            stock=stock,
                            trade_time__gte=datetime.combine(trade_date_dt, datetime.min.time()),
                            trade_time__lt=datetime.combine(next_day, datetime.min.time())
                        ).order_by('trade_time')[:50000]
                        tick_records = await sync_to_async(list)(tick_qs.values('trade_time', 'price', 'volume', 'type'))
                        data['tick_data'] = pd.DataFrame(tick_records) if tick_records else None
                        print(f"📊 [逐笔数据_v3] 记录数: {len(tick_records) if tick_records else 0}")
                    except Exception as e:
                        print(f"⚠️ [逐笔数据_v3] 获取失败: {e}")
                        data['tick_data'] = None
                else:
                    print(f"⚠️ [逐笔数据_v3] 模型不存在")
                    data['tick_data'] = None
            # 3. 获取筹码分布数据 - 简化处理
            print(f"🔍 [筹码模型_v3] 开始获取筹码数据")
            chips_model = self.get_chips_model(stock_code)
            if chips_model is None:
                print(f"❌ [筹码模型_v3] 模型获取失败!")
                data['chip_dists'] = []
                data['chip_dist_current'] = pd.DataFrame()
            else:
                try:
                    from stock_models.stock_basic import StockInfo
                    stock = await sync_to_async(StockInfo.objects.get)(stock_code=stock_code)
                    # 获取当前日期的筹码分布
                    chip_dist_current_qs = chips_model.objects.filter(stock=stock, trade_time=trade_date_dt)
                    chip_dist_current_list = await sync_to_async(list)(chip_dist_current_qs.values('price', 'percent'))
                    print(f"📊 [当前筹码查询_v3] 原始记录数: {len(chip_dist_current_list)}")
                    if chip_dist_current_list:
                        data['chip_dist_current'] = pd.DataFrame(chip_dist_current_list)
                        print(f"📊 [当前筹码_v3] DataFrame记录数: {len(data['chip_dist_current'])}")
                    else:
                        data['chip_dist_current'] = pd.DataFrame()
                        print(f"⚠️ [当前筹码_v3] 无当日筹码数据")
                    # 获取历史筹码分布数据
                    get_dates_between_func = sync_to_async(TradeCalendar.get_trade_dates_between, thread_sensitive=True)
                    trade_dates = await get_dates_between_func(start_date, trade_date_dt - timedelta(days=1))
                    print(f"📅 [历史筹码_v3] 获取 {len(trade_dates) if trade_dates else 0} 个交易日的数据")
                    # 分批获取历史筹码数据
                    historical_by_date = {}
                    if trade_dates:
                        for trade_date_obj in trade_dates:
                            daily_chips_qs = chips_model.objects.filter(stock=stock, trade_time=trade_date_obj)
                            daily_chips_list = await sync_to_async(list)(daily_chips_qs.values('price', 'percent'))
                            if daily_chips_list:
                                historical_by_date[str(trade_date_obj)] = daily_chips_list
                    # 转换为需要的格式：每日一个字典列表
                    data['chip_dists'] = list(historical_by_date.values())
                    print(f"📊 [历史筹码分组_v3] 共 {len(historical_by_date)} 个交易日有筹码数据")
                except Exception as e:
                    print(f"❌ [筹码模型_v3] 获取数据失败: {e}")
                    data['chip_dists'] = []
                    data['chip_dist_current'] = pd.DataFrame()
            # 4. 获取日线数据（用于换手率）
            print(f"🔍 [日线数据_v3] 开始获取日线数据")
            daily_model = self.get_daily_data_model(stock_code)
            if daily_model:
                try:
                    from stock_models.stock_basic import StockInfo
                    stock = await sync_to_async(StockInfo.objects.get)(stock_code=stock_code)
                    daily_qs = daily_model.objects.filter(
                        stock=stock, 
                        trade_time__gte=start_date, 
                        trade_time__lte=trade_date_dt
                    ).order_by('trade_time').values('trade_time', 'vol', 'amount')
                    daily_data = await sync_to_async(list)(daily_qs)
                    data['daily_data'] = pd.DataFrame(daily_data) if daily_data else pd.DataFrame()
                    print(f"📊 [日线数据_v3] 记录数: {len(daily_data)}")
                except Exception as e:
                    print(f"⚠️ [日线数据_v3] 获取失败: {e}")
                    data['daily_data'] = pd.DataFrame()
            else:
                print(f"⚠️ [日线数据_v3] 模型不存在")
                data['daily_data'] = pd.DataFrame()
            # 5. 获取自由流通股本
            print(f"🔍 [自由流通股本_v3] 开始获取流通股本")
            try:
                basic_qs = StockDailyBasic.objects.filter(stock__stock_code=stock_code, trade_time=trade_date_dt)
                basic_data = await sync_to_async(basic_qs.first)()
                if basic_data and basic_data.free_share:
                    data['float_shares'] = float(basic_data.free_share) * 10000
                    print(f"📊 [自由流通股本_v3] 从数据库获取: {basic_data.free_share}万 → {data['float_shares']}股")
                else:
                    # 尝试获取最近的有效数据
                    print(f"⚠️ [自由流通股本_v3] 当日数据不存在，尝试获取最近数据")
                    recent_basic_qs = StockDailyBasic.objects.filter(stock__stock_code=stock_code, trade_time__lt=trade_date_dt).order_by('-trade_time')
                    recent_basic_data = await sync_to_async(recent_basic_qs.first)()
                    if recent_basic_data and recent_basic_data.free_share:
                        data['float_shares'] = float(recent_basic_data.free_share) * 10000
                        print(f"📊 [自由流通股本_v3] 使用最近数据({recent_basic_data.trade_time}): {data['float_shares']}股")
                    else:
                        data['float_shares'] = 100000000
                        print(f"⚠️ [自由流通股本_v3] 使用默认值: {data['float_shares']}股")
            except Exception as e:
                print(f"⚠️ [自由流通股本_v3] 获取失败: {e}, 使用默认值")
                data['float_shares'] = 100000000
            # 6. 计算价格范围
            if 'chip_dist_current' in data and not data['chip_dist_current'].empty:
                price_min = data['chip_dist_current']['price'].min()
                price_max = data['chip_dist_current']['price'].max()
                data['price_range'] = (price_min, price_max)
                print(f"📈 [价格范围-当前_v3] {price_min:.2f} - {price_max:.2f}")
            else:
                # 从历史数据中提取所有价格
                all_prices = []
                if 'chip_dists' in data and data['chip_dists']:
                    for daily_dist in data['chip_dists']:
                        if daily_dist:
                            for item in daily_dist:
                                if isinstance(item, dict) and 'price' in item:
                                    all_prices.append(item['price'])
                if all_prices:
                    price_min = min(all_prices)
                    price_max = max(all_prices)
                    padding = (price_max - price_min) * 0.1
                    price_min = max(0.01, price_min - padding)
                    price_max = price_max + padding
                    data['price_range'] = (price_min, price_max)
                    print(f"📈 [价格范围-历史_v3] {price_min:.2f} - {price_max:.2f}, 基于{len(all_prices)}个价格点")
                else:
                    data['price_range'] = (1.0, 100.0)
                    print(f"⚠️ [价格范围_v3] 无价格数据，使用默认范围: 1.0 - 100.0")
            # 7. 计算日换手率
            if not data['daily_data'].empty and 'vol' in data['daily_data'].columns and data['float_shares'] > 0:
                data['daily_turnover'] = data['daily_data']['vol'] * 100 / data['float_shares']
                print(f"📊 [日换手率_v3] 计算完成，数据长度: {len(data['daily_turnover'])}")
            else:
                print(f"⚠️ [日换手率_v3] 计算失败，数据不全")
                data['daily_turnover'] = pd.Series()
            print(f"✅ [数据获取完成_v3] 共获取{len(data)}个数据集")
            print(f"📋 [数据摘要_v3] 分钟数据: {len(data.get('minute_data', pd.DataFrame()))}行")
            print(f"📋 [数据摘要_v3] 筹码历史天数: {len(data.get('chip_dists', []))}")
            print(f"📋 [数据摘要_v3] 当前筹码条数: {len(data.get('chip_dist_current', pd.DataFrame()))}")
            return data
        except Exception as e:
            logger.error(f"获取数据失败 {stock_code}: {e}", exc_info=True)
            print(f"❌ [数据获取异常_v3] {e}")
            import traceback
            traceback.print_exc()
            return {}

    def _build_price_grid_and_chip_matrix(self, chip_dists: List[Dict], price_range: Tuple[float, float]) -> Tuple[np.ndarray, np.ndarray]:
        """建立价格网格和筹码分布矩阵"""
        try:
            min_price, max_price = price_range
            print(f"🔨 [构建矩阵] 输入价格范围: {min_price:.2f} - {max_price:.2f}")
            # 检查价格范围有效性
            if max_price <= min_price or max_price <= 0 or min_price <= 0:
                print(f"⚠️ [构建矩阵] 价格范围无效，使用默认范围")
                min_price, max_price = 1.0, 100.0
            # 确保价格范围有足够的差值
            price_diff = max_price - min_price
            if price_diff < 0.01:
                print(f"⚠️ [构建矩阵] 价格差值过小: {price_diff:.4f}, 扩展范围")
                padding = max(1.0, min_price * 0.1)
                min_price = max(0.01, min_price - padding)
                max_price = max_price + padding
            price_grid = np.linspace(min_price, max_price, self.price_grid_size)
            print(f"📊 [构建矩阵] 价格网格: {len(price_grid)}个点, {min_price:.2f}-{max_price:.2f}")
            # 检查是否有历史筹码数据
            if not chip_dists:
                print(f"⚠️ [构建矩阵] 无历史筹码分布数据，创建默认矩阵")
                chip_matrix = np.ones((1, len(price_grid))) / len(price_grid)
                print(f"📊 [构建矩阵] 创建默认筹码矩阵: 形状={chip_matrix.shape}")
                return price_grid, chip_matrix
            print(f"📊 [构建矩阵] 处理 {len(chip_dists)} 天的历史筹码数据")
            # 将历史筹码分布插值到价格网格
            chip_matrix = np.zeros((len(chip_dists), len(price_grid)))
            valid_days = 0
            for i, chip_dist in enumerate(chip_dists):
                if not chip_dist:
                    print(f"⚠️ [构建矩阵] 第{i}天筹码数据为空")
                    chip_matrix[i, :] = np.ones(len(price_grid)) / len(price_grid)
                    continue
                # 修复关键问题：处理单个字典的情况
                if isinstance(chip_dist, dict):
                    print(f"⚠️ [构建矩阵] 第{i}天数据是单个字典，转换为列表")
                    chip_dist = [chip_dist]
                # 确保chip_dist是列表
                if not isinstance(chip_dist, list):
                    print(f"⚠️ [构建矩阵] 第{i}天数据格式错误: {type(chip_dist)}")
                    chip_matrix[i, :] = np.ones(len(price_grid)) / len(price_grid)
                    continue
                try:
                    df = pd.DataFrame(chip_dist)
                    if df.empty or 'price' not in df.columns or 'percent' not in df.columns:
                        print(f"⚠️ [构建矩阵] 第{i}天DataFrame格式错误，使用均匀分布")
                        chip_matrix[i, :] = np.ones(len(price_grid)) / len(price_grid)
                        continue
                    if len(df) < 2:
                        print(f"⚠️ [构建矩阵] 第{i}天数据点过少: {len(df)}个")
                        chip_matrix[i, :] = np.ones(len(price_grid)) / len(price_grid)
                        continue
                    # 线性插值
                    f = interp1d(df['price'], df['percent'], bounds_error=False, fill_value=0)
                    chip_matrix[i, :] = f(price_grid)
                    valid_days += 1
                    if i < 3:  # 打印前3天的调试信息
                        print(f"📊 [构建矩阵] 第{i}天插值成功: {len(df)}点→{len(price_grid)}网格")
                        print(f"📊 [构建矩阵] 第{i}天价格范围: {df['price'].min():.2f}-{df['price'].max():.2f}")
                        print(f"📊 [构建矩阵] 第{i}天比例总和: {df['percent'].sum():.4f}")
                except Exception as e:
                    print(f"⚠️ [构建矩阵] 第{i}天处理失败: {e}")
                    print(f"📊 [构建矩阵] 第{i}天数据类型: {type(chip_dist)}, 长度: {len(chip_dist) if hasattr(chip_dist, '__len__') else 'N/A'}")
                    chip_matrix[i, :] = np.ones(len(price_grid)) / len(price_grid)
            print(f"📊 [构建矩阵] 有效处理天数: {valid_days}/{len(chip_dists)}")
            # 归一化
            row_sums = chip_matrix.sum(axis=1, keepdims=True)
            chip_matrix = np.divide(chip_matrix, row_sums, out=np.zeros_like(chip_matrix), where=row_sums != 0)
            # 检查最终矩阵
            if chip_matrix.shape[0] == 0 or chip_matrix.shape[1] == 0:
                print(f"❌ [构建矩阵] 最终矩阵维度异常: {chip_matrix.shape}")
                chip_matrix = np.ones((max(len(chip_dists), 1), len(price_grid))) / len(price_grid)
            print(f"✅ [构建矩阵完成] 价格网格形状={price_grid.shape}, 筹码矩阵形状={chip_matrix.shape}")
            print(f"📊 [构建矩阵] 筹码矩阵统计: 总和={chip_matrix.sum():.4f}, 均值={chip_matrix.mean():.6f}")
            return price_grid, chip_matrix
        except Exception as e:
            logger.error(f"建立价格网格失败: {e}")
            print(f"❌ [构建矩阵异常] {e}")
            import traceback
            traceback.print_exc()
            default_price_grid = np.linspace(1.0, 100.0, self.price_grid_size)
            default_chip_matrix = np.ones((1, len(default_price_grid))) / len(default_price_grid)
            return default_price_grid, default_chip_matrix

    def _calculate_minute_volume_distribution(self, minute_data: pd.DataFrame, price_grid: np.ndarray) -> np.ndarray:
        """
        使用1分钟数据计算成交量分布
        """
        try:
            if minute_data is None or minute_data.empty:
                print(f"⚠️ [_calculate_minute_volume_distribution] 分钟数据为空")
                # 返回零分布，让上层处理
                return np.zeros(len(price_grid))
            volume_dist = np.zeros(len(price_grid))
            total_volume = 0
            for _, row in minute_data.iterrows():
                minute_volume = row['vol']
                total_volume += minute_volume
                low_price = row['low']
                high_price = row['high']
                # 找到价格区间对应的网格索引
                start_idx = np.searchsorted(price_grid, low_price, side='left')
                end_idx = np.searchsorted(price_grid, high_price, side='right')
                if start_idx < end_idx:
                    # 在价格区间内均匀分布成交量
                    interval_volume = minute_volume / (end_idx - start_idx)
                    volume_dist[start_idx:end_idx] += interval_volume
                else:
                    # 价格区间很窄，直接分配到最近的网格
                    mid_price = (low_price + high_price) / 2
                    nearest_idx = np.argmin(np.abs(price_grid - mid_price))
                    volume_dist[nearest_idx] += minute_volume
            print(f"📊 [_calculate_minute_volume_distribution] 总成交量: {total_volume}股, 分布总和: {volume_dist.sum():.0f}股")
            # 如果分钟数据量太小，使用日线数据补充
            if total_volume < 1000:  # 小于1000股，认为数据不准确
                print(f"⚠️ [_calculate_minute_volume_distribution] 分钟成交量过小 ({total_volume}股)，可能不准确")
            return volume_dist
            
        except Exception as e:
            logger.error(f"计算分钟成交量分布失败: {e}")
            print(f"❌ [_calculate_minute_volume_distribution] 计算失败: {e}")
            return np.zeros(len(price_grid))

    def _enhance_with_tick_data(self,base_dist: np.ndarray,tick_data: pd.DataFrame,price_grid: np.ndarray) -> np.ndarray:
        """
        使用逐笔数据增强成交量分布
        """
        try:
            if tick_data is None or tick_data.empty:
                return base_dist
            # 基于逐笔数据计算更精确的分布
            tick_prices = tick_data['price'].values
            tick_volumes = tick_data['volume'].values
            # 使用核密度估计
            if len(tick_prices) > 10:
                # 训练KDE模型
                kde = KernelDensity(bandwidth=0.01, kernel='gaussian')
                # 按成交量加权
                sample_weights = tick_volumes / tick_volumes.sum()
                # 需要重采样以应用权重
                resampled_prices = []
                for price, weight in zip(tick_prices, sample_weights):
                    # 按权重比例重复价格
                    n_repeats = max(1, int(weight * 1000))
                    resampled_prices.extend([price] * n_repeats)
                if resampled_prices:
                    kde.fit(np.array(resampled_prices).reshape(-1, 1))
                    # 计算每个网格点的密度
                    log_density = kde.score_samples(price_grid.reshape(-1, 1))
                    tick_density = np.exp(log_density)
                    # 归一化并乘以总成交量
                    total_volume = tick_volumes.sum()
                    tick_dist = tick_density / tick_density.sum() * total_volume
                    # 合并基础分布和逐笔分布（加权平均）
                    alpha = 0.7  # 逐笔数据权重
                    enhanced_dist = alpha * tick_dist + (1 - alpha) * base_dist
                    return enhanced_dist
            return base_dist
        except Exception as e:
            logger.error(f"逐笔数据增强失败: {e}")
            return base_dist
    
    def _calculate_turnover_matrix(self, volume_dist: np.ndarray, chip_matrix: np.ndarray, float_shares: float) -> np.ndarray:
        """计算换手率矩阵 - 修复换手率计算"""
        try:
            if float_shares <= 0 or len(volume_dist) == 0 or chip_matrix.size == 0:
                print(f"⚠️ [_calculate_turnover_matrix] 输入数据无效")
                return np.zeros_like(chip_matrix)
            
            # 计算日换手率
            daily_volume = volume_dist.sum()
            daily_turnover_rate = daily_volume / float_shares if float_shares > 0 else 0.02
            print(f"📊 [_calculate_turnover_matrix] 日成交量: {daily_volume:.0f}, 流通股: {float_shares:.0f}, 日换手率: {daily_turnover_rate:.4%}")
            
            # 使用当前筹码分布计算各价格区间的相对换手率
            if chip_matrix.shape[0] > 0:
                chip_dist_current = chip_matrix[-1, :]
            else:
                chip_dist_current = np.ones(len(volume_dist)) / len(volume_dist)
            
            # 避免除零
            chip_dist_current = np.maximum(chip_dist_current, 1e-6)
            
            # 🚨 关键修复：使用正确的换手率计算方法
            # 每个价格区间的成交量 / (该价格区间筹码 * 总流通股本)
            turnover_by_price = np.zeros_like(volume_dist)
            for i in range(len(volume_dist)):
                if chip_dist_current[i] > 0:
                    # 该价格区间筹码量 = 筹码比例 * 总流通股本
                    chips_at_price = chip_dist_current[i] * float_shares
                    if chips_at_price > 0:
                        # 该价格区间换手率 = 成交量 / 筹码量
                        turnover_by_price[i] = volume_dist[i] / chips_at_price
            
            # 归一化到日换手率
            total_turnover = turnover_by_price.sum()
            if total_turnover > 0:
                # 按比例缩放，使平均换手率等于日换手率
                scaling_factor = daily_turnover_rate / (total_turnover / len(turnover_by_price))
                price_turnover = turnover_by_price * scaling_factor
            else:
                price_turnover = np.ones(len(volume_dist)) * daily_turnover_rate
            
            # 限制合理范围：单日换手率不应超过50%
            price_turnover = np.clip(price_turnover, 0.001, 0.5)
            
            print(f"📊 [_calculate_turnover_matrix] 价格换手率统计: 均值={price_turnover.mean():.4%}, 范围={price_turnover.min():.4%}-{price_turnover.max():.4%}")
            
            # 创建换手率矩阵
            turnover_matrix = np.tile(price_turnover, (chip_matrix.shape[0], 1))
            
            # 应用时间衰减
            if chip_matrix.shape[0] > 1:
                time_weights = np.exp(-np.arange(chip_matrix.shape[0]) / 20)
                turnover_matrix = turnover_matrix * time_weights[:, np.newaxis]
            
            return turnover_matrix
            
        except Exception as e:
            print(f"❌ [_calculate_turnover_matrix] 计算失败: {e}")
            import traceback
            traceback.print_exc()
            return np.zeros_like(chip_matrix)

    def _optimize_turnover_parameters(self,turnover_matrix: np.ndarray,chip_matrix: np.ndarray,daily_turnover_series: pd.Series) -> Dict[str, float]:
        """
        优化换手概率参数
        """
        try:
            if turnover_matrix.size == 0 or chip_matrix.size == 0:
                return {'alpha': 0.5, 'beta': 0.3, 'gamma': 0.2}
            # 简化优化：使用最小二乘法
            def objective(params):
                alpha, beta = params
                # 预测换手率 = alpha * 成交量分布 + beta * 筹码集中度
                chip_concentration = np.std(chip_matrix, axis=1)
                predicted = alpha * turnover_matrix.mean(axis=1) + beta * chip_concentration
                # 与实际日换手率比较
                if daily_turnover_series is not None and len(daily_turnover_series) == len(predicted):
                    actual = daily_turnover_series.values[-len(predicted):]
                    loss = np.mean((predicted - actual) ** 2)
                else:
                    loss = 1.0
                return loss
            # 初始猜测和边界
            initial_guess = [0.5, 0.3]
            bounds = [(0, 1), (0, 1)]
            # 优化
            result = minimize(objective, initial_guess, bounds=bounds, method='L-BFGS-B')
            return {
                'alpha': result.x[0],
                'beta': result.x[1],
                'gamma': 0.2,  # 固定时间衰减参数
                'success': result.success,
                'loss': result.fun
            }
        except Exception as e:
            logger.error(f"优化参数失败: {e}")
            return {'alpha': 0.5, 'beta': 0.3, 'gamma': 0.2}
    
    def _calculate_holding_matrix(self, chip_matrix: np.ndarray, turnover_matrix: np.ndarray, params: Dict[str, float]) -> np.ndarray:
        """计算持有时间矩阵 - 修复比例错误"""
        try:
            n_days = chip_matrix.shape[0]
            n_prices = chip_matrix.shape[1]
            
            # 初始化持有时间矩阵
            holding_matrix = np.zeros((n_prices, self.max_holding_days))
            
            # 🚨 关键修复：确保初始分配总和为1
            # 每个价格区间的筹码总和应为1（100%）
            for price_idx in range(n_prices):
                # 初始分配：短线30%，中线40%，长线30%
                holding_matrix[price_idx, 0] = 0.3  # 0天持有（当天买入）
                holding_matrix[price_idx, 30] = 0.4  # 30天持有
                holding_matrix[price_idx, 90] = 0.3  # 90天持有
            
            # 行归一化
            for i in range(n_prices):
                row_sum = holding_matrix[i, :].sum()
                if row_sum > 0:
                    holding_matrix[i, :] = holding_matrix[i, :] / row_sum
            
            if n_days <= 1:
                return holding_matrix
            
            # 模拟每天的筹码流动
            for day_idx in range(1, min(n_days, 60)):
                # 获取当天的换手率
                if day_idx < turnover_matrix.shape[0]:
                    day_turnover_rate = turnover_matrix[day_idx, :]
                else:
                    day_turnover_rate = turnover_matrix[-1, :]
                
                # 🚨 关键修复：确保换手率合理
                day_turnover_rate = np.clip(day_turnover_rate, 0.001, 0.3)
                
                new_holding_matrix = np.zeros((n_prices, self.max_holding_days))
                
                for price_idx in range(n_prices):
                    current_turnover = day_turnover_rate[price_idx]
                    
                    # 处理当前持有的筹码
                    for holding_day in range(self.max_holding_days):
                        current_chips = holding_matrix[price_idx, holding_day]
                        if current_chips > 0:
                            # 持有时间越长，换手率越低
                            turnover_decay = max(0.1, 1.0 - holding_day / 200)
                            effective_turnover = current_turnover * turnover_decay
                            
                            # 继续持有
                            if holding_day < self.max_holding_days - 1:
                                new_holding_matrix[price_idx, holding_day + 1] += current_chips * (1 - effective_turnover)
                            else:
                                # 最大持有期，不再增加天数
                                new_holding_matrix[price_idx, holding_day] += current_chips * (1 - effective_turnover)
                            
                            # 换手卖出后成为短线筹码
                            new_holding_matrix[price_idx, 0] += current_chips * effective_turnover
                    
                    # 🚨 关键修复：当天新增筹码，使用合理的比例
                    # 当天该价格区间新增的筹码占全部新增筹码的比例
                    if day_idx < chip_matrix.shape[0]:
                        daily_chip_change = np.abs(chip_matrix[day_idx, price_idx] - chip_matrix[day_idx-1, price_idx])
                        if daily_chip_change > 0:
                            # 新增筹码分配：70%为短线，30%根据市场情况分配
                            new_chips_ratio = daily_chip_change * 0.01  # 使用小的比例
                            new_holding_matrix[price_idx, 0] += new_chips_ratio * 0.7
                            if day_idx > 10:
                                new_holding_matrix[price_idx, 20] += new_chips_ratio * 0.2
                            if day_idx > 30:
                                new_holding_matrix[price_idx, 60] += new_chips_ratio * 0.1
                
                # 行归一化
                for i in range(n_prices):
                    row_sum = new_holding_matrix[i, :].sum()
                    if row_sum > 0:
                        new_holding_matrix[i, :] = new_holding_matrix[i, :] / row_sum
                
                holding_matrix = new_holding_matrix.copy()
            
            # 最终验证
            for i in range(n_prices):
                row_sum = holding_matrix[i, :].sum()
                if abs(row_sum - 1.0) > 0.01:
                    print(f"⚠️ 第{i}行筹码总和异常: {row_sum:.4f}")
                    holding_matrix[i, :] = holding_matrix[i, :] / row_sum
            
            return holding_matrix
            
        except Exception as e:
            print(f"❌ [_calculate_holding_matrix] 计算失败: {e}")
            # 返回合理的默认矩阵
            default_matrix = np.zeros((n_prices, self.max_holding_days))
            for i in range(n_prices):
                default_matrix[i, 0] = 0.3
                default_matrix[i, 30] = 0.4
                default_matrix[i, 90] = 0.3
                row_sum = default_matrix[i, :].sum()
                if row_sum > 0:
                    default_matrix[i, :] = default_matrix[i, :] / row_sum
            return default_matrix

    def _calculate_holding_factors(self,holding_matrix: np.ndarray,chip_matrix: np.ndarray,data_dict: Dict[str, any]) -> Dict[str, float]:
        """计算持有时间相关因子 - 优化短线筹码计算"""
        try:
            factors = {}
            if holding_matrix.size == 0:
                return self._get_default_factors()
            print(f"📊 [_calculate_holding_factors] 持有矩阵形状: {holding_matrix.shape}")
            # 基础计算
            n_prices = holding_matrix.shape[0]
            # 计算各种持有时间的筹码比例
            short_term_mask = np.arange(self.max_holding_days) < 5
            mid_term_mask = (np.arange(self.max_holding_days) >= 5) & (np.arange(self.max_holding_days) < 60)
            long_term_mask = np.arange(self.max_holding_days) >= 60
            short_term_ratios = []
            mid_term_ratios = []
            long_term_ratios = []
            for i in range(n_prices):
                row_sum = holding_matrix[i, :].sum()
                if row_sum > 0:
                    row = holding_matrix[i, :] / row_sum
                    short_term_ratios.append(row[short_term_mask].sum())
                    mid_term_ratios.append(row[mid_term_mask].sum())
                    long_term_ratios.append(row[long_term_mask].sum())
            if short_term_ratios:
                base_short = float(np.mean(short_term_ratios))
                base_mid = float(np.mean(mid_term_ratios))
                base_long = float(np.mean(long_term_ratios))
            else:
                base_short = 0.2
                base_mid = 0.3
                base_long = 0.5
            print(f"📊 [基础计算] 短线: {base_short:.2%}, 中线: {base_mid:.2%}, 长线: {base_long:.2%}")
            # 根据换手率调整短线筹码
            if 'daily_turnover' in data_dict and isinstance(data_dict['daily_turnover'], pd.Series):
                turnover_series = data_dict['daily_turnover']
                if len(turnover_series) >= 5:
                    recent_turnover = turnover_series.iloc[-5:].mean()
                    # 关键改进：换手率与短线筹码关系
                    # 高换手率（>10%）：短线筹码应该更高（30-50%）
                    # 中换手率（5-10%）：适中（20-30%）
                    # 低换手率（<5%）：较低（10-20%）
                    if recent_turnover > 0.1:  # 换手率>10%
                        target_short = min(0.5, base_short * 1.5)
                        print(f"📊 [换手率调整] 高换手率{recent_turnover:.2%}，提升短线筹码")
                    elif recent_turnover > 0.05:  # 换手率5-10%
                        target_short = min(0.35, base_short * 1.2)
                        print(f"📊 [换手率调整] 中换手率{recent_turnover:.2%}，适度提升短线筹码")
                    else:  # 换手率<5%
                        target_short = max(0.1, base_short * 0.8)
                        print(f"📊 [换手率调整] 低换手率{recent_turnover:.2%}，降低短线筹码")
                    # 调整比例
                    short_change = target_short - base_short
                    total_other = base_mid + base_long
                    if total_other > 0:
                        factors['short_term_ratio'] = target_short
                        factors['mid_term_ratio'] = max(0.05, base_mid - short_change * (base_mid / total_other))
                        factors['long_term_ratio'] = max(0.05, base_long - short_change * (base_long / total_other))
                    else:
                        factors['short_term_ratio'] = base_short
                        factors['mid_term_ratio'] = base_mid
                        factors['long_term_ratio'] = base_long
                else:
                    factors['short_term_ratio'] = base_short
                    factors['mid_term_ratio'] = base_mid
                    factors['long_term_ratio'] = base_long
            else:
                factors['short_term_ratio'] = base_short
                factors['mid_term_ratio'] = base_mid
                factors['long_term_ratio'] = base_long
            # 应用A股长线模型（如果可用）
            if 'chip_dist_current' in data_dict and 'daily_turnover' in data_dict:
                try:
                    market_env = {'daily_turnover': data_dict['daily_turnover']}
                    long_term_model_result = self._calculate_china_a_share_long_term_factors(
                        stock_code='',
                        chip_dist_current=data_dict['chip_dist_current'],
                        historical_data=data_dict.get('daily_data', pd.DataFrame()),
                        market_environment=market_env
                    )
                    if 'final_long_term_ratio' in long_term_model_result:
                        model_long_term = long_term_model_result['final_long_term_ratio']
                        # 融合：A股模型主要影响长线比例
                        blend_factor = 0.6  # 60%权重给A股模型
                        final_long_term = model_long_term * blend_factor + factors['long_term_ratio'] * (1 - blend_factor)
                        # 调整其他比例
                        long_change = final_long_term - factors['long_term_ratio']
                        total_short_mid = factors['short_term_ratio'] + factors['mid_term_ratio']
                        if total_short_mid > 0:
                            factors['long_term_ratio'] = final_long_term
                            factors['short_term_ratio'] = max(0.05, factors['short_term_ratio'] - long_change * (factors['short_term_ratio'] / total_short_mid))
                            factors['mid_term_ratio'] = max(0.05, factors['mid_term_ratio'] - long_change * (factors['mid_term_ratio'] / total_short_mid))
                        print(f"📊 [模型融合] A股模型长线: {model_long_term:.2%}, 最终长线: {final_long_term:.2%}")
                except Exception as e:
                    print(f"⚠️ [模型融合] A股长线模型失败: {e}")
            # 计算平均持有时间
            holding_days = np.arange(self.max_holding_days)
            avg_days = 0
            valid_rows = 0
            for i in range(n_prices):
                row_sum = holding_matrix[i, :].sum()
                if row_sum > 0:
                    row = holding_matrix[i, :] / row_sum
                    avg_days += np.sum(row * holding_days)
                    valid_rows += 1
            factors['avg_holding_days'] = float(avg_days / valid_rows if valid_rows > 0 else 100.0)
            # 计算其他因子
            self._calculate_additional_factors(factors, holding_matrix, data_dict)
            # 最终归一化确保总和为1
            total = factors.get('short_term_ratio', 0) + factors.get('mid_term_ratio', 0) + factors.get('long_term_ratio', 0)
            if total > 0:
                factors['short_term_ratio'] = factors.get('short_term_ratio', 0) / total
                factors['mid_term_ratio'] = factors.get('mid_term_ratio', 0) / total
                factors['long_term_ratio'] = factors.get('long_term_ratio', 0) / total
            print(f"✅ [_calculate_holding_factors] 最终比例:")
            print(f"   短线筹码: {factors.get('short_term_ratio', 0):.2%}")
            print(f"   中线筹码: {factors.get('mid_term_ratio', 0):.2%}")
            print(f"   长线筹码: {factors.get('long_term_ratio', 0):.2%}")
            print(f"   平均持有: {factors.get('avg_holding_days', 0):.1f}天")
            return factors
        except Exception as e:
            print(f"❌ [_calculate_holding_factors] 计算失败: {e}")
            import traceback
            traceback.print_exc()
            return self._get_default_factors()

    def _calculate_china_a_share_long_term_factors(self,stock_code: str,chip_dist_current: pd.DataFrame,historical_data: pd.DataFrame,market_environment: Dict) -> Dict[str, float]:
        """中国A股特有的长线筹码因子计算模型 - 简化稳定版本"""
        long_term_factors = {}
        try:
            print(f"🔍 [A股长线模型] 开始计算 {stock_code} 的长线筹码结构")
            # 1. 基础价格分析
            price_current = chip_dist_current['price'].mean() if not chip_dist_current.empty else 10.0
            # 2. 历史高点估算
            price_history_high = self._estimate_historical_high(historical_data, price_current)
            # 3. 高位套牢盘计算
            long_term_factors['high_position_lock_ratio'] = self._calculate_high_position_lock_ratio(chip_dist_current, price_history_high)
            # 4. 机构持仓分析（简化）
            institutional_ratio = self._estimate_institutional_holding_simple(market_environment)
            long_term_factors['institutional_holding_ratio'] = institutional_ratio
            # 5. 周期位置判断
            cycle_position = self._identify_market_cycle_position_simple(historical_data, price_current)
            # 6. 根据周期位置设定长线筹码预期
            expected_long_term = self._get_expected_long_term_by_cycle(cycle_position)
            long_term_factors['expected_long_term_ratio'] = expected_long_term
            # 7. 筹码集中度
            concentration = self._calculate_chip_concentration_simple(chip_dist_current)
            long_term_factors['concentration_based_long_term'] = concentration
            # 8. 综合计算
            final_long_term = self._calculate_final_long_term_ratio(long_term_factors, market_environment)
            long_term_factors['final_long_term_ratio'] = final_long_term
            print(f"✅ [A股长线模型] 最终长线筹码比例: {final_long_term:.2%}")
            return long_term_factors
        except Exception as e:
            print(f"❌ [A股长线模型] 计算失败: {e}")
            import traceback
            traceback.print_exc()
            return {'final_long_term_ratio': 0.35}

    def _calculate_high_position_lock_ratio(self,chip_dist: pd.DataFrame,historical_high: float) -> float:
        """计算高位套牢盘比例（中国A股特有）"""
        try:
            if chip_dist.empty or historical_high <= 0:
                return 0.0
            current_price = chip_dist['price'].mean()
            # 定义高位区域：历史高点的80%以上
            high_threshold = historical_high * 0.8
            # 计算高位套牢盘
            high_mask = chip_dist['price'] >= high_threshold
            if not high_mask.any():
                return 0.0
            high_chip_sum = chip_dist.loc[high_mask, 'percent'].sum()
            total_chip_sum = chip_dist['percent'].sum()
            if total_chip_sum > 0:
                ratio = high_chip_sum / total_chip_sum
                print(f"📊 [高位套牢] 当前价: {current_price:.2f}, 历史高点: {historical_high:.2f}, 阈值: {high_threshold:.2f}, 套牢比例: {ratio:.2%}")
                return float(ratio)
            return 0.0
        except:
            return 0.0

    def _calculate_additional_factors(self,factors: Dict[str, float],holding_matrix: np.ndarray,data_dict: Dict[str, any]) -> None:
        """计算额外的筹码因子"""
        try:
            # 1. 高位筹码沉淀比例（90%分位以上）
            if 'price_grid' in data_dict and 'chip_dist_current' in data_dict:
                price_grid = data_dict['price_grid']
                chip_dist_current = data_dict['chip_dist_current']
                if len(price_grid) > 0 and not chip_dist_current.empty:
                    # 找到90%分位价格
                    sorted_prices = np.sort(price_grid)
                    idx_90 = int(len(sorted_prices) * 0.9)
                    price_90 = sorted_prices[idx_90]
                    # 计算价格高于90%分位的筹码比例
                    high_mask = chip_dist_current['price'] >= price_90
                    if high_mask.any():
                        high_chip_sum = chip_dist_current.loc[high_mask, 'percent'].sum()
                        total_chip_sum = chip_dist_current['percent'].sum()
                        if total_chip_sum > 0:
                            high_chip_ratio = high_chip_sum / total_chip_sum
                            factors['high_position_lock_ratio_90'] = float(high_chip_ratio)
                            print(f"📊 [_calculate_additional_factors] 高位筹码沉淀比例: {high_chip_ratio:.4f}")
            # 2. 主力成本区间锁定比例（50分位±10%）
            if 'chip_dist_current' in data_dict and not data_dict['chip_dist_current'].empty:
                chip_dist_current = data_dict['chip_dist_current']
                sorted_chips = chip_dist_current.sort_values('price')
                cumsum = sorted_chips['percent'].cumsum()
                if (cumsum >= 0.5).any():
                    idx_50 = (cumsum >= 0.5).idxmax()
                    cost_50pct = sorted_chips.loc[idx_50, 'price']
                    lower_bound = cost_50pct * 0.9
                    upper_bound = cost_50pct * 1.1
                    main_mask = (chip_dist_current['price'] >= lower_bound) & (chip_dist_current['price'] <= upper_bound)
                    main_chip_sum = chip_dist_current.loc[main_mask, 'percent'].sum()
                    total_chip_sum = chip_dist_current['percent'].sum()
                    if total_chip_sum > 0:
                        main_range_ratio = main_chip_sum / total_chip_sum
                        factors['main_cost_range_ratio'] = float(main_range_ratio)
                        print(f"📊 [_calculate_additional_factors] 主力成本区间锁定比例: {main_range_ratio:.4f}")
            # 3. 换手率调整因子
            if 'daily_turnover' in data_dict and isinstance(data_dict['daily_turnover'], pd.Series):
                recent_turnover = data_dict['daily_turnover'].iloc[-5:].mean() if len(data_dict['daily_turnover']) >= 5 else 0
                factors['turnover_adjustment'] = float(recent_turnover)
                print(f"📊 [_calculate_additional_factors] 5日平均换手率: {recent_turnover:.2%}")
            # 4. 筹码集中度
            if 'chip_dist_current' in data_dict and not data_dict['chip_dist_current'].empty:
                chip_dist = data_dict['chip_dist_current']
                if not chip_dist.empty:
                    chip_mean = chip_dist['price'].mean()
                    if chip_mean > 0:
                        chip_std = chip_dist['price'].std()
                        chip_concentration = chip_std / chip_mean
                        factors['chip_concentration_ratio'] = float(chip_concentration)
                        print(f"📊 [_calculate_additional_factors] 筹码集中度: {chip_concentration:.4f}")
        except Exception as e:
            print(f"⚠️ [_calculate_additional_factors] 计算额外因子失败: {e}")

    def _calculate_mid_position_lock_ratio(self,chip_dist: pd.DataFrame,current_price: float) -> float:
        """计算中位套牢盘比例"""
        try:
            if chip_dist.empty or current_price <= 0:
                return 0.0
            # 定义中位区域：当前价格的±15%
            lower_bound = current_price * 0.85
            upper_bound = current_price * 1.15
            # 计算中位套牢盘
            mid_mask = (chip_dist['price'] >= lower_bound) & (chip_dist['price'] <= upper_bound)
            if not mid_mask.any():
                return 0.0
            mid_chip_sum = chip_dist.loc[mid_mask, 'percent'].sum()
            total_chip_sum = chip_dist['percent'].sum()
            if total_chip_sum > 0:
                ratio = mid_chip_sum / total_chip_sum
                print(f"📊 [中位套牢] 当前价: {current_price:.2f}, 区间: [{lower_bound:.2f}, {upper_bound:.2f}], 中位比例: {ratio:.2%}")
                return float(ratio)
            return 0.0
        except Exception as e:
            print(f"⚠️ [_calculate_mid_position_lock_ratio] 计算失败: {e}")
            return 0.0

    def _calculate_unlocking_pressure(self,float_shares_change: float,stock_code: str) -> float:
        """计算解禁压力比例"""
        try:
            # 简化解禁压力估算
            # 实际应用中应从数据库获取解禁数据
            if float_shares_change > 0:
                # 流通股本增加，解禁压力大
                pressure = min(0.5, float_shares_change * 2)
                return float(pressure)
            return 0.0
        except:
            return 0.0

    def _calculate_chip_concentration_ratio(self,chip_dist: pd.DataFrame) -> float:
        """计算筹码集中度比率"""
        try:
            if chip_dist.empty or len(chip_dist) < 3:
                return 0.5
            # 使用价格标准差与价格范围的比值
            price_range = chip_dist['price'].max() - chip_dist['price'].min()
            if price_range > 0:
                price_std = chip_dist['price'].std()
                concentration = 1.0 - (price_std / price_range)
                return float(max(0.0, min(1.0, concentration)))
            return 0.5
        except:
            return 0.5

    def _estimate_institutional_holding(self,turnover_20d: float,volume_ratio: float) -> float:
        """估算机构持仓比例"""
        try:
            # 基于换手率和量比估算机构持仓
            # 换手率越低，机构持仓通常越高
            # 量比稳定在1附近，表明交易正常，机构可能持仓稳定
            if turnover_20d < 0.01:  # 换手率低于1%
                base_ratio = 0.7
            elif turnover_20d < 0.02:  # 换手率1-2%
                base_ratio = 0.6
            elif turnover_20d < 0.05:  # 换手率2-5%
                base_ratio = 0.5
            elif turnover_20d < 0.1:   # 换手率5-10%
                base_ratio = 0.3
            else:                      # 换手率高于10%
                base_ratio = 0.2
            # 量比调整：量比接近1，机构持仓稳定
            volume_adjustment = 1.0 - min(0.3, abs(volume_ratio - 1.0) * 0.5)
            final_ratio = base_ratio * volume_adjustment
            return min(0.8, max(0.1, final_ratio))
        except:
            return 0.4

    def _estimate_historical_high(self,historical_data: pd.DataFrame,current_price: float) -> float:
        """估算历史高点"""
        try:
            if historical_data.empty:
                return current_price * 1.5
            # 尝试获取价格数据
            for col in ['high', 'high_qfq', 'close', 'close_qfq', 'price']:
                if col in historical_data.columns:
                    high_value = historical_data[col].max()
                    if high_value > current_price:
                        return float(high_value)
            return current_price * 1.5
        except:
            return current_price * 1.5

    def _estimate_institutional_holding_simple(self,market_environment: Dict) -> float:
        """简化机构持仓估算"""
        try:
            if 'daily_turnover' not in market_environment:
                return 0.4
            turnover_series = market_environment['daily_turnover']
            if isinstance(turnover_series, pd.Series) and len(turnover_series) >= 20:
                turnover_20d = turnover_series.iloc[-20:].mean()
            else:
                turnover_20d = 0.05
            # 基于换手率的简单估算
            if turnover_20d < 0.01:  # 低于1%
                return 0.7
            elif turnover_20d < 0.02:  # 1-2%
                return 0.6
            elif turnover_20d < 0.05:  # 2-5%
                return 0.5
            elif turnover_20d < 0.1:   # 5-10%
                return 0.3
            else:                      # 高于10%
                return 0.2
        except:
            return 0.4

    def _identify_market_cycle_position_simple(self,historical_data: pd.DataFrame,current_price: float) -> str:
        """简化市场周期位置判断"""
        try:
            if historical_data.empty or len(historical_data) < 20:
                return 'consolidation'
            # 获取收盘价序列
            close_series = None
            for col in ['close', 'close_qfq', 'price', 'value']:
                if col in historical_data.columns:
                    close_series = historical_data[col]
                    break
            if close_series is None:
                return 'consolidation'
            if len(close_series) < 20:
                return 'consolidation'
            # 计算20日涨幅
            if len(close_series) >= 20:
                price_20d_ago = close_series.iloc[-20]
                price_change_20d = (current_price - price_20d_ago) / price_20d_ago if price_20d_ago > 0 else 0
                if price_change_20d > 0.15:  # 20日涨幅超过15%
                    return 'lifting'
                elif price_change_20d < -0.1:  # 20日跌幅超过10%
                    return 'accumulation'
                else:
                    return 'consolidation'
            return 'consolidation'
        except:
            return 'consolidation'

    def _get_expected_long_term_by_cycle(self,cycle_position: str) -> float:
        """根据周期位置获取预期长线比例"""
        cycle_map = {
            'accumulation': 0.6,   # 吸筹期：长线筹码高
            'lifting': 0.4,        # 拉升期：适中
            'distribution': 0.2,    # 派发期：低
            'decline': 0.3,        # 下跌期：适中
            'consolidation': 0.35   # 整理期：中等
        }
        return cycle_map.get(cycle_position, 0.35)

    def _calculate_chip_concentration_simple(self,chip_dist: pd.DataFrame) -> float:
        """简化筹码集中度计算"""
        try:
            if chip_dist.empty or len(chip_dist) < 3:
                return 0.5
            # 计算价格变异系数
            price_mean = chip_dist['price'].mean()
            price_std = chip_dist['price'].std()
            if price_mean > 0:
                cv = price_std / price_mean
                # 变异系数越小，集中度越高
                concentration = 1.0 - min(1.0, cv)
                return float(concentration)
            return 0.5
        except:
            return 0.5

    def _calculate_final_long_term_ratio(self,long_term_factors: Dict,market_environment: Dict) -> float:
        """计算最终长线比例"""
        try:
            # 权重配置
            weights = {
                'high_position': 0.25,
                'institutional': 0.30,
                'expected': 0.25,
                'concentration': 0.20
            }
            total = 0.0
            weight_sum = 0.0
            # 高位套牢盘
            if 'high_position_lock_ratio' in long_term_factors:
                total += long_term_factors['high_position_lock_ratio'] * weights['high_position']
                weight_sum += weights['high_position']
            # 机构持仓
            if 'institutional_holding_ratio' in long_term_factors:
                total += long_term_factors['institutional_holding_ratio'] * weights['institutional']
                weight_sum += weights['institutional']
            # 周期预期
            if 'expected_long_term_ratio' in long_term_factors:
                total += long_term_factors['expected_long_term_ratio'] * weights['expected']
                weight_sum += weights['expected']
            # 集中度
            if 'concentration_based_long_term' in long_term_factors:
                total += long_term_factors['concentration_based_long_term'] * weights['concentration']
                weight_sum += weights['concentration']
            if weight_sum > 0:
                base_ratio = total / weight_sum
            else:
                base_ratio = 0.35
            # 换手率调整
            if 'daily_turnover' in market_environment:
                turnover_series = market_environment['daily_turnover']
                if isinstance(turnover_series, pd.Series) and len(turnover_series) >= 20:
                    turnover_20d = turnover_series.iloc[-20:].mean()
                    # 换手率越低，长线筹码应越高
                    turnover_adjust = 1.0 - min(0.5, turnover_20d * 1.5)
                    base_ratio = base_ratio * turnover_adjust
            # 限制范围
            final_ratio = min(0.7, max(0.05, base_ratio))
            return float(final_ratio)
        except:
            return 0.35

    def _identify_market_cycle_position(self,historical_data: pd.DataFrame) -> str:
        """识别市场周期位置"""
        try:
            if historical_data.empty or len(historical_data) < 60:
                return 'consolidation'
            # 计算技术指标
            close_prices = historical_data['close'] if 'close' in historical_data.columns else historical_data.iloc[:, 0]
            # 计算均线
            ma20 = close_prices.rolling(20).mean()
            ma60 = close_prices.rolling(60).mean()
            if len(ma20) < 1 or len(ma60) < 1:
                return 'consolidation'
            current_close = close_prices.iloc[-1]
            current_ma20 = ma20.iloc[-1]
            current_ma60 = ma60.iloc[-1]
            # 判断周期位置
            if current_close > current_ma20 > current_ma60:  # 多头排列
                if (current_close - current_ma60) / current_ma60 > 0.3:  # 涨幅超过30%
                    return 'distribution'  # 可能处于派发期
                else:
                    return 'lifting'  # 拉升期
            elif current_close < current_ma20 < current_ma60:  # 空头排列
                if (current_ma60 - current_close) / current_close > 0.2:  # 跌幅超过20%
                    return 'accumulation'  # 可能处于吸筹期
                else:
                    return 'decline'  # 下跌期
            else:
                return 'consolidation'  # 整理期
        except:
            return 'consolidation'

    def _validate_results(self,holding_matrix: np.ndarray,factors: Dict[str, float],data_dict: Dict[str, any]) -> Dict[str, any]:
        """验证计算结果合理性 - 修复验证逻辑错误"""
        validation = {'is_valid': True,'warnings': [],'checks_passed': 0,'total_checks': 5}
        try:
            # 修复检查1：持有时间矩阵总和应为n_prices（每行和为1）
            n_prices = holding_matrix.shape[0]
            row_sums = holding_matrix.sum(axis=1)
            avg_row_sum = np.mean(row_sums)
            print(f"📊 [_validate_results] 验证检查: 矩阵形状={holding_matrix.shape}, 平均行和={avg_row_sum:.6f}")
            if abs(avg_row_sum - 1.0) > 0.01:
                warning_msg = f"持有矩阵行和异常: 平均{avg_row_sum:.4f}, 期望1.0"
                validation['warnings'].append(warning_msg)
                validation['is_valid'] = False
                print(f"⚠️ [_validate_results] {warning_msg}")
                print(f"📊 [_validate_results] 行和范围: {row_sums.min():.6f} - {row_sums.max():.6f}")
            else:
                validation['checks_passed'] += 1
                print(f"✅ [_validate_results] 检查1通过: 持有矩阵行和正常")
            # 检查2：短线+中线+长线比例应接近1
            sum_ratios = factors.get('short_term_ratio', 0) + factors.get('mid_term_ratio', 0) + factors.get('long_term_ratio', 0)
            if abs(sum_ratios - 1.0) > 0.05:
                warning_msg = f"筹码比例总和异常: {sum_ratios:.4f}"
                validation['warnings'].append(warning_msg)
                validation['is_valid'] = False
                print(f"⚠️ [_validate_results] {warning_msg}")
            else:
                validation['checks_passed'] += 1
                print(f"✅ [_validate_results] 检查2通过: 筹码比例总和正常")
            # 检查3：平均持有时间应在合理范围内
            avg_days = factors.get('avg_holding_days', 0)
            if avg_days < 1 or avg_days > 500:
                warning_msg = f"平均持有时间异常: {avg_days:.1f}天"
                validation['warnings'].append(warning_msg)
                validation['is_valid'] = False
                print(f"⚠️ [_validate_results] {warning_msg}")
            else:
                validation['checks_passed'] += 1
                print(f"✅ [_validate_results] 检查3通过: 平均持有时间正常")
            # 修复检查4：与换手率一致性（使用更合理的预期）
            if 'daily_turnover' in data_dict and isinstance(data_dict['daily_turnover'], pd.Series):
                recent_turnover = data_dict['daily_turnover'].iloc[-5:].mean() if len(data_dict['daily_turnover']) >= 5 else 0
                # 修复预期计算：短线比例 ≈ 换手率 × 3（更合理的经验公式）
                expected_short_term = min(recent_turnover * 3, 0.8)
                actual_short_term = factors.get('short_term_ratio', 0)
                diff_threshold = 0.2  # 允许20%的差异
                if abs(actual_short_term - expected_short_term) > diff_threshold:
                    warning_msg = f"短线比例与换手率差异较大: 实际{actual_short_term:.2%}, 预期{expected_short_term:.2%} (换手率{recent_turnover:.2%})"
                    validation['warnings'].append(warning_msg)
                    print(f"⚠️ [_validate_results] {warning_msg}")
                else:
                    validation['checks_passed'] += 1
                    print(f"✅ [_validate_results] 检查4通过: 短线比例与换手率一致")
            # 检查5：价格区间合理性
            if 'price_range' in data_dict:
                price_min, price_max = data_dict['price_range']
                if price_max <= price_min or price_max <= 0:
                    warning_msg = f"价格区间异常: {price_min}-{price_max}"
                    validation['warnings'].append(warning_msg)
                    validation['is_valid'] = False
                    print(f"⚠️ [_validate_results] {warning_msg}")
                else:
                    validation['checks_passed'] += 1
                    print(f"✅ [_validate_results] 检查5通过: 价格区间正常")
            validation['score'] = validation['checks_passed'] / validation['total_checks']
            print(f"📊 [_validate_results] 验证完成: 通过{validation['checks_passed']}/{validation['total_checks']}项检查, 分数={validation['score']:.2f}")
            print(f"📊 [_validate_results] 警告数量: {len(validation['warnings'])}")
            return validation
        except Exception as e:
            logger.error(f"验证结果失败: {e}")
            warning_msg = f"验证过程异常: {str(e)}"
            validation['warnings'].append(warning_msg)
            validation['is_valid'] = False
            print(f"❌ [_validate_results] {warning_msg}")
            return validation

    def _get_default_result(self, stock_code: str = "", trade_date: str = "") -> Dict[str, any]:
        """获取默认结果（计算失败时返回）"""
        return {
            'stock_code': stock_code,
            'trade_date': trade_date,
            'holding_matrix': np.array([]),
            'price_grid': np.array([]),  # 确保有price_grid字段
            'factors': self._get_default_factors(),
            'validation': {
                'is_valid': False,
                'warnings': ['计算失败'],  # 确保有warnings字段
                'checks_passed': 0,
                'total_checks': 5,
                'score': 0
            },
            'calc_status': 'failed',
            'calc_time': datetime.now()
        }

    def _get_default_factors(self) -> Dict[str, float]:
        """获取默认因子值"""
        return {
            'short_term_ratio': 0.2,
            'mid_term_ratio': 0.3,
            'long_term_ratio': 0.5,
            'avg_holding_days': 100.0,
            'concentration_adjusted_holding': 100.0,
            'high_position_lock_ratio_90': 0.0,
            'main_cost_range_ratio': 0.5,
            'turnover_adjustment': 0.02,
            'price_position': 0.5
        }

    def save_holding_matrix_to_db(self, stock_code: str, trade_date: str, result: Dict[str, any]) -> bool:
        """
        将持有时间矩阵保存到数据库
        需要先创建对应的数据库模型
        """
        try:
            print(f"💾 [保存矩阵] 开始保存持有矩阵: {stock_code} {trade_date}")
            # 压缩矩阵数据（可以保存为JSON或二进制）
            import json
            import base64
            import pickle
            ChipHoldingMatrixModel = get_chip_holding_matrix_model_by_code(stock_code)
            print(f"💾 [保存矩阵] 获取模型: {ChipHoldingMatrixModel}")
            # 将矩阵转换为可存储格式
            if 'holding_matrix' in result and result['holding_matrix'].size > 0:
                # 获取价格网格
                price_grid_data = []
                if 'price_grid' in result and result['price_grid'].size > 0:
                    price_grid_data = result['price_grid'].tolist()
                    print(f"💾 [保存矩阵] 价格网格数据: {len(price_grid_data)}个点")
                # 获取验证警告
                validation_warnings_data = []
                if 'validation' in result and 'warnings' in result['validation']:
                    validation_warnings_data = result['validation']['warnings']
                    print(f"💾 [保存矩阵] 验证警告: {len(validation_warnings_data)}条")
                # 方法1：保存为JSON（适合小矩阵）
                try:
                    matrix_data = {
                        'matrix': result['holding_matrix'].tolist(),
                        'price_grid': price_grid_data
                    }
                    # 将JSON数据转换为字符串
                    matrix_json = json.dumps(matrix_data, ensure_ascii=False)
                    print(f"💾 [保存矩阵] JSON数据长度: {len(matrix_json)}")
                except Exception as e:
                    print(f"⚠️ [保存矩阵] JSON转换失败: {e}")
                    matrix_json = "{}"
                # 方法2：保存为压缩二进制（推荐）
                compressed_data = b""  # 初始化为空字节串
                try:
                    matrix_bytes = pickle.dumps(result['holding_matrix'])
                    # 修复编码问题：确保encode方法的正确使用
                    compressed_data = base64.b64encode(matrix_bytes)
                    print(f"💾 [保存矩阵] 压缩数据长度: {len(compressed_data)} 字节")
                except Exception as e:
                    print(f"⚠️ [保存矩阵] 二进制压缩失败: {e}")
                    import traceback
                    traceback.print_exc()
                # 准备保存的数据 - 只包含 ChipHoldingMatrix 模型中存在的字段
                # 获取因子值并清理 NaN
                short_term_ratio = result['factors'].get('short_term_ratio', 0)
                mid_term_ratio = result['factors'].get('mid_term_ratio', 0)
                long_term_ratio = result['factors'].get('long_term_ratio', 0)
                avg_holding_days = result['factors'].get('avg_holding_days', 0)
                validation_score = result.get('validation', {}).get('score', 0)
                # 清理 NaN 值，转换为 None
                def clean_nan(value):
                    import math
                    if value is None:
                        return None
                    if isinstance(value, float) and math.isnan(value):
                        return None
                    return value
                defaults = {
                    'short_term_ratio': clean_nan(short_term_ratio),
                    'mid_term_ratio': clean_nan(mid_term_ratio),
                    'long_term_ratio': clean_nan(long_term_ratio),
                    'avg_holding_days': clean_nan(avg_holding_days),
                    'matrix_data': matrix_json,  # 保存JSON数据
                    'compressed_matrix': compressed_data,  # 保存压缩数据（已经是bytes）
                    'price_grid': price_grid_data,  # 保存价格网格
                    'validation_warnings': validation_warnings_data,  # 保存验证警告
                    'calc_status': result.get('calc_status', 'failed'),
                    'validation_score': clean_nan(validation_score),
                }
                # 检查所有浮点字段是否为 None，如果是则设置默认值
                float_fields = ['short_term_ratio', 'mid_term_ratio', 'long_term_ratio', 'avg_holding_days', 'validation_score']
                for field in float_fields:
                    if defaults[field] is None:
                        if field == 'validation_score':
                            defaults[field] = 0.0
                        elif field == 'avg_holding_days':
                            defaults[field] = 100.0
                        else:
                            defaults[field] = 0.0
                        print(f"⚠️ [保存矩阵] 字段 {field} 为NaN，已设置为默认值: {defaults[field]}")
                print(f"💾 [保存矩阵] 准备保存的字段: {list(defaults.keys())}")
                print(f"💾 [保存矩阵] 价格网格字段: {'有数据' if price_grid_data else '空'}")
                print(f"💾 [保存矩阵] 验证警告字段: {'有数据' if validation_warnings_data else '空'}")
                print(f"💾 [保存矩阵] 字段值检查:")
                for field in float_fields:
                    print(f"  {field}: {defaults[field]} (type: {type(defaults[field])})")
                # 转换trade_date为date对象
                try:
                    trade_date_dt = datetime.strptime(trade_date, "%Y-%m-%d").date()
                    print(f"💾 [保存矩阵] 交易日期转换: {trade_date} -> {trade_date_dt}")
                except:
                    try:
                        trade_date_dt = datetime.strptime(trade_date, "%Y%m%d").date()
                    except:
                        print(f"❌ [保存矩阵] 日期格式无法解析: {trade_date}")
                        return False
                # 获取股票对象
                from stock_models.stock_basic import StockInfo
                try:
                    stock = StockInfo.objects.get(stock_code=stock_code)
                except StockInfo.DoesNotExist:
                    print(f"❌ [保存矩阵] 股票不存在: {stock_code}")
                    return False
                # 创建或更新记录
                try:
                    holding_record, created = ChipHoldingMatrixModel.objects.update_or_create(
                        stock=stock,
                        trade_time=trade_date_dt,
                        defaults=defaults
                    )
                    print(f"💾 [保存矩阵] 保存{'成功' if created else '已更新'}: ID={holding_record.id}")
                    return True
                except Exception as e:
                    print(f"❌ [保存矩阵] 数据库操作失败: {e}")
                    import traceback
                    traceback.print_exc()
                    return False
            else:
                print(f"⚠️ [保存矩阵] 无有效的持有矩阵数据")
                return False
        except Exception as e:
            logger.error(f"保存持有矩阵失败 {stock_code} {trade_date}: {e}")
            print(f"❌ [保存矩阵异常] {e}")
            import traceback
            traceback.print_exc()
            return False

    # 快速计算函数（保持向后兼容）
    def calculate_holding_factors_daily(stock_code: str, trade_date: str) -> Dict[str, float]:
        """
        快速计算持有时间因子（简化接口）
        """
        service = ChipHoldingService(use_tick_data=True)
        result = service.calculate_holding_matrix_daily(stock_code, trade_date)
        return result.get('factors', {})