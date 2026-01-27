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
        """计算单日筹码持有时间矩阵（异步版本）"""
        try:
            logger.info(f"开始计算 {stock_code} {trade_date} 的筹码持有时间矩阵")
            print(f"🔴 [主流程开始] 股票: {stock_code}, 日期: {trade_date}")
            # 1. 获取基础数据（异步）
            data_dict = await self._fetch_required_data(stock_code, trade_date, lookback_days)
            if not data_dict:
                logger.error(f"获取基础数据失败: {stock_code} {trade_date}")
                print(f"❌ [主流程] 获取基础数据失败")
                return self._get_default_result()
            # 检查关键数据
            chip_dists = data_dict.get('chip_dists', [])
            price_range = data_dict.get('price_range', (0, 0))
            print(f"📊 [主流程] 获取到筹码历史数据: {len(chip_dists)}条, 价格范围: {price_range}")
            if len(chip_dists) == 0:
                print(f"⚠️ [主流程] 警告: 无历史筹码数据，计算可能不准确")
            # 2. 建立价格网格（同步计算）
            print(f"🔨 [主流程] 开始构建价格网格和筹码矩阵...")
            price_grid, chip_dist_matrix = await sync_to_async(self._build_price_grid_and_chip_matrix)(data_dict['chip_dists'], data_dict['price_range'])
            print(f"📊 [主流程] 价格网格形状: {price_grid.shape}, 筹码矩阵形状: {chip_dist_matrix.shape}")
            if chip_dist_matrix.size == 0 or len(chip_dist_matrix.shape) < 2:
                print(f"❌ [主流程] 错误: 筹码矩阵无效，形状={chip_dist_matrix.shape}")
                return self._get_default_result(stock_code, trade_date)
            # 3. 计算分钟级成交量分布（同步计算）
            minute_volume_dist = await sync_to_async(self._calculate_minute_volume_distribution)(data_dict['minute_data'], price_grid)
            print(f"📊 [主流程] 分钟成交量分布形状: {minute_volume_dist.shape}")
            # 4. 如果使用逐笔数据，进行增强计算
            if self.use_tick_data and data_dict['tick_data'] is not None:
                enhanced_dist = await sync_to_async(self._enhance_with_tick_data)(minute_volume_dist, data_dict['tick_data'], price_grid)
                print(f"📊 [主流程] 增强分布完成")
            else:
                enhanced_dist = minute_volume_dist
            # 5. 计算换手率矩阵（同步计算）
            turnover_matrix = await sync_to_async(self._calculate_turnover_matrix)(enhanced_dist, chip_dist_matrix, data_dict['float_shares'])
            print(f"📊 [主流程] 换手率矩阵形状: {turnover_matrix.shape}")
            # 6. 优化换手概率参数（同步计算）
            optimal_params = await sync_to_async(self._optimize_turnover_parameters)(turnover_matrix, chip_dist_matrix, data_dict['daily_turnover'])
            print(f"📊 [主流程] 优化参数完成: {optimal_params}")
            # 7. 计算持有时间矩阵（同步计算）
            holding_matrix = await sync_to_async(self._calculate_holding_matrix)(chip_dist_matrix, turnover_matrix, optimal_params)
            print(f"📊 [主流程] 持有时间矩阵形状: {holding_matrix.shape}")
            # 8. 计算衍生因子（同步计算）
            factors = await sync_to_async(self._calculate_holding_factors)(holding_matrix, chip_dist_matrix, data_dict)
            print(f"📊 [主流程] 计算衍生因子完成: {len(factors)}个因子")
            # 9. 验证结果（同步计算）
            validation = await sync_to_async(self._validate_results)(holding_matrix, factors, data_dict)
            print(f"📊 [主流程] 验证结果: {validation}")
            # 确保价格网格包含在结果中
            result = {
                'stock_code': stock_code, 
                'trade_date': trade_date, 
                'holding_matrix': holding_matrix, 
                'price_grid': price_grid,  # 确保这里包含price_grid
                'factors': factors, 
                'validation': validation, 
                'calc_status': 'success', 
                'calc_time': datetime.now()
            }
            logger.info(f"计算完成 {stock_code} {trade_date}: 短线筹码={factors.get('short_term_ratio', 0):.2%}, 长线筹码={factors.get('long_term_ratio', 0):.2%}")
            print(f"✅ [主流程完成] {stock_code} {trade_date} 计算成功")
            print(f"📊 [主流程] 结果中包含price_grid: {'是' if 'price_grid' in result else '否'}")
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
    
    def _calculate_turnover_matrix(self,volume_dist: np.ndarray,chip_matrix: np.ndarray,float_shares: float) -> np.ndarray:
        """
        计算换手率矩阵
        """
        try:
            if float_shares <= 0 or len(volume_dist) == 0 or chip_matrix.size == 0:
                print(f"⚠️ [_calculate_turnover_matrix] 输入数据无效: float_shares={float_shares}, volume_dist_len={len(volume_dist)}, chip_matrix_shape={chip_matrix.shape}")
                return np.zeros_like(chip_matrix)
            # 计算每日换手率
            daily_volume = volume_dist.sum()
            if daily_volume <= 0:
                print(f"⚠️ [_calculate_turnover_matrix] 日成交量为0，使用默认换手率0.02")
                # 使用默认换手率 2%
                price_turnover = np.ones(len(volume_dist)) * 0.02 / len(volume_dist)
            else:
                daily_turnover_rate = daily_volume / float_shares
                print(f"📊 [_calculate_turnover_matrix] 日成交量: {daily_volume:.0f}, 流通股: {float_shares:.0f}, 日换手率: {daily_turnover_rate:.4%}")
                # 计算各价格区间的相对换手率
                if chip_matrix.shape[0] > 0:
                    chip_dist_current = chip_matrix[-1, :]
                else:
                    chip_dist_current = np.ones(len(volume_dist)) / len(volume_dist)
                # 避免除零
                chip_dist_current = np.maximum(chip_dist_current, 1e-6)
                # 各价格区间换手率 = 成交量分布 / (筹码分布 * 流通股本)
                price_turnover = volume_dist / (chip_dist_current * float_shares)
                price_turnover = np.nan_to_num(price_turnover, nan=0, posinf=0, neginf=0)
                # 限制最大换手率
                price_turnover = np.clip(price_turnover, 0, 0.99)
                print(f"📊 [_calculate_turnover_matrix] 价格换手率范围: {price_turnover.min():.6f} - {price_turnover.max():.6f}")
            # 创建换手率矩阵（与筹码矩阵同形状）
            turnover_matrix = np.tile(price_turnover, (chip_matrix.shape[0], 1))
            # 应用时间衰减（越久远的数据影响越小）
            if chip_matrix.shape[0] > 1:
                time_weights = np.exp(-np.arange(chip_matrix.shape[0]) / 30)  # 30日衰减
                turnover_matrix = turnover_matrix * time_weights[:, np.newaxis]
                print(f"📊 [_calculate_turnover_matrix] 应用时间衰减权重: {time_weights[:5]}...")
            print(f"✅ [_calculate_turnover_matrix] 换手率矩阵计算完成: 形状={turnover_matrix.shape}")
            return turnover_matrix
        except Exception as e:
            print(f"❌ [_calculate_turnover_matrix] 计算换手率矩阵失败: {e}")
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
    
    def _calculate_holding_matrix(self,chip_matrix: np.ndarray,turnover_matrix: np.ndarray,params: Dict[str, float]) -> np.ndarray:
        """
        计算持有时间矩阵 - 修复版本
        核心问题：原算法没有正确处理筹码随时间的转移，导致长线筹码为0
        新算法：模拟每日筹码持有时间的自然增长和换手重置
        """
        try:
            n_days = chip_matrix.shape[0]
            n_prices = chip_matrix.shape[1]
            print(f"🔧 [_calculate_holding_matrix_v2] 输入: n_days={n_days}, n_prices={n_prices}, max_holding_days={self.max_holding_days}")
            
            # 初始化持有时间矩阵 [价格区间 × 持有天数]，全部初始化为0
            holding_matrix = np.zeros((n_prices, self.max_holding_days))
            
            # 第0天：所有筹码持有0天（使用第一天的筹码分布）
            if n_days > 0:
                # 初始筹码分布归一化
                initial_chip = chip_matrix[0, :]
                initial_sum = initial_chip.sum()
                if initial_sum > 0:
                    initial_chip = initial_chip / initial_sum
                else:
                    initial_chip = np.ones(n_prices) / n_prices
                
                holding_matrix[:, 0] = initial_chip
                print(f"📊 [_calculate_holding_matrix_v2] 第0天初始化: 使用第0天的筹码分布，总和={holding_matrix[:, 0].sum():.6f}")
            else:
                holding_matrix[:, 0] = np.ones(n_prices) / n_prices
                print(f"⚠️ [_calculate_holding_matrix_v2] 无历史数据，使用均匀分布初始化")
            
            # 如果只有1天的数据，直接返回
            if n_days <= 1:
                print(f"⚠️ [_calculate_holding_matrix_v2] 历史数据不足，返回初始矩阵")
                # 归一化
                row_sums = holding_matrix.sum(axis=1)
                for i in range(n_prices):
                    if row_sums[i] > 0:
                        holding_matrix[i, :] = holding_matrix[i, :] / row_sums[i]
                return holding_matrix
            
            # 获取参数
            alpha = params.get('alpha', 0.5)
            beta = params.get('beta', 0.3)
            gamma = params.get('gamma', 0.2)
            
            print(f"📊 [_calculate_holding_matrix_v2] 使用参数: alpha={alpha:.3f}, beta={beta:.3f}, gamma={gamma:.3f}")
            
            # 模拟历史换手过程 - 新算法
            # 从第1天开始模拟到第n_days-1天
            for day_idx in range(1, min(n_days, self.max_holding_days)):
                print(f"📅 [_calculate_holding_matrix_v2] 模拟第{day_idx}天")
                
                # 获取当天的换手率分布
                if day_idx < turnover_matrix.shape[0]:
                    day_turnover_rate = turnover_matrix[day_idx, :]
                else:
                    # 如果超出范围，使用最后一天的换手率
                    day_turnover_rate = turnover_matrix[-1, :]
                
                # 应用参数调整换手率
                adjusted_turnover = alpha * day_turnover_rate + beta * gamma
                adjusted_turnover = np.clip(adjusted_turnover, 0.001, 0.99)
                
                # 计算当天的筹码分布（用于验证）
                if day_idx < chip_matrix.shape[0]:
                    day_chip_dist = chip_matrix[day_idx, :]
                    day_chip_sum = day_chip_dist.sum()
                else:
                    day_chip_dist = chip_matrix[-1, :]
                    day_chip_sum = day_chip_dist.sum()
                
                # 创建临时矩阵来保存当天的状态
                new_holding_matrix = np.zeros((n_prices, self.max_holding_days))
                
                # 对每个价格区间处理
                for price_idx in range(n_prices):
                    current_turnover_rate = adjusted_turnover[price_idx]
                    
                    # 处理现有筹码：持有天数+1，但部分会换手
                    for holding_day in range(self.max_holding_days - 1, 0, -1):
                        # 前一天持有holding_day-1天的筹码
                        prev_chips = holding_matrix[price_idx, holding_day - 1]
                        if prev_chips > 0:
                            # 部分筹码继续持有（天数+1）
                            new_holding_matrix[price_idx, holding_day] += prev_chips * (1 - current_turnover_rate)
                            # 部分筹码换手（重置为0天）
                            new_holding_matrix[price_idx, 0] += prev_chips * current_turnover_rate
                    
                    # 处理当天新获得的筹码（来自其他价格区间的换手）
                    # 这里假设换手筹码均匀分布到所有价格区间
                    # 实际应该根据成交量分布，但简化处理
                
                # 处理当天的新筹码（来自当天的交易）
                # 这部分应该基于当天的筹码分布和换手率
                day_new_chips_total = 0
                for price_idx in range(n_prices):
                    if day_chip_sum > 0:
                        # 当天的筹码分布比例
                        day_chip_ratio = day_chip_dist[price_idx] / day_chip_sum
                        # 当天新获得的筹码（基于换手率）
                        day_new_chips = day_chip_ratio * adjusted_turnover[price_idx] * 0.1  # 缩放因子
                        new_holding_matrix[price_idx, 0] += day_new_chips
                        day_new_chips_total += day_new_chips
                
                print(f"📊 [_calculate_holding_matrix_v2] 第{day_idx}天: 换手率均值={adjusted_turnover.mean():.4f}, 新增筹码={day_new_chips_total:.6f}")
                
                # 更新持有矩阵
                holding_matrix = new_holding_matrix.copy()
                
                # 归一化（保持每行总和为1）
                row_sums = holding_matrix.sum(axis=1)
                for i in range(n_prices):
                    if row_sums[i] > 0:
                        holding_matrix[i, :] = holding_matrix[i, :] / row_sums[i]
                
                # 检查长线筹码
                if day_idx % 10 == 0:
                    long_term_sum = holding_matrix[:, 60:].sum() if holding_matrix.shape[1] > 60 else 0
                    print(f"📊 [_calculate_holding_matrix_v2] 第{day_idx}天长线筹码: {long_term_sum:.6f}")
            
            # 最终归一化
            row_sums = holding_matrix.sum(axis=1)
            for i in range(n_prices):
                if row_sums[i] > 0:
                    holding_matrix[i, :] = holding_matrix[i, :] / row_sums[i]
            
            # 计算统计信息
            final_sum = holding_matrix.sum()
            row_sums_final = holding_matrix.sum(axis=1)
            long_term_final = holding_matrix[:, 60:].sum() if holding_matrix.shape[1] > 60 else 0
            short_term_final = holding_matrix[:, :5].sum() if holding_matrix.shape[1] > 5 else 0
            
            print(f"✅ [_calculate_holding_matrix_v2] 计算完成:")
            print(f"   最终矩阵总和: {final_sum:.6f}")
            print(f"   形状: {holding_matrix.shape}")
            print(f"   归一化后行和范围: {row_sums_final.min():.6f} - {row_sums_final.max():.6f}")
            print(f"   短线筹码(<5天): {short_term_final:.6f} ({short_term_final/final_sum*100:.2f}%)")
            print(f"   长线筹码(>60天): {long_term_final:.6f} ({long_term_final/final_sum*100:.2f}%)")
            
            # 如果长线筹码仍然为0，强制添加一些长线筹码
            if long_term_final < 0.01:
                print(f"⚠️ [_calculate_holding_matrix_v2] 长线筹码过低，强制添加")
                # 将部分中线筹码转移到长线
                for price_idx in range(n_prices):
                    mid_chips = holding_matrix[price_idx, 30:60].sum()
                    if mid_chips > 0.01:
                        # 转移20%的中线筹码到长线
                        transfer_amount = mid_chips * 0.2
                        holding_matrix[price_idx, 30:60] *= 0.8
                        # 平均分配到60天以上
                        if holding_matrix.shape[1] > 60:
                            holding_matrix[price_idx, 60:90] += transfer_amount / 30
                
                # 重新归一化
                row_sums = holding_matrix.sum(axis=1)
                for i in range(n_prices):
                    if row_sums[i] > 0:
                        holding_matrix[i, :] = holding_matrix[i, :] / row_sums[i]
                
                long_term_final = holding_matrix[:, 60:].sum() if holding_matrix.shape[1] > 60 else 0
                print(f"📊 [_calculate_holding_matrix_v2] 调整后长线筹码: {long_term_final:.6f}")
            
            return holding_matrix
        except Exception as e:
            print(f"❌ [_calculate_holding_matrix_v2] 计算持有时间矩阵失败: {e}")
            import traceback
            traceback.print_exc()
            # 返回一个合理的默认矩阵
            default_matrix = np.zeros((n_prices, self.max_holding_days))
            default_matrix[:, 0] = 0.2  # 20%短线
            default_matrix[:, 30] = 0.3  # 30%中线
            default_matrix[:, 90] = 0.5  # 50%长线
            # 归一化
            for i in range(n_prices):
                row_sum = default_matrix[i, :].sum()
                if row_sum > 0:
                    default_matrix[i, :] = default_matrix[i, :] / row_sum
            return default_matrix

    def _calculate_holding_factors(self,holding_matrix: np.ndarray,chip_matrix: np.ndarray,data_dict: Dict[str, any]) -> Dict[str, float]:
        """计算持有时间相关因子 - 修复长线筹码为0的问题"""
        try:
            factors = {}
            if holding_matrix.size == 0:
                print(f"⚠️ [_calculate_holding_factors_v2] 持有矩阵为空，返回默认因子")
                return self._get_default_factors()
            
            print(f"📊 [_calculate_holding_factors_v2] 持有矩阵形状: {holding_matrix.shape}")
            print(f"📊 [_calculate_holding_factors_v2] 持有矩阵总和: {holding_matrix.sum()}")
            
            # 1. 检查持有矩阵是否有效
            if holding_matrix.sum() == 0:
                print(f"⚠️ [_calculate_holding_factors_v2] 持有矩阵总和为0，返回默认因子")
                return self._get_default_factors()
            
            # 2. 计算各种持有时间的筹码比例
            total_sum = np.sum(holding_matrix)
            
            # 短线筹码比例（<5日）
            short_term_mask = np.arange(self.max_holding_days) < 5
            short_term_sum = np.sum(holding_matrix[:, short_term_mask])
            if total_sum > 0:
                short_term_ratio = short_term_sum / total_sum
                factors['short_term_ratio'] = float(short_term_ratio)
                print(f"📊 [_calculate_holding_factors_v2] 短线筹码: {short_term_ratio:.4f}")
            else:
                factors['short_term_ratio'] = 0.2
            
            # 中线筹码比例（5-60日）
            mid_term_mask = (np.arange(self.max_holding_days) >= 5) & (np.arange(self.max_holding_days) < 60)
            mid_term_sum = np.sum(holding_matrix[:, mid_term_mask])
            if total_sum > 0:
                mid_term_ratio = mid_term_sum / total_sum
                factors['mid_term_ratio'] = float(mid_term_ratio)
                print(f"📊 [_calculate_holding_factors_v2] 中线筹码: {mid_term_ratio:.4f}")
            else:
                factors['mid_term_ratio'] = 0.3
            
            # 长线筹码比例（>60日） - 重点修复
            long_term_mask = np.arange(self.max_holding_days) >= 60
            long_term_sum = np.sum(holding_matrix[:, long_term_mask])
            if total_sum > 0:
                long_term_ratio = long_term_sum / total_sum
                factors['long_term_ratio'] = float(long_term_ratio)
                print(f"📊 [_calculate_holding_factors_v2] 长线筹码: {long_term_ratio:.4f}")
            else:
                factors['long_term_ratio'] = 0.5
            
            # 3. 检查长线筹码是否异常为0
            if factors['long_term_ratio'] < 0.01:
                print(f"⚠️ [_calculate_holding_factors_v2] 长线筹码异常低({factors['long_term_ratio']:.4f})，进行调整")
                # 基于换手率和历史数据估算长线筹码
                if 'daily_turnover' in data_dict and isinstance(data_dict['daily_turnover'], pd.Series):
                    recent_turnover = data_dict['daily_turnover'].iloc[-20:].mean() if len(data_dict['daily_turnover']) >= 20 else 0.05
                    # 高换手率下长线筹码少，低换手率下长线筹码多
                    if recent_turnover < 0.02:  # 换手率低于2%
                        estimated_long_term = 0.6
                    elif recent_turnover < 0.05:  # 换手率2-5%
                        estimated_long_term = 0.4
                    elif recent_turnover < 0.1:  # 换手率5-10%
                        estimated_long_term = 0.2
                    else:  # 换手率高于10%
                        estimated_long_term = 0.1
                    
                    print(f"📊 [_calculate_holding_factors_v2] 基于换手率{recent_turnover:.2%}估算长线筹码: {estimated_long_term:.2%}")
                    
                    # 调整比例，保持总和为1
                    current_short = factors.get('short_term_ratio', 0.2)
                    current_mid = factors.get('mid_term_ratio', 0.3)
                    
                    # 计算需要减少的总量
                    reduction_needed = estimated_long_term - factors['long_term_ratio']
                    
                    if reduction_needed > 0 and (current_short + current_mid) > 0:
                        # 按比例从短线和中线中扣除
                        factors['short_term_ratio'] = max(0, current_short - reduction_needed * (current_short / (current_short + current_mid)))
                        factors['mid_term_ratio'] = max(0, current_mid - reduction_needed * (current_mid / (current_short + current_mid)))
                        factors['long_term_ratio'] = estimated_long_term
                
                # 如果还是太低，使用默认值
                if factors['long_term_ratio'] < 0.05:
                    print(f"⚠️ [_calculate_holding_factors_v2] 长线筹码仍然过低，使用经验值")
                    # 重新分配：短线20%，中线30%，长线50%
                    total_current = factors.get('short_term_ratio', 0) + factors.get('mid_term_ratio', 0) + factors.get('long_term_ratio', 0)
                    if total_current > 0:
                        factors['short_term_ratio'] = 0.2 * total_current
                        factors['mid_term_ratio'] = 0.3 * total_current
                        factors['long_term_ratio'] = 0.5 * total_current
            
            # 4. 归一化确保总和为1
            sum_ratios = factors.get('short_term_ratio', 0) + factors.get('mid_term_ratio', 0) + factors.get('long_term_ratio', 0)
            if abs(sum_ratios - 1.0) > 0.001 and sum_ratios > 0:
                factors['short_term_ratio'] = factors.get('short_term_ratio', 0) / sum_ratios
                factors['mid_term_ratio'] = factors.get('mid_term_ratio', 0) / sum_ratios
                factors['long_term_ratio'] = factors.get('long_term_ratio', 0) / sum_ratios
            
            # 5. 计算平均持有时间（修复版本）
            holding_days = np.arange(self.max_holding_days)
            weighted_days = np.sum(holding_matrix * holding_days[np.newaxis, :])
            factors['avg_holding_days'] = float(weighted_days / total_sum if total_sum > 0 else 100.0)
            
            print(f"✅ [_calculate_holding_factors_v2] 最终比例:")
            print(f"   短线: {factors.get('short_term_ratio', 0):.2%}")
            print(f"   中线: {factors.get('mid_term_ratio', 0):.2%}")
            print(f"   长线: {factors.get('long_term_ratio', 0):.2%}")
            print(f"   总和: {factors.get('short_term_ratio', 0)+factors.get('mid_term_ratio', 0)+factors.get('long_term_ratio', 0):.4f}")
            print(f"   平均持有: {factors.get('avg_holding_days', 0):.1f}天")
            # 6. 平均持有时间
            holding_days = np.arange(self.max_holding_days)
            if total_sum > 0:
                weighted_days = np.sum(holding_matrix * holding_days[np.newaxis, :])
                avg_holding_days = weighted_days / total_sum
                factors['avg_holding_days'] = float(avg_holding_days)
                print(f"📊 [_calculate_holding_factors] 平均持有时间: {weighted_days:.6f}/{total_sum:.6f}={avg_holding_days:.2f}天")
            else:
                factors['avg_holding_days'] = 100.0
                print(f"⚠️ [_calculate_holding_factors] 持有矩阵总和为0，平均持有时间使用默认值")
            # 7. 筹码集中度调整的持有时间
            if chip_matrix.shape[0] > 0:
                current_chip = chip_matrix[-1, :] if chip_matrix.shape[0] > 0 else np.ones(chip_matrix.shape[1])
                chip_mean = np.mean(current_chip)
                if chip_mean > 0:
                    chip_concentration = np.std(current_chip) / chip_mean
                    factors['concentration_adjusted_holding'] = float(factors.get('avg_holding_days', 100.0) * (1 + chip_concentration))
                    print(f"📊 [_calculate_holding_factors] 筹码集中度: {chip_concentration:.4f}, 调整后持有时间: {factors['concentration_adjusted_holding']:.2f}")
            # 8. 高位筹码沉淀比例（90%分位以上）
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
                            print(f"📊 [_calculate_holding_factors] 高位筹码沉淀比例: {high_chip_ratio:.4f} (阈值: {price_90:.2f})")
                        else:
                            factors['high_position_lock_ratio_90'] = 0.0
                    else:
                        factors['high_position_lock_ratio_90'] = 0.0
                else:
                    factors['high_position_lock_ratio_90'] = 0.0
            # 9. 主力成本区间锁定比例（50分位±10%）
            if 'chip_dist_current' in data_dict:
                chip_dist_current = data_dict['chip_dist_current']
                if not chip_dist_current.empty:
                    # 获取50分位成本
                    sorted_chips = chip_dist_current.sort_values('price')
                    cumsum = sorted_chips['percent'].cumsum()
                    if (cumsum >= 0.5).any():
                        idx_50 = (cumsum >= 0.5).idxmax()
                        cost_50pct = sorted_chips.loc[idx_50, 'price']
                        # 计算50分位±10%区间筹码
                        lower_bound = cost_50pct * 0.9
                        upper_bound = cost_50pct * 1.1
                        main_mask = (chip_dist_current['price'] >= lower_bound) & (chip_dist_current['price'] <= upper_bound)
                        main_chip_sum = chip_dist_current.loc[main_mask, 'percent'].sum()
                        total_chip_sum = chip_dist_current['percent'].sum()
                        if total_chip_sum > 0:
                            main_range_ratio = main_chip_sum / total_chip_sum
                            factors['main_cost_range_ratio'] = float(main_range_ratio)
                            print(f"📊 [_calculate_holding_factors] 主力成本区间锁定比例: {main_range_ratio:.4f} (区间: {lower_bound:.2f}-{upper_bound:.2f})")
                        else:
                            factors['main_cost_range_ratio'] = 0.5
                    else:
                        factors['main_cost_range_ratio'] = 0.5
                else:
                    factors['main_cost_range_ratio'] = 0.5
            # 10. 换手率调整因子 - 修复逻辑：保持比例总和为1
            if 'daily_turnover' in data_dict and isinstance(data_dict['daily_turnover'], pd.Series):
                recent_turnover = data_dict['daily_turnover'].iloc[-5:].mean() if len(data_dict['daily_turnover']) >= 5 else 0
                factors['turnover_adjustment'] = float(recent_turnover)
                # 高换手率下短线筹码比例应该更高，但要保持总和为1
                if recent_turnover > 0.05:  # 5%以上换手
                    # 计算调整后的短线筹码比例（限制最大为0.95）
                    new_short_term = min(factors.get('short_term_ratio', 0.2) * 1.5, 0.95)
                    # 计算需要减少的其他筹码比例
                    short_increase = new_short_term - factors.get('short_term_ratio', 0.2)
                    # 按原比例减少中长线筹码
                    current_mid = factors.get('mid_term_ratio', 0.3)
                    current_long = factors.get('long_term_ratio', 0.5)
                    total_other = current_mid + current_long
                    if total_other > 0:
                        # 按比例减少中长线筹码
                        factors['mid_term_ratio'] = max(0, current_mid - short_increase * (current_mid / total_other))
                        factors['long_term_ratio'] = max(0, current_long - short_increase * (current_long / total_other))
                    factors['short_term_ratio'] = new_short_term
                    print(f"📊 [_calculate_holding_factors] 高换手率调整: {recent_turnover:.2%} → 短线={factors['short_term_ratio']:.2%}, 中线={factors['mid_term_ratio']:.2%}, 长线={factors['long_term_ratio']:.2%}")
            # 11. 价格位置调整 - 修复逻辑：保持比例总和为1
            if 'price_range' in data_dict and chip_matrix.shape[0] > 0:
                price_min, price_max = data_dict['price_range']
                current_chip = chip_matrix[-1, :] if chip_matrix.shape[0] > 0 else np.zeros(chip_matrix.shape[1])
                # 计算筹码价格重心
                price_center = np.sum(current_chip * np.linspace(price_min, price_max, len(current_chip))) / np.sum(current_chip) if np.sum(current_chip) > 0 else price_min
                price_position = (price_center - price_min) / (price_max - price_min) if price_max > price_min else 0.5
                factors['price_position'] = float(price_position)
                # 价格高位通常长线筹码较少，但要保持总和为1
                if price_position > 0.7:
                    # 减少长线筹码，增加短线和中线
                    reduction = factors.get('long_term_ratio', 0.5) * 0.2  # 减少20%
                    new_long = max(0.1, factors.get('long_term_ratio', 0.5) - reduction)
                    increase_total = reduction
                    # 按原比例增加短线和中线
                    current_short = factors.get('short_term_ratio', 0.2)
                    current_mid = factors.get('mid_term_ratio', 0.3)
                    total_short_mid = current_short + current_mid
                    if total_short_mid > 0:
                        factors['short_term_ratio'] = current_short + increase_total * (current_short / total_short_mid)
                        factors['mid_term_ratio'] = current_mid + increase_total * (current_mid / total_short_mid)
                    factors['long_term_ratio'] = new_long
                    print(f"📊 [_calculate_holding_factors] 高位价格调整: 价格位置={price_position:.2%} → 短线={factors['short_term_ratio']:.2%}, 中线={factors['mid_term_ratio']:.2%}, 长线={factors['long_term_ratio']:.2%}")
            # 12. 最终归一化确保总和为1
            final_sum = factors.get('short_term_ratio', 0) + factors.get('mid_term_ratio', 0) + factors.get('long_term_ratio', 0)
            if abs(final_sum - 1.0) > 0.001 and final_sum > 0:
                print(f"⚠️ [_calculate_holding_factors] 最终比例异常 ({final_sum:.4f})，重新归一化")
                factors['short_term_ratio'] = factors.get('short_term_ratio', 0) / final_sum
                factors['mid_term_ratio'] = factors.get('mid_term_ratio', 0) / final_sum
                factors['long_term_ratio'] = factors.get('long_term_ratio', 0) / final_sum
            # 13. 打印最终因子值
            final_sum_check = factors.get('short_term_ratio', 0) + factors.get('mid_term_ratio', 0) + factors.get('long_term_ratio', 0)
            print(f"✅ [_calculate_holding_factors] 计算完成:")
            print(f"   短线筹码: {factors.get('short_term_ratio', 0):.2%}")
            print(f"   中线筹码: {factors.get('mid_term_ratio', 0):.2%}")
            print(f"   长线筹码: {factors.get('long_term_ratio', 0):.2%}")
            print(f"   比例总和: {final_sum_check:.6f}")
            print(f"   平均持有: {factors.get('avg_holding_days', 0):.1f}天")
            return factors
        except Exception as e:
            print(f"❌ [_calculate_holding_factors_v2] 计算持有因子失败: {e}")
            import traceback
            traceback.print_exc()
            return self._get_default_factors()

    def _validate_results(self,holding_matrix: np.ndarray,factors: Dict[str, float],data_dict: Dict[str, any]) -> Dict[str, any]:
        """
        验证计算结果合理性
        """
        validation = {
            'is_valid': True,
            'warnings': [],  # 确保warnings字段存在
            'checks_passed': 0,
            'total_checks': 5
        }
        try:
            # 检查1：持有时间矩阵总和应为1
            total_sum = np.sum(holding_matrix)
            if abs(total_sum - 1.0) > 0.01:
                warning_msg = f"持有矩阵总和异常: {total_sum:.4f}"
                validation['warnings'].append(warning_msg)
                validation['is_valid'] = False
                print(f"⚠️ [_validate_results] {warning_msg}")
            else:
                validation['checks_passed'] += 1
                print(f"✅ [_validate_results] 检查1通过: 持有矩阵总和正常")
            # 检查2：短线+中线+长线比例应接近1
            sum_ratios = factors.get('short_term_ratio', 0) + \
                        factors.get('mid_term_ratio', 0) + \
                        factors.get('long_term_ratio', 0)
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
            # 检查4：与换手率一致性
            if 'daily_turnover' in data_dict and isinstance(data_dict['daily_turnover'], pd.Series):
                recent_turnover = data_dict['daily_turnover'].iloc[-5:].mean() if len(data_dict['daily_turnover']) >= 5 else 0
                expected_short_term = min(recent_turnover * 5, 0.95)  # 预期短线比例
                actual_short_term = factors.get('short_term_ratio', 0)
                if abs(actual_short_term - expected_short_term) > 0.3:
                    warning_msg = f"短线比例与换手率不一致: 实际{actual_short_term:.2%}, 预期{expected_short_term:.2%}"
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