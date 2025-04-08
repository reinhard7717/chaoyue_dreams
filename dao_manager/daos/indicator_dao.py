import logging
from typing import Any, List, Optional, Union, Dict, Type
import pandas as pd
import numpy as np
from decimal import Decimal, InvalidOperation
from asgiref.sync import sync_to_async
from django.db import models
from django.db import transaction
from django.utils import timezone

# 假设你有 StockBasicDAO 和 Cache 相关类
from dao_manager.base_dao import BaseDAO

from core.constants import TimeLevel, FIB_PERIODS, FINTA_OHLCV_MAP


logger = logging.getLogger("services")

class IndicatorDAO(BaseDAO):
    """
    指标数据访问对象，负责指标数据的读取和存储
    """
    def __init__(self):
        from dao_manager.daos.stock_basic_dao import StockBasicDAO
        from utils.cache_get import StockIndicatorsCacheGet
        from utils.cache_set import StockIndicatorsCacheSet
        # 依赖注入基础DAO和缓存工具
        self.stock_basic_dao = StockBasicDAO()
        self.cache_get = StockIndicatorsCacheGet() # 假设 CacheGet 实例已配置好
        self.cache_set = StockIndicatorsCacheSet() # 假设 CacheSet 实例已配置好

    @staticmethod
    def _safe_decimal(value, default=None) -> Optional[Decimal]:
        """安全地将值转换为 Decimal"""
        if value is None or value == '':
            return default
        try:
            # 如果已经是 Decimal，直接返回
            if isinstance(value, Decimal):
                return value
            # 从字符串或浮点数转换，推荐先转字符串保证精度
            return Decimal(str(value))
        except (InvalidOperation, ValueError, TypeError):
            logger.warning(f"无法将值 '{value}' (类型: {type(value)}) 转换为 Decimal", exc_info=False)
            return default

    @staticmethod
    def _safe_int(value, default=None) -> Optional[int]:
        """安全地将值转换为 Integer"""
        if value is None or value == '':
            return default
        try:
             # 如果已经是 Int，直接返回
            if isinstance(value, int):
                return value
            # 处理可能带小数点的字符串或浮点数
            return int(float(value))
        except (ValueError, TypeError):
            logger.warning(f"无法将值 '{value}' (类型: {type(value)}) 转换为 Integer", exc_info=False)
            return default

    @staticmethod
    def _safe_datetime(value, default=None) -> Optional[timezone.datetime]:
        """安全地将值转换为 timezone-aware datetime"""
        if value is None:
            return default
        try:
            # pandas.to_datetime 是一个强大的解析器
            dt = pd.to_datetime(value)
            # 确保时区感知，如果已经是，则不变；如果是 naive，则设置为默认时区
            if timezone.is_naive(dt):
                return timezone.make_aware(dt, timezone.get_default_timezone())
            return dt
        except (ValueError, TypeError):
             logger.warning(f"无法将值 '{value}' (类型: {type(value)}) 转换为 Datetime", exc_info=False)
             return default

    async def get_history_time_trades_by_limit(self, stock_code: str, time_level: Union[TimeLevel, str], limit: int = 1000) -> Optional[List['StockTimeTrade']]:
        """
        获取指定股票和时间级别的最新分时成交数据。
        优先从 Redis 缓存获取（处理字典格式），失败则从数据库获取。
        返回按时间升序排列的 StockTimeTrade 模型实例列表。
        """
        from stock_models.stock_basic import StockTimeTrade
        stock = await self.stock_basic_dao.get_stock_by_code(stock_code)
        if not stock:
            logger.warning(f"无法找到股票信息: {stock_code}")
            return None

        time_level_str = time_level.value if isinstance(time_level, TimeLevel) else str(time_level)

        # 1. 尝试从Redis获取数据
        try:
            # 假设 cache_get.history_time_trade_by_limit 返回 List[Dict] 或 None
            cache_data = await self.cache_get.history_time_trade_by_limit(stock_code, time_level_str, limit)
        except Exception as e:
            logger.error(f"从 Redis 获取缓存数据时出错 for {stock_code} {time_level_str}: {e}", exc_info=True)
            cache_data = None # 出错则认为缓存未命中

        if cache_data:
            logger.debug(f"从缓存获取到 {stock_code} {time_level_str} 历史数据 (limit={limit})，共 {len(cache_data)} 条，进行转换...")
            model_instances = []
            conversion_errors = 0
            for item_dict in cache_data:
                try:
                    # 将字典转换为 StockTimeTrade 模型实例
                    trade_time = self._safe_datetime(item_dict.get('trade_time'))
                    if not trade_time: # 如果时间无效，跳过此条记录
                        logger.warning(f"缓存数据中发现无效的 trade_time: {item_dict.get('trade_time')}")
                        conversion_errors += 1
                        continue

                    instance = StockTimeTrade(
                        stock=stock, # 使用前面获取的 StockInfo 实例
                        time_level=time_level_str,
                        trade_time=trade_time,
                        open_price=self._safe_decimal(item_dict.get('open_price')),
                        high_price=self._safe_decimal(item_dict.get('high_price')),
                        low_price=self._safe_decimal(item_dict.get('low_price')),
                        close_price=self._safe_decimal(item_dict.get('close_price')),
                        volume=self._safe_int(item_dict.get('volume')),
                        turnover=self._safe_decimal(item_dict.get('turnover')),
                        amplitude=self._safe_decimal(item_dict.get('amplitude')),
                        turnover_rate=self._safe_decimal(item_dict.get('turnover_rate')),
                        price_change_percent=self._safe_decimal(item_dict.get('price_change_percent')),
                        price_change_amount=self._safe_decimal(item_dict.get('price_change_amount')),
                    )
                    model_instances.append(instance)
                except Exception as e_conv:
                    conversion_errors += 1
                    logger.error(f"转换缓存字典为 StockTimeTrade 实例时出错: {e_conv}. Dict: {item_dict}", exc_info=False) # 避免过多日志

            if conversion_errors > 0:
                 logger.warning(f"转换缓存数据时遇到 {conversion_errors} 个错误 for {stock_code} {time_level_str}")

            if not model_instances:
                logger.warning(f"缓存数据转换后为空列表 for {stock_code} {time_level_str}")
                # 这里可以选择返回 None 或继续尝试从数据库获取
                # 为了与原逻辑一致（缓存命中直接返回），如果转换后为空，我们返回 None
                return None

            # 重要：确保返回的数据是按时间升序排列的
            # 假设缓存存储的是最新的 N 条（时间降序），需要反转
            # 如果缓存本身就是升序存储，则不需要 reverse()
            model_instances.sort(key=lambda x: x.trade_time) # 使用 sort 保证升序

            logger.debug(f"成功从缓存转换 {len(model_instances)} 条 StockTimeTrade 实例 for {stock_code} {time_level_str}")
            return model_instances

        # 2. 如果缓存未命中或处理失败，从数据库获取数据
        logger.debug(f"缓存未命中或处理失败 for {stock_code} {time_level_str}，从数据库获取...")
        try:
            # 获取最新的 N 条 (降序)
            data_qs = StockTimeTrade.objects.filter(
                stock=stock,
                time_level=time_level_str
            ).order_by('-trade_time')[:limit]

            # 异步执行查询并转换为列表
            data_list = await sync_to_async(list)(data_qs)

            if not data_list:
                logger.warning(f"数据库中未找到 {stock_code} {time_level_str} 的历史数据")
                return None

            logger.debug(f"从数据库获取到 {stock_code} {time_level_str} {len(data_list)} 条历史数据")

            # 计算指标需要升序数据，反转列表
            data_list.reverse()

            # (可选) 将从数据库获取的数据存入缓存
            # 注意：这里需要决定是缓存模型实例还是字典。
            # 如果要缓存字典以匹配读取逻辑，需要在这里进行转换。
            try:
                # 转换为字典列表以便缓存
                data_to_cache = [
                    {
                        'stock': trade.stock.stock_code, # 存代码而非实例
                        'time_level': trade.time_level,
                        # 将 datetime 转换为 ISO 格式字符串以便 JSON 序列化
                        'trade_time': trade.trade_time.isoformat() if trade.trade_time else None,
                        # 将 Decimal 转换为字符串
                        'open_price': str(trade.open_price) if trade.open_price is not None else None,
                        'high_price': str(trade.high_price) if trade.high_price is not None else None,
                        'low_price': str(trade.low_price) if trade.low_price is not None else None,
                        'close_price': str(trade.close_price) if trade.close_price is not None else None,
                        'volume': trade.volume, # int 可以直接序列化
                        'turnover': str(trade.turnover) if trade.turnover is not None else None,
                        'amplitude': str(trade.amplitude) if trade.amplitude is not None else None,
                        'turnover_rate': str(trade.turnover_rate) if trade.turnover_rate is not None else None,
                        'price_change_percent': str(trade.price_change_percent) if trade.price_change_percent is not None else None,
                        'price_change_amount': str(trade.price_change_amount) if trade.price_change_amount is not None else None,
                    }
                    # 注意：缓存的数据应该是升序还是降序？取决于你的缓存策略
                    # 如果 history_time_trade_by_limit 期望缓存的是最新的 N 条（降序）
                    # 那么应该在 data_list.reverse() 之前转换并缓存原始的 data_list
                    # 这里假设缓存升序数据（与函数最终返回一致）
                    for trade in data_list # 使用已反转（升序）的列表
                ]
                await self.cache_set.history_time_trade_by_limit(stock_code, time_level_str, limit, data_to_cache)
                logger.debug(f"已将从数据库获取的数据（字典格式）存入缓存 for {stock_code} {time_level_str}")
            except Exception as e_cache_set:
                 logger.error(f"将数据存入缓存失败 for {stock_code} {time_level_str}: {e_cache_set}", exc_info=True)


            return data_list # 返回模型实例列表

        except Exception as e_db:
            logger.error(f"从数据库获取股票[{stock_code}] {time_level_str} 级别分时成交数据失败: {str(e_db)}", exc_info=True)
            return None

    # ... (IndicatorDAO 的其他方法，如 get_history_ohlcv_df, _save_indicator_data_generic 等保持不变)
    async def get_history_ohlcv_df(self, stock_code: str, time_level: Union[TimeLevel, str], limit: int = 1000) -> Optional[pd.DataFrame]:
        """
        获取历史数据并转换为 finta 需要的 DataFrame 格式。
        返回的 DataFrame 按时间升序排列。
        (此方法逻辑不变，因为它依赖于 get_history_time_trades_by_limit 返回 List[StockTimeTrade])
        """
        history_trades = await self.get_history_time_trades_by_limit(stock_code, time_level, limit)
        if not history_trades:
            logger.warning(f"get_history_time_trades_by_limit 未返回数据 for {stock_code} {time_level}")
            return None

        try:
            # 将模型列表转换为 DataFrame
            data = [
                {
                    # 使用 getattr 安全访问，并处理 None
                    'trade_time': getattr(trade, 'trade_time', None),
                    'open_price': float(getattr(trade, 'open_price', None)) if getattr(trade, 'open_price', None) is not None else np.nan,
                    'high_price': float(getattr(trade, 'high_price', None)) if getattr(trade, 'high_price', None) is not None else np.nan,
                    'low_price': float(getattr(trade, 'low_price', None)) if getattr(trade, 'low_price', None) is not None else np.nan,
                    'close_price': float(getattr(trade, 'close_price', None)) if getattr(trade, 'close_price', None) is not None else np.nan,
                    'volume': int(getattr(trade, 'volume', 0)) if getattr(trade, 'volume', None) is not None else 0, # finta 需要整数或浮点数
                    'turnover': float(getattr(trade, 'turnover', None)) if getattr(trade, 'turnover', None) is not None else np.nan,
                }
                for trade in history_trades if getattr(trade, 'trade_time', None) is not None # 过滤掉没有时间的记录
            ]
            if not data:
                logger.warning(f"从 StockTimeTrade 实例转换的数据列表为空: {stock_code} {time_level}")
                return None

            df = pd.DataFrame(data)

            if df.empty:
                logger.warning(f"转换后的 DataFrame 为空: {stock_code} {time_level}")
                return None

            # 重命名列以匹配 finta 要求
            df.rename(columns=FINTA_OHLCV_MAP, inplace=True)

            # 将 trade_time 设置为索引
            df['trade_time'] = pd.to_datetime(df['trade_time'], utc=True)
            df.set_index('trade_time', inplace=True)

            # --- 新增：去除重复索引 ---
            initial_len = len(df)
            if df.index.has_duplicates:
                logger.warning(f"发现重复的时间戳索引 for {stock_code} {time_level}，将进行去重处理 (保留最后一个)")
                # 保留每个重复时间戳的最后一条记录
                df = df[~df.index.duplicated(keep='last')]
                logger.info(f"索引去重完成 for {stock_code} {time_level}，记录数从 {initial_len} 变为 {len(df)}")
            # --- 去重结束 ---

            # 确保数据按时间升序排列 (虽然 get_history_time_trades_by_limit 已保证升序)
            df.sort_index(ascending=True, inplace=True)

            # 验证必要的列是否存在
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_cols):
                logger.error(f"DataFrame 缺少必要列: {stock_code} {time_level}. 需要: {required_cols}, 实际: {df.columns.tolist()}")
                return None

            # 移除完全是 NaN 的行 (如果需要)
            # df.dropna(subset=required_cols, how='all', inplace=True) # 谨慎使用，可能移除计算指标需要的数据点

            # 检查是否有足够的非 NaN 数据行
            if df[required_cols].isnull().all(axis=1).sum() == len(df):
                 logger.warning(f"处理后 DataFrame 只包含 NaN 值: {stock_code} {time_level}")
                 return None # 如果全是 NaN，返回 None

            # logger.debug(f"成功为 {stock_code} {time_level} 创建 OHLCV DataFrame，形状: {df.shape}")
            return df

        except Exception as e:
            logger.error(f"转换 {stock_code} {time_level} 历史数据为 DataFrame 失败: {str(e)}", exc_info=True)
            return None

    # --- 其他 DAO 方法 ---
    @staticmethod
    def _prepare_decimal(value) -> Optional[Decimal]:
        """将计算结果（可能是 float 或 NaN）安全地转换为 Decimal 或 None"""
        if value is None or (isinstance(value, (float, np.floating)) and (np.isnan(value) or np.isinf(value))):
            return None
        try:
            # 使用之前定义的更安全的转换函数
            return IndicatorDAO._safe_decimal(value)
            # return Decimal(str(value)).quantize(Decimal("0.0001")) # 保留4位小数，根据模型调整
        except (InvalidOperation, TypeError):
            logger.warning(f"无法将值 {value} (类型: {type(value)}) 转换为 Decimal", exc_info=False)
            return None

    @staticmethod
    def _prepare_int(value) -> Optional[int]:
        """将计算结果安全地转换为 int 或 None"""
        if value is None or (isinstance(value, (float, np.floating)) and (np.isnan(value) or np.isinf(value))):
            return None
        try:
             # 使用之前定义的更安全的转换函数
            return IndicatorDAO._safe_int(value)
            # return int(value)
        except (ValueError, TypeError):
            logger.warning(f"无法将值 {value} (类型: {type(value)}) 转换为 Integer", exc_info=False)
            return None

    # --- 修改后的通用保存方法 ---
    async def _save_indicator_data_generic(self, stock_info: 'StockInfo', time_level: Union[TimeLevel, str],
        indicator_df: pd.DataFrame, model_class: Type[models.Model],
        field_map: Dict[str, str], # DataFrame 列名 -> 模型字段名 的映射
        # unique_fields 列表现在内部定义，基于通用模式
    ):
        """
        通用的指标数据保存方法，使用 BaseDao 的 _save_all_to_db_native_upsert 进行批量处理。
        Args:
            stock_info (StockInfo): 股票信息实例.
            time_level (Union[TimeLevel, str]): 时间级别.
            indicator_df (pd.DataFrame): 包含指标数据的 DataFrame (索引必须是 trade_time).
            model_class (Type[models.Model]): 要保存到的 Django 模型类.
            field_map (Dict[str, str]): DataFrame 列名到模型字段名的映射.
        """
        from stock_models.stock_basic import StockInfo
        time_level_str = time_level.value if isinstance(time_level, TimeLevel) else str(time_level)
        model_name = model_class.__name__ # 获取模型名称用于日志记录

        # 检查输入 DataFrame 是否有效
        if indicator_df is None or indicator_df.empty:
            logger.warning(f"[{model_name}] 无指标数据可保存 for {stock_info.stock_code} {time_level_str}")
            return

        # 1. 准备 data_list (List[Dict]) 以供批量操作
        data_to_save: List[Dict[str, Any]] = []
        preparation_failed_count = 0 # 记录准备数据阶段失败的行数

        # 确保 DataFrame 索引是 DatetimeIndex
        if not isinstance(indicator_df.index, pd.DatetimeIndex):
            try:
                indicator_df.index = pd.to_datetime(indicator_df.index)
            except Exception as e_idx:
                logger.error(f"[{model_name}] 无法将索引转换为 DatetimeIndex for {stock_info.stock_code} {time_level_str}: {e_idx}", exc_info=True)
                return # 没有有效的时间索引无法继续

        # logger.debug(f"[{model_name}] 开始准备 {len(indicator_df)} 条记录的批量数据 for {stock_info.stock_code} {time_level_str}")

        # 遍历 DataFrame 的每一行来构建字典列表
        for trade_time, row in indicator_df.iterrows():
            try:
                # 确保 trade_time 是时区感知的 (aware)
                # 假设数据库和 Django 设置使用 UTC 或默认时区
                default_tz = timezone.get_default_timezone()
                if timezone.is_naive(trade_time):
                    aware_trade_time = timezone.make_aware(trade_time, default_tz)
                else:
                    # 如果已经是 aware，确保它是默认时区，避免混合时区问题
                    aware_trade_time = trade_time.astimezone(default_tz)

                # 构建基础记录字典，包含唯一标识字段
                # 注意：外键字段在字典中通常使用 "_id" 后缀，值为外键的主键
                record_data = {
                    'stock_id': stock_info.pk, # 使用 StockInfo 实例的主键
                    'time_level': time_level_str,
                    'trade_time': aware_trade_time,
                }
                has_valid_indicator_value = False # 标记此行是否有有效的指标值

                # 根据 field_map 添加指标值
                for df_col, model_field in field_map.items():
                    if df_col in row: # 检查 DataFrame 中是否存在该列
                        value = row[df_col] # 获取原始值
                        prepared_value = None # 初始化准备好的值

                        # 获取模型字段实例以判断目标类型 (可选但推荐)
                        try:
                            model_field_instance = model_class._meta.get_field(model_field)
                        except models.FieldDoesNotExist:
                            logger.warning(f"[{model_name}] 模型字段 '{model_field}' 不存在，跳过映射 for column '{df_col}'")
                            continue # 跳过这个字段

                        # 根据目标模型字段类型准备数据
                        if isinstance(model_field_instance, (models.DecimalField)):
                            prepared_value = self._prepare_decimal(value)
                        elif isinstance(model_field_instance, (models.BigIntegerField, models.IntegerField)):
                            prepared_value = self._prepare_int(value)
                        elif isinstance(model_field_instance, (models.FloatField)):
                            # 确保是有限的浮点数，NaN/inf 存为 None
                            if value is not None and isinstance(value, (float, np.floating)) and np.isfinite(value):
                                prepared_value = float(value)
                            else:
                                prepared_value = None
                        else: # 其他类型如 CharField 等
                            # 存储非 NaN 值，其他转为 None
                            if value is not None and not (isinstance(value, float) and np.isnan(value)):
                                prepared_value = value
                            else:
                                prepared_value = None

                        # 将准备好的值添加到记录字典中
                        record_data[model_field] = prepared_value
                        # 如果准备好的值不是 None，说明此行至少有一个有效指标
                        if prepared_value is not None:
                            has_valid_indicator_value = True
                    # else: 如果 DataFrame 行中缺少映射的列，则忽略

                # 只有当该行包含至少一个有效的（非 None）指标值时，才将其添加到待保存列表
                if has_valid_indicator_value:
                    data_to_save.append(record_data)
                # else: # 可以选择记录被跳过的行
                #    logger.debug(f"[{model_name}] 跳过行 at {aware_trade_time} for {stock_info.stock_code}，因为所有指标值均为 None.")

            except Exception as e_prep:
                # 捕获准备单行数据时可能发生的错误
                preparation_failed_count += 1
                logger.error(f"[{model_name}] 准备数据时出错 for {stock_info.stock_code} at {trade_time}: {e_prep}", exc_info=False) # 避免过多日志

        # 如果在准备阶段有失败的记录，记录警告
        if preparation_failed_count > 0:
             logger.warning(f"[{model_name}] 在数据准备阶段跳过或失败了 {preparation_failed_count} 条记录 for {stock_info.stock_code} {time_level_str}")

        # 如果没有准备好任何有效数据，则直接返回
        if not data_to_save:
            logger.warning(f"[{model_name}] 没有准备好任何有效数据进行批量保存 for {stock_info.stock_code} {time_level_str}")
            return

        # 2. 定义用于冲突检测的唯一字段列表
        # 这些字段组合必须在数据库模型 Meta 中定义了 unique_together 约束，
        # 或者数据库层面有对应的联合唯一索引。
        # 对于指标数据，通常是 股票、时间级别、交易时间 的组合。
        unique_fields = ['stock_id', 'time_level', 'trade_time']

        # 3. 调用继承的批量 Upsert 方法
        # logger.info(f"[{model_name}] 准备调用 _save_all_to_db_native_upsert 处理 {len(data_to_save)} 条有效记录 for {stock_info.stock_code} {time_level_str}")
        try:
            # 调用基类方法，传入模型类、准备好的数据列表和唯一字段列表
            # extra_fields 不需要，因为所有字段已包含在 data_to_save 的字典中
            result = await self._save_all_to_db_native_upsert(
                model_class=model_class,
                data_list=data_to_save,
                unique_fields=unique_fields,
                # **extra_fields 参数在这里不需要传递
            )
            # 基类方法中应包含详细的日志记录，这里可以只记录一个概要
            logger.info(f"[{model_name}] 批量保存/更新调用完成 for {stock_info.stock_code} {time_level_str}. 结果: {result}")

        except Exception as e_bulk:
            # 捕获调用基类方法本身可能抛出的意外错误（尽管基类方法应处理其内部错误）
            logger.error(f"[{model_name}] 调用 _save_all_to_db_native_upsert 时发生意外错误 for {stock_info.stock_code} {time_level_str}: {e_bulk}", exc_info=True)


    # --- 各指标具体的保存方法 (保持不变) ---
    async def save_atr_fib(self, stock_info: 'StockInfo', time_level: Union[TimeLevel, str], df: pd.DataFrame):
        from stock_models.indicator.atr import StockAtrFIB
        field_map = {f'ATR_{p}': f'atr{p}' for p in FIB_PERIODS if f'ATR_{p}' in df.columns}
        await self._save_indicator_data_generic(stock_info, time_level, df, StockAtrFIB, field_map)

    async def save_boll(self, stock_info: 'StockInfo', time_level: Union[TimeLevel, str], df: pd.DataFrame):
        from stock_models.indicator.boll import StockBOLLIndicator
        field_map = {'BB_UPPER': 'upper', 'BB_MIDDLE': 'mid', 'BB_LOWER': 'lower'}
        await self._save_indicator_data_generic(stock_info, time_level, df, StockBOLLIndicator, field_map)

    async def save_cci_fib(self, stock_info: 'StockInfo', time_level: Union[TimeLevel, str], df: pd.DataFrame):
        from stock_models.indicator.cci import StockCciFIB
        field_map = {f'{p} period CCI': f'cci{p}' for p in FIB_PERIODS if f'{p} period CCI' in df.columns}
        await self._save_indicator_data_generic(stock_info, time_level, df, StockCciFIB, field_map)

    async def save_cmf_fib(self, stock_info: 'StockInfo', time_level: Union[TimeLevel, str], df: pd.DataFrame):
        from stock_models.indicator.cmf import StockCmfFIB
        field_map = {f'{p} period CMF': f'cmf{p}' for p in FIB_PERIODS if f'{p} period CMF' in df.columns}
        await self._save_indicator_data_generic(stock_info, time_level, df, StockCmfFIB, field_map)

    async def save_dmi_fib(self, stock_info: 'StockInfo', time_level: Union[TimeLevel, str], df: pd.DataFrame):
        from stock_models.indicator.dmi import StockDmiFIB
        field_map = {}
        for p in [13, 21, 34, 55, 89, 144, 233]:
             if f'+DI_{p}' in df.columns: field_map[f'+DI_{p}'] = f'plus_di{p}'
             if f'-DI_{p}' in df.columns: field_map[f'-DI_{p}'] = f'minus_di{p}'
             if f'ADX_{p}' in df.columns: field_map[f'ADX_{p}'] = f'adx{p}'
             if f'ADXR_{p}' in df.columns: field_map[f'ADXR_{p}'] = f'adxr{p}'
        await self._save_indicator_data_generic(stock_info, time_level, df, StockDmiFIB, field_map)

    async def save_ichimoku(self, stock_info: 'StockInfo', time_level: Union[TimeLevel, str], df: pd.DataFrame):
        from stock_models.indicator.ichimoku import StockIchimoku
        field_map = {
            'TENKAN': 'tenkan_sen', 'KIJUN': 'kijun_sen', 'CHIKOU': 'chikou_span',
            'SENKOU A': 'senkou_span_a', 'SENKOU B': 'senkou_span_b'
        }
        await self._save_indicator_data_generic(stock_info, time_level, df, StockIchimoku, field_map)

    async def save_kdj_fib(self, stock_info: 'StockInfo', time_level: Union[TimeLevel, str], df: pd.DataFrame):
        from stock_models.indicator.kdj import StockKDJFIB
        field_map = {}
        for p in FIB_PERIODS:
            if f'K_{p}' in df.columns: field_map[f'K_{p}'] = f'k_{p}'
            if f'D_{p}' in df.columns: field_map[f'D_{p}'] = f'd_{p}'
            if f'J_{p}' in df.columns: field_map[f'J_{p}'] = f'j_{p}'
        await self._save_indicator_data_generic(stock_info, time_level, df, StockKDJFIB, field_map)

    async def save_ema_fib(self, stock_info: 'StockInfo', time_level: Union[TimeLevel, str], df: pd.DataFrame):
        from stock_models.indicator.ma import StockEmaFIB
        field_map = {f'EMA_{p}': f'ema{p}' for p in FIB_PERIODS if f'EMA_{p}' in df.columns}
        await self._save_indicator_data_generic(stock_info, time_level, df, StockEmaFIB, field_map)

    async def save_amount_ma_fib(self, stock_info: 'StockInfo', time_level: Union[TimeLevel, str], df: pd.DataFrame):
        from stock_models.indicator.ma import StockAmountMaFIB
        field_map = {f'AMT_MA_{p}': f'amt_ma{p}' for p in FIB_PERIODS if f'AMT_MA_{p}' in df.columns}
        await self._save_indicator_data_generic(stock_info, time_level, df, StockAmountMaFIB, field_map)

    async def save_macd_fib(self, stock_info: 'StockInfo', time_level: Union[TimeLevel, str], df: pd.DataFrame):
        from stock_models.indicator.macd import StockMACDFIB
        field_map = {'MACD': 'diff', 'SIGNAL': 'dea', 'MACD_HIST': 'macd'}
        for p in FIB_PERIODS:
            if f'EMA_{p}' in df.columns: field_map[f'EMA_{p}'] = f'ema{p}'
        await self._save_indicator_data_generic(stock_info, time_level, df, StockMACDFIB, field_map)

    async def save_mfi_fib(self, stock_info: 'StockInfo', time_level: Union[TimeLevel, str], df: pd.DataFrame):
        from stock_models.indicator.mfi import StockMfiFIB
        field_map = {f'MFI_{p}': f'mfi{p}' for p in FIB_PERIODS if f'MFI_{p}' in df.columns}
        await self._save_indicator_data_generic(stock_info, time_level, df, StockMfiFIB, field_map)

    async def save_mom_fib(self, stock_info: 'StockInfo', time_level: Union[TimeLevel, str], df: pd.DataFrame):
        from stock_models.indicator.mom import StockMomFIB
        field_map = {f'MOM_{p}': f'mom{p}' for p in FIB_PERIODS if f'MOM_{p}' in df.columns}
        await self._save_indicator_data_generic(stock_info, time_level, df, StockMomFIB, field_map)

    async def save_obv(self, stock_info: 'StockInfo', time_level: Union[TimeLevel, str], df: pd.DataFrame):
        from stock_models.indicator.obv import StockObvFIB
        field_map = {'OBV': 'obv'}
        await self._save_indicator_data_generic(stock_info, time_level, df, StockObvFIB, field_map)

    async def save_roc_fib(self, stock_info: 'StockInfo', time_level: Union[TimeLevel, str], df: pd.DataFrame):
        from stock_models.indicator.roc import StockRocFIB
        field_map = {f'ROC_{p}': f'roc{p}' for p in FIB_PERIODS if f'ROC_{p}' in df.columns}
        await self._save_indicator_data_generic(stock_info, time_level, df, StockRocFIB, field_map)

    async def save_amount_roc_fib(self, stock_info: 'StockInfo', time_level: Union[TimeLevel, str], df: pd.DataFrame):
        from stock_models.indicator.roc import StockAmountRocFIB
        field_map = {f'AROC_{p}': f'aroc{p}' for p in FIB_PERIODS if f'AROC_{p}' in df.columns}
        await self._save_indicator_data_generic(stock_info, time_level, df, StockAmountRocFIB, field_map)

    async def save_rsi_fib(self, stock_info: 'StockInfo', time_level: Union[TimeLevel, str], df: pd.DataFrame):
        from stock_models.indicator.rsi import StockRsiFIB
        field_map = {f'RSI_{p}': f'rsi{p}' for p in FIB_PERIODS if f'RSI_{p}' in df.columns}
        await self._save_indicator_data_generic(stock_info, time_level, df, StockRsiFIB, field_map)

    async def save_sar(self, stock_info: 'StockInfo', time_level: Union[TimeLevel, str], df: pd.DataFrame):
        from stock_models.indicator.sar import StockSar
        field_map = {'SAR': 'sar'}
        await self._save_indicator_data_generic(stock_info, time_level, df, StockSar, field_map)

    async def save_vroc_fib(self, stock_info: 'StockInfo', time_level: Union[TimeLevel, str], df: pd.DataFrame):
        from stock_models.indicator.vroc import StockVrocFIB
        field_map = {f'VROC_{p}': f'vroc{p}' for p in FIB_PERIODS if f'VROC_{p}' in df.columns}
        await self._save_indicator_data_generic(stock_info, time_level, df, StockVrocFIB, field_map)

    async def save_vwap(self, stock_info: 'StockInfo', time_level: Union[TimeLevel, str], df: pd.DataFrame):
        from stock_models.indicator.vwap import StockVwap
        field_map = {'VWAP': 'vwap'}
        await self._save_indicator_data_generic(stock_info, time_level, df, StockVwap, field_map)

    async def save_wr_fib(self, stock_info: 'StockInfo', time_level: Union[TimeLevel, str], df: pd.DataFrame):
        from stock_models.indicator.wr import StockWrFIB
        field_map = {f'WR_{p}': f'wr{p}' for p in FIB_PERIODS if f'WR_{p}' in df.columns}
        await self._save_indicator_data_generic(stock_info, time_level, df, StockWrFIB, field_map)



