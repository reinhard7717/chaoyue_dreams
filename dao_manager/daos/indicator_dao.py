import asyncio
import logging
from typing import Any, List, Optional, Set, Union, Dict, Type
import pandas as pd
from django.db.models import Max # <--- 确保导入 Max
import numpy as np
from decimal import Decimal, InvalidOperation
from asgiref.sync import sync_to_async
from django.db import models
from django.db import transaction
from django.utils import timezone
from django.core.exceptions import FieldDoesNotExist
from dao_manager.base_dao import BaseDAO
from core.constants import TimeLevel, FIB_PERIODS, FINTA_OHLCV_MAP
from stock_models.stock_basic import StockTimeTrade
from utils.cache_get import StockIndicatorsCacheGet
from utils.cache_manager import CacheManager
from utils.cache_set import StockIndicatorsCacheSet


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
        self.cache_manager = None
        self.cache_get = None
        self.cache_set = None

    async def initialize_cache_objects(self):
        self.cache_manager = CacheManager()  # 先实例化
        await self.cache_manager.initialize()  # 然后 await 其异步初始化方法，如果存在

        self.cache_set = StockIndicatorsCacheSet()  # 先实例化
        await self.cache_set.initialize()  # 添加异步初始化方法，如果需要

        self.cache_get = StockIndicatorsCacheGet()  # 先实例化
        await self.cache_get.initialize()  # 添加异步初始化方法，如果需要


    async def get_history_time_trades_by_limit(self, stock_code: str, time_level: Union[TimeLevel, str], limit: int = 1000) -> Optional[List[StockTimeTrade]]:
        """
        获取指定股票、时间级别和数量限制的历史分时交易数据。

        查询顺序:
        1. 尝试从 Redis 缓存获取数据 (期望格式为 List[Dict])。
        2. 如果缓存命中，手动将字典列表转换为 `StockTimeTrade` 模型实例列表。
        3. 如果缓存未命中或转换失败，则从数据库查询。
        4. 返回按交易时间升序排列的模型实例列表。

        Args:
            stock_code: 股票代码。
            time_level: 时间周期级别 (可以是 TimeLevel 枚举或其字符串表示)。
            limit: 需要获取的最新数据条数。

        Returns:
            包含 `StockTimeTrade` 模型实例的列表 (按时间升序)，如果找不到数据或出错则返回 None。
        """
        # 确保缓存对象已初始化
        if self.cache_get is None or self.stock_basic_dao is None:
            await self.initialize_cache_objects()

        # 获取股票基础信息对象
        stock = await self.stock_basic_dao.get_stock_by_code(stock_code)
        if not stock:
            logger.warning(f"无法找到股票信息: {stock_code}")
            return None

        # 标准化时间级别为字符串
        time_level_str = time_level.value if isinstance(time_level, TimeLevel) else str(time_level)

        # --- 1. 尝试从 Redis 缓存获取数据 ---
        cache_data: Optional[List[Dict]] = None
        try:
            cache_data = await self.cache_get.history_time_trade_by_limit(stock_code, time_level_str, limit)
        except Exception as e:
            logger.error(f"从 Redis 获取缓存数据时出错 for {stock_code} {time_level_str}: {e}", exc_info=True)
            cache_data = None

        # --- 2. 处理缓存数据 (如果命中) ---
        if cache_data and isinstance(cache_data, list):
            # logger.debug(f"缓存命中: 获取到 {stock_code} {time_level_str} 历史数据 {len(cache_data)} 条 (limit={limit})，进行手动转换...")
            model_instances = []
            conversion_errors = 0
            for item_dict_str in cache_data:
                try:
                    if isinstance(item_dict_str, bytes):
                        item_dict = self.cache_manager._deserialize(item_dict_str)
                    else:
                        item_dict = item_dict_str
                    # 手动将字典转换为 StockTimeTrade 模型实例
                    trade_time = self._safe_datetime(item_dict.get('trade_time'))
                    if not trade_time: # 如果时间无效，跳过此条记录
                        logger.warning(f"缓存数据中发现无效的 trade_time: {item_dict.get('trade_time')}")
                        conversion_errors += 1
                        continue

                    # 注意：这里不设置 id，因为缓存字典中没有
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
                        # 其他 StockTimeTrade 可能有的字段，如果缓存中有，也需要在这里添加转换逻辑
                    )
                    model_instances.append(instance)
                except Exception as e_conv:
                    conversion_errors += 1
                    # 记录转换单个字典时的错误，避免打印过多日志可以选择不显示 exc_info
                    logger.error(f"转换缓存字典为 StockTimeTrade 实例时出错: {e_conv}. Dict: {item_dict}", exc_info=False)

            if conversion_errors > 0:
                 logger.warning(f"转换缓存数据时遇到 {conversion_errors} 个错误 for {stock_code} {time_level_str}")

            if not model_instances:
                logger.warning(f"缓存数据转换后为空列表 for {stock_code} {time_level_str}，将尝试从数据库获取。")
                # 不返回 None，继续尝试数据库
            else:
                # 按交易时间升序排序
                model_instances.sort(key=lambda x: x.trade_time)
                logger.debug(f"成功从缓存转换 {len(model_instances)} 条 StockTimeTrade 实例 for {stock_code} {time_level_str}")
                return model_instances # 返回成功转换的实例列表

        # --- 3. 缓存未命中或处理失败，从数据库获取 ---
        logger.debug(f"缓存未命中或处理失败 for {stock_code} {time_level_str}，从数据库获取...")
        try:
            # 查询数据库，按时间降序获取最新的 limit 条
            data_qs = StockTimeTrade.objects.filter(
                stock=stock,
                time_level=time_level_str
            ).order_by('-trade_time')[:limit]

            # 异步执行查询
            data_list = await sync_to_async(list)(data_qs)

            if not data_list:
                logger.warning(f"数据库中未找到 {stock_code} {time_level_str} 的历史数据")
                return None

            logger.debug(f"从数据库获取到 {stock_code} {time_level_str} {len(data_list)} 条历史数据")

            # 反转列表，得到升序排列
            data_list.reverse()

            return data_list # 返回从数据库获取并排序后的列表

        except Exception as e_db:
            logger.error(f"从数据库获取股票[{stock_code}] {time_level_str} 级别分时成交数据失败: {str(e_db)}", exc_info=True)
            return None # 查询失败返回 None

    # 
    async def get_history_ohlcv_df(self, stock_code: str, time_level: Union[TimeLevel, str], limit: int = 1000) -> Optional[pd.DataFrame]:
        """
        获取历史数据并转换为 finta 需要的 DataFrame 格式。
        返回的 DataFrame 按时间升序排列。
        (此方法逻辑不变，因为它依赖于 get_history_time_trades_by_limit 返回 List[StockTimeTrade])
        """
        time_level_str = ""
        if time_level == "5m":
            time_level_str = '5'
        elif time_level == '15m':
            time_level_str = '15'
        elif time_level == '30m':
            time_level_str = '30'
        elif time_level == '60m':
            time_level_str = '60'
        elif time_level == 'D':
            time_level_str = 'day'
        elif time_level == "W":
            time_level_str = 'week'
        elif time_level == "M":
            time_level_str = 'month'
        else:
            time_level_str = time_level
        history_trades = await self.get_history_time_trades_by_limit(stock_code, time_level_str, limit)
        if not history_trades:
            logger.warning(f"get_history_time_trades_by_limit 未返回数据 for {stock_code} {time_level}")
            return None
        try:
            # 1. 将模型列表转换为 DataFrame
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
            # 2. 对非数值列应用分类类型 (在 df 创建之后)
            # 确保在访问 df.columns 之前 df 已经被定义
            for col in df.columns:
                # 检查列是否存在且类型为 object，并且不是时间索引列
                if col in df.columns and df[col].dtype == 'object' and col != 'trade_time':
                    try:
                        df[col] = df[col].astype('category')
                    except Exception as e_cat:
                        # 如果转换失败，记录警告但继续
                        logger.warning(f"转换列 '{col}' 为 category 类型失败: {e_cat}")
            # 3. 重命名列以匹配 finta 要求
            df.rename(columns=FINTA_OHLCV_MAP, inplace=True)
            # 4. 将 trade_time 设置为索引
            df['trade_time'] = pd.to_datetime(df['trade_time'], utc=True)
            df.set_index('trade_time', inplace=True)
            # 5. 去除重复索引
            initial_len = len(df)
            if df.index.has_duplicates:
                # logger.warning(f"发现重复的时间戳索引 for {stock_code} {time_level}，将进行去重处理 (保留最后一个)")
                # 保留每个重复时间戳的最后一条记录
                df = df[~df.index.duplicated(keep='last')]
                # logger.info(f"索引去重完成 for {stock_code} {time_level}，记录数从 {initial_len} 变为 {len(df)}")
            # --- 去重结束 ---
            # 6. 确保数据按时间升序排列
            df.sort_index(ascending=True, inplace=True)
            # 7. 验证必要的列是否存在
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_cols):
                logger.error(f"DataFrame 缺少必要列: {stock_code} {time_level}. 需要: {required_cols}, 实际: {df.columns.tolist()}")
                return None
            # 移除完全是 NaN 的行 (如果需要)
            # df.dropna(subset=required_cols, how='all', inplace=True) # 谨慎使用，可能移除计算指标需要的数据点
            # 8. 检查是否有足够的非 NaN 数据行
            if df[required_cols].isnull().all(axis=1).sum() == len(df):
                 logger.warning(f"处理后 DataFrame 只包含 NaN 值: {stock_code} {time_level}")
                 return None # 如果全是 NaN，返回 None
            return df

        except Exception as e:
            logger.error(f"转换 {stock_code} {time_level} 历史数据为 DataFrame 失败: {str(e)}", exc_info=True)
            return None

    # --- 其他 DAO 方法 ---
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

    # --- 通用保存方法 ---
    # --- 确保 _save_indicator_data_generic 处理时区感知的索引 ---
    # (如果之前未处理，请确保此方法中的索引处理部分正确设置时区)
    async def _save_indicator_data_generic(self, stock_info: 'StockInfo', time_level: Union[TimeLevel, str],
        indicator_df: pd.DataFrame, model_class: Type['models.Model'],
        field_map: Dict[str, str],
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

        # 在处理 DataFrame 之前，可以对非数值列应用分类类型
        # 例如 time_level 列（如果存在于 DataFrame 中）
        if 'time_level' in indicator_df.columns and indicator_df['time_level'].dtype == 'object':
            indicator_df['time_level'] = indicator_df['time_level'].astype('category')

        # 检查输入 DataFrame 是否有效
        if indicator_df is None or indicator_df.empty:
            logger.warning(f"[{model_name}] 无指标数据可保存 for {stock_info.stock_code} {time_level_str}")
            return

        # 1. 准备 data_list (List[Dict]) 以供批量操作
        data_to_save: List[Dict[str, Any]] = []
        preparation_failed_count = 0 # 记录准备数据阶段失败的行数

        # 确保 DataFrame 索引是 DatetimeIndex 且时区感知
        if not isinstance(indicator_df.index, pd.DatetimeIndex):
            try:
                indicator_df.index = pd.to_datetime(indicator_df.index)
            except Exception as e_idx:
                logger.error(f"[{model_name}] 无法将索引转换为 DatetimeIndex for {stock_info.stock_code} {time_level_str}: {e_idx}", exc_info=True)
                return

        default_tz = timezone.get_default_timezone()
        if indicator_df.index.tz is None:
            try:
                indicator_df.index = indicator_df.index.tz_localize(default_tz)
            except Exception as e_tz:
                 logger.error(f"[{model_name}] 无法本地化索引时区 for {stock_info.stock_code} {time_level_str}: {e_tz}", exc_info=True)
                 return # 时区处理失败则无法继续
        elif indicator_df.index.tz != default_tz:
             indicator_df.index = indicator_df.index.tz_convert(default_tz)
            

        # logger.info(f"[{model_name}] 开始准备 {len(indicator_df)} 条记录的批量数据 for {stock_info.stock_code} {time_level_str}")

        # 遍历 DataFrame 的每一行来构建字典列表
        for trade_time, row in indicator_df.iterrows():
            try:
                # 确保 trade_time 是时区感知的 (aware)
                # 假设数据库和 Django 设置使用 UTC 或默认时区
                default_tz = timezone.get_current_timezone()
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
                    'trade_time': aware_trade_time, # 直接使用带时区的索引
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
            # logger.warning(f"[{model_name}] 没有准备好任何有效数据进行批量保存 for {stock_info.stock_code} {time_level_str}")
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
            # logger.info(f"[{model_name}] 批量保存/更新调用完成 for {stock_info.stock_code} {time_level_str}. 结果: {result}")

        except Exception as e_bulk:
            # 捕获调用基类方法本身可能抛出的意外错误（尽管基类方法应处理其内部错误）
            logger.error(f"[{model_name}] 调用 _save_all_to_db_native_upsert 时发生意外错误 for {stock_info.stock_code} {time_level_str}: {e_bulk}", exc_info=True)


    # --- 各指标具体的读取方法 ---
    @staticmethod
    def _find_closest_fib_period(target_period: int) -> int:
        """找到斐波那契周期列表中最接近目标周期的值"""
        if not FIB_PERIODS:
            raise ValueError("FIB_PERIODS constant is not defined or empty.")
        # 计算每个斐波那契周期与目标周期的绝对差值
        diffs = {period: abs(period - target_period) for period in FIB_PERIODS}
        # 找到差值最小的那个周期
        closest_period = min(diffs, key=diffs.get)
        return closest_period

    # --- 新增方法：获取最新指标时间戳 ---
    async def get_latest_indicator_timestamp(self, stock_info: 'StockInfo', time_level: str, model_class: Type['models.Model']) -> Optional[timezone.datetime]:
        """
        获取指定模型、股票和时间级别的最新 trade_time。

        Args:
            stock_info (StockInfo): 股票信息实例.
            time_level (str): 时间级别字符串.
            model_class (Type[models.Model]): 指标模型类.

        Returns:
            Optional[timezone.datetime]: 数据库中最新的 trade_time (时区感知)，如果不存在则返回 None。
        """
        model_name = model_class.__name__
        try:
            # 异步执行数据库查询
            latest_time_result = await sync_to_async(
                model_class.objects.filter(
                    stock=stock_info,
                    time_level=time_level
                ).aggregate
            )(latest_trade_time=Max('trade_time')) # 使用 Max 聚合

            latest_timestamp = latest_time_result.get('latest_trade_time')

            if latest_timestamp:
                # 确保返回的是时区感知的时间
                default_tz = timezone.get_default_timezone() # 获取默认时区
                if timezone.is_naive(latest_timestamp):
                    # 如果数据库返回的是 naive 时间，假定它是默认时区
                    # logger.warning(f"[{model_name}] 从数据库获取的最新时间戳 {latest_timestamp} 是 naive 的，将假定为默认时区。")
                    return timezone.make_aware(latest_timestamp, default_tz)
                else:
                    # 如果已经是 aware，确保它是默认时区
                    return latest_timestamp.astimezone(default_tz)
            else:
                # logger.debug(f"[{model_name}] 未找到 {stock_info.stock_code} {time_level} 的现有数据。")
                return None # 没有找到记录

        except Exception as e:
            logger.error(f"[{model_name}] 查询最新时间戳失败 for {stock_info.stock_code} {time_level}: {e}", exc_info=True)
            return None # 查询出错也返回 None

    # --- 新增方法：获取指定范围内的已存在时间戳 ---
    async def get_existing_timestamps_for_range(
        self,
        stock_info: 'StockInfo',
        time_level: str,
        model_class: Type['models.Model'],
        timestamps_to_check: List[pd.Timestamp]
    ) -> Set[pd.Timestamp]:
        """
        查询数据库，返回在给定时间戳列表(timestamps_to_check)中已经存在的记录的时间戳集合。

        Args:
            stock_info (StockInfo): 股票信息实例.
            time_level (str): 时间级别字符串.
            model_class (Type[models.Model]): 指标模型类.
            timestamps_to_check (List[pd.Timestamp]): 需要检查是否存在的时间戳列表 (必须是时区感知的).

        Returns:
            Set[pd.Timestamp]: 在数据库中已存在的时间戳集合 (时区感知).
        """
        model_name = model_class.__name__
        existing_timestamps = set()
        if not timestamps_to_check:
            return existing_timestamps

        # 确保输入的时间戳是时区感知的
        if not all(ts.tzinfo is not None for ts in timestamps_to_check):
             logger.warning(f"[{model_name}] 传递给 get_existing_timestamps_for_range 的时间戳列表包含 naive datetime，可能导致查询错误。")
             # 可以选择在这里强制转换时区或直接返回空集合/抛出错误
             # return existing_timestamps

        try:
            # 使用 __in 查询优化性能
            # 注意：数据库中的 trade_time 字段也必须是时区感知的才能正确比较
            query = model_class.objects.filter(
                stock=stock_info,
                time_level=time_level,
                trade_time__in=timestamps_to_check # 查询时间戳是否在列表中
            ).values_list('trade_time', flat=True) # 只获取 trade_time 字段

            # 异步执行查询
            found_timestamps = await sync_to_async(list)(query)

            # 转换为 Pandas Timestamps 并确保时区一致性
            default_tz = timezone.get_default_timezone()
            for ts in found_timestamps:
                if isinstance(ts, pd.Timestamp):
                    if ts.tzinfo is None:
                        existing_timestamps.add(ts.tz_localize(default_tz))
                    else:
                        existing_timestamps.add(ts.tz_convert(default_tz))
                else: # 处理 datetime.datetime 对象
                    if timezone.is_naive(ts):
                         existing_timestamps.add(timezone.make_aware(ts, default_tz))
                    else:
                         existing_timestamps.add(ts.astimezone(default_tz))

            # logger.debug(f"[{model_name}] 在 {len(timestamps_to_check)} 个待查时间戳中，找到 {len(existing_timestamps)} 个已存在记录 for {stock_info.stock_code} {time_level}")
            return existing_timestamps

        except Exception as e:
            logger.error(f"[{model_name}] 查询已存在时间戳失败 for {stock_info.stock_code} {time_level}: {e}", exc_info=True)
            return set() # 查询出错返回空集合

    # --- 新增方法：检查特定时间戳的指标记录是否存在 ---
    async def check_indicator_exists_at_timestamp(
        self,
        stock_info: 'StockInfo',
        time_level: str,
        model_class: Type['models.Model'],
        timestamp_to_check: pd.Timestamp
    ) -> bool:
        """
        检查数据库中是否存在指定股票、时间级别和特定时间戳的指标记录。

        Args:
            stock_info (StockInfo): 股票信息实例.
            time_level (str): 时间级别字符串.
            model_class (Type[models.Model]): 指标模型类.
            timestamp_to_check (pd.Timestamp): 需要检查是否存在的时间戳 (必须是时区感知的).

        Returns:
            bool: 如果记录存在则返回 True，否则返回 False。
        """
        model_name = model_class.__name__

        # 确保输入的时间戳是时区感知的
        if timestamp_to_check.tzinfo is None:
             logger.error(f"[{model_name}] 传递给 check_indicator_exists_at_timestamp 的时间戳是 naive 的，无法进行检查。")
             return False # 或者抛出错误

        try:
            # 使用 exists() 进行高效检查
            exists = await sync_to_async(
                model_class.objects.filter(
                    stock=stock_info,
                    time_level=time_level,
                    trade_time=timestamp_to_check # 直接比较时区感知的时间戳
                ).exists
            )()
            return exists

        except Exception as e:
            logger.error(f"[{model_name}] 检查时间戳 {timestamp_to_check} 是否存在失败 for {stock_info.stock_code} {time_level}: {e}", exc_info=True)
            return False # 查询出错也认为不存在，以便后续尝试计算









