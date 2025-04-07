# self_built_indicators_dao.py
import json
import logging
import operator
from functools import reduce
from typing import List, Optional, Dict, Type, Tuple, Any, Union
from datetime import datetime, date
from decimal import Decimal, InvalidOperation

import pandas as pd
import numpy as np
from django.db import transaction, models, Q
from django.utils import timezone
from asgiref.sync import sync_to_async

# 假设模型在 'your_app.models'
from stock_models.stock_basic import StockInfo
# 假设你的 CacheManager 和 CacheKey 在 utils 中
from utils.cache_manager import CacheManager
from utils.cash_key import StockCashKey # 假设有这个类用于生成缓存键

logger = logging.getLogger("dao")
CACHE_TIMEOUT = 60 * 60 * 2 # 缓存时间，例如2小时

class SelfBuiltIndicatorsDao:
    """
    用于存储和获取自行计算的技术指标数据的 DAO
    """
    def __init__(self):
        self.cache_manager = CacheManager()
        self.cache_key_generator = StockCashKey() # 假设使用 StockCashKey 生成键

    async def _dataframe_to_list_of_dicts(self,
                                          stock_code: str,
                                          kline_period: str,
                                          indicator_model_class: Type[models.Model],
                                          indicator_df: pd.DataFrame) -> Tuple[List[Dict[str, Any]], int]:
        """
        将指标 DataFrame 转换为用于保存的字典列表。
        同时进行数据类型和精度的处理。

        Returns:
            Tuple[List[Dict[str, Any]], int]: 包含处理后的数据字典列表和跳过的无效记录数。
        """
        data_list = []
        skipped_count = 0
        indicator_name = indicator_model_class._meta.verbose_name

        # 获取 Stock 对象一次
        try:
            # 使用 sync_to_async 获取 stock 对象
            stock = await sync_to_async(StockInfo.objects.get)(code=stock_code)
        except StockInfo.DoesNotExist:
            logger.error(f"错误: 转换 DataFrame 时未找到股票 {stock_code}。")
            return [], indicator_df.shape[0] if indicator_df is not None else 0 # 返回空列表和所有行都跳过

        # 将 NaN 替换为 None
        indicator_df = indicator_df.replace({np.nan: None})

        for timestamp, row in indicator_df.iterrows():
            py_timestamp = timestamp.to_pydatetime()
            # 确保时间戳是 naive 或 aware，与数据库设置匹配
            # 如果 Django 设置 USE_TZ=True，确保 py_timestamp 是 aware
            # if timezone.is_aware(py_timestamp):
            #     py_timestamp = timezone.make_naive(py_timestamp, timezone.get_default_timezone()) # 如果数据库是 naive
            # elif timezone.is_naive(py_timestamp) and settings.USE_TZ:
            #      py_timestamp = timezone.make_aware(py_timestamp, timezone.get_default_timezone()) # 如果数据库是 aware

            item_dict = {
                'stock': stock, # 直接存 Stock 对象引用
                'timestamp': py_timestamp,
                'period': kline_period,
            }
            valid_data = False
            for col_name, value in row.items():
                # 检查模型是否有此字段，并且值不是 None
                if hasattr(indicator_model_class, col_name) and value is not None:
                    try:
                        field = indicator_model_class._meta.get_field(col_name)
                        # 类型转换和精度处理
                        if isinstance(field, models.DecimalField):
                            processed_value = Decimal(str(value)).quantize(Decimal('0.0001')) # 保留4位小数
                        elif isinstance(field, (models.BigIntegerField, models.IntegerField)):
                            processed_value = int(value)
                        # 可以添加对 BooleanField, DateField 等的处理
                        else:
                            processed_value = value # 其他类型直接赋值
                        item_dict[col_name] = processed_value
                        valid_data = True
                    except (InvalidOperation, ValueError, TypeError) as convert_error:
                        logger.warning(f"警告: 转换字段 {col_name} 值 {value} 失败 ({stock_code}, {kline_period}, {py_timestamp}): {convert_error}")
                        continue # 跳过这个字段

            if valid_data:
                data_list.append(item_dict)
            else:
                skipped_count += 1

        if skipped_count > 0:
             logger.info(f"指标 {indicator_name} ({stock_code}, {kline_period}): "
                         f"转换 DataFrame 时跳过 {skipped_count} 条无效数据记录。")
        return data_list, skipped_count

    async def _save_all_to_db(self,
                              model_class: Type[models.Model],
                              data_list: List[Dict[str, Any]],
                              unique_fields: List[str]) -> Dict:
        """
        优化的通用异步数据批量处理方法 (基于提供的逻辑)。
        注意：此方法现在是 SelfBuiltIndicatorsDao 的内部方法。

        Args:
            model_class: Django模型类
            data_list: 要处理的数据列表，每项都是包含模型字段和值的字典 (包括 stock 对象引用)
            unique_fields: 用于确定唯一记录的字段列表 (例如 ['stock', 'timestamp', 'period'])

        Returns:
            dict: 包含创建、更新和跳过的记录数
        """
        if not data_list:
            logger.warning(f"未提供任何数据用于处理 - {model_class.__name__}")
            return {'创建': 0, '更新': 0, '跳过': 0}

        created_count = 0
        updated_count = 0
        skipped_count = 0
        batch_size = 1000 # 调整批量大小

        for i in range(0, len(data_list), batch_size):
            batch = data_list[i:i + batch_size]

            # 使用 sync_to_async 包装整个事务处理逻辑
            @sync_to_async
            @transaction.atomic
            def process_batch_sync():
                nonlocal created_count, updated_count, skipped_count
                batch_created = 0
                batch_updated = 0
                batch_skipped = 0

                # 1. 构建查询条件和待处理项映射
                items_to_process_map = {}
                query_filters = []
                for item in batch:
                    filter_kwargs = {}
                    valid_key = True
                    for field in unique_fields:
                        value = item.get(field)
                        if value is None:
                            valid_key = False
                            break
                        filter_kwargs[field] = value
                    if not valid_key:
                        batch_skipped += 1
                        continue

                    query_condition = Q(**filter_kwargs)
                    query_filters.append(query_condition)
                    # 创建唯一键 (处理 Stock 对象)
                    key_parts = []
                    for field in sorted(unique_fields):
                        value = item.get(field)
                        key_parts.append((field, value.pk if isinstance(value, models.Model) else value))
                    unique_key = tuple(key_parts)
                    items_to_process_map[unique_key] = item

                # 2. 批量查询现有记录
                existing_records_map = {}
                if query_filters:
                    combined_query = reduce(operator.or_, query_filters)
                    existing_queryset = model_class.objects.filter(combined_query)
                    for record in existing_queryset:
                        key_parts = []
                        for field in sorted(unique_fields):
                            value = getattr(record, field)
                            key_parts.append((field, value.pk if isinstance(value, models.Model) else value))
                        unique_key = tuple(key_parts)
                        existing_records_map[unique_key] = record

                # 3. 分离创建和更新
                to_create_instances = []
                to_update_instances = []
                update_field_names = set()

                for unique_key, item_data in items_to_process_map.items():
                    if unique_key in existing_records_map:
                        # 更新逻辑
                        record = existing_records_map[unique_key]
                        has_changes = False
                        current_update_fields = set()
                        for field, value in item_data.items():
                            if field not in unique_fields: # 只比较非唯一键字段
                                if hasattr(record, field):
                                    current_value = getattr(record, field)
                                    is_different = False
                                    # 更健壮的比较
                                    if value is None and current_value is not None: is_different = True
                                    elif value is not None and current_value is None: is_different = True
                                    elif value is not None and current_value is not None:
                                        if isinstance(current_value, Decimal) and isinstance(value, Decimal): is_different = current_value != value
                                        elif isinstance(current_value, float) and isinstance(value, float): is_different = not np.isclose(current_value, value, atol=1e-9, equal_nan=True) # equal_nan 处理 NaN 比较
                                        elif isinstance(current_value, (datetime, date)) and isinstance(value, (datetime, date)): is_different = current_value != value # 假设时区已处理好
                                        else: is_different = current_value != value

                                    if is_different:
                                        setattr(record, field, value)
                                        has_changes = True
                                        current_update_fields.add(field)

                        if has_changes:
                            to_update_instances.append(record)
                            update_field_names.update(current_update_fields)
                            batch_updated += 1
                        else:
                            batch_skipped += 1
                    else:
                        # 创建逻辑
                        if 'stock' in item_data and isinstance(item_data['stock'], StockInfo):
                             to_create_instances.append(model_class(**item_data))
                             batch_created += 1
                        else:
                             logger.error(f"创建记录时缺少有效的 'stock' 对象: {item_data}")
                             batch_skipped += 1

                # 4. 批量创建
                if to_create_instances:
                    try:
                        # ignore_conflicts=True 可以在唯一键冲突时跳过，避免因重复查询导致的问题，但会隐藏错误
                        model_class.objects.bulk_create(to_create_instances, ignore_conflicts=True)
                        # 如果 ignore_conflicts=True，实际创建数可能少于 len(to_create_instances)
                        # 这里仍然按尝试创建的数量计数，或者需要更复杂的逻辑来精确计数
                    except Exception as e:
                        logger.error(f"批量创建 {model_class.__name__} 失败: {e}")
                        batch_skipped += len(to_create_instances)
                        batch_created = 0

                # 5. 批量更新
                if to_update_instances and update_field_names:
                    valid_update_fields = [f for f in update_field_names if hasattr(model_class, f) and f not in unique_fields]
                    if valid_update_fields:
                        try:
                            model_class.objects.bulk_update(to_update_instances, valid_update_fields)
                        except Exception as e:
                            logger.error(f"批量更新 {model_class.__name__} 失败: {e}")
                            batch_skipped += len(to_update_instances)
                            batch_updated = 0

                return {'created': batch_created, 'updated': batch_updated, 'skipped': batch_skipped}

            # 执行异步批处理
            batch_result = await process_batch_sync()
            created_count += batch_result['created']
            updated_count += batch_result['updated']
            skipped_count += batch_result['skipped']

        final_result = {
            '创建': created_count,
            '更新': updated_count,
            '跳过': skipped_count
        }
        logger.info(f"完成 {model_class.__name__} 数据批量处理: {final_result}")
        return final_result

    async def save_indicator_data(self,
                                  stock_code: str,
                                  kline_period: str,
                                  indicator_model_class: Type[models.Model],
                                  indicator_df: pd.DataFrame):
        """
        将计算出的指标 DataFrame 保存到数据库。
        内部调用 _dataframe_to_list_of_dicts 和 _save_all_to_db。
        """
        indicator_name = indicator_model_class._meta.verbose_name
        logger.info(f"开始保存指标 {indicator_name} for {stock_code} ({kline_period})...")

        # 1. 将 DataFrame 转换为字典列表，并获取跳过的数量
        data_list, initial_skipped = await self._dataframe_to_list_of_dicts(
            stock_code, kline_period, indicator_model_class, indicator_df
        )

        if not data_list:
            logger.warning(f"没有有效的指标数据需要保存 for {stock_code} ({kline_period}) - {indicator_name}")
            return {'创建': 0, '更新': 0, '跳过': initial_skipped}

        # 2. 调用优化的保存方法
        unique_fields = ['stock', 'timestamp', 'period'] # 指标模型的唯一键字段
        result = await self._save_all_to_db(
            model_class=indicator_model_class,
            data_list=data_list,
            unique_fields=unique_fields
        )
        # 将转换过程中跳过的数量加到最终结果
        result['跳过'] += initial_skipped
        return result

    async def get_indicator_data(self,
                                 stock_code: str,
                                 kline_period: str,
                                 indicator_model_class: Type[models.Model],
                                 start_date: Optional[Union[str, date, datetime]] = None,
                                 end_date: Optional[Union[str, date, datetime]] = None) -> list[dict] | None:
        """
        从数据库获取指定范围的已存储指标数据，并使用 Redis 缓存。
        """
        indicator_name = indicator_model_class._meta.model_name
        # 使用 cache_key_generator 生成缓存键
        try:
            cache_key =  self.cache_key_generator.indicator_data(
                indicator_name, stock_code, kline_period, start_date, end_date
            )
        except AttributeError:
             cache_key = f"indicator:{indicator_name}:{stock_code}:{kline_period}:{start_date}:{end_date}"

        # 尝试从缓存获取
        cached_data = self.cache_manager.get(cache_key)
        if cached_data:
            logger.debug(f"从缓存命中获取指标: {cache_key}")
            # 假设缓存的是 Python 列表字典
            if isinstance(cached_data, list):
                return cached_data
            try:
                # 如果是 JSON 字符串，尝试解析
                return json.loads(cached_data)
            except (json.JSONDecodeError, TypeError):
                 logger.warning(f"缓存数据解析失败，将从数据库重新获取: {cache_key}")
                 pass # 继续从数据库获取

        # --- 从数据库获取逻辑 ---
        try:
            stock = await sync_to_async(StockInfo.objects.get)(code=stock_code)

            @sync_to_async
            def _get_from_db():
                queryset = indicator_model_class.objects.filter(
                    stock=stock,
                    period=kline_period
                ).order_by('timestamp')
                # 时间范围过滤
                if start_date:
                    queryset = queryset.filter(timestamp__gte=start_date)
                if end_date:
                    # ... (时间范围处理逻辑，同上个版本) ...
                    if isinstance(end_date, str):
                        try:
                            end_date_dt = datetime.strptime(end_date, '%Y-%m-%d').date()
                            queryset = queryset.filter(timestamp__date__lte=end_date_dt)
                        except ValueError:
                             try:
                                 end_date_dt = datetime.fromisoformat(end_date)
                                 queryset = queryset.filter(timestamp__lte=end_date_dt)
                             except ValueError: logger.warning(f"无法解析的结束日期格式: {end_date}")
                    elif isinstance(end_date, datetime): queryset = queryset.filter(timestamp__lte=end_date)
                    elif isinstance(end_date, date): queryset = queryset.filter(timestamp__date__lte=end_date)

                field_names = [f.name for f in indicator_model_class._meta.fields if f.name not in ['id', 'stock']]
                # 使用 .values() 获取字典列表
                return list(queryset.values(*field_names))

            result_list = await _get_from_db()

            # 序列化处理 (Decimal -> float, datetime -> str)
            serialized_result = []
            for item in result_list:
                serialized_item = {}
                for key, value in item.items():
                    if isinstance(value, Decimal):
                        serialized_item[key] = float(value)
                    elif isinstance(value, (datetime, date)):
                        serialized_item[key] = value.isoformat()
                    else:
                        serialized_item[key] = value
                serialized_result.append(serialized_item)

            # 缓存结果
            # 确保缓存的数据是 Python 原生类型或 JSON 字符串
            self.cache_manager.set(cache_key, serialized_result, timeout=CACHE_TIMEOUT)
            logger.debug(f"指标数据已存入缓存: {cache_key}")
            return serialized_result

        except StockInfo.DoesNotExist:
            logger.error(f"错误: 获取指标 {indicator_name} 时未找到股票 {stock_code}。")
            return None
        except Exception as e:
            logger.error(f"获取指标 {indicator_name} ({stock_code}, {kline_period}) 时出错: {e}")
            return None

