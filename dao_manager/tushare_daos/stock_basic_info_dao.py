
import asyncio
import logging
from asgiref.sync import sync_to_async
from typing import TYPE_CHECKING, Dict, List, Any, Optional
import time
from django.db import OperationalError
import numpy as np
import pandas as pd
from utils.cache_manager import CacheManager
from django.contrib.auth.models import AbstractUser
from dao_manager.base_dao import BaseDAO
from stock_models.stock_basic import HSConst, StockCompany, StockInfo
from utils.cache_get import UserCacheGet, StockInfoCacheGet
from utils.cache_set import StockInfoCacheSet, UserCacheSet
from django.contrib.auth import get_user_model
from users.models import FavoriteStock
from utils.data_format_process import StockInfoFormatProcess
    

logger = logging.getLogger("dao")
User = get_user_model()

class StockBasicInfoDao(BaseDAO):
    def __init__(self, cache_manager_instance: CacheManager):
        # 调用 super() 时，将 cache_manager_instance 传递进去
        super().__init__(cache_manager_instance=cache_manager_instance, model_class=None)
        self.data_format_process = StockInfoFormatProcess(cache_manager_instance)
        self.stock_cache_set = StockInfoCacheSet(self.cache_manager)
        self.stock_cache_get = StockInfoCacheGet(self.cache_manager)
        self.user_cache_set = UserCacheSet(self.cache_manager)
        self.user_cache_get = UserCacheGet(self.cache_manager)

    async def get_stock_list(self) -> List['StockInfo']:
        """
        【V2.1 - 向量化缓存版】获取所有股票的基本信息
        优化：
        1. 使用 Pandas 批量处理数据库查询结果，替代循环格式化。
        2. 批量写入缓存，减少 Redis 交互次数。
        """
        return_data = []
        try:
            cached_data = await self.stock_cache_get.all_stocks()
            if cached_data:
                cached_data.sort(key=lambda x: x.get('stock_code', ''))
                for stock_dict in cached_data:
                    if stock_dict.get('list_status') == 'L' and not stock_dict.get('stock_code', '').endswith('.BJ'):
                        return_data.append(StockInfo(**stock_dict))
            if return_data:
                return return_data
        except Exception as e:
            logger.error(f"从缓存获取股票列表失败: {e}", exc_info=True)
        try:
            # 使用 values() 获取字典列表，直接转换为 DataFrame，比 list(objects) 更快
            qs = StockInfo.objects.filter(list_status='L').exclude(stock_code__endswith='.BJ').order_by('stock_code').values()
            stock_list = await sync_to_async(list)(qs)
            
            if stock_list:
                # 转换为 DataFrame 进行批量处理
                df = pd.DataFrame(stock_list)
                
                # 处理日期列 (转为字符串以存入缓存)
                date_cols = ['list_date', 'delist_date']
                for col in date_cols:
                    if col in df.columns:
                        df[col] = df[col].astype(str).replace({'NaT': None, 'nan': None, 'None': None})

                # 转换为字典列表
                data_to_cache = df.where(pd.notnull(df), None).to_dict('records')
                
                # 批量写入缓存
                # 注意：stock_cache_set.stock_basic_info 是单条写入，这里可以使用 pipeline 或并发
                # 为了保持接口一致性，这里使用 asyncio.gather 并发写入单条缓存
                # 但更优的是实现一个 mset 方法
                cache_tasks = [self.stock_cache_set.stock_basic_info(d['stock_code'], d) for d in data_to_cache]
                await asyncio.gather(*cache_tasks)
                
                # 写入全量列表缓存
                await self.stock_cache_set.all_stocks(data_to_cache)
                
                # 重建模型对象列表返回
                return_data = [StockInfo(**d) for d in stock_list] # 注意这里用原始 list 重建，避免日期格式问题
                
        except Exception as e:
            logger.error(f"从数据库读取股票列表失败: {e}", exc_info=True)
        return return_data

    async def get_stock_by_code(self, stock_code: str) -> Optional['StockInfo']:
        """
        【V1.1 - 异步修复版】获取单个股票信息
        优化：修复了 time.sleep 阻塞事件循环的 Bug，改为 await asyncio.sleep。
        """
        retry = 3
        for i in range(retry):
            try:
                stock = await sync_to_async(lambda: StockInfo.objects.filter(stock_code=stock_code).first())()
                if stock:
                    cache_data = self.data_format_process.set_stock_info_basic_data(stock)
                    await self.stock_cache_set.stock_basic_info(stock.stock_code, cache_data)
                    return stock
                break
            except OperationalError as e:
                print(f"数据库连接丢失，重试第{i+1}次: {e}")
                await asyncio.sleep(0.2) # 修正：使用异步 sleep
        return None

    async def get_stocks_by_codes(self, stock_codes: List[str]) -> Dict[str, 'StockInfo']:
        """
        【V2.1 - 批量获取优化版】批量获取股票信息，返回以stock_code为key的字典。
        优化：使用 sync_to_async(list) 一次性拉取数据，比 async for 逐行迭代减少了大量 await 上下文切换开销。
        """
        if not stock_codes:
            return {}
        retry_count = 3
        for attempt in range(retry_count):
            try:
                # 优化：在线程池中一次性获取所有对象，避免 async for 的频繁挂起
                stocks = await sync_to_async(list)(
                    StockInfo.objects.filter(stock_code__in=stock_codes)
                )
                return {stock.stock_code: stock for stock in stocks}
            except OperationalError as e:
                logger.warning(f"数据库连接丢失，正在进行第 {attempt + 1}/{retry_count} 次重试... 错误: {e}")
                if attempt + 1 == retry_count:
                    logger.error("数据库连接重试失败，放弃操作。")
                    raise
                await asyncio.sleep(0.5 * (attempt + 1))
            except Exception as e:
                logger.error(f"批量查找股票信息时发生未知异常: {e}", exc_info=True)
                raise
        return {}

    async def get_all_favorite_stocks(self) -> Optional[List[Dict]]:
        """
        【V2.0 - 序列化优化版】获取所有用户的自选股。
        优化：使用 .values() 直接获取字典，避免模型实例化的巨大开销。
        """
        print("调试: 开始获取所有用户的自选股数据...")
        try:
            # 优化：仅查询需要的字段，利用数据库别名直接重命名关联字段
            # F表达式或直接指定关联字段名
            qs = FavoriteStock.objects.order_by('stock__stock_code').values(
                'id', 'user_id', 'added_at', 'note', 'is_pinned', 'tags',
                stock_code_val=F('stock__stock_code'),
                stock_name_val=F('stock__stock_name')
            )
            
            raw_data = await sync_to_async(list)(qs)
            
            if not raw_data:
                print("调试: 数据库中没有找到任何自选股记录。")
                return []
                
            print(f"调试: 从数据库成功获取 {len(raw_data)} 条自选股记录。")
            
            # 快速重构字典键名以匹配前端需求
            fav_datas = [
                {
                    "id": item['id'],
                    "user_id": item['user_id'],
                    "stock_code": item['stock_code_val'],
                    "stock_name": item['stock_name_val'],
                    "added_at": item['added_at'],
                    "note": item['note'],
                    "is_pinned": item['is_pinned'],
                    "tags": item['tags'],
                }
                for item in raw_data
            ]
            return fav_datas
        except Exception as e:
            logger.error(f"从数据库获取所有自选股失败: {e}", exc_info=True)
            return None

    # --- 增加一个更常用的“获取指定用户自选股”的方法 ---
    async def get_user_favorite_stocks(self, user: AbstractUser) -> Optional[List[Dict]]:
        """
        【V2.0 - 序列化优化版】获取指定用户的自选股列表。
        优化：使用 .values() 替代模型查询，大幅降低内存占用和CPU时间。
        """
        if not user or not user.is_authenticated:
            print("调试: 用户未提供或未认证，无法获取自选股。")
            return []
        print(f"调试: 开始获取用户 {user.username} (ID: {user.id}) 的自选股数据...")
        try:
            # 优化：使用 values() 避免 N+1 和模型实例化
            qs = FavoriteStock.objects.filter(user=user).values(
                'id', 'added_at', 'note', 'is_pinned', 'tags',
                stock_code_val=F('stock__stock_code'),
                stock_name_val=F('stock__stock_name')
            ).order_by('-is_pinned', '-added_at') # 保持默认排序
            
            raw_data = await sync_to_async(list)(qs)
            
            if not raw_data:
                print(f"调试: 用户 {user.username} 没有任何自选股记录。")
                return []
                
            print(f"调试: 成功为用户 {user.username} 获取 {len(raw_data)} 条自选股记录。")
            
            fav_datas = [
                {
                    "id": item['id'],
                    "stock_code": item['stock_code_val'],
                    "stock_name": item['stock_name_val'],
                    "added_at": item['added_at'],
                    "note": item['note'],
                    "is_pinned": item['is_pinned'],
                    "tags": item['tags'],
                }
                for item in raw_data
            ]
            return fav_datas
        except Exception as e:
            logger.error(f"为用户 {user.username} 获取自选股失败: {e}", exc_info=True)
            return None

    async def save_stocks(self) -> Dict:
        """
        【V2.2 - 向量化增强版】通过tushare获取股票数据并保存到数据库
        优化：
        1. 保持 Pandas 向量化清洗逻辑。
        2. 优化空值替换逻辑，确保 None 值能被 Django ORM 正确处理。
        """
        df = self.ts_pro.stock_basic(**{
            "ts_code": "", "name": "", "exchange": "", "market": "", "is_hs": "", "list_status": "", "limit": "", "offset": ""
        }, fields=[
            "ts_code", "symbol", "name", "area", "industry", "cnspell", "market", "list_date", "act_name", "act_ent_type",
            "fullname", "enname", "exchange", "curr_type", "list_status", "delist_date", "is_hs"
        ])
        if df is None or df.empty:
            return {}
        
        # 向量化清洗
        df.replace(['nan', 'NaN', ''], np.nan, inplace=True)
        
        # 日期转换
        df['list_date'] = pd.to_datetime(df['list_date'], format='%Y%m%d', errors='coerce').dt.date
        df['delist_date'] = pd.to_datetime(df['delist_date'], format='%Y%m%d', errors='coerce').dt.date
        
        rename_map = {
            'ts_code': 'stock_code',
            'name': 'stock_name',
            'fullname': 'full_name',
            'enname': 'en_name',
            'cnspell': 'cn_spell',
            'market': 'market_type',
            'curr_type': 'currency_type',
            'act_name': 'actual_controller',
            'act_ent_type': 'actual_controller_type'
        }
        df.rename(columns=rename_map, inplace=True)
        
        # 筛选数据库字段
        model_fields = {f.name for f in StockInfo._meta.get_fields()}
        db_cols = [col for col in df.columns if col in model_fields]
        db_df = df[db_cols]
        
        # 转换为字典列表，并将 NaN 替换为 None
        stock_dicts = db_df.where(pd.notnull(db_df), None).to_dict('records')
        
        # 准备缓存数据
        cache_cols = ['stock_code', 'stock_name', 'list_status', 'list_date', 'delist_date', 'exchange', 'market_type', 'is_hs', 'industry']
        cache_df = df[[col for col in cache_cols if col in df.columns]]
        cache_dicts = cache_df.where(pd.notnull(cache_df), None).to_dict('records')
        
        # 并发写入缓存
        cache_tasks = [self.stock_cache_set.stock_basic_info(d['stock_code'], d) for d in cache_dicts if d.get('stock_code')]
        await asyncio.gather(*cache_tasks)
        await self.stock_cache_set.all_stocks(cache_dicts)
        
        if stock_dicts:
            result = await self._save_all_to_db_native_upsert(
                model_class=StockInfo,
                data_list=stock_dicts,
                unique_fields=['stock_code']
            )
            return result
        return {}

    async def save_company_info(self) -> Dict:
        """
        【V2.1 - 完全向量化版】通过tushare获取所有公司信息并保存到数据库
        优化：
        1. 移除 itertuples 循环，使用 Pandas map 实现 stock 对象的向量化映射。
        2. 直接从 DataFrame 生成字典列表，大幅提升构建速度。
        """
        print("调试: 开始执行 save_company_info 任务...")
        try:
            print("调试: 正在从Tushare API拉取所有上市公司基本信息...")
            df = self.ts_pro.stock_company(**{
                "ts_code": "", "exchange": "", "status": "L",
            }, fields=[
                "ts_code", "com_name", "chairman", "manager", "secretary", "reg_capital", "setup_date", "province",
                "city", "introduction", "website", "email", "office", "business_scope", "employees", "main_business", "exchange"
            ])
            if df.empty:
                logger.info("Tushare API没有返回任何公司信息，任务提前结束。")
                return {"status": "success", "message": "No data returned from API.", "saved_count": 0}
            
            # 数据清洗
            df = df.replace(['nan', 'NaN', ''], np.nan)
            
            # 批量获取 StockInfo 对象
            unique_ts_codes = df['ts_code'].unique().tolist()
            print(f"调试: 从API获取了 {len(df)} 条公司数据，涉及 {len(unique_ts_codes)} 个独立股票代码。")
            stock_map = await self.get_stocks_by_codes(unique_ts_codes)
            print(f"调试: 批量从数据库获取了 {len(stock_map)} 个股票对象。")
            
            # 向量化映射 Stock 对象
            df['stock'] = df['ts_code'].map(stock_map)
            
            # 过滤无效数据 (无对应股票或无公司名)
            df = df.dropna(subset=['stock', 'com_name'])
            
            if df.empty:
                logger.info("经过筛选后，没有需要保存到数据库的公司数据。")
                return {"status": "success", "message": "No new data to save.", "saved_count": 0}
            
            # 字段重命名以匹配模型 (假设模型字段名与API字段名有对应关系，这里需根据实际模型调整)
            # StockCompany 模型字段通常与 API 字段一致，除了外键 'stock'
            # 如果有不一致，需在此处 rename
            
            # 转换为字典列表
            # 注意：这里假设 DataFrame 列名与 StockCompany 模型字段名一致
            # 排除 ts_code 列，因为已经映射为 stock 对象
            cols_to_keep = [c for c in df.columns if c != 'ts_code']
            final_df = df[cols_to_keep]
            
            data_dicts_to_save = final_df.where(pd.notnull(final_df), None).to_dict('records')
            
            print(f"调试: 准备批量保存 {len(data_dicts_to_save)} 条公司数据到数据库...")
            result = await self._save_all_to_db_native_upsert(
                model_class=StockCompany,
                data_list=data_dicts_to_save,
                unique_fields=['stock']
            )
            saved_count = len(data_dicts_to_save)
            logger.info(f"成功批量保存 {saved_count} 条公司信息。")
            return result
            
        except Exception as e:
            logger.error(f"保存公司信息时发生严重错误: {e}", exc_info=True)
            raise

    async def save_hs_const(self) -> Dict:
        """
        【V2.1 - 向量化优化版】通过tushare获取沪深港通成分股信息并保存到数据库
        优化：保持向量化逻辑，确保空值处理的健壮性。
        """
        df_sh = self.ts_pro.hs_const(hs_type='SH')
        df_sz = self.ts_pro.hs_const(hs_type='SZ')
        df = pd.concat([df_sh, df_sz], ignore_index=True)
        if df is None or df.empty:
            return {}
        
        df.replace(['nan', 'NaN', ''], np.nan, inplace=True)
        df.dropna(subset=['ts_code'], inplace=True)
        if df.empty:
            return {}
            
        unique_ts_codes = df['ts_code'].unique().tolist()
        stock_map = await self.get_stocks_by_codes(unique_ts_codes)
        
        df['stock'] = df['ts_code'].map(stock_map)
        df.dropna(subset=['stock'], inplace=True)
        
        if df.empty:
            logger.warning("所有沪深港通成分股都无法关联到已知的股票信息，任务终止。")
            return {}
            
        df['in_date'] = pd.to_datetime(df['in_date'], format='%Y%m%d', errors='coerce').dt.date
        df['out_date'] = pd.to_datetime(df['out_date'], format='%Y%m%d', errors='coerce').dt.date
        df.rename(columns={'hs_type': 'hs_type_code'}, inplace=True)
        
        final_df = df[['stock', 'in_date', 'out_date', 'is_new', 'hs_type_code']]
        
        # 转换为字典列表，处理 NaN -> None
        stock_dicts = final_df.where(pd.notnull(final_df), None).to_dict('records')
        
        if stock_dicts:
            result = await self._save_all_to_db_native_upsert(
                model_class=HSConst,
                data_list=stock_dicts,
                unique_fields=['stock']
            )
            return result
        return {}




















