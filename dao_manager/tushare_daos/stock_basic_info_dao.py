
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
        获取所有股票的基本信息
        Returns:
            List[StockInfo]: 股票基本信息列表（正常上市状态）
        """
        return_data = []
        try:
            # 尝试从缓存获取
            cached_data = await self.stock_cache_get.all_stocks()
            if cached_data:
                # 按stock_code排序
                cached_data.sort(key=lambda x: x.get('stock_code', ''))
                # 将缓存数据转换为模型实例列表            
                for stock_dict in cached_data:
                    # logger.info(f"get_stock_list: {stock_dict}")
                    if stock_dict.get('list_status') == 'L' and not stock_dict.get('stock_code', '').endswith('.BJ'):
                        return_data.append(StockInfo(**stock_dict))
            if return_data:
                return return_data  # 直接返回模型实例列表
        except Exception as e:
            logger.error(f"从缓存获取股票列表失败: {e}",exc_info=True)
        try:
            # 从数据库读取
            return_data = await sync_to_async(
                lambda: list(StockInfo.objects.filter(list_status='L').exclude(stock_code__endswith='.BJ').order_by('stock_code')),
                thread_sensitive=True
            )()
            if return_data:
                data_to_cache = []
                for stock in return_data:
                    stock_dict = self.data_format_process.set_stock_info_basic_data(stock)
                    data_to_cache.append(stock_dict)
                    await self.stock_cache_set.stock_basic_info(stock.stock_code, stock_dict)
                await self.stock_cache_set.all_stocks(data_to_cache)
        except Exception as e:
            logger.error(f"从数据库读取股票列表失败: {e}", exc_info=True)
        return return_data

    async def get_stock_by_code(self, stock_code: str) -> Optional['StockInfo']:
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
                time.sleep(0.2)
        return None

    async def get_stocks_by_codes(self, stock_codes: List[str]) -> Dict[str, 'StockInfo']:
        """
        【V2 - 最佳实践版】批量获取股票信息，返回以stock_code为key的字典。
        优化点:
        1. 使用Django 5原生异步ORM，避免sync_to_async的开销。
        2. 使用异步字典推导式，一步完成查询和字典构建。
        3. 增强了异常处理，只对可恢复的OperationalError进行重试，其他错误则向上抛出。
        """
        if not stock_codes:
            return {}
        retry_count = 3
        for attempt in range(retry_count):
            try:
                # 【代码修改处】使用异步字典推导式，直接查询并构建字典
                # StockInfo.objects.filter(...) 返回一个异步查询集 (QuerySet)
                # 'async for' 会异步地从数据库获取每一条记录
                # 整个表达式一步到位，高效且优雅
                return {
                    stock.stock_code: stock
                    async for stock in StockInfo.objects.filter(stock_code__in=stock_codes)
                }
            except OperationalError as e:
                # 数据库连接错误是可重试的
                logger.warning(f"数据库连接丢失，正在进行第 {attempt + 1}/{retry_count} 次重试... 错误: {e}")
                if attempt + 1 == retry_count:
                    # 如果是最后一次重试，则记录错误并向上抛出异常
                    logger.error("数据库连接重试失败，放弃操作。")
                    raise  # 重新抛出异常，让上层调用者知道操作失败了
                await asyncio.sleep(0.5 * (attempt + 1))  # 增加重试等待时间
            except Exception as e:
                # 【代码修改处】对于其他所有非预期的异常，直接记录并向上抛出
                # 这可以防止隐藏如字段错误、类型错误等编程错误
                logger.error(f"批量查找股票信息时发生未知异常: {e}", exc_info=True)
                raise # 向上抛出，让调用方知道发生了严重错误
        # 理论上，由于上面的逻辑总会返回或抛出异常，代码不会执行到这里。
        # 但为了代码完整性，保留一个返回。
        return {}

    async def get_all_favorite_stocks(self) -> Optional[List[Dict]]:
        """
        获取所有用户的自选股，并按股票代码排序 (已优化)
        1. 使用 `select_related('stock')` 解决 N+1 查询问题。
        2. 使用 `order_by('stock__stock_code')` 直接在数据库层面完成排序。
        3. 返回结构化的字典列表。
        """
        print("调试: 开始获取所有用户的自选股数据...")
        try:
            # 使用 select_related('stock') 来预先加载关联的 StockInfo 对象，避免 N+1 查询。
            # 使用 order_by('stock__stock_code') 让数据库直接按股票代码排序。
            favorite_stocks_query = FavoriteStock.objects.select_related('stock').order_by('stock__stock_code')
            # 使用 sync_to_async 异步执行整个查询
            favorite_stock_objects = await sync_to_async(list)(favorite_stocks_query)
            if not favorite_stock_objects:
                print("调试: 数据库中没有找到任何自选股记录。")
                return []
            print(f"调试: 从数据库成功获取 {len(favorite_stock_objects)} 条自选股记录。")
            # 将查询结果转换为字典列表
            fav_datas = [
                {
                    "id": fav.id,
                    "user_id": fav.user_id, # fav.user_id 是外键的ID，可以直接获取
                    "stock_code": fav.stock.stock_code if fav.stock else None,
                    "stock_name": fav.stock.stock_name if fav.stock else None,
                    "added_at": fav.added_at,
                    "note": fav.note,
                    "is_pinned": fav.is_pinned,
                    "tags": fav.tags,
                }
                for fav in favorite_stock_objects
            ]
            return fav_datas
        except Exception as e:
            logger.error(f"从数据库获取所有自选股失败: {e}", exc_info=True)
            return None

    # --- 增加一个更常用的“获取指定用户自选股”的方法 ---
    async def get_user_favorite_stocks(self, user: AbstractUser) -> Optional[List[Dict]]:
        """
        获取指定用户的自选股列表，并按默认排序（置顶、添加时间）
        """
        if not user or not user.is_authenticated:
            print("调试: 用户未提供或未认证，无法获取自选股。")
            return []
        print(f"调试: 开始获取用户 {user.username} (ID: {user.id}) 的自选股数据...")
        try:
            # 1. 使用 filter(user=user) 筛选指定用户的记录
            # 2. select_related('stock') 同样用于性能优化
            # 3. 使用模型 Meta 中定义的默认排序 ['-is_pinned', '-added_at']
            user_favorites_query = FavoriteStock.objects.filter(user=user).select_related('stock')
            # 异步执行查询
            favorite_stock_objects = await sync_to_async(list)(user_favorites_query)
            if not favorite_stock_objects:
                print(f"调试: 用户 {user.username} 没有任何自选股记录。")
                return []
            print(f"调试: 成功为用户 {user.username} 获取 {len(favorite_stock_objects)} 条自选股记录。")
            # 转换为字典列表
            fav_datas = [
                {
                    "id": fav.id,
                    "stock_code": fav.stock.stock_code if fav.stock else None,
                    "stock_name": fav.stock.stock_name if fav.stock else None,
                    "added_at": fav.added_at,
                    "note": fav.note,
                    "is_pinned": fav.is_pinned,
                    "tags": fav.tags,
                }
                for fav in favorite_stock_objects
            ]
            return fav_datas
        except Exception as e:
            logger.error(f"为用户 {user.username} 获取自选股失败: {e}", exc_info=True)
            return None

    async def save_stocks(self) -> Dict:
        """
        【V2.1 健壮性增强版】通过tushare获取股票数据并保存到数据库
        - 核心优化: 使用Pandas的向量化操作替代了原有的 `itertuples()` 循环，大幅提升了数据处理效率。
        - 健壮性增强:
          1. 扩展了列名映射，确保API字段能正确对应模型字段。
          2. 在保存到数据库前，动态筛选DataFrame的列，只保留模型中存在的字段，避免因API返回多余字段（如'symbol'）导致程序崩溃。
          3. 修正了缓存列名，与重命名后的列名保持一致。
        """
        # 从Tushare API获取所有股票基本信息
        df = self.ts_pro.stock_basic(**{
            "ts_code": "", "name": "", "exchange": "", "market": "", "is_hs": "", "list_status": "", "limit": "", "offset": ""
        }, fields=[
            "ts_code", "symbol", "name", "area", "industry", "cnspell", "market", "list_date", "act_name", "act_ent_type",
            "fullname", "enname", "exchange", "curr_type", "list_status", "delist_date", "is_hs"
        ])
        if df is None or df.empty:
            return {}
        # --- 开始向量化处理 ---
        # 1. 数据清洗：将各种空值表示统一为None
        df.replace(['nan', 'NaN', ''], np.nan, inplace=True)
        df = df.where(pd.notnull(df), None)
        # 2. 向量化转换日期列
        df['list_date'] = pd.to_datetime(df['list_date'], format='%Y%m%d', errors='coerce').dt.date
        df['delist_date'] = pd.to_datetime(df['delist_date'], format='%Y%m%d', errors='coerce').dt.date
        # 3. 向量化重命名列以匹配模型字段
        # [代码修改处] 扩展rename_map以匹配所有模型字段
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
        # 4. 准备用于数据库和缓存的数据
        # [代码修改处] 获取模型所有字段名，用于筛选DataFrame的列，避免因API返回多余字段（如'symbol'）导致错误
        model_fields = [f.name for f in StockInfo._meta.get_fields()]
        # [代码修改处] 筛选出DataFrame中与模型字段匹配的列，用于数据库保存
        db_cols = [col for col in df.columns if col in model_fields]
        db_df = df[db_cols]
        stock_dicts = db_df.to_dict('records')
        # 缓存只需要部分基础列
        # [代码修改处] 修正缓存列名，将 'market' 改为 'market_type' 以匹配重命名后的列
        cache_cols = ['stock_code', 'stock_name', 'list_status', 'list_date', 'delist_date', 'exchange', 'market_type', 'is_hs', 'industry']
        cache_df = df[[col for col in cache_cols if col in df.columns]]
        cache_dicts = cache_df.to_dict('records')
        # --- 向量化处理结束，原有的itertuples()循环已被移除 ---
        # 5. 并发执行缓存写入任务
        cache_tasks = [self.stock_cache_set.stock_basic_info(d['stock_code'], d) for d in cache_dicts if d.get('stock_code')]
        await asyncio.gather(*cache_tasks)
        await self.stock_cache_set.all_stocks(cache_dicts)
        # 6. 批量保存到数据库
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
        【V2 - 优化版】通过tushare获取所有公司信息并保存到数据库
        优化点:
        1. [核心] 使用 `get_stocks_by_codes` 方法进行批量查询，彻底解决N+1数据库查询问题，性能提升巨大。
        2. 采用“先从API拉取，再批量关联数据库信息”的高效模式。
        3. 增加了对源数据（API返回数据）的完整性校验。
        4. 优化了代码结构，使其更清晰、高效和健壮。
        """
        print("调试: 开始执行 save_company_info 任务...")
        try:
            # 1. 一次性从Tushare拉取所有公司数据
            print("调试: 正在从Tushare API拉取所有上市公司基本信息...")
            df = self.ts_pro.stock_company(**{
                "ts_code": "", "exchange": "", "status": "L", # 通常只获取上市状态的公司
            }, fields=[
                "ts_code", "com_name", "chairman", "manager", "secretary", "reg_capital", "setup_date", "province",
                "city", "introduction", "website", "email", "office", "business_scope", "employees", "main_business", "exchange"
            ])
            if df.empty:
                logger.info("Tushare API没有返回任何公司信息，任务提前结束。")
                print("调试: Tushare API返回空数据帧，任务结束。")
                return {"status": "success", "message": "No data returned from API.", "saved_count": 0}
            # 2. 数据清洗
            df = df.replace(['nan', 'NaN', ''], np.nan)
            df = df.where(pd.notnull(df), None)
            # 3. [代码修改处] 批量获取所有相关的股票基础信息对象
            unique_ts_codes = df['ts_code'].unique().tolist()
            print(f"调试: 从API获取了 {len(df)} 条公司数据，涉及 {len(unique_ts_codes)} 个独立股票代码。")
            # [代码修改处] 使用 get_stocks_by_codes 方法，一次性查询数据库，解决N+1问题
            stock_map = await self.get_stocks_by_codes(unique_ts_codes)
            print(f"调试: 批量从数据库获取了 {len(stock_map)} 个股票对象。")
            # 4. 准备批量写入的数据
            data_dicts_to_save = []
            for row in df.itertuples():
                # [代码修改处] 从预先查好的映射中获取股票对象，高效且无N+1问题
                stock_instance = stock_map.get(row.ts_code)
                # 进行健壮性检查
                if stock_instance and row.com_name:
                    company_dict = self.data_format_process.set_company_info_data(stock_instance, row)
                    data_dicts_to_save.append(company_dict)
                else:
                    if not stock_instance:
                        logger.warning(f"在数据库中未找到股票代码 {row.ts_code} 的基础信息，已跳过该公司信息。")
                    else:
                        logger.warning(f"API返回的股票 {row.ts_code} 公司名称(com_name)为空，已跳过。")
            # 5. [代码修改处] 批量写入数据库
            saved_count = 0
            if data_dicts_to_save:
                print(f"调试: 准备批量保存 {len(data_dicts_to_save)} 条公司数据到数据库...")
                # [代码修改处] unique_fields 应该直接关联到 stock 对象，而不是 stock_code 字符串
                # 假设 _save_all_to_db_native_upsert 能处理外键对象
                result = await self._save_all_to_db_native_upsert(
                    model_class=StockCompany,
                    data_list=data_dicts_to_save,
                    unique_fields=['stock'] # 使用外键字段'stock'作为唯一约束
                )
                saved_count = len(data_dicts_to_save)
                logger.info(f"成功批量保存 {saved_count} 条公司信息。")
                print(f"调试: 成功保存 {saved_count} 条数据。")
                return result # 返回upsert的结果
            else:
                logger.info("经过筛选后，没有需要保存到数据库的公司数据。")
                print("调试: 经过筛选后，没有需要保存的数据。")
                return {"status": "success", "message": "No new data to save.", "saved_count": 0}
        except Exception as e:
            logger.error(f"保存公司信息时发生严重错误: {e}", exc_info=True)
            print(f"调试: 发生异常: {e}")
            raise

    async def save_hs_const(self) -> Dict:
        """
        【V2.0 向量化与N+1优化版】通过tushare获取沪深港通成分股信息并保存到数据库
        - 核心优化:
          1. 【消除N+1查询】通过一次性批量获取所有 `StockInfo` 对象，彻底解决了原先在循环中频繁查询数据库的性能瓶颈。
          2. 【向量化处理】使用Pandas的向量化操作替代了原有的 `itertuples()` 循环，大幅提升了数据处理效率。
        """
        # 获取沪股通和深股通成分
        df_sh = self.ts_pro.hs_const(hs_type='SH')
        df_sz = self.ts_pro.hs_const(hs_type='SZ')
        # 纵向合并
        df = pd.concat([df_sh, df_sz], ignore_index=True)
        if df is None or df.empty:
            return {}
        # --- 开始向量化处理 ---
        # 1. 数据清洗
        df.replace(['nan', 'NaN', ''], np.nan, inplace=True)
        df.dropna(subset=['ts_code'], inplace=True)
        if df.empty:
            return {}
        # 2. 批量获取关联的StockInfo对象 (消除N+1查询)
        unique_ts_codes = df['ts_code'].unique().tolist()
        stock_map = await self.get_stocks_by_codes(unique_ts_codes)
        # 3. 向量化映射、转换和选择
        df['stock'] = df['ts_code'].map(stock_map)
        df.dropna(subset=['stock'], inplace=True)
        if df.empty:
            logger.warning("所有沪深港通成分股都无法关联到已知的股票信息，任务终止。")
            return {}
        df['in_date'] = pd.to_datetime(df['in_date'], format='%Y%m%d', errors='coerce').dt.date
        df['out_date'] = pd.to_datetime(df['out_date'], format='%Y%m%d', errors='coerce').dt.date
        df.rename(columns={'hs_type': 'hs_type_code'}, inplace=True)
        final_df = df[['stock', 'in_date', 'out_date', 'is_new', 'hs_type_code']]
        # 4. 转换为字典列表
        stock_dicts = final_df.where(pd.notnull(final_df), None).to_dict('records')
        # --- 向量化处理结束，原有的itertuples()循环和N+1查询已被移除 ---
        if stock_dicts:
            # 批量保存到数据库
            result = await self._save_all_to_db_native_upsert(
                model_class=HSConst,
                data_list=stock_dicts,
                unique_fields=['stock'] # 优化：直接使用stock对象作为唯一键
            )
            return result
        return {}




















