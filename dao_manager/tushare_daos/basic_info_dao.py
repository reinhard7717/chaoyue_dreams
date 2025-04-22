
import logging
from asgiref.sync import sync_to_async
from typing import Dict, List, Any, Optional
from utils import cache_constants as cc
from dao_manager.base_dao import BaseDAO
from stock_models.stock_basic import CompanyInfo, StockInfo
from utils.cache_get import UserCacheGet
from utils.cache_set import UserCacheSet
from utils.data_format_process import StockInfoFormatTuShare

logger = logging.getLogger("dao")

class BasicInfoDao(BaseDAO):
    def __init__(self):
        super().__init__(None, None, 3600)
        self.data_format_process = StockInfoFormatTuShare()
        from utils.cache_get import StockInfoCacheGet
        from utils.cash_key import StockCashKey
        from utils.data_format_process import StockInfoFormatProcess
        from utils.cache_manager import CacheManager
        from api_manager.apis.stock_basic_api import StockBasicAPI
        self.api = StockBasicAPI()
        self.cache_manager = CacheManager()  # 初始化缓存管理器
        self.data_format_process = StockInfoFormatTuShare()
        self.cache_key = StockCashKey()
        self.cache_get = StockInfoCacheGet()
        self.user_cache_set = UserCacheSet()
        self.user_cache_get = UserCacheGet()

    async def get_stock_list(self) -> List['StockInfo']:
        """
        获取所有股票的基本信息
        
        Returns:
            List[StockInfo]: 股票基本信息列表（已过滤掉 stock_name 中包含“退”或“债”字的股票）
            过滤逻辑：如果 stock_name 中包含“退”字或“债”字（或的关系），则排除。
        """
        from stock_models.stock_basic import StockInfo
        try:
            # 尝试从缓存获取
            cached_data = await self.cache_get.all_stocks()
            if cached_data:
                # 将缓存数据转换为模型实例列表
                return_data = [StockInfo(**stock_dict) for stock_dict in cached_data]
                # 过滤掉 stock_name 中包含“退”或“债”字的股票
                # 解释：仅保留不包含“退”并且不包含“债”的股票，等价于排除包含“退”或“债”
                filtered_data = [item for item in return_data if '退' not in item.stock_name and '债' not in item.stock_name]
                # 排序
                sorted_data = sorted(filtered_data, key=lambda x: x.stock_code)
                return sorted_data  # 返回过滤并排序后的列表
        except Exception as e:
            logger.error(f"从缓存获取股票列表失败: {e}",exc_info=True)
        
        stocks = []
        try:
            # 从数据库读取
            get_stocks_sync = sync_to_async(
                lambda: list(StockInfo.objects.order_by('stock_code')),
                thread_sensitive=True  # 对于 ORM 操作，通常建议设置为 True
            )
            stocks = await get_stocks_sync()
            if stocks:
                # 过滤掉 stock_name 中包含“退”或“债”字的股票
                filtered_stocks = [item for item in stocks if '退' not in item.stock_name and '债' not in item.stock_name]
                # 缓存过滤后的股票列表
                await self.set_cache_stocks(filtered_stocks)
                # 排序
                sorted_stocks = sorted(filtered_stocks, key=lambda x: x.stock_code)
                return sorted_stocks
        except Exception as e:
            logger.error(f"从数据库读取股票列表失败: {e}")
        
        # 如果数据库中没有数据，从API获取并保存
        logger.info("数据库中没有股票数据，从API获取")
        await self.fetch_and_save_stocks()
        # 从数据库读取
        get_stocks_sync = sync_to_async(
            lambda: list(StockInfo.objects.filter(list_status='L').order_by('stock_code')),
            thread_sensitive=True
        )
        stocks = await get_stocks_sync()
        # 过滤掉 stock_name 中包含“退”或“债”字的股票
        filtered_stocks = [item for item in stocks if '退' not in item.stock_name and '债' not in item.stock_name]
        # 排序
        sorted_stocks = sorted(filtered_stocks, key=lambda x: x.stock_code)
        return sorted_stocks

    async def get_stock_by_code(self, stock_code: str) -> Optional['StockInfo']:
        """
        根据股票代码获取股票信息
        Args:
            stock_code: 股票代码
        Returns:
            Optional[StockInfo]: 股票信息
        """
        from stock_models.stock_basic import StockInfo
        stock = StockInfo(stock_code="", stock_name="", exchange="")
        # 使用CacheManager生成标准化缓存键
        cache_key = self.cache_key.stock_data(stock_code)
        # 尝试从缓存获取，指定模型类进行自动转换
        stock = await self.cache_manager.get_model(cache_key, StockInfo)
        if stock:
            return stock
            
        # 从数据库获取
        # logger.info(f"get_stock_by_code从数据库获取股票: {cache_key}, {stock_code}")
        stock = await sync_to_async(lambda: StockInfo.objects.filter(stock_code=stock_code).first())()
        # 如果数据库中有数据，缓存并返回
        if stock:
            cache_data = self.data_format_process.set_stock_info_data(stock)
            # logger.info(f"get_stock_by_code,cache_data: {cache_data}, type: {type(cache_data)}")
            # *** 正确调用 CacheManager 缓存数据 ***
            success = await self.cache_manager.set(
                key=cache_key,
                data=cache_data,
                timeout=self.cache_manager.get_timeout('st')
            )
            # logger.info(f"get_stock_by_code,success: {success}")
            return stock

    async def get_favorite_stocks_by_user(self, user: 'User') -> List['FavoriteStock']:  
        """
        获取用户自选股
        Args:
            user: 用户
        """
        from django.contrib.auth.models import User
        from users.models import FavoriteStock
        # 从缓存获取
        fav_datas = []
        # fav_datas = await self.user_cache_get.user_favorites(user.id)
        # if fav_datas:
        #     return fav_datas
        # 从数据库获取
        items = FavoriteStock.objects.filter(user=user)
        for item in items:
            fav_data = self.user_data_format_process.set_user_favorites(user.id, item)
            fav_datas.append(fav_data)
            await self.user_cache_set.user_favorites(user.id, item)
        return fav_datas

    async def get_all_favorite_stocks(self) -> Optional[List[Dict]]:
        """
        获取所有自选股
        """
        from users.models import FavoriteStock
        fav_datas = []
        
        try:
            # 使用 sync_to_async 包装 ORM 查询
            items = await sync_to_async(list)(FavoriteStock.objects.all())
            for item in items:
                # 原逻辑中 self.user_data_format_process.set_user_favorites(user.id, item) 依赖 user.id，
                # 但这里没有 user 参数。这可能是错误。假设这是一个通用的格式化操作，
                # 这里直接使用 item 对象或一个通用的格式化方法。如果有特定方法可用，请替换。
                # 为保持业务逻辑，我假设格式化是可选的，先直接使用 item，然后缓存。
                # 如果 self.user_data_format_process 有通用方法，可以在这里调整。
                fav_data = item  # 临时修正：直接使用模型对象，避免 user.id 错误
                # 如果需要严格保持原逻辑，且有通用格式化方法，可以添加：fav_data = self.data_format_process.set_favorite_stock(item)
                fav_datas.append(fav_data)  # fav_data 现在是 FavoriteStock 对象
                # await self.user_cache_set.all_favorites(item)  # 保持缓存逻辑不变
        except Exception as e:
            logger.error(f"从数据库获取所有自选股失败: {e}")
            return None  # 返回 None 表示失败
        
        return fav_datas  # 返回列表，包含 FavoriteStock 对象

    async def save_stocks(self, stock_code: str) -> Optional['StockInfo']:
        """
        通过tushare获取股票数据并保存到数据库
        """
        from stock_models.stock_basic import StockInfo
        stock_dicts = []
        df = self.ts_pro.stock_basic(**{
            "ts_code": "", "name": "", "exchange": "", "market": "", "is_hs": "", "list_status": "", "limit": "", "offset": ""
        }, fields=[
            "ts_code", "symbol", "name", "area", "industry", "cnspell", "market", "list_date", "act_name", 
            "act_ent_type", "fullname", "exchange", "list_status"
        ])
         # 使用CacheManager生成标准化缓存键
        cache_key = self.cache_key.stock_data(stock_code)
        cache_timeout = self.cache_manager.get_timeout(cc.TYPE_STATIC)
        for row in df.itertuples():
            stock_dict = self.data_format_process.set_stock_info_data(row)
            stock_dicts.append(stock_dict)
            await self.cache_manager.set(key=cache_key, data=stock_dict, timeout=self.cache_manager.get_timeout('st'))
        if stock_dicts is not None:
            result = await self._save_all_to_db_native_upsert(
                    model_class=StockInfo,
                    data_list=stock_dicts,
                    unique_fields=['stock_code'] # ORM 能处理 stock 实例
                )
        return result

    async def save_company_info(self, stock_code: str) -> Optional['CompanyInfo']:
        """
        通过tushare获取公司信息并保存到数据库
        """
        # 拉取数据
        df = self.ts_pro.stock_company(**{
            "ts_code": "", "exchange": "", "status": "", "limit": "", "offset": ""
        }, fields=["ts_code", "com_name", "com_id", "chairman", "manager", "secretary", "reg_capital", "setup_date", "province",
            "city", "introduction", "website", "email", "office", "business_scope", "employees", "main_business", "exchange"
        ])
        company_dicts = []
        for row in df.itertuples():
            company_info = {

            }
            print(row.ts_code, row.com_name, row.chairman)
        # 使用CacheManager生成标准化缓存键
        cache_key = self.cache_key.generate_key('st', 'company', stock_code)


















