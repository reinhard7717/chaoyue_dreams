
import logging
from dao_manager.base_dao import BaseDAO
from stock_models.index import IndexInfo
from utils.cache_get import IndexCacheGet
from utils.cache_set import IndexCacheSet
from utils.data_format_process import IndexDataFormatProcess


logger = logging.getLogger("dao")

class IndexDAO(BaseDAO):
    def __init__(self):
        super().__init__(None, None, 3600)
        self.data_format_process = IndexDataFormatProcess()
        self.index_cache_set = IndexCacheSet()
        self.index_cache_get = IndexCacheGet()

    async def get_index_list(self) -> List['IndexInfo']:
        """
        获取所有指数的基本信息
        Returns:
            List[StockInfo]: 指数基本信息列表
        """
        return_data = []
        # 先从缓存中获取
        index_list = await self.index_cache_get.all_indexes()
        if index_list:
            for index_dict in index_list:
                return_data.append(index_dict)
            return return_data
        # 从数据库获取
        # logger.info(f"get_stock_by_code从数据库获取股票: {cache_key}, {stock_code}")
        index_list = await sync_to_async(lambda: list(IndexInfo.objects.all()))()
        # 如果数据库中有数据，缓存并返回
        if index_list:
            data_to_cache = []
            for index in index_list:
                index_dict = self.data_format_process.set_index_info_data(index)
                data_to_cache.append(index_dict)
                await self.index_cache_set.index_info(index.index_code, index_dict)
            await self.index_cache_set.all_indexes(data_to_cache)
        return return_data

    async def get_index_info(self, index_code) -> Optional['IndexInfo']:
        """
        获得指数信息
        Args
        """
        pass

    async def save_index_info(self, index_code):
        """
        保存指数信息到数据库
        接口：index_basic，可以通过数据工具调试和查看数据。
        描述：获取指数基础信息。
        """
        # 拉取数据
        df = self.ts_pro.index_basic(**{
            "ts_code": "", "market": "", "publisher": "", "category": "", "name": "", "limit": "", "offset": ""
        }, fields=[
            "ts_code", "name", "market", "publisher", "category", "base_date", "base_point", "list_date",
            "fullname", "index_type", "weight_rule", "desc", "exp_date"
        ])
        index_dicts = []
        if df is not None:
            df = df.replace(['nan', 'NaN', ''], None)  # 先把字符串nan等变成None
            for row in df.itertuples():
                index_dict = self.data_format_process.set_index_info_data(row)
                index_dicts.append(index_dict)
                await self.index_cache_set.index_info(row.ts_code, index_dict)



