from api_manager.apis.stock_indicators_api import StockIndicatorsAPI
from dao_manager.base_dao import BaseDAO
from dao_manager.daos.stock_basic_dao import StockBasicDAO
from dao_manager.daos.user_dao import UserDAO
from utils.cash_key import StockCashKey
from utils.data_format_process import StockIndicatorsDataFormatProcess

class StockTimeTradeDAO(BaseDAO):
    def __init__(self):
        """初始化StockIndicatorsDAO"""
        super().__init__(None, None, 3600)  # 基类使用None作为model_class，因为本DAO管理多个模型
        self.api = StockIndicatorsAPI()
        self.stock_basic_dao = StockBasicDAO()
        self.cache_timeout = 300  # 默认缓存5分钟
        self.cache_limit = 333 # 定义缓存数量上限
        self.user_dao = UserDAO()
        self.cache_key = StockCashKey()
        self.data_format_process = StockIndicatorsDataFormatProcess()
        self.cache_manager = None
        self.cache_get = None
        self.cache_set = None
