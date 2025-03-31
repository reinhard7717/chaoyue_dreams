import logging
import asyncio
from typing import Dict, List, Any, Optional, Union, Set, Tuple, Type
from datetime import datetime, date
from decimal import Decimal

from django.db import transaction
from django.core.cache import cache
from asgiref.sync import sync_to_async
from django.db.models import Q
from django.db.models.base import Model

from api_manager.apis.stock_indicators_api import StockIndicatorsAPI, TimeLevel
from api_manager.mappings.stock_indicators_mapping import BOLL_INDICATOR_MAPPING, KDJ_INDICATOR_MAPPING, MA_INDICATOR_MAPPING, MACD_INDICATOR_MAPPING, TIME_TRADE_MAPPING
from dao_manager.base_dao import BaseDAO
from dao_manager.daos.stock_basic_dao import StockBasicDAO
from dao_manager.daos.user_dao import UserDAO
from stock_models.stock_indicators import StockBOLLIndicator, StockKDJIndicator, StockMACDIndicator, StockMAIndicator, StockTimeTrade
from utils.cache_manager import CacheManager

logger = logging.getLogger(__name__)

TIME_LEVELS = ['5','15','30','60','Day','Day_qfq','Day_hfq','Week','Week_qfq','Week_hfq','Month','Month_qfq','Month_hfq','Year','Year_qfq','Year_hfq']

class StockIndicatorsDAO(BaseDAO):
    """
    股票技术指标DAO，整合所有相关的技术指标访问功能
    """
    
    def __init__(self):
        """初始化StockIndicatorsDAO"""
        super().__init__(None, None, 3600)  # 基类使用None作为model_class，因为本DAO管理多个模型
        self.api = StockIndicatorsAPI()
        self.cache_manager = CacheManager()
        self.stock_basic_dao = StockBasicDAO()
        self.cache_timeout = 300  # 默认缓存5分钟
        self.user_dao = UserDAO()

        logger.info("初始化StockIndicatorsDAO")
    
    # ================= 分时成交数据相关方法 =================
    
    async def get_latest_time_trade(self, stock_code: str, time_level: Union[TimeLevel, str]) -> Optional[StockTimeTrade]:
        """
        获取最新的分时成交数据
        
        Args:
            stock_code: 股票代码
            time_level: 时间级别
            
        Returns:
            Optional[StockTimeTrade]: 最新的分时成交数据
        """
        return await self._get_latest_indicator(
            model_class=StockTimeTrade,
            stock_code=stock_code,
            time_level=time_level,
            api_method=self.api.get_time_trade_data,
            mapping=TIME_TRADE_MAPPING,
            cache_prefix="time_trade"
        )
    
    # 还需要修改
    async def get_favorite_stocks_latest_time_trade(self) -> Optional[StockTimeTrade]:
        """
        获取自选股最新分时成交数据
        
        Returns:
            Optional[StockTimeTrade]: 自选股最新分时成交数据
        """
        # 获取自选股
        favorite_stocks = await self.user_dao.get_all_favorite_stocks()
        # 获取自选股最新分时成交数据
        for stock in favorite_stocks:
            # 使用CacheManager生成标准化缓存键
            cache_key = self.cache_manager.generate_key('st', 'stock_time_trade', stock.stock_code)
            cached_data = self.cache_manager.get(cache_key)
            if cached_data:
                return cached_data
        # 从数据库中获取    
        stock_time_trade = await sync_to_async(StockTimeTrade.objects.filter)(stock_code=stock.stock_code, time_level='Day')
        if stock_time_trade:
            # 序列化对象列表
            serialized_stock_time_trade = self._serialize_model(stock_time_trade)
            # 使用CacheManager缓存数据
            self.cache_manager.set(cache_key, serialized_stock_time_trade, timeout=self.cache_manager.get_timeout('st'))
            return serialized_stock_time_trade
        return None

    async def get_history_time_trades(self, stock_code: str, time_level: Union[TimeLevel, str], 
                                    limit: int = 1000) -> List[StockTimeTrade]:
        """
        获取历史分时成交数据
        
        Args:
            stock_code: 股票代码
            time_level: 时间级别
            limit: 返回记录数量限制
            
        Returns:
            List[StockTimeTrade]: 历史分时成交数据列表
        """
        return await self._get_history_indicators(
            model_class=StockTimeTrade,
            stock_code=stock_code,
            time_level=time_level,
            api_method=self.api.get_time_trade_data,
            mapping=TIME_TRADE_MAPPING,
            cache_prefix="time_trade",
            limit=limit
        )
    
    async def fetch_and_save_latest_time_trade(self, stock_code: str, time_level: Union[TimeLevel, str]) -> Optional[StockTimeTrade]:
        """
        从API获取并保存最新分时成交数据
        
        Args:
            stock_code: 股票代码
            time_level: 时间级别
            
        Returns:
            Optional[StockTimeTrade]: 保存的数据
        """
        # 获取股票信息
        stock = await self.stock_basic_dao.get_stock_by_code(stock_code)
        if not stock:
            return []
        try:
            api_data = await self.api.get_time_trade(stock.stock_code, time_level)
                
            if not api_data:
                logger.warning(f"API未返回{stock}的{time_level}级别时间序列数据")
                return []
            
            data_dicts = []
            data_dict = {
                'stock': stock,
                'time_level': time_level,
                'trade_time': self._parse_datetime(api_data.get('d')),  # 交易时间
                'open_price': self._parse_number(api_data.get('o')),  # 开盘价
                'high_price': self._parse_number(api_data.get('h')),  # 最高价
                'low_price': self._parse_number(api_data.get('l')),  # 最低价
                'close_price': self._parse_number(api_data.get('c')),  # 收盘价
                'volume': self._parse_number(api_data.get('v')),  # 成交量
                'turnover': self._parse_number(api_data.get('e')),  # 成交额
                'amplitude': self._parse_number(api_data.get('zf')),  # 振幅
                'turnover_rate': self._parse_number(api_data.get('hs')),  # 换手率
                'price_change_percent': self._parse_number(api_data.get('zd')),  # 涨跌幅
                'price_change_amount': self._parse_number(api_data.get('zde')),  # 涨跌额   
            }
            data_dicts.append(data_dict)

            # 保存数据
            logger.info(f"开始保存{stock}股票{time_level}级别分时成交数据")
            result = await self._save_all_to_db(
                model_class=StockTimeTrade,
                data_list=data_dicts,
                unique_fields=['stock', 'time_level', 'trade_time']
            )
            
            logger.info(f"{stock}股票{time_level}级别分时成交数据保存完成，结果: {result}")
            return result
        except Exception as e:
            logger.error(f"保存{stock}股票{time_level}级别  分时成交数据出错: {str(e)}")
            logger.debug(f"错误数据内容: {data_dicts if 'data_dicts' in locals() else '未获取到数据'}")
            return {'创建': 0, '更新': 0, '跳过': 0}

    async def fetch_and_save_latest_time_trade_by_stock_code(self, stock_code: str) -> Optional[StockTimeTrade]:
        """
        从API获取并保存最新股票分时成交数据
        
        Args:
            stock_code: 股票代码
        
        Returns:
            Optional[StockTimeTrade]: 保存的数据
        """
        # 获取股票信息
        stock = await self.stock_basic_dao.get_stock_by_code(stock_code)
        if not stock:
            logger.warning(f"股票代码[{stock_code}]不存在，无法获取时间序列数据")
            return []
        try:
            data_dicts = []
            for time_level in TIME_LEVELS:
                api_data = await self.api.get_time_trade(stock.stock_code, time_level)
                data_dict = {
                    'stock': stock,
                    'time_level': time_level,
                    'trade_time': self._parse_datetime(api_data.get('d')),  # 交易时间
                    'open_price': self._parse_number(api_data.get('o')),  # 开盘价
                    'high_price': self._parse_number(api_data.get('h')),  # 最高价
                    'low_price': self._parse_number(api_data.get('l')),  # 最低价
                    'close_price': self._parse_number(api_data.get('c')),  # 收盘价
                    'volume': self._parse_number(api_data.get('v')),  # 成交量
                    'turnover': self._parse_number(api_data.get('e')),  # 成交额
                    'amplitude': self._parse_number(api_data.get('zf')),  # 振幅
                    'turnover_rate': self._parse_number(api_data.get('hs')),  # 换手率
                    'price_change_percent': self._parse_number(api_data.get('zd')),  # 涨跌幅
                    'price_change_amount': self._parse_number(api_data.get('zde')),  # 涨跌额   
                }
                data_dicts.append(data_dict)
            if not data_dicts:
                logger.warning(f"API未返回{stock}股票的{time_level}级别时间序列数据")
                return []
            
            # 保存数据
            logger.info(f"开始保存{stock}股票分时成交数据")
            result = await self._save_all_to_db(
                model_class=StockTimeTrade,
                data_list=data_dicts,
                unique_fields=['stock', 'time_level', 'trade_time']
            )
            
            logger.info(f"{stock}股票分时成交数据保存完成，结果: {result}")
            return result
        except Exception as e:
            logger.error(f"保存{stock}股票分时成交数据出错: {str(e)}")
            logger.debug(f"错误数据内容: {data_dicts if 'data_dicts' in locals() else '未获取到数据'}")
            return {'创建': 0, '更新': 0, '跳过': 0}

    async def fetch_and_save_latest_time_trade_by_time_level(self, time_level: str) -> Optional[StockTimeTrade]:
        """
        从API获取并保存最新股票分时成交数据
        
        Args:
            time_level: 时间级别
        """
        stocks = await self.stock_basic_dao.get_stock_list()
        try:
            data_dicts = []
            for stock in stocks:
                for time_level in TIME_LEVELS:
                    api_data = await self.api.get_time_trade(stock.stock_code, time_level)
                    data_dict = {
                        'stock': stock,
                        'time_level': time_level,
                        'trade_time': self._parse_datetime(api_data.get('d')),  # 交易时间
                        'open_price': self._parse_number(api_data.get('o')),  # 开盘价
                        'high_price': self._parse_number(api_data.get('h')),  # 最高价
                        'low_price': self._parse_number(api_data.get('l')),  # 最低价
                        'close_price': self._parse_number(api_data.get('c')),  # 收盘价
                        'volume': self._parse_number(api_data.get('v')),  # 成交量
                        'turnover': self._parse_number(api_data.get('e')),  # 成交额
                        'amplitude': self._parse_number(api_data.get('zf')),  # 振幅
                        'turnover_rate': self._parse_number(api_data.get('hs')),  # 换手率
                        'price_change_percent': self._parse_number(api_data.get('zd')),  # 涨跌幅
                        'price_change_amount': self._parse_number(api_data.get('zde')),  # 涨跌额   
                    }
                    data_dicts.append(data_dict)
            if not data_dicts:
                logger.warning(f"API未返回{time_level}级别的所有股票时间序列数据")
                return []
            
            # 保存数据
            logger.info("开始保存自选股分时成交数据")
            result = await self._save_all_to_db(
                model_class=StockTimeTrade,
                data_list=data_dicts,
                unique_fields=['stock', 'time_level', 'trade_time']
            )
            
            logger.info(f"自选股分时成交数据保存完成，结果: {result}")
            return result
        except Exception as e:
            logger.error(f"保存自选股分时成交数据出错: {str(e)}")
            logger.debug(f"错误数据内容: {data_dicts if 'data_dicts' in locals() else '未获取到数据'}")
            return {'创建': 0, '更新': 0, '跳过': 0}

    async def fetch_and_save_favorite_stocks_latest_time_trade(self) -> Optional[StockTimeTrade]:
        """
        从API获取并保存最新自选股分时成交数据
            
        Returns:
            Optional[StockTimeTrade]: 保存的数据
        """
        favorite_stocks = await self.user_dao.get_all_favorite_stocks()
        try:
            data_dicts = []
            for stock in favorite_stocks:
                for time_level in TIME_LEVELS:
                    api_data = await self.api.get_time_trade(stock.stock_code, time_level)
                    data_dict = {
                        'stock': stock,
                        'time_level': time_level,
                        'trade_time': self._parse_datetime(api_data.get('d')),  # 交易时间
                        'open_price': self._parse_number(api_data.get('o')),  # 开盘价
                        'high_price': self._parse_number(api_data.get('h')),  # 最高价
                        'low_price': self._parse_number(api_data.get('l')),  # 最低价
                        'close_price': self._parse_number(api_data.get('c')),  # 收盘价
                        'volume': self._parse_number(api_data.get('v')),  # 成交量
                        'turnover': self._parse_number(api_data.get('e')),  # 成交额
                        'amplitude': self._parse_number(api_data.get('zf')),  # 振幅
                        'turnover_rate': self._parse_number(api_data.get('hs')),  # 换手率
                        'price_change_percent': self._parse_number(api_data.get('zd')),  # 涨跌幅
                        'price_change_amount': self._parse_number(api_data.get('zde')),  # 涨跌额   
                    }
                    data_dicts.append(data_dict)
            if not data_dicts:
                logger.warning(f"API未返回自选股（get_all_favorite_stocks）的{time_level}级别时间序列数据")
                return []
            # 保存数据
            logger.info("开始保存自选股分时成交数据")
            result = await self._save_all_to_db(
                model_class=StockTimeTrade,
                data_list=data_dicts,
                unique_fields=['stock', 'time_level', 'trade_time']
            )
            
            logger.info(f"自选股分时成交数据保存完成，结果: {result}")
            return result
        except Exception as e:
            logger.error(f"保存自选股分时成交数据出错: {str(e)}")
            logger.debug(f"错误数据内容: {data_dicts if 'data_dicts' in locals() else '未获取到数据'}")
            return {'创建': 0, '更新': 0, '跳过': 0}

    async def fetch_and_save_all_latest_time_trade(self) -> Optional[StockTimeTrade]:
        """
        从API获取并保存所有最新股票分时成交数据
        """
        stocks = await self.stock_basic_dao.get_stock_list()
        for stock in stocks:
            await self.fetch_and_save_latest_time_trade_by_stock_code(stock.stock_code)
        return {'创建': 0, '更新': 0, '跳过': 0}

    async def fetch_and_save_history_time_trade(self, stock_code: str, time_level: Union[TimeLevel, str], limit: int = 1000) -> Optional[StockTimeTrade]:
        """
        从API获取并保存历史股票分时成交数据
        
        Args:
            stock_code: 股票代码
            time_level: 时间级别
            limit: 返回记录数量限制
            
        Returns:
            Optional[StockTimeTrade]: 保存的数据
        """
        # 获取股票信息
        stock = await self.stock_basic_dao.get_stock_by_code(stock_code)
        if not stock:
            return []
        try:
            api_datas = await self.api.get_history_trade(stock.stock_code, time_level)
                
            if not api_datas:
                logger.warning(f"API未返回{stock}的{time_level}级别历史时间序列数据")
                return []
            
            data_dicts = []
            for api_data in api_datas:
                data_dict = {
                    'stock': stock,
                    'time_level': time_level,
                    'trade_time': self._parse_datetime(api_data.get('d')),  # 交易时间
                    'open_price': self._parse_number(api_data.get('o')),  # 开盘价
                    'high_price': self._parse_number(api_data.get('h')),  # 最高价
                    'low_price': self._parse_number(api_data.get('l')),  # 最低价
                    'close_price': self._parse_number(api_data.get('c')),  # 收盘价
                    'volume': self._parse_number(api_data.get('v')),  # 成交量
                    'turnover': self._parse_number(api_data.get('e')),  # 成交额
                    'amplitude': self._parse_number(api_data.get('zf')),  # 振幅
                    'turnover_rate': self._parse_number(api_data.get('hs')),  # 换手率
                    'price_change_percent': self._parse_number(api_data.get('zd')),  # 涨跌幅
                    'price_change_amount': self._parse_number(api_data.get('zde')),  # 涨跌额   
                }
                data_dicts.append(data_dict)

            # 保存数据
            logger.info(f"开始保存{stock}股票{time_level}级别历史分时成交数据")
            result = await self._save_all_to_db(
                model_class=StockTimeTrade,
                data_list=data_dicts,
                unique_fields=['stock', 'time_level', 'trade_time']
            )
            
            logger.info(f"{stock}股票{time_level}级别历史分时成交数据保存完成，结果: {result}")
            return result
        except Exception as e:
            logger.error(f"保存{stock}股票{time_level}级别历史分时成交数据出错: {str(e)}")
            logger.debug(f"错误数据内容: {data_dicts if 'data_dicts' in locals() else '未获取到数据'}")
            return {'创建': 0, '更新': 0, '跳过': 0}

    async def fetch_and_save_history_time_trade_by_stock_code(self, stock_code: str) -> Optional[StockTimeTrade]:
        """
        从API获取并保存历史股票分时成交数据
        
        Args:
            stock_code: 股票代码
        """
        # 获取股票信息
        stock = await self.stock_basic_dao.get_stock_by_code(stock_code)
        if not stock:
            return []
        try:
            data_dicts = []
            total_result = {'创建': 0, '更新': 0, '跳过': 0}
            for time_level in TIME_LEVELS:
                api_datas = await self.api.get_history_trade(stock.stock_code, time_level)
                for api_item in api_datas:
                    data_dict = {
                        'stock': stock,
                        'time_level': api_item.get('t'),
                        'trade_time': self._parse_datetime(api_item.get('d')),  # 交易时间
                        'open_price': self._parse_number(api_item.get('o')),  # 开盘价
                        'high_price': self._parse_number(api_item.get('h')),  # 最高价
                        'low_price': self._parse_number(api_item.get('l')),  # 最低价
                        'close_price': self._parse_number(api_item.get('c')),  # 收盘价
                        'volume': self._parse_number(api_item.get('v')),  # 成交量
                        'turnover': self._parse_number(api_item.get('e')),  # 成交额
                        'amplitude': self._parse_number(api_item.get('zf')),  # 振幅
                        'turnover_rate': self._parse_number(api_item.get('hs')),  # 换手率
                        'price_change_percent': self._parse_number(api_item.get('zd')),  # 涨跌幅
                        'price_change_amount': self._parse_number(api_item.get('zde')),  # 涨跌额   
                    }
                    data_dicts.append(data_dict)
                # 当数据量超过10万时，保存一次
                if len(data_dicts) >= 100000:
                    logger.info(f"数据量达到{len(data_dicts)}，开始保存批次数据")
                    batch_result = await self._save_all_to_db(
                        model_class=StockTimeTrade,
                        data_list=data_dicts,
                        unique_fields=['stock', 'time_level', 'trade_time']
                    )
                    logger.info(f"批次数据保存完成，结果: {batch_result}")
                    # 累加结果
                    for key in total_result:
                        total_result[key] += batch_result.get(key, 0)
                    # 清空数据列表，准备下一批
                    data_dicts = []
                logger.warning(f"当前data_dicts总量: {len(data_dicts)}")
            if not api_datas:
                logger.warning(f"API未返回{stock.stock_code}股票的{time_level}级别历史分时成交数据")
            
            # 保存剩余数据
            if data_dicts:
                logger.info(f"开始保存剩余{len(data_dicts)}条数据")
                final_result = await self._save_all_to_db(
                    model_class=StockTimeTrade,
                    data_list=data_dicts,
                    unique_fields=['stock', 'time_level', 'trade_time']
                )
                logger.info(f"剩余数据保存完成，结果: {final_result}")
                
                # 累加最终结果
                for key in total_result:
                    total_result[key] += final_result.get(key, 0)
            
            logger.info(f"所有股票历史KDJ指标数据保存完成，总结果: {total_result}")
            return total_result
        except Exception as e:
            logger.error(f"保存{stock}股票分时成交数据出错: {str(e)}")
            logger.debug(f"错误数据内容: {data_dicts if 'data_dicts' in locals() else '未获取到数据'}")
            return {'创建': 0, '更新': 0, '跳过': 0}

    async def fetch_and_save_history_time_trade_by_time_level(self, time_level: str) -> Optional[StockTimeTrade]:
        """
        从API获取并保存历史股票分时成交数据
        
        Args:
            time_level: 时间级别
        """
        stocks = await self.stock_basic_dao.get_stock_list()
        try:
            data_dicts = []
            total_result = {'创建': 0, '更新': 0, '跳过': 0}
            for stock in stocks:
                for time_level in TIME_LEVELS:
                    api_datas = await self.api.get_history_trade(stock.stock_code, time_level)
                    for api_item in api_datas:
                        data_dict = {
                            'stock': stock,
                            'time_level': api_item.get('t'),
                            'trade_time': self._parse_datetime(api_item.get('d')),  # 交易时间
                            'open_price': self._parse_number(api_item.get('o')),  # 开盘价
                            'high_price': self._parse_number(api_item.get('h')),  # 最高价
                            'low_price': self._parse_number(api_item.get('l')),  # 最低价
                            'close_price': self._parse_number(api_item.get('c')),  # 收盘价
                            'volume': self._parse_number(api_item.get('v')),  # 成交量
                            'turnover': self._parse_number(api_item.get('e')),  # 成交额
                            'amplitude': self._parse_number(api_item.get('zf')),  # 振幅
                            'turnover_rate': self._parse_number(api_item.get('hs')),  # 换手率
                            'price_change_percent': self._parse_number(api_item.get('zd')),  # 涨跌幅
                            'price_change_amount': self._parse_number(api_item.get('zde')),  # 涨跌额   
                        }
                        data_dicts.append(data_dict)
                    # 当数据量超过10万时，保存一次
                    if len(data_dicts) >= 100000:
                        logger.info(f"数据量达到{len(data_dicts)}，开始保存批次数据")
                        batch_result = await self._save_all_to_db(
                            model_class=StockTimeTrade,
                            data_list=data_dicts,
                            unique_fields=['stock', 'time_level', 'trade_time']
                        )
                        logger.info(f"批次数据保存完成，结果: {batch_result}")
                        # 累加结果
                        for key in total_result:
                            total_result[key] += batch_result.get(key, 0)
                        # 清空数据列表，准备下一批
                        data_dicts = []
                    logger.warning(f"当前data_dicts总量: {len(data_dicts)}")
                if not api_datas:
                    logger.warning(f"API未返回{stock.stock_code}股票的{time_level}级别历史分时成交数据")
            
            # 保存剩余数据
            if data_dicts:
                logger.info(f"开始保存剩余{len(data_dicts)}条数据")
                final_result = await self._save_all_to_db(
                    model_class=StockTimeTrade,
                    data_list=data_dicts,
                    unique_fields=['stock', 'time_level', 'trade_time']
                )
                logger.info(f"剩余数据保存完成，结果: {final_result}")
                
                # 累加最终结果
                for key in total_result:
                    total_result[key] += final_result.get(key, 0)
            
            logger.info(f"所有股票历史KDJ指标数据保存完成，总结果: {total_result}")
            return total_result
        except Exception as e:
            logger.error(f"保存自选股分时成交数据出错: {str(e)}")
            logger.debug(f"错误数据内容: {data_dicts if 'data_dicts' in locals() else '未获取到数据'}")
            return {'创建': 0, '更新': 0, '跳过': 0}
    
    async def fetch_and_save_all_history_time_trade(self) -> Optional[StockTimeTrade]:
        """
        从API获取并保存所有历史股票分时成交数据
        """
        stocks = await self.stock_basic_dao.get_stock_list()
        try:
            data_dicts = []
            total_result = {'创建': 0, '更新': 0, '跳过': 0}
            for stock in stocks:
                for time_level in TIME_LEVELS:
                    api_datas = await self.api.get_history_trade(stock.stock_code, time_level)
                    for api_item in api_datas:
                        data_dict = {
                            'stock': stock,
                            'time_level': api_item.get('t'),
                            'trade_time': self._parse_datetime(api_item.get('d')),  # 交易时间
                            'open_price': self._parse_number(api_item.get('o')),  # 开盘价
                            'high_price': self._parse_number(api_item.get('h')),  # 最高价
                            'low_price': self._parse_number(api_item.get('l')),  # 最低价
                            'close_price': self._parse_number(api_item.get('c')),  # 收盘价
                            'volume': self._parse_number(api_item.get('v')),  # 成交量
                            'turnover': self._parse_number(api_item.get('e')),  # 成交额
                            'amplitude': self._parse_number(api_item.get('zf')),  # 振幅
                            'turnover_rate': self._parse_number(api_item.get('hs')),  # 换手率
                            'price_change_percent': self._parse_number(api_item.get('zd')),  # 涨跌幅
                            'price_change_amount': self._parse_number(api_item.get('zde')),  # 涨跌额   
                        }
                        data_dicts.append(data_dict)
                    # 当数据量超过10万时，保存一次
                    if len(data_dicts) >= 100000:
                        logger.info(f"数据量达到{len(data_dicts)}，开始保存批次数据")
                        batch_result = await self._save_all_to_db(
                            model_class=StockTimeTrade,
                            data_list=data_dicts,
                            unique_fields=['stock', 'time_level', 'trade_time']
                        )
                        logger.info(f"批次数据保存完成，结果: {batch_result}")
                        # 累加结果
                        for key in total_result:
                            total_result[key] += batch_result.get(key, 0)
                        # 清空数据列表，准备下一批
                        data_dicts = []
                    logger.warning(f"当前data_dicts总量: {len(data_dicts)}")
                if not api_datas:
                    logger.warning(f"API未返回{stock.stock_code}股票的{time_level}级别历史分时成交数据")
            
            # 保存剩余数据
            if data_dicts:
                logger.info(f"开始保存剩余{len(data_dicts)}条数据")
                final_result = await self._save_all_to_db(
                    model_class=StockTimeTrade,
                    data_list=data_dicts,
                    unique_fields=['stock', 'time_level', 'trade_time']
                )
                logger.info(f"剩余数据保存完成，结果: {final_result}")
                
                # 累加最终结果
                for key in total_result:
                    total_result[key] += final_result.get(key, 0)
            
            logger.info(f"所有股票历史KDJ指标数据保存完成，总结果: {total_result}")
            return total_result
        except Exception as e:
            logger.error(f"保存{stock}股票历史分时成交数据出错: {str(e)}")
            logger.debug(f"错误数据内容: {data_dicts if 'data_dicts' in locals() else '未获取到数据'}")
            return {'创建': 0, '更新': 0, '跳过': 0}
    
    async def refresh_time_trade(self, stock_code: str, time_level: Union[TimeLevel, str]) -> Optional[StockTimeTrade]:
        """
        刷新分时成交数据
        
        Args:
            stock_code: 股票代码
            time_level: 时间级别
            
        Returns:
            Optional[StockTimeTrade]: 最新的分时成交数据
        """
        return await self._refresh_indicator(
            model_class=StockTimeTrade,
            stock_code=stock_code,
            time_level=time_level,
            api_method=self.api.get_time_trade_data,
            mapping=TIME_TRADE_MAPPING,
            cache_prefix="time_trade"
        )
    
    # ================= KDJ指标相关方法 =================
    
    async def get_latest_kdj(self, stock_code: str, time_level: Union[TimeLevel, str]) -> Optional[StockKDJIndicator]:
        """
        获取最新的KDJ指标数据
        
        Args:
            stock_code: 股票代码
            time_level: 时间级别
            
        Returns:
            Optional[StockKDJIndicator]: 最新的KDJ指标数据
        """
        return await self._get_latest_indicator(
            model_class=StockKDJIndicator,
            stock_code=stock_code,
            time_level=time_level,
            api_method=self.api.get_kdj_data,
            mapping=KDJ_INDICATOR_MAPPING,
            cache_prefix="kdj"
        )
    
    async def fetch_and_save_latest_kdj(self, stock_code: str, time_level: str) -> Optional[StockKDJIndicator]:
        """
        从API获取并保存最新KDJ指标数据
        
        Args:
            stock_code: 股票代码
            time_level: 时间级别
        """
        # 获取股票信息
        stock = await self.stock_basic_dao.get_stock_by_code(stock_code)
        if not stock:
            return {'创建': 0, '更新': 0, '跳过': 0}
        try:
            api_data = await self.api.get_kdj(stock.stock_code, time_level)
                
            if not api_data:
                logger.warning(f"API未返回{stock}的{time_level}级别KDJ指标数据")
                return []
            
            data_dicts = []
            data_dict = {
                'stock': stock,
                'time_level': time_level,
                'trade_time': self._parse_datetime(api_data.get('t')),  # 交易时间
                'k_value': self._parse_number(api_data.get('k')),  # K值
                'd_value': self._parse_number(api_data.get('d')),  # D值
                'j_value': self._parse_number(api_data.get('j')),  # J值
            }
            data_dicts.append(data_dict)

            # 保存数据
            logger.info(f"开始保存{stock}股票{time_level}级别KDJ指标数据")
            result = await self._save_all_to_db(
                model_class=StockKDJIndicator,
                data_list=data_dicts,
                unique_fields=['stock', 'time_level', 'trade_time']
            )
            
            logger.info(f"{stock}股票{time_level}级别KDJ指标数据保存完成，结果: {result}")
            return result
        except Exception as e:
            logger.error(f"保存{stock}股票{time_level}级别KDJ指标数据出错: {str(e)}")
            logger.debug(f"错误数据内容: {data_dicts if 'data_dicts' in locals() else '未获取到数据'}")
            return {'创建': 0, '更新': 0, '跳过': 0}

    async def fetch_and_save_latest_kdj_by_stock_code(self, stock_code: str) -> Optional[StockKDJIndicator]:
        """
        从API获取并保存最新KDJ指标数据
        
        Args:
            stock_code: 股票代码
        """
        # 获取股票信息
        stock = await self.stock_basic_dao.get_stock_by_code(stock_code)
        if not stock:
            logger.warning(f"股票代码[{stock_code}]不存在，无法获取KDJ指标数据")
            return []
        try:
            data_dicts = []
            for time_level in TIME_LEVELS:
                api_data = await self.api.get_kdj(stock.stock_code, time_level)
                data_dict = {
                    'stock': stock,
                    'time_level': time_level,
                    'trade_time': self._parse_datetime(api_data.get('t')),  # 交易时间
                    'k_value': self._parse_number(api_data.get('k')),  # K值
                    'd_value': self._parse_number(api_data.get('d')),  # D值
                    'j_value': self._parse_number(api_data.get('j')),  # J值
                }
                data_dicts.append(data_dict)
            if not data_dicts:
                logger.warning(f"API未返回{stock}股票的{time_level}级别KDJ指标数据")
                return []
            # 保存数据
            logger.info(f"开始保存{stock}股票KDJ指标数据")
            result = await self._save_all_to_db(
                model_class=StockKDJIndicator,
                data_list=data_dicts,
                unique_fields=['stock', 'time_level', 'trade_time']
            )
            
            logger.info(f"{stock}股票KDJ指标数据保存完成，结果: {result}")
            return result
        except Exception as e:
            logger.error(f"保存{stock}股票KDJ指标数据出错: {str(e)}")
            logger.debug(f"错误数据内容: {data_dicts if 'data_dicts' in locals() else '未获取到数据'}")
            return {'创建': 0, '更新': 0, '跳过': 0}

    async def fetch_and_save_latest_kdj_by_time_level(self, time_level: str) -> Optional[StockKDJIndicator]:
        """
        从API获取并保存最新KDJ指标数据
        
        Args:
            time_level: 时间级别
        """
        stocks = await self.stock_basic_dao.get_stock_list()
        try:
            data_dicts = []
            for stock in stocks:
                for time_level in TIME_LEVELS:
                    api_data = await self.api.get_kdj(stock.stock_code, time_level)
                    data_dict = {
                        'stock': stock,
                        'time_level': time_level,
                        'trade_time': self._parse_datetime(api_data.get('t')),  # 交易时间
                        'k_value': self._parse_number(api_data.get('k')),  # K值
                        'd_value': self._parse_number(api_data.get('d')),  # D值
                        'j_value': self._parse_number(api_data.get('j')),  # J值
                    }
                    data_dicts.append(data_dict)
            if not data_dicts:
                logger.warning(f"API未返回{time_level}级别的所有股票KDJ指标数据")
                return {'创建': 0, '更新': 0, '跳过': 0}
            # 保存数据
            logger.info(f"开始保存{time_level}级别KDJ指标数据")
            result = await self._save_all_to_db(
                model_class=StockKDJIndicator,
                data_list=data_dicts,
                unique_fields=['stock', 'time_level', 'trade_time']
            )
            
            logger.info(f"{time_level}级别KDJ指标数据保存完成，结果: {result}")
            return result
        except Exception as e:
            logger.error(f"保存{time_level}级别KDJ指标数据出错: {str(e)}")
            logger.debug(f"错误数据内容: {data_dicts if 'data_dicts' in locals() else '未获取到数据'}")
            return {'创建': 0, '更新': 0, '跳过': 0}

    async def fetch_and_save_favorite_stocks_latest_kdj(self) -> Optional[StockKDJIndicator]:
        """
        从API获取并保存自选股最新KDJ指标数据
        """
        favorite_stocks = await self.user_dao.get_all_favorite_stocks()
        try:
            data_dicts = []
            for stock in favorite_stocks:
                for time_level in TIME_LEVELS:
                    api_data = await self.api.get_kdj(stock.stock_code, time_level)
                    data_dict = {
                        'stock': stock,
                        'time_level': time_level,
                        'trade_time': self._parse_datetime(api_data.get('t')),  # 交易时间
                        'k_value': self._parse_number(api_data.get('k')),  # K值
                        'd_value': self._parse_number(api_data.get('d')),  # D值
                        'j_value': self._parse_number(api_data.get('j')),  # J值
                    }
                    data_dicts.append(data_dict)
            if not data_dicts:
                logger.warning(f"API未返回自选股（get_all_favorite_stocks）的{time_level}级别KDJ指标数据")
                return {'创建': 0, '更新': 0, '跳过': 0}
            
            # 保存数据
            logger.info("开始保存自选股KDJ指标数据")
            result = await self._save_all_to_db(
                model_class=StockKDJIndicator,
                data_list=data_dicts,
                unique_fields=['stock', 'time_level', 'trade_time']
            )
            
            logger.info(f"自选股KDJ指标数据保存完成，结果: {result}")
            return result
        except Exception as e:
            logger.error(f"保存自选股KDJ指标数据出错: {str(e)}")
            logger.debug(f"错误数据内容: {data_dicts if 'data_dicts' in locals() else '未获取到数据'}")
            return {'创建': 0, '更新': 0, '跳过': 0}

    async def fetch_and_save_all_latest_kdj(self) -> Optional[StockKDJIndicator]:
        """
        从API获取并保存所有股票最新KDJ指标数据
        """
        stocks = await self.stock_basic_dao.get_stock_list()
        try:
            data_dicts = []
            for stock in stocks:
                for time_level in TIME_LEVELS:
                    api_data = await self.api.get_kdj(stock.stock_code, time_level)
                    data_dict = {
                        'stock': stock,
                        'time_level': time_level,
                        'trade_time': self._parse_datetime(api_data.get('t')),  # 交易时间
                        'k_value': self._parse_number(api_data.get('k')),  # K值
                        'd_value': self._parse_number(api_data.get('d')),  # D值
                        'j_value': self._parse_number(api_data.get('j')),  # J值
                    }
                    data_dicts.append(data_dict)
            if not data_dicts:
                logger.warning(f"API未返回{stock}股票的{time_level}级别KDJ指标数据")
                return []
            # 保存数据
            logger.info(f"开始保存{stock}股票KDJ指标数据")
            result = await self._save_all_to_db(
                model_class=StockKDJIndicator,
                data_list=data_dicts,
                unique_fields=['stock', 'time_level', 'trade_time']
            )
            
            logger.info(f"{stock}股票KDJ指标数据保存完成，结果: {result}")
            return result
        except Exception as e:
            logger.error(f"保存{stock}股票KDJ指标数据出错: {str(e)}")
            logger.debug(f"错误数据内容: {data_dicts if 'data_dicts' in locals() else '未获取到数据'}")
            return {'创建': 0, '更新': 0, '跳过': 0}

    # ================= 历史KDJ指标相关方法 =================
    async def get_history_kdj(self, stock_code: str, time_level: str, 
                             limit: int = 1000) -> List[StockKDJIndicator]:
        """
        获取历史KDJ指标数据
        
        Args:
            stock_code: 股票代码
            time_level: 时间级别
            limit: 返回记录数量限制
            
        Returns:
            List[StockKDJIndicator]: 历史KDJ指标数据列表
        """
        return await self._get_history_indicators(
            model_class=StockKDJIndicator,
            stock_code=stock_code,
            time_level=time_level,
            api_method=self.api.get_kdj_data,
            mapping=KDJ_INDICATOR_MAPPING,
            cache_prefix="kdj",
            limit=limit
        )
    
    async def fetch_and_save_history_kdj(self, stock_code: str, time_level: str, limit: int = 1000) -> Optional[StockKDJIndicator]:
        """
        从API获取并保存历史KDJ指标数据
        """
        stock = await self.stock_basic_dao.get_stock_by_code(stock_code)
        if not stock:
            return {'创建': 0, '更新': 0, '跳过': 0}
        try:
            api_data = await self.api.get_history_kdj(stock.stock_code, time_level)
                
            if not api_data:
                logger.warning(f"API未返回{stock}的{time_level}级别KDJ指标数据")
                return {'创建': 0, '更新': 0, '跳过': 0}
            
            data_dicts = []
            data_dict = {
                'stock': stock,
                'time_level': time_level,
                'trade_time': self._parse_datetime(api_data.get('t')),  # 交易时间
                'k_value': self._parse_number(api_data.get('k')),  # K值
                'd_value': self._parse_number(api_data.get('d')),  # D值
                'j_value': self._parse_number(api_data.get('j')),  # J值
            }
            data_dicts.append(data_dict)

            # 保存数据
            logger.info(f"开始保存{stock}股票{time_level}级别KDJ指标数据")
            result = await self._save_all_to_db(
                model_class=StockKDJIndicator,
                data_list=data_dicts,
                unique_fields=['stock', 'time_level', 'trade_time']
            )
            
            logger.info(f"{stock}股票{time_level}级别KDJ指标数据保存完成，结果: {result}")
            return result
        except Exception as e:
            logger.error(f"保存{stock}股票{time_level}级别KDJ指标数据出错: {str(e)}")
            logger.debug(f"错误数据内容: {data_dicts if 'data_dicts' in locals() else '未获取到数据'}")
            return {'创建': 0, '更新': 0, '跳过': 0}

    async def fetch_and_save_history_kdj_by_stock_code(self, stock_code: str) -> Optional[StockKDJIndicator]:
        """
        从API获取并保存股票历史KDJ指标数据
        """
        # 获取股票信息
        stock = await self.stock_basic_dao.get_stock_by_code(stock_code)
        if not stock:
            return {'创建': 0, '更新': 0, '跳过': 0}
        try:
            data_dicts = []
            total_result = {'创建': 0, '更新': 0, '跳过': 0}
            for time_level in TIME_LEVELS:
                api_datas = await self.api.get_history_kdj(stock.stock_code, time_level)
                for api_data in api_datas:
                    data_dict = {
                        'stock': stock,
                        'time_level': time_level,
                        'trade_time': self._parse_datetime(api_data.get('t')),  # 交易时间
                        'k_value': self._parse_number(api_data.get('k')),  # K值
                        'd_value': self._parse_number(api_data.get('d')),  # D值
                        'j_value': self._parse_number(api_data.get('j')),  # J值
                    }
                    data_dicts.append(data_dict)
                # 当数据量超过10万时，保存一次
                if len(data_dicts) >= 100000:
                    logger.info(f"数据量达到{len(data_dicts)}，开始保存批次数据")
                    batch_result = await self._save_all_to_db(
                        model_class=StockKDJIndicator,
                        data_list=data_dicts,
                        unique_fields=['stock', 'time_level', 'trade_time']
                    )
                    logger.info(f"批次数据保存完成，结果: {batch_result}")
                    # 累加结果
                    for key in total_result:
                        total_result[key] += batch_result.get(key, 0)
                    # 清空数据列表，准备下一批
                    data_dicts = []
                logger.warning(f"当前data_dicts总量: {len(data_dicts)}")
            if not api_datas:
                logger.warning(f"API未返回{stock.stock_code}股票的{time_level}级别历史KDJ指标数据")
            
            # 保存剩余数据
            if data_dicts:
                logger.info(f"开始保存剩余{len(data_dicts)}条数据")
                final_result = await self._save_all_to_db(
                    model_class=StockKDJIndicator,
                    data_list=data_dicts,
                    unique_fields=['stock', 'time_level', 'trade_time']
                )
                logger.info(f"剩余数据保存完成，结果: {final_result}")
                
                # 累加最终结果
                for key in total_result:
                    total_result[key] += final_result.get(key, 0)
            
            logger.info(f"所有股票历史KDJ指标数据保存完成，总结果: {total_result}")
            return total_result
        except Exception as e:
            logger.error(f"保存{stock}股票历史KDJ指标数据出错: {str(e)}")
            logger.debug(f"错误数据内容: {data_dicts if 'data_dicts' in locals() else '未获取到数据'}")
            return {'创建': 0, '更新': 0, '跳过': 0}

    async def fetch_and_save_history_kdj_by_time_level(self, time_level: str) -> Optional[StockKDJIndicator]:
        """
        从API获取并保存所有股票历史KDJ指标数据
        """
        stocks = await self.stock_basic_dao.get_stock_list()
        try:
            data_dicts = []
            total_result = {'创建': 0, '更新': 0, '跳过': 0}
            for stock in stocks:
                for time_level in TIME_LEVELS:
                    api_datas = await self.api.get_history_kdj(stock.stock_code, time_level)
                    for api_data in api_datas:
                        data_dict = {
                            'stock': stock,
                            'time_level': time_level,
                            'trade_time': self._parse_datetime(api_data.get('t')),  # 交易时间
                            'k_value': self._parse_number(api_data.get('k')),  # K值
                            'd_value': self._parse_number(api_data.get('d')),  # D值
                            'j_value': self._parse_number(api_data.get('j')),  # J值
                        }
                        data_dicts.append(data_dict)
                    # 当数据量超过10万时，保存一次
                    if len(data_dicts) >= 100000:
                        logger.info(f"数据量达到{len(data_dicts)}，开始保存批次数据")
                        batch_result = await self._save_all_to_db(
                            model_class=StockKDJIndicator,
                            data_list=data_dicts,
                            unique_fields=['stock', 'time_level', 'trade_time']
                        )
                        logger.info(f"批次数据保存完成，结果: {batch_result}")
                        # 累加结果
                        for key in total_result:
                            total_result[key] += batch_result.get(key, 0)
                        # 清空数据列表，准备下一批
                        data_dicts = []
                    logger.warning(f"当前data_dicts总量: {len(data_dicts)}")
                if not api_datas:
                    logger.warning(f"API未返回{stock.stock_code}股票的{time_level}级别历史KDJ指标数据")
            
            # 保存剩余数据
            if data_dicts:
                logger.info(f"开始保存剩余{len(data_dicts)}条数据")
                final_result = await self._save_all_to_db(
                    model_class=StockKDJIndicator,
                    data_list=data_dicts,
                    unique_fields=['stock', 'time_level', 'trade_time']
                )
                logger.info(f"剩余数据保存完成，结果: {final_result}")
                
                # 累加最终结果
                for key in total_result:
                    total_result[key] += final_result.get(key, 0)
            
            logger.info(f"所有股票历史KDJ指标数据保存完成，总结果: {total_result}")
            return total_result
        except Exception as e:
            logger.error(f"保存{time_level}级别历史KDJ指标数据出错: {str(e)}")
            logger.debug(f"错误数据内容: {data_dicts if 'data_dicts' in locals() else '未获取到数据'}")
            return {'创建': 0, '更新': 0, '跳过': 0}

    async def fetch_and_save_favorite_stocks_history_kdj(self) -> Optional[StockKDJIndicator]:
        """
        从API获取并保存自选股历史KDJ指标数据
        """
        favorite_stocks = await self.user_dao.get_all_favorite_stocks()
        try:
            data_dicts = []
            total_result = {'创建': 0, '更新': 0, '跳过': 0}
            for stock in favorite_stocks:
                for time_level in TIME_LEVELS:
                    api_datas = await self.api.get_history_kdj(stock.stock_code, time_level)
                    for api_data in api_datas:
                        data_dict = {
                            'stock': stock,
                            'time_level': time_level,
                            'trade_time': self._parse_datetime(api_data.get('t')),  # 交易时间
                            'k_value': self._parse_number(api_data.get('k')),  # K值
                            'd_value': self._parse_number(api_data.get('d')),  # D值
                            'j_value': self._parse_number(api_data.get('j')),  # J值
                        }
                        data_dicts.append(data_dict)
                    # 当数据量超过10万时，保存一次
                    if len(data_dicts) >= 100000:
                        logger.info(f"数据量达到{len(data_dicts)}，开始保存批次数据")
                        batch_result = await self._save_all_to_db(
                            model_class=StockKDJIndicator,
                            data_list=data_dicts,
                            unique_fields=['stock', 'time_level', 'trade_time']
                        )
                        logger.info(f"批次数据保存完成，结果: {batch_result}")
                        # 累加结果
                        for key in total_result:
                            total_result[key] += batch_result.get(key, 0)
                        # 清空数据列表，准备下一批
                        data_dicts = []
                    logger.warning(f"当前data_dicts总量: {len(data_dicts)}")
                if not api_datas:
                    logger.warning(f"API未返回{stock.stock_code}股票的{time_level}级别历史KDJ指标数据")
            
            # 保存剩余数据
            if data_dicts:
                logger.info(f"开始保存剩余{len(data_dicts)}条数据")
                final_result = await self._save_all_to_db(
                    model_class=StockKDJIndicator,
                    data_list=data_dicts,
                    unique_fields=['stock', 'time_level', 'trade_time']
                )
                logger.info(f"剩余数据保存完成，结果: {final_result}")
                
                # 累加最终结果
                for key in total_result:
                    total_result[key] += final_result.get(key, 0)
            
            logger.info(f"所有股票历史KDJ指标数据保存完成，总结果: {total_result}")
            return total_result
        except Exception as e:
            logger.error(f"保存自选股历史KDJ指标数据出错: {str(e)}")
            logger.debug(f"错误数据内容: {data_dicts if 'data_dicts' in locals() else '未获取到数据'}")
            return {'创建': 0, '更新': 0, '跳过': 0}

    async def fetch_and_save_all_history_kdj(self) -> Optional[StockMACDIndicator]:
        """
        从API获取并保存所有股票历史KDJ指标数据
        """
        stocks = await self.stock_basic_dao.get_stock_list()
        try:
            data_dicts = []
            total_result = {'创建': 0, '更新': 0, '跳过': 0}
            for stock in stocks:
                for time_level in TIME_LEVELS:
                    api_datas = await self.api.get_history_kdj(stock.stock_code, time_level)
                    for api_data in api_datas:
                        data_dict = {
                            'stock': stock,
                            'time_level': time_level,
                            'trade_time': self._parse_datetime(api_data.get('t')),  # 交易时间
                            'k_value': self._parse_number(api_data.get('k')),  # K值
                            'd_value': self._parse_number(api_data.get('d')),  # D值
                            'j_value': self._parse_number(api_data.get('j')),  # J值
                        }
                        data_dicts.append(data_dict)
                    # 当数据量超过10万时，保存一次
                    if len(data_dicts) >= 100000:
                        logger.info(f"数据量达到{len(data_dicts)}，开始保存批次数据")
                        batch_result = await self._save_all_to_db(
                            model_class=StockKDJIndicator,
                            data_list=data_dicts,
                            unique_fields=['stock', 'time_level', 'trade_time']
                        )
                        logger.info(f"批次数据保存完成，结果: {batch_result}")
                        # 累加结果
                        for key in total_result:
                            total_result[key] += batch_result.get(key, 0)
                        # 清空数据列表，准备下一批
                        data_dicts = []
                    logger.warning(f"当前data_dicts总量: {len(data_dicts)}")
                if not api_datas:
                    logger.warning(f"API未返回{stock.stock_code}股票的{time_level}级别历史KDJ指标数据")
            
            # 保存剩余数据
            if data_dicts:
                logger.info(f"开始保存剩余{len(data_dicts)}条数据")
                final_result = await self._save_all_to_db(
                    model_class=StockKDJIndicator,
                    data_list=data_dicts,
                    unique_fields=['stock', 'time_level', 'trade_time']
                )
                logger.info(f"剩余数据保存完成，结果: {final_result}")
                
                # 累加最终结果
                for key in total_result:
                    total_result[key] += final_result.get(key, 0)
            
            logger.info(f"所有股票历史KDJ指标数据保存完成，总结果: {total_result}")
            return total_result
        except Exception as e:
            logger.error(f"保存所有股票历史KDJ指标数据出错: {str(e)}")
            logger.debug(f"错误数据内容: {data_dicts if 'data_dicts' in locals() else '未获取到数据'}")
            return {'创建': 0, '更新': 0, '跳过': 0}
        
    async def refresh_kdj(self, stock_code: str, time_level: Union[TimeLevel, str]) -> Optional[StockKDJIndicator]:
        """
        刷新KDJ指标数据
        
        Args:
            stock_code: 股票代码
            time_level: 时间级别
            
        Returns:
            Optional[StockKDJIndicator]: 最新的KDJ指标数据
        """
        return await self._refresh_indicator(
            model_class=StockKDJIndicator,
            stock_code=stock_code,
            time_level=time_level,
            api_method=self.api.get_kdj_data,
            mapping=KDJ_INDICATOR_MAPPING,
            cache_prefix="kdj"
        )
    
    # ================= MACD指标相关方法 =================
    
    async def get_latest_macd(self, stock_code: str, time_level: Union[TimeLevel, str]) -> Optional[StockMACDIndicator]:
        """
        获取最新的MACD指标数据
        
        Args:
            stock_code: 股票代码
            time_level: 时间级别
            
        Returns:
            Optional[StockMACDIndicator]: 最新的MACD指标数据
        """
        return await self._get_latest_indicator(
            model_class=StockMACDIndicator,
            stock_code=stock_code,
            time_level=time_level,
            api_method=self.api.get_macd_data,
            mapping=MACD_INDICATOR_MAPPING,
            cache_prefix="macd"
        )

    async def fetch_and_save_latest_macd(self, stock_code: str, time_level: str) -> Optional[StockMACDIndicator]:
        """
        从API获取并保存最新MACD指标数据
        """
        # 获取股票信息
        stock = await self.stock_basic_dao.get_stock_by_code(stock_code)
        if not stock:
            return {'创建': 0, '更新': 0, '跳过': 0}
        try:
            api_data = await self.api.get_macd(stock.stock_code, time_level)
                
            if not api_data:
                logger.warning(f"API未返回{stock}的{time_level}级别最新MACD指标数据")
                return {'创建': 0, '更新': 0, '跳过': 0}
            
            data_dicts = []
            data_dict = {
                'stock': stock,
                'time_level': time_level,
                'trade_time': self._parse_datetime(api_data.get('t')),  # 交易时间
                'diff': self._parse_number(api_data.get('diff')),  # DIFF值
                'dea': self._parse_number(api_data.get('dea')),  # DEA值
                'macd': self._parse_number(api_data.get('macd')),  # MACD值
                'ema12': self._parse_number(api_data.get('ema12')),  # EMA(12)值
                'ema26': self._parse_number(api_data.get('ema26')),  # EMA(26)值
            }
            data_dicts.append(data_dict)

            # 保存数据
            logger.info(f"开始保存{stock}股票{time_level}级别最新MACD指标数据")
            result = await self._save_all_to_db(
                model_class=StockMACDIndicator,
                data_list=data_dicts,
                unique_fields=['stock', 'time_level', 'trade_time']
            )
            
            logger.info(f"{stock}股票{time_level}级别最新MACD指标数据保存完成，结果: {result}")
            return result
        except Exception as e:
            logger.error(f"保存{stock}股票{time_level}级别最新MACD指标数据出错: {str(e)}")
            logger.debug(f"错误数据内容: {data_dicts if 'data_dicts' in locals() else '未获取到数据'}")
            return {'创建': 0, '更新': 0, '跳过': 0}

    async def fetch_and_save_latest_macd_by_stock_code(self, stock_code: str) -> Optional[StockMACDIndicator]:
        """
        从API获取并保存所有股票最新MACD指标数据
        """
        # 获取股票信息
        stock = await self.stock_basic_dao.get_stock_by_code(stock_code)
        if not stock:
            return {'创建': 0, '更新': 0, '跳过': 0}
        try:
            data_dicts = []
            for time_level in TIME_LEVELS:
                api_data = await self.api.get_macd(stock.stock_code, time_level)
                data_dict = {
                    'stock': stock,
                    'time_level': time_level,
                    'trade_time': self._parse_datetime(api_data.get('t')),  # 交易时间
                    'diff': self._parse_number(api_data.get('diff')),  # DIFF值
                    'dea': self._parse_number(api_data.get('dea')),  # DEA值
                    'macd': self._parse_number(api_data.get('macd')),  # MACD值
                    'ema12': self._parse_number(api_data.get('ema12')),  # EMA(12)值
                    'ema26': self._parse_number(api_data.get('ema26')),  # EMA(26)值
                }
                data_dicts.append(data_dict)
            if not data_dicts:
                logger.warning(f"API未返回{stock}股票的{time_level}级别最新MACD指标数据")
                return {'创建': 0, '更新': 0, '跳过': 0}            
            # 保存数据
            logger.info(f"开始保存{stock}股票最新MACD指标数据")
            result = await self._save_all_to_db(
                model_class=StockMACDIndicator,
                data_list=data_dicts,
                unique_fields=['stock', 'time_level', 'trade_time']
            )
            
            logger.info(f"{stock}股票最新MACD指标数据保存完成，结果: {result}")
            return result
        except Exception as e:
            logger.error(f"保存{stock}股票最新MACD指标数据出错: {str(e)}")
            logger.debug(f"错误数据内容: {data_dicts if 'data_dicts' in locals() else '未获取到数据'}")
            return {'创建': 0, '更新': 0, '跳过': 0}

    async def fetch_and_save_latest_macd_by_time_level(self, time_level: str) -> Optional[StockMACDIndicator]:
        """
        从API获取并保存所有股票最新MACD指标数据
        """
        stocks = await self.stock_basic_dao.get_stock_list()
        try:
            data_dicts = []
            for stock in stocks:
                for time_level in TIME_LEVELS:
                    api_data = await self.api.get_macd(stock.stock_code, time_level)
                    data_dict = {
                        'stock': stock,
                        'time_level': time_level,
                        'trade_time': self._parse_datetime(api_data.get('t')),  # 交易时间
                        'diff': self._parse_number(api_data.get('diff')),  # DIFF值
                        'dea': self._parse_number(api_data.get('dea')),  # DEA值
                        'macd': self._parse_number(api_data.get('macd')),  # MACD值
                        'ema12': self._parse_number(api_data.get('ema12')),  # EMA(12)值
                        'ema26': self._parse_number(api_data.get('ema26')),  # EMA(26)值
                    }
                    data_dicts.append(data_dict)
            if not data_dicts:
                logger.warning(f"API未返回{time_level}级别的所有股票最新MACD指标数据")
                return {'创建': 0, '更新': 0, '跳过': 0}
            # 保存数据
            logger.info(f"开始保存{time_level}级别最新MACD指标数据")
            result = await self._save_all_to_db(
                model_class=StockMACDIndicator,
                data_list=data_dicts,
                unique_fields=['stock', 'time_level', 'trade_time']
            )
            
            logger.info(f"{time_level}级别最新MACD指标数据保存完成，结果: {result}")
            return result
        except Exception as e:
            logger.error(f"保存{time_level}级别最新MACD指标数据出错: {str(e)}")
            logger.debug(f"错误数据内容: {data_dicts if 'data_dicts' in locals() else '未获取到数据'}")
            return {'创建': 0, '更新': 0, '跳过': 0}

    async def fetch_and_save_favorite_stocks_latest_macd(self) -> Optional[StockMACDIndicator]:
        """
        从API获取并保存自选股最新MACD指标数据
        """
        favorite_stocks = await self.user_dao.get_all_favorite_stocks()
        try:
            data_dicts = []
            for stock in favorite_stocks:
                for time_level in TIME_LEVELS:
                    api_data = await self.api.get_macd(stock.stock_code, time_level)
                    data_dict = {
                        'stock': stock,
                        'time_level': time_level,
                        'trade_time': self._parse_datetime(api_data.get('t')),  # 交易时间
                        'diff': self._parse_number(api_data.get('diff')),  # DIFF值
                        'dea': self._parse_number(api_data.get('dea')),  # DEA值
                        'macd': self._parse_number(api_data.get('macd')),  # MACD值
                        'ema12': self._parse_number(api_data.get('ema12')),  # EMA(12)值
                        'ema26': self._parse_number(api_data.get('ema26')),  # EMA(26)值
                    }
                    data_dicts.append(data_dict)
            if not data_dicts:
                logger.warning(f"API未返回自选股（get_all_favorite_stocks）的{time_level}级别最新MACD指标数据")
                return {'创建': 0, '更新': 0, '跳过': 0}
            
            # 保存数据
            logger.info("开始保存自选股最新MACD指标数据")
            result = await self._save_all_to_db(
                model_class=StockMACDIndicator,
                data_list=data_dicts,
                unique_fields=['stock', 'time_level', 'trade_time']
            )
            
            logger.info(f"自选股最新MACD指标数据保存完成，结果: {result}")
            return result
        except Exception as e:
            logger.error(f"保存自选股最新MACD指标数据出错: {str(e)}")
            logger.debug(f"错误数据内容: {data_dicts if 'data_dicts' in locals() else '未获取到数据'}")
            return {'创建': 0, '更新': 0, '跳过': 0}

    async def fetch_and_save_all_latest_macd(self) -> Optional[StockMACDIndicator]:
        """
        从API获取并保存所有股票最新MACD指标数据
        """
        stocks = await self.stock_basic_dao.get_stock_list()
        try:
            data_dicts = []
            for stock in stocks:
                for time_level in TIME_LEVELS:
                    api_data = await self.api.get_macd(stock.stock_code, time_level)
                    data_dict = {
                        'stock': stock,
                        'time_level': time_level,
                        'trade_time': self._parse_datetime(api_data.get('t')),  # 交易时间
                        'diff': self._parse_number(api_data.get('diff')),  # DIFF值
                        'dea': self._parse_number(api_data.get('dea')),  # DEA值
                        'macd': self._parse_number(api_data.get('macd')),  # MACD值
                        'ema12': self._parse_number(api_data.get('ema12')),  # EMA(12)值
                        'ema26': self._parse_number(api_data.get('ema26')),  # EMA(26)值
                    }
                    data_dicts.append(data_dict)
            if not data_dicts:
                logger.warning(f"API未返回所有股票的{time_level}级别最新MACD指标数据")
                return {'创建': 0, '更新': 0, '跳过': 0}
            # 保存数据
            logger.info(f"开始保存所有股票最新MACD指标数据")
            result = await self._save_all_to_db(
                model_class=StockMACDIndicator,
                data_list=data_dicts,
                unique_fields=['stock', 'time_level', 'trade_time']
            )
            
            logger.info(f"所有股票最新MACD指标数据保存完成，结果: {result}")
            return result
        except Exception as e:
            logger.error(f"保存所有股票最新MACD指标数据出错: {str(e)}")
            logger.debug(f"错误数据内容: {data_dicts if 'data_dicts' in locals() else '未获取到数据'}")
            return {'创建': 0, '更新': 0, '跳过': 0}

    # ================= 历史MACD指标相关方法 =================
    async def get_history_macd(self, stock_code: str, time_level: Union[TimeLevel, str], 
                             limit: int = 1000) -> List[StockMACDIndicator]:
        """
        获取历史MACD指标数据
        
        Args:
            stock_code: 股票代码
            time_level: 时间级别
            limit: 返回记录数量限制
            
        Returns:
            List[StockMACDIndicator]: 历史MACD指标数据列表
        """
        return await self._get_history_indicators(
            model_class=StockMACDIndicator,
            stock_code=stock_code,
            time_level=time_level,
            api_method=self.api.get_macd_data,
            mapping=MACD_INDICATOR_MAPPING,
            cache_prefix="macd",
            limit=limit
        )
    
    async def fetch_and_save_history_macd(self, stock_code: str, time_level: str) -> Optional[StockMACDIndicator]:
        """
        从API获取并保存历史MACD指标数据
        Args:
            stock_code: 股票代码
            time_level: 时间级别
        Returns:
            Optional[StockMACDIndicator]: 保存结果
        """
        stock = await self.stock_basic_dao.get_stock_by_code(stock_code)
        if not stock:
            return {'创建': 0, '更新': 0, '跳过': 0}
        try:
            api_data = await self.api.get_history_macd(stock.stock_code, time_level)
                
            if not api_data:
                logger.warning(f"API未返回{stock}的{time_level}级别历史MACD指标数据")
                return {'创建': 0, '更新': 0, '跳过': 0}
            
            data_dicts = []
            data_dict = {
                'stock': stock,
                'time_level': time_level,
                'trade_time': self._parse_datetime(api_data.get('t')),  # 交易时间
                'diff': self._parse_number(api_data.get('diff')),  # DIFF值
                'dea': self._parse_number(api_data.get('dea')),  # DEA值
                'macd': self._parse_number(api_data.get('macd')),  # MACD值
                'ema12': self._parse_number(api_data.get('ema12')),  # EMA(12)值
                'ema26': self._parse_number(api_data.get('ema26')),  # EMA(26)值
            }
            data_dicts.append(data_dict)

            # 保存数据
            logger.info(f"开始保存{stock}股票{time_level}级别历史MACD指标数据")
            result = await self._save_all_to_db(
                model_class=StockMACDIndicator,
                data_list=data_dicts,
                unique_fields=['stock', 'time_level', 'trade_time']
            )
            
            logger.info(f"{stock}股票{time_level}级别历史MACD指标数据保存完成，结果: {result}")
            return result
        except Exception as e:
            logger.error(f"保存{stock}股票{time_level}级别历史MACD指标数据出错: {str(e)}")
            logger.debug(f"错误数据内容: {data_dicts if 'data_dicts' in locals() else '未获取到数据'}")
            return {'创建': 0, '更新': 0, '跳过': 0}

    async def fetch_and_save_history_macd_by_stock_code(self, stock_code: str) -> Optional[StockMACDIndicator]:
        """
        从API获取并保存所有股票历史MACD指标数据
        Args:
            stock_code: 股票代码
        Returns:
            Optional[StockMACDIndicator]: 保存结果
        """
        # 获取股票信息
        stock = await self.stock_basic_dao.get_stock_by_code(stock_code)
        if not stock:
            return {'创建': 0, '更新': 0, '跳过': 0}
        try:
            data_dicts = []
            total_result = {'创建': 0, '更新': 0, '跳过': 0}
            for time_level in TIME_LEVELS:
                api_datas = await self.api.get_history_macd(stock.stock_code, time_level)
                for api_data in api_datas:
                    data_dict = {
                        'stock': stock,
                        'time_level': time_level,
                        'trade_time': self._parse_datetime(api_data.get('t')),  # 交易时间
                        'diff': self._parse_number(api_data.get('diff')),  # DIFF值
                        'dea': self._parse_number(api_data.get('dea')),  # DEA值
                        'macd': self._parse_number(api_data.get('macd')),  # MACD值
                        'ema12': self._parse_number(api_data.get('ema12')),  # EMA(12)值
                        'ema26': self._parse_number(api_data.get('ema26')),  # EMA(26)值
                    }
                    data_dicts.append(data_dict)
                # 当数据量超过10万时，保存一次
                if len(data_dicts) >= 100000:
                    logger.info(f"数据量达到{len(data_dicts)}，开始保存批次数据")
                    batch_result = await self._save_all_to_db(
                        model_class=StockMACDIndicator,
                        data_list=data_dicts,
                        unique_fields=['stock', 'time_level', 'trade_time']
                    )
                    logger.info(f"批次数据保存完成，结果: {batch_result}")
                    # 累加结果
                    for key in total_result:
                        total_result[key] += batch_result.get(key, 0)
                    # 清空数据列表，准备下一批
                    data_dicts = []
                logger.warning(f"当前data_dicts总量: {len(data_dicts)}")
            if not api_datas:
                logger.warning(f"API未返回{stock.stock_code}股票的{time_level}级别历史MACD指标数据")
            
            # 保存剩余数据
            if data_dicts:
                logger.info(f"开始保存剩余{len(data_dicts)}条数据")
                final_result = await self._save_all_to_db(
                    model_class=StockMACDIndicator,
                    data_list=data_dicts,
                    unique_fields=['stock', 'time_level', 'trade_time']
                )
                logger.info(f"剩余数据保存完成，结果: {final_result}")
                
                # 累加最终结果
                for key in total_result:
                    total_result[key] += final_result.get(key, 0)
            
            logger.info(f"所有股票历史MACD指标数据保存完成，总结果: {total_result}")
            return total_result
        except Exception as e:
            logger.error(f"保存{stock}股票历史MACD指标数据出错: {str(e)}")
            logger.debug(f"错误数据内容: {data_dicts if 'data_dicts' in locals() else '未获取到数据'}")
            return {'创建': 0, '更新': 0, '跳过': 0}

    async def fetch_and_save_history_macd_by_time_level(self, time_level: str) -> Optional[StockMACDIndicator]:
        """
        从API获取并保存所有股票历史MACD指标数据
        Args:
            time_level: 时间级别
        Returns:
            Optional[StockMACDIndicator]: 保存结果
        """
        stocks = await self.stock_basic_dao.get_stock_list()
        try:
            data_dicts = []
            total_result = {'创建': 0, '更新': 0, '跳过': 0}
            for stock in stocks:
                for time_level in TIME_LEVELS:
                    api_datas = await self.api.get_history_macd(stock.stock_code, time_level)
                    for api_data in api_datas:
                        data_dict = {
                            'stock': stock,
                            'time_level': time_level,
                            'trade_time': self._parse_datetime(api_data.get('t')),  # 交易时间
                            'diff': self._parse_number(api_data.get('diff')),  # DIFF值
                            'dea': self._parse_number(api_data.get('dea')),  # DEA值
                            'macd': self._parse_number(api_data.get('macd')),  # MACD值
                            'ema12': self._parse_number(api_data.get('ema12')),  # EMA(12)值
                            'ema26': self._parse_number(api_data.get('ema26')),  # EMA(26)值
                        }
                        data_dicts.append(data_dict)
                    # 当数据量超过10万时，保存一次
                    if len(data_dicts) >= 100000:
                        logger.info(f"数据量达到{len(data_dicts)}，开始保存批次数据")
                        batch_result = await self._save_all_to_db(
                            model_class=StockMACDIndicator,
                            data_list=data_dicts,
                            unique_fields=['stock', 'time_level', 'trade_time']
                        )
                        logger.info(f"批次数据保存完成，结果: {batch_result}")
                        # 累加结果
                        for key in total_result:
                            total_result[key] += batch_result.get(key, 0)
                        # 清空数据列表，准备下一批
                        data_dicts = []
                    logger.warning(f"当前data_dicts总量: {len(data_dicts)}")
                if not api_datas:
                    logger.warning(f"API未返回{stock.stock_code}股票的{time_level}级别历史MACD指标数据")
            
            # 保存剩余数据
            if data_dicts:
                logger.info(f"开始保存剩余{len(data_dicts)}条数据")
                final_result = await self._save_all_to_db(
                    model_class=StockMACDIndicator,
                    data_list=data_dicts,
                    unique_fields=['stock', 'time_level', 'trade_time']
                )
                logger.info(f"剩余数据保存完成，结果: {final_result}")
                
                # 累加最终结果
                for key in total_result:
                    total_result[key] += final_result.get(key, 0)
            
            logger.info(f"所有股票历史MACD指标数据保存完成，总结果: {total_result}")
            return total_result
        except Exception as e:
            logger.error(f"保存{time_level}级别历史MACD指标数据出错: {str(e)}")
            logger.debug(f"错误数据内容: {data_dicts if 'data_dicts' in locals() else '未获取到数据'}")
            return {'创建': 0, '更新': 0, '跳过': 0}

    async def fetch_and_save_favorite_stocks_history_macd(self) -> Optional[StockMACDIndicator]:
        """
        从API获取并保存自选股历史MACD指标数据
        """
        favorite_stocks = await self.user_dao.get_all_favorite_stocks()
        try:
            data_dicts = []
            total_result = {'创建': 0, '更新': 0, '跳过': 0}
            for stock in favorite_stocks:
                for time_level in TIME_LEVELS:
                    api_datas = await self.api.get_history_macd(stock.stock_code, time_level)
                    for api_data in api_datas:
                        data_dict = {
                            'stock': stock,
                            'time_level': time_level,
                            'trade_time': self._parse_datetime(api_data.get('t')),  # 交易时间
                            'diff': self._parse_number(api_data.get('diff')),  # DIFF值
                            'dea': self._parse_number(api_data.get('dea')),  # DEA值
                            'macd': self._parse_number(api_data.get('macd')),  # MACD值
                            'ema12': self._parse_number(api_data.get('ema12')),  # EMA(12)值
                            'ema26': self._parse_number(api_data.get('ema26')),  # EMA(26)值
                        }
                        data_dicts.append(data_dict)
                    # 当数据量超过10万时，保存一次
                    if len(data_dicts) >= 100000:
                        logger.info(f"数据量达到{len(data_dicts)}，开始保存批次数据")
                        batch_result = await self._save_all_to_db(
                            model_class=StockMACDIndicator,
                            data_list=data_dicts,
                            unique_fields=['stock', 'time_level', 'trade_time']
                        )
                        logger.info(f"批次数据保存完成，结果: {batch_result}")
                        # 累加结果
                        for key in total_result:
                            total_result[key] += batch_result.get(key, 0)
                        # 清空数据列表，准备下一批
                        data_dicts = []
                    logger.warning(f"当前data_dicts总量: {len(data_dicts)}")
                if not api_datas:
                    logger.warning(f"API未返回{stock.stock_code}股票的{time_level}级别历史MACD指标数据")
            
            # 保存剩余数据
            if data_dicts:
                logger.info(f"开始保存剩余{len(data_dicts)}条数据")
                final_result = await self._save_all_to_db(
                    model_class=StockMACDIndicator,
                    data_list=data_dicts,
                    unique_fields=['stock', 'time_level', 'trade_time']
                )
                logger.info(f"剩余数据保存完成，结果: {final_result}")
                
                # 累加最终结果
                for key in total_result:
                    total_result[key] += final_result.get(key, 0)
            
            logger.info(f"所有股票历史MACD指标数据保存完成，总结果: {total_result}")
            return total_result
        except Exception as e:
            logger.error(f"保存自选股历史MACD指标数据出错: {str(e)}")
            logger.debug(f"错误数据内容: {data_dicts if 'data_dicts' in locals() else '未获取到数据'}")
            return {'创建': 0, '更新': 0, '跳过': 0}

    async def fetch_and_save_all_history_macd(self) -> Optional[StockMACDIndicator]:
        """
        从API获取并保存所有股票历史MACD指标数据
        """
        stocks = await self.stock_basic_dao.get_stock_list()
        try:
            data_dicts = []
            total_result = {'创建': 0, '更新': 0, '跳过': 0}
            for stock in stocks:
                for time_level in TIME_LEVELS:
                    api_datas = await self.api.get_history_macd(stock.stock_code, time_level)
                    for api_data in api_datas:
                        data_dict = {
                            'stock': stock,
                            'time_level': time_level,
                            'trade_time': self._parse_datetime(api_data.get('t')),  # 交易时间
                            'diff': self._parse_number(api_data.get('diff')),  # DIFF值
                            'dea': self._parse_number(api_data.get('dea')),  # DEA值
                            'macd': self._parse_number(api_data.get('macd')),  # MACD值
                            'ema12': self._parse_number(api_data.get('ema12')),  # EMA(12)值
                            'ema26': self._parse_number(api_data.get('ema26')),  # EMA(26)值
                        }
                        data_dicts.append(data_dict)
                    # 当数据量超过10万时，保存一次
                    if len(data_dicts) >= 100000:
                        logger.info(f"数据量达到{len(data_dicts)}，开始保存批次数据")
                        batch_result = await self._save_all_to_db(
                            model_class=StockMACDIndicator,
                            data_list=data_dicts,
                            unique_fields=['stock', 'time_level', 'trade_time']
                        )
                        logger.info(f"批次数据保存完成，结果: {batch_result}")
                        # 累加结果
                        for key in total_result:
                            total_result[key] += batch_result.get(key, 0)
                        # 清空数据列表，准备下一批
                        data_dicts = []
                    logger.warning(f"当前data_dicts总量: {len(data_dicts)}")
                if not api_datas:
                    logger.warning(f"API未返回{stock.stock_code}股票的{time_level}级别历史MACD指标数据")
            
            # 保存剩余数据
            if data_dicts:
                logger.info(f"开始保存剩余{len(data_dicts)}条数据")
                final_result = await self._save_all_to_db(
                    model_class=StockMACDIndicator,
                    data_list=data_dicts,
                    unique_fields=['stock', 'time_level', 'trade_time']
                )
                logger.info(f"剩余数据保存完成，结果: {final_result}")
                
                # 累加最终结果
                for key in total_result:
                    total_result[key] += final_result.get(key, 0)
            
            logger.info(f"所有股票历史MACD指标数据保存完成，总结果: {total_result}")
            return total_result
        except Exception as e:
            logger.error(f"保存所有股票历史MACD指标数据出错: {str(e)}")
            logger.debug(f"错误数据内容: {data_dicts if 'data_dicts' in locals() else '未获取到数据'}")
            return {'创建': 0, '更新': 0, '跳过': 0}
        
    async def refresh_macd(self, stock_code: str, time_level: Union[TimeLevel, str]) -> Optional[StockMACDIndicator]:
        """
        刷新MACD指标数据
        
        Args:
            stock_code: 股票代码
            time_level: 时间级别
            
        Returns:
            Optional[StockMACDIndicator]: 最新的MACD指标数据
        """
        return await self._refresh_indicator(
            model_class=StockMACDIndicator,
            stock_code=stock_code,
            time_level=time_level,
            api_method=self.api.get_macd_data,
            mapping=MACD_INDICATOR_MAPPING,
            cache_prefix="macd"
        )
    
    # ================= MA指标相关方法 =================
    
    async def get_latest_ma(self, stock_code: str, time_level: Union[TimeLevel, str]) -> Optional[StockMAIndicator]:
        """
        获取最新的MA指标数据
        
        Args:
            stock_code: 股票代码
            time_level: 时间级别
            
        Returns:
            Optional[StockMAIndicator]: 最新的MA指标数据
        """
        return await self._get_latest_indicator(
            model_class=StockMAIndicator,
            stock_code=stock_code,
            time_level=time_level,
            api_method=self.api.get_ma_data,
            mapping=MA_INDICATOR_MAPPING,
            cache_prefix="ma"
        )
    
    async def fetch_and_save_latest_ma(self, stock_code: str, time_level: str) -> Optional[StockMAIndicator]:
        """
        从API获取并保存股票历史MA指标数据
        Args:
            stock_code: 股票代码
            time_level: 时间级别
        Returns:
            Optional[StockMAIndicator]: 保存结果
        """
        # 获取股票信息
        stock = await self.stock_basic_dao.get_stock_by_code(stock_code)
        if not stock:
            return {'创建': 0, '更新': 0, '跳过': 0}
        try:
            api_data = await self.api.get_ma(stock.stock_code, time_level)
                
            if not api_data:
                logger.warning(f"API未返回{stock}的{time_level}级别最新MA指标数据")
                return {'创建': 0, '更新': 0, '跳过': 0}
            
            data_dicts = []
            data_dict = {
                'stock': stock,
                'time_level': time_level,
                'trade_time': self._parse_datetime(api_data.get('t')),  # 交易时间
                'ma3': self._parse_number(api_data.get('ma3')),  # MA3值
                'ma5': self._parse_number(api_data.get('ma5')),  # MA5值
                'ma10': self._parse_number(api_data.get('ma10')),  # MA10值
                'ma15': self._parse_number(api_data.get('ma15')),  # MA15值
                'ma20': self._parse_number(api_data.get('ma20')),  # MA20值
                'ma30': self._parse_number(api_data.get('ma30')),  # MA30值 
                'ma60': self._parse_number(api_data.get('ma60')),  # MA60值
                'ma120': self._parse_number(api_data.get('ma120')),  # MA120值
                'ma200': self._parse_number(api_data.get('ma200')),  # MA200值
                'ma250': self._parse_number(api_data.get('ma250')),  # MA250值
            }
            data_dicts.append(data_dict)

            # 保存数据
            logger.info(f"开始保存{stock}股票{time_level}级别最新MA指标数据")
            result = await self._save_all_to_db(
                model_class=StockMAIndicator,
                data_list=data_dicts,
                unique_fields=['stock', 'time_level', 'trade_time']
            )
            
            logger.info(f"{stock}股票{time_level}级别最新MA指标数据保存完成，结果: {result}")
            return result
        except Exception as e:
            logger.error(f"保存{stock}股票{time_level}级别最新MA指标数据出错: {str(e)}")
            logger.debug(f"错误数据内容: {data_dicts if 'data_dicts' in locals() else '未获取到数据'}")
            return {'创建': 0, '更新': 0, '跳过': 0}

    async def fetch_and_save_latest_ma_by_stock_code(self, stock_code: str) -> Optional[StockMAIndicator]:
        """
        从API获取并保存股票历史MA指标数据
        Args:
            stock_code: 股票代码
        Returns:
            Optional[StockMAIndicator]: 保存结果
        """
        # 获取股票信息
        stock = await self.stock_basic_dao.get_stock_by_code(stock_code)
        if not stock:
            return {'创建': 0, '更新': 0, '跳过': 0}
        try:
            data_dicts = []
            for time_level in TIME_LEVELS:
                api_data = await self.api.get_ma(stock.stock_code, time_level)
                data_dict = {
                    'stock': stock,
                    'time_level': time_level,
                    'trade_time': self._parse_datetime(api_data.get('t')),  # 交易时间
                    'ma3': self._parse_number(api_data.get('ma3')),  # MA3值
                    'ma5': self._parse_number(api_data.get('ma5')),  # MA5值
                    'ma10': self._parse_number(api_data.get('ma10')),  # MA10值
                    'ma15': self._parse_number(api_data.get('ma15')),  # MA15值
                    'ma20': self._parse_number(api_data.get('ma20')),  # MA20值
                    'ma30': self._parse_number(api_data.get('ma30')),  # MA30值
                    'ma60': self._parse_number(api_data.get('ma60')),  # MA60值
                    'ma120': self._parse_number(api_data.get('ma120')),  # MA120值
                    'ma200': self._parse_number(api_data.get('ma200')),  # MA200值
                    'ma250': self._parse_number(api_data.get('ma250')),  # MA250值
                }
                data_dicts.append(data_dict)
            if not data_dicts:
                logger.warning(f"API未返回{stock}股票的{time_level}级别最新MA指标数据")
                return {'创建': 0, '更新': 0, '跳过': 0}
            # 保存数据
            logger.info(f"开始保存{stock}股票最新MA指标数据")
            result = await self._save_all_to_db(
                model_class=StockMAIndicator,
                data_list=data_dicts,
                unique_fields=['stock', 'time_level', 'trade_time']
            )
            
            logger.info(f"{stock}股票最新MA指标数据保存完成，结果: {result}")
            return result
        except Exception as e:
            logger.error(f"保存{stock}股票最新MA指标数据出错: {str(e)}")
            logger.debug(f"错误数据内容: {data_dicts if 'data_dicts' in locals() else '未获取到数据'}")
            return {'创建': 0, '更新': 0, '跳过': 0}

    async def fetch_and_save_latest_ma_by_time_level(self, time_level: str) -> Optional[StockMAIndicator]:
        """
        从API获取并保存所有股票历史MA指标数据
        Args:
            time_level: 时间级别
        Returns:
            Optional[StockMAIndicator]: 保存结果
        """
        stocks = await self.stock_basic_dao.get_stock_list()
        try:
            data_dicts = []  
            for stock in stocks:
                for time_level in TIME_LEVELS:
                    api_data = await self.api.get_ma(stock.stock_code, time_level)
                    data_dict = {
                        'stock': stock,
                        'time_level': time_level,
                        'trade_time': self._parse_datetime(api_data.get('t')),  # 交易时间
                        'ma3': self._parse_number(api_data.get('ma3')),  # MA3值
                        'ma5': self._parse_number(api_data.get('ma5')),  # MA5值
                        'ma10': self._parse_number(api_data.get('ma10')),  # MA10值
                        'ma15': self._parse_number(api_data.get('ma15')),  # MA15值
                        'ma20': self._parse_number(api_data.get('ma20')),  # MA20值
                        'ma30': self._parse_number(api_data.get('ma30')),  # MA30值
                        'ma60': self._parse_number(api_data.get('ma60')),  # MA60值
                        'ma120': self._parse_number(api_data.get('ma120')),  # MA120值
                        'ma200': self._parse_number(api_data.get('ma200')),  # MA200值
                        'ma250': self._parse_number(api_data.get('ma250')),  # MA250值
                    }
                    data_dicts.append(data_dict)
            if not data_dicts:
                logger.warning(f"API未返回{time_level}级别的所有股票最新MA指标数据")
                return {'创建': 0, '更新': 0, '跳过': 0}
            # 保存数据
            logger.info(f"开始保存{time_level}级别最新MA指标数据")
            result = await self._save_all_to_db(
                model_class=StockMAIndicator,
                data_list=data_dicts,
                unique_fields=['stock', 'time_level', 'trade_time']
            )
            
            logger.info(f"{time_level}级别最新MA指标数据保存完成，结果: {result}")
            return result
        except Exception as e:
            logger.error(f"保存{time_level}级别最新MA指标数据出错: {str(e)}")
            logger.debug(f"错误数据内容: {data_dicts if 'data_dicts' in locals() else '未获取到数据'}")
            return {'创建': 0, '更新': 0, '跳过': 0}

    async def fetch_and_save_favorite_stocks_latest_ma(self) -> Optional[StockMAIndicator]:
        """
        从API获取并保存自选股最新MA指标数据
        """
        favorite_stocks = await self.user_dao.get_all_favorite_stocks()
        try:
            data_dicts = []
            for stock in favorite_stocks:
                for time_level in TIME_LEVELS:
                    api_data = await self.api.get_ma(stock.stock_code, time_level)
                    data_dict = {
                        'stock': stock,
                        'time_level': time_level,
                        'trade_time': self._parse_datetime(api_data.get('t')),  # 交易时间
                        'ma3': self._parse_number(api_data.get('ma3')),  # MA3值
                        'ma5': self._parse_number(api_data.get('ma5')),  # MA5值
                        'ma10': self._parse_number(api_data.get('ma10')),  # MA10值
                        'ma15': self._parse_number(api_data.get('ma15')),  # MA15值
                        'ma20': self._parse_number(api_data.get('ma20')),  # MA20值
                        'ma30': self._parse_number(api_data.get('ma30')),  # MA30值
                        'ma60': self._parse_number(api_data.get('ma60')),  # MA60值
                        'ma120': self._parse_number(api_data.get('ma120')),  # MA120值
                        'ma200': self._parse_number(api_data.get('ma200')),  # MA200值
                        'ma250': self._parse_number(api_data.get('ma250')),  # MA250值
                    }
                    data_dicts.append(data_dict)
            if not data_dicts:
                logger.warning(f"API未返回自选股（get_all_favorite_stocks）的{time_level}级别最新MA指标数据")
                return {'创建': 0, '更新': 0, '跳过': 0}
            # 保存数据
            logger.info("开始保存自选股最新MA指标数据")
            result = await self._save_all_to_db(
                model_class=StockMAIndicator,
                data_list=data_dicts,
                unique_fields=['stock', 'time_level', 'trade_time']
            )
            
            logger.info(f"自选股最新MA指标数据保存完成，结果: {result}")
            return result
        except Exception as e:
            logger.error(f"保存自选股最新MA指标数据出错: {str(e)}")
            logger.debug(f"错误数据内容: {data_dicts if 'data_dicts' in locals() else '未获取到数据'}")
            return {'创建': 0, '更新': 0, '跳过': 0}

    async def fetch_and_save_all_latest_ma(self) -> Optional[StockMAIndicator]:
        """
        从API获取并保存所有股票最新MA指标数据
        """
        stocks = await self.stock_basic_dao.get_stock_list()
        try:
            data_dicts = []
            for stock in stocks:
                for time_level in TIME_LEVELS:
                    api_data = await self.api.get_ma(stock.stock_code, time_level)
                    data_dict = {
                        'stock': stock,
                        'time_level': time_level,
                        'trade_time': self._parse_datetime(api_data.get('t')),  # 交易时间
                        'ma3': self._parse_number(api_data.get('ma3')),  # MA3值
                        'ma5': self._parse_number(api_data.get('ma5')),  # MA5值
                        'ma10': self._parse_number(api_data.get('ma10')),  # MA10值
                        'ma15': self._parse_number(api_data.get('ma15')),  # MA15值
                        'ma20': self._parse_number(api_data.get('ma20')),  # MA20值
                        'ma30': self._parse_number(api_data.get('ma30')),  # MA30值
                        'ma60': self._parse_number(api_data.get('ma60')),  # MA60值
                        'ma120': self._parse_number(api_data.get('ma120')),  # MA120值
                        'ma200': self._parse_number(api_data.get('ma200')),  # MA200值
                        'ma250': self._parse_number(api_data.get('ma250')),  # MA250值
                    }
                    data_dicts.append(data_dict)
            if not data_dicts:
                logger.warning(f"API未返回所有股票的{time_level}级别最新MA指标数据")
                return {'创建': 0, '更新': 0, '跳过': 0}
            # 保存数据
            logger.info(f"开始保存所有股票最新MA指标数据")
            result = await self._save_all_to_db(
                model_class=StockMAIndicator,
                data_list=data_dicts,
                unique_fields=['stock', 'time_level', 'trade_time']
            )            
            logger.info(f"所有股票最新MA指标数据保存完成，结果: {result}")
            return result
        except Exception as e:
            logger.error(f"保存所有股票最新MA指标数据出错: {str(e)}")
            logger.debug(f"错误数据内容: {data_dicts if 'data_dicts' in locals() else '未获取到数据'}")
            return {'创建': 0, '更新': 0, '跳过': 0}
            
    # ================= 历史MA指标相关方法 =================
    async def get_history_ma(self, stock_code: str, time_level: Union[TimeLevel, str], 
                           limit: int = 1000) -> List[StockMAIndicator]:
        """
        获取历史MA指标数据
        
        Args:
            stock_code: 股票代码
            time_level: 时间级别
            limit: 返回记录数量限制
            
        Returns:
            List[StockMAIndicator]: 历史MA指标数据列表
        """
        return await self._get_history_indicators(
            model_class=StockMAIndicator,
            stock_code=stock_code,
            time_level=time_level,
            api_method=self.api.get_ma_data,
            mapping=MA_INDICATOR_MAPPING,
            cache_prefix="ma",
            limit=limit
        )
    
    async def fetch_and_save_history_ma(self, stock_code: str, time_level: str) -> Optional[StockMAIndicator]:
        """
        从API获取并保存股票历史MA指标数据
        Args:
            stock_code: 股票代码
            time_level: 时间级别
        Returns:
            Optional[StockMAIndicator]: 保存结果
        """
        stock = await self.stock_basic_dao.get_stock_by_code(stock_code)
        if not stock:
            return {'创建': 0, '更新': 0, '跳过': 0}
        try:
            api_datas = await self.api.get_history_ma(stock.stock_code, time_level)
            if not api_datas:
                logger.warning(f"API未返回{stock}的{time_level}级别历史MA指标数据")
                return {'创建': 0, '更新': 0, '跳过': 0}
            
            data_dicts = []
            for api_data in api_datas:
                data_dict = {
                    'stock': stock,
                    'time_level': time_level,
                    'trade_time': self._parse_datetime(api_data.get('t')),  # 交易时间
                    'ma3': self._parse_number(api_data.get('ma3')),  # MA3值    
                    'ma5': self._parse_number(api_data.get('ma5')),  # MA5值
                    'ma10': self._parse_number(api_data.get('ma10')),  # MA10值
                    'ma15': self._parse_number(api_data.get('ma15')),  # MA15值
                    'ma20': self._parse_number(api_data.get('ma20')),  # MA20值
                    'ma30': self._parse_number(api_data.get('ma30')),  # MA30值
                    'ma60': self._parse_number(api_data.get('ma60')),  # MA60值
                    'ma120': self._parse_number(api_data.get('ma120')),  # MA120值
                    'ma200': self._parse_number(api_data.get('ma200')),  # MA200值
                    'ma250': self._parse_number(api_data.get('ma250')),  # MA250值   
                }
                data_dicts.append(data_dict)

            # 保存数据
            logger.info(f"开始保存{stock}股票{time_level}级别历史MA指标数据")
            result = await self._save_all_to_db(
                model_class=StockMAIndicator,
                data_list=data_dicts,
                unique_fields=['stock', 'time_level', 'trade_time']
            )
            
            logger.info(f"{stock}股票{time_level}级别历史MA指标数据保存完成，结果: {result}")
            return result
        except Exception as e:
            logger.error(f"保存{stock}股票{time_level}级别历史MA指标数据出错: {str(e)}")
            logger.debug(f"错误数据内容: {data_dicts if 'data_dicts' in locals() else '未获取到数据'}")
            return {'创建': 0, '更新': 0, '跳过': 0}

    async def fetch_and_save_history_ma_by_stock_code(self, stock_code: str) -> Optional[StockMAIndicator]:
        """
        从API获取并保存股票历史MA指标数据
        Args:
            stock_code: 股票代码
        Returns:
            Optional[StockMAIndicator]: 保存结果
        """
        # 获取股票信息
        stock = await self.stock_basic_dao.get_stock_by_code(stock_code)
        if not stock:
            return {'创建': 0, '更新': 0, '跳过': 0}
        try:
            data_dicts = []
            total_result = {'创建': 0, '更新': 0, '跳过': 0}
            for time_level in TIME_LEVELS:
                api_datas = await self.api.get_history_ma(stock_code, time_level)
                for api_item in api_datas:
                    data_dict = {
                        'stock': stock,
                        'time_level': time_level,
                        'trade_time': self._parse_datetime(api_item.get('t')),  # 交易时间
                        'ma3': self._parse_number(api_item.get('ma3')),  # MA3值
                        'ma5': self._parse_number(api_item.get('ma5')),  # MA5值
                        'ma10': self._parse_number(api_item.get('ma10')),  # MA10值
                        'ma15': self._parse_number(api_item.get('ma15')),  # MA15值
                        'ma20': self._parse_number(api_item.get('ma20')),  # MA20值
                        'ma30': self._parse_number(api_item.get('ma30')),  # MA30值
                        'ma60': self._parse_number(api_item.get('ma60')),  # MA60值
                        'ma120': self._parse_number(api_item.get('ma120')),  # MA120值
                        'ma200': self._parse_number(api_item.get('ma200')),  # MA200值
                        'ma250': self._parse_number(api_item.get('ma250')),  # MA250值
                    }
                    data_dicts.append(data_dict)
                # 当数据量超过10万时，保存一次
                if len(data_dicts) >= 100000:
                    logger.info(f"数据量达到{len(data_dicts)}，开始保存批次数据")
                    batch_result = await self._save_all_to_db(
                        model_class=StockMAIndicator,
                        data_list=data_dicts,
                        unique_fields=['stock', 'time_level', 'trade_time']
                    )
                    logger.info(f"批次数据保存完成，结果: {batch_result}")
                    # 累加结果
                    for key in total_result:
                        total_result[key] += batch_result.get(key, 0)
                    # 清空数据列表，准备下一批
                    data_dicts = []
                logger.warning(f"当前data_dicts总量: {len(data_dicts)}")
                if not api_datas:
                    logger.warning(f"API未返回{stock.stock_code}股票的{time_level}级别历史BOLL指标数据")
            
            # 保存剩余数据
            if data_dicts:
                logger.info(f"开始保存剩余{len(data_dicts)}条数据")
                final_result = await self._save_all_to_db(
                    model_class=StockMAIndicator,
                    data_list=data_dicts,
                    unique_fields=['stock', 'time_level', 'trade_time']
                )
                logger.info(f"剩余数据保存完成，结果: {final_result}")
                
                # 累加最终结果
                for key in total_result:
                    total_result[key] += final_result.get(key, 0)
            
            logger.info(f"所有股票历史MA指标数据保存完成，总结果: {total_result}")
            return total_result
        except Exception as e:
            logger.error(f"保存{stock}股票历史MA指标数据出错: {str(e)}")
            logger.debug(f"错误数据内容: {data_dicts if 'data_dicts' in locals() else '未获取到数据'}")
            return {'创建': 0, '更新': 0, '跳过': 0}

    async def fetch_and_save_history_ma_by_time_level(self, time_level: str) -> Optional[StockMAIndicator]:
        """
        从API获取并保存所有股票历史MA指标数据
        Args:
            time_level: 时间级别
        Returns:
            Optional[StockMAIndicator]: 保存结果
        """
        stocks = await self.stock_basic_dao.get_stock_list()
        try:
            data_dicts = []
            total_result = {'创建': 0, '更新': 0, '跳过': 0}
            for stock in stocks:
                for time_level in TIME_LEVELS:
                    api_datas = await self.api.get_history_ma(stock.stock_code, time_level)
                    for api_item in api_datas:
                        data_dict = {
                            'stock': stock,
                            'time_level': time_level,
                            'trade_time': self._parse_datetime(api_item.get('t')),  # 交易时间
                            'ma3': self._parse_number(api_item.get('ma3')),  # MA3值
                            'ma5': self._parse_number(api_item.get('ma5')),  # MA5值
                            'ma10': self._parse_number(api_item.get('ma10')),  # MA10值
                            'ma15': self._parse_number(api_item.get('ma15')),  # MA15值
                            'ma20': self._parse_number(api_item.get('ma20')),  # MA20值
                            'ma30': self._parse_number(api_item.get('ma30')),  # MA30值
                            'ma60': self._parse_number(api_item.get('ma60')),  # MA60值
                            'ma120': self._parse_number(api_item.get('ma120')),  # MA120值
                            'ma200': self._parse_number(api_item.get('ma200')),  # MA200值
                            'ma250': self._parse_number(api_item.get('ma250')),  # MA250值
                        }
                        data_dicts.append(data_dict)
                    # 当数据量超过10万时，保存一次
                    if len(data_dicts) >= 100000:
                        logger.info(f"数据量达到{len(data_dicts)}，开始保存批次数据")
                        batch_result = await self._save_all_to_db(
                            model_class=StockMAIndicator,
                            data_list=data_dicts,
                            unique_fields=['stock', 'time_level', 'trade_time']
                        )
                        logger.info(f"批次数据保存完成，结果: {batch_result}")
                        # 累加结果
                        for key in total_result:
                            total_result[key] += batch_result.get(key, 0)
                        # 清空数据列表，准备下一批
                        data_dicts = []
                    logger.warning(f"当前data_dicts总量: {len(data_dicts)}")
                if not api_datas:
                    logger.warning(f"API未返回{stock.stock_code}股票的{time_level}级别历史BOLL指标数据")
            
            # 保存剩余数据
            if data_dicts:
                logger.info(f"开始保存剩余{len(data_dicts)}条数据")
                final_result = await self._save_all_to_db(
                    model_class=StockMAIndicator,
                    data_list=data_dicts,
                    unique_fields=['stock', 'time_level', 'trade_time']
                )
                logger.info(f"剩余数据保存完成，结果: {final_result}")
                
                # 累加最终结果
                for key in total_result:
                    total_result[key] += final_result.get(key, 0)
            
            logger.info(f"所有股票历史MA指标数据保存完成，总结果: {total_result}")
            return total_result
        except Exception as e:
            logger.error(f"保存{time_level}级别所有股票历史MA指标数据出错: {str(e)}")
            logger.debug(f"错误数据内容: {data_dicts if 'data_dicts' in locals() else '未获取到数据'}")
            return {'创建': 0, '更新': 0, '跳过': 0}

    async def fetch_and_save_favorite_stocks_history_ma(self) -> Optional[StockMAIndicator]:
        """
        从API获取并保存自选股历史MA指标数据
        """
        favorite_stocks = await self.user_dao.get_all_favorite_stocks()
        try:
            data_dicts = []
            total_result = {'创建': 0, '更新': 0, '跳过': 0}
            for stock in favorite_stocks:
                for time_level in TIME_LEVELS:
                    api_datas = await self.api.get_history_ma(stock.stock_code, time_level)
                    for api_item in api_datas:
                        data_dict = {
                            'stock': stock,
                            'time_level': time_level,
                            'trade_time': self._parse_datetime(api_item.get('t')),  # 交易时间
                            'ma3': self._parse_number(api_item.get('ma3')),  # MA3值
                            'ma5': self._parse_number(api_item.get('ma5')),  # MA5值
                            'ma10': self._parse_number(api_item.get('ma10')),  # MA10值
                            'ma15': self._parse_number(api_item.get('ma15')),  # MA15值
                            'ma20': self._parse_number(api_item.get('ma20')),  # MA20值
                            'ma30': self._parse_number(api_item.get('ma30')),  # MA30值
                            'ma60': self._parse_number(api_item.get('ma60')),  # MA60值
                            'ma120': self._parse_number(api_item.get('ma120')),  # MA120值
                            'ma200': self._parse_number(api_item.get('ma200')),  # MA200值
                            'ma250': self._parse_number(api_item.get('ma250')),  # MA250值
                        }
                        data_dicts.append(data_dict)
                    # 当数据量超过10万时，保存一次
                    if len(data_dicts) >= 100000:
                        logger.info(f"数据量达到{len(data_dicts)}，开始保存批次数据")
                        batch_result = await self._save_all_to_db(
                            model_class=StockMAIndicator,
                            data_list=data_dicts,
                            unique_fields=['stock', 'time_level', 'trade_time']
                        )
                        logger.info(f"批次数据保存完成，结果: {batch_result}")
                        # 累加结果
                        for key in total_result:
                            total_result[key] += batch_result.get(key, 0)
                        # 清空数据列表，准备下一批
                        data_dicts = []
                    logger.warning(f"当前data_dicts总量: {len(data_dicts)}")
                if not api_datas:
                    logger.warning(f"API未返回{stock.stock_code}股票的{time_level}级别历史BOLL指标数据")
            
            # 保存剩余数据
            if data_dicts:
                logger.info(f"开始保存剩余{len(data_dicts)}条数据")
                final_result = await self._save_all_to_db(
                    model_class=StockMAIndicator,
                    data_list=data_dicts,
                    unique_fields=['stock', 'time_level', 'trade_time']
                )
                logger.info(f"剩余数据保存完成，结果: {final_result}")
                
                # 累加最终结果
                for key in total_result:
                    total_result[key] += final_result.get(key, 0)
            
            logger.info(f"所有股票历史MA指标数据保存完成，总结果: {total_result}")
            return total_result
        except Exception as e:
            logger.error(f"保存自选股历史MA指标数据出错: {str(e)}")
            logger.debug(f"错误数据内容: {data_dicts if 'data_dicts' in locals() else '未获取到数据'}")
            return {'创建': 0, '更新': 0, '跳过': 0}

    async def fetch_and_save_all_history_ma(self) -> Optional[StockMAIndicator]:
        """
        从API获取并保存所有股票历史MA指标数据
        """
        stocks = await self.stock_basic_dao.get_stock_list()
        try:
            data_dicts = []
            total_result = {'创建': 0, '更新': 0, '跳过': 0}
            for stock in stocks:
                for time_level in TIME_LEVELS:
                    api_datas = await self.api.get_history_ma(stock.stock_code, time_level)
                    for api_item in api_datas:
                        data_dict = {
                            'stock': stock,
                            'time_level': time_level,
                            'trade_time': self._parse_datetime(api_item.get('t')),  # 交易时间
                            'ma3': self._parse_number(api_item.get('ma3')),  # MA3值
                            'ma5': self._parse_number(api_item.get('ma5')),  # MA5值
                            'ma10': self._parse_number(api_item.get('ma10')),  # MA10值
                            'ma15': self._parse_number(api_item.get('ma15')),  # MA15值
                            'ma20': self._parse_number(api_item.get('ma20')),  # MA20值
                            'ma30': self._parse_number(api_item.get('ma30')),  # MA30值
                            'ma60': self._parse_number(api_item.get('ma60')),  # MA60值
                            'ma120': self._parse_number(api_item.get('ma120')),  # MA120值
                            'ma200': self._parse_number(api_item.get('ma200')),  # MA200值
                            'ma250': self._parse_number(api_item.get('ma250')),  # MA250值
                        }
                        data_dicts.append(data_dict)
                    # 当数据量超过10万时，保存一次
                    if len(data_dicts) >= 100000:
                        logger.info(f"数据量达到{len(data_dicts)}，开始保存批次数据")
                        batch_result = await self._save_all_to_db(
                            model_class=StockMAIndicator,
                            data_list=data_dicts,
                            unique_fields=['stock', 'time_level', 'trade_time']
                        )
                        logger.info(f"批次数据保存完成，结果: {batch_result}")
                        # 累加结果
                        for key in total_result:
                            total_result[key] += batch_result.get(key, 0)
                        # 清空数据列表，准备下一批
                        data_dicts = []
                    logger.warning(f"当前data_dicts总量: {len(data_dicts)}")
                if not api_datas:
                    logger.warning(f"API未返回{stock.stock_code}股票的{time_level}级别历史BOLL指标数据")
            
            # 保存剩余数据
            if data_dicts:
                logger.info(f"开始保存剩余{len(data_dicts)}条数据")
                final_result = await self._save_all_to_db(
                    model_class=StockMAIndicator,
                    data_list=data_dicts,
                    unique_fields=['stock', 'time_level', 'trade_time']
                )
                logger.info(f"剩余数据保存完成，结果: {final_result}")
                
                # 累加最终结果
                for key in total_result:
                    total_result[key] += final_result.get(key, 0)
            
            logger.info(f"所有股票历史MA指标数据保存完成，总结果: {total_result}")
            return total_result
        except Exception as e:
            logger.error(f"保存所有股票历史MA指标数据出错: {str(e)}")
            logger.debug(f"错误数据内容: {data_dicts if 'data_dicts' in locals() else '未获取到数据'}")
            return {'创建': 0, '更新': 0, '跳过': 0}

    async def refresh_ma(self, stock_code: str, time_level: Union[TimeLevel, str]) -> Optional[StockMAIndicator]:
        """
        刷新MA指标数据
        
        Args:
            stock_code: 股票代码
            time_level: 时间级别
            
        Returns:
            Optional[StockMAIndicator]: 最新的MA指标数据
        """
        return await self._refresh_indicator(
            model_class=StockMAIndicator,
            stock_code=stock_code,
            time_level=time_level,
            api_method=self.api.get_ma_data,
            mapping=MA_INDICATOR_MAPPING,
            cache_prefix="ma"
        )
    
    # ================= BOLL指标相关方法 =================
    async def get_latest_boll(self, stock_code: str, time_level: Union[TimeLevel, str]) -> Optional[StockBOLLIndicator]:
        """
        获取最新的BOLL指标数据
        
        Args:
            stock_code: 股票代码
            time_level: 时间级别
            
        Returns:
            Optional[StockBOLLIndicator]: 最新的BOLL指标数据
        """
        return await self._get_latest_indicator(
            model_class=StockBOLLIndicator,
            stock_code=stock_code,
            time_level=time_level,
            api_method=self.api.get_boll_data,
            mapping=BOLL_INDICATOR_MAPPING,
            cache_prefix="boll"
        )
    
    async def fetch_and_save_latest_boll(self, stock_code: str, time_level: str) -> Optional[StockBOLLIndicator]:
        """
        从API获取并保存股票最新BOLL指标数据
        Args:
            stock_code: 股票代码
            time_level: 时间级别
        Returns:
            Optional[StockBOLLIndicator]: 保存结果
        """
        # 获取股票信息
        stock = await self.stock_basic_dao.get_stock_by_code(stock_code)
        if not stock:
            return {'创建': 0, '更新': 0, '跳过': 0}
        try:
            api_data = await self.api.get_boll(stock.stock_code, time_level)
                
            if not api_data:
                logger.warning(f"API未返回{stock}的{time_level}级别最新BOLL指标数据")
                return {'创建': 0, '更新': 0, '跳过': 0}
            
            data_dicts = []
            data_dict = {
                'stock': stock,
                'time_level': time_level,
                'trade_time': self._parse_datetime(api_data.get('t')),  # 交易时间
                'upper': self._parse_number(api_data.get('upper')),  # 上轨
                'lower': self._parse_number(api_data.get('lower')),  # 下轨
                'mid': self._parse_number(api_data.get('mid')),  # 中轨
            }
            data_dicts.append(data_dict)

            # 保存数据
            logger.info(f"开始保存{stock}股票{time_level}级别最新BOLL指标数据")
            result = await self._save_all_to_db(
                model_class=StockBOLLIndicator,
                data_list=data_dicts,
                unique_fields=['stock', 'time_level', 'trade_time']
            )
            
            logger.info(f"{stock}股票{time_level}级别最新BOLL指标数据保存完成，结果: {result}")
            return result
        except Exception as e:
            logger.error(f"保存{stock}股票{time_level}级别最新BOLL指标数据出错: {str(e)}")
            logger.debug(f"错误数据内容: {data_dicts if 'data_dicts' in locals() else '未获取到数据'}")
            return {'创建': 0, '更新': 0, '跳过': 0}

    async def fetch_and_save_latest_boll_by_stock_code(self, stock_code: str) -> Optional[StockBOLLIndicator]:
        """
        从API获取并保存股票最新BOLL指标数据
        Args:
            stock_code: 股票代码
        Returns:
            Optional[StockBOLLIndicator]: 保存结果
        """
        # 获取股票信息
        stock = await self.stock_basic_dao.get_stock_by_code(stock_code)
        if not stock:
            return {'创建': 0, '更新': 0, '跳过': 0}
        try:
            data_dicts = []
            for time_level in TIME_LEVELS:
                api_data = await self.api.get_boll(stock.stock_code, time_level)
                data_dict = {
                    'stock': stock,
                    'time_level': time_level,
                    'trade_time': self._parse_datetime(api_data.get('t')),  # 交易时间
                    'upper': self._parse_number(api_data.get('u')),  # 上轨
                    'lower': self._parse_number(api_data.get('d')),  # 下轨
                    'mid': self._parse_number(api_data.get('m')),  # 中轨
                }
                data_dicts.append(data_dict)
            if not data_dicts:
                logger.warning(f"API未返回{stock}股票的{time_level}级别最新BOLL指标数据")
                return {'创建': 0, '更新': 0, '跳过': 0}
            # 保存数据
            logger.info(f"开始保存{stock}股票最新BOLL指标数据")
            result = await self._save_all_to_db(
                model_class=StockBOLLIndicator,
                data_list=data_dicts,
                unique_fields=['stock', 'time_level', 'trade_time']
            )
            
            logger.info(f"{stock}股票最新BOLL指标数据保存完成，结果: {result}")
            return result
        except Exception as e:
            logger.error(f"保存{stock}股票最新BOLL指标数据出错: {str(e)}")
            logger.debug(f"错误数据内容: {data_dicts if 'data_dicts' in locals() else '未获取到数据'}")
            return {'创建': 0, '更新': 0, '跳过': 0}

    async def fetch_and_save_latest_boll_by_time_level(self, time_level: str) -> Optional[StockBOLLIndicator]:
        """
        从API获取并保存所有股票最新BOLL指标数据
        Args:
            time_level: 时间级别
        Returns:
            Optional[StockBOLLIndicator]: 保存结果
        """
        stocks = await self.stock_basic_dao.get_stock_list()
        try:
            data_dicts = []
            for stock in stocks:
                for time_level in TIME_LEVELS:
                    api_data = await self.api.get_boll(stock.stock_code, time_level)
                    data_dict = {
                        'stock': stock,
                        'time_level': time_level,
                        'trade_time': self._parse_datetime(api_data.get('t')),  # 交易时间
                        'upper': self._parse_number(api_data.get('u')),  # 上轨
                        'lower': self._parse_number(api_data.get('d')),  # 下轨
                        'mid': self._parse_number(api_data.get('m')),  # 中轨
                    }
                    data_dicts.append(data_dict)
            if not data_dicts:
                logger.warning(f"API未返回{time_level}级别的所有股票最新BOLL指标数据")
                return {'创建': 0, '更新': 0, '跳过': 0}
            # 保存数据
            logger.info(f"开始保存{time_level}级别所有股票最新BOLL指标数据")
            result = await self._save_all_to_db(
                model_class=StockBOLLIndicator,
                data_list=data_dicts,
                unique_fields=['stock', 'time_level', 'trade_time']
            )
            
            logger.info(f"{time_level}级别所有股票最新BOLL指标数据保存完成，结果: {result}")
            return result
        except Exception as e:
            logger.error(f"保存{time_level}级别所有股票最新BOLL指标数据出错: {str(e)}")
            logger.debug(f"错误数据内容: {data_dicts if 'data_dicts' in locals() else '未获取到数据'}")
            return {'创建': 0, '更新': 0, '跳过': 0}
        
    async def fetch_and_save_favorite_stocks_latest_boll(self) -> Optional[StockBOLLIndicator]:
        """
        从API获取并保存自选股最新BOLL指标数据
        """
        favorite_stocks = await self.user_dao.get_all_favorite_stocks()
        try:
            data_dicts = []
            for stock in favorite_stocks:
                for time_level in TIME_LEVELS:
                    api_data = await self.api.get_boll(stock.stock_code, time_level)
                    data_dict = {
                        'stock': stock,
                        'time_level': time_level,
                        'trade_time': self._parse_datetime(api_data.get('t')),  # 交易时间
                        'upper': self._parse_number(api_data.get('u')),  # 上轨
                        'lower': self._parse_number(api_data.get('d')),  # 下轨
                        'mid': self._parse_number(api_data.get('m')),  # 中轨
                    }
                    data_dicts.append(data_dict)
            if not data_dicts:
                logger.warning(f"API未返回自选股（get_all_favorite_stocks）的{time_level}级别最新BOLL指标数据")
                return {'创建': 0, '更新': 0, '跳过': 0}
            # 保存数据
            logger.info("开始保存自选股最新BOLL指标数据")
            result = await self._save_all_to_db(
                model_class=StockBOLLIndicator,
                data_list=data_dicts,
                unique_fields=['stock', 'time_level', 'trade_time']
            )
            
            logger.info(f"自选股最新BOLL指标数据保存完成，结果: {result}")
            return result
        except Exception as e:
            logger.error(f"保存自选股最新BOLL指标数据出错: {str(e)}")
            logger.debug(f"错误数据内容: {data_dicts if 'data_dicts' in locals() else '未获取到数据'}")
            return {'创建': 0, '更新': 0, '跳过': 0}
        
    async def fetch_and_save_all_latest_boll(self) -> Optional[StockBOLLIndicator]:
        """
        从API获取并保存所有股票最新BOLL指标数据
        """
        stocks = await self.stock_basic_dao.get_stock_list()
        try:
            data_dicts = []
            for stock in stocks:
                for time_level in TIME_LEVELS:
                    api_data = await self.api.get_boll(stock.stock_code, time_level)
                    data_dict = {
                        'stock': stock,
                        'time_level': time_level,
                        'trade_time': self._parse_datetime(api_data.get('t')),  # 交易时间
                        'upper': self._parse_number(api_data.get('u')),  # 上轨
                        'lower': self._parse_number(api_data.get('d')),  # 下轨
                        'mid': self._parse_number(api_data.get('m')),  # 中轨
                    }
                    data_dicts.append(data_dict)
            if not data_dicts:
                logger.warning(f"API未返回所有股票的{time_level}级别最新BOLL指标数据")
                return {'创建': 0, '更新': 0, '跳过': 0}
            
            # 保存数据
            logger.info(f"开始保存所有股票最新BOLL指标数据")
            result = await self._save_all_to_db(
                model_class=StockBOLLIndicator,
                data_list=data_dicts,
                unique_fields=['stock', 'time_level', 'trade_time']
            )
            
            logger.info(f"所有股票最新BOLL指标数据保存完成，结果: {result}")
            return result
        except Exception as e:
            logger.error(f"保存所有股票最新BOLL指标数据出错: {str(e)}")
            logger.debug(f"错误数据内容: {data_dicts if 'data_dicts' in locals() else '未获取到数据'}")
            return {'创建': 0, '更新': 0, '跳过': 0}

    # ================= 历史BOLL指标相关方法 =================
    async def get_history_boll(self, stock_code: str, time_level: Union[TimeLevel, str], 
                             limit: int = 1000) -> List[StockBOLLIndicator]:
        """
        获取历史BOLL指标数据
        
        Args:
            stock_code: 股票代码
            time_level: 时间级别
            limit: 返回记录数量限制
            
        Returns:
            List[StockBOLLIndicator]: 历史BOLL指标数据列表
        """
        return await self._get_history_indicators(
            model_class=StockBOLLIndicator,
            stock_code=stock_code,
            time_level=time_level,
            api_method=self.api.get_boll_data,
            mapping=BOLL_INDICATOR_MAPPING,
            cache_prefix="boll",
            limit=limit
        )
    
    async def fetch_and_save_history_boll(self, stock_code: str, time_level: str) -> Optional[StockBOLLIndicator]: 
        """
        从API获取并保存股票历史BOLL指标数据
        Args:
            stock_code: 股票代码
            time_level: 时间级别
        Returns:
            Optional[StockBOLLIndicator]: 保存结果
        """
        # 获取股票信息
        stock = await self.stock_basic_dao.get_stock_by_code(stock_code)
        if not stock:
            return {'创建': 0, '更新': 0, '跳过': 0}
        try:
            api_datas = await self.api.get_history_boll(stock.stock_code, time_level)
                
            if not api_datas:
                logger.warning(f"API未返回{stock}的{time_level}级别历史BOLL指标数据")
                return {'创建': 0, '更新': 0, '跳过': 0}
            
            data_dicts = []
            for api_data in api_datas:
                data_dict = {
                    'stock': stock,
                    'time_level': time_level,
                    'trade_time': self._parse_datetime(api_data.get('t')),  # 交易时间
                    'upper': self._parse_number(api_data.get('upper')),  # 上轨
                    'lower': self._parse_number(api_data.get('lower')),  # 下轨
                    'mid': self._parse_number(api_data.get('mid')),  # 中轨
                }
                data_dicts.append(data_dict)

            # 保存数据
            logger.info(f"开始保存{stock}股票{time_level}级别历史BOLL指标数据")
            result = await self._save_all_to_db(
                model_class=StockBOLLIndicator,
                data_list=data_dicts,
                unique_fields=['stock', 'time_level', 'trade_time']
            )
            
            logger.info(f"{stock}股票{time_level}级别历史BOLL指标数据保存完成，结果: {result}")
            return result
        except Exception as e:
            logger.error(f"保存{stock}股票{time_level}级别历史BOLL指标数据出错: {str(e)}")
            logger.debug(f"错误数据内容: {data_dicts if 'data_dicts' in locals() else '未获取到数据'}")
            return {'创建': 0, '更新': 0, '跳过': 0}
        
    async def fetch_and_save_history_boll_by_stock_code(self, stock_code: str) -> Optional[StockBOLLIndicator]:
        """
        从API获取并保存股票历史BOLL指标数据
        """
        # 获取股票信息
        stock = await self.stock_basic_dao.get_stock_by_code(stock_code)
        if not stock:
            return {'创建': 0, '更新': 0, '跳过': 0}
        try:
            data_dicts = []
            total_result = {'创建': 0, '更新': 0, '跳过': 0}
            for time_level in TIME_LEVELS:
                api_datas = await self.api.get_history_boll(stock_code, time_level)
                for api_data in api_datas:
                    data_dict = {
                        'stock': stock,
                        'time_level': time_level,
                        'trade_time': self._parse_datetime(api_data.get('t')),  # 交易时间
                        'upper': self._parse_number(api_data.get('u')),  # 上轨
                        'lower': self._parse_number(api_data.get('d')),  # 下轨
                        'mid': self._parse_number(api_data.get('m')),  # 中轨
                    }
                    data_dicts.append(data_dict)
                    
                    # 当数据量超过10万时，保存一次
                    if len(data_dicts) >= 100000:
                        logger.info(f"数据量达到{len(data_dicts)}，开始保存批次数据")
                        batch_result = await self._save_all_to_db(
                            model_class=StockBOLLIndicator,
                            data_list=data_dicts,
                            unique_fields=['stock', 'time_level', 'trade_time']
                        )
                        logger.info(f"批次数据保存完成，结果: {batch_result}")
                        
                        # 累加结果
                        for key in total_result:
                            total_result[key] += batch_result.get(key, 0)
                        
                        # 清空数据列表，准备下一批
                        data_dicts = []
                    
                    logger.warning(f"当前data_dicts总量: {len(data_dicts)}")
                if not api_datas:
                    logger.warning(f"API未返回{stock.stock_code}股票的{time_level}级别历史BOLL指标数据")
            
            # 保存剩余数据
            if data_dicts:
                logger.info(f"开始保存剩余{len(data_dicts)}条数据")
                final_result = await self._save_all_to_db(
                    model_class=StockBOLLIndicator,
                    data_list=data_dicts,
                    unique_fields=['stock', 'time_level', 'trade_time']
                )
                logger.info(f"剩余数据保存完成，结果: {final_result}")
                
                # 累加最终结果
                for key in total_result:
                    total_result[key] += final_result.get(key, 0)
            
            logger.info(f"所有股票历史BOLL指标数据保存完成，总结果: {total_result}")
            return total_result
        except Exception as e:
            logger.error(f"保存{stock}股票历史BOLL指标数据出错: {str(e)}")
            logger.debug(f"错误数据内容: {data_dicts if 'data_dicts' in locals() else '未获取到数据'}")
            return {'创建': 0, '更新': 0, '跳过': 0}

    async def fetch_and_save_history_boll_by_time_level(self, time_level: str) -> Optional[StockBOLLIndicator]:
        """
        从API获取并保存所有股票历史BOLL指标数据
        Args:
            time_level: 时间级别
        Returns:
            Optional[StockBOLLIndicator]: 保存结果
        """
        stocks = await self.stock_basic_dao.get_stock_list()
        try:
            data_dicts = []
            total_result = {'创建': 0, '更新': 0, '跳过': 0}
            for stock in stocks:
                for time_level in TIME_LEVELS:
                    api_datas = await self.api.get_history_boll(stock.stock_code, time_level)
                    for api_data in api_datas:
                        data_dict = {
                            'stock': stock,
                            'time_level': time_level,
                            'trade_time': self._parse_datetime(api_data.get('t')),  # 交易时间
                            'upper': self._parse_number(api_data.get('u')),  # 上轨
                            'lower': self._parse_number(api_data.get('d')),  # 下轨
                            'mid': self._parse_number(api_data.get('m')),  # 中轨
                        }
                        data_dicts.append(data_dict)
                    
                    # 当数据量超过10万时，保存一次
                    if len(data_dicts) >= 100000:
                        logger.info(f"数据量达到{len(data_dicts)}，开始保存批次数据")
                        batch_result = await self._save_all_to_db(
                            model_class=StockBOLLIndicator,
                            data_list=data_dicts,
                            unique_fields=['stock', 'time_level', 'trade_time']
                        )
                        logger.info(f"批次数据保存完成，结果: {batch_result}")
                        
                        # 累加结果
                        for key in total_result:
                            total_result[key] += batch_result.get(key, 0)
                        
                        # 清空数据列表，准备下一批
                        data_dicts = []
                    
                    logger.warning(f"当前data_dicts总量: {len(data_dicts)}")
                if not api_datas:
                    logger.warning(f"API未返回{stock.stock_code}股票的{time_level}级别历史BOLL指标数据")
            
            # 保存剩余数据
            if data_dicts:
                logger.info(f"开始保存剩余{len(data_dicts)}条数据")
                final_result = await self._save_all_to_db(
                    model_class=StockBOLLIndicator,
                    data_list=data_dicts,
                    unique_fields=['stock', 'time_level', 'trade_time']
                )
                logger.info(f"剩余数据保存完成，结果: {final_result}")
                
                # 累加最终结果
                for key in total_result:
                    total_result[key] += final_result.get(key, 0)
            
            logger.info(f"所有股票历史BOLL指标数据保存完成，总结果: {total_result}")
            return total_result
        except Exception as e:
            logger.error(f"保存{time_level}级别所有股票历史BOLL指标数据出错: {str(e)}")
            logger.debug(f"错误数据内容: {data_dicts if 'data_dicts' in locals() else '未获取到数据'}")
            return {'创建': 0, '更新': 0, '跳过': 0}

    async def fetch_and_save_favorite_stocks_history_boll(self) -> Optional[StockBOLLIndicator]:
        """
        从API获取并保存自选股历史BOLL指标数据
        """
        favorite_stocks = await self.user_dao.get_all_favorite_stocks()
        try:
            data_dicts = []
            total_result = {'创建': 0, '更新': 0, '跳过': 0}
            for stock in favorite_stocks:
                for time_level in TIME_LEVELS:
                    api_datas = await self.api.get_history_boll(stock.stock_code, time_level)
                    for api_data in api_datas:
                        data_dict = {
                            'stock': stock,
                            'time_level': time_level,
                            'trade_time': self._parse_datetime(api_data.get('t')),  # 交易时间
                            'upper': self._parse_number(api_data.get('u')),  # 上轨
                            'lower': self._parse_number(api_data.get('d')),  # 下轨
                            'mid': self._parse_number(api_data.get('m')),  # 中轨
                        }
                        data_dicts.append(data_dict)
                    
                    # 当数据量超过10万时，保存一次
                    if len(data_dicts) >= 100000:
                        logger.info(f"数据量达到{len(data_dicts)}，开始保存批次数据")
                        batch_result = await self._save_all_to_db(
                            model_class=StockBOLLIndicator,
                            data_list=data_dicts,
                            unique_fields=['stock', 'time_level', 'trade_time']
                        )
                        logger.info(f"批次数据保存完成，结果: {batch_result}")
                        
                        # 累加结果
                        for key in total_result:
                            total_result[key] += batch_result.get(key, 0)
                        
                        # 清空数据列表，准备下一批
                        data_dicts = []
                    
                    logger.warning(f"当前data_dicts总量: {len(data_dicts)}")
                if not api_datas:
                    logger.warning(f"API未返回{stock.stock_code}股票的{time_level}级别历史BOLL指标数据")
            
            # 保存剩余数据
            if data_dicts:
                logger.info(f"开始保存剩余{len(data_dicts)}条数据")
                final_result = await self._save_all_to_db(
                    model_class=StockBOLLIndicator,
                    data_list=data_dicts,
                    unique_fields=['stock', 'time_level', 'trade_time']
                )
                logger.info(f"剩余数据保存完成，结果: {final_result}")
                
                # 累加最终结果
                for key in total_result:
                    total_result[key] += final_result.get(key, 0)
            
            logger.info(f"所有股票历史BOLL指标数据保存完成，总结果: {total_result}")
            return total_result
        except Exception as e:
            logger.error(f"保存自选股历史BOLL指标数据出错: {str(e)}")
            logger.debug(f"错误数据内容: {data_dicts if 'data_dicts' in locals() else '未获取到数据'}")
            return {'创建': 0, '更新': 0, '跳过': 0}
            
    async def fetch_and_save_all_history_boll(self) -> Optional[StockBOLLIndicator]:
        """
        从API获取并保存所有股票历史BOLL指标数据
        """
        stocks = await self.stock_basic_dao.get_stock_list()
        try:
            data_dicts = []
            total_result = {'创建': 0, '更新': 0, '跳过': 0}
            
            for stock in stocks:
                logger.warning(f"开始获取{stock.stock_code}股票历史BOLL指标数据")
                for time_level in TIME_LEVELS:
                    api_datas = await self.api.get_history_boll(stock.stock_code, time_level)
                    for api_data in api_datas:
                        data_dict = {
                            'stock': stock,
                            'time_level': time_level,
                            'trade_time': self._parse_datetime(api_data.get('t')),  # 交易时间
                            'upper': self._parse_number(api_data.get('u')),  # 上轨
                            'lower': self._parse_number(api_data.get('d')),  # 下轨
                            'mid': self._parse_number(api_data.get('m')),  # 中轨
                        }
                        data_dicts.append(data_dict)                    
                    # 当数据量超过10万时，保存一次
                    if len(data_dicts) >= 100000:
                        logger.info(f"数据量达到{len(data_dicts)}，开始保存批次数据")
                        batch_result = await self._save_all_to_db(
                            model_class=StockBOLLIndicator,
                            data_list=data_dicts,
                            unique_fields=['stock', 'time_level', 'trade_time']
                        )
                        logger.info(f"批次数据保存完成，结果: {batch_result}")
                        # 累加结果
                        for key in total_result:
                            total_result[key] += batch_result.get(key, 0)
                        # 清空数据列表，准备下一批
                        data_dicts = []
                    logger.warning(f"当前data_dicts总量: {len(data_dicts)}")
                if not api_datas:
                    logger.warning(f"API未返回{stock.stock_code}股票的{time_level}级别历史BOLL指标数据")
            
            # 保存剩余数据
            if data_dicts:
                logger.info(f"开始保存剩余{len(data_dicts)}条数据")
                final_result = await self._save_all_to_db(
                    model_class=StockBOLLIndicator,
                    data_list=data_dicts,
                    unique_fields=['stock', 'time_level', 'trade_time']
                )
                logger.info(f"剩余数据保存完成，结果: {final_result}")
                
                # 累加最终结果
                for key in total_result:
                    total_result[key] += final_result.get(key, 0)
            
            logger.info(f"所有股票历史BOLL指标数据保存完成，总结果: {total_result}")
            return total_result
        
        except Exception as e:
            logger.error(f"保存所有股票历史BOLL指标数据出错: {str(e)}")
            logger.debug(f"错误数据内容: {data_dicts if 'data_dicts' in locals() else '未获取到数据'}")
            return {'创建': 0, '更新': 0, '跳过': 0}

    async def refresh_boll(self, stock_code: str, time_level: Union[TimeLevel, str]) -> Optional[StockBOLLIndicator]:
        """
        刷新BOLL指标数据
        
        Args:
            stock_code: 股票代码
            time_level: 时间级别
            
        Returns:
            Optional[StockBOLLIndicator]: 最新的BOLL指标数据
        """
        return await self._refresh_indicator(
            model_class=StockBOLLIndicator,
            stock_code=stock_code,
            time_level=time_level,
            api_method=self.api.get_boll_data,
            mapping=BOLL_INDICATOR_MAPPING,
            cache_prefix="boll"
        )
