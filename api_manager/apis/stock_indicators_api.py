import asyncio
import logging
from typing import Dict, List, Any, Optional, Union
from enum import Enum

from api_manager.base_api import BaseAPI


logger = logging.getLogger(__name__)

class TimeLevel(str, Enum):
    """股票分时级别枚举"""
    MIN_5 = "5"         # 5分钟
    MIN_15 = "15"       # 15分钟
    MIN_30 = "30"       # 30分钟
    MIN_60 = "60"       # 60分钟
    DAY = "Day"         # 日线
    DAY_QFQ = "Day_qfq" # 日线前复权
    DAY_HFQ = "Day_hfq" # 日线后复权
    WEEK = "Week"       # 周线
    WEEK_QFQ = "Week_qfq" # 周线前复权
    WEEK_HFQ = "Week_hfq" # 周线后复权
    MONTH = "Month"     # 月线
    MONTH_QFQ = "Month_qfq" # 月线前复权
    MONTH_HFQ = "Month_hfq" # 月线后复权
    YEAR = "Year"       # 年线
    YEAR_QFQ = "Year_qfq" # 年线前复权
    YEAR_HFQ = "Year_hfq" # 年线后复权


class StockIndicatorsAPI(BaseAPI):
    """
    股票分时和技术指标API
    提供股票分时交易数据和各类技术指标的API接口调用
    """
    async def get_time_trade(self, stock_code: str, time_level: Union[TimeLevel, str]) -> Dict[str, Any]:
        """
        获取最新分时交易数据
        Args:
            stock_code: 股票代码
            time_level: 分时级别，可以是TimeLevel枚举或对应的字符串
        Returns:
            Dict[str, Any]: 最新分时交易数据
        """
        endpoint = f"/data/time/real/time/{stock_code}/{time_level}"
        # logger.info(f"获取最新分时交易数据: {stock_code}, 级别: {time_level}")
        return await self.get(endpoint, expected_type='dict')
    
    async def get_kdj(self, stock_code: str, time_level: Union[TimeLevel, str]) -> Dict[str, Any]:
        """
        获取最新分时KDJ(9,3,3)指标数据
        Args:
            stock_code: 股票代码
            time_level: 分时级别，可以是TimeLevel枚举或对应的字符串
        Returns:
            Dict[str, Any]: 最新KDJ指标数据
        """
        endpoint = f"/data/time/real/kdj/{stock_code}/{time_level}"
        # logger.info(f"获取最新KDJ指标数据: {stock_code}, 级别: {time_level}")
        return await self.get(endpoint, expected_type='dict')
    
    async def get_macd(self, stock_code: str, time_level: Union[TimeLevel, str]) -> Dict[str, Any]:
        """
        获取最新分时MACD指标数据
        Args:
            stock_code: 股票代码
            time_level: 分时级别，可以是TimeLevel枚举或对应的字符串
        Returns:
            Dict[str, Any]: 最新MACD指标数据
        """
        endpoint = f"/data/time/real/macd/{stock_code}/{time_level}"
        # logger.info(f"获取最新MACD指标数据: {stock_code}, 级别: {time_level}")
        return await self.get(endpoint, expected_type='dict')
    
    async def get_ma(self, stock_code: str, time_level: Union[TimeLevel, str]) -> Dict[str, Any]:
        """
        获取最新分时MA指标数据
        Args:
            stock_code: 股票代码
            time_level: 分时级别，可以是TimeLevel枚举或对应的字符串
        Returns:
            Dict[str, Any]: 最新MA指标数据
        """
        endpoint = f"/data/time/real/ma/{stock_code}/{time_level}"
        # logger.info(f"获取最新MA指标数据: {stock_code}, 级别: {time_level}")
        return await self.get(endpoint, expected_type='dict')
    
    async def get_boll(self, stock_code: str, time_level: Union[TimeLevel, str]) -> Dict[str, Any]:
        """
        获取最新分时BOLL(20,2)指标数据
        Args:
            stock_code: 股票代码
            time_level: 分时级别，可以是TimeLevel枚举或对应的字符串
        Returns:
            Dict[str, Any]: 最新BOLL指标数据
        """
        endpoint = f"/data/time/real/boll/{stock_code}/{time_level}"
        # logger.info(f"获取最新BOLL指标数据: {stock_code}, 级别: {time_level}")
        return await self.get(endpoint, expected_type='dict')
    
    async def get_history_trade(self, stock_code: str, time_level: Union[str]) -> List[Dict[str, Any]]:
        """
        获取历史分时交易数据，增加对返回数据量为1的重试机制。
        Args:
            stock_code: 股票代码
            time_level: 分时级别 (字符串)
        Returns:
            List[Dict[str, Any]]: 历史分时交易数据列表，或包含错误信息的字典
        """
        endpoint = f"/data/time/history/trade/{stock_code}/{time_level}"
        
        # 初始尝试获取数据
        api_data = await self.get(endpoint, expected_type='list')
        
        # 检查初始获取结果是否需要重试
        if isinstance(api_data, list) and len(api_data) == 1:
            logger.warning(f"获取历史分时交易数据 {stock_code} ({time_level}) 初始仅返回1条记录，启动重试机制...")
            
            for attempt in range(self.max_retry_count):
                # 计算重试延迟
                retry_delay = min(
                    self.retry_delay * (self.retry_delay_factor ** attempt),
                    self.max_retry_delay
                )
                logger.info(f"将在 {retry_delay:.1f} 秒后重试 ({attempt + 1}/{self.max_retry_count}) 获取 {stock_code} ({time_level})")
                await asyncio.sleep(retry_delay)
                
                # 执行重试获取
                retry_api_data = await self.get(endpoint, expected_type='list')
                
                # 检查重试结果
                if isinstance(retry_api_data, list) and len(retry_api_data) > 1:
                    logger.info(f"重试成功: 获取历史分时交易数据 {stock_code} ({time_level}) 成功获取 {len(retry_api_data)} 条记录。")
                    api_data = retry_api_data # 使用重试成功的数据
                    break # 成功获取到多于1条数据，退出重试循环
                elif isinstance(retry_api_data, list) and len(retry_api_data) == 1:
                    logger.warning(f"重试 {attempt + 1}/{self.max_retry_count} 后，获取 {stock_code} ({time_level}) 仍只返回1条记录。")
                    # 继续下一次重试
                else:
                    # 如果重试返回错误或非列表类型，记录错误并停止重试，保留原始的单条数据
                    logger.error(f"重试 {attempt + 1}/{self.max_retry_count} 获取 {stock_code} ({time_level}) 失败或返回非预期类型: {retry_api_data}")
                    # 保留第一次获取的单条数据 api_data
                    break 
            else:
                # 如果循环正常结束（即所有重试都失败了，仍然是1条数据）
                logger.warning(f"获取历史分时交易数据 {stock_code} ({time_level}) 在重试 {self.max_retry_count} 次后仍只返回1条记录。将使用此单条记录。")
                
        # 对最终获取的数据进行处理（排序）
        if isinstance(api_data, list):
            # 确保 'd' 键存在，如果不存在则使用空字符串或其他默认值排序，避免 KeyError
            api_data.sort(key=lambda x: x.get('d', ''), reverse=True) 
            logger.info(f"最终获取历史分时交易数据: {stock_code}, 级别: {time_level}, 数据量: {len(api_data)}")
        elif isinstance(api_data, dict) and 'error' in api_data:
             logger.error(f"获取历史分时交易数据失败: {stock_code}, 级别: {time_level}, 错误: {api_data['error']}")
        else:
             # 如果 api_data 不是列表也不是包含 error 的字典，记录警告
             logger.warning(f"获取历史分时交易数据返回非预期类型: {stock_code}, 级别: {time_level}, 类型: {type(api_data)}, 数据: {str(api_data)[:100]}...") # 记录部分数据以供调试
             # 根据业务需求，这里可以返回空列表或原始数据
             # return [] 

        return api_data
    
    async def get_history_kdj(self, stock_code: str, time_level: Union[TimeLevel, str]) -> List[Dict[str, Any]]:
        """
        获取历史分时KDJ(9,3,3)指标数据
        Args:
            stock_code: 股票代码
            time_level: 分时级别，可以是TimeLevel枚举或对应的字符串
        Returns:
            List[Dict[str, Any]]: 历史KDJ指标数据列表
        """
        endpoint = f"/data/time/history/kdj/{stock_code}/{time_level}"
        # logger.info(f"获取历史KDJ指标数据: {stock_code}, 级别: {time_level}")
        api_data = await self.get(endpoint, expected_type='list')
        api_data.sort(key=lambda x: x['t'], reverse=True)
        return api_data
    
    async def get_history_macd(self, stock_code: str, time_level: Union[TimeLevel, str]) -> List[Dict[str, Any]]:
        """
        获取历史分时MACD指标数据
        Args:
            stock_code: 股票代码
            time_level: 分时级别，可以是TimeLevel枚举或对应的字符串
        Returns:
            List[Dict[str, Any]]: 历史MACD指标数据列表
        """
        endpoint = f"/data/time/history/macd/{stock_code}/{time_level}"
        # logger.info(f"获取历史MACD指标数据: {stock_code}, 级别: {time_level}")
        api_data = await self.get(endpoint, expected_type='list')
        api_data.sort(key=lambda x: x['t'], reverse=True)
        return api_data
    
    async def get_history_ma(self, stock_code: str, time_level: Union[TimeLevel, str]) -> List[Dict[str, Any]]:
        """
        获取历史分时MA指标数据
        Args:
            stock_code: 股票代码
            time_level: 分时级别，可以是TimeLevel枚举或对应的字符串
        Returns:
            List[Dict[str, Any]]: 历史MA指标数据列表
        """
        endpoint = f"/data/time/history/ma/{stock_code}/{time_level}"
        # logger.info(f"获取历史MA指标数据: {stock_code}, 级别: {time_level}")
        api_data = await self.get(endpoint, expected_type='list')
        api_data.sort(key=lambda x: x['t'], reverse=True)
        return api_data
    
    async def get_history_boll(self, stock_code: str, time_level: Union[TimeLevel, str]) -> List[Dict[str, Any]]:
        """
        获取历史分时BOLL(20,2)指标数据
        Args:
            stock_code: 股票代码
            time_level: 分时级别，可以是TimeLevel枚举或对应的字符串
        Url: /data/time/history/boll/{stock_code}/{time_level}
        Returns:
            List[Dict[str, Any]]: 历史BOLL指标数据列表
        """
        endpoint = f"/data/time/history/boll/{stock_code}/{time_level}"
        # logger.info(f"获取历史BOLL指标数据: {stock_code}, 级别: {time_level}")
        api_data = await self.get(endpoint, expected_type='list')
        api_data.sort(key=lambda x: x['t'], reverse=True)
        return api_data
