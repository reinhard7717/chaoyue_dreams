# api_manager/baseapi.py

import json
import logging
import asyncio
import aiohttp
from typing import Dict, Any, Optional, List, Union
from django.conf import settings

from .licence_manager import LicenceManager

logger = logging.getLogger(__name__)

class BaseAPI:
    """
    基础API调用类，提供异步HTTP请求方法
    """
    
    def __init__(self):
        """
        初始化BaseAPI
        """
        self.base_url = settings.API_BASE_URL
        self.timeout = settings.API_REQUEST_TIMEOUT
        self.licence_manager = LicenceManager()
        logger.info(f"初始化BaseAPI，基础URL: {self.base_url}")
    
    async def _make_request(self, url: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """
        执行异步HTTP请求
        
        Args:
            url: API请求的URL
            params: 请求参数字典
            
        Returns:
            请求响应的JSON数据
        
        Raises:
            Exception: 请求失败时抛出异常
        """
        if params is None:
            params = {}
        
        # 确保params中有licence参数
        if 'licence' not in params:
            params['licence'] = self.licence_manager.get_licence()
        
        full_url = f"{self.base_url}{url}"
        logger.debug(f"请求URL: {full_url}, 参数: {params}")
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(full_url, params=params, timeout=self.timeout) as response:
                    # 检查响应状态
                    response.raise_for_status()
                    
                    # 获取响应内容
                    content_type = response.headers.get('Content-Type', '')
                    if 'application/json' in content_type:
                        data = await response.json()
                    else:
                        text = await response.text()
                        # 尝试将文本解析为JSON
                        try:
                            data = json.loads(text)
                        except json.JSONDecodeError:
                            # 如果不是JSON，则直接返回文本
                            data = text
                    
                    logger.debug(f"响应状态码: {response.status}, 响应内容类型: {content_type}")
                    return data
        except aiohttp.ClientError as e:
            logger.error(f"请求失败: {url}, 错误: {str(e)}")
            raise Exception(f"API请求失败: {str(e)}")
        except asyncio.TimeoutError:
            logger.error(f"请求超时: {url}")
            raise Exception(f"API请求超时")
        except Exception as e:
            logger.error(f"未知错误: {url}, 错误: {str(e)}")
            raise
    
    async def get(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """
        发送GET请求到指定的API端点
        
        Args:
            endpoint: API端点
            params: 请求参数
            
        Returns:
            API的响应数据
        """
        return await self._make_request(endpoint, params)
