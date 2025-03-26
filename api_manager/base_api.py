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
    基础API类，提供通用的HTTP请求方法
    """
    
    def __init__(self):
        self.session = None
        self.base_url = settings.API_BASE_URL if hasattr(settings, 'API_BASE_URL') else ""
        self.headers = {}
        self.timeout = settings.API_REQUEST_TIMEOUT
        self.licence_manager = LicenceManager()
        logger.info(f"初始化BaseAPI，基础URL: {self.base_url}")
    
    async def _make_request(self, method: str, url: str, params: Dict = None, data: Dict = None) -> Any:
        """
        发送HTTP请求
        
        Args:
            method: 请求方法（GET、POST等）
            url: 请求URL
            params: URL参数
            data: 请求体数据
            
        Returns:
            Any: 响应数据
        """
        session_created = False
        try:
            if not self.session:
                self.session = aiohttp.ClientSession()
                session_created = True
            
            # 获取licence并添加到URL中
            licence = self.licence_manager.get_licence()
            
            # 确保base_url不以斜杠结尾，url不以斜杠开头
            base_url = self.base_url.rstrip('/')
            url = url.lstrip('/')
            
            # 清理URL中的股票名称部分
            if '-' in url:
                url = url.split('-')[0]
            
            # 构建完整的URL
            separator = '&' if '?' in url else '?'
            url_with_licence = f"{url}{separator}licence={licence}"
            full_url = f"{base_url}/{url_with_licence}"
            logger.warning(f"请求URL: {full_url}")
            async with self.session.request(method, full_url, params=params, json=data, headers=self.headers, timeout=self.timeout) as response:
                if response.status == 404:
                    error_text = await response.text()
                    logger.warning(f"资源不存在(404): {full_url}, 响应: {error_text}")
                    # 尝试将错误响应解析为JSON
                    try:
                        return json.loads(error_text)
                    except json.JSONDecodeError:
                        return error_text
                else:
                    # 获取响应内容
                    content_type = response.headers.get('Content-Type', '')
                    text = await response.text()
                    logger.debug(f"响应状态码: {response.status}, 响应内容类型: {content_type}")
                    return text
        except aiohttp.ClientError as e:
            logger.error(f"HTTP客户端错误: {str(e)}")
            return f"HTTP客户端错误: {str(e)}"
        except asyncio.TimeoutError:
            logger.error(f"请求超时: {url}")
            return "请求超时"
        except Exception as e:
            logger.error(f"API请求出错: {str(e)}")
            return f"未知错误: {str(e)}"
        finally:
            # 如果是新创建的会话且只用于一次请求，则关闭它
            if session_created:
                await self.close()
    
    async def get(self, url: str, params: Dict = None) -> Any:
        """
        发送GET请求
        
        Args:
            url: 请求URL
            params: URL参数
            
        Returns:
            Any: 响应数据
        """
        return await self._make_request('GET', url, params=params)
    
    async def post(self, url: str, data: Dict = None) -> Any:
        """
        发送POST请求
        
        Args:
            url: 请求URL
            data: 请求体数据
            
        Returns:
            Any: 响应数据
        """
        return await self._make_request('POST', url, data=data)
    
    async def close(self):
        """
        关闭HTTP会话
        """
        if self.session:
            await self.session.close()
            self.session = None
            
    async def __aenter__(self):
        """
        异步上下文管理器入口
        """
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """
        异步上下文管理器出口，确保会话关闭
        """
        await self.close()
