# api_manager/baseapi.py

import json
import logging
import asyncio
import aiohttp
import re
import time
from typing import Dict, Any, Optional, List, Union
from django.conf import settings

from .licence_manager import LicenceManager

logger = logging.getLogger(__name__)

class BaseAPI:
    """
    基础API类，提供通用的HTTP请求方法，自动识别API类型
    """
    
    def __init__(self):
        """
        初始化BaseAPI
        """
        self.session = None
        self.base_url = settings.API_BASE_URL if hasattr(settings, 'API_BASE_URL') else ""
        self.headers = {}
        self.timeout = getattr(settings, 'API_REQUEST_TIMEOUT', 30)
        self.licence_manager = LicenceManager()
        
        # 检查是否有可用的license
        if not getattr(settings, 'API_LICENCES', []):
            logger.warning("未在settings中找到API_LICENCES配置，或配置为空列表")
        
        # 从settings中获取URL模式映射
        self._url_patterns = getattr(settings, 'API_URL_PATTERNS', {})
        
        # 重试机制配置
        self._max_retry_count = getattr(settings, 'API_MAX_RETRY_COUNT', 5)
        self._retry_delay = getattr(settings, 'API_RETRY_DELAY', 2.0)
        self._retry_delay_factor = getattr(settings, 'API_RETRY_DELAY_FACTOR', 1.5)
        self._max_retry_delay = getattr(settings, 'API_MAX_RETRY_DELAY', 30.0)
        
        # 频率限制错误模式匹配
        self._rate_limit_patterns = [
            r'503请求过于频繁',
            r'请求频率过高',
            r'超出请求限制',
            r'请求过于频繁',
            r'too many requests',
            r'rate limit exceeded',
            r'请稍后再试',
        ]
        
        # 默认使用专业版
        self._user_type = 'pro'
        
        logger.info(f"初始化BaseAPI，基础URL: {self.base_url}，用户类型: {self._user_type}")
    
    def _detect_api_type(self, url: str) -> str:
        """
        根据URL自动识别API类型
        
        Args:
            url: 请求URL
            
        Returns:
            str: 识别出的API类型
        """
        url_lower = url.lower()
        
        # 按照API类型检查URL模式
        for api_type, patterns in self._url_patterns.items():
            for pattern in patterns:
                if re.search(pattern, url_lower):
                    # logger.debug(f"API类型识别: {url} -> {api_type}")
                    return api_type
                    
        logger.warning(f"未能识别API类型: {url}，使用默认类型")
        return 'default'  # 默认API类型
    
    def _is_rate_limited_response(self, response_text: str) -> bool:
        """
        判断响应是否表示请求频率受限
        
        Args:
            response_text: 响应文本
            
        Returns:
            bool: 是否频率受限
        """
        # 如果响应为空，直接返回False
        if not response_text:
            return False
            
        # 检查是否匹配任一频率限制模式
        for pattern in self._rate_limit_patterns:
            if re.search(pattern, response_text, re.IGNORECASE):
                return True
                
        return False
    
    async def _get_licence_with_retry(self, api_type: str) -> str:
        """
        获取license，如果所有license都不可用，则等待一段时间后重试
        
        Args:
            api_type: API类型
            
        Returns:
            str: license字符串
        """
        retry_count = 0
        current_delay = self._retry_delay
        
        while retry_count < self._max_retry_count:
            licence = self.licence_manager.get_licence(api_type=api_type, user_type=self._user_type)
            
            # 检查license是否可用 (licence_manager返回空字符串表示无可用license)
            if licence:
                return licence
            
            # 所有license都不可用，等待后重试
            retry_count += 1
            if retry_count < self._max_retry_count:
                logger.warning(f"所有license都达到速率限制或处于冷却状态，将在{current_delay:.1f}秒后重试 ({retry_count}/{self._max_retry_count})")
                await asyncio.sleep(current_delay)
                
                # 增加下次重试的等待时间，但不超过最大等待时间
                current_delay = min(current_delay * self._retry_delay_factor, self._max_retry_delay)
            else:
                logger.error(f"达到最大重试次数({self._max_retry_count})，将强制使用license")
        
        # 如果达到最大重试次数仍然没有可用license，强制使用一个license
        return self.licence_manager.get_licence(api_type=api_type, user_type=self._user_type, ignore_limits=True)
    
    async def _make_request(self, method: str, url: str, params: Dict = None, data: Dict = None, 
                          retry_count: int = 0, original_url: str = None) -> Any:
        """
        发送HTTP请求
        
        Args:
            method: 请求方法（GET、POST等）
            url: 请求URL
            params: URL参数
            data: 请求体数据
            retry_count: 当前重试次数
            original_url: 原始URL，用于重试时跟踪
            
        Returns:
            Any: 响应数据
        """
        # 保存原始URL用于重试
        if original_url is None:
            original_url = url
            
        # 检查是否超过最大重试次数
        if retry_count > self._max_retry_count:
            logger.error(f"请求 {original_url} 超过最大重试次数 {self._max_retry_count}")
            return {"error": f"超过最大重试次数 {self._max_retry_count}", "url": original_url}
        
        session_created = False
        try:
            if not self.session:
                self.session = aiohttp.ClientSession()
                session_created = True
            
            # 自动检测API类型
            api_type = self._detect_api_type(url)
            
            # 获取licence并添加到URL中（带重试机制）
            licence = await self._get_licence_with_retry(api_type)
            if not licence:
                logger.error(f"无法获取可用license，请检查API_LICENCES配置")
                return {"error": "无法获取可用license，请检查API_LICENCES配置"}
            
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
            
            # 只在首次请求或重试时记录日志，避免日志过多
            # if retry_count == 0:
            #     logger.debug(f"请求URL: {full_url}, API类型: {api_type}, 用户类型: {self._user_type}")
            # else:
            #     logger.debug(f"重试请求({retry_count}/{self._max_retry_count}): {full_url}")
            logger.warning(f"请求URL: {full_url}, API类型: {api_type}, 用户类型: {self._user_type}")
            async with self.session.request(
                method, 
                full_url, 
                params=params, 
                json=data, 
                headers=self.headers, 
                timeout=self.timeout
            ) as response:
                # 获取响应内容
                text = await response.text()
                content_type = response.headers.get('Content-Type', '')
                
                # 检查响应内容是否表示频率限制
                if self._is_rate_limited_response(text) or response.status == 429:
                    logger.warning(f"检测到频率限制响应: {response.status}, 内容: {text[:100]}")
                    
                    # 报告错误
                    self.licence_manager.report_error(licence)
                    
                    # 计算下次重试时间
                    retry_after = response.headers.get('Retry-After')
                    retry_delay = self._retry_delay * (self._retry_delay_factor ** retry_count)
                    retry_delay = min(retry_delay, self._max_retry_delay)
                    
                    # 如果服务器指定了重试时间，优先使用
                    if retry_after:
                        try:
                            retry_delay = int(retry_after)
                        except ValueError:
                            pass
                    
                    logger.info(f"将在 {retry_delay:.1f} 秒后重试请求 ({retry_count + 1}/{self._max_retry_count})")
                    await asyncio.sleep(retry_delay)
                    
                    # 递归调用自身重试请求
                    return await self._make_request(
                        method, 
                        original_url, 
                        params, 
                        data, 
                        retry_count + 1, 
                        original_url
                    )
                elif response.status == 404:
                    logger.warning(f"资源不存在(404): {full_url}, 响应: {text}")
                    # 报告错误
                    self.licence_manager.report_error(licence)
                    # 尝试将错误响应解析为JSON
                    try:
                        return json.loads(text)
                    except json.JSONDecodeError:
                        return text
                elif response.status >= 400:
                    logger.error(f"HTTP错误 {response.status}: {full_url}, 响应: {text}")
                    # 报告错误
                    self.licence_manager.report_error(licence)
                    return text
                else:
                    # logger.debug(f"响应状态码: {response.status}, 响应内容类型: {content_type}")
                    
                    # 成功请求，重置错误计数
                    self.licence_manager.reset_error_count(licence)
                    
                    # 检查响应内容是否暗示错误，尽管状态码是200
                    if self._is_rate_limited_response(text):
                        logger.warning(f"检测到状态码为200但内容暗示频率限制: {text[:100]}")
                        
                        # 报告错误
                        self.licence_manager.report_error(licence)
                        
                        # 计算下次重试时间
                        retry_delay = self._retry_delay * (self._retry_delay_factor ** retry_count)
                        retry_delay = min(retry_delay, self._max_retry_delay)
                        
                        logger.info(f"将在 {retry_delay:.1f} 秒后重试请求 ({retry_count + 1}/{self._max_retry_count})")
                        await asyncio.sleep(retry_delay)
                        
                        # 递归调用自身重试请求
                        return await self._make_request(
                            method, 
                            original_url, 
                            params, 
                            data, 
                            retry_count + 1, 
                            original_url
                        )
                    
                    # 如果内容类型是JSON或看起来像JSON，尝试解析
                    if 'application/json' in content_type or text.strip().startswith('{') or text.strip().startswith('['):
                        try:
                            return json.loads(text)
                        except json.JSONDecodeError:
                            logger.warning(f"无法解析JSON响应: {text[:100]}...")
                            return text
                    return text
        except aiohttp.ClientError as e:
            logger.error(f"HTTP客户端错误: {str(e)}")
            # 报告错误
            if 'licence' in locals():
                self.licence_manager.report_error(licence)
                
            # 针对网络相关错误，尝试重试
            if retry_count < self._max_retry_count:
                retry_delay = self._retry_delay * (self._retry_delay_factor ** retry_count)
                retry_delay = min(retry_delay, self._max_retry_delay)
                logger.info(f"网络错误，将在 {retry_delay:.1f} 秒后重试 ({retry_count + 1}/{self._max_retry_count})")
                await asyncio.sleep(retry_delay)
                return await self._make_request(
                    method, 
                    original_url, 
                    params, 
                    data, 
                    retry_count + 1, 
                    original_url
                )
            return f"HTTP客户端错误: {str(e)}"
        except asyncio.TimeoutError:
            logger.error(f"请求超时: {url}")
            # 报告错误
            if 'licence' in locals():
                self.licence_manager.report_error(licence)
                
            # 针对超时错误，尝试重试
            if retry_count < self._max_retry_count:
                retry_delay = self._retry_delay * (self._retry_delay_factor ** retry_count)
                retry_delay = min(retry_delay, self._max_retry_delay)
                logger.info(f"请求超时，将在 {retry_delay:.1f} 秒后重试 ({retry_count + 1}/{self._max_retry_count})")
                await asyncio.sleep(retry_delay)
                return await self._make_request(
                    method, 
                    original_url, 
                    params, 
                    data, 
                    retry_count + 1, 
                    original_url
                )
            return "请求超时"
        except Exception as e:
            logger.error(f"API请求出错: {str(e)}")
            # 报告错误
            if 'licence' in locals():
                self.licence_manager.report_error(licence)
                
            # 针对其他错误，尝试重试
            if retry_count < self._max_retry_count:
                retry_delay = self._retry_delay * (self._retry_delay_factor ** retry_count)
                retry_delay = min(retry_delay, self._max_retry_delay)
                logger.info(f"未知错误，将在 {retry_delay:.1f} 秒后重试 ({retry_count + 1}/{self._max_retry_count})")
                await asyncio.sleep(retry_delay)
                return await self._make_request(
                    method, 
                    original_url, 
                    params, 
                    data, 
                    retry_count + 1, 
                    original_url
                )
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
            # logger.debug("HTTP会话已关闭")
            
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
