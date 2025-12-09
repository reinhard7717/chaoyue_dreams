# 修改版 api_manager/baseapi.py

import json
import logging
import asyncio
import aiohttp
import re
from typing import Dict, Any, Optional, List, Union
from django.conf import settings

logger = logging.getLogger(__name__)

class BaseAPI:
    """
    简化版基础API类，提供轮流使用license的请求功能和重试机制
    """
    def __init__(self):
        """初始化BaseAPI"""
        self._session = None
        # 修改为http方式
        base_url = settings.API_BASE_URL if hasattr(settings, 'API_BASE_URL') else ""
        if base_url.startswith('https://'):
            base_url = base_url.replace('https://', 'http://')
        self.base_url = base_url
        self.timeout = getattr(settings, 'API_REQUEST_TIMEOUT', 30)
        # license管理
        self.licences = getattr(settings, 'API_LICENCES_IG507', [])
        if not self.licences:
            logger.warning("未在settings中找到API_LICENCES_IG507配置，或配置为空列表")
        # 各license的使用情况记录和冷却状态
        self.licence_usage = {lic: {"count": 0, "last_used": 0, "errors": 0, "cooldown_until": 0} 
                              for lic in self.licences}
        # API类型和URL模式映射
        self.url_patterns = getattr(settings, 'API_URL_PATTERNS', {})
        # API类型对应的访问限制配置
        self.api_limits = getattr(settings, 'API_LIMITS', {
            'default': {'max_requests': 50, 'window_seconds': 60}
        })
        # 重试配置
        self.max_retry_count = getattr(settings, 'API_MAX_RETRY_COUNT', 5)
        self.retry_delay = getattr(settings, 'API_RETRY_DELAY', 2.0)
        self.retry_delay_factor = getattr(settings, 'API_RETRY_DELAY_FACTOR', 1.5)
        self.max_retry_delay = getattr(settings, 'API_MAX_RETRY_DELAY', 30.0)
        # 频率限制错误模式匹配
        self.rate_limit_patterns = [
            r'503请求过于频繁',
            r'请求频率过高',
            r'超出请求限制',
            r'请求过于频繁',
            r'too many requests',
            r'rate limit exceeded',
            r'请稍后再试',
        ]
        # logger.info(f"初始化BaseAPI，基础URL: {self.base_url}，可用license数: {len(self.licences)}")
    @property
    async def session(self):
        """获取或创建aiohttp会话"""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session
    def _detect_api_type(self, url: str) -> str:
        """根据URL识别API类型"""
        if url is None:
            logger.warning("URL为None，使用默认API类型")
            return 'default'
        url_lower = url.lower()
        for api_type, patterns in self.url_patterns.items():
            for pattern in patterns:
                if re.search(pattern, url_lower):
                    return api_type
                    
        return 'default'
    def _is_rate_limited(self, response_text: str) -> bool:
        """判断响应是否表示请求频率受限"""
        if not response_text:
            return False
        for pattern in self.rate_limit_patterns:
            if re.search(pattern, response_text, re.IGNORECASE):
                return True
                
        return False
    def _get_licence(self, api_type: str) -> Optional[str]:
        """获取当前可用的license"""
        current_time = asyncio.get_event_loop().time()
        api_limit = self.api_limits.get(api_type, self.api_limits.get('default'))
        # 过滤出不在冷却期的license
        available_licences = [
            lic for lic in self.licences 
            if self.licence_usage[lic]["cooldown_until"] <= current_time
        ]
        if not available_licences:
            # logger.warning("所有license都在冷却期，等待最快可用的一个")
            # 找到最快可用的license
            cooldown_times = [(lic, self.licence_usage[lic]["cooldown_until"]) for lic in self.licences]
            cooldown_times.sort(key=lambda x: x[1])
            return cooldown_times[0][0]  # 返回最快可用的license
        # 按使用次数排序，优先使用次数少的
        usage_sorted = sorted(
            available_licences, 
            key=lambda lic: self.licence_usage[lic]["count"]
        )
        # 选择使用次数最少的license
        selected_licence = usage_sorted[0]
        # 更新使用记录
        self.licence_usage[selected_licence]["count"] += 1
        self.licence_usage[selected_licence]["last_used"] = current_time
        return selected_licence
    def _report_error(self, licence: str, is_rate_limit: bool = False):
        """报告license使用出错"""
        if licence not in self.licence_usage:
            return
        current_time = asyncio.get_event_loop().time()
        self.licence_usage[licence]["errors"] += 1
        # 如果是频率限制错误，进入冷却期
        if is_rate_limit:
            # 冷却时间根据错误次数增加，但不超过最大冷却时间
            cooldown_seconds = min(1 * (2 ** (self.licence_usage[licence]["errors"] - 1)), 3600)
            self.licence_usage[licence]["cooldown_until"] = current_time + cooldown_seconds
            # logger.warning(f"License {licence} 触发频率限制，进入冷却期 {cooldown_seconds} 秒")
    def _reset_errors(self, licence: str):
        """重置license错误计数"""
        if licence in self.licence_usage:
            self.licence_usage[licence]["errors"] = 0
    def _parse_response(self, text: str, expected_type: str = None) -> Any:
        """
        解析响应内容
        Args:
            text: 响应文本
            expected_type: 期望的返回数据类型('list', 'dict', None等)
        Returns:
            Any: 解析后的数据
        """
        if not text:
            return {} if expected_type == 'dict' else [] if expected_type == 'list' else None
        try:
            data = json.loads(text)
            # 类型检查与转换
            if expected_type == 'list' and isinstance(data, dict):
                # 尝试从字典中提取列表类型的字段
                for key, value in data.items():
                    if isinstance(value, list):
                        logger.debug(f"从响应字典的 '{key}' 字段中提取列表数据")
                        return value
                # 如果没有列表字段，但有items字段
                if 'items' in data:
                    return data['items']
                # 找不到列表，返回空列表
                logger.warning(f"期望列表类型，但收到字典且无法提取列表: {text[:100]}...")
                return []
                
            elif expected_type == 'dict' and isinstance(data, list):
                # 列表转字典
                if data and len(data) > 0:
                    logger.debug("将响应列表的第一项作为字典返回")
                    return data[0] if isinstance(data[0], dict) else {"value": data[0]}
                else:
                    return {}
                
            return data
        except json.JSONDecodeError as e:
            # logger.warning(f"JSON解析错误: {str(e)}, 文本: {text[:100]}...")
            # 根据期望类型返回默认值
            if expected_type == 'dict':
                return {}
            elif expected_type == 'list':
                return []
            else:
                return text
    async def _make_request(self, method: str, url: str, params: Dict = None, data: Dict = None, 
                          headers: Dict = None, expected_type: str = None, retry_count: int = 0) -> Any:
        """发送HTTP请求，带重试机制"""
        if url is None:
            logger.error("请求URL不能为None")
            return {"error": "请求URL不能为空"}
        # 保存原始URL（不含license）
        original_url = url
        try:
            # 获取会话
            session = await self.session
            # 检测API类型
            api_type = self._detect_api_type(url)
            # 获取license
            licence = self._get_licence(api_type)
            if not licence:
                logger.error("无法获取可用license")
                return {"error": "无法获取可用license"}
            # 构建完整URL
            base_url = self.base_url.rstrip('/')
            url = url.lstrip('/')
            # 清理URL，去除可能的股票名称
            if '-' in url:
                url = url.split('-')[0]
            # 添加license参数
            separator = '&' if '?' in url else '?'
            url_with_licence = f"{url}{separator}licence={licence}"
            full_url = f"{base_url}/{url_with_licence}"
            logger.debug(f"请求URL: {full_url}, API类型: {api_type}")
            # 合并请求头
            request_headers = {'Accept': 'application/json'}
            if headers:
                request_headers.update(headers)
            # 发送请求
            async with session.request(
                method, 
                full_url, 
                params=params, 
                json=data, 
                headers=request_headers, 
                timeout=self.timeout
            ) as response:
                text = await response.text()
                # 检查是否频率限制
                is_rate_limited = self._is_rate_limited(text) or response.status == 429
                if is_rate_limited:
                    # logger.warning(f"检测到频率限制，licence: {licence}")
                    self._report_error(licence, is_rate_limit=True)
                    if retry_count < self.max_retry_count:
                        retry_delay = min(
                            self.retry_delay * (self.retry_delay_factor ** retry_count),
                            self.max_retry_delay
                        )
                        # logger.info(f"将在 {retry_delay:.1f} 秒后重试请求 ({retry_count + 1}/{self.max_retry_count})")
                        await asyncio.sleep(retry_delay)
                        return await self._make_request(
                            method, original_url, params, data, headers, expected_type, retry_count + 1
                        )
                    else:
                        return {"error": "请求频率过高，请稍后再试"}
                elif response.status >= 400:
                    logger.error(f"HTTP错误 {response.status}: {full_url}")
                    self._report_error(licence)
                    if response.status >= 500 and retry_count < self.max_retry_count:
                        retry_delay = min(
                            self.retry_delay * (self.retry_delay_factor ** retry_count),
                            self.max_retry_delay
                        )
                        logger.info(f"服务器错误，将在 {retry_delay:.1f} 秒后重试 ({retry_count + 1}/{self.max_retry_count})")
                        await asyncio.sleep(retry_delay)
                        return await self._make_request(
                            method, original_url, params, data, headers, expected_type, retry_count + 1
                        )
                    # 尝试解析响应
                    parsed_data = self._parse_response(text, expected_type)
                    if isinstance(parsed_data, dict) and not parsed_data.get("error"):
                        parsed_data["error"] = f"HTTP错误 {response.status}"
                    return parsed_data
                else:
                    # 成功请求，重置错误
                    self._reset_errors(licence)
                    # 解析响应
                    return self._parse_response(text, expected_type)
                    
        except aiohttp.ClientError as e:
            logger.error(f"HTTP客户端错误: {str(e)}")
            if retry_count < self.max_retry_count:
                retry_delay = min(
                    self.retry_delay * (self.retry_delay_factor ** retry_count),
                    self.max_retry_delay
                )
                logger.info(f"网络错误，将在 {retry_delay:.1f} 秒后重试 ({retry_count + 1}/{self.max_retry_count})")
                await asyncio.sleep(retry_delay)
                return await self._make_request(
                    method, original_url, params, data, headers, expected_type, retry_count + 1
                )
            return {"error": f"网络错误: {str(e)}"}
        except asyncio.TimeoutError:
            logger.error(f"请求超时: {url}")
            if retry_count < self.max_retry_count:
                retry_delay = min(
                    self.retry_delay * (self.retry_delay_factor ** retry_count),
                    self.max_retry_delay
                )
                logger.info(f"请求超时，将在 {retry_delay:.1f} 秒后重试 ({retry_count + 1}/{self.max_retry_count})")
                await asyncio.sleep(retry_delay)
                return await self._make_request(
                    method, original_url, params, data, headers, expected_type, retry_count + 1
                )
            return {"error": "请求超时"}
        except Exception as e:
            logger.error(f"API请求出错: {str(e)}")
            return {"error": f"未知错误: {str(e)}"}
    async def get(self, url: str, params: Dict = None, headers: Dict = None, expected_type: str = None) -> Any:
        """
        发送GET请求
        Args:
            url: 请求URL
            params: URL参数
            headers: 请求头
            expected_type: 期望的返回数据类型('list', 'dict', None等)
        Returns:
            Any: 响应数据
        """
        if url is None:
            logger.error("GET请求的URL不能为None")
            return {"error": "请求URL不能为空"}
        return await self._make_request('GET', url, params=params, headers=headers, expected_type=expected_type)
    async def post(self, url: str, data: Dict = None, headers: Dict = None, expected_type: str = None) -> Any:
        """
        发送POST请求
        Args:
            url: 请求URL
            data: 请求体数据
            headers: 请求头
            expected_type: 期望的返回数据类型('list', 'dict', None等)
        Returns:
            Any: 响应数据
        """
        if url is None:
            logger.error("POST请求的URL不能为None")
            return {"error": "请求URL不能为空"}
        return await self._make_request('POST', url, data=data, headers=headers, expected_type=expected_type)
    async def close(self):
        """关闭HTTP会话"""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None
            logger.debug("HTTP会话已关闭")
    async def __aenter__(self):
        """异步上下文管理器入口"""
        return self
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口，确保会话关闭"""
        await self.close()
