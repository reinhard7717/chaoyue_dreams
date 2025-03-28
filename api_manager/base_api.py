# api_manager/baseapi.py

import json
import logging
import asyncio
import aiohttp
import re
import time
from typing import Dict, Any, Optional, List, Union
from django.conf import settings
from bs4 import BeautifulSoup
from datetime import datetime

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
        self._session = None
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
        
        # 错误类型配置
        self._error_configs = {
            'rate_limit': {
                'max_retries': 5,
                'base_delay': 5.0,
                'delay_factor': 2.0,
                'max_delay': 60.0
            },
            'network': {
                'max_retries': 3,
                'base_delay': 2.0,
                'delay_factor': 1.5,
                'max_delay': 10.0
            },
            'timeout': {
                'max_retries': 3,
                'base_delay': 2.0,
                'delay_factor': 1.5,
                'max_delay': 10.0
            },
            'server_error': {
                'max_retries': 3,
                'base_delay': 2.0,
                'delay_factor': 1.5,
                'max_delay': 10.0
            }
        }
        
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

        # 响应解析相关正则表达式
        self._key_value_pattern = re.compile(r'([^:=\s]+)[:\s=]+(.+?)(?=\n|$)')
        self._table_pattern = re.compile(r'<table.*?>(.*?)</table>', re.DOTALL)
        self._row_pattern = re.compile(r'<tr.*?>(.*?)</tr>', re.DOTALL)
        self._cell_pattern = re.compile(r'<t[dh].*?>(.*?)</t[dh]>', re.DOTALL)
        
        logger.info(f"初始化BaseAPI，基础URL: {self.base_url}，用户类型: {self._user_type}")
    
    @property
    async def session(self):
        """
        获取或创建aiohttp会话
        """
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session
    
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
                          retry_count: int = 0, original_url: str = None, error_type: str = None,
                          expected_type: str = None) -> Any:
        """
        发送HTTP请求，包含重试机制和错误处理
        
        Args:
            method: HTTP方法
            url: 请求URL
            params: URL参数
            data: 请求体数据
            retry_count: 当前重试次数
            original_url: 原始URL（用于重试）
            error_type: 错误类型（用于重试）
            expected_type: 期望的返回数据类型('list', 'dict', None等)
            
        Returns:
            Any: 响应数据
        """
        try:
            # 获取会话
            session = await self.session
            
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
            
            logger.debug(f"请求URL: {full_url}, API类型: {api_type}, 用户类型: {self._user_type}")
            
            async with session.request(
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
                    
                    # 获取频率限制配置
                    error_config = self._error_configs['rate_limit']
                    max_retries = error_config['max_retries']
                    base_delay = error_config['base_delay']
                    delay_factor = error_config['delay_factor']
                    max_delay = error_config['max_delay']
                    
                    # 计算下次重试时间
                    retry_after = response.headers.get('Retry-After')
                    retry_delay = base_delay * (delay_factor ** retry_count)
                    retry_delay = min(retry_delay, max_delay)
                    
                    # 如果服务器指定了重试时间，优先使用
                    if retry_after:
                        try:
                            retry_delay = int(retry_after)
                        except ValueError:
                            pass
                    
                    if retry_count < max_retries:
                        logger.info(f"将在 {retry_delay:.1f} 秒后重试请求 ({retry_count + 1}/{max_retries})")
                        await asyncio.sleep(retry_delay)
                        
                        # 递归调用自身重试请求
                        return await self._make_request(
                            method, 
                            original_url, 
                            params, 
                            data, 
                            retry_count + 1, 
                            original_url,
                            'rate_limit'
                        )
                    else:
                        logger.error(f"达到最大重试次数({max_retries})，请求失败")
                        return {"error": "请求频率过高，请稍后再试"}
                        
                elif response.status == 404:
                    logger.warning(f"资源不存在(404): {full_url}, 响应: {text}")
                    # 报告错误
                    self.licence_manager.report_error(licence)
                    # 尝试将错误响应解析为JSON
                    try:
                        return json.loads(text)
                    except json.JSONDecodeError:
                        return text
                elif response.status >= 500:
                    logger.error(f"服务器错误 {response.status}: {full_url}, 响应: {text}")
                    # 报告错误
                    self.licence_manager.report_error(licence)
                    
                    # 获取服务器错误配置
                    error_config = self._error_configs['server_error']
                    max_retries = error_config['max_retries']
                    base_delay = error_config['base_delay']
                    delay_factor = error_config['delay_factor']
                    max_delay = error_config['max_delay']
                    
                    if retry_count < max_retries:
                        retry_delay = base_delay * (delay_factor ** retry_count)
                        retry_delay = min(retry_delay, max_delay)
                        logger.info(f"服务器错误，将在 {retry_delay:.1f} 秒后重试 ({retry_count + 1}/{max_retries})")
                        await asyncio.sleep(retry_delay)
                        return await self._make_request(
                            method, 
                            original_url, 
                            params, 
                            data, 
                            retry_count + 1, 
                            original_url,
                            'server_error'
                        )
                    else:
                        return {"error": "服务器错误，请稍后再试"}
                elif response.status >= 400:
                    logger.error(f"HTTP错误 {response.status}: {full_url}, 响应: {text}")
                    # 报告错误
                    self.licence_manager.report_error(licence)
                    return text
                else:
                    # 成功请求，重置错误计数
                    self.licence_manager.reset_error_count(licence)
                    
                    # 检查响应内容是否暗示错误，尽管状态码是200
                    if self._is_rate_limited_response(text):
                        logger.warning(f"检测到状态码为200但内容暗示频率限制: {text[:100]}")
                        
                        # 报告错误
                        self.licence_manager.report_error(licence)
                        
                        # 获取频率限制配置
                        error_config = self._error_configs['rate_limit']
                        max_retries = error_config['max_retries']
                        base_delay = error_config['base_delay']
                        delay_factor = error_config['delay_factor']
                        max_delay = error_config['max_delay']
                        
                        if retry_count < max_retries:
                            retry_delay = base_delay * (delay_factor ** retry_count)
                            retry_delay = min(retry_delay, max_delay)
                            logger.info(f"将在 {retry_delay:.1f} 秒后重试请求 ({retry_count + 1}/{max_retries})")
                            await asyncio.sleep(retry_delay)
                            return await self._make_request(
                                method, 
                                original_url, 
                                params, 
                                data, 
                                retry_count + 1, 
                                original_url,
                                'rate_limit'
                            )
                        else:
                            return {"error": "请求频率过高，请稍后再试"}
                    
                    # 解析响应内容
                    parsed_data = self._parse_response(text, content_type, expected_type)
                    return parsed_data
        except aiohttp.ClientError as e:
            logger.error(f"HTTP客户端错误: {str(e)}")
            # 报告错误
            if 'licence' in locals():
                self.licence_manager.report_error(licence)
                
            # 获取网络错误配置
            error_config = self._error_configs['network']
            max_retries = error_config['max_retries']
            base_delay = error_config['base_delay']
            delay_factor = error_config['delay_factor']
            max_delay = error_config['max_delay']
            
            if retry_count < max_retries:
                retry_delay = base_delay * (delay_factor ** retry_count)
                retry_delay = min(retry_delay, max_delay)
                logger.info(f"网络错误，将在 {retry_delay:.1f} 秒后重试 ({retry_count + 1}/{max_retries})")
                await asyncio.sleep(retry_delay)
                return await self._make_request(
                    method, 
                    original_url, 
                    params, 
                    data, 
                    retry_count + 1, 
                    original_url,
                    'network'
                )
            else:
                return {"error": f"网络错误: {str(e)}"}
        except asyncio.TimeoutError:
            logger.error(f"请求超时: {url}")
            # 报告错误
            if 'licence' in locals():
                self.licence_manager.report_error(licence)
                
            # 获取超时错误配置
            error_config = self._error_configs['timeout']
            max_retries = error_config['max_retries']
            base_delay = error_config['base_delay']
            delay_factor = error_config['delay_factor']
            max_delay = error_config['max_delay']
            
            if retry_count < max_retries:
                retry_delay = base_delay * (delay_factor ** retry_count)
                retry_delay = min(retry_delay, max_delay)
                logger.info(f"请求超时，将在 {retry_delay:.1f} 秒后重试 ({retry_count + 1}/{max_retries})")
                await asyncio.sleep(retry_delay)
                return await self._make_request(
                    method, 
                    original_url, 
                    params, 
                    data, 
                    retry_count + 1, 
                    original_url,
                    'timeout'
                )
            else:
                return {"error": "请求超时"}
        except Exception as e:
            logger.error(f"API请求出错: {str(e)}")
            # 报告错误
            if 'licence' in locals():
                self.licence_manager.report_error(licence)
            return {"error": f"未知错误: {str(e)}"}

    def _parse_response(self, response_text: str, content_type: str = '', expected_type: str = None) -> Any:
        """
        解析API响应，处理不同的数据格式
        
        Args:
            response_text: API响应文本
            content_type: 响应内容类型
            expected_type: 期望的数据类型(list, dict, str等)
            
        Returns:
            解析后的结构化数据
        """
        # 如果响应为空，返回空结果
        if not response_text:
            return {} if expected_type == 'dict' else [] if expected_type == 'list' else None
        
        # 尝试解析JSON
        if response_text.strip().startswith('{') or response_text.strip().startswith('[') or 'application/json' in content_type:
            try:
                data = json.loads(response_text)
                return data
            except json.JSONDecodeError:
                logger.warning(f"JSON解析失败: {response_text[:100]}...")
        
        # 检查是否为HTML响应
        if '<html' in response_text.lower() or '<table' in response_text.lower() or 'text/html' in content_type:
            return self._parse_html_response(response_text, expected_type)
        
        # 检查是否为键值对文本
        if ('=' in response_text or ':' in response_text) and '\n' in response_text:
            result = self._parse_key_value_text(response_text)
            # 如果期望列表但解析出字典
            if expected_type == 'list' and isinstance(result, dict) and result:
                return [result]
            return result
        
        # 检查是否为CSV格式
        if ',' in response_text and '\n' in response_text:
            return self._parse_csv_text(response_text)
        
        # 返回原始响应
        logger.debug(f"无法识别的响应格式，返回原始内容")
        return response_text
    
    def _parse_html_response(self, html_content: str, expected_type: str = None) -> Union[List[Dict[str, Any]], Dict[str, Any]]:
        """
        解析HTML响应
        
        Args:
            html_content: HTML内容
            expected_type: 期望的数据类型
            
        Returns:
            解析后的数据结构
        """
        try:
            # 使用BeautifulSoup解析
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # 查找表格
            tables = soup.find_all('table')
            if tables:
                # 有表格，解析为列表
                result = []
                # 获取第一个表格(通常是主要数据)
                table = tables[0]
                
                # 获取表头
                headers = []
                header_row = table.find('tr')
                if header_row:
                    headers = [th.get_text().strip() for th in header_row.find_all(['th', 'td'])]
                
                # 解析数据行
                for row in table.find_all('tr')[1:] if headers else table.find_all('tr'):
                    cells = [td.get_text().strip() for td in row.find_all(['td', 'th'])]
                    if headers:
                        # 使用表头作为键
                        row_data = {headers[i]: self._convert_value(cells[i]) for i in range(min(len(headers), len(cells)))}
                    else:
                        # 没有表头，使用索引作为键
                        row_data = {f"column_{i}": self._convert_value(cell) for i, cell in enumerate(cells)}
                    
                    result.append(row_data)
                
                return result
            else:
                # 无表格，尝试解析为键值对
                result = {}
                # 查找所有段落和div
                elements = soup.find_all(['p', 'div', 'span'])
                
                for element in elements:
                    text = element.get_text().strip()
                    # 查找冒号或等号分隔的键值对
                    match = re.search(r'([^:=]+)[:\s=]+(.+)', text)
                    if match:
                        key = match.group(1).strip()
                        value = match.group(2).strip()
                        result[key] = self._convert_value(value)
                
                # 如果期望列表但解析出字典
                if expected_type == 'list' and isinstance(result, dict):
                    return [result] if result else []
                
                return result
        except Exception as e:
            logger.error(f"HTML解析失败: {str(e)}")
            return [] if expected_type == 'list' else {}
    
    def _parse_key_value_text(self, text_content: str) -> Dict[str, Any]:
        """
        解析键值对文本
        
        Args:
            text_content: 文本内容
            
        Returns:
            Dict[str, Any]: 解析后的字典
        """
        result = {}
        
        try:
            # 按行分割
            lines = text_content.strip().split('\n')
            
            for line in lines:
                # 跳过空行
                if not line.strip():
                    continue
                    
                # 匹配键值对
                match = self._key_value_pattern.search(line)
                if match:
                    key = match.group(1).strip()
                    value = match.group(2).strip()
                    result[key] = self._convert_value(value)
        except Exception as e:
            logger.error(f"键值对文本解析失败: {str(e)}")
        
        return result
    
    def _parse_csv_text(self, text_content: str) -> List[Dict[str, Any]]:
        """
        解析CSV格式文本
        
        Args:
            text_content: CSV文本内容
            
        Returns:
            List[Dict[str, Any]]: 解析后的数据列表
        """
        result = []
        
        try:
            # 按行分割
            lines = [line.strip() for line in text_content.strip().split('\n') if line.strip()]
            
            if not lines:
                return result
                
            # 解析表头
            headers = [h.strip() for h in lines[0].split(',')]
            
            # 解析数据行
            for i in range(1, len(lines)):
                values = lines[i].split(',')
                row_data = {}
                
                for j in range(min(len(headers), len(values))):
                    row_data[headers[j]] = self._convert_value(values[j].strip())
                    
                result.append(row_data)
        except Exception as e:
            logger.error(f"CSV文本解析失败: {str(e)}")
        
        return result
    
    def _convert_value(self, value_str: str) -> Any:
        """
        尝试将字符串值转换为适当的类型
        
        Args:
            value_str: 字符串值
            
        Returns:
            Any: 转换后的值
        """
        if not value_str or not isinstance(value_str, str):
            return value_str
            
        # 去除两端空白
        value_str = value_str.strip()
        
        # 尝试转换为数字
        try:
            # 检查是否包含小数点
            if '.' in value_str:
                # 尝试转换为浮点数
                return float(value_str)
            else:
                # 尝试转换为整数
                return int(value_str)
        except ValueError:
            pass
            
        # 尝试转换为布尔值
        if value_str.lower() in ('true', 'yes', '是', 't', 'y'):
            return True
        elif value_str.lower() in ('false', 'no', '否', 'f', 'n'):
            return False
            
        # 尝试转换为日期时间
        for fmt in ['%Y-%m-%d %H:%M:%S', '%Y-%m-%d', '%H:%M:%S']:
            try:
                return datetime.strptime(value_str, fmt)
            except ValueError:
                pass
                
        # 无法转换，保持字符串类型
        return value_str
    
    async def get(self, url: str, params: Dict = None, expected_type: str = None) -> Any:
        """
        发送GET请求
        
        Args:
            url: 请求URL
            params: URL参数
            expected_type: 期望的返回数据类型('list', 'dict', None等)
            
        Returns:
            Any: 响应数据
        """
        return await self._make_request('GET', url, params=params, expected_type=expected_type)
    
    async def post(self, url: str, data: Dict = None, expected_type: str = None) -> Any:
        """
        发送POST请求
        
        Args:
            url: 请求URL
            data: 请求体数据
            expected_type: 期望的返回数据类型('list', 'dict', None等)
            
        Returns:
            Any: 响应数据
        """
        return await self._make_request('POST', url, data=data, expected_type=expected_type)
    
    async def close(self):
        """
        关闭HTTP会话
        """
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None
            logger.debug("HTTP会话已关闭")
            
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
