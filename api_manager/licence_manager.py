# api_manager/licence_manager.py

import time
import logging
import threading
from django.conf import settings
from collections import defaultdict
import math

logger = logging.getLogger(__name__)

class LicenceManager:
    """
    License管理器，用于轮番使用多个licence，并实现API访问限制和错误处理
    """
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """
        单例模式实现
        """
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(LicenceManager, cls).__new__(cls)
                cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """
        初始化License管理器
        """
        if not self._initialized:
            self._licenses = settings.API_LICENCES if hasattr(settings, 'API_LICENCES') else []
            self._current_index = 0
            self._last_use_time = {}  # 记录每个license最后使用时间
            self._request_history = defaultdict(list)  # 记录每个license请求历史
            self._error_count = defaultdict(int)  # 记录每个license错误次数
            self._is_active = {lic: True for lic in self._licenses}  # license状态
            self._cooldown_until = {}  # license冷却时间
            self._request_counts = defaultdict(lambda: defaultdict(int))  # 记录每个license不同API类型的请求次数
            
            # 从settings中获取速率限制和错误处理配置
            self._rate_limits = getattr(settings, 'API_RATE_LIMITS', {})
            self._error_settings = getattr(settings, 'API_ERROR_SETTINGS', {
                'error_threshold': 5,
                'base_cooldown': 60,
                'max_cooldown': 300,
                'error_backoff': 2.0,
            })
            
            self._initialized = True
            logger.info(f"初始化License管理器，共有{len(self._licenses)}个license")
    
    def get_licence(self, api_type='default', user_type='basic', ignore_limits=False):
        """
        获取下一个可用的license
        
        Args:
            api_type: API类型 (realtime, basic, index, market, fund_flow, technical, default)
            user_type: 用户类型 (basic, pro)
            ignore_limits: 是否忽略速率限制，仅在紧急情况下使用
            
        Returns:
            str: license字符串
        """
        with self._lock:
            # 如果没有license，直接返回空字符串
            if not self._licenses:
                logger.error("没有可用的license")
                return ""
            
            # 尝试多次获取可用的license
            attempts = len(self._licenses)
            while attempts > 0:
                licence = self._licenses[self._current_index]
                self._current_index = (self._current_index + 1) % len(self._licenses)
                
                # 检查license是否可用
                if ignore_limits or self._is_licence_available(licence, api_type, user_type):
                    # 记录使用情况
                    current_time = time.time()
                    self._last_use_time[licence] = current_time
                    self._request_history[licence].append(current_time)
                    self._request_counts[licence][api_type] += 1
                    
                    # 清理旧的请求历史记录
                    self._clean_request_history(licence)
                    
                    logger.debug(f"使用license: {licence}，API类型：{api_type}，用户类型：{user_type}")
                    return licence
                
                attempts -= 1
            
            if ignore_limits:
                # 如果忽略限制，强制返回一个license
                licence = self._licenses[0]
                current_time = time.time()
                self._last_use_time[licence] = current_time
                self._request_history[licence].append(current_time)
                self._request_counts[licence][api_type] += 1
                logger.warning(f"强制使用license: {licence}，即使它已达到速率限制")
                return licence
                
            logger.error(f"所有license都达到速率限制或处于冷却状态，API类型：{api_type}，用户类型：{user_type}")
            return ""  # 返回空字符串表示没有可用license    
        
    def _is_licence_available(self, licence, api_type='default', user_type='basic'):
        """
        检查license是否可用
        
        Args:
            licence: 要检查的license
            api_type: API类型
            user_type: 用户类型
            
        Returns:
            bool: 是否可用
        """
        current_time = time.time()
        
        # 检查license是否激活
        if not self._is_active.get(licence, True):
            return False
        
        # 检查license是否处于冷却期
        cooldown_until = self._cooldown_until.get(licence, 0)
        if cooldown_until > current_time:
            logger.debug(f"License {licence} 处于冷却期，还需等待 {cooldown_until - current_time:.1f} 秒")
            return False
        
        # 获取API类型的速率限制
        if api_type in self._rate_limits and user_type in self._rate_limits[api_type]:
            limit_config = self._rate_limits[api_type][user_type]
            rate = limit_config.get('rate', 0.167)  # 默认每6秒一个请求
            burst = limit_config.get('burst', 1)
            
            # 计算时间窗口
            window_size = 1 / rate if rate > 0 else 6
            
            # 获取时间窗口内的请求次数
            window_start = current_time - window_size
            requests_in_window = sum(1 for t in self._request_history[licence] if t > window_start)
            
            # 检查是否超过突发限制
            if requests_in_window >= burst:
                logger.debug(f"License {licence} 达到速率限制: {requests_in_window}/{burst} 请求在 {window_size:.1f} 秒内")
                return False
        
        return True
    
    def _clean_request_history(self, licence):
        """
        清理旧的请求历史记录，只保留最近60秒的
        
        Args:
            licence: 要清理的license
        """
        current_time = time.time()
        cutoff_time = current_time - 60  # 只保留60秒内的记录
        self._request_history[licence] = [t for t in self._request_history[licence] if t > cutoff_time]
    
    def get_least_used_licence(self, api_type='default', user_type='basic'):
        """
        获取最近使用次数最少的license
        
        Args:
            api_type: API类型
            user_type: 用户类型
            
        Returns:
            str: license字符串
        """
        with self._lock:
            if not self._licenses:
                logger.error("没有可用的license")
                return ""
            
            # 优先选择可用且从未使用过的license
            never_used = [lic for lic in self._licenses 
                         if lic not in self._last_use_time 
                         and self._is_licence_available(lic, api_type, user_type)]
            if never_used:
                licence = never_used[0]
                current_time = time.time()
                self._last_use_time[licence] = current_time
                self._request_history[licence].append(current_time)
                self._request_counts[licence][api_type] += 1
                return licence
            
            # 其次选择可用且使用时间最久远的license
            available_licenses = [lic for lic in self._last_use_time 
                                if self._is_licence_available(lic, api_type, user_type)]
            if available_licenses:
                oldest_license = min(available_licenses, key=lambda x: self._last_use_time[x])
                current_time = time.time()
                self._last_use_time[oldest_license] = current_time
                self._request_history[oldest_license].append(current_time)
                self._request_counts[oldest_license][api_type] += 1
                return oldest_license
            
            # 如果没有可用的license，返回使用时间最久远的
            if self._last_use_time:
                oldest_license = min(self._last_use_time.keys(), key=lambda x: self._last_use_time[x])
                return oldest_license
            
            # 如果还是没有，返回第一个license
            return self._licenses[0] if self._licenses else ""
    
    def report_error(self, licence):
        """
        报告license发生错误
        
        Args:
            licence: 发生错误的license
        """
        with self._lock:
            if licence not in self._licenses:
                return
            
            self._error_count[licence] += 1
            error_threshold = self._error_settings.get('error_threshold', 5)
            
            if self._error_count[licence] >= error_threshold:
                # 达到错误阈值，进入冷却期
                cooldown_time = min(
                    self._error_settings.get('base_cooldown', 60) * 
                    math.pow(self._error_settings.get('error_backoff', 2.0), 
                            (self._error_count[licence] - error_threshold) / error_threshold),
                    self._error_settings.get('max_cooldown', 300)
                )
                
                current_time = time.time()
                self._cooldown_until[licence] = current_time + cooldown_time
                logger.warning(f"License {licence} 已进入冷却期 {cooldown_time:.1f} 秒，错误次数: {self._error_count[licence]}")
    
    def reset_error_count(self, licence):
        """
        重置license的错误计数
        
        Args:
            licence: 要重置的license
        """
        with self._lock:
            if licence in self._error_count:
                self._error_count[licence] = 0
    
    def set_licence_status(self, licence, active=True):
        """
        设置license状态
        
        Args:
            licence: 要设置的license
            active: 是否激活
        """
        with self._lock:
            if licence in self._licenses:
                self._is_active[licence] = active
                logger.info(f"License {licence} 状态已设置为 {'激活' if active else '禁用'}")
    
    def get_licence_stats(self):
        """
        获取所有license的统计信息
        
        Returns:
            dict: license统计信息
        """
        with self._lock:
            stats = {}
            current_time = time.time()
            
            for licence in self._licenses:
                stats[licence] = {
                    'active': self._is_active.get(licence, True),
                    'last_used': self._last_use_time.get(licence, 0),
                    'time_since_last_used': current_time - self._last_use_time.get(licence, current_time),
                    'error_count': self._error_count.get(licence, 0),
                    'in_cooldown': current_time < self._cooldown_until.get(licence, 0),
                    'cooldown_remaining': max(0, self._cooldown_until.get(licence, 0) - current_time),
                    'request_counts': dict(self._request_counts[licence]),
                    'total_requests': sum(self._request_counts[licence].values()),
                }
            
            return stats
