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
            # 从settings中读取API_LICENCES_IG507配置
            self._licenses = getattr(settings, 'API_LICENCES_IG507', [])
            if not self._licenses:
                logger.warning("未在settings中找到API_LICENCES_IG507配置，或配置为空列表")
            else:
                logger.info(f"从settings中加载了{len(self._licenses)}个license")
            self._current_index = 0
            self._last_use_time = {}  # 记录每个license最后使用时间
            self._request_history = defaultdict(list)  # 记录每个license请求历史
            self._error_count = defaultdict(int)  # 记录每个license错误次数
            self._is_active = {lic: True for lic in self._licenses}  # license状态
            self._cooldown_until = {}  # license冷却时间
            self._request_counts = defaultdict(lambda: defaultdict(int))  # 记录每个license不同API类型的请求次数
            self._consecutive_errors = defaultdict(int)  # 记录每个license连续错误次数
            self._success_count = defaultdict(int)  # 记录每个license成功请求次数
            self._error_history = defaultdict(list)  # 记录每个license的错误历史
            self._api_type_stats = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))  # 记录每个license的API类型统计
            # 从settings中获取速率限制和错误处理配置
            self._rate_limits = getattr(settings, 'API_RATE_LIMITS', {})
            self._error_settings = getattr(settings, 'API_ERROR_SETTINGS', {})
            # 检查必要的配置是否存在
            if not self._rate_limits:
                logger.error("未在settings中找到API_RATE_LIMITS配置")
            if not self._error_settings:
                logger.error("未在settings中找到API_ERROR_SETTINGS配置")
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
        # 获取API类型的速率限制，如果找不到特定类型，使用默认配置
        limit_config = None
        if api_type in self._rate_limits and user_type in self._rate_limits[api_type]:
            limit_config = self._rate_limits[api_type][user_type]
        else:
            # 使用默认配置
            default_config = self._rate_limits.get('default', {})
            if user_type in default_config:
                limit_config = default_config[user_type]
                logger.debug(f"使用默认配置: default.{user_type}")
            else:
                logger.error(f"未找到API_RATE_LIMITS配置: {api_type}.{user_type} 或 default.{user_type}")
                return False
        if limit_config:
            rate = limit_config.get('rate')
            burst = limit_config.get('burst')
            error_window = limit_config.get('error_window')
            min_success_rate = limit_config.get('min_success_rate')
            if any(x is None for x in [rate, burst, error_window, min_success_rate]):
                logger.error(f"API_RATE_LIMITS配置不完整: {api_type}.{user_type}")
                return False
        else:
            logger.error(f"未找到有效的API_RATE_LIMITS配置: {api_type}.{user_type}")
            return False
        # 计算时间窗口
        window_size = 1 / rate if rate > 0 else 10
        # 获取时间窗口内的请求次数
        window_start = current_time - window_size
        requests_in_window = sum(1 for t in self._request_history[licence] if t > window_start)
        # 检查是否超过突发限制
        if requests_in_window >= burst:
            logger.debug(f"License {licence} 达到速率限制: {requests_in_window}/{burst} 请求在 {window_size:.1f} 秒内")
            return False
        # 检查错误率
        error_history = [t for t in self._error_history[licence] if t > current_time - error_window]
        total_requests = len([t for t in self._request_history[licence] if t > current_time - error_window])
        if total_requests > 0:
            error_rate = len(error_history) / total_requests
            if error_rate > (1 - min_success_rate):
                logger.warning(f"License {licence} 错误率过高: {error_rate:.2%} > {(1 - min_success_rate):.2%}")
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
    def report_error(self, licence, error_type=None):
        """
        报告license发生错误
        Args:
            licence: 发生错误的license
            error_type: 错误类型（可选）
        """
        with self._lock:
            if licence not in self._licenses:
                return
            current_time = time.time()
            # 更新错误计数
            self._error_count[licence] += 1
            self._consecutive_errors[licence] += 1
            self._error_history[licence].append(current_time)
            # 限制错误历史记录大小
            default_config = self._rate_limits.get('default', {}).get('basic', {})
            max_history_size = default_config.get('error_history_size')
            if max_history_size is None:
                logger.error("未找到API_RATE_LIMITS.default.basic.error_history_size配置")
                return
                
            if len(self._error_history[licence]) > max_history_size:
                self._error_history[licence] = self._error_history[licence][-max_history_size:]
            # 更新API类型统计
            if error_type:
                self._api_type_stats[licence][error_type]['errors'] += 1
            error_threshold = self._error_settings.get('error_threshold')
            consecutive_error_threshold = self._error_settings.get('consecutive_error_threshold')
            base_cooldown = self._error_settings.get('base_cooldown')
            max_cooldown = self._error_settings.get('max_cooldown')
            error_backoff = self._error_settings.get('error_backoff')
            # 检查每个配置项是否存在
            missing_settings = []
            if error_threshold is None:
                missing_settings.append('error_threshold')
            if consecutive_error_threshold is None:
                missing_settings.append('consecutive_error_threshold')
            if base_cooldown is None:
                missing_settings.append('base_cooldown')
            if max_cooldown is None:
                missing_settings.append('max_cooldown')
            if error_backoff is None:
                missing_settings.append('error_backoff')
            if missing_settings:
                logger.error(f"API_ERROR_SETTINGS配置不完整，缺少以下配置项：{', '.join(missing_settings)}")
                return
            # 检查是否达到连续错误阈值
            if self._consecutive_errors[licence] >= consecutive_error_threshold:
                # 达到连续错误阈值，进入冷却期
                cooldown_time = min(
                    base_cooldown * 
                    math.pow(error_backoff,
                            (self._consecutive_errors[licence] - consecutive_error_threshold) / consecutive_error_threshold),
                    max_cooldown
                )
                self._cooldown_until[licence] = current_time + cooldown_time
                logger.warning(f"License {licence} 已进入冷却期 {cooldown_time:.1f} 秒，连续错误次数: {self._consecutive_errors[licence]}")
            # 检查是否达到总错误阈值
            if self._error_count[licence] >= error_threshold:
                # 达到错误阈值，进入冷却期
                cooldown_time = min(
                    base_cooldown * 
                    math.pow(error_backoff,
                            (self._error_count[licence] - error_threshold) / error_threshold),
                    max_cooldown
                )
                self._cooldown_until[licence] = current_time + cooldown_time
                logger.warning(f"License {licence} 已进入冷却期 {cooldown_time:.1f} 秒，错误次数: {self._error_count[licence]}")
    def reset_error_count(self, licence, api_type=None):
        """
        重置license的错误计数
        Args:
            licence: 要重置的license
            api_type: API类型（可选）
        """
        with self._lock:
            if licence in self._error_count:
                self._success_count[licence] += 1
                success_threshold = self._error_settings.get('success_threshold')
                if success_threshold is None:
                    logger.error("未找到API_ERROR_SETTINGS.success_threshold配置")
                    return
                # 更新API类型统计
                if api_type:
                    self._api_type_stats[licence][api_type]['success'] += 1
                # 如果成功请求次数达到阈值，重置错误计数
                if self._success_count[licence] >= success_threshold:
                    self._error_count[licence] = 0
                    self._consecutive_errors[licence] = 0
                    self._success_count[licence] = 0
                    logger.info(f"License {licence} 成功请求次数达到阈值，重置错误计数")
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
                # 获取默认的错误窗口配置
                default_config = self._rate_limits.get('default', {}).get('basic', {})
                error_window = default_config.get('error_window')
                if error_window is None:
                    logger.error("未找到API_RATE_LIMITS.default.basic.error_window配置")
                    continue
                # 计算错误率
                error_history = [t for t in self._error_history[licence] if t > current_time - error_window]
                total_requests = len([t for t in self._request_history[licence] if t > current_time - error_window])
                error_rate = len(error_history) / total_requests if total_requests > 0 else 0
                # 计算API类型统计
                api_stats = {}
                for api_type, type_stats in self._api_type_stats[licence].items():
                    total = type_stats.get('success', 0) + type_stats.get('errors', 0)
                    if total > 0:
                        api_stats[api_type] = {
                            'success_rate': type_stats.get('success', 0) / total,
                            'total_requests': total,
                            'success_count': type_stats.get('success', 0),
                            'error_count': type_stats.get('errors', 0)
                        }
                stats[licence] = {
                    'active': self._is_active.get(licence, True),
                    'last_used': self._last_use_time.get(licence, 0),
                    'time_since_last_used': current_time - self._last_use_time.get(licence, current_time),
                    'error_count': self._error_count.get(licence, 0),
                    'consecutive_errors': self._consecutive_errors.get(licence, 0),
                    'success_count': self._success_count.get(licence, 0),
                    'in_cooldown': current_time < self._cooldown_until.get(licence, 0),
                    'cooldown_remaining': max(0, self._cooldown_until.get(licence, 0) - current_time),
                    'request_counts': dict(self._request_counts[licence]),
                    'total_requests': sum(self._request_counts[licence].values()),
                    'error_rate': error_rate,
                    'api_stats': api_stats
                }
            return stats
