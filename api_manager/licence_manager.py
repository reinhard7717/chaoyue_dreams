# api_manager/licence_manager.py

import time
import logging
import threading
from django.conf import settings

logger = logging.getLogger(__name__)

class LicenceManager:
    """
    License管理器，用于轮番使用多个licence
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
            self._licenses = settings.API_LICENCES
            self._current_index = 0
            self._last_use_time = {}  # 记录每个license最后使用时间
            self._initialized = True
            logger.info(f"初始化License管理器，共有{len(self._licenses)}个license")
    
    def get_licence(self):
        """
        获取下一个可用的license
        
        Returns:
            str: license字符串
        """
        with self._lock:
            # 如果没有license，直接返回空字符串
            if not self._licenses:
                logger.error("没有可用的license")
                return ""
            
            # 选择当前索引的license
            licence = self._licenses[self._current_index]
            
            # 更新最后使用时间
            self._last_use_time[licence] = time.time()
            
            # 更新索引，轮换使用license
            self._current_index = (self._current_index + 1) % len(self._licenses)
            
            logger.debug(f"使用license: {licence}")
            return licence
    
    def get_least_used_licence(self):
        """
        获取最近使用次数最少的license
        
        Returns:
            str: license字符串
        """
        with self._lock:
            if not self._licenses:
                logger.error("没有可用的license")
                return ""
            
            # 如果某些license从未使用过，优先使用
            never_used = [lic for lic in self._licenses if lic not in self._last_use_time]
            if never_used:
                licence = never_used[0]
                self._last_use_time[licence] = time.time()
                return licence
            
            # 否则选择最久未使用的license
            oldest_license = min(self._last_use_time.keys(), key=lambda x: self._last_use_time[x])
            self._last_use_time[oldest_license] = time.time()
            return oldest_license
