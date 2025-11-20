import re # 导入正则表达式模块
import time # 导入时间模块
from django.core.cache import cache # 导入Django缓存，我们将使用Redis
from django.http import HttpResponseForbidden # 导入Forbidden响应

# 定义一个正则表达式列表，匹配常见的扫描路径
# 这些路径是从您的日志中提取的，并增加了一些常见的扫描目标
BLOCKED_PATH_PATTERNS = [
    re.compile(r"\.php$", re.IGNORECASE), # 匹配所有以 .php 结尾的请求
    re.compile(r"\.env$", re.IGNORECASE), # 匹配所有以 .env 结尾的请求
    re.compile(r"^/owa/"), # 匹配OWA（Outlook Web Access）路径
    re.compile(r"^/ips/"), # 匹配日志中出现的 /ips/ 路径
    re.compile(r"^/pma/"), # 匹配 phpMyAdmin 的常见路径
    re.compile(r"^/phpmyadmin/"), # 匹配 phpMyAdmin 的常见路径
    re.compile(r"^/wordpress/"), # 匹配 WordPress 的常见路径
    re.compile(r"^/wp-admin/"), # 匹配 WordPress 管理后台
    re.compile(r"\.git/config$"), # 匹配暴露的git配置
    re.compile(r"\.svn/entries$"), # 匹配暴露的svn信息
]

# 定义速率限制的配置
# 60秒内最多允许120次请求
RATE_LIMIT_SECONDS = 60
RATE_LIMIT_REQUESTS = 1200

class SecurityMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response
    def __call__(self, request):
        # 1. 路径扫描防御
        # ----------------
        path = request.path_info
        for pattern in BLOCKED_PATH_PATTERNS:
            if pattern.search(path):
                print(f"DEBUG: [SecurityMiddleware] Blocked scan attempt for path: {path} from IP: {self.get_client_ip(request)}")
                return HttpResponseForbidden("Access Denied.") # 如果匹配到恶意路径，直接返回403 Forbidden
        # 2. IP速率限制防御 (使用Redis)
        # --------------------------
        ip = self.get_client_ip(request)
        if ip:
            # 使用Redis缓存来存储IP的请求记录
            cache_key = f"rate-limit:{ip}"
            request_count = cache.get(cache_key, 0) + 1
            # cache.set(key, value, timeout)
            # 如果是第一次请求，我们会设置一个过期时间
            if request_count == 1:
                cache.set(cache_key, request_count, RATE_LIMIT_SECONDS)
            else:
                cache.set(cache_key, request_count) # 对于已存在的key，仅更新值，不改变过期时间
            if request_count > RATE_LIMIT_REQUESTS:
                print(f"DEBUG: [SecurityMiddleware] Rate limit exceeded for IP: {ip}")
                # 当超过速率限制时，可以考虑将IP加入黑名单
                # 例如：cache.set(f"blacklist:{ip}", True, timeout=3600) # 封禁1小时
                return HttpResponseForbidden("Rate limit exceeded. Access Denied.") # 返回403 Forbidden
        response = self.get_response(request)
        return response
    def get_client_ip(self, request):
        """获取客户端IP地址"""
        # 尝试从 X-Forwarded-For 头获取IP，这在Nginx等反向代理后是必须的
        x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
        if x_forwarded_for:
            ip = x_forwarded_for.split(',')[0]
        else:
            # 否则，从 REMOTE_ADDR 获取IP
            ip = request.META.get('REMOTE_ADDR')
        return ip

