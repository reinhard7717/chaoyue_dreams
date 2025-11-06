"""
启动Celery Beat的脚本
自动检测操作系统类型，在Windows下使用优化配置
"""
import os
import sys
import subprocess
import platform
import redis
import re
import django
from pathlib import Path

def check_database_migration():
    """
    检查数据库迁移
    确保django_celery_beat的数据库表已经创建
    """
    try:
        # 设置Django环境
        os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'chaoyue_dreams.settings')
        django.setup()
        # 检查django_celery_beat的表是否存在
        from django.db import connections
        from django.db.utils import OperationalError
        conn = connections['default']
        try:
            cursor = conn.cursor()
            cursor.execute("SHOW TABLES LIKE 'django_celery_beat_periodictask'")
            result = cursor.fetchone()
            return result is not None
        except OperationalError:
            return False
    except Exception as e:
        print(f"检查数据库迁移异常: {str(e)}")
        return False

def run_database_migration():
    """
    执行数据库迁移
    """
    try:
        # 获取项目根目录
        project_root = Path(__file__).resolve().parent
        # 构建命令
        cmd = [
            sys.executable,  # Python解释器路径
            str(project_root / 'manage.py'),
            'migrate',
            'django_celery_beat',
        ]
        print(f"执行数据库迁移命令: {' '.join(cmd)}")
        # 执行命令
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"数据库迁移失败: {str(e)}")
        print(e.stdout)
        print(e.stderr)
        return False

def check_redis_connection():
    """
    测试Redis连接
    检查Celery配置中的Redis连接是否可用
    """
    try:
        # 设置Django环境
        os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'chaoyue_dreams.settings')
        django.setup()
        from django.conf import settings
        # 解析Redis连接信息
        broker_url = settings.CELERY_BROKER_URL
        pattern = r'redis://(?::(.*))?@(.*):(\d+)/(\d+)'
        match = re.match(pattern, broker_url)
        if match:
            password, host, port, db = match.groups()
            port = int(port)
            db = int(db)
        else:
            # 使用默认设置
            password = getattr(settings, 'REDIS_PASSWORD', None)
            host = getattr(settings, 'REDIS_HOST', 'localhost')
            port = getattr(settings, 'REDIS_PORT', 6379)
            db = 1
        # 测试Redis连接
        client = redis.Redis(
            host=host,
            port=port,
            password=password,
            db=db,
            socket_timeout=5,
            socket_connect_timeout=5
        )
        response = client.ping()
        print(f"Redis连接测试: {'成功' if response else '失败'}")
        return response
    except Exception as e:
        print(f"Redis连接测试异常: {str(e)}")
        return False

def start_celery_beat():
    """
    启动Celery Beat
    根据操作系统类型选择合适的配置
    """
    # 检查数据库迁移
    if not check_database_migration():
        print("正在执行数据库迁移...")
        if not run_database_migration():
            print("数据库迁移失败，无法启动Celery Beat")
            sys.exit(1)
    # 检查Redis连接
    if not check_redis_connection():
        print("Redis连接失败，无法启动Celery Beat")
        sys.exit(1)
    # 检测操作系统类型
    is_windows = platform.system().lower() == 'windows'
    # 设置Django环境
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'chaoyue_dreams.settings')
    # 获取项目根目录
    project_root = Path(__file__).resolve().parent
    # 构建命令
    cmd = [
        sys.executable,  # Python解释器路径
        '-m',
        'celery',
        '-A',
        'chaoyue_dreams.celery_windows' if is_windows else 'chaoyue_dreams',
        'beat',
        '-l',
        'info',
        '--scheduler',
        'django_celery_beat.schedulers:DatabaseScheduler',
    ]
    print(f"启动命令: {' '.join(cmd)}")
    try:
        # 执行命令
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("手动停止Celery Beat...")
    except subprocess.CalledProcessError as e:
        print(f"启动Celery Beat失败: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    start_celery_beat() 