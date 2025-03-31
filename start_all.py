"""
启动脚本，同时启动Django服务器、Celery Worker和Beat
"""
import os
import sys
import subprocess
import time
import platform
from pathlib import Path

def start_django_server():
    """
    启动Django服务器
    """
    # 获取项目根目录
    project_root = Path(__file__).resolve().parent
    
    # 构建命令
    cmd = [
        sys.executable,  # Python解释器路径
        str(project_root / 'manage.py'),
        'runserver',
        '0.0.0.0:8000'
    ]
    
    print(f"启动Django服务器: {' '.join(cmd)}")
    
    # 使用非阻塞方式启动
    return subprocess.Popen(
        cmd, 
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True
    )

def start_celery_worker():
    """
    启动Celery Worker
    """
    # 获取项目根目录
    project_root = Path(__file__).resolve().parent
    
    # 检测操作系统类型
    is_windows = platform.system().lower() == 'windows'
    
    # 构建命令
    cmd = [
        sys.executable,  # Python解释器路径
        '-m',
        'celery',
        '-A',
        'chaoyue_dreams.celery_windows' if is_windows else 'chaoyue_dreams',
        'worker',
        '-l',
        'info'
    ]
    
    # Windows下的特殊参数
    if is_windows:
        cmd.extend([
            '--pool=solo',
            '--concurrency=1',
        ])
    
    print(f"启动Celery Worker: {' '.join(cmd)}")
    
    # 使用非阻塞方式启动
    return subprocess.Popen(
        cmd, 
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True
    )

def start_celery_beat():
    """
    启动Celery Beat
    """
    # 获取项目根目录
    project_root = Path(__file__).resolve().parent
    
    # 检测操作系统类型
    is_windows = platform.system().lower() == 'windows'
    
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
    
    print(f"启动Celery Beat: {' '.join(cmd)}")
    
    # 使用非阻塞方式启动
    return subprocess.Popen(
        cmd, 
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True
    )

def print_output(process, prefix):
    """
    非阻塞地打印进程输出
    """
    line = process.stdout.readline()
    if line:
        print(f"[{prefix}] {line.strip()}")
        return True
    return False

def main():
    """
    主函数，启动所有服务
    """
    # 设置Django环境
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'chaoyue_dreams.settings')
    
    # 启动Django服务器
    django_process = start_django_server()
    time.sleep(2)  # 等待Django启动
    
    # 启动Celery Worker
    worker_process = start_celery_worker()
    time.sleep(2)  # 等待Worker启动
    
    # 启动Celery Beat
    beat_process = start_celery_beat()
    
    processes = [
        (django_process, "Django"),
        (worker_process, "Worker"),
        (beat_process, "Beat")
    ]
    
    try:
        # 持续打印输出
        while all(p[0].poll() is None for p in processes):
            any_output = False
            for process, prefix in processes:
                if print_output(process, prefix):
                    any_output = True
            
            if not any_output:
                time.sleep(0.1)
    
    except KeyboardInterrupt:
        print("\n正在关闭所有服务...")
    finally:
        # 关闭所有进程
        for process, name in processes:
            if process.poll() is None:
                print(f"正在关闭{name}服务...")
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    print(f"强制关闭{name}服务...")
                    process.kill()

if __name__ == "__main__":
    main() 