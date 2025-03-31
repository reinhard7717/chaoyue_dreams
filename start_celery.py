import subprocess
import sys
import os
from pathlib import Path

def start_celery_worker(concurrency=4):
    """
    启动Celery worker
    :param concurrency: worker进程数
    """
    # 获取项目根目录
    project_root = Path(__file__).resolve().parent
    
    # 构建命令
    cmd = [
        'celery',
        '-A',
        'chaoyue_dreams',
        'worker',
        '-l',
        'info',
        '--concurrency',
        str(concurrency),
        '--max-tasks-per-child',
        '1000',  # 每个worker处理的最大任务数
        '--max-memory-per-child',
        '200000',  # 每个worker的最大内存使用量（KB）
        '--prefetch-multiplier',
        '1',  # 预取任务数
        '--time-limit',
        '3600',  # 任务超时时间（秒）
        '--soft-time-limit',
        '3000',  # 软超时时间（秒）
        '--without-heartbeat',  # 禁用心跳
        '--without-mingle',  # 禁用mingle
        '--without-gossip',  # 禁用gossip
    ]
    
    try:
        # 启动worker
        process = subprocess.Popen(
            cmd,
            cwd=str(project_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        # 实时输出日志
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.strip())
        
        # 获取错误输出
        _, stderr = process.communicate()
        if stderr:
            print(f"Error: {stderr}", file=sys.stderr)
        
        # 检查是否有错误
        if process.returncode != 0:
            print(f"Celery worker启动失败，错误代码: {process.returncode}")
            sys.exit(1)
        
        return process.returncode
        
    except KeyboardInterrupt:
        print("\nStopping Celery worker...")
        process.terminate()
        return 0
    except Exception as e:
        print(f"Error starting Celery worker: {e}", file=sys.stderr)
        return 1

def start_celery_beat():
    """
    启动Celery beat
    """
    # 获取项目根目录
    project_root = Path(__file__).resolve().parent
    
    # 构建命令
    cmd = [
        'celery',
        '-A',
        'chaoyue_dreams',
        'beat',
        '-l',
        'info',
        '--max-interval',
        '300',  # beat最大循环间隔（秒）
    ]
    
    try:
        # 启动beat
        process = subprocess.Popen(
            cmd,
            cwd=str(project_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        # 实时输出日志
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.strip())
        
        # 获取错误输出
        _, stderr = process.communicate()
        if stderr:
            print(f"Error: {stderr}", file=sys.stderr)
        
        # 检查是否有错误
        if process.returncode != 0:
            print(f"Celery beat启动失败，错误代码: {process.returncode}")
            sys.exit(1)
        
        return process.returncode
        
    except KeyboardInterrupt:
        print("\nStopping Celery beat...")
        process.terminate()
        return 0
    except Exception as e:
        print(f"Error starting Celery beat: {e}", file=sys.stderr)
        return 1

if __name__ == '__main__':
    # 从命令行参数获取worker数量
    concurrency = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    
    # 启动worker
    print(f"Starting Celery worker with {concurrency} processes...")
    worker_code = start_celery_worker(concurrency)
    
    if worker_code != 0:
        print(f"Celery worker exited with code {worker_code}")
        sys.exit(worker_code) 