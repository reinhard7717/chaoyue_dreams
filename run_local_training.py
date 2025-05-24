#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import django
import logging
from pathlib import Path
import argparse
import time
import multiprocessing
from tqdm.contrib.concurrent import process_map
import sys
import torch # 导入 torch 用于检查 CUDA

# --- 配置 Django 设置 ---
# 请将 'your_project_name.settings' 替换为您的实际 Django 项目的 settings 模块路径。
# 例如: 如果您的项目根目录下有一个名为 'myproject' 的应用包含了 settings.py,
# 并且 'E:\chaoyue_dreams' 是项目根目录 (PYTHONPATH 包含此路径或其父目录),
# 那么这里可能是 'myproject.settings' 或类似的路径。
# 如果 settings.py 在项目根目录下的一个名为 'config' 的文件夹中，那么可能是 'config.settings'
DJANGO_SETTINGS_MODULE_NAME = 'chaoyue_dreams.settings' # <--- 【重要】请用户根据实际项目结构修改此行

# 在主进程中尝试加载 Django settings，用于获取固定的路径配置。子进程需要独立加载。
DJANGO_SETTINGS_AVAILABLE_IN_MAIN = False
django_settings_module_main = None

try:
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', DJANGO_SETTINGS_MODULE_NAME)
    # 检查是否已经 setup，避免重复调用
    if not django.apps.apps.ready:
        django.setup()
    from django.conf import settings as imported_django_settings
    django_settings_module_main = imported_django_settings
    DJANGO_SETTINGS_AVAILABLE_IN_MAIN = True
    print(f"DEBUG: Django settings '{DJANGO_SETTINGS_MODULE_NAME}' 已成功加载和配置 (主进程)。")
except Exception as e:
    print(f"CRITICAL: 无法加载或配置 Django settings '{DJANGO_SETTINGS_MODULE_NAME}' (主进程): {e}") # 修改行：将 WARNING 改为 CRITICAL，因为路径现在必须从 settings 获取
    print("CRITICAL: 脚本无法获取必要的路径配置 (STRATEGY_DATA_DIR, INDICATOR_PARAMETERS_CONFIG_PATH)。请确保 Django settings 配置正确。") # 修改行：更新错误信息
    sys.exit("Django settings 加载失败，无法获取路径配置。") # 修改行：如果 settings 加载失败，直接退出

# --- 模块导入 ---
# 注意：这些导入在主进程中执行。工作进程需要确保它们能访问到这些模块。
try:
    # 假设这些模块不直接依赖于 Django ORM 的连接，只依赖 settings 中的路径等配置
    from services.indicator_services import IndicatorService
    from dao_manager.tushare_daos.stock_basic_info_dao import StockBasicInfoDao
    # 导入 train_transformer_model 函数所在的模块
    # 假设 train_transformer_model 在 strategies.transformer_trainer 模块中
    # 请根据您的实际文件结构修改此行
    from strategies.transformer_trainer import train_transformer_model # 导入 train_transformer_model 函数
    from strategies.trend_following_strategy import TrendFollowingStrategy # 导入 TrendFollowingStrategy 类

    print("DEBUG: IndicatorService, StockBasicInfoDao, train_transformer_model, TrendFollowingStrategy 类/函数已成功导入。")
except ImportError as e:
    print(f"CRITICAL: 无法导入核心业务模块 (如 IndicatorService, TrendFollowingStrategy, train_transformer_model 等): {e}")
    print("CRITICAL: 请确保这些模块存在于 PYTHONPATH 中，并且 Django 环境已正确配置（如果它们依赖 Django）。")
    sys.exit("核心业务模块导入失败，脚本无法继续执行。")

# --- 配置主进程日志记录器 ---
# 在 __main__ 块中会重新配置，这里只是定义一个 logger 变量
logger = logging.getLogger(__name__)

# --- 工作进程的训练函数 ---
def _worker_train_single_stock(
    stock_code: str,
    model_base_dir_str: str, # 修改行：仍然需要接收路径字符串
    params_file_str: str,    # 修改行：仍然需要接收路径字符串
    django_settings_module_name: str # 传递 settings 模块名称
    ):
    """
    工作进程函数：负责为单个股票执行 Transformer 模型训练。
    接收路径字符串是因为子进程不直接继承父进程的 Path 对象或 settings 模块状态。
    """
    # 在工作进程中重新配置日志
    worker_logger = logging.getLogger(f'worker_{os.getpid()}')
    if not worker_logger.handlers:
        if worker_logger.parent and worker_logger.parent.handlers:
             worker_logger.parent.handlers.clear()
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(f'%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        worker_logger.addHandler(handler)
        worker_logger.setLevel(logging.INFO)
        worker_logger.propagate = False

    worker_logger.info(f"工作进程 {os.getpid()} 日志配置完成。")

    # 在工作进程中尝试加载 Django settings
    try:
        os.environ.setdefault('DJANGO_SETTINGS_MODULE', django_settings_module_name)
        if not django.apps.apps.ready:
             django.setup()
        worker_logger.info(f"工作进程 {os.getpid()}: Django settings '{django_settings_module_name}' 已加载。")
    except Exception as e:
        worker_logger.error(f"工作进程 {os.getpid()}: 无法加载或配置 Django settings '{django_settings_module_name}': {e}", exc_info=True)
        return {"stock_code": stock_code, "status": "failed", "message": f"工作进程 Django 配置失败: {e}"}

    worker_logger.info(f"工作进程 {os.getpid()} 开始处理股票: {stock_code}")

    try:
        # 在工作进程中构建 Path 对象
        model_base_dir = Path(model_base_dir_str)
        params_file = Path(params_file_str)
        item_path = model_base_dir / stock_code
        prepared_data_path = item_path / "prepared_data"
        trained_model_path = item_path / "trained_model"

        # 再次检查 npz 文件是否存在 (虽然主进程已过滤，但作为安全检查)
        if not prepared_data_path.is_dir() or not list(prepared_data_path.glob("*.npz")):
             worker_logger.warning(f"[{stock_code}] 工作进程发现预处理数据缺失，跳过训练。")
             return {"stock_code": stock_code, "status": "skipped_no_npz", "message": "预处理数据缺失"}

        # 在工作进程中实例化策略类
        strategy = TrendFollowingStrategy(params_file=params_file, base_data_dir=model_base_dir)
        worker_logger.info(f"[{stock_code}] 工作进程实例化 TrendFollowingStrategy。")

        # 调用策略的训练方法
        training_successful = strategy.train_transformer_model_from_prepared_data(stock_code)

        if training_successful and \
           strategy.transformer_model and \
           strategy.feature_scaler and \
           strategy.target_scaler and \
           strategy.selected_feature_names_for_transformer:
            worker_logger.info(f"[{stock_code}] Transformer 模型训练成功完成。")
            return {"stock_code": stock_code, "status": "success", "message": "训练成功"}
        else:
            worker_logger.error(f"[{stock_code}] Transformer 模型训练逻辑执行完毕，但未能成功标记为训练或必要组件缺失。")
            return {"stock_code": stock_code, "status": "failed", "message": "训练方法返回失败或组件缺失"}

    except Exception as e:
        worker_logger.error(f"工作进程 {os.getpid()} 为股票 {stock_code} 执行训练时发生意外错误: {e}", exc_info=True)
        return {"stock_code": stock_code, "status": "failed", "message": f"训练过程中发生异常: {e}"}

# --- 主进程的批量训练函数 ---
# 修改行：移除 model_base_dir_path_str 和 params_file_path_str 参数
def run_local_transformer_training_batch(
    processing_order: str = 'asc',
    num_processes: int = 1
    ):
    print("DEBUG: !!!!! 函数入口：run_local_transformer_training_batch !!!!!")
    logger.info("开始执行本地 Transformer 模型批量训练任务...")
    logger.info(f"配置：进程数 = {num_processes}, 处理顺序 = {processing_order}")

    # --- 从 Django settings 获取和验证路径 ---
    # 修改开始：直接从 settings 获取路径，不再接受参数
    if not DJANGO_SETTINGS_AVAILABLE_IN_MAIN:
         logger.critical("错误：Django settings 未成功加载，无法获取路径配置。")
         return {"status": "error", "message": "Django settings 未加载。"}

    actual_model_base_dir_str = None
    actual_params_file_str = None

    if hasattr(django_settings_module_main, 'STRATEGY_DATA_DIR'):
        actual_model_base_dir_str = str(django_settings_module_main.STRATEGY_DATA_DIR) # 修改行：获取并转换为字符串
        actual_model_base_dir = Path(actual_model_base_dir_str) # 修改行：转换为 Path 对象进行验证
        logger.info(f"从 Django settings 获取模型根目录: '{actual_model_base_dir_str}'")
    else:
        logger.critical("错误：Django settings 中未找到 STRATEGY_DATA_DIR 配置。")
        return {"status": "error", "message": "Django settings 中缺少 STRATEGY_DATA_DIR。"}

    if hasattr(django_settings_module_main, 'INDICATOR_PARAMETERS_CONFIG_PATH'):
        actual_params_file_str = str(django_settings_module_dir_main.INDICATOR_PARAMETERS_CONFIG_PATH) # 修改行：获取并转换为字符串
        actual_params_file = Path(actual_params_file_str) # 修改行：转换为 Path 对象进行验证
        logger.info(f"从 Django settings 获取指标参数文件路径: '{actual_params_file_str}'")
    else:
        logger.critical("错误：Django settings 中未找到 INDICATOR_PARAMETERS_CONFIG_PATH 配置。")
        return {"status": "error", "message": "Django settings 中缺少 INDICATOR_PARAMETERS_CONFIG_PATH。"}

    # 修改结束：直接从 settings 获取路径

    # --- 验证路径 ---
    if not actual_params_file.is_file():
        logger.critical(f"错误：从 Django settings 获取的指标参数文件 '{actual_params_file}' 不存在或不是一个文件。") # 修改行：更新错误信息
        return {"status": "error", "message": f"指标参数文件 '{actual_params_file}' 未找到或无效。"}

    if not actual_model_base_dir.is_dir():
        logger.critical(f"错误：从 Django settings 获取的模型根目录 '{actual_model_base_dir}' 不存在或不是一个目录。") # 修改行：更新错误信息
        return {"status": "error", "message": f"模型根目录 '{actual_model_base_dir}' 未找到或无效。"}

    # --- 收集需要训练的股票列表 ---
    all_item_names = [item.name for item in actual_model_base_dir.iterdir() if item.is_dir()]
    all_item_names.sort() # 默认按字典序正序排序

    if processing_order == 'desc':
        all_item_names.reverse()
        logger.info(f"将按倒序处理股票文件夹。")
    else:
        logger.info(f"将按正序处理股票文件夹。")

    total_stock_folders = len(all_item_names)
    logger.info(f"在 '{actual_model_base_dir}' 中找到 {total_stock_folders} 个潜在的股票文件夹。")

    stocks_to_train = []
    skipped_due_to_no_npz = 0
    skipped_due_to_existing_pth = 0
    processed_stock_folders_filter = 0 # 用于过滤阶段的计数

    logger.info("开始过滤需要训练的股票...")
    for item_name in all_item_names:
        processed_stock_folders_filter += 1
        item_path = actual_model_base_dir / item_name
        prepared_data_path = item_path / "prepared_data"
        trained_model_path = item_path / "trained_model"

        if not prepared_data_path.is_dir():
            skipped_due_to_no_npz += 1
            continue

        npz_files = list(prepared_data_path.glob("*.npz"))
        if not npz_files:
            skipped_due_to_no_npz += 1
            continue

        pth_files = []
        if trained_model_path.is_dir():
            pth_files = list(trained_model_path.glob("*.pth"))

        if pth_files:
            skipped_due_to_existing_pth += 1
            continue

        # 如果通过所有检查，则添加到待训练列表
        stocks_to_train.append(item_name)

    total_stocks_to_train = len(stocks_to_train)
    logger.info(f"过滤完成。总计 {total_stock_folders} 个文件夹，跳过 {skipped_due_to_no_npz} (无数据) + {skipped_due_to_existing_pth} (已训练) = {skipped_due_to_no_npz + skipped_due_to_existing_pth} 个。")
    logger.info(f"将训练 {total_stocks_to_train} 个股票模型。")

    if total_stocks_to_train == 0:
        summary_message = "没有需要训练的股票模型。"
        logger.info(summary_message)
        return {
            "status": "completed",
            "message": summary_message,
            "successfully_trained": 0,
            "skipped_no_npz": skipped_due_to_no_npz,
            "skipped_existing_pth": skipped_due_to_existing_pth,
            "failed_training": 0,
            "total_processed_folders": processed_stock_folders_filter,
            "total_discovered_folders": total_stock_folders
        }

    # --- 执行并行训练 ---
    logger.info(f"使用 {num_processes} 个进程开始并行训练...")
    start_time = time.time()

    # 存储并行训练的结果
    results = []
    failed_training_count = 0
    successfully_trained_count = 0

    # 使用 process_map 进行并行处理，并显示进度条
    # 将从 settings 获取的路径字符串传递给 worker 函数
    try:
        results = process_map(
            _worker_train_single_stock,
            stocks_to_train,
            actual_model_base_dir_str, # 修改行：传递从 settings 获取的字符串路径
            actual_params_file_str,    # 修改行：传递从 settings 获取的字符串路径
            DJANGO_SETTINGS_MODULE_NAME,
            max_workers=num_processes,
            chunksize=1
        )
    except Exception as e:
        logger.critical(f"并行处理过程中发生致命错误: {e}", exc_info=True)
        failed_training_count = total_stocks_to_train
        results = [{"stock_code": stock, "status": "failed", "message": f"并行处理框架错误: {e}"} for stock in stocks_to_train]

    # --- 处理并行训练结果 ---
    for result in results:
        if result and isinstance(result, dict):
            if result.get("status") == "success":
                successfully_trained_count += 1
            elif result.get("status") == "failed":
                failed_training_count += 1
                logger.error(f"股票 {result.get('stock_code', '未知')} 训练失败: {result.get('message', '未知错误')}")
        else:
             logger.error(f"收到无效的工作进程结果: {result}")
             failed_training_count += 1

    end_time = time.time()
    duration = end_time - start_time

    summary_message = (
        f"\n--- 本地 Transformer 模型批量训练完成 ---\n"
        f"总计发现的股票文件夹数量: {total_stock_folders}.\n"
        f"过滤后待训练数量: {total_stocks_to_train}.\n"
        f"因缺少NPZ文件或目录而跳过的数量: {skipped_due_to_no_npz}.\n"
        f"因已存在PTH模型而跳过的数量: {skipped_due_to_existing_pth}.\n"
        f"成功训练的模型数量: {successfully_trained_count}.\n"
        f"训练失败的数量: {failed_training_count}.\n"
        f"总耗时: {duration:.2f} 秒.\n"
        f"------------------------------------------"
    )
    logger.info(summary_message)

    # --- MPS 相关提示 ---
    if num_processes > 1 and torch.cuda.is_available() and torch.cuda.device_count() == 1:
         logger.info("\n--- 单 GPU 使用多进程 (结合 MPS) 注意事项 ---")
         logger.info("检测到您使用多进程 (processes > 1) 且只有一块 CUDA 设备。")
         logger.info("在单 GPU 上，多个训练进程会竞争 GPU 资源。")
         logger.info("为了提高 GPU 利用率和并发性，建议您在系统层面配置并启动 NVIDIA MPS (Multi-Process Service)。")
         logger.info("如果 MPS 守护进程正在运行，本脚本启动的多个进程将自动尝试作为 MPS 客户端共享 GPU 资源。")
         logger.info("配置 MPS 的步骤通常包括：")
         logger.info("1. 确保您的 NVIDIA 驱动支持 MPS (CUDA 12.4 支持)。")
         logger.info("2. 启动 MPS 控制守护进程 (通常需要 root 权限)。例如：")
         logger.info("   sudo nvidia-cuda-mps-control -d")
         logger.info("3. 运行本脚本，并设置 --processes 参数为您希望并发的进程数。")
         logger.info("请注意：")
         logger.info(" - MPS 不能增加 GPU 显存容量。您能同时运行的进程数受限于所有进程的峰值显存总和是否超过 16GB。")
         logger.info(" - 实际的加速效果取决于您的模型大小、批量大小、数据加载效率以及 MPS 的调度效率。")
         logger.info(" - 建议监控 GPU 状态 (如使用 `nvidia-smi` 或 `nvidia-smi pmon`) 来观察 MPS 是否生效以及资源利用情况。")
         logger.info(" - 结束时，请记得停止 MPS 守护进程 (如果不再需要)。例如：")
         logger.info("   echo quit | nvidia-cuda-mps-control")
         logger.info("-------------------------------------------\n")
    elif num_processes > 1 and torch.cuda.is_available() and torch.cuda.device_count() > 1:
         logger.info("\n--- 多 GPU 使用多进程注意事项 ---")
         logger.info(f"检测到您使用多进程 (processes > 1) 且有多块 CUDA 设备 ({torch.cuda.device_count()} 块)。")
         logger.info("在这种情况下，每个进程默认会尝试使用可用的第一块 GPU (设备 0)。")
         logger.info("为了充分利用多块 GPU，您可能需要手动为每个进程分配不同的 GPU。")
         logger.info("一种常见方法是在启动每个进程前设置 CUDA_VISIBLE_DEVICES 环境变量。")
         logger.info("例如，如果您有 4 块 GPU (0, 1, 2, 3) 并希望启动 4 个进程，可以考虑编写一个 wrapper 脚本，")
         logger.info("在启动第一个进程前设置 CUDA_VISIBLE_DEVICES=0，第二个设置 CUDA_VISIBLE_DEVICES=1，以此类推。")
         logger.info("或者修改 worker 函数，使其接收一个 device_id 参数，并在函数开始时设置 torch.cuda.set_device(device_id)。")
         logger.info("-----------------------------------\n")
    elif num_processes == 1 and torch.cuda.is_available():
         logger.info("\n--- 单进程单 GPU 训练 ---")
         logger.info("您正在使用单进程进行训练。在单 GPU 环境下，这通常是最高效的训练方式，因为它避免了进程切换和资源竞争的开销。")
         logger.info("---------------------------\n")


    return {
        "status": "completed",
        "message": summary_message,
        "successfully_trained": successfully_trained_count,
        "skipped_no_npz": skipped_due_to_no_npz,
        "skipped_existing_pth": skipped_due_to_existing_pth,
        "failed_training": failed_training_count,
        "total_processed_folders": processed_stock_folders_filter,
        "total_discovered_folders": total_stock_folders,
        "total_to_train": total_stocks_to_train,
        "duration_seconds": duration
    }

# 主执行块
if __name__ == '__main__':
    # 在主进程中配置日志
    root_logger = logging.getLogger()
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.INFO)

    # 导入 torch 检查 CUDA 可用性
    try:
        if torch.cuda.is_available():
            logger.info(f"CUDA 可用。设备数量: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                logger.info(f"设备 {i}: {torch.cuda.get_device_name(i)}")
        else:
            logger.warning("CUDA 不可用。训练将使用 CPU。")
    except ImportError:
        logger.critical("未安装 PyTorch。训练将无法进行。请安装 PyTorch。")
        sys.exit("PyTorch 未安装。")
    except Exception as e:
        logger.warning(f"检查 CUDA 时发生错误: {e}")


    logger.info("--- 脚本开始执行 (run_local_training.py) ---")

    parser = argparse.ArgumentParser(description="本地批量训练 Transformer 模型脚本。")
    # 修改开始：移除参数定义
    # parser.add_argument(
    #     "--params-file",
    #     type=str,
    #     help="指标参数文件的绝对或相对路径。如果未提供，则尝试从 Django settings 获取。"
    # )
    # parser.add_argument(
    #     "--strategy-data-dir",
    #     type=str,
    #     help="包含各个股票代码子文件夹的模型根目录的绝对或相对路径。如果未提供，则尝试从 Django settings 获取。"
    # )
    # 修改结束：移除参数定义

    parser.add_argument(
        "--order",
        type=str,
        choices=['asc', 'desc'],
        default='asc',
        help="处理股票文件夹的顺序。'asc' 为正序 (默认)，'desc' 为倒序。"
    )
    parser.add_argument(
        "--processes",
        type=int,
        default=1,
        help="用于并行训练的工作进程数量。设置为 1 表示串行执行。在单 GPU 环境下，结合 MPS 可能带来性能提升，但受显存限制。"
    )
    args = parser.parse_args()

    # 修改行：更新日志信息，不再打印已移除的参数
    logger.info(f"命令行参数已解析。Order: '{args.order}', Processes: {args.processes}")
    logger.info(f"路径配置将从 Django settings '{DJANGO_SETTINGS_MODULE_NAME}' 中读取。") # 修改行：提示路径来源

    # 检查进程数是否有效
    if args.processes < 1:
        logger.error("错误：进程数 (--processes) 必须大于等于 1。")
        sys.exit("无效的进程数参数。")

    # 启动批量训练任务
    # 修改行：不再传递 model_base_dir_path_str 和 params_file_path_str 参数
    results = run_local_transformer_training_batch(
        processing_order=args.order,
        num_processes=args.processes
    )

    logger.info(f"\n--- 执行结果摘要 ---")
    if results and isinstance(results, dict):
        for key, value in results.items():
            if key == "message" and isinstance(value, str):
                logger.info(f"{key}:")
                for line in value.strip().split('\n'):
                     logger.info(line)
            else:
                logger.info(f"{key}: {value}")
    else:
        logger.error(f"执行返回了意外的结果或错误: {results}")

    logger.info("\n--- 脚本执行完毕 ---")

