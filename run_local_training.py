#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import django
import logging
from pathlib import Path
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed # 引入并行处理模块
import torch # 引入torch，用于设置多进程启动方式和检查CUDA

# --- 配置 Django 设置 ---
# 请将 'your_project_name.settings' 替换为您的实际 Django 项目的 settings 模块路径。
DJANGO_SETTINGS_MODULE_NAME = 'chaoyue_dreams.settings' # <--- 【重要】请用户根据实际项目结构修改此行

# 在主进程中尝试加载 Django settings
DJANGO_SETTINGS_AVAILABLE_IN_MAIN = False
django_settings_module_in_main = None

try:
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', DJANGO_SETTINGS_MODULE_NAME)
    django.setup()
    from django.conf import settings as imported_django_settings_main
    django_settings_module_in_main = imported_django_settings_main
    DJANGO_SETTINGS_AVAILABLE_IN_MAIN = True
    print(f"DEBUG: Django settings '{DJANGO_SETTINGS_MODULE_NAME}' 已在主进程中成功加载和配置。")
except Exception as e:
    print(f"WARNING: 无法在主进程中加载或配置 Django settings '{DJANGO_SETTINGS_MODULE_NAME}': {e}")
    print("WARNING: 脚本将尝试在没有 Django settings 的情况下运行 (主进程)。子进程将再次尝试加载。")



# --- 模块导入 ---
try:
    # 导入核心业务模块。这些模块的实际实例化和使用将在子进程中进行。
    from services.indicator_services import IndicatorService
    from dao_manager.tushare_daos.stock_basic_info_dao import StockBasicInfoDao
    from strategies.trend_following_strategy import TrendFollowingStrategy
    print("DEBUG: IndicatorService, StockBasicInfoDao, TrendFollowingStrategy 类已成功导入。")
except ImportError as e:
    print(f"CRITICAL: 无法导入核心业务模块 (如 IndicatorService, TrendFollowingStrategy 等): {e}")
    print("CRITICAL: 请确保这些模块存在于 PYTHONPATH 中，并且 Django 环境已正确配置（如果它们依赖 Django）。")
    import sys
    sys.exit("核心业务模块导入失败，脚本无法继续执行。")
# --- 模块导入结束 ---


# 配置主进程日志记录器
logger = logging.getLogger(__name__)

# --- 新增函数: 用于子进程中重新配置日志 ---
def _setup_child_process_logging(stock_name):
    """为每个子进程配置独立的日志记录器。"""
    # 清除父进程可能继承的日志处理器，避免冲突
    root = logging.getLogger()
    for handler in root.handlers[:]:
        root.removeHandler(handler)
    # 为子进程添加新的日志处理器
    logging.basicConfig(
        level=logging.INFO,
        format=f'%(asctime)s - CHILD_PROC[{os.getpid()}] - STOCK({stock_name}) - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )
    return logging.getLogger() # 返回新的子进程日志器

# --- 新增函数: 单个股票的训练逻辑 (在子进程中执行) ---
def _train_single_stock_model(item_name: str, django_settings_module_name: str):
    """
    为单个股票代码执行 Transformer 模型训练。
    此函数将在子进程中运行。
    """
    stock_logger = _setup_child_process_logging(item_name)
    stock_logger.info(f"子进程开始处理股票: {item_name}")
    # --- 子进程中重新加载 Django settings ---
    # 这是关键，确保每个子进程有自己独立的Django环境
    try:
        os.environ.setdefault('DJANGO_SETTINGS_MODULE', django_settings_module_name)
        django.setup()
        from django.conf import settings as imported_django_settings_child
        stock_logger.debug(f"子进程成功加载Django settings '{django_settings_module_name}'.")
    except Exception as e:
        stock_logger.error(f"子进程无法加载或配置 Django settings '{django_settings_module_name}': {e}")
        return {
            "item_name": item_name,
            "status": "failed",
            "reason": f"Django settings load error in child: {e}",
            "type": "django_config_error"
        }
    # --- 严格按照要求：从 Django settings 中读取路径 ---
    actual_model_base_dir_child = None
    actual_params_file_child = None
    if hasattr(imported_django_settings_child, 'STRATEGY_DATA_DIR'):
        actual_model_base_dir_child = Path(imported_django_settings_child.STRATEGY_DATA_DIR)
        print(f"子进程从 Django settings 获取模型根目录: '{actual_model_base_dir_child}'")
    else:
        stock_logger.error(f"错误：子进程 Django settings 中未找到 STRATEGY_DATA_DIR。")
        return {"item_name": item_name, "status": "failed", "reason": "STRATEGY_DATA_DIR not in Django settings in child.", "type": "path_config_error"}
    if hasattr(imported_django_settings_child, 'INDICATOR_PARAMETERS_CONFIG_PATH'):
        actual_params_file_child = Path(imported_django_settings_child.INDICATOR_PARAMETERS_CONFIG_PATH)
        stock_logger.debug(f"子进程从 Django settings 获取指标参数文件路径: '{actual_params_file_child}'")
    else:
        stock_logger.error(f"错误：子进程 Django settings 中未找到 INDICATOR_PARAMETERS_CONFIG_PATH。")
        return {"item_name": item_name, "status": "failed", "reason": "INDICATOR_PARAMETERS_CONFIG_PATH not in Django settings in child.", "type": "path_config_error"}
    if not actual_params_file_child.is_file():
        stock_logger.error(f"错误：子进程中指定的指标参数文件 '{actual_params_file_child}' 不存在或不是一个文件。")
        return {"item_name": item_name, "status": "failed", "reason": f"指标参数文件 '{actual_params_file_child}' 未找到。", "type": "file_not_found"}
    if not actual_model_base_dir_child.is_dir():
        stock_logger.error(f"错误：子进程中指定的模型根目录 '{actual_model_base_dir_child}' 不存在或不是一个目录。")
        return {"item_name": item_name, "status": "failed", "reason": f"模型根目录 '{actual_model_base_dir_child}' 未找到。", "type": "dir_not_found"}
    # --- 单个股票训练的核心逻辑 ---
    item_path = actual_model_base_dir_child / item_name
    prepared_data_path = item_path / "prepared_data"
    trained_model_path = actual_model_base_dir_child / "trained_model"
    if not prepared_data_path.is_dir():
        stock_logger.warning(f"[{item_name}] 预处理数据目录 '{prepared_data_path}' 不存在，跳过。")
        return {"item_name": item_name, "status": "skipped", "reason": "no_prepared_data_dir", "type": "skip"}
    npz_files = list(prepared_data_path.glob("*.npz"))
    if not npz_files:
        # stock_logger.info(f"[{item_name}] 在 '{prepared_data_path}' 中未找到 *.npz 文件，跳过训练。")
        return {"item_name": item_name, "status": "skipped", "reason": "no_npz_files", "type": "skip"}
    pth_files = []
    if trained_model_path.is_dir():
        pth_files = list(trained_model_path.glob("*.pth"))
    if pth_files:
        # stock_logger.info(f"[{item_name}] 已存在PTH模型，跳过。")
        return {"item_name": item_name, "status": "skipped", "reason": "existing_pth", "type": "skip"}
    stock_logger.info(f"开始为股票 {item_name} 执行 Transformer 模型训练...")
    # 实例化 TrendFollowingStrategy 必须在子进程内部，因为它可能持有数据库连接或其他资源
    try:
        strategy = TrendFollowingStrategy(params_file=actual_params_file_child, base_data_dir=actual_model_base_dir_child)
        training_successful = strategy.train_transformer_model_from_prepared_data(item_name)
        if training_successful and \
           strategy.transformer_model and \
           strategy.feature_scaler and \
           strategy.target_scaler and \
           strategy.selected_feature_names_for_transformer:
            stock_logger.info(f"[{item_name}] Transformer 模型训练成功完成。")
            return {"item_name": item_name, "status": "success", "type": "trained"}
        else:
            stock_logger.error(f"[{item_name}] Transformer 模型训练逻辑执行完毕，但未能成功标记为训练或必要组件缺失。")
            return {"item_name": item_name, "status": "failed", "reason": "training_flag_false_or_missing_components", "type": "training_error"}
    except Exception as e:
        stock_logger.error(f"为股票 {item_name} 执行 Transformer 模型训练时发生意外错误: {e}", exc_info=True)
        return {"item_name": item_name, "status": "failed", "reason": str(e), "type": "exception"}


def run_local_transformer_training_batch(
    model_base_dir_path_str: str = None, # 命令行参数，但在子进程中会从Django settings重新读取
    params_file_path_str: str = None,   # 命令行参数，但在子进程中会从Django settings重新读取
    processing_order: str = 'asc',
    num_processes: int = None #参数: 并行进程数
    ):
    logger.info("开始执行本地 Transformer 模型批量训练任务 (多进程并行)...")
    # --- 主进程中的路径解析和验证 ---
    # 这些路径会用于确定要遍历哪些文件夹，但实际传递给子进程的
    # 是一个标志，让子进程自行从Django settings读取。
    # 这里只是为了保证在主进程中这些路径是可用的，避免后续错误。
    # 优先从 Django settings 获取
    actual_model_base_dir = None
    if DJANGO_SETTINGS_AVAILABLE_IN_MAIN and django_settings_module_in_main and hasattr(django_settings_module_in_main, 'STRATEGY_DATA_DIR'):
        actual_model_base_dir = Path(django_settings_module_in_main.STRATEGY_DATA_DIR)
        logger.info(f"INFO: 使用 Django settings 的模型根目录: '{actual_model_base_dir}'")
    else:
        logger.error("错误：模型根目录 (STRATEGY_DATA_DIR) 未在 Django settings 中找到。")
        if model_base_dir_path_str:
            logger.warning(f"警告：Django settings 中未找到 STRATEGY_DATA_DIR，但命令行参数提供了 '{model_base_dir_path_str}'。此参数将仅用于列出文件夹，实际训练子进程仍会尝试从 Django settings 获取。")
            actual_model_base_dir = Path(model_base_dir_path_str) # 仅用于获取股票列表
        else:
            logger.error("错误：模型根目录未在 Django settings 或命令行参数中指定。")
            return {"status": "error", "message": "模型根目录配置未找到。"}
    actual_params_file = None
    if DJANGO_SETTINGS_AVAILABLE_IN_MAIN and django_settings_module_in_main and hasattr(django_settings_module_in_main, 'INDICATOR_PARAMETERS_CONFIG_PATH'):
        actual_params_file = Path(django_settings_module_in_main.INDICATOR_PARAMETERS_CONFIG_PATH)
        logger.info(f"INFO: 使用 Django settings 的指标参数文件路径: '{actual_params_file}'")
    else:
        logger.error("错误：指标参数文件 (INDICATOR_PARAMETERS_CONFIG_PATH) 未在 Django settings 中找到。")
        if params_file_path_str:
            logger.warning(f"警告：Django settings 中未找到 INDICATOR_PARAMETERS_CONFIG_PATH，但命令行参数提供了 '{params_file_path_str}'。此参数将仅用于路径校验，实际训练子进程仍会尝试从 Django settings 获取。")
            actual_params_file = Path(params_file_path_str) # 仅用于路径校验
        else:
            logger.error("错误：指标参数文件未在 Django settings 或命令行参数中指定。")
            return {"status": "error", "message": "指标参数文件配置未找到。"}
    # 路径存在性校验
    if not actual_model_base_dir.is_dir():
        logger.error(f"错误：指定的模型根目录 '{actual_model_base_dir}' 不存在或不是一个目录。")
        return {"status": "error", "message": f"模型根目录 '{actual_model_base_dir}' 未找到。"}
    if not actual_params_file.is_file():
        logger.error(f"错误：指定的指标参数文件 '{actual_params_file}' 不存在或不是一个文件。")
        return {"status": "error", "message": f"指标参数文件 '{actual_params_file}' 未找到。"}

    # 获取所有股票文件夹名称并根据参数排序
    all_item_names = [item.name for item in actual_model_base_dir.iterdir() if item.is_dir()]
    all_item_names.sort(reverse=(processing_order == 'desc')) # 根据 order 参数排序
    total_stock_folders = len(all_item_names)
    logger.info(f"在 '{actual_model_base_dir}' 中找到 {total_stock_folders} 个潜在的股票文件夹。")
    if not all_item_names:
        logger.warning("未找到任何股票文件夹进行处理。")
        return {"status": "completed", "message": "未找到任何股票文件夹进行处理。", "successfully_trained": 0, "skipped_no_npz": 0, "skipped_existing_pth": 0, "failed_training": 0, "total_processed_folders": 0, "total_discovered_folders": 0}
    # --- 并行执行 ---
    # 确定最大工作进程数
    max_workers = os.cpu_count() if num_processes is None else num_processes
    if max_workers <= 0:
        max_workers = 1 # 至少一个进程
    logger.info(f"将使用 {max_workers} 个工作进程进行并行训练。")
    if torch.cuda.is_available():
        logger.info(f"CUDA可用。将尝试利用GPU。V100 MPS功能（如已配置）应自动生效，允许多进程共享GPU资源。")
    else:
        logger.warning("CUDA不可用。训练将在CPU上进行，这可能非常慢。")
    results_list = []
    # 使用 'spawn' 启动方法，对PyTorch和CUDA环境更稳定和安全
    # 确保此代码块在 `if __name__ == '__main__':` 内部运行
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=torch.multiprocessing.get_context('spawn')) as executor:
        futures = []
        for item_name in all_item_names:
            futures.append(
                executor.submit(
                    _train_single_stock_model,
                    item_name,
                    DJANGO_SETTINGS_MODULE_NAME # 传递 Django settings 模块名
                )
            )
        # 收集并处理子进程的结果
        for i, future in enumerate(as_completed(futures)):
            try:
                result = future.result()
                results_list.append(result)
                logger.info(f"完成 {i+1}/{total_stock_folders} 个股票训练。股票 '{result.get('item_name', 'UNKNOWN')}' 状态: {result.get('status', 'N/A')}")
            except Exception as exc:
                logger.error(f"一个训练任务在执行过程中发生异常: {exc}", exc_info=True)
                # 为异常情况添加一个失败结果，以便统计
                results_list.append({"item_name": "UNKNOWN_ERROR_PROCESS", "status": "failed", "reason": str(exc), "type": "process_exception"})
    # --- 结果汇总 ---
    successfully_trained_count = sum(1 for r in results_list if r.get("status") == "success")
    skipped_due_to_no_npz = sum(1 for r in results_list if r.get("reason") in ["no_npz_files", "no_prepared_data_dir"])
    skipped_due_to_existing_pth = sum(1 for r in results_list if r.get("reason") == "existing_pth")
    failed_training_count = sum(1 for r in results_list if r.get("status") == "failed")
    processed_stock_folders = len(results_list) # 实际尝试处理的股票数量
    summary_message = (
        f"\n本地 Transformer 模型批量训练完成。\n"
        f"总共发现的股票文件夹数量: {total_stock_folders}.\n"
        f"实际尝试处理的股票文件夹数量: {processed_stock_folders}.\n"
        f"成功训练的模型数量: {successfully_trained_count}.\n"
        f"因缺少NPZ文件或目录而跳过的数量: {skipped_due_to_no_npz}.\n"
        f"因已存在PTH模型而跳过的数量: {skipped_due_to_existing_pth}.\n"
        f"训练失败的数量: {failed_training_count}."
    )
    logger.info(summary_message)
    return {
        "status": "completed",
        "message": summary_message,
        "successfully_trained": successfully_trained_count,
        "skipped_no_npz": skipped_due_to_no_npz,
        "skipped_existing_pth": skipped_due_to_existing_pth,
        "failed_training": failed_training_count,
        "total_processed_folders": processed_stock_folders,
        "total_discovered_folders": total_stock_folders
    }

# 主执行块
if __name__ == '__main__':
    # 确保 PyTorch 使用 'spawn' 启动方法，这对于多进程和CUDA至关重要
    # 必须在任何创建进程池的操作之前调用
    try:
        torch.multiprocessing.set_start_method('spawn', force=True)
        print("DEBUG: PyTorch multiprocessing start method set to 'spawn'.")
    except RuntimeError:
        print("WARNING: PyTorch multiprocessing start method already set.")
    # 配置主进程的日志记录器
    root_logger = logging.getLogger()
    if root_logger.hasHandlers():
        root_logger.handlers.clear() # 清除默认可能存在的处理器
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - MAIN_PROCESS - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )
    print("--- 脚本开始执行 (run_local_training.py) ---")
    parser = argparse.ArgumentParser(description="本地批量训练 Transformer 模型脚本。")
    parser.add_argument(
        "--params-file",
        type=str,
        help="指标参数文件的绝对或相对路径。此参数仅作为信息提供，实际将固定从 Django settings 获取。"
    )
    parser.add_argument(
        "--strategy-data-dir",
        type=str,
        help="包含各个股票代码子文件夹的模型根目录的绝对或相对路径。此参数仅作为信息提供，实际将固定从 Django settings 获取。"
    )
    parser.add_argument(
        "--order",
        type=str,
        choices=['asc', 'desc'],
        default='asc',
        help="处理股票文件夹的顺序。'asc' 为正序 (默认)，'desc' 为倒序。"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None, # 默认None，表示使用CPU核心数
        help="用于并行训练的工作进程数量。默认情况下将使用CPU核心数。"
    )
    args = parser.parse_args()
    logger.info(f"命令行参数已解析。Params File (信息): '{args.params_file}', Strategy Data Dir (信息): '{args.strategy_data_dir}', Order: '{args.order}', Workers: '{args.workers}'")
    results = run_local_transformer_training_batch(
        model_base_dir_path_str=args.strategy_data_dir,
        params_file_path_str=args.params_file,
        processing_order=args.order,
        num_processes=args.workers # 传递进程数
    )
    print(f"\n--- 执行结果摘要 ---")
    if results and isinstance(results, dict):
        for key, value in results.items():
            if key == "message" and isinstance(value, str):
                print(f"{key}:")
                print(value.strip())
            else:
                print(f"{key}: {value}")
    else:
        print(f"执行返回了意外的结果或错误: {results}")
    print("\n--- 脚本执行完毕 ---")

