#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import django
import logging
from pathlib import Path
import argparse
import torch

# --- 配置 Django 设置 ---
DJANGO_SETTINGS_MODULE_NAME = 'chaoyue_dreams.settings'  # 根据实际项目修改

try:
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', DJANGO_SETTINGS_MODULE_NAME)
    django.setup()
    from django.conf import settings as imported_django_settings_main
    DJANGO_SETTINGS_AVAILABLE_IN_MAIN = True
    print(f"DEBUG: Django settings '{DJANGO_SETTINGS_MODULE_NAME}' 已在主进程中成功加载和配置。")
except Exception as e:
    print(f"WARNING: 无法在主进程中加载或配置 Django settings '{DJANGO_SETTINGS_MODULE_NAME}': {e}")
    DJANGO_SETTINGS_AVAILABLE_IN_MAIN = False
    imported_django_settings_main = None

# --- 模块导入 ---
try:
    from services.indicator_services import IndicatorService
    from dao_manager.tushare_daos.stock_basic_info_dao import StockBasicInfoDao
    from strategies.trend_following_strategy import TrendFollowingStrategy
    print("DEBUG: IndicatorService, StockBasicInfoDao, TrendFollowingStrategy 类已成功导入。")
except ImportError as e:
    print(f"CRITICAL: 无法导入核心业务模块: {e}")
    import sys
    sys.exit("核心业务模块导入失败，脚本无法继续执行。")

# 配置主进程日志记录器
logger = logging.getLogger(__name__)

# --- 单个股票的训练逻辑（主进程直接调用） ---
def train_single_stock_model(item_name, django_settings_module):
    logger.info(f"开始处理股票: {item_name}")
    # 路径读取
    if hasattr(django_settings_module, 'STRATEGY_DATA_DIR'):
        actual_model_base_dir = Path(django_settings_module.STRATEGY_DATA_DIR)
    else:
        logger.error("Django settings 中未找到 STRATEGY_DATA_DIR。")
        return {"item_name": item_name, "status": "failed", "reason": "no_STRATEGY_DATA_DIR"}
    if hasattr(django_settings_module, 'INDICATOR_PARAMETERS_CONFIG_PATH'):
        actual_params_file = Path(django_settings_module.INDICATOR_PARAMETERS_CONFIG_PATH)
    else:
        logger.error("Django settings 中未找到 INDICATOR_PARAMETERS_CONFIG_PATH。")
        return {"item_name": item_name, "status": "failed", "reason": "no_INDICATOR_PARAMETERS_CONFIG_PATH"}
    if not actual_params_file.is_file():
        logger.error(f"指标参数文件 '{actual_params_file}' 不存在。")
        return {"item_name": item_name, "status": "failed", "reason": "params_file_not_found"}
    if not actual_model_base_dir.is_dir():
        logger.error(f"模型根目录 '{actual_model_base_dir}' 不存在。")
        return {"item_name": item_name, "status": "failed", "reason": "model_base_dir_not_found"}
    item_path = actual_model_base_dir / item_name
    prepared_data_path = item_path / "prepared_data"
    trained_model_path = actual_model_base_dir / "trained_model"
    if not prepared_data_path.is_dir():
        logger.warning(f"[{item_name}] 预处理数据目录 '{prepared_data_path}' 不存在，跳过。")
        return {"item_name": item_name, "status": "skipped", "reason": "no_prepared_data_dir"}
    npz_files = list(prepared_data_path.glob("*.npz"))
    if not npz_files:
        return {"item_name": item_name, "status": "skipped", "reason": "no_npz_files"}
    pth_files = []
    if trained_model_path.is_dir():
        pth_files = list(trained_model_path.glob(f"best_transformer_model_{item_name}.pth"))
    if pth_files:
        return {"item_name": item_name, "status": "skipped", "reason": "existing_pth"}
    logger.info(f"开始为股票 {item_name} 执行 Transformer 模型训练...")
    try:
        strategy = TrendFollowingStrategy(params_file=actual_params_file, base_data_dir=actual_model_base_dir)
        training_successful = strategy.train_transformer_model_from_prepared_data(item_name)
        if training_successful and \
           strategy.transformer_model and \
           strategy.feature_scaler and \
           strategy.target_scaler and \
           strategy.selected_feature_names_for_transformer:
            logger.info(f"[{item_name}] Transformer 模型训练成功完成。")
            return {"item_name": item_name, "status": "success"}
        else:
            logger.error(f"[{item_name}] 训练逻辑执行完毕，但未能成功标记为训练或必要组件缺失。")
            return {"item_name": item_name, "status": "failed", "reason": "training_flag_false_or_missing_components"}
    except Exception as e:
        logger.error(f"为股票 {item_name} 执行 Transformer 模型训练时发生意外错误: {e}", exc_info=True)
        return {"item_name": item_name, "status": "failed", "reason": str(e)}

def run_local_transformer_training_batch_single(
    model_base_dir_path_str=None,
    params_file_path_str=None,
    processing_order='asc'
):
    logger.info("开始执行本地 Transformer 模型批量训练任务 (单进程)...")
    # 路径解析
    if DJANGO_SETTINGS_AVAILABLE_IN_MAIN and imported_django_settings_main and hasattr(imported_django_settings_main, 'STRATEGY_DATA_DIR'):
        actual_model_base_dir = Path(imported_django_settings_main.STRATEGY_DATA_DIR)
        logger.info(f"使用 Django settings 的模型根目录: '{actual_model_base_dir}'")
    else:
        logger.error("模型根目录 (STRATEGY_DATA_DIR) 未在 Django settings 中找到。")
        if model_base_dir_path_str:
            logger.warning(f"命令行参数提供了 '{model_base_dir_path_str}'。")
            actual_model_base_dir = Path(model_base_dir_path_str)
        else:
            logger.error("模型根目录未在 Django settings 或命令行参数中指定。")
            return {"status": "error", "message": "模型根目录配置未找到。"}
    if DJANGO_SETTINGS_AVAILABLE_IN_MAIN and imported_django_settings_main and hasattr(imported_django_settings_main, 'INDICATOR_PARAMETERS_CONFIG_PATH'):
        actual_params_file = Path(imported_django_settings_main.INDICATOR_PARAMETERS_CONFIG_PATH)
        logger.info(f"使用 Django settings 的指标参数文件路径: '{actual_params_file}'")
    else:
        logger.error("指标参数文件 (INDICATOR_PARAMETERS_CONFIG_PATH) 未在 Django settings 中找到。")
        if params_file_path_str:
            logger.warning(f"命令行参数提供了 '{params_file_path_str}'。")
            actual_params_file = Path(params_file_path_str)
        else:
            logger.error("指标参数文件未在 Django settings 或命令行参数中指定。")
            return {"status": "error", "message": "指标参数文件配置未找到。"}
    if not actual_model_base_dir.is_dir():
        logger.error(f"指定的模型根目录 '{actual_model_base_dir}' 不存在或不是一个目录。")
        return {"status": "error", "message": f"模型根目录 '{actual_model_base_dir}' 未找到。"}
    if not actual_params_file.is_file():
        logger.error(f"指定的指标参数文件 '{actual_params_file}' 不存在或不是一个文件。")
        return {"status": "error", "message": f"指标参数文件 '{actual_params_file}' 未找到。"}
    all_item_names = [item.name for item in actual_model_base_dir.iterdir() if item.is_dir()]
    all_item_names.sort(reverse=(processing_order == 'desc'))
    total_stock_folders = len(all_item_names)
    logger.info(f"在 '{actual_model_base_dir}' 中找到 {total_stock_folders} 个潜在的股票文件夹。")
    if not all_item_names:
        logger.warning("未找到任何股票文件夹进行处理。")
        return {"status": "completed", "message": "未找到任何股票文件夹进行处理。", "successfully_trained": 0, "skipped_no_npz": 0, "skipped_existing_pth": 0, "failed_training": 0, "total_processed_folders": 0, "total_discovered_folders": 0}
    results_list = []
    for i, item_name in enumerate(all_item_names):
        result = train_single_stock_model(item_name, imported_django_settings_main)
        results_list.append(result)
        logger.info(f"完成 {i+1}/{total_stock_folders} 个股票训练。股票 '{result.get('item_name', 'UNKNOWN')}' 状态: {result.get('status', 'N/A')} 原因： {result.get('reason', 'N/A')}")
    # 结果汇总
    successfully_trained_count = sum(1 for r in results_list if r.get("status") == "success")
    skipped_due_to_no_npz = sum(1 for r in results_list if r.get("reason") in ["no_npz_files", "no_prepared_data_dir"])
    skipped_due_to_existing_pth = sum(1 for r in results_list if r.get("reason") == "existing_pth")
    failed_training_count = sum(1 for r in results_list if r.get("status") == "failed")
    processed_stock_folders = len(results_list)
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
    # 配置主进程的日志记录器
    root_logger = logging.getLogger()
    if root_logger.hasHandlers():
        root_logger.handlers.clear()
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - MAIN_PROCESS - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )
    print("--- 脚本开始执行 (run_local_training_single.py) ---")
    parser = argparse.ArgumentParser(description="本地批量训练 Transformer 模型脚本（单进程版）。")
    parser.add_argument(
        "--params-file",
        type=str,
        help="指标参数文件的绝对或相对路径。仅用于路径校验。"
    )
    parser.add_argument(
        "--strategy-data-dir",
        type=str,
        help="包含各个股票代码子文件夹的模型根目录的绝对或相对路径。仅用于路径校验。"
    )
    parser.add_argument(
        "--order",
        type=str,
        choices=['asc', 'desc'],
        default='asc',
        help="处理股票文件夹的顺序。'asc' 为正序 (默认)，'desc' 为倒序。"
    )
    args = parser.parse_args()
    logger.info(f"命令行参数已解析。Params File: '{args.params_file}', Strategy Data Dir: '{args.strategy_data_dir}', Order: '{args.order}'")
    results = run_local_transformer_training_batch_single(
        model_base_dir_path_str=args.strategy_data_dir,
        params_file_path_str=args.params_file,
        processing_order=args.order
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
