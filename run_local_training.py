#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import django # 修改：导入 django 模块
import logging
from pathlib import Path
import argparse

# --- 修改开始: 配置 Django 设置 ---
# 请将 'your_project_name.settings' 替换为您的实际 Django 项目的 settings 模块路径。
# 例如: 如果您的项目根目录下有一个名为 'myproject' 的应用包含了 settings.py,
# 并且 'E:\chaoyue_dreams' 是项目根目录 (PYTHONPATH 包含此路径或其父目录),
# 那么这里可能是 'myproject.settings' 或类似的路径。
# 如果 settings.py 在项目根目录下的一个名为 'config' 的文件夹中，那么可能是 'config.settings'
DJANGO_SETTINGS_MODULE_NAME = 'chaoyue_dreams.settings' # <--- 【重要】请用户根据实际项目结构修改此行

DJANGO_SETTINGS_AVAILABLE = False # 修改：先假定 Django 设置不可用
django_settings_module = None # 修改：初始化为 None

try:
    # 尝试设置 DJANGO_SETTINGS_MODULE 环境变量
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', DJANGO_SETTINGS_MODULE_NAME)
    # 初始化 Django 设置
    django.setup() # 修改：调用 django.setup()
    # 如果 django.setup() 成功，则可以安全地导入 Django settings
    from django.conf import settings as imported_django_settings
    django_settings_module = imported_django_settings
    DJANGO_SETTINGS_AVAILABLE = True
    print(f"DEBUG: Django settings '{DJANGO_SETTINGS_MODULE_NAME}' 已成功加载和配置。")
except Exception as e: # 修改：捕获更通用的异常，因为 django.setup() 可能抛出多种错误
    print(f"WARNING: 无法加载或配置 Django settings '{DJANGO_SETTINGS_MODULE_NAME}': {e}")
    print("WARNING: 脚本将尝试在没有 Django settings 的情况下运行。某些 Django 特定功能可能受限或失败。")
    # 保持 DJANGO_SETTINGS_AVAILABLE = False 和 django_settings_module = None
# --- 修改结束 ---


# --- 模块导入 ---
# 这里的导入依赖于 Django 的初始化（间接通过模型）
try:
    from services.indicator_services import IndicatorService
    from dao_manager.tushare_daos.stock_basic_info_dao import StockBasicInfoDao
    from strategies.trend_following_strategy import TrendFollowingStrategy
    # 修改：更新调试信息，确认导入来源
    print("DEBUG: IndicatorService, StockBasicInfoDao, TrendFollowingStrategy 类已成功导入。")
except ImportError as e:
    # 修改：提供更具体的错误信息
    print(f"CRITICAL: 无法导入核心业务模块 (如 IndicatorService, TrendFollowingStrategy 等): {e}")
    print("CRITICAL: 请确保这些模块存在于 PYTHONPATH 中，并且 Django 环境已正确配置（如果它们依赖 Django）。")
    import sys
    sys.exit("核心业务模块导入失败，脚本无法继续执行。")
# --- 模块导入结束 ---


# 配置日志记录器
logger = logging.getLogger(__name__)
# (详细配置移至 `if __name__ == '__main__':`)

def run_local_transformer_training_batch(
    model_base_dir_path_str: str = None,
    params_file_path_str: str = None
    ):
    print("DEBUG: !!!!! 函数入口：run_local_transformer_training_batch !!!!!") # 函数入口调试信息
    logger.info("开始执行本地 Transformer 模型批量训练任务...")

    # --- 解析和验证路径 ---
    actual_model_base_dir = None
    if model_base_dir_path_str:
        actual_model_base_dir = Path(model_base_dir_path_str)
        print(f"DEBUG: 使用来自参数的模型根目录: '{actual_model_base_dir}'") # 调试信息：模型根目录来源
    # 修改：使用上面安全获取的 django_settings_module
    elif DJANGO_SETTINGS_AVAILABLE and django_settings_module and hasattr(django_settings_module, 'STRATEGY_DATA_DIR'):
        actual_model_base_dir = Path(django_settings_module.STRATEGY_DATA_DIR)
        print(f"DEBUG: 使用 Django settings 的模型根目录: '{actual_model_base_dir}'") # 调试信息：模型根目录来源
    else:
        logger.error("错误：模型根目录 (STRATEGY_DATA_DIR) 未指定。请通过命令行参数 (--strategy-data-dir) 或 Django settings 提供。")
        # 增加调试信息
        if not DJANGO_SETTINGS_AVAILABLE:
            print("DEBUG: Django settings 未成功加载，无法获取 STRATEGY_DATA_DIR。")
        elif not django_settings_module:
             print("DEBUG: django_settings_module 为空，无法获取 STRATEGY_DATA_DIR。")
        elif not hasattr(django_settings_module, 'STRATEGY_DATA_DIR'):
            print(f"DEBUG: Django settings 中未找到 STRATEGY_DATA_DIR 属性。Django settings available: {DJANGO_SETTINGS_AVAILABLE}")

        return {"status": "error", "message": "模型根目录配置未找到。"}

    actual_params_file = None
    if params_file_path_str:
        actual_params_file = Path(params_file_path_str)
        print(f"DEBUG: 使用来自参数的指标参数文件路径: '{actual_params_file}'") # 调试信息：参数文件路径来源
    # 修改：使用上面安全获取的 django_settings_module
    elif DJANGO_SETTINGS_AVAILABLE and django_settings_module and hasattr(django_settings_module, 'INDICATOR_PARAMETERS_CONFIG_PATH'):
        actual_params_file = Path(django_settings_module.INDICATOR_PARAMETERS_CONFIG_PATH)
        print(f"DEBUG: 使用 Django settings 的指标参数文件路径: '{actual_params_file}'") # 调试信息：参数文件路径来源
    else:
        logger.error("错误：指标参数文件 (INDICATOR_PARAMETERS_CONFIG_PATH) 未指定。请通过命令行参数 (--params-file) 或 Django settings 提供。")
        # 增加调试信息
        if not DJANGO_SETTINGS_AVAILABLE:
            print("DEBUG: Django settings 未成功加载，无法获取 INDICATOR_PARAMETERS_CONFIG_PATH。")
        elif not django_settings_module:
            print("DEBUG: django_settings_module 为空，无法获取 INDICATOR_PARAMETERS_CONFIG_PATH。")
        elif not hasattr(django_settings_module, 'INDICATOR_PARAMETERS_CONFIG_PATH'):
            print(f"DEBUG: Django settings 中未找到 INDICATOR_PARAMETERS_CONFIG_PATH 属性。Django settings available: {DJANGO_SETTINGS_AVAILABLE}")
        return {"status": "error", "message": "指标参数文件配置未找到。"}

    if not actual_params_file.is_file():
        logger.error(f"错误：指定的指标参数文件 '{actual_params_file}' 不存在或不是一个文件。")
        print(f"DEBUG: 验证失败: 指标参数文件 '{actual_params_file}' 不是有效文件。") # 调试信息
        return {"status": "error", "message": f"指标参数文件 '{actual_params_file}' 未找到。"}

    if not actual_model_base_dir.is_dir():
        logger.error(f"错误：指定的模型根目录 '{actual_model_base_dir}' 不存在或不是一个目录。")
        print(f"DEBUG: 验证失败: 模型根目录 '{actual_model_base_dir}' 不是有效目录。") # 调试信息
        return {"status": "error", "message": f"模型根目录 '{actual_model_base_dir}' 未找到。"}

    successfully_trained_count = 0
    skipped_due_to_no_npz = 0
    skipped_due_to_existing_pth = 0
    failed_training_count = 0
    processed_stock_folders = 0

    stock_folders = [item for item in actual_model_base_dir.iterdir() if item.is_dir()]
    total_stock_folders = len(stock_folders)
    print(f"DEBUG: 在 '{actual_model_base_dir}' 中找到 {total_stock_folders} 个潜在的股票文件夹。") # 调试信息：发现的股票文件夹数量

    for item_idx, item_path in enumerate(stock_folders):
        processed_stock_folders += 1
        stock_code = item_path.name
        # 修改：使用 f-string 格式化 print 输出
        print(f"\nDEBUG: --- 正在处理股票 [{stock_code}] ({processed_stock_folders}/{total_stock_folders}) ---")

        prepared_data_path = item_path / "prepared_data"
        trained_model_path = item_path / "trained_model"
        # print(f"DEBUG: [{stock_code}] 预处理数据路径: '{prepared_data_path}'") # 调试信息
        # print(f"DEBUG: [{stock_code}] 已训练模型路径: '{trained_model_path}'") # 调试信息

        if not prepared_data_path.is_dir():
            logger.warning(f"[{stock_code}] 预处理数据目录 '{prepared_data_path}' 不存在，跳过。")
            print(f"DEBUG: [{stock_code}] 目录 '{prepared_data_path}' 未找到。跳过。") # 调试信息
            skipped_due_to_no_npz +=1 # 修正：变量名统一
            continue

        npz_files = list(prepared_data_path.glob("*.npz"))
        if not npz_files:
            logger.info(f"[{stock_code}] 在 '{prepared_data_path}' 中未找到 *.npz 文件，跳过训练。")
            print(f"DEBUG: [{stock_code}] 在 '{prepared_data_path}' 中未找到 NPZ 文件。跳过。") # 调试信息
            skipped_due_to_no_npz += 1
            continue
        # print(f"DEBUG: [{stock_code}] 找到 NPZ 文件: {[f.name for f in npz_files]}") # 调试信息

        pth_files = []
        if trained_model_path.is_dir():
            pth_files = list(trained_model_path.glob("*.pth"))

        if pth_files:
            # logger.info(f"[{stock_code}] 在 '{trained_model_path}' 中已找到 PTH 模型文件 (例如: '{pth_files[0].name}'), 跳过训练。")
            # print(f"DEBUG: [{stock_code}] 在 '{trained_model_path}' 中找到 PTH 文件: {[f.name for f in pth_files]}。跳过。") # 调试信息
            skipped_due_to_existing_pth += 1
            continue
        print(f"DEBUG: [{stock_code}] 在 '{trained_model_path}' 中未找到 PTH 文件 (或目录不存在)。准备训练。") # 调试信息

        logger.info(f"开始为股票 {stock_code} 执行 Transformer 模型训练...")
        # 修改：确认使用的是导入的类，之前的注释有点误导，实际是从 strategies.trend_following_strategy 导入的
        print(f"DEBUG: [{stock_code}] 初始化 TrendFollowingStrategy (已从 'strategies.trend_following_strategy' 模块导入)...")
        strategy = TrendFollowingStrategy(params_file=actual_params_file, base_data_dir=actual_model_base_dir)

        try:
            training_successful = strategy.train_transformer_model_from_prepared_data(stock_code)
            if training_successful and \
               strategy.transformer_model and \
               strategy.feature_scaler and \
               strategy.target_scaler and \
               strategy.selected_feature_names_for_transformer: # 修改：保持原有逻辑的条件检查
                logger.info(f"[{stock_code}] Transformer 模型训练成功完成。")
                successfully_trained_count += 1
            else:
                logger.error(f"[{stock_code}] Transformer 模型训练逻辑执行完毕，但未能成功标记为训练或必要组件缺失。")
                print(f"ERROR: [{stock_code}] 训练方法返回失败或组件未正确设置。Training successful flag: {training_successful}") # 调试信息
                failed_training_count += 1
        except Exception as e:
            logger.error(f"为股票 {stock_code} 执行 Transformer 模型训练时发生意外错误: {e}", exc_info=True)
            print(f"ERROR: [{stock_code}] 训练过程中发生异常: {e}") # 调试信息
            failed_training_count += 1

    summary_message = (
        f"\n本地 Transformer 模型批量训练完成。\n"
        f"总共检查的股票文件夹数量: {processed_stock_folders} (在 {total_stock_folders} 个已发现的文件夹中).\n"
        f"成功训练的模型数量: {successfully_trained_count}.\n"
        f"因缺少NPZ文件或目录而跳过的数量: {skipped_due_to_no_npz}.\n"
        f"因已存在PTH模型而跳过的数量: {skipped_due_to_existing_pth}.\n"
        f"训练失败的数量: {failed_training_count}."
    )
    logger.info(summary_message)
    print(f"DEBUG: {summary_message}") # 调试信息
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
    # 修改：将日志配置移到更前面，确保所有日志都按此配置输出
    # 但要注意，如果 django.setup() 过程中有日志，可能不会遵循此格式，因为那时 logging 可能还未配置
    # 不过，django.setup() 之前的 print 语句可以帮助调试
    # 确保清除现有处理器，以避免日志重复
    root_logger = logging.getLogger()
    if root_logger.hasHandlers():
        root_logger.handlers.clear()
    
    logging.basicConfig(
        level=logging.INFO, # 日志级别
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', # 日志格式
        handlers=[
            logging.StreamHandler() # 输出到控制台
        ]
    )
    print("--- 脚本开始执行 (run_local_training.py) ---") # 脚本开始标记

    parser = argparse.ArgumentParser(description="本地批量训练 Transformer 模型脚本。")
    parser.add_argument(
        "--params-file",
        type=str,
        help="指标参数文件的绝对或相对路径。如果未提供，则尝试从 Django settings 获取。"
    )
    parser.add_argument(
        "--strategy-data-dir",
        type=str,
        help="包含各个股票代码子文件夹的模型根目录的绝对或相对路径。如果未提供，则尝试从 Django settings 获取。"
    )
    args = parser.parse_args()

    print(f"INFO: 命令行参数已解析。Params File: '{args.params_file}', Strategy Data Dir: '{args.strategy_data_dir}'") # 命令行参数解析信息
    
    # 在调用 run_local_transformer_training_batch 之前，Django 设置应该已经尝试加载
    # 所以函数内部的 DJANGO_SETTINGS_AVAILABLE 和 django_settings_module 会是正确的状态
    results = run_local_transformer_training_batch(
        model_base_dir_path_str=args.strategy_data_dir,
        params_file_path_str=args.params_file
    )
    
    print(f"\n--- 执行结果摘要 ---") # 执行结果摘要开始标记
    if results and isinstance(results, dict):
        for key, value in results.items():
            if key == "message" and isinstance(value, str):
                # 修改：确保 message 打印时保留换行符
                print(f"{key}:")
                print(value.strip())
            else:
                print(f"{key}: {value}")
    else:
        print(f"执行返回了意外的结果或错误: {results}")

    print("\n--- 脚本执行完毕 ---") # 脚本结束标记

