#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from functools import partial
import json
import os
from typing import Optional
import django
import logging
from pathlib import Path
import argparse
import optuna
import numpy as np #导入 numpy

# --- 配置 Django 设置 ---
# 请将 'your_project_name.settings' 替换为您的实际 Django 项目的 settings 模块路径。
# 例如: 如果您的项目根目录下有一个名为 'myproject' 的应用包含了 settings.py,
# 并且 'E:\chaoyue_dreams' 是项目根目录 (PYTHONPATH 包含此路径或其父目录),
# 那么这里可能是 'myproject.settings' 或类似的路径。
# 如果 settings.py 在项目根目录下的一个名为 'config' 的文件夹中，那么可能是 'config.settings'
DJANGO_SETTINGS_MODULE_NAME = 'chaoyue_dreams.settings' # <--- 【重要】请用户根据实际项目结构修改此行

DJANGO_SETTINGS_AVAILABLE = False
django_settings_module = None

try:
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', DJANGO_SETTINGS_MODULE_NAME)
    django.setup()
    from django.conf import settings as imported_django_settings
    django_settings_module = imported_django_settings
    DJANGO_SETTINGS_AVAILABLE = True
    print(f"DEBUG: Django settings '{DJANGO_SETTINGS_MODULE_NAME}' 已成功加载和配置。")
except Exception as e:
    print(f"WARNING: 无法加载或配置 Django settings '{DJANGO_SETTINGS_MODULE_NAME}': {e}")
    print("WARNING: 脚本将尝试在没有 Django settings 的情况下运行。某些 Django 特定功能可能受限或失败。")



# --- 模块导入 ---
try:
    from services.indicator_services import IndicatorService
    from dao_manager.tushare_daos.stock_basic_info_dao import StockBasicInfoDao
    from strategies.trend_following_strategy import TrendFollowingStrategy
    print("DEBUG: IndicatorService, StockBasicInfoDao, TrendFollowingStrategy 类已成功导入。")
except ImportError as e:
    print(f"CRITICAL: 无法导入核心业务模块 (如 IndicatorService, TrendBasicInfoDao 等): {e}")
    print("CRITICAL: 请确保这些模块存在于 PYTHONPATH 中，并且 Django 环境已正确配置（如果它们依赖 Django）。")
    import sys
    sys.exit("核心业务模块导入失败，脚本无法继续执行。")
# --- 模块导入结束 ---


# 配置日志记录器
logger = logging.getLogger(__name__)

# 1. 在文件顶部或 main 里，提前生成所有合法组合
d_model_choices = [64, 96]
nhead_choices = [8]
dmodel_nhead_strs = []
for d in d_model_choices:
    for n in nhead_choices:
        if d % n == 0:
            dmodel_nhead_strs.append(f"{d}_{n}")

def objective(trial, strategy, item_name, epochs):
    # 1. 采样参数
    dmodel_nhead_str = trial.suggest_categorical("dmodel_nhead", ['64_8'])  # 只用最优结构
    d_model, nhead = map(int, dmodel_nhead_str.split("_"))
    dim_feedforward = trial.suggest_int("dim_feedforward", 384, 448, step=16)
    nlayers = trial.suggest_int("nlayers", 10, 12)
    dropout = trial.suggest_float("dropout", 0.12, 0.20)
    batch_size = trial.suggest_int("batch_size", 384, 448, step=16)
    learning_rate = trial.suggest_float("learning_rate", 0.0002, 0.0006, log=True)
    weight_decay = trial.suggest_float("weight_decay", 0.0001, 0.001, log=True)
    clip_grad_norm = trial.suggest_float("clip_grad_norm", 0.3, 0.6)
    warmup_epochs = trial.suggest_int("warmup_epochs", 6, 8)
    warmup_start_lr = trial.suggest_float("warmup_start_lr", 2e-6, 1e-5, log=True)
    early_stopping_patience = trial.suggest_int("early_stopping_patience", 25, 30)
    reduce_lr_patience = trial.suggest_int("reduce_lr_patience", 5, 7)
    reduce_lr_factor = trial.suggest_float("reduce_lr_factor", 0.25, 0.40)
    min_lr = trial.suggest_float("min_lr", 5e-7, 2e-6, log=True)
    # 固定激活函数为 gelu
    activation = "gelu"
    # 固定学习率调度器为 CosineAnnealingLR
    lr_scheduler = "CosineAnnealingLR"
    transformer_hyperparams = {
        "transformer_model_config": {
            "d_model": d_model,
            "nhead": nhead,
            "dim_feedforward": dim_feedforward,
            "nlayers": nlayers,
            "dropout": dropout,
            "activation": activation,
            "lr_scheduler": lr_scheduler
        },
        "transformer_training_config": {
            "epochs": epochs,  # 只用外部传入的 epochs
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "warmup_epochs": warmup_epochs,
            "warmup_start_lr": warmup_start_lr,
            "weight_decay": weight_decay,
            "early_stopping_patience": early_stopping_patience,
            "reduce_lr_patience": reduce_lr_patience,
            "reduce_lr_factor": reduce_lr_factor,
            "min_lr": min_lr,
            "clip_grad_norm": clip_grad_norm,
            "use_amp": True,
            "optimizer": "adamw",
            "loss": "mse",
            "monitor_metric": "val_mae",
            "verbose": 1,
            "tensorboard_log_dir": "logs/tensorboard"
        }
    }
    print(f"\n[Optuna][{item_name}] Trial {trial.number} 开始，采样参数如下：")
    for k, v in transformer_hyperparams["transformer_model_config"].items():
        print(f"  [Model] {k}: {v}")
    for k, v in transformer_hyperparams["transformer_training_config"].items():
        print(f"  [Train] {k}: {v}")
    print(f"[Optuna][{item_name}] Trial {trial.number} 开始，采样参数: d_model={transformer_hyperparams['transformer_model_config']['d_model']}, "
          f"nhead={transformer_hyperparams['transformer_model_config']['nhead']}, learning_rate={transformer_hyperparams['transformer_training_config']['learning_rate']:.6e}")
    # 初始化 val_mae 为 NaN，以避免 NoneType 错误
    val_mae = float('nan')
    
    try:
        # 将 train_transformer_model_from_prepared_data 的返回值赋给一个临时变量
        returned_val_mae = strategy.train_transformer_model_from_prepared_data(
            item_name,
            transformer_hyperparams=transformer_hyperparams,
            only_return_val_metric=True,
            trial=trial  # 传入 trial
        )
        # 检查返回值的有效性
        if returned_val_mae is None or (isinstance(returned_val_mae, float) and np.isnan(returned_val_mae)):
            # 如果返回值为 None 或 NaN，说明训练未完成或被内部剪枝，或者返回了无效值
            print(f"[Optuna][{item_name}] Trial {trial.number} 训练未完成或被内部剪枝，未返回有效 val_mae。")
            val_mae = float('nan') # 确保 val_mae 是 NaN，Optuna 会将其视为较差的结果
        else:
            # 如果返回了有效值，则更新 val_mae 并打印
            val_mae = returned_val_mae
            print(f"[Optuna][{item_name}] Trial {trial.number} 训练完成，val_mae={val_mae:.6f}")
        return val_mae
    except optuna.exceptions.TrialPruned:
        # 这个异常块是 Optuna 期望的，当 Trial 被剪枝时，会重新抛出此异常
        print(f"[Optuna][{item_name}] Trial {trial.number} 被早停 (Pruned)。")
        raise # 重新抛出异常，让 Optuna 内部处理
    except Exception as e:
        # 捕获其他所有异常，打印错误信息，并返回一个极大值作为指标，表示该 Trial 失败
        print(f"[Optuna][{item_name}] Trial {trial.number} 发生异常: {e}")
        import traceback
        traceback.print_exc()
        return 1e6  # 出错时返回极大值

def run_local_transformer_training_batch(
    model_base_dir_path_str: str = None,
    params_file_path_str: str = None,
    processing_order: str = 'asc',
    n_trials: int = 20,
    epochs: int = 10,
    n_jobs: int = 1, # n_jobs 参数，默认为 1 (串行)
    gpu_id: Optional[int] = None # gpu_id 参数，默认为 None (不指定)
    ):
    logger.info("开始执行本地 Transformer 模型批量训练任务...")
    # 设置 MPS 相关的环境变量
    # 确保 MPS Daemon 已经运行：sudo nvidia-cuda-mps-control -d
    # 建议将 CUDA_MPS_PIPE_DIRECTORY 设置为一个唯一的、可访问的目录
    mps_pipe_dir = os.getenv('CUDA_MPS_PIPE_DIRECTORY', '/tmp/nvidia-mps')
    os.makedirs(mps_pipe_dir, exist_ok=True)
    os.environ['CUDA_MPS_PIPE_DIRECTORY'] = mps_pipe_dir
    # 对于 V100 (Volta架构) 或更新的 GPU，可以增加最大连接数以提高 MPS 效率
    os.environ['CUDA_DEVICE_MAX_CONNECTIONS'] = '32' # 默认是 16，可以尝试 32 或更高
    logger.info(f"已设置 CUDA_MPS_PIPE_DIRECTORY={os.environ['CUDA_MPS_PIPE_DIRECTORY']}")
    logger.info(f"已设置 CUDA_DEVICE_MAX_CONNECTIONS={os.environ['CUDA_DEVICE_MAX_CONNECTIONS']}")
    if gpu_id is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        logger.info(f"已设置 CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")
    else:
        # 如果未指定 GPU ID，则不设置 CUDA_VISIBLE_DEVICES，让 PyTorch 自动选择
        # 或者根据需要，可以显式地清除它，以防之前有设置
        if 'CUDA_VISIBLE_DEVICES' in os.environ:
            del os.environ['CUDA_VISIBLE_DEVICES']
        logger.info("未指定 GPU ID，CUDA_VISIBLE_DEVICES 未设置。")
    
    # --- 解析和验证路径 ---
    actual_model_base_dir = None
    if model_base_dir_path_str:
        actual_model_base_dir = Path(model_base_dir_path_str)
    elif DJANGO_SETTINGS_AVAILABLE and django_settings_module and hasattr(django_settings_module, 'STRATEGY_DATA_DIR'):
        actual_model_base_dir = Path(django_settings_module.STRATEGY_DATA_DIR)
    else:
        logger.error("错误：模型根目录 (STRATEGY_DATA_DIR) 未指定。请通过命令行参数 (--strategy-data-dir) 或 Django settings 提供。")
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
        # print(f"DEBUG: 使用来自参数的指标参数文件路径: '{actual_params_file}'")
    elif DJANGO_SETTINGS_AVAILABLE and django_settings_module and hasattr(django_settings_module, 'INDICATOR_PARAMETERS_CONFIG_PATH'):
        actual_params_file = Path(django_settings_module.INDICATOR_PARAMETERS_CONFIG_PATH)
        # print(f"DEBUG: 使用 Django settings 的指标参数文件路径: '{actual_params_file}'")
    else:
        logger.error("错误：指标参数文件 (INDICATOR_PARAMETERS_CONFIG_PATH) 未指定。请通过命令行参数 (--params-file) 或 Django settings 提供。")
        if not DJANGO_SETTINGS_AVAILABLE:
            print("DEBUG: Django settings 未成功加载，无法获取 INDICATOR_PARAMETERS_CONFIG_PATH。")
        elif not django_settings_module:
            print("DEBUG: django_settings_module 为空，无法获取 INDICATOR_PARAMETERS_CONFIG_PATH。")
        elif not hasattr(django_settings_module, 'INDICATOR_PARAMETERS_CONFIG_PATH'):
            print(f"DEBUG: Django settings 中未找到 INDICATOR_PARAMETERS_CONFIG_PATH 属性。Django settings available: {DJANGO_SETTINGS_AVAILABLE}")
        return {"status": "error", "message": "指标参数文件配置未找到。"}
    if not actual_params_file.is_file():
        logger.error(f"错误：指定的指标参数文件 '{actual_params_file}' 不存在或不是一个文件。")
        return {"status": "error", "message": f"指标参数文件 '{actual_params_file}' 未找到。"}
    if not actual_model_base_dir.is_dir():
        logger.error(f"错误：指定的模型根目录 '{actual_model_base_dir}' 不存在或不是一个目录。")
        return {"status": "error", "message": f"模型根目录 '{actual_model_base_dir}' 未找到。"}
    successfully_trained_count = 0
    skipped_due_to_no_npz = 0
    skipped_due_to_existing_pth = 0
    failed_training_count = 0
    processed_stock_folders = 0
    # 获取所有股票文件夹名称并根据参数排序
    all_item_names = [item.name for item in actual_model_base_dir.iterdir() if item.is_dir()] # 获取所有子目录名称
    all_item_names.sort() # 默认按字典序正序排序
    if processing_order == 'desc': # 如果参数是倒序
        all_item_names.reverse() # 将列表反转，实现倒序
        print(f"INFO: 将按倒序处理股票文件夹。") # 提示信息
    else:
        print(f"INFO: 将按正序处理股票文件夹。") # 提示信息
    total_stock_folders = len(all_item_names) # 总数现在是排序后的列表长度
    print(f"DEBUG: 在 '{actual_model_base_dir}' 中找到 {total_stock_folders} 个潜在的股票文件夹。")
    # 遍历排序/反转后的 item_name 列表
    for item_idx, item_name in enumerate(all_item_names):
        processed_stock_folders += 1
        item_path = actual_model_base_dir / item_name # 根据 item_name 构建完整路径
        prepared_data_path = item_path / "prepared_data"
        trained_model_path = item_path / "trained_model"
        if not prepared_data_path.is_dir():
            logger.warning(f"[{item_name}] 预处理数据目录 '{prepared_data_path}' 不存在，跳过。")
            skipped_due_to_no_npz += 1
            continue
        npz_files = list(prepared_data_path.glob("*.npz"))
        if not npz_files:
            logger.info(f"[{item_name}] 在 '{prepared_data_path}' 中未找到 *.npz 文件，跳过训练。")
            skipped_due_to_no_npz += 1
            continue
        pth_files = []
        if trained_model_path.is_dir():
            pth_files = list(trained_model_path.glob("*.pth"))
        if pth_files:
            skipped_due_to_existing_pth += 1
            continue
        logger.info(f"开始为股票 {item_name} 执行 Transformer 模型训练...")
        print(f"DEBUG: [{item_name}] 初始化 TrendFollowingStrategy (已从 'strategies.trend_following_strategy' 模块导入)...")
        strategy = TrendFollowingStrategy(params_file=actual_params_file, base_data_dir=actual_model_base_dir)
        try:
            # 贝叶斯优化
            print(f"[Optuna][{item_name}] 贝叶斯优化开始，启用 MedianPruner 早停策略。最大试验次数: {n_trials}，并行进程数: {n_jobs}") # 打印 n_jobs
            study = optuna.create_study(
                direction="minimize",
                pruner=optuna.pruners.MedianPruner(n_warmup_steps=3),
                # 为每个股票代码创建一个独立的 SQLite 数据库文件，避免多进程冲突
                storage=f"sqlite:///{item_path}/optuna_study_{item_name}_2.db",
                study_name=f"transformer_tuning_{item_name}_2",
                load_if_exists=True  #：如果已存在则直接加载
            )
            print("开始 Optuna 超参数优化。")
            objective_with_epochs = partial(objective, strategy=strategy, item_name=item_name, epochs=epochs)
            # 将 n_jobs 参数传递给 study.optimize
            study.optimize(objective_with_epochs, n_trials=n_trials, n_jobs=n_jobs)
            best_params = study.best_params
            print(f"贝叶斯优化结束，最优参数: {best_params}")
            # 用最优参数做最终训练
            training_successful = strategy.train_transformer_model_from_prepared_data(
                item_name,
                transformer_hyperparams=best_params,
                only_return_val_metric=False
            )
            if training_successful and \
            strategy.transformer_model and \
            strategy.feature_scaler and \
            strategy.target_scaler and \
            strategy.selected_feature_names_for_transformer:
                logger.info(f"[{item_name}] Transformer 贝叶斯优化+最终训练成功完成。")
                successfully_trained_count += 1
                # 将 best_params 保存到股票对应的 trained_model 目录下
                best_params_filepath = trained_model_path / f"{item_name}_best_params.json"
                os.makedirs(trained_model_path, exist_ok=True) # 确保目录存在
                with open(best_params_filepath, "w") as f:
                    json.dump(best_params, f, indent=2, ensure_ascii=False)
                print(f"[Optuna][{item_name}] 最优参数已保存到 {best_params_filepath}")
                print(f"[Optuna][{item_name}] 贝叶斯优化结束。最优参数: {study.best_params}")
            else:
                logger.error(f"[{item_name}] Transformer 贝叶斯优化+最终训练逻辑执行完毕，但未能成功标记为训练或必要组件缺失。")
                failed_training_count += 1
        except Exception as e:
            logger.error(f"为股票 {item_name} 执行 Transformer 贝叶斯优化+最终训练时发生意外错误: {e}", exc_info=True)
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
    root_logger = logging.getLogger()
    if root_logger.hasHandlers():
        root_logger.handlers.clear()
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )
    print("--- 脚本开始执行 (run_local_optuna_signal.py) ---")
    # 添加 MPS 启动提示
    print("\n--- MPS (Multi-Process Service) 模式提示 ---")
    print("为了充分利用 V100 显卡的 MPS 模式进行并行训练，请确保您已在后台启动 MPS Daemon。")
    print("启动命令 (可能需要 sudo):")
    print("  sudo nvidia-cuda-mps-control -d")
    print("停止命令:")
    print("  echo quit | nvidia-cuda-mps-control")
    print("-------------------------------------------\n")
    
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
    # 添加 --order 参数
    parser.add_argument(
        "--order",
        type=str,
        choices=['asc', 'desc'], # 限制输入为 'asc' 或 'desc'
        default='asc', # 默认值为 'asc' (正序)
        help="处理股票文件夹的顺序。'asc' 为正序 (默认)，'desc' 为倒序。" # 帮助信息
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=20,
        help="Optuna 贝叶斯优化的 trial 次数，默认 20"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="每次 trial 训练的 epoch 数，默认 10"
    )
    # n_jobs 和 gpu_id 参数
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Optuna 并行运行的 Trial 数量。设置为 -1 表示使用所有可用的 CPU 核心。默认 1 (串行)。"
    )
    parser.add_argument(
        "--gpu-id",
        type=int,
        default=None,
        help="指定用于训练的 GPU ID (例如 0, 1)。如果未指定，则不设置 CUDA_VISIBLE_DEVICES。"
    )
    args = parser.parse_args()
    print(f"INFO: 命令行参数已解析。Params File: '{args.params_file}', Strategy Data Dir: '{args.strategy_data_dir}', Order: '{args.order}', N_Trials: {args.n_trials}, Epochs: {args.epochs}, N_Jobs: {args.n_jobs}, GPU_ID: {args.gpu_id}") # 打印新的参数值
    results = run_local_transformer_training_batch(
        model_base_dir_path_str=args.strategy_data_dir,
        params_file_path_str=args.params_file,
        processing_order=args.order,
        n_trials=args.n_trials,
        epochs=args.epochs,
        n_jobs=args.n_jobs, # 传递 n_jobs 参数
        gpu_id=args.gpu_id # 传递 gpu_id 参数
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

