# tasks/tushare/train_transformer_tasks.py
import json
import os
import logging
import asyncio
from pathlib import Path # 导入 asyncio

from celery import group
from django.conf import settings
import numpy as np
# 假设 StockBasicInfoDao 存在且可用
from dao_manager.tushare_daos.stock_basic_info_dao import StockBasicInfoDao
import pandas as pd
# 假设 celery 实例存在且可用
from chaoyue_dreams.celery import app as celery_app
# 假设 IndicatorService 存在且可用
from services.indicator_services import IndicatorService
# 导入修改后的 TrendFollowingStrategy 类
from strategies.trend_following_strategy import TrendFollowingStrategy
# 导入 prepare_data_for_transformer 函数
from strategies.utils.deep_learning_utils import prepare_data_for_transformer

logger = logging.getLogger("tasks")

# 任务：准备 Transformer 训练数据并保存
# 修改任务名称以更准确地反映其功能：处理股票数据以进行 Transformer 训练
@celery_app.task(bind=True, name='tasks.tushare.train_transformer_tasks.process_stock_data_for_transformer_training')
def schedule_transformer_data_processing(self, params_file: str = None, base_data_dir: str = None, base_bars_to_request: int = 11200):
    """
    调度器任务：
    1. 获取股票代码列表。
    2. 检查每个股票对应的根目录 (base_data_dir / 股票代码) 是否存在。
    3. 如果根目录存在，检查下级目录 prepared_data 是否存在。
    4. 如果 prepared_data 存在，检查其中是否存在 .npz 文件。
    5. 为目录不存在、prepared_data 不存在或 prepared_data 中没有 .npz 文件的股票创建并分派一个 Transformer 数据处理任务到指定队列。
    这个任务由 Celery Beat 调度，用于触发数据处理的多进程处理。
    :param params_file: 策略参数文件路径
    :param base_data_dir: 模型和数据保存的根目录 (包含各个股票子目录)
    :param base_bars_to_request: 请求的基础 K 线数据量 (用于数据准备任务)
    """
    logger.info(f"任务启动: schedule_transformer_data_processing (调度器模式) - 检查股票数据目录状态并分派缺失任务") # 修改行：日志信息更新
    # 优先使用传入参数，否则使用 Django settings
    if params_file is None:
        # 检查 settings 是否可用以及属性是否存在
        if not hasattr(settings, 'INDICATOR_PARAMETERS_CONFIG_PATH'):
                logger.error("错误：指标参数文件路径未提供且 Django settings 中未配置 INDICATOR_PARAMETERS_CONFIG_PATH。")
                return {"status": "error", "message": "指标参数文件路径未配置", "dispatched_tasks": 0}
        params_file = settings.INDICATOR_PARAMETERS_CONFIG_PATH
    if base_data_dir is None:
        # 检查 settings 是否可用以及属性是否存在
        if not hasattr(settings, 'STRATEGY_DATA_DIR'):
                logger.error("错误：基础数据目录未提供且 Django settings 中未配置 STRATEGY_DATA_DIR。")
                return {"status": "error", "message": "基础数据目录未配置", "dispatched_tasks": 0}
        base_data_dir = settings.STRATEGY_DATA_DIR
    # 确保 base_data_dir 是一个 Path 对象
    base_data_path = Path(base_data_dir)
    if not base_data_path.is_dir():
            logger.error(f"错误：配置的基础数据目录 '{base_data_dir}' 不存在或不是一个目录。")
            return {"status": "error", "message": f"基础数据目录 '{base_data_dir}' 无效", "dispatched_tasks": 0}
    total_dispatched_tasks = 0
    total_skipped_tasks = 0
    total_stocks_checked = 0
    try:
        stock_basic_dao = StockBasicInfoDao()
        # 使用 asyncio.run 来执行异步方法
        all_stocks = asyncio.run(stock_basic_dao.get_stock_list())
        if not all_stocks:
            logger.warning("未获取到股票列表，跳过数据处理任务分派。")
            return {"status": "warning", "message": "未获取到股票列表", "dispatched_tasks": 0}
        logger.info(f"成功获取 {len(all_stocks)} 个股票代码，开始检查股票数据目录状态...") # 修改行：日志信息更新
        for stock in all_stocks:
            total_stocks_checked += 1
            stock_code = stock.stock_code
            # 构建当前股票数据根目录的预期路径 (例如: /data/strategy_models/000001)
            expected_stock_data_root = base_data_path / stock_code
            # 构建 prepared_data 子目录的预期路径
            prepared_data_path = expected_stock_data_root / "prepared_data" # 新增行：构建 prepared_data 目录路径
            should_schedule = False # 新增行：标志是否需要调度任务
            reason = "" # 新增行：记录调度或跳过的原因

            # 检查股票数据根目录是否存在
            if not expected_stock_data_root.is_dir():
                should_schedule = True
                reason = f"股票根目录 '{expected_stock_data_root}' 不存在" # 新增行：记录原因
                logger.info(f"股票 {stock_code}: {reason}，标记为需要调度.") # 新增行：日志说明原因
            else:
                # 如果根目录存在，检查 prepared_data 子目录
                if not prepared_data_path.is_dir(): # 新增行：检查 prepared_data 目录是否存在
                    should_schedule = True
                    reason = f"prepared_data 子目录 '{prepared_data_path}' 不存在" # 新增行：记录原因
                    logger.info(f"股票 {stock_code}: 根目录存在，但 {reason}，标记为需要调度.") # 新增行：日志说明原因
                else:
                    # 如果 prepared_data 子目录存在，检查是否有 .npz 文件
                    # 使用 glob 查找所有 .npz 文件，并用 any() 判断是否存在至少一个
                    has_npz_files = any(prepared_data_path.glob("*.npz")) # 新增行：检查是否存在 .npz 文件
                    if not has_npz_files: # 新增行：如果不存在 .npz 文件
                        should_schedule = True
                        reason = f"prepared_data 子目录 '{prepared_data_path}' 中不存在 .npz 文件" # 新增行：记录原因
                        logger.info(f"股票 {stock_code}: 根目录和 prepared_data 子目录存在，但 {reason}，标记为需要调度.") # 新增行：日志说明原因
                    else:
                        # 根目录、prepared_data 子目录都存在，且存在 .npz 文件，则跳过
                        should_schedule = False # 新增行：明确不需要调度
                        reason = f"股票根目录和 prepared_data 子目录存在且包含 .npz 文件" # 新增行：记录原因
                        logger.info(f"跳过 {stock_code} 的 Transformer 数据处理任务分派 ({reason}).") # 修改行：日志信息说明跳过原因
                        total_skipped_tasks += 1

            # 根据 should_schedule 标志决定是否分派任务
            if should_schedule: # 新增行：根据标志决定是否调度
                logger.info(f"分派 {stock_code} 的 Transformer 数据处理任务到 'Train_Transformer_Prepare_Data' 队列 (原因: {reason}).") # 修改行：日志说明调度原因
                # 调用任务
                prepare_task_signature = schedule_transformer_data_processing.s(
                    stock_code=stock_code,
                    params_file=params_file,
                    model_dir=base_data_dir, # 注意：这里传递的是基础数据目录
                    base_bars=base_bars_to_request
                ).set(queue="Train_Transformer_Prepare_Data")
                prepare_task_signature.apply_async()
                total_dispatched_tasks += 1

        logger.info(f"任务结束: schedule_transformer_data_processing (调度器模式) - 共检查 {total_stocks_checked} 个股票，分派 {total_dispatched_tasks} 个任务，跳过 {total_skipped_tasks} 个任务。") # 修改行：日志信息总结
        return {
            "status": "completed",
            "message": f"共检查 {total_stocks_checked} 个股票，分派 {total_dispatched_tasks} 个任务，跳过 {total_skipped_tasks} 个任务。",
            "dispatched_tasks": total_dispatched_tasks,
            "skipped_tasks": total_skipped_tasks,
            "total_stocks_checked": total_stocks_checked
        }
    except Exception as e:
        logger.error(f"执行 schedule_transformer_data_processing (调度器模式) 时出错: {e}", exc_info=True)
        return {"status": "error", "message": str(e), "dispatched_tasks": total_dispatched_tasks, "skipped_tasks": total_skipped_tasks, "total_stocks_checked": total_stocks_checked}

# 调度器任务：仅调度数据准备任务
# 修改任务名称以匹配新的 prepare 任务名称
@celery_app.task(bind=True, name='tasks.tushare.train_transformer_tasks.schedule_transformer_data_processing') # 修改行：修改任务名称
def schedule_transformer_data_processing(self, params_file: str = None, base_data_dir: str = None, base_bars_to_request: int = 11200):
    """
    调度器任务：
    1. 获取股票代码列表。
    2. 检查每个股票对应的根目录 (base_data_dir / 股票代码) 是否存在。
    3. 如果根目录存在，检查下级目录 prepared_data 是否存在。
    4. 如果 prepared_data 存在，检查其中是否存在 .npz 文件。
    5. 为目录不存在、prepared_data 不存在或 prepared_data 中没有 .npz 文件的股票创建并分派一个 Transformer 数据处理任务到指定队列。
    这个任务由 Celery Beat 调度，用于触发数据处理的多进程处理。
    :param params_file: 策略参数文件路径
    :param base_data_dir: 模型和数据保存的根目录 (包含各个股票子目录)
    :param base_bars_to_request: 请求的基础 K 线数据量 (用于数据准备任务)
    """
    logger.info(f"任务启动: schedule_transformer_data_processing (调度器模式) - 检查股票数据目录状态并分派缺失任务") # 修改行：日志信息更新
    # 优先使用传入参数，否则使用 Django settings
    if params_file is None:
        # 检查 settings 是否可用以及属性是否存在
        if not hasattr(settings, 'INDICATOR_PARAMETERS_CONFIG_PATH'):
            logger.error("错误：指标参数文件路径未提供且 Django settings 中未配置 INDICATOR_PARAMETERS_CONFIG_PATH。")
            return {"status": "error", "message": "指标参数文件路径未配置", "dispatched_tasks": 0}
        params_file = settings.INDICATOR_PARAMETERS_CONFIG_PATH
    if base_data_dir is None:
        # 检查 settings 是否可用以及属性是否存在
        if not hasattr(settings, 'STRATEGY_DATA_DIR'):
            logger.error("错误：基础数据目录未提供且 Django settings 中未配置 STRATEGY_DATA_DIR。")
            return {"status": "error", "message": "基础数据目录未配置", "dispatched_tasks": 0}
        base_data_dir = settings.STRATEGY_DATA_DIR
    # 确保 base_data_dir 是一个 Path 对象
    base_data_path = Path(base_data_dir)
    if not base_data_path.is_dir():
        logger.error(f"错误：配置的基础数据目录 '{base_data_dir}' 不存在或不是一个目录。")
        return {"status": "error", "message": f"基础数据目录 '{base_data_dir}' 无效", "dispatched_tasks": 0}
    total_dispatched_tasks = 0
    total_skipped_tasks = 0
    total_stocks_checked = 0
    try:
        stock_basic_dao = StockBasicInfoDao()
        # 使用 asyncio.run 来执行异步方法
        all_stocks = asyncio.run(stock_basic_dao.get_stock_list())
        if not all_stocks:
            logger.warning("未获取到股票列表，跳过数据处理任务分派。")
            return {"status": "warning", "message": "未获取到股票列表", "dispatched_tasks": 0}
        logger.info(f"成功获取 {len(all_stocks)} 个股票代码，开始检查股票数据目录状态...") # 修改行：日志信息更新
        for stock in all_stocks:
            total_stocks_checked += 1
            stock_code = stock.stock_code
            # 构建当前股票数据根目录的预期路径 (例如: /data/strategy_models/000001)
            expected_stock_data_root = base_data_path / stock_code
            # 构建 prepared_data 子目录的预期路径
            prepared_data_path = expected_stock_data_root / "prepared_data" # 新增行：构建 prepared_data 目录路径
            should_schedule = False # 新增行：标志是否需要调度任务
            reason = "" # 新增行：记录调度或跳过的原因

            # 检查股票数据根目录是否存在
            if not expected_stock_data_root.is_dir():
                should_schedule = True
                reason = f"股票根目录 '{expected_stock_data_root}' 不存在" # 新增行：记录原因
                logger.info(f"股票 {stock_code}: {reason}，标记为需要调度.") # 新增行：日志说明原因
            else:
                # 如果根目录存在，检查 prepared_data 子目录
                if not prepared_data_path.is_dir(): # 新增行：检查 prepared_data 目录是否存在
                    should_schedule = True
                    reason = f"prepared_data 子目录 '{prepared_data_path}' 不存在" # 新增行：记录原因
                    logger.info(f"股票 {stock_code}: 根目录存在，但 {reason}，标记为需要调度.") # 新增行：日志说明原因
                else:
                    # 如果 prepared_data 子目录存在，检查是否有 .npz 文件
                    # 使用 glob 查找所有 .npz 文件，并用 any() 判断是否存在至少一个
                    has_npz_files = any(prepared_data_path.glob("*.npz")) # 新增行：检查是否存在 .npz 文件
                    if not has_npz_files: # 新增行：如果不存在 .npz 文件
                        should_schedule = True
                        reason = f"prepared_data 子目录 '{prepared_data_path}' 中不存在 .npz 文件" # 新增行：记录原因
                        logger.info(f"股票 {stock_code}: 根目录和 prepared_data 子目录存在，但 {reason}，标记为需要调度.") # 新增行：日志说明原因
                    else:
                        # 根目录、prepared_data 子目录都存在，且存在 .npz 文件，则跳过
                        should_schedule = False # 新增行：明确不需要调度
                        reason = f"股票根目录和 prepared_data 子目录存在且包含 .npz 文件" # 新增行：记录原因
                        logger.info(f"跳过 {stock_code} 的 Transformer 数据处理任务分派 ({reason}).") # 修改行：日志信息说明跳过原因
                        total_skipped_tasks += 1

            # 根据 should_schedule 标志决定是否分派任务
            if should_schedule: # 新增行：根据标志决定是否调度
                logger.info(f"分派 {stock_code} 的 Transformer 数据处理任务到 'Train_Transformer_Prepare_Data' 队列 (原因: {reason}).") # 修改行：日志说明调度原因
                # 调用任务，确保使用原始的任务函数名
                prepare_task_signature = schedule_transformer_data_processing.s(
                    stock_code=stock_code,
                    params_file=params_file,
                    model_dir=base_data_dir, # 注意：这里传递的是基础数据目录
                    base_bars=base_bars_to_request
                ).set(queue="Train_Transformer_Prepare_Data")
                prepare_task_signature.apply_async()
                total_dispatched_tasks += 1

        logger.info(f"任务结束: schedule_transformer_data_processing (调度器模式) - 共检查 {total_stocks_checked} 个股票，分派 {total_dispatched_tasks} 个任务，跳过 {total_skipped_tasks} 个任务。") # 修改行：日志信息总结
        return {
            "status": "completed",
            "message": f"共检查 {total_stocks_checked} 个股票，分派 {total_dispatched_tasks} 个任务，跳过 {total_skipped_tasks} 个任务。",
            "dispatched_tasks": total_dispatched_tasks,
            "skipped_tasks": total_skipped_tasks,
            "total_stocks_checked": total_stocks_checked
        }
    except Exception as e:
        logger.error(f"执行 schedule_transformer_data_processing (调度器模式) 时出错: {e}", exc_info=True)
        return {"status": "error", "message": str(e), "dispatched_tasks": total_dispatched_tasks, "skipped_tasks": total_skipped_tasks, "total_stocks_checked": total_stocks_checked}

# 任务1：使用 IndicatorService 准备数据、使用策略生成规则信号 (包括 Transformer 目标列)，之后保存进文件
@celery_app.task(bind=True, name='tasks.tushare.train_transformer_tasks.process_stock_data_stage1', queue='Train_Transformer_Prepare_Data')
def process_stock_data_stage1(self, stock_code: str, params_file: str = None, model_dir: str = None, base_bars: int = 10000):
    """
    任务 1: 为 Transformer 训练准备数据和生成规则信号 (阶段 1 和 2)。
    将结果保存到文件供后续任务使用。
    """
    task_id_str = f"任务 {self.request.id if self.request else 'UnknownID'}"
    logger.info(f"{task_id_str}：开始为 {stock_code} 执行 Transformer 数据处理阶段 1 & 2...")
    try:
        # 记录策略实例化信息
        logger.info(f"{task_id_str} [{stock_code}]：实例化 TrendFollowingStrategy...")
        # 实例化策略，传递参数文件路径和模型目录 (用于确定保存路径)
        strategy = TrendFollowingStrategy(params_file=params_file, base_data_dir=model_dir)
        logger.info(f"{task_id_str} [{stock_code}]：TrendFollowingStrategy 实例化完毕，策略名: '{strategy.strategy_name}'。")
        # 检查策略参数是否为空
        if not strategy.params:
            logger.error(f"{task_id_str} [{stock_code}]：CRITICAL TASK HALT: strategy.params 为空。参数文件可能未加载或无效。任务无法继续。")
            return {"status": "error", "stock_code": stock_code, "message": "策略参数 strategy.params 为空，任务终止。"}
        # 检查 Transformer 相关参数是否存在且非空
        if 'trend_following_params' not in strategy.params or not strategy.tf_params:
            logger.error(f"{task_id_str} [{stock_code}]：CRITICAL TASK HALT: 'trend_following_params' 在 strategy.params 中缺失或 strategy.tf_params 为空。任务无法继续。")
            return {"status": "error", "stock_code": stock_code, "message": "'trend_following_params' 缺失或为空，任务终止。"}
        logger.info(f"{task_id_str} [{stock_code}]：策略参数检查通过 (params 和 tf_params 非空)。")
        logger.info(f"{task_id_str} [{stock_code}]：确认 Transformer 窗口大小 (用于后续的TimeSeriesDataset): {strategy.transformer_window_size}")
        # 阶段 1: 使用 IndicatorService 准备数据
        logger.info(f"{task_id_str} [{stock_code}]：开始使用 IndicatorService 准备数据 (阶段 1)...")
        data_df, actual_indicator_configs = asyncio.run(strategy.prepare_data(stock_code=stock_code, base_needed_count=base_bars))
        if data_df is None or data_df.empty:
            logger.warning(f"{task_id_str} [{stock_code}]：未能获取准备好的数据 (data_df为空)。")
            return {"status": "warning", "stock_code": stock_code, "message": "未能获取准备好的数据"}
        logger.info(f"{task_id_str} [{stock_code}]：准备好的数据获取完成 (阶段 1)，{len(data_df)}行，{len(data_df.columns)}列。")
        logger.info(f"{task_id_str} [{stock_code}]：获取到 {len(actual_indicator_configs)} 个实际使用的指标配置。")
        # 阶段 2: 使用策略生成规则信号 (包括 Transformer 目标列)
        logger.info(f"{task_id_str} [{stock_code}]：调用策略 generate_signals 方法生成规则信号及中间数据 (阶段 2)...")
        data_with_all_signals = strategy.generate_signals(data=data_df, stock_code=stock_code, indicator_configs=actual_indicator_configs)
        if data_with_all_signals is None or data_with_all_signals.empty:
            logger.error(f"{task_id_str} [{stock_code}]：策略 generate_signals 方法返回空或None DataFrame。")
            raise RuntimeError("策略 generate_signals 方法执行失败或未返回数据。")
        logger.info(f"{task_id_str} [{stock_code}]：策略 generate_signals 方法执行完毕 (阶段 2)，返回DataFrame {data_with_all_signals.shape}。")
        # 获取 Transformer 目标列名
        transformer_target_column = strategy.transformer_target_column
        if transformer_target_column not in data_with_all_signals.columns:
            logger.error(f"{task_id_str} [{stock_code}]：返回的 DataFrame 中缺少 Transformer 目标列 '{transformer_target_column}'。")
            raise RuntimeError(f"策略未生成 Transformer 目标列 '{transformer_target_column}'。")
        logger.info(f"{task_id_str} [{stock_code}]：Transformer 目标列 '{transformer_target_column}' 存在。")
        # 确定用于 prepare_data_for_transformer 的初始特征列
        # 这是 data_with_all_signals 中除了目标列之外的所有列
        all_columns = data_with_all_signals.columns.tolist()
        required_columns_for_transformer = [col for col in all_columns if col != transformer_target_column]
        logger.info(f"{task_id_str} [{stock_code}]：用于 prepare_data_for_transformer 的初始特征列数: {len(required_columns_for_transformer)}")
        # --- 新增：保存阶段 1 和 2 的结果到文件 ---
        logger.info(f"{task_id_str} [{stock_code}]：开始保存阶段 1 & 2 的中间数据...")
        dataframe_path, metadata_path = strategy.save_intermediate_data(
            stock_code,
            data_with_all_signals,
            transformer_target_column,
            strategy.tf_params, # 保存 tf_params 供阶段 2 使用
            strategy.transformer_window_size, # 保存 window_size 供后续使用 (虽然 prepare_data_for_transformer 不直接用，但它是流程的一部分)
            required_columns_for_transformer # 保存初始特征列列表
        )
        logger.info(f"{task_id_str} [{stock_code}]：阶段 1 & 2 中间数据已保存。")
        # --- 新增结束 ---
        # 返回保存的文件路径和 stock_code，供阶段 2 任务使用
        return {
            "status": "success",
            "stock_code": stock_code,
            "dataframe_path": dataframe_path,
            "metadata_path": metadata_path,
            "message": "阶段 1 & 2 数据处理并保存成功。"
        }
    except Exception as e:
        logger.error(f"{task_id_str} [{stock_code}]：执行阶段 1 或 2 时出错: {e}", exc_info=True)
        # 返回错误状态和 stock_code
        return {"status": "error", "stock_code": stock_code, "message": f"阶段 1 或 2 执行失败: {e}"}

# 任务2：从文件读取数据，准备 Transformer 训练数据并保存 (阶段 3 和 4)
@celery_app.task(bind=True, name='tasks.tushare.train_transformer_tasks.process_stock_data_stage2', queue='Train_Transformer_Prepare_Data')
def process_stock_data_stage2(self, stock_code: str, dataframe_path: str, metadata_path: str, params_file: str = None, model_dir: str = None):
    """
    任务 2: 从文件读取数据，准备 Transformer 训练数据并保存 (阶段 3 和 4)。
    """
    task_id_str = f"任务 {self.request.id if self.request else 'UnknownID'}"
    logger.info(f"{task_id_str}：开始为 {stock_code} 执行 Transformer 数据处理阶段 3 & 4...")
    try:
        # 记录策略实例化信息 (需要 params_file 和 model_dir 来调用 save_prepared_data)
        logger.info(f"{task_id_str} [{stock_code}]：实例化 TrendFollowingStrategy (用于加载中间数据和保存最终结果)...")
        strategy = TrendFollowingStrategy(params_file=params_file, base_data_dir=model_dir)
        logger.info(f"{task_id_str} [{stock_code}]：TrendFollowingStrategy 实例化完毕。")
        # --- 新增：从文件加载阶段 1 和 2 的结果 ---
        logger.info(f"{task_id_str} [{stock_code}]：开始从文件加载中间数据...")
        data_for_prep, transformer_target_column, tf_params, transformer_window_size, required_columns_for_transformer = strategy.load_intermediate_data(
            dataframe_path, metadata_path
        )
        logger.info(f"{task_id_str} [{stock_code}]：中间数据加载完成，DataFrame shape: {data_for_prep.shape}。")
        logger.info(f"{task_id_str} [{stock_code}]：加载的 Transformer 目标列: '{transformer_target_column}'")
        logger.info(f"{task_id_str} [{stock_code}]：加载的 tf_params: {tf_params}")
        logger.info(f"{task_id_str} [{stock_code}]：加载的 Transformer 窗口大小: {transformer_window_size}")
        logger.info(f"{task_id_str} [{stock_code}]：加载的初始特征列数: {len(required_columns_for_transformer)}")
        # --- 新增结束 ---

        # 阶段 3: 准备 Transformer 训练数据 (特征选择、标准化、分割)
        # --- 修改：调用新的 strategy.process_loaded_intermediate_data 方法 ---
        logger.info(f"{task_id_str} [{stock_code}]：调用 strategy.process_loaded_intermediate_data 方法准备 Transformer 数据 (阶段 3)...")
        features_scaled_train, targets_scaled_train, \
        features_scaled_val, targets_scaled_val, \
        features_scaled_test, targets_scaled_test, \
        feature_scaler, target_scaler, \
        selected_feature_names = strategy.process_loaded_intermediate_data(
            data_for_prep=data_for_prep,
            transformer_target_column=transformer_target_column,
            tf_params=tf_params,
            required_columns_for_transformer=required_columns_for_transformer
        )
        # --- 修改结束 ---

        logger.info(f"{task_id_str} [{stock_code}]：Transformer 数据准备调用成功 (阶段 3)。")
        # 检查数据准备结果是否有效
        if features_scaled_train is None or (isinstance(features_scaled_train, np.ndarray) and features_scaled_train.shape[0] == 0) or \
           targets_scaled_train is None or (isinstance(targets_scaled_train, np.ndarray) and targets_scaled_train.shape[0] == 0) or \
           feature_scaler is None or target_scaler is None or not selected_feature_names:
             logger.error(f"{task_id_str} [{stock_code}]：Transformer 数据准备后，训练集为空或 Scaler/特征列表未成功生成。")
             raise ValueError("Transformer 数据准备后，训练集为空或 Scaler/特征列表未成功生成。")
        logger.info(f"{task_id_str} [{stock_code}]：Transformer 数据准备完成 (阶段 3)。训练集 shape: {features_scaled_train.shape}, 最终特征数: {len(selected_feature_names)}")
        # 阶段 4: 保存准备好的数据和 Scaler
        # logger.info(f"{task_id_str} [{stock_code}]：开始保存准备好的数据和 Scaler (阶段 4)...")
        strategy.save_prepared_data(
            stock_code,
            features_scaled_train, targets_scaled_train,
            features_scaled_val, targets_scaled_val,
            features_scaled_test, targets_scaled_test,
            feature_scaler, target_scaler,
            selected_feature_names
        )
        logger.info(f"{task_id_str} [{stock_code}]：准备好的 Transformer 数据和 Scaler 已成功保存 (阶段 4)。")
        # --- 新增：清理中间文件 (可选，根据需求决定是否清理) ---
        try:
            os.remove(dataframe_path)
            os.remove(metadata_path)
            logger.info(f"{task_id_str} [{stock_code}]：已清理中间数据文件。")
        except OSError as e:
            logger.warning(f"{task_id_str} [{stock_code}]：清理中间数据文件失败: {e}")
        # --- 新增结束 ---
        # 返回成功状态和相关信息
        return {"status": "success", "stock_code": stock_code, "train_samples": features_scaled_train.shape[0], "final_features": len(selected_feature_names)}
    except FileNotFoundError:
        logger.error(f"{task_id_str} [{stock_code}]：中间数据文件未找到，任务终止。")
        return {"status": "error", "stock_code": stock_code, "message": "中间数据文件未找到，任务终止。"}
    except Exception as e:
        logger.error(f"{task_id_str} [{stock_code}]：执行阶段 3 或 4 时出错: {e}", exc_info=True)
        # 返回错误状态和 stock_code
        return {"status": "error", "stock_code": stock_code, "message": f"阶段 3 或 4 执行失败: {e}"}

# 任务1的调度器任务
@celery_app.task(bind=True, name='tasks.tushare.train_transformer_tasks.schedule_stage1_tasks', queue="celery")
def schedule_stage1_tasks(self, stock_codes: list, params_file: str = None, model_dir: str = None, base_bars: int = 10000):
    """
    任务 3: 调度并行执行 process_stock_data_stage1 任务。
    """
    task_id_str = f"调度任务 {self.request.id if self.request else 'UnknownID'}"
    logger.info(f"{task_id_str}：开始调度 {len(stock_codes)} 个股票的阶段 1 任务...")
    # 创建一个任务组 (group)
    stage1_tasks = group(
        process_stock_data_stage1.s(stock_code, params_file, model_dir, base_bars)
        for stock_code in stock_codes
    )
    # 异步执行任务组
    result = stage1_tasks.apply_async()
    logger.info(f"{task_id_str}：已提交阶段 1 任务组，任务组 ID: {result.id}")
    # 返回任务组 ID 或其他相关信息
    return {"status": "submitted", "task_group_id": result.id, "stock_count": len(stock_codes)}

# 任务2的调度器任务
@celery_app.task(bind=True, name='tasks.tushare.train_transformer_tasks.schedule_stage2_tasks', queue="celery")
def schedule_stage2_tasks(self, stage1_results: list, params_file: str = None, model_dir: str = None):
    """
    任务 4: 调度并行执行 process_stock_data_stage2 任务。
    接收阶段 1 调度任务的结果列表作为输入。
    """
    task_id_str = f"调度任务 {self.request.id if self.request else 'UnknownID'}"
    logger.info(f"{task_id_str}：开始调度阶段 2 任务...")
    stage2_tasks_params = []
    successful_stage1_count = 0
    # 遍历阶段 1 的结果，收集成功任务的输出作为阶段 2 的输入
    for result in stage1_results:
        # 检查结果是否是成功的字典，并且包含必要的文件路径
        if isinstance(result, dict) and result.get("status") == "success" and \
           "stock_code" in result and "dataframe_path" in result and "metadata_path" in result:
            stage2_tasks_params.append({
                "stock_code": result["stock_code"],
                "dataframe_path": result["dataframe_path"],
                "metadata_path": result["metadata_path"]
            })
            successful_stage1_count += 1
        else:
            # 记录阶段 1 失败或无效的结果
            stock_code = result.get("stock_code", "UnknownStock") if isinstance(result, dict) else "UnknownStock"
            logger.warning(f"{task_id_str}：阶段 1 任务为股票 {stock_code} 失败或返回无效结果，跳过阶段 2 调度。结果: {result}")
    if not stage2_tasks_params:
        logger.warning(f"{task_id_str}：没有成功的阶段 1 任务结果，无需调度阶段 2 任务。")
        return {"status": "skipped", "message": "没有成功的阶段 1 任务结果。"}
    logger.info(f"{task_id_str}：将为 {len(stage2_tasks_params)} 个股票调度阶段 2 任务 (来自 {successful_stage1_count} 个成功的阶段 1 任务)。")
    # 创建一个任务组 (group)
    stage2_tasks = group(
        process_stock_data_stage2.s(
            params["stock_code"],
            params["dataframe_path"],
            params["metadata_path"],
            params_file, # 将 params_file 传递给阶段 2
            model_dir # 将 model_dir 传递给阶段 2
        )
        for params in stage2_tasks_params
    )
    # 异步执行任务组
    result = stage2_tasks.apply_async()
    logger.info(f"{task_id_str}：已提交阶段 2 任务组，任务组 ID: {result.id}")
    # 返回任务组 ID 或其他相关信息
    return {"status": "submitted", "task_group_id": result.id, "stock_count": len(stage2_tasks_params)}


# 任务：训练 Transformer 模型 (从已准备数据加载)
@celery_app.task(bind=True, name='tasks.tushare.train_transformer_tasks.batch_train_following_strategy_transformer', queue="Train_Transformer_Model")
def batch_train_following_strategy_transformer(self, stock_code: str, params_file: str = "", model_dir=""):
    print(f"DEBUG: !!!!! Celery task ENTRY for {stock_code} !!!!!") # 修改：增加一个醒目的print语句
    logger.info(f"开始执行 {stock_code} 的 Transformer 模型训练任务...")
    # 实例化策略，传递参数文件和模型目录
    params_file=settings.INDICATOR_PARAMETERS_CONFIG_PATH
    model_dir = settings.STRATEGY_DATA_DIR # 修改行：使用设置的模型目录
    strategy = TrendFollowingStrategy(params_file=params_file, base_data_dir=model_dir)
    try:
        # strategy.train_transformer_model_from_prepared_data 内部会加载数据并训练
        strategy.train_transformer_model_from_prepared_data(stock_code)
        if strategy.transformer_model and strategy.feature_scaler and \
           strategy.target_scaler and strategy.selected_feature_names_for_transformer:
            logger.info(f"[{stock_code}] Transformer 模型训练任务完成。")
            return {"status": "success", "stock_code": stock_code}
        else:
            logger.error(f"[{stock_code}] Transformer 模型训练任务失败，模型或相关组件未成功加载/训练。")
            # 训练失败时，抛出异常以便 Celery 标记任务失败
            raise RuntimeError("Transformer 模型训练任务失败，模型或相关组件未成功加载/训练。")
    except Exception as e:
        logger.error(f"执行股票 {stock_code} 的 Transformer 模型训练任务时发生意外错误: {e}", exc_info=True)
        raise e # 重新抛出异常

# 调度器任务: 训练 Transformer 模型
@celery_app.task(bind=True, name='tasks.tushare.train_transformer_tasks.schedule_transformer_training_chain')
def schedule_transformer_training_chain(self): # 参数名一致性
    """
    调度器任务：训练 Transformer 模型
    这个任务由 Celery Beat 调度。
    """
    logger.info(f"任务启动: schedule_transformer_training_chain (调度器模式) - 获取股票列表并分派任务")
    try:
        total_dispatched_chains = 0
        stock_basic_dao = StockBasicInfoDao()
        all_stocks = asyncio.run(stock_basic_dao.get_stock_list())
        if not all_stocks:
             logger.warning("未获取到股票列表，跳过任务链分派。")
             return {"status": "warning", "message": "未获取到股票列表", "dispatched_chains": 0}
        for stock in all_stocks:
            stock_code = stock.stock_code
            logger.info(f"创建 {stock_code} 的 Transformer 数据处理和模型训练任务链...") # 修改行：日志信息
            # 定义模型训练任务签名 (调用 batch_train_following_strategy_transformer)
            train_task_signature = batch_train_following_strategy_transformer.s(
                stock_code=stock_code
            ).set().apply_async() # 指定模型训练队列 (新的队列名建议)

            total_dispatched_chains += 1

        logger.info(f"任务结束: schedule_transformer_training_chain (调度器模式) - 共分派 {total_dispatched_chains} 个任务") # 修改行：日志信息
        return {"status": "success", "dispatched_chains": total_dispatched_chains}

    except Exception as e:
        logger.error(f"执行 schedule_transformer_training_chain (调度器模式) 时出错: {e}", exc_info=True) # 修改行：日志信息
        return {"status": "error", "message": str(e), "dispatched_chains": 0}

