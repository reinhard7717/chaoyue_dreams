import joblib
import numpy as np
import os
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# scaler 文件路径
scaler_path = r'E:\chaoyue_dreams\models\000001.SZ\prepared_data\trend_following_transformer_target_scaler.save' # 从日志中获取的路径

if os.path.exists(scaler_path):
    try:
        # 加载 scaler 对象
        target_scaler = joblib.load(scaler_path)
        logger.info(f"成功加载缩放器文件: {scaler_path}")

        # 检查加载的对象是否是 MinMaxScaler 或具有相关属性
        if hasattr(target_scaler, 'data_min_') and hasattr(target_scaler, 'data_max_'):
            # data_min_ 和 data_max_ 属性存储了原始数据的最小值和最大值
            # 【代码修改】从 NumPy 数组中取出标量值
            original_min = target_scaler.data_min_[0]
            original_max = target_scaler.data_max_[0]
            logger.info(f"原始信号 ('final_rule_signal') 的最小值: {original_min}")
            logger.info(f"原始信号 ('final_rule_signal') 的最大值: {original_max}")

            # 也可以查看缩放的目标范围 (通常是 [0, 1])
            if hasattr(target_scaler, 'feature_range_'):
                 logger.info(f"信号被缩放到的目标范围: {target_scaler.feature_range_}")

            # 计算原始信号的范围
            # 【代码修改】从 NumPy 数组中取出标量值
            original_range = target_scaler.data_range_[0]
            logger.info(f"原始信号 ('final_rule_signal') 的范围 (max - min): {original_range}")

            # 将之前日志中的真实 MAE 与原始范围进行对比
            true_mae_from_log = 0.3368 # 从日志 Epoch 157 或测试集评估结果中获取的 val_true_mae 或 test_true_mae
            logger.info(f"测试集真实 MAE (从日志获取): {true_mae_from_log}")
            if original_range > 0:
                 # 【代码修改】确保计算使用的是标量值
                 mae_percentage_of_range = (true_mae_from_log / original_range) * 100
                 # 【代码修改】确保格式化的是标量值
                 logger.info(f"测试集真实 MAE 占原始信号范围的百分比: {mae_percentage_of_range:.2f}%")
            else:
                 logger.warning("原始信号范围为零，无法计算 MAE 占范围的百分比。")

        else:
            logger.warning("加载的对象不是预期的 MinMaxScaler 或缺少 data_min_/data_max_ 属性。")

    except Exception as e:
        logger.error(f"加载或检查缩放器时出错: {e}")
else:
    logger.error(f"未找到缩放器文件: {scaler_path}")

