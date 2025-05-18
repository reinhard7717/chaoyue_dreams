import joblib
import numpy as np
import os
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 根目录，所有股票代码文件夹都在这里
base_dir = r'E:\chaoyue_dreams\models'

# 从日志获取的测试集真实 MAE
true_mae_from_log = 0.3368

# 遍历 base_dir 下所有文件夹（假设每个文件夹名是 stock_code）
for stock_code in os.listdir(base_dir):
    stock_dir = os.path.join(base_dir, stock_code)
    if not os.path.isdir(stock_dir):
        continue  # 不是文件夹跳过

    scaler_path = os.path.join(stock_dir, 'prepared_data', 'trend_following_transformer_target_scaler.save')
    if os.path.exists(scaler_path):
        try:
            # 加载 scaler 对象
            target_scaler = joblib.load(scaler_path)
            # logger.info(f"[{stock_code}] 成功加载缩放器文件: {scaler_path}")

            # 检查加载的对象是否是 MinMaxScaler 或具有相关属性
            if hasattr(target_scaler, 'data_min_') and hasattr(target_scaler, 'data_max_'):
                # 从 NumPy 数组中取出标量值
                original_min = target_scaler.data_min_[0]
                original_max = target_scaler.data_max_[0]
                # logger.info(f"[{stock_code}] 原始信号 ('final_rule_signal') 的最小值: {original_min}")
                # logger.info(f"[{stock_code}] 原始信号 ('final_rule_signal') 的最大值: {original_max}")

                # if hasattr(target_scaler, 'feature_range_'):
                    # logger.info(f"[{stock_code}] 信号被缩放到的目标范围: {target_scaler.feature_range_}")

                original_range = target_scaler.data_range_[0]
                # logger.info(f"[{stock_code}] 原始信号 ('final_rule_signal') 的范围 (max - min): {original_range}")

                if original_range > 0:
                    mae_percentage_of_range = (true_mae_from_log / original_range) * 100
                    # logger.info(f"[{stock_code}] 测试集真实 MAE (从日志获取): {true_mae_from_log}")
                    logger.info(f"[{stock_code}] 测试集真实 MAE 占原始信号范围的百分比: {mae_percentage_of_range:.4f}%")
                else:
                    logger.warning(f"[{stock_code}] 原始信号范围为零，无法计算 MAE 占范围的百分比。")
            else:
                logger.warning(f"[{stock_code}] 加载的对象不是预期的 MinMaxScaler 或缺少 data_min_/data_max_ 属性。")

        except Exception as e:
            logger.error(f"[{stock_code}] 加载或检查缩放器时出错: {e}")
    # else:
    #     logger.warning(f"[{stock_code}] 未找到缩放器文件: {scaler_path}")
