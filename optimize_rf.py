import pandas as pd
import numpy as np
import logging
import os
import json
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error # 使用MAE作为评估指标
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.feature_selection import VarianceThreshold, SelectFromModel
# 暂时不使用PCA和XGBoost，简化示例，如果需要可以加入
# from sklearn.decomposition import PCA
# try:
#     import xgboost as xgb
#     XGB_AVAILABLE = True
# except ImportError:
#     xgb = None
#     XGB_AVAILABLE = False


# 导入 BayesianOptimization
from bayes_opt import BayesianOptimization

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("BayesianOptimization_RF")

# --- 模拟数据和参数加载 ---
# 实际应用中，您会从文件或数据库加载数据
def load_sample_data(num_samples=1000, num_features=30, random_state=42):
    """模拟生成一些时间序列数据"""
    np.random.seed(random_state)
    # 模拟特征数据，加入一些时间序列相关性
    data = pd.DataFrame(np.random.randn(num_samples, num_features), columns=[f'feature_{i}' for i in range(num_features)])
    # 模拟生成一个目标信号，可能依赖于一些特征的移动平均和噪音
    data['target_base'] = data[[f'feature_{i}' for i in range(5)]].sum(axis=1).rolling(window=10).mean().shift(-10) # 目标是未来10个周期的某个值
    data['final_signal'] = np.tanh(data['target_base'] + np.random.randn(num_samples) * 0.5) * 100 # 模拟一个有范围的信号，例如-100到100

    # 填充目标列的NaN (通常是未来数据或计算 awal 的NaN)
    data['final_signal'].fillna(method='ffill', inplace=True)
    data['final_signal'].fillna(method='bfill', inplace=True)
    data['final_signal'].fillna(0, inplace=True) # 如果仍有NaN

    # 填充特征的NaN (如果模拟数据生成中也可能产生)
    data.ffill(inplace=True)
    data.bfill(inplace=True)
    data.fillna(0, inplace=True)

    logger.info(f"模拟数据生成完成，形状: {data.shape}")
    return data

# 模拟参数加载，复用 indicator_parameters.json 中与数据处理和LSTM相关的部分作为参考
# 实际应用中，您可以创建或修改一个专门用于RF优化的参数配置
def load_optimization_params(params_file="indicator_parameters.json"):
    """模拟从参数文件加载用于优化的配置"""
    # 为了简化，直接硬编码或从现有文件结构中提取 relevant 部分
    # 假设我们从 trend_following_params 中提取数据准备和模型相关的参数
    try:
         if os.path.exists(params_file):
             with open(params_file, 'r', encoding='utf-8') as f:
                 all_params = json.load(f)
             tf_params = all_params.get("trend_following_params", {})
             # 提取适用于RF数据准备的参数
             opt_params = {
                 "required_columns": [f'feature_{i}' for i in range(30)], # 模拟使用的特征列
                 "target_column": tf_params.get('lstm_target_column', 'final_signal'),
                 "scaler_type": tf_params.get('lstm_scaler_type', 'minmax'), # 特征缩放器
                 "target_scaler_type": tf_params.get('lstm_target_scaler_type', 'minmax'), # 目标缩放器
                 "train_split": tf_params.get('lstm_train_split', 0.7),
                 "val_split": tf_params.get('lstm_val_split', 0.15),
                 "apply_variance_threshold": tf_params.get('lstm_apply_variance_threshold', False),
                 "variance_threshold_value": tf_params.get('lstm_variance_threshold_value', 0.01),
                 "use_feature_selection": tf_params.get('lstm_use_feature_selection', True),
                 "feature_selector_model": tf_params.get('lstm_feature_selector_model', 'rf'), # 注意：这里RF是用于特征选择的模型
                 "max_features_fs": tf_params.get('lstm_max_features_fs', 20),
                 "feature_selection_threshold": tf_params.get('lstm_feature_selection_threshold', 'median'),
                 "use_pca": tf_params.get('lstm_use_pca', False), # RF通常不需要PCA
                 "pca_n_components": tf_params.get('lstm_pca_n_components', 0.99),
             }
             logger.info(f"从 {params_file} 加载了优化相关参数。")
             return opt_params
         else:
             logger.warning(f"参数文件 {params_file} 不存在，使用硬编码的默认参数。")
             # 硬编码默认参数作为后备
             return {
                 "required_columns": [f'feature_{i}' for i in range(30)],
                 "target_column": 'final_signal',
                 "scaler_type": 'minmax',
                 "target_scaler_type": 'minmax',
                 "train_split": 0.7,
                 "val_split": 0.15,
                 "apply_variance_threshold": False,
                 "variance_threshold_value": 0.01,
                 "use_feature_selection": True,
                 "feature_selector_model": 'rf',
                 "max_features_fs": 20,
                 "feature_selection_threshold": 'median',
                 "use_pca": False,
                 "pca_n_components": 0.99,
             }
    except Exception as e:
        logger.error(f"加载或解析参数文件 {params_file} 时出错，使用硬编码的默认参数: {e}", exc_info=True)
        return {
             "required_columns": [f'feature_{i}' for i in range(30)],
             "target_column": 'final_signal',
             "scaler_type": 'minmax',
             "target_scaler_type": 'minmax',
             "train_split": 0.7,
             "val_split": 0.15,
             "apply_variance_threshold": False,
             "variance_threshold_value": 0.01,
             "use_feature_selection": True,
             "feature_selector_model": 'rf',
             "max_features_fs": 20,
             "feature_selection_threshold": 'median',
             "use_pca": False,
             "pca_n_components": 0.99,
         }


# --- 适合 Random Forest 的数据准备函数 ---
# 适配 prepare_data_for_lstm 的逻辑，但不进行窗口化
def prepare_data_for_rf(data: pd.DataFrame, params: dict):
    """
    准备用于 Random Forest 训练的时间序列数据，包括特征处理、缩放和按时间顺序数据集分割。
    适配了 prepare_data_for_lstm 的部分逻辑，但不进行窗口化。
    """
    logger.info("开始准备 Random Forest 数据...")
    required_columns = params.get('required_columns')
    target_column = params.get('target_column', 'final_signal')
    scaler_type = params.get('scaler_type', 'minmax')
    target_scaler_type = params.get('target_scaler_type', 'minmax')
    train_split = params.get('train_split', 0.7)
    val_split = params.get('val_split', 0.15)
    apply_variance_threshold = params.get('apply_variance_threshold', False)
    variance_threshold_value = params.get('variance_threshold_value', 0.01)
    use_feature_selection = params.get('use_feature_selection', True)
    feature_selector_model = params.get('feature_selector_model', 'rf')
    max_features_fs = params.get('max_features_fs', 20)
    feature_selection_threshold = params.get('feature_selection_threshold', 'median')
    # PCA 对 RF 不是必需的，这里简化不包含 PCA 逻辑，如果需要可以添加

    # 1. 检查目标列和特征列
    if target_column not in data.columns:
        logger.error(f"目标列 '{target_column}' 不存在。")
        raise ValueError(f"目标列 '{target_column}' 不存在。")

    initial_feature_columns = [col for col in required_columns if col in data.columns and col != target_column]
    if not initial_feature_columns:
        logger.error("根据 required_columns 筛选后，没有可用的特征列。")
        raise ValueError("没有可用的特征列。")

    # 2. 处理 NaN 值 (在分割前填充整个数据集)
    data_filled = data.ffill().bfill()
    data_filled[initial_feature_columns + [target_column]].fillna(0, inplace=True) # 对特征和目标填充剩余NaN

    features_processed_flat = data_filled.loc[:, initial_feature_columns].values
    targets_original_flat = data_filled.loc[:, target_column].values
    current_feature_columns = initial_feature_columns[:]

    logger.info(f"初始特征维度 (处理NaN后): {features_processed_flat.shape[1]}")

    # 3. (可选) 方差阈值过滤
    if apply_variance_threshold and features_processed_flat.shape[0] > 1 and features_processed_flat.shape[1] > 0:
        try:
            selector_var = VarianceThreshold(threshold=variance_threshold_value)
            features_processed_flat = selector_var.fit_transform(features_processed_flat)
            selected_indices_var = selector_var.get_support(indices=True)
            current_feature_columns = [current_feature_columns[i] for i in selected_indices_var]
            logger.info(f"方差阈值 ({variance_threshold_value}) 选择后维度: {features_processed_flat.shape[1]}")
        except Exception as e:
             logger.warning(f"应用方差阈值时出错: {e}, 跳过。", exc_info=True)

    # 检查处理后的特征是否为空
    num_initial_features = features_processed_flat.shape[1]
    if num_initial_features == 0:
         logger.error("经过 NaN 处理和方差阈值过滤后，特征维度为零。无法继续。")
         return np.array([]), np.array([]), np.array([]), np.array([]), None, None, 0


    # 4. 按时间顺序分割数据集 (在特征工程和缩放前)
    n_samples_flat = features_processed_flat.shape[0]
    if n_samples_flat == 0:
        logger.error("处理后的数据为空，无法进行分割。")
        return np.array([]), np.array([]), np.array([]), np.array([]), None, None, 0

    if train_split + val_split > 1.0:
         logger.error("训练集和验证集比例之和必须小于等于1。")
         raise ValueError("分割比例错误。")

    n_train_flat = int(n_samples_flat * train_split)
    n_val_flat = int(n_samples_flat * val_split)

    if n_train_flat == 0:
         logger.error("数据分割后训练集为空。无法进行特征工程和缩放。")
         return np.array([]), np.array([]), np.array([]), np.array([]), None, None, 0


    flat_features_train = features_processed_flat[:n_train_flat]
    flat_targets_train = targets_original_flat[:n_train_flat]
    flat_features_val = features_processed_flat[n_train_flat : n_train_flat + n_val_flat]
    flat_targets_val = targets_original_flat[n_train_flat : n_train_flat + n_val_flat]
    # 测试集数据可以准备但不返回，因为优化通常只在 train/val 上进行
    # flat_features_test = features_processed_flat[n_train_flat + n_val_flat:]
    # flat_targets_test = targets_original_flat[n_train_flat + n_val_flat:]


    logger.info(f"数据分割完成 (按时间顺序)，平坦训练集: {flat_features_train.shape[0]} 条，平坦验证集: {flat_features_val.shape[0]} 条")


    # 5. (可选) 在训练集上拟合特征选择转换器 (SelectFromModel)
    features_transformed_train = flat_features_train
    features_transformed_val = flat_features_val
    # transformed_feature_columns = current_feature_columns[:] # 这里我们不需要列名列表，只需要数量

    if use_feature_selection and flat_features_train.shape[1] > 1 and flat_features_train.shape[0] > 1:
        try:
            # 使用 RandomForestRegressor 作为特征重要性评估模型
            if feature_selector_model.lower() == 'rf':
                 selector_model_instance = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            # elif feature_selector_model.lower() == 'xgb' and XGB_AVAILABLE:
            #      selector_model_instance = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42, n_jobs=-1)
            else:
                 logger.warning(f"不支持的特征选择模型: {feature_selector_model} 或 XGBoost 不可用，使用 RandomForest。")
                 selector_model_instance = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

            # 确定选择方式 (数量或阈值)
            if max_features_fs is not None and max_features_fs > 0:
                 actual_max_features = min(max_features_fs, flat_features_train.shape[1])
                 selector = SelectFromModel(selector_model_instance, max_features=actual_max_features, threshold=-np.inf)
                 logger.info(f"使用 {feature_selector_model} 选择最重要的 {actual_max_features} 个特征。")
            else:
                 selector = SelectFromModel(selector_model_instance, threshold=feature_selection_threshold)
                 logger.info(f"使用 {feature_selector_model} 和阈值 '{feature_selection_threshold}' 选择特征。")


            selector.fit(flat_features_train, flat_targets_train)
            features_transformed_train = selector.transform(flat_features_train)
            # 检查验证集是否非空再转换
            if flat_features_val.shape[0] > 0:
                 features_transformed_val = selector.transform(flat_features_val)
            else:
                 features_transformed_val = np.array([])

            logger.info(f"基于模型选择转换完成。训练集形状: {features_transformed_train.shape}, 验证集形状: {features_transformed_val.shape}")

        except Exception as e:
             logger.warning(f"应用基于模型的特征选择时出错: {e}, 跳过。", exc_info=True)
             features_transformed_train = flat_features_train # 特征选择失败，保留原特征
             features_transformed_val = flat_features_val


    num_final_features = features_transformed_train.shape[1]
    if num_final_features == 0:
         logger.error("经过所有预处理步骤后，训练集特征维度为零。无法继续。")
         return np.array([]), np.array([]), np.array([]), np.array([]), None, None, 0

    logger.info(f"最终用于缩放的特征维度: {num_final_features}")


    # 6. 对处理后的特征和原始目标变量进行最终缩放 (在训练集上拟合)

    # 特征缩放
    if scaler_type.lower() == 'minmax':
        feature_scaler = MinMaxScaler()
    elif scaler_type.lower() == 'standard':
        feature_scaler = StandardScaler()
    else:
        logger.warning(f"不支持的特征缩放器类型: {scaler_type}，使用默认MinMaxScaler。")
        feature_scaler = MinMaxScaler()

    if features_transformed_train.shape[0] > 0 and features_transformed_train.shape[1] > 0:
         feature_scaler.fit(features_transformed_train)
         X_train = feature_scaler.transform(features_transformed_train)
         X_val = feature_scaler.transform(features_transformed_val) if features_transformed_val.shape[0] > 0 else np.array([])
         logger.info(f"最终特征缩放完成 (使用 {scaler_type} scaler)。")
    else:
         logger.warning("处理后的训练集特征数据为空，跳过最终特征缩放。")
         X_train = features_transformed_train
         X_val = features_transformed_val
         feature_scaler = None


    # 目标变量缩放
    if target_scaler_type.lower() == 'minmax':
        target_scaler = MinMaxScaler()
    elif target_scaler_type.lower() == 'standard':
        target_scaler = StandardScaler()
    else:
        logger.warning(f"不支持的目标变量缩放器类型: {target_scaler_type}，使用默认MinMaxScaler。")
        target_scaler = MinMaxScaler()

    if flat_targets_train.shape[0] > 0:
        target_scaler.fit(flat_targets_train.reshape(-1, 1))
        y_train = target_scaler.transform(flat_targets_train.reshape(-1, 1)).flatten()
        y_val = target_scaler.transform(flat_targets_val.reshape(-1, 1)).flatten() if flat_targets_val.shape[0] > 0 else np.array([])
        logger.info(f"目标变量缩放完成 (使用 {target_scaler_type} scaler)。")
    else:
        logger.warning("原始训练集目标变量数据为空，跳过目标变量缩放。")
        y_train = flat_targets_train
        y_val = flat_targets_val
        target_scaler = None


    # 返回处理好的训练集和验证集数据，以及目标变量缩放器，用于逆缩放计算MAE
    # feature_scaler 通常不需要在目标函数中使用，但 target_scaler 需要
    return X_train, y_train, X_val, y_val, feature_scaler, target_scaler, num_final_features


# --- 定义 Bayesian Optimization 的目标函数 ---
# 这个函数将被优化器反复调用，以不同的超参数组合来评估模型性能
def rf_objective(n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features_count):
    """
    Random Forest Regressor 的优化目标函数。
    接收超参数，训练模型，并返回验证集上的负MAE（因为需要最大化）。
    超参数由优化器提供，通常是浮点数，需要转换为整数或适当类型。
    """
    # 确保超参数是正确的类型和范围
    n_estimators = int(round(n_estimators)) # 树的数量必须是整数
    max_depth = int(round(max_depth)) if max_depth > 1 else 1 # 最大深度必须是整数，且至少为1
    min_samples_split = int(round(min_samples_split)) # 分裂所需的最小样本数，整数
    min_samples_leaf = int(round(min_samples_leaf)) # 叶节点所需的最小样本数，整数
    max_features_count = int(round(max_features_count)) # 寻找最佳分裂时考虑的特征数，整数

    # 根据 sklearn 文档，min_samples_split 至少为 2，min_samples_leaf 至少为 1
    min_samples_split = max(2, min_samples_split)
    min_samples_leaf = max(1, min_samples_leaf)

    # 确保 max_features_count 不超过实际特征数量
    # 实际特征数量在 prepare_data_for_rf 中确定并返回了 num_final_features
    # 为了在这里使用它，我们需要将它或整个准备好的数据作为全局/外部变量传入
    # 或者在 prepare_data_for_rf 中获取后，传递给优化器创建时的 object_function
    # 这里假设 num_final_features 已经可以访问到 (通过外部变量或闭包)
    # num_final_features 是 prepare_data_for_rf 返回的第五个值
    # 在实际运行时，X_train, y_train, X_val, y_val, target_scaler, num_features 会在调用 optimizer.maximize 前准备好
    # objective 函数可以通过访问外部 scope 的变量来获取这些数据
    # 假设外部已经定义了 X_train, y_train, X_val, y_val, target_scaler, num_final_features
    global X_train_opt, y_train_opt, X_val_opt, y_val_opt, target_scaler_opt, num_final_features_opt

    # 确保 max_features_count 不超过训练数据的实际特征数
    actual_max_features = max(1, min(max_features_count, num_final_features_opt)) # 不能超过特征总数，且至少为1

    logger.debug(f"尝试超参数: n_estimators={n_estimators}, max_depth={max_depth}, "
                 f"min_samples_split={min_samples_split}, min_samples_leaf={min_samples_leaf}, "
                 f"max_features={actual_max_features} (requested={max_features_count})")

    try:
        # 构建 Random Forest Regressor 模型
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=actual_max_features, # 使用处理后的特征数量
            random_state=42, # 保证可复现性
            n_jobs=-1 # 使用所有可用核心加速
        )

        # 训练模型
        model.fit(X_train_opt, y_train_opt)

        # 在验证集上进行预测 (预测结果是缩放后的)
        y_pred_scaled = model.predict(X_val_opt)

        # 将预测结果和真实值逆缩放回原始范围，以便计算可解释的MAE
        # 确保 target_scaler_opt 已成功 fit 且 X_val_opt/y_val_opt 非空
        if target_scaler_opt is not None and X_val_opt.shape[0] > 0:
            # 逆缩放预测值 (需要 reshape 成二维)
            y_pred_original = target_scaler_opt.inverse_transform(y_pred_scaled.reshape(-1, 1))
            # 逆缩放真实值 (需要 reshape 成二维)
            y_val_original = target_scaler_opt.inverse_transform(y_val_opt.reshape(-1, 1))

            # 计算原始范围上的 MAE
            mae_original = mean_absolute_error(y_val_original, y_pred_original)

            # 贝叶斯优化是最大化目标函数，所以返回负的MAE
            score = -mae_original
            logger.debug(f"MAE (原始范围): {mae_original:.4f}, Score: {score:.4f}")
            return score
        else:
            logger.warning("验证集为空或目标缩放器不可用，无法计算有效的MAE。返回一个很小的负值。")
            return -1e9 # 返回一个很小的负值表示失败或不可用

    except Exception as e:
        logger.error(f"模型训练或评估过程中出错，超参数: {e}", exc_info=True)
        # 发生错误时，返回一个很小的负值，表示这组超参数不好或无效
        return -1e9

# --- 执行贝叶斯优化 ---
if __name__ == "__main__":
    # 1. 加载参数和数据
    optimization_params = load_optimization_params()
    # 模拟数据，这里假设所有特征都使用
    sample_data = load_sample_data(num_samples=2000, num_features=30) # 使用更多数据以便分割

    # 2. 准备数据
    # 准备数据时，将结果存储在外部变量中，供 rf_objective 函数访问
    try:
        X_train_opt, y_train_opt, X_val_opt, y_val_opt, feature_scaler_opt, target_scaler_opt, num_final_features_opt = prepare_data_for_rf(sample_data, optimization_params)
        if X_train_opt.shape[0] == 0 or X_val_opt.shape[0] == 0:
             logger.error("数据准备失败，训练集或验证集为空。无法进行优化。")
        else:
            logger.info(f"数据准备完成。训练集形状: {X_train_opt.shape}, 验证集形状: {X_val_opt.shape}")
            logger.info(f"训练集目标形状: {y_train_opt.shape}, 验证集目标形状: {y_val_opt.shape}")
            logger.info(f"最终特征数量: {num_final_features_opt}")

            # 3. 定义超参数搜索空间
            # pbounds 是一个字典，键是目标函数的参数名，值是该参数的搜索范围 (min, max)
            # BayesianOptimization 默认处理浮点数范围，整数需要手动在目标函数中转换和取整
            pbounds = {
                'n_estimators': (50, 500),       # 树的数量，范围50到500
                'max_depth': (5, 30),            # 树的最大深度，范围5到30
                'min_samples_split': (2, 20),    # 分裂所需的最小样本数，范围2到20
                'min_samples_leaf': (1, 10),     # 叶节点所需的最小样本数，范围1到10
                # 寻找最佳分裂时考虑的特征数
                # 直接搜索特征数量，范围1到最终特征数量 num_final_features_opt
                'max_features_count': (1, num_final_features_opt),
            }

            # 4. 初始化贝叶斯优化器
            optimizer = BayesianOptimization(
                f=rf_objective,         # 要优化的目标函数
                pbounds=pbounds,        # 超参数搜索空间
                random_state=1,         # 随机种子，用于结果复现
                verbose=2               # verbose=0: 不打印; verbose=1: 打印发现的改进; verbose=2: 打印所有探针
            )

            # 5. 运行优化
            logger.info("开始运行贝叶斯优化...")
            # init_points: 随机探索的初始点数
            # n_iter: 高斯过程回归的迭代次数 (贝叶斯优化步骤)
            optimizer.maximize(
                init_points=5,  # 初始随机探索5次
                n_iter=25       # 进行25次贝叶斯优化迭代
            )

            # 6. 获取最优结果
            logger.info("\n贝叶斯优化完成.")
            logger.info(f"最优超参数: {optimizer.max['params']}")
            logger.info(f"最优得分 (负MAE，越接近0越好): {optimizer.max['target']:.4f}")

            # 打印原始MAE（通过将负得分取反）
            best_mae_original = -optimizer.max['target']
            logger.info(f"对应的最佳 MAE (原始范围): {best_mae_original:.4f}")

            # 可以选择使用最优参数重新训练最终模型，并在测试集上评估
            # 注意：我们没有返回测试集，实际应用中您需要在 prepare_data_for_rf 中也返回测试集，
            # 并在优化完成后用最优参数训练模型，然后在测试集上进行最终评估。

    except Exception as e:
         logger.error(f"运行贝叶斯优化主流程出错: {e}", exc_info=True)

