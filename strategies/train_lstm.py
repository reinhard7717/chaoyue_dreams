import pandas as pd
from strategies.trend_following_strategy import TrendFollowingStrategy

def train_lstm_for_stock(strategy: TrendFollowingStrategy, data: pd.DataFrame, stock_code: str):
    """
    LSTM训练入口。准备数据、生成规则信号、训练并保存模型。
    """
    # 1. 检查数据
    if data is None or data.empty:
        raise ValueError("训练数据为空")
    # 2. 生成规则信号
    data = data.copy()
    data['final_signal'] = strategy.generate_signals(data, stock_code)
    # 3. 训练并保存模型
    strategy.train_and_save_lstm_model(data, stock_code)
    print(f"LSTM模型训练完成并保存，股票代码: {stock_code}")

if __name__ == '__main__':
    # 1. 加载参数和数据
    params_file = "strategies/indicator_parameters.json"
    stock_code = "600410.SH"
    # 你需要根据实际情况加载历史数据
    data = pd.read_csv(f"your_train_data/{stock_code}.csv")  # 或其它方式
    # 2. 初始化策略
    strategy = TrendFollowingStrategy(params_file)
    # 3. 训练
    train_lstm_for_stock(strategy, data, stock_code)
