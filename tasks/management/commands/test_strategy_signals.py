import asyncio
import pandas as pd
from typing import Dict, Any, List, Optional

# 假设你的文件结构和导入路径如下
from services.indicator_services import IndicatorService # 假设 IndicatorService 在 services 包中
from strategies.macd_rsi_kdj_boll_strategy import MacdRsiKdjBollEnhancedStrategy # 假设策略类在这个路径

async def test_strategy_signals(stock_code: str):
    """
    测试指定股票代码的策略信号生成过程。
    """
    # 1. 初始化服务和策略实例
    indicator_service = IndicatorService()

    # 2. 定义策略参数 (确保这些参数与策略类和 prepare_strategy_dataframe 的需求一致)
    #    这些参数会传递给策略实例，并且 prepare_strategy_dataframe 会从中提取所需周期
    strategy_params: Dict[str, Any] = {
        # --- 必须与 MacdRsiKdjBollEnhancedStrategy 的 __init__ 匹配 ---
        'rsi_period': 14, 'rsi_oversold': 30, 'rsi_overbought': 70,
        'kdj_period_k': 9, 'kdj_period_d': 3, 'kdj_period_j': 3, 'kdj_oversold': 20, 'kdj_overbought': 80,
        'boll_period': 20, 'boll_std_dev': 2,
        'macd_fast': 12, 'macd_slow': 26, 'macd_signal': 9,
        'cci_period': 14, 'cci_threshold': 100,
        'mfi_period': 14, 'mfi_oversold': 20, 'mfi_overbought': 80,
        'roc_period': 12,
        'dmi_period': 14, 'adx_threshold': 20,
        'sar_step': 0.02, 'sar_max': 0.2,
        'amount_ma_period': 20,
        'obv_ma_period': 10,
        'cmf_period': 20,
        'weights': {'5': 0.1, '15': 0.4, '30': 0.3, '60': 0.2},
        'score_thresholds': {'strong_buy': 2.5, 'buy': 0.8, 'sell': -0.8, 'strong_sell': -2.5},
        'volume_confirmation': True,
        'volume_tf': '15', # 量能确认的时间周期，prepare_strategy_dataframe 需要
        'check_bearish_divergence': True,
        'divergence_price_period': 5,
        'divergence_threshold_cmf': -0.05,
    }
    strategy_instance = MacdRsiKdjBollEnhancedStrategy(params=strategy_params)

    # 3. 定义策略所需的时间周期
    timeframes: List[str] = ['5', '15', '30', '60'] # 与策略的 self.timeframes 匹配

    # 4. 准备策略数据帧
    print(f"[{stock_code}] 正在准备策略数据...")
    # 注意：limit_per_tf 需要足够大，以覆盖所有指标计算所需的回溯期 + 你想分析的时间段
    # 例如，如果最长指标周期是 233，你可能需要至少 233 + 50 = 283 条数据，甚至更多
    # 1200 是一个相对安全的值，但可能影响性能
    strategy_df: Optional[pd.DataFrame] = await indicator_service.prepare_strategy_dataframe(
        stock_code=stock_code,
        timeframes=timeframes,
        strategy_params=strategy_params, # 将完整的策略参数传给 prepare_strategy_dataframe
        limit_per_tf=1200 # 获取每个时间周期/指标的最新记录数
    )

    # 5. 检查数据准备是否成功
    if strategy_df is None or strategy_df.empty:
        print(f"[{stock_code}] 策略数据准备失败或为空，无法生成信号。")
        return

    print(f"[{stock_code}] 策略数据准备完成，形状: {strategy_df.shape}")
    print(f"[{stock_code}] 数据帧包含列: {strategy_df.columns.tolist()}")
    # print(f"[{stock_code}] 数据帧尾部样本:\n{strategy_df.tail()}") # 可以取消注释查看数据

    # 6. 生成信号
    print(f"[{stock_code}] 正在生成策略信号...")
    try:
        signals: pd.Series = strategy_instance.generate_signals(strategy_df)
        print(f"[{stock_code}] 策略信号生成完成。")

        # 7. 查看或使用信号结果
        # 打印最后几条信号
        print(f"[{stock_code}] 最新的信号:")
        print(signals.tail(10)) # 打印最后10条信号

        # 你可以在这里添加更多分析或处理信号的代码
        # 例如，统计不同信号的数量
        print("\n信号统计:")
        print(signals.value_counts())

        # 获取中间计算结果（如果需要调试）
        intermediate_data = strategy_instance.get_intermediate_data()
        if intermediate_data is not None:
            print("\n中间计算数据 (最后5行):")
            print(intermediate_data.tail())


    except Exception as e:
        print(f"[{stock_code}] 生成信号时出错: {e}")
        import traceback
        traceback.print_exc()


# --- 如何运行这个测试函数 ---
async def main():
    # 设置你想测试的股票代码
    target_stock_code = '000001' # 例如：平安银行
    # 确保 Django 环境已设置 (如果需要访问数据库模型和设置)
    # import django
    # django.setup()
    await test_strategy_signals(target_stock_code)

if __name__ == "__main__":
    # 在异步环境中运行 main 函数
    # 注意：如果你在 Django 项目中运行，可能需要不同的方式来启动异步任务
    # 例如，在 management command 中使用 asyncio.run()
    # 或者在异步视图/服务中直接 await main()
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("测试被中断。")
