# tasks/management/commands/test_strategy_signals.py

import asyncio
import pandas as pd
from typing import Dict, Any, List, Optional
from django.core.management.base import BaseCommand

# --- 确保这里的导入路径相对于你的项目结构是正确的 ---
from services.indicator_services import IndicatorService
from strategies.macd_rsi_kdj_boll_strategy import MacdRsiKdjBollEnhancedStrategy

# -----------------------------------------------------

# 将之前定义的 test_strategy_signals 函数放在这里
async def test_strategy_signals(stock_code: str):
    """
    测试指定股票代码的策略信号生成过程。
    (代码与上一个回答中的 test_strategy_signals 函数相同)
    """
    # 1. 初始化服务和策略实例
    indicator_service = IndicatorService()

    # 2. 定义策略参数
    strategy_params: Dict[str, Any] = {
        # --- 完整的策略参数 ---
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
        'volume_tf': '15',
        'check_bearish_divergence': True,
        'divergence_price_period': 5,
        'divergence_threshold_cmf': -0.05,
    }
    strategy_instance = MacdRsiKdjBollEnhancedStrategy(params=strategy_params)

    # 3. 定义策略所需的时间周期
    timeframes: List[str] = ['5', '15', '30', '60']

    # 4. 准备策略数据帧
    print(f"[{stock_code}] 正在准备策略数据...")
    strategy_df: Optional[pd.DataFrame] = await indicator_service.prepare_strategy_dataframe(
        stock_code=stock_code,
        timeframes=timeframes,
        strategy_params=strategy_params,
        limit_per_tf=1200
    )

    # 5. 检查数据准备是否成功
    if strategy_df is None or strategy_df.empty:
        print(f"[{stock_code}] 策略数据准备失败或为空，无法生成信号。")
        return

    print(f"[{stock_code}] 策略数据准备完成，形状: {strategy_df.shape}")
    # print(f"[{stock_code}] 数据帧包含列: {strategy_df.columns.tolist()}") # 取消注释以查看列
    # print(f"[{stock_code}] 数据帧尾部样本:\n{strategy_df.tail()}") # 取消注释以查看数据

    # 6. 生成信号
    print(f"[{stock_code}] 正在生成策略信号...")
    # print("Columns in strategy_df:", strategy_df.columns)
    try:
        signals: pd.Series = strategy_instance.generate_signals(strategy_df)
        print(f"[{stock_code}] 策略信号生成完成。")

        # 7. 查看或使用信号结果
        print(f"[{stock_code}] 最新的信号:")
        print(signals.tail(10))
        print("\n信号统计:")
        print(signals.value_counts())

        intermediate_data = strategy_instance.get_intermediate_data()
        if intermediate_data is not None:
            print("\n中间计算数据 (最后5行):")
            print(intermediate_data.tail())

    except Exception as e:
        print(f"[{stock_code}] 生成信号时出错: {e}")
        import traceback
        traceback.print_exc()


# --- Django Management Command 类 ---
class Command(BaseCommand):
    help = '测试指定股票代码的策略信号生成'

    def add_arguments(self, parser):
        # 添加一个位置参数 stock_code
        parser.add_argument('stock_code', type=str, help='要测试的股票代码 (例如: 000001)')

    def handle(self, *args, **options):
        # 从 options 字典中获取 stock_code 参数
        stock_code_to_test = options['stock_code']

        self.stdout.write(self.style.SUCCESS(f'开始测试策略信号 for {stock_code_to_test}...'))

        # 使用 asyncio.run() 来运行异步的 test_strategy_signals 函数
        try:
            asyncio.run(test_strategy_signals(stock_code=stock_code_to_test))
            self.stdout.write(self.style.SUCCESS(f'策略信号测试完成 for {stock_code_to_test}.'))
        except Exception as e:
            self.stderr.write(self.style.ERROR(f'测试过程中发生错误: {e}'))
            import traceback
            traceback.print_exc() # 打印详细错误堆栈

