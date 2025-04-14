# tasks/management/commands/test_strategy_signals.py

import asyncio
import pandas as pd
from typing import Dict, Any, List, Optional
from django.core.management.base import BaseCommand
import logging # Import logging

# --- 新增：导入时区处理库 ---
try:
    import tzlocal
    from zoneinfo import ZoneInfo # zoneinfo 是 Python 3.9+ 内置库
except ImportError:
    tzlocal = None
    ZoneInfo = None
    print("警告：无法导入 'tzlocal' 或 'zoneinfo'。时间将不会转换为本地时间。请运行 'pip install tzlocal'")
# ---------------------------

# --- 确保这里的导入路径相对于你的项目结构是正确的 ---
# Assuming the service correctly provides all necessary base indicator columns
from dao_manager.daos.stock_basic_dao import StockBasicDAO
from services.indicator_services import IndicatorService
# Import the updated strategy class
from strategies.macd_rsi_kdj_boll_strategy import MacdRsiKdjBollEnhancedStrategy
# -----------------------------------------------------

logger = logging.getLogger(__name__) # Use standard logging

async def test_strategy_scores(stock_code: str): # Renamed function for clarity
    """
    测试指定股票代码的策略评分生成过程 (0-100分)。
    """
    # --- 获取本地时区 ---
    local_tz = None
    local_tz_name = "系统默认" # 用于打印
    if tzlocal:
        try:
            local_tz = tzlocal.get_localzone() # 获取本地时区对象
            local_tz_name = str(local_tz) # 获取时区名称字符串
            print(f"检测到本地时区: {local_tz_name}")
        except Exception as tz_e:
            print(f"获取本地时区时出错: {tz_e}. 时间将不会转换。")
            local_tz = None
    # --------------------
    # 1. 初始化服务和策略实例
    indicator_service = IndicatorService()
    stock_basic_dao = StockBasicDAO()

    # 2. 定义策略参数 (使用新策略的参数)
    strategy_params: Dict[str, Any] = {
        # --- 指标周期参数 ---
        'rsi_period': 12,
        'rsi_oversold': 30,
        'rsi_overbought': 70,
        'rsi_extreme_oversold': 20, # 新增
        'rsi_extreme_overbought': 80, # 新增
        'kdj_period_k': 9,
        'kdj_period_d': 3,
        'kdj_period_j': 3,
        'kdj_oversold': 20,
        'kdj_overbought': 80,
        'boll_period': 20,
        'boll_std_dev': 2,
        'macd_fast': 10,
        'macd_slow': 26,
        'macd_signal': 9,
        'cci_period': 14,
        'cci_threshold': 100,
        'cci_extreme_threshold': 200, # 新增
        'mfi_period': 14,
        'mfi_oversold': 20,
        'mfi_overbought': 80,
        'mfi_extreme_oversold': 10, # 新增
        'mfi_extreme_overbought': 90, # 新增
        'roc_period': 12,
        'dmi_period': 14,
        'adx_threshold': 20,
        'adx_strong_threshold': 30, # 新增
        'sar_step': 0.02,
        'sar_max': 0.2,
        'amount_ma_period': 20,
        'obv_ma_period': 10,
        'cmf_period': 20,
        # --- 权重 ---
        'weights': {'5': 0.1, '15': 0.4, '30': 0.3, '60': 0.2},
        # --- 移除了 score_thresholds ---
        # --- 量能确认与调整 ---
        'volume_confirmation': True,
        'volume_tf': '15',
        'volume_confirm_boost': 1.1,  # 新增
        'volume_fail_penalty': 0.8,   # 新增
        'divergence_penalty': 0.3,    # 新增
        # --- 量能背离检查 ---
        'check_bearish_divergence': True,
        'divergence_price_period': 5,
        'divergence_threshold_cmf': -0.05,
        'divergence_threshold_mfi': 40, # 新增 (虽然在策略中是可选的，但这里包含以保持一致)
    }
    # 使用更新后的策略类
    strategy_instance = MacdRsiKdjBollEnhancedStrategy(params=strategy_params)

    # 3. 定义策略所需的时间周期
    timeframes: List[str] = strategy_instance.timeframes # 从策略实例获取
    stock = await stock_basic_dao.get_stock_by_code(stock_code)
    # 4. 准备策略数据帧
    print(f"[{stock}] 正在准备策略数据...")
    # 注意：确保 IndicatorService.prepare_strategy_dataframe 能够根据
    # strategy_instance.get_required_columns() 返回所有必需的列
    strategy_df: Optional[pd.DataFrame] = await indicator_service.prepare_strategy_dataframe(
        stock_code=stock_code,
        timeframes=timeframes,
        strategy_params=strategy_params, # 函数定义需要这个参数
        limit_per_tf=1200
    )

    # 5. 检查数据准备是否成功
    if strategy_df is None or strategy_df.empty:
        print(f"[{stock}] 策略数据准备失败或为空，无法生成评分。")
        logger.error(f"[{stock}] Strategy data preparation failed or returned empty.")
        return

    print(f"[{stock}] 策略数据准备完成，形状: {strategy_df.shape}")
    # print(f"[{stock_code}] 数据帧包含列: {strategy_df.columns.tolist()}")
    # print(f"[{stock_code}] 数据帧尾部样本:\n{strategy_df.tail()}")

    # 6. 生成评分
    print(f"[{stock}] 正在生成策略评分 (0-100)...")
    try:
        # 调用 generate_signals，但结果现在是分数
        scores: pd.Series = strategy_instance.run(strategy_df) # 使用 run 方法更标准
        print(f"[{stock}] 策略评分生成完成。")

        # 7. 查看或使用评分结果 (加入时区转换)
        if scores is not None and not scores.empty:
            scores_display = scores.copy() # 创建副本以进行显示转换
            if local_tz and isinstance(scores_display.index, pd.DatetimeIndex):
                try:
                    if scores_display.index.tz is None:
                        # 如果是幼稚类型，先假定为 UTC 再转本地
                        print(f"[{stock}] 评分索引为时区幼稚类型，假定为 UTC 并转换为 {local_tz_name}...")
                        scores_display.index = scores_display.index.tz_localize('UTC').tz_convert(local_tz)
                    else:
                        # 如果是感知类型，直接转本地
                        print(f"[{stock}] 评分索引时区为 {scores_display.index.tz}，转换为 {local_tz_name}...")
                        scores_display.index = scores_display.index.tz_convert(local_tz)
                except Exception as convert_e:
                    print(f"[{stock}] 转换评分索引时区时出错: {convert_e}")

            print(f"\n[{stock}] 最新的评分 (最后10条，时间：{local_tz_name}):")
            print(scores_display.tail(10)) # 打印转换后的副本

            print("\n评分统计描述:")
            print(scores.describe()) # 统计描述使用原始数据

            nan_count = scores.isna().sum()
            if nan_count > 0:
                print(f"\n警告: 生成的评分中包含 {nan_count} 个 NaN 值。")
        else:
            print(f"\n[{stock}] 未能获取有效的评分结果 (scores is None or empty)。")
            print(f"[{stock}] scores 对象: {scores}")
        # 查看中间数据 (加入时区转换)
        intermediate_data = strategy_instance.get_intermediate_data()
        if intermediate_data is not None:
            intermediate_data_display = intermediate_data.copy() # 创建副本
            if local_tz and isinstance(intermediate_data_display.index, pd.DatetimeIndex):
                 try:
                    if intermediate_data_display.index.tz is None:
                        # 幼稚类型处理
                        # print(f"[{stock_code}] 中间数据索引为时区幼稚类型，假定为 UTC 并转换为 {local_tz_name}...")
                        intermediate_data_display.index = intermediate_data_display.index.tz_localize('UTC').tz_convert(local_tz)
                    else:
                        # 感知类型处理
                        # print(f"[{stock_code}] 中间数据索引时区为 {intermediate_data_display.index.tz}，转换为 {local_tz_name}...")
                        intermediate_data_display.index = intermediate_data_display.index.tz_convert(local_tz)
                 except Exception as convert_e:
                    print(f"[{stock}] 转换中间数据索引时区时出错: {convert_e}")

            # print(f"\n中间计算数据 (最后5行，时间：{local_tz_name}):")
            # with pd.option_context('display.max_rows', 5, 'display.max_columns', None):
            #     print(intermediate_data_display.tail()) # 打印转换后的副本
        else:
             print("\n无法获取中间计算数据。")

    except ValueError as ve:
        # 特别处理列缺失错误 (如果 run 方法中没有完全抑制)
        print(f"[{stock}] 生成评分时可能发生错误 (例如缺少列): {ve}")
        logger.error(f"[{stock}] ValueError during score generation: {ve}", exc_info=True)
        # print("可用列:", strategy_df.columns.tolist()) # 打印可用列帮助调试
    except Exception as e:
        print(f"[{stock}] 生成评分时发生未知错误: {e}")
        logger.error(f"[{stock}] Unexpected error during score generation: {e}", exc_info=True)
        # import traceback # 在 manage.py command 中，Django 会处理异常打印
        # traceback.print_exc()


# --- Django Management Command 类 ---
class Command(BaseCommand):
    help = '测试指定股票代码的策略评分生成 (0-100分)'

    def add_arguments(self, parser):
        parser.add_argument('stock_code', type=str, help='要测试的股票代码 (例如: 000001)')

    def handle(self, *args, **options):
        stock_code_to_test = options['stock_code']

        self.stdout.write(self.style.SUCCESS(f'开始测试策略评分 for {stock_code_to_test}...'))

        try:
            # 运行更新后的异步测试函数
            asyncio.run(test_strategy_scores(stock_code=stock_code_to_test))
            self.stdout.write(self.style.SUCCESS(f'策略评分测试完成 for {stock_code_to_test}.'))
        except Exception as e:
            # Django Command 会自动处理未捕获的异常并打印堆栈，但这里可以加个日志
            logger.error(f"Error during command execution for {stock_code_to_test}: {e}", exc_info=True)
            self.stderr.write(self.style.ERROR(f'测试过程中发生错误: {e}'))
            # 不需要手动打印 traceback，manage.py 会处理

