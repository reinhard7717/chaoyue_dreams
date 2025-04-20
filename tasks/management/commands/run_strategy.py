# strategies/management/commands/run_strategy.py

import logging
from datetime import datetime
from django.core.management.base import BaseCommand, CommandError
import pandas as pd # 可选：用于可能的结果展示

# 导入你的策略类
from strategies.trend_following_strategy import TrendFollowingStrategy
from strategies.trend_reversal_strategy import TrendReversalStrategy
from strategies.t_plus_0_strategy import TPlus0Strategy
# from strategies import strategy_utils # utils 被策略内部使用，通常无需直接在此导入

# --- 数据加载逻辑的占位符 ---
# !!! 你需要用你自己的实际实现替换这部分代码 !!!
# 它应该根据 required_columns 获取 OHLCV 和预计算的指标
def load_strategy_data(stock_code: str, start_date: str | None, end_date: str | None, required_columns: list) -> pd.DataFrame:
    """
    占位函数：用于加载策略所需的数据 (OHLCV + 指标)。
    请将其替换为你的实际数据获取逻辑 (例如, 使用 IndicatorService)。
    :param stock_code: 股票代码
    :param start_date: 开始日期 (YYYY-MM-DD)
    :param end_date: 结束日期 (YYYY-MM-DD)
    :param required_columns: 策略所需的列名列表
    :return: 包含所需数据的 Pandas DataFrame
    """
    logger = logging.getLogger(__name__)
    logger.warning("正在使用占位数据加载函数 load_strategy_data。请务必替换为实际实现！")
    # 示例：模拟加载数据
    # 在实际场景中，你会根据 stock_code, dates, required_columns 查询数据库/缓存/API。
    # 返回的数据结构需要符合策略的预期。
    # 为演示起见，这里返回一个空的 DataFrame 以避免错误。
    # 确保索引是 DatetimeIndex (如果可能)。
    # dummy_dates = pd.date_range(start=start_date or '2023-01-01', end=end_date or '2023-01-10', freq='D')
    # dummy_data = {col: pd.Series(index=dummy_dates, dtype=float) for col in required_columns}
    # return pd.DataFrame(dummy_data)
    print(f"--- 占位符：假装正在为 {stock_code} 加载数据 ---")
    print(f"--- 需要的列: {', '.join(required_columns)} ---")
    # 返回空 DataFrame 以允许命令在没有真实数据源的情况下运行
    return pd.DataFrame(columns=required_columns)
# --- 占位符结束 ---


logger = logging.getLogger(__name__) # 获取当前模块的 logger

class Command(BaseCommand):
    help = '运行选定的交易策略，针对给定的股票代码。' # 命令的帮助信息

    def add_arguments(self, parser):
        # 定义命令接收的参数
        parser.add_argument(
            'strategy_name', # 参数名
            type=str,        # 参数类型
            choices=['trend_following', 'trend_reversal', 't_plus_0'], # 可选值
            help='要运行的策略名称。' # 参数说明
        )
        parser.add_argument(
            'stock_code',
            type=str,
            help='要运行策略的股票代码 (例如, 600519.SH)。'
        )
        parser.add_argument(
            '--start-date', # 可选参数，以 '--' 开头
            type=str,
            default=None, # 默认值
            help='分析的开始日期 (格式: YYYY-MM-DD)。'
        )
        parser.add_argument(
            '--end-date',
            type=str,
            default=None, # 默认值
            help='分析的结束日期 (格式: YYYY-MM-DD)。默认为今天。'
        )
        parser.add_argument(
            '--params-file',
            type=str,
            default="strategies/indicator_parameters.json", # 参数文件的默认路径
            help='指标参数 JSON 文件的路径。'
        )
        # 如果需要，可以添加更多参数 (例如 --output-file, --save-to-db)

    def handle(self, *args, **options):
        # 命令执行的入口点
        strategy_name = options['strategy_name'] # 获取命令行传入的参数值
        stock_code = options['stock_code']
        start_date = options['start_date']
        end_date = options['end_date'] or datetime.now().strftime('%Y-%m-%d') # 如果未提供结束日期，则使用当前日期
        params_file = options['params_file']

        # 使用 self.stdout 输出信息到控制台
        self.stdout.write(f"开始为股票 '{stock_code}' 运行策略 '{strategy_name}'...")
        logger.info(f"执行命令: strategy={strategy_name}, stock={stock_code}, start={start_date}, end={end_date}, params={params_file}")

        strategy_instance = None # 初始化策略实例变量
        try:
            # 1. 实例化选定的策略
            self.stdout.write("正在初始化策略...")
            if strategy_name == 'trend_following':
                strategy_instance = TrendFollowingStrategy(params_file=params_file)
            elif strategy_name == 'trend_reversal':
                strategy_instance = TrendReversalStrategy(params_file=params_file)
            elif strategy_name == 't_plus_0':
                strategy_instance = TPlus0Strategy(params_file=params_file)
            else:
                # 这个情况理论上会被 'choices' 捕获，但为了健壮性可以保留
                raise CommandError(f"未知的策略名称: {strategy_name}")

            self.stdout.write(f"策略 '{strategy_instance.strategy_name}' 初始化完成。")

            # 2. 在加载数据前获取策略所需的列
            required_columns = strategy_instance.get_required_columns()
            if not required_columns:
                 logger.warning(f"策略 '{strategy_name}' 未指定必需列。数据加载可能不完整。")
                 # 你可以决定这是否是一个错误或只是警告

            # 3. 加载数据 (!!! 必须替换占位符 !!!)
            self.stdout.write("正在加载数据...")
            try:
                # *** 将 load_strategy_data 替换为你真实的数据加载函数 ***
                data = load_strategy_data(stock_code, start_date, end_date, required_columns)
            except Exception as e:
                 logger.error(f"为 {stock_code} 加载数据失败: {e}", exc_info=True) # 记录详细错误信息
                 raise CommandError(f"数据加载失败: {e}")

            # 检查是否成功加载到数据
            if data.empty:
                 logger.warning(f"未能为股票 '{stock_code}' 在指定时间段内加载到数据。无法生成信号。")
                 self.stdout.write(self.style.WARNING("未加载到数据，跳过信号生成。")) # 使用 Django 的样式输出警告
                 return # 如果没有数据则优雅退出

            self.stdout.write(f"数据加载完成。数据行数: {len(data)}。数据列数: {len(data.columns)}")

            # 4. 生成信号
            self.stdout.write("正在生成信号...")
            try:
                # 调用策略实例的 generate_signals 方法
                final_signals = strategy_instance.generate_signals(data)
            except Exception as e:
                 logger.error(f"为策略 {strategy_name} (股票 {stock_code}) 生成信号时出错: {e}", exc_info=True)
                 raise CommandError(f"信号生成失败: {e}")


            # 5. 输出/记录结果
            self.stdout.write("信号生成完成。详细分析请查看日志文件。")
            # 检查是否有有效的信号返回
            if final_signals is not None and not final_signals.empty:
                self.stdout.write("\n--- 最新信号 ---")
                # 打印最后 5 条信号/评分，方便快速查看
                self.stdout.write(str(final_signals.tail()))

                # 每个策略的 analyze_signals 方法已经在内部记录了详细分析
                # 你也可以选择在这里获取分析结果的 DataFrame 并打印：
                # analysis_df = strategy_instance.get_analysis_results() # 假设你在策略类中添加了这个 getter 方法
                # if analysis_df is not None and not analysis_df.empty:
                #    self.stdout.write("\n--- 分析摘要 ---")
                #    # 使用 to_string() 打印完整的 DataFrame
                #    self.stdout.write(analysis_df.to_string())
            else:
                self.stdout.write(self.style.WARNING("未能生成最终信号。"))


        except FileNotFoundError as e:
            # 捕获参数文件未找到的错误
            logger.error(f"参数文件未找到: {e}")
            raise CommandError(f"参数文件错误: {e}")
        except ImportError as e:
             # 捕获导入策略或其依赖时的错误
             logger.error(f"导入策略或依赖失败: {e}", exc_info=True)
             raise CommandError(f"导入错误: {e}")
        except ValueError as e: # 捕获策略内部的参数验证错误
             logger.error(f"参数验证错误: {e}", exc_info=True)
             raise CommandError(f"参数验证错误: {e}")
        except Exception as e:
            # 捕获其他所有未预料到的异常
            logger.error(f"发生未预料的错误: {e}", exc_info=True)
            raise CommandError(f"发生未预料的错误: {e}")

        # 命令成功结束
        self.stdout.write(self.style.SUCCESS(f"成功完成股票 '{stock_code}' 的策略 '{strategy_name}'。"))

