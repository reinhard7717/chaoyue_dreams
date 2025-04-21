# tasks/management/commands/test_strategy_signals.py

import asyncio
import pandas as pd
import numpy as np
import json
from typing import Dict, Any, Optional # 简化
from django.core.management.base import BaseCommand, CommandError
import logging
from django.utils import timezone
from decimal import Decimal
from asgiref.sync import sync_to_async
from django.core.cache import cache

from dao_manager.daos.stock_basic_dao import StockBasicDAO
# from dao_manager.daos.stock_realtime_dao import StockRealtimeDAO # 保留以备T+0策略未来可能的需求
from stock_models.stock_analytics import StockScoreAnalysis # 导入分析结果模型
from stock_models.stock_basic import StockInfo
# from stock_models.stock_realtime import StockRealtimeData # 可能不再需要
# from utils.cache_manager import CacheManager # 可能不再直接需要，由 timezone 处理
from django.db import models # 用于 aggregate

# --- 导入时区处理库 (如果保存数据库需要) ---
try:
    import tzlocal
    from zoneinfo import ZoneInfo
except ImportError:
    tzlocal = None
    ZoneInfo = None
    print("警告：无法导入 'tzlocal' 或 'zoneinfo'。数据库保存可能受时区影响。")

# --- 导入项目模块 ---
from services.indicator_services import IndicatorService # 依赖这个服务来准备数据
# --- 导入需要运行的策略 ---
from strategies.trend_following_strategy import TrendFollowingStrategy
from strategies.trend_reversal_strategy import TrendReversalStrategy
from strategies.t_plus_0_strategy import TPlus0Strategy


logger = logging.getLogger(__name__)

# --- 辅助函数：清理潜在的无效数值 (如果保存数据库需要) ---
def clean_value(value, default=None):
    """
    将 NaN, inf, -inf, NaT, None 等无效值转换为指定的默认值 (通常是 None)。
    处理 numpy 类型和 Decimal 的特殊情况。
    """
    if pd.isna(value):
        return default
    if isinstance(value, float) and not np.isfinite(value):
        return default
    if isinstance(value, Decimal) and (value.is_nan() or value.is_infinite()):
        return default
    # 检查 numpy 数字类型 (包括 int, float 等)
    if isinstance(value, np.number) and not np.isfinite(value):
        return default
    # 处理 numpy 数组元素 (例如通过 .item() 获取)
    if hasattr(value, 'item'):
        try:
            item_value = value.item()
            # 再次检查转换后的 Python 类型
            if isinstance(item_value, float) and not np.isfinite(item_value):
                return default
            # 检查是否是pandas的Timestamp NaT 或 Python 的 None
            if pd.isna(item_value):
                return default
            return item_value
        except (ValueError, TypeError): # .item() 可能对某些类型失败
            return default
    return value


# --- 【移除】旧策略相关的辅助函数 ---
# 移除 get_find_peaks_params, detect_divergence, detect_kline_patterns, analyze_score_trend

class Command(BaseCommand):
    help = '测试指定的A股策略并生成信号/评分 (TrendFollowing, TrendReversal, TPlus0)'

    def add_arguments(self, parser):
        parser.add_argument('--stock_code', type=str, required=True, help='指定要测试的股票代码')
        parser.add_argument('--params_file', type=str, default="strategies/indicator_parameters.json", help='策略参数 JSON 文件路径')
        parser.add_argument('--save_analysis', action='store_true', help='将分析结果保存到 StockScoreAnalysis 模型')
        # 可以添加 --tail 参数来只显示最后 N 条结果
        parser.add_argument('--tail', type=int, default=5, help='显示最后 N 条计算结果')


    async def handle_async(self, *args, **options):
        stock_code = options['stock_code']
        params_file = options['params_file']
        save_analysis = options['save_analysis']
        tail_n = options['tail']

        stock_basic_dao = StockBasicDAO()
        stock_info = await stock_basic_dao.get_stock_by_code(stock_code)
        if not stock_info:
            raise CommandError(f"找不到股票代码: {stock_code}")

        # 1. 实例化服务和需要运行的策略
        indicator_service = IndicatorService()
        # --- 修改字典值的类型提示为 Any ---
        strategies_to_run: Dict[str, Any] = {}
        try:
            # --- 按需实例化策略 ---
            strategies_to_run['trend_following'] = TrendFollowingStrategy(params_file=params_file)
            strategies_to_run['trend_reversal'] = TrendReversalStrategy(params_file=params_file)
            strategies_to_run['t_plus_0'] = TPlus0Strategy(params_file=params_file)
            # 可以在这里添加或移除其他策略
            self.stdout.write(f"将要运行的策略: {', '.join(s.strategy_name for s in strategies_to_run.values())}")
        except (FileNotFoundError, ValueError, ImportError, KeyError) as e: # 捕捉更多可能的初始化错误
            raise CommandError(f"初始化策略时出错: {e}")

        # 2. 准备数据 (假设基于 params_file 准备所有策略所需数据)
        # 注意: 这假设 indicator_service.prepare_strategy_dataframe 能够根据
        # 单个 params_file 文件准备所有已实例化策略所需的数据列。
        # 如果策略的数据需求差异很大（例如需要不同的API源或预处理），可能需要调整数据准备逻辑。
        self.stdout.write(f"[{stock_info}] 正在使用 IndicatorService 准备数据 (基于 {params_file})...")
        try:
            # 传递参数文件路径，服务内部应解析所有需要的指标参数
            data_df = await indicator_service.prepare_strategy_dataframe(
                stock_code=stock_code,
                params_file=params_file # 传递参数文件路径
            )
        except Exception as prep_err:
             raise CommandError(f"[{stock_info}] 调用 prepare_strategy_dataframe 准备数据时出错: {prep_err}")

        if data_df is None or data_df.empty:
            self.stdout.write(self.style.WARNING(f"[{stock_info}] 未能准备足够的数据 (prepare_strategy_dataframe 返回空)。"))
            return

        self.stdout.write(f"[{stock_info}] 数据准备完成，Shape: {data_df.shape}。开始生成策略信号...")

        # 3. 循环运行每个策略
        results = {}
        # --- 修改 strategy 的类型提示为 Any ---
        for name, strategy in strategies_to_run.items():
            self.stdout.write(f"\n--- 运行策略: {strategy.strategy_name} ---")
            try:
                # 生成信号/评分 (传递数据的副本以防策略修改原始数据)
                final_output = strategy.generate_signals(data_df.copy(), stock_code)

                if final_output is None or final_output.empty:
                    self.stdout.write(self.style.WARNING(f"[{stock_info}] 策略 '{strategy.strategy_name}' 未生成有效输出。"))
                    continue

                intermediate_df = strategy.get_intermediate_data()
                results[name] = {'output': final_output, 'intermediate': intermediate_df, 'strategy_obj': strategy}

                # 打印最新结果
                self.stdout.write(self.style.SUCCESS(f"策略 {strategy.strategy_name} 计算完成 for {stock_info}"))

                if intermediate_df is not None and not intermediate_df.empty:
                    # self.stdout.write(f"--- 最新 {tail_n} 条中间结果 for {strategy.strategy_name} ---")
                    # 尝试打印常见的或重要的列，否则打印所有列
                    common_cols = ['final_signal', 'final_score', 't0_signal', 'base_score_volume_adjusted', 'reversal_confirmation_signal', 'alignment_signal', 'long_term_context']
                    cols_to_print = [col for col in common_cols if col in intermediate_df.columns]

                    # 如果没有通用列，尝试添加最终输出列（如果它是Series且不在intermediate中）
                    primary_col_name = None
                    if isinstance(final_output, pd.Series) and final_output.name:
                         primary_col_name = final_output.name
                    elif 'final_signal' in intermediate_df.columns:
                         primary_col_name = 'final_signal'
                    elif 't0_signal' in intermediate_df.columns:
                         primary_col_name = 't0_signal'
                    elif 'final_score' in intermediate_df.columns:
                         primary_col_name = 'final_score'
                    elif 'score' in intermediate_df.columns: # 检查模型主字段名
                         primary_col_name = 'score'

                    if not cols_to_print and primary_col_name and primary_col_name not in intermediate_df.columns:
                         # 临时将 final_output 添加到要打印的副本中
                         temp_print_df = intermediate_df.tail(tail_n).copy()
                         temp_print_df[primary_col_name] = final_output.tail(tail_n)
                         cols_to_print = [primary_col_name] + intermediate_df.columns.tolist()
                         # 确保只包含存在的列
                         valid_cols_to_print = [col for col in cols_to_print if col in temp_print_df.columns]
                         if valid_cols_to_print:
                              print(temp_print_df[valid_cols_to_print].to_string())
                         else:
                              print(temp_print_df.to_string()) # Fallback
                         cols_to_print = None # 防止下面再次打印
                    elif not cols_to_print and primary_col_name:
                         cols_to_print = [primary_col_name] # 至少打印主输出列

                    if cols_to_print: # 如果找到了要打印的列
                         valid_cols_to_print = [col for col in cols_to_print if col in intermediate_df.columns]
                         if valid_cols_to_print:
                              print(intermediate_df[valid_cols_to_print].tail(tail_n).to_string())
                         else:
                              print(intermediate_df.tail(tail_n).to_string()) # Fallback
                    elif cols_to_print is None: # 表示上面已经打印过了
                         pass
                    else: # cols_to_print 是空列表，但上面没打印
                         self.stdout.write("无法找到特定列，显示所有中间列:")
                         print(intermediate_df.tail(tail_n).to_string())

                else: # 只有最终输出 Series 可用
                    # self.stdout.write(f"--- 最新 {tail_n} 条输出 for {strategy.strategy_name} ---")
                    # print(final_output.tail(tail_n).to_string())
                    pass
                analysis_results = strategy.analyze_signals
                # print(f"analysis_results: {analysis_results}")

                # --- 可选: 保存分析结果到数据库 ---
                # 确保 intermediate_df 存在才尝试保存
                if save_analysis:
                    if intermediate_df is not None and not intermediate_df.empty:
                        self.stdout.write(f"[{stock_info}] 正在为策略 '{strategy.strategy_name}' 保存分析结果...")
                        try:
                            # 传递中间结果DataFrame, 股票信息, 和策略对象本身
                            await self.save_analysis_results(intermediate_df, stock_info, strategy) # strategy 是 Any 类型
                            self.stdout.write(self.style.SUCCESS(f"[{stock_info}] 策略 '{strategy.strategy_name}' 的分析结果已保存。"))
                        except CommandError as cmd_err: # 特别捕捉模型字段缺失错误
                             raise cmd_err # 重新抛出，停止执行
                        except Exception as db_err:
                            self.stdout.write(self.style.ERROR(f"[{stock_info}] 保存策略 '{strategy.strategy_name}' 的分析结果时出错: {db_err}"))
                    else:
                        self.stdout.write(self.style.WARNING(f"[{stock_info}] 策略 '{strategy.strategy_name}' 没有可用的中间数据 (intermediate_data)，无法保存。"))

            except Exception as sig_err:
                self.stdout.write(self.style.ERROR(f"[{stock_info}] 执行策略 '{strategy.strategy_name}' 时出错: {sig_err}"))
                logger.error(f"[{stock_info}] 执行策略 '{strategy.strategy_name}' 时出错: {sig_err}", exc_info=True)
                # 选择是否继续下一个策略，这里我们选择继续
                continue

    # --- 修改 save_analysis_results 中 strategy 的类型提示为 Any ---
    async def save_analysis_results(self, analysis_df: pd.DataFrame, stock_info: StockInfo, strategy: Any):
        """
        将指定策略生成的分析结果 DataFrame 保存到 StockScoreAnalysis 模型。
        从 analysis_df 中读取数据。
        **重要提示:** 此函数假定 StockScoreAnalysis 模型已包含 `strategy_name` 字段。
                      请确保已执行数据库迁移添加此字段。
        """
        strategy_name = strategy.strategy_name
        params = strategy.params # 从策略对象获取参数

        # --- 确定用于存储的时间级别 ---
        # 优先使用策略对象上的 focus_timeframe 属性
        analysis_tf = getattr(strategy, 'focus_timeframe', None)
        if not analysis_tf:
            # 如果策略对象没有 focus_timeframe，尝试从参数中推断
            strategy_params_key = f"{strategy_name.lower().replace('strategy', '').rstrip('_')}_params" # e.g., 'trend_following_params'
            analysis_tf = params.get(strategy_params_key, {}).get('focus_timeframe')

            if not analysis_tf:
                # 最后尝试从 base_scoring 获取第一个时间框架
                analysis_tf = params.get('base_scoring', {}).get('timeframes', ['unknown'])[0]
                if analysis_tf == 'unknown':
                    logger.error(f"[{stock_info.stock_code} - {strategy_name}] 无法确定分析时间框架 (未找到 focus_timeframe)。跳过保存。")
                    return

            logger.warning(f"[{stock_info.stock_code} - {strategy_name}] 策略对象缺少 'focus_timeframe' 属性。已推断使用时间框架: '{analysis_tf}'。")

        self.stdout.write(f"[{stock_info.stock_code} - {strategy_name}] 使用时间级别 '{analysis_tf}' 保存分析结果。")

        # --- 数据准备 ---
        analysis_df_filled = analysis_df.copy()
        analysis_df_filled.replace({np.nan: None, pd.NaT: None}, inplace=True)
        # 将所有可能数值列转为 float (因为模型字段多改为 FloatField), 保持整数信号列为整数
        int_signal_cols = [
            'alignment_signal', 'long_term_context', 'strong_reversal_confirmation',
            'macd_hist_divergence', 'rsi_divergence', 'mfi_divergence', 'obv_divergence',
            'kline_pattern', 'rsi_obos_reversal', 'stoch_obos_reversal',
            'cci_obos_reversal', 'bb_reversal', 'volume_spike', 't0_signal'
        ]
        for col in analysis_df_filled.columns:
            # 检查是否是数值类型且不是明确的整数信号列
            if pd.api.types.is_numeric_dtype(analysis_df_filled[col]) and col not in int_signal_cols:
                try:
                    # 尝试转换为 float，但如果列中只包含整数，保持为整数以避免不必要的 .0
                    if analysis_df_filled[col].dropna().apply(float.is_integer).all():
                         # 如果所有非 NaN 值都是整数形式的浮点数，可以尝试转为 Int64
                         # 注意：这要求 Pandas >= 1.0 且无 NaN 值才能直接转 int
                         # 为了兼容性和处理 NaN，通常还是转 float 更安全，数据库层面处理类型
                         analysis_df_filled[col] = analysis_df_filled[col].astype(float)
                    else:
                         analysis_df_filled[col] = analysis_df_filled[col].astype(float)
                except (ValueError, TypeError, pd.errors.IntCastingNaNError):
                    logger.debug(f"无法将列 {col} 转换为 float/int，跳过类型转换。")
                    pass
            # 对于明确的整数信号列，尝试转为可空的整数类型
            elif col in int_signal_cols:
                 try:
                     # 使用 Int64 (大写I) 支持 NaN 的整数类型
                     analysis_df_filled[col] = analysis_df_filled[col].astype('Int64')
                 except (ValueError, TypeError):
                      logger.debug(f"无法将整数信号列 {col} 转换为 Int64，跳过类型转换。")
                      pass


        # --- 获取数据库中该股票、策略和时间级别的最新时间戳 ---
        target_tz = timezone.get_default_timezone() # 获取 Django 默认时区
        try:
            # 注意Django ORM 的 filter 对 DateTimeField 时区的处理
            # 直接比较 aware datetime 对象是推荐的方式
            latest_timestamp_in_db_obj = await StockScoreAnalysis.objects.filter(
                stock=stock_info,
                # timestamp__tz=target_tz, # 不建议这样过滤，直接比较 aware datetime
                time_level=analysis_tf,
                strategy_name=strategy_name
            ).aggregate(latest_ts=models.Max('timestamp'))
            latest_timestamp_in_db = latest_timestamp_in_db_obj['latest_ts']

        except Exception as e:
             if 'strategy_name' in str(e).lower():
                  logger.error(f"[{stock_info.stock_code} - {strategy_name}] 查询数据库时出错: 'StockScoreAnalysis' 模型似乎缺少 'strategy_name' 字段。请添加该字段并执行迁移。错误: {e}")
                  raise CommandError(f"Database model 'StockScoreAnalysis' is missing the required 'strategy_name' field. Please add it via migration.") from e
             else:
                  logger.error(f"[{stock_info.stock_code} - {strategy_name}] 查询最新时间戳时出错: {e}")
                  latest_timestamp_in_db = None # 继续尝试插入，但不跳过

        # --- 时区处理 for latest_timestamp_in_db ---
        if latest_timestamp_in_db and timezone.is_naive(latest_timestamp_in_db):
            # 如果数据库返回的是 naive 时间戳，假设它是 UTC 或 Django 的 TIME_ZONE
            # 这里假设它是 Django 的 TIME_ZONE
            latest_timestamp_in_db = timezone.make_aware(latest_timestamp_in_db, target_tz)
            logger.debug(f"Database returned naive timestamp, assuming target timezone {target_tz}")
        elif latest_timestamp_in_db:
             latest_timestamp_in_db = latest_timestamp_in_db.astimezone(target_tz) # 确保与 target_tz 一致

        saved_count = 0
        skipped_count = 0
        error_count = 0

        # --- 确定主要得分/信号列 ---
        primary_col = None
        # 优先使用 'final_signal' 或 'final_score' 或 'score' (与模型字段匹配)
        if 'score' in analysis_df_filled.columns:
             primary_col = 'score'
        elif 'final_signal' in analysis_df_filled.columns:
            primary_col = 'final_signal'
        elif 'final_score' in analysis_df_filled.columns:
            primary_col = 'final_score'
        elif 't0_signal' in analysis_df_filled.columns and isinstance(strategy, TPlus0Strategy):
            primary_col = 't0_signal'

        if primary_col is None:
            logger.warning(f"[{stock_info} - {strategy_name}] 未能在中间数据中识别主要得分/信号列 (如 'score', 'final_signal', 't0_signal')。将尝试保存 'score' 为 None。")

        # --- 迭代并保存数据 ---
        async for timestamp, row in sync_to_async(analysis_df_filled.iterrows, thread_sensitive=False)():
            # 确保 timestamp 是 timezone-aware 且与数据库时区一致
            if pd.isna(timestamp): # 跳过无效时间戳的行
                logger.warning(f"[{stock_info.stock_code} - {strategy_name}] 跳过无效时间戳的行数据。")
                continue

            current_ts_aware = None
            try:
                if timezone.is_naive(timestamp):
                     current_ts_aware = timezone.make_aware(timestamp, target_tz)
                else:
                     current_ts_aware = timestamp.astimezone(target_tz)
            except Exception as tz_err:
                logger.error(f"[{stock_info.stock_code} - {strategy_name}] 处理时间戳 {timestamp} 时区时出错: {tz_err}. 跳过此行。")
                error_count += 1
                continue


            # 跳过数据库中已有的或更早的数据
            if latest_timestamp_in_db and current_ts_aware <= latest_timestamp_in_db:
                 skipped_count += 1
                 continue

            # --- 构建要保存到 StockScoreAnalysis 的数据 ---
            analysis_data = {
                'stock': stock_info,
                'timestamp': current_ts_aware, # 使用重命名后的字段
                'time_level': analysis_tf,
                'strategy_name': strategy_name,
                # --- 主要得分/信号 (填充到 score 字段) ---
                'score': clean_value(row.get(primary_col)), # 使用 score 字段
                # --- 保存与新模型字段对应的中间结果 ---
                'base_score_raw': clean_value(row.get('base_score_raw')),
                'base_score_volume_adjusted': clean_value(row.get('base_score_volume_adjusted')),
                'alignment_signal': clean_value(row.get('alignment_signal')),
                'long_term_context': clean_value(row.get('long_term_context')),
                'ema_score_5': clean_value(row.get('ema_score_5')),
                'ema_score_13': clean_value(row.get('ema_score_13')),
                'ema_score_21': clean_value(row.get('ema_score_21')),
                'ema_score_55': clean_value(row.get('ema_score_55')),
                'ema_score_233': clean_value(row.get('ema_score_233')),
                'ema_strength_13_55': clean_value(row.get('ema_strength_13_55')),
                'score_momentum': clean_value(row.get('score_momentum')),
                'score_volatility': clean_value(row.get('score_volatility')),
                'reversal_confirmation_signal': clean_value(row.get('reversal_confirmation_signal')),
                'strong_reversal_confirmation': clean_value(row.get('strong_reversal_confirmation')),
                'macd_hist_divergence': clean_value(row.get('macd_hist_divergence')),
                'rsi_divergence': clean_value(row.get('rsi_divergence')),
                'mfi_divergence': clean_value(row.get('mfi_divergence')),
                'obv_divergence': clean_value(row.get('obv_divergence')),
                'kline_pattern': clean_value(row.get('kline_pattern')),
                'rsi_obos_reversal': clean_value(row.get('rsi_obos_reversal')),
                'stoch_obos_reversal': clean_value(row.get('stoch_obos_reversal')),
                'cci_obos_reversal': clean_value(row.get('cci_obos_reversal')),
                'bb_reversal': clean_value(row.get('bb_reversal')),
                'volume_spike': clean_value(row.get('volume_spike')),
                't0_signal': clean_value(row.get('t0_signal')),
                # T+0 策略的 deviation 可能存储在 price_vwap_deviation 字段
                'price_vwap_deviation': clean_value(row.get('deviation')),
                # 可选的快照字段
                'close_price': clean_value(row.get(f'close_{analysis_tf}')),
                'vwap': clean_value(row.get(f'vwap_{analysis_tf}') or row.get('vwap')),
                # --- 元数据 ---
                'params_snapshot': json.dumps({ # 使用 params_snapshot 字段
                    'strategy_name': strategy_name,
                    'analysis_tf': analysis_tf,
                    'focus_tf': getattr(strategy, 'focus_timeframe', 'N/A'),
                    # 可以添加更多参数信息
                    # 'params': params # 可能过大，选择性添加
                 }, ensure_ascii=False, default=str),
            }

            # --- 清理无效的键并准备 defaults ---
            key_fields = ['stock', 'timestamp', 'time_level', 'strategy_name']
            if any(k not in analysis_data or analysis_data[k] is None for k in key_fields):
                 logger.error(f"[{stock_info.stock_code} - {strategy_name}] 缺少用于 update_or_create 的关键识别字段或其值为 None: {key_fields}")
                 error_count += 1
                 continue

            model_fields = [f.name for f in StockScoreAnalysis._meta.get_fields()]
            defaults_cleaned = {
                k: v for k, v in analysis_data.items()
                if k not in key_fields and k in model_fields # 确保字段存在于模型中
            }
            # 显式设置 updated_at (即使模型中有 auto_now=True, 在 aupdate_or_create 中指定可确保更新)
            # defaults_cleaned['updated_at'] = timezone.now() # 如果 auto_now=True 正常工作，则不需要

            # --- 执行数据库操作 ---
            try:
                 # 使用 aupdate_or_create
                 obj, created = await StockScoreAnalysis.objects.aupdate_or_create(
                     stock=analysis_data['stock'],
                     timestamp=analysis_data['timestamp'],
                     time_level=analysis_data['time_level'],
                     strategy_name=analysis_data['strategy_name'],
                     defaults=defaults_cleaned
                 )
                 saved_count += 1
            except Exception as e:
                 if 'strategy_name' in str(e).lower() and "Unknown field" in str(e):
                     logger.error(f"[{stock_info.stock_code} - {strategy_name}] 保存数据时出错: 'StockScoreAnalysis' 模型似乎缺少 'strategy_name' 字段。请添加该字段并执行迁移。错误: {e}", exc_info=False)
                     raise CommandError(f"Database model 'StockScoreAnalysis' is missing the required 'strategy_name' field. Please add it via migration.") from e
                 elif 'constraint' in str(e).lower() and 'unique' in str(e).lower():
                      logger.warning(f"[{stock_info.stock_code} - {strategy_name}] 保存时间戳 {analysis_data['timestamp']} 时可能遇到唯一约束冲突 (可能由并发写入引起): {e}", exc_info=False)
                      error_count += 1 # 计为错误
                 else:
                     logger.error(f"[{stock_info.stock_code} - {strategy_name}] 保存时间戳 {analysis_data['timestamp']} 的分析数据时出错: {e}", exc_info=False)
                     error_count += 1


        self.stdout.write(f"[{stock_info} - {strategy_name}] 数据库保存完成: 新增/更新 {saved_count} 条, 跳过 {skipped_count} 条, 失败 {error_count} 条。")


    def handle(self, *args, **options):
        # 运行异步处理
        asyncio.run(self.handle_async(*args, **options))

