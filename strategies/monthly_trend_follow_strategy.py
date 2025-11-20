# 文件: strategies/monthly_trend_follow_strategy.py
import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from utils.cache_manager import CacheManager
from services.indicator_services import IndicatorService
from dao_manager.tushare_daos.stock_basic_info_dao import StockBasicInfoDao

logger = logging.getLogger(__name__)

class MonthlyTrendFollowStrategy:
    """
    月线趋势跟踪策略 (V6.1 - 验证版)
    - 增加最终验证日志，清晰展示历史信号与当日信号的区别。
    - 修复报告生成中的格式问题。
    - 策略逻辑已于V6.0版修复并确认正确。
    """
    def __init__(self):
        """
        构造函数，初始化服务和事件循环。
        """
        cache_manager = CacheManager()
        self.stock_basic_dao = StockBasicInfoDao(cache_manager)
        self.indicator_service = IndicatorService(cache_manager)
        self.favorite_stock_set = None
        self.params = None
        try:
            self.loop = asyncio.get_running_loop()
        except RuntimeError:
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
    async def apply_strategy(self, df: pd.DataFrame, params: dict) -> pd.DataFrame:
        """
        【核心策略应用函数 V5.0 - 终极修复版】
        - 确保所有信号都通过返回Series并使用.loc赋值，彻底解决信号覆盖问题。
        """
        if df is None or df.empty:
            logger.warning("输入DataFrame为空，无法应用策略。")
            return pd.DataFrame()
        original_len = len(df) # 记录原始数据长度用于日志
        if start_date_str:
            try:
                # 将日期字符串转换为pandas可比较的日期对象
                start_date = pd.to_datetime(start_date_str).date()
                # 筛选出索引日期大于等于起始日期的数据，并重新赋值给df
                # 使用 .copy() 避免后续操作出现 SettingWithCopyWarning
                df = df[df.index.date >= start_date].copy()
                # print(f"调试信息 (月线策略): 已应用起始日期 {start_date_str}。策略计算的数据从 {original_len} 行过滤至 {len(df)} 行。")
            except (ValueError, TypeError) as e:
                # 如果日期格式错误，则记录错误并继续处理全部数据
                logger.error(f"无效的起始日期格式: '{start_date_str}'。错误: {e}。将处理全部历史记录。")
                print(f"调试信息 (月线策略): 起始日期 '{start_date_str}' 格式无效，将计算全部历史数据。")
        self.params = params
        # --- 步骤 1-5: 核心信号链计算 ---
        df.loc[:, 'signal_monthly_accumulation'] = self._check_monthly_accumulation(df, self.params.get('monthly_accumulation_params', {}))
        df.loc[:, 'signal_monthly_breakout'] = self._check_monthly_breakout(df, self.params.get('monthly_breakout_params', {}))
        df.loc[:, 'washout_score'] = self._calculate_washout_score(df, self.params.get('final_washout_params', {}))
        washout_signal_threshold = self.params.get('final_washout_params', {}).get('score_threshold', 1)
        df.loc[:, 'signal_final_washout'] = df['washout_score'] >= washout_signal_threshold
        # ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
        # 实施方案三，放宽吸筹前提条件
        # 原逻辑: was_accumulating = df['signal_monthly_accumulation'].shift(1).fillna(False)
        # 新逻辑: 不再要求紧邻的上一个月必须吸筹，而是检查过去约3个月（60个交易日）内是否出现过吸筹信号。
        # 这种方式可以捕捉到“吸筹后盘整一两个月再突破”的股票，显著增加信号数量。
        recent_accumulation_window = 60 # 定义“近期”的时间窗口为60个交易日（约3个月）
        had_recent_accumulation = df['signal_monthly_accumulation'].rolling(window=recent_accumulation_window, min_periods=1).sum().shift(1) > 0
        was_accumulating = had_recent_accumulation.fillna(False)
        # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲
        is_breakout_month = was_accumulating & (df['signal_monthly_breakout'] | df['signal_final_washout'])
        df.loc[:, 'signal_breakout_initiation'] = (is_breakout_month) & (is_breakout_month.shift(1) == False)
        df.loc[:, 'signal_ma_rejection'] = self._check_ma_rejection_signal(df, self.params.get('ma_rejection_filter_params', {}))
        df.loc[:, 'signal_box_rejection'] = self._check_box_rejection_signal(df, self.params.get('box_rejection_filter_params', {}))
        is_rejection_day = (df['signal_ma_rejection'] == -1) | (df['signal_box_rejection'] == -1)
        breakout_event_group = df['signal_breakout_initiation'].cumsum()
        rejections_in_group_so_far = is_rejection_day.groupby(breakout_event_group).cumsum()
        has_no_rejection_yet = (rejections_in_group_so_far == 0)
        df.loc[:, 'signal_breakout_trigger'] = is_breakout_month & has_no_rejection_yet
        # --- 步骤 6: 打印调试信息 ---
        # print("\n---【策略逻辑链调试】---")
        # print(f"【步骤1】月线吸筹信号总数: {df['signal_monthly_accumulation'].sum()}")
        # ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
        # 更新调试信息的文本以匹配新的逻辑
        # print(f"【步骤1.5】'近期有吸筹'(was_accumulating)信号总数: {was_accumulating.sum()}")
        # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲
        # print(f"【步骤2】月线突破信号总数: {df['signal_monthly_breakout'].sum()}")
        # print(f"【步骤3】最后洗盘信号总数: {df['signal_final_washout'].sum()}")
        # print(f"【步骤4】'初始突破条件'(is_breakout_month)信号总数: {is_breakout_month.sum()}")
        # print(f"【步骤4.5】'突破启动日'(signal_breakout_initiation)信号总数: {df['signal_breakout_initiation'].sum()}")
        # print(f"【步骤5】'拒绝信号日'总数: {is_rejection_day.sum()}")
        # print(f"【步骤6】最终'突破观察'(signal_breakout_trigger)信号总数: {df['signal_breakout_trigger'].sum()}")
        # --- 步骤 7: 基于干净的信号计算买入点 ---
        df.loc[:, 'signal_pullback_entry'] = self._find_pullback_entry_signal(df, self.params.get('pullback_entry_params', {}))
        df.loc[:, 'signal_continuation_entry'] = self._find_continuation_entry_signal(df, self.params.get('continuation_entry_params', {}))
        # --- 步骤 8: 计算止盈信号 (采用新的安全模式) ---
        df.loc[:, 'take_profit_signal'] = self.apply_take_profit_rules(df, self.params.get('take_profit_params', {}))
        # logger.info(f"策略应用完成。")
        return df
    def _score_and_generate_report(self, signal_row: pd.Series, stock_code: str, params: dict) -> Dict:
        """
        【V6.2 报告重构版】对最新信号进行评分并生成报告。
        - 修复了止盈信号与吸筹信号同时出现时，报告文本逻辑矛盾的问题。
        - 建立了信号优先级制度，确保报告的逻辑自洽性。
        """
        strategy_params = params.get('strategy_params', {}).get('monthly_trend_follow', {})
        signal_type = "无明确信号"
        total_score = 0
        main_analysis_parts = []
        context_analysis_parts = []
        acc_params = strategy_params.get('monthly_accumulation_params', {})
        brk_params = strategy_params.get('monthly_breakout_params', {})
        lookback_months = brk_params.get('lookback_months', 6)
        period_high_col = f'consolidation_period_high_M_{lookback_months}'
        trix_col_m = f"TRIX_{acc_params.get('trix_period', 12)}_{acc_params.get('trix_signal_period', 9)}_M"
        coppock_col_m = f"COPPOCK_{acc_params.get('coppock_long_roc', 14)}_{acc_params.get('coppock_short_roc', 11)}_{acc_params.get('coppock_wma', 10)}_M"
        # 步骤1: 准备好可能用到的“长期趋势分析”文本块，但不立即使用
        is_accumulating = signal_row.get('signal_monthly_accumulation', False)
        accumulation_analysis_text = []
        if is_accumulating:
            accumulation_analysis_text.append("--- 长期趋势分析 (月线) ---")
            accumulation_analysis_text.append("核心发现: **识别到月线级别趋势反转和主力吸筹迹象。**")
            if coppock_col_m in signal_row and signal_row[coppock_col_m] > 0:
                accumulation_analysis_text.append(f"  - 估波指标(Coppock): {signal_row[coppock_col_m]:.2f} > 0，发出经典长线买入信号。")
            if trix_col_m in signal_row and trix_col_m.replace('_M', '_M_prev') in signal_row and not pd.isna(signal_row[trix_col_m]) and not pd.isna(signal_row[trix_col_m.replace('_M', '_M_prev')]) and signal_row[trix_col_m] > signal_row[trix_col_m.replace('_M', '_M_prev')]:
                 accumulation_analysis_text.append(f"  - 三重平滑均线(TRIX): 指标值({signal_row[trix_col_m]:.2f})持续上升，表明长期趋势正在转强。")
            accumulation_analysis_text.append("  - 形态特征: 股价在长期均线附近盘整，为突破蓄力。")
        # 步骤2: 根据信号优先级，确定报告的主基调
        # 优先级最高：止盈信号
        if signal_row.get('take_profit_signal', 0) > 0:
            signal_type = "趋势止盈"
            total_score = 0
            main_analysis_parts.append("--- 卖出信号分析 ---")
            tp_type = signal_row['take_profit_signal']
            if tp_type == 1: reason = "原因: 股价已接近前期重要压力位。"
            elif tp_type == 2: reason = "原因: 股价从近期高点回撤，触发移动止盈。"
            elif tp_type == 3: reason = "原因: 技术指标显示市场过热或趋势转弱(如RSI超买)。"
            else: reason = ""
            main_analysis_parts.append(f"核心发现: **触发止盈条件，建议考虑获利了结或减仓。{reason}**")
            # 如果存在吸筹背景，则作为上下文补充，而不是作为矛盾的建议
            if accumulation_analysis_text:
                context_analysis_parts.append("\n--- 背景说明 ---")
                context_analysis_parts.append("注意: 该止盈信号是在一个良好的月线吸筹长线背景下出现的短期战术信号，适合了结前期利润，未来仍可关注新的介入机会。")
        # 优先级次之：风险信号
        elif signal_row.get('signal_ma_rejection', 0) == -1 or signal_row.get('signal_box_rejection', 0) == -1:
            signal_type = "压力位拒绝(风险)"
            total_score = 10
            main_analysis_parts.append("--- 风险信号分析 ---")
            main_analysis_parts.append("核心发现: **在关键压力位出现带量长上影线，为看跌拒绝信号，注意风险！**")
        # 优先级第三：买入信号
        elif signal_row.get('signal_pullback_entry', 0) > 0:
            signal_type = "回踩买入(高)"
            total_score = 95
            # 买入信号的理由，就是长期趋势向好
            if accumulation_analysis_text: main_analysis_parts.extend(accumulation_analysis_text)
            main_analysis_parts.append("\n--- 买入信号分析 (日线) ---")
            main_analysis_parts.append("核心发现: **在突破后出现健康的缩量回踩，确认为黄金买入点！**")
        elif signal_row.get('signal_continuation_entry', 0) > 0:
            signal_type = "追击买入(中)"
            total_score = 90
            if accumulation_analysis_text: main_analysis_parts.extend(accumulation_analysis_text)
            main_analysis_parts.append("\n--- 买入信号分析 (日线) ---")
            main_analysis_parts.append("核心发现: **突破后未回踩，持续沿短期均线上攻，确认为强势追击信号！**")
        # 优先级第四：观察信号
        elif signal_row.get('signal_breakout_trigger', False):
            signal_type = "突破观察(低)"
            total_score = 85
            if accumulation_analysis_text: main_analysis_parts.extend(accumulation_analysis_text)
            main_analysis_parts.append("\n--- 观察信号分析 (日线) ---")
            main_analysis_parts.append("核心发现: **已触发突破观察信号，建议密切关注后续回踩或追击机会。**")
            if signal_row.get('signal_monthly_breakout', False) and period_high_col in signal_row and not pd.isna(signal_row[period_high_col]):
                main_analysis_parts.append(f"  - 价格突破: 当月价格已突破长达 {lookback_months} 个月的盘整区间顶部({signal_row[period_high_col]:.2f})。")
        # 最低优先级：仅有长期状态信号，无短期事件
        elif is_accumulating:
            signal_type = "月线吸筹(观察)"
            total_score = 75 # 给予一个基础观察分
            main_analysis_parts.extend(accumulation_analysis_text)
            main_analysis_parts.append("\n--- 策略状态 ---")
            main_analysis_parts.append("核心发现: **股票处于长期吸筹阶段，但尚未出现明确的日线级别交易信号，建议纳入观察池持续跟踪。**")
        # 步骤3: 组合报告文本
        if signal_row.get('washout_score', 0) > 0:
            main_analysis_parts.append(f"\n洗盘分析: 检测到洗盘行为，强度评分为 {signal_row['washout_score']} 分。")
        report_text = "\n".join(main_analysis_parts + context_analysis_parts)
        full_report = f"""*** 最新信号分析报告 ({stock_code}) ***
            买入信号评分: {total_score} / 100
            信号类型: {signal_type}
            信号日期: {signal_row.name.strftime('%Y-%m-%d')}
            {report_text}
            """
        return {
            "analysis_text": full_report,
            "buy_score": total_score,
            "signal_type": signal_type
        }
    def prepare_db_records(self, stock_code: str, result_df: pd.DataFrame, params: dict) -> List[Dict[str, Any]]:
        """
        【V7.2 标准化版】将策略分析结果转换为用于数据库存储的标准化字典列表。
        - 修正时间戳键名为 `trade_time` 以匹配 V3.1 模型标准。
        """
        signal_conditions = (
            (result_df.get('signal_pullback_entry', 0) > 0) |
            (result_df.get('signal_continuation_entry', 0) > 0) |
            (result_df.get('take_profit_signal', 0) > 0) |
            (result_df.get('signal_ma_rejection', 0) != 0) |
            (result_df.get('signal_box_rejection', 0) != 0) |
            (result_df.get('signal_breakout_trigger', False))
        )
        df_with_signals = result_df[signal_conditions].copy()
        if df_with_signals.empty:
            return []
        records = []
        strategy_name = params.get('strategy_info', {}).get('name', 'MonthlyTrendFollow_V7')
        score_threshold = params.get('entry_scoring_params', {}).get('score_threshold', 90)
        for timestamp, row in df_with_signals.iterrows():
            entry_score = 0.0
            is_pullback = bool(row.get('signal_pullback_entry', 0) > 0)
            is_continuation = bool(row.get('signal_continuation_entry', 0) > 0)
            is_breakout_obs = bool(row.get('signal_breakout_trigger', False))
            exit_code = int(row.get('take_profit_signal', 0))
            if exit_code > 0: entry_score = 0.0
            elif row.get('signal_ma_rejection', 0) == -1 or row.get('signal_box_rejection', 0) == -1: entry_score = 10.0
            elif is_pullback: entry_score = 95.0
            elif is_continuation: entry_score = 90.0
            elif is_breakout_obs: entry_score = 85.0
            is_final_entry_signal = entry_score >= score_threshold
            rejection_code = 0
            if row.get('signal_ma_rejection', 0) == -1: rejection_code += 1
            if row.get('signal_box_rejection', 0) == -1: rejection_code += 2
            triggered_playbooks = []
            signal_columns = [
                'signal_monthly_accumulation', 'signal_monthly_breakout', 'signal_final_washout',
                'signal_breakout_initiation', 'signal_breakout_trigger', 'signal_pullback_entry',
                'signal_continuation_entry'
            ]
            for col in signal_columns:
                if row.get(col, False):
                    triggered_playbooks.append(col.upper())
            if exit_code > 0: triggered_playbooks.append(f'TAKE_PROFIT_CODE_{exit_code}')
            if rejection_code > 0: triggered_playbooks.append(f'REJECTION_CODE_{rejection_code}')
            context_snapshot = {k: v for k, v in row.items() if pd.notna(v)}
            record = {
                "stock_code": stock_code,
                # ▼▼▼ 修改/新增 ▼▼▼
                # 解释: 字段名从 'signal_time' 修正为 'trade_time'，与模型标准保持一致。
                "trade_time": timestamp,
                # ▲▲▲ 修改/新增 ▲▲▲
                "timeframe": "D",
                "strategy_name": strategy_name,
                "close_price": row.get('close_D'),
                "entry_score": entry_score,
                "entry_signal": is_final_entry_signal,
                "exit_signal_code": exit_code,
                "is_pullback_entry": is_pullback,
                "is_continuation_entry": is_continuation,
                "is_breakout_trigger": is_breakout_obs,
                "rejection_code": rejection_code,
                "washout_score": int(row.get('washout_score', 0)),
                "triggered_playbooks": triggered_playbooks,
                "context_snapshot": context_snapshot,
            }
            records.append(record)
        return records
    def run_analysis(self, stock_code: str, params_file: str = "config/monthly_trend_follow_strategy.json", trade_time: Optional[str] = None, data_df: Optional[pd.DataFrame] = None, start_date_str: Optional[str] = None) -> Tuple[Optional[pd.DataFrame], Optional[List[Dict[str, Any]]]]:
        """
        【V7.1 重构版 - 新增起始日期计算支持】运行单只股票的完整策略分析流程。
        - 现在返回一个符合通用信号日志模型的记录列表。
        - 新增功能: 增加了 start_date_str 参数，用于指定策略计算的起始点。
        """
        if self.favorite_stock_set is None:
            favorite_stocks = self.loop.run_until_complete(self.stock_basic_dao.get_all_favorite_stocks())
            self.favorite_stock_set = {stock.stock_id for stock in favorite_stocks}
        if data_df is None:
            # 数据准备阶段不变，依然加载全历史数据以计算准确的基础指标
            df_base, _ = self.loop.run_until_complete(self.indicator_service.prepare_data_for_strategy(stock_code=stock_code, params_file=params_file, trade_time=trade_time))
        else:
            df_base = data_df.copy()
        if df_base is None or df_base.empty:
            logger.warning(f"为股票 {stock_code} 准备数据失败，分析终止。")
            return None, None
        try:
            with open(params_file, 'r', encoding='utf-8') as f:
                params = json.load(f)
        except Exception as e:
            logger.error(f"加载策略参数文件 '{params_file}' 失败: {e}")
            return None, None
        final_df = self.loop.run_until_complete(self.apply_strategy(df_base, params, start_date_str=start_date_str))
        if final_df.empty:
            return None, None
        # 调用新的、标准化的记录准备方法
        db_records = self.prepare_db_records(stock_code, final_df, params)
        # if db_records:
        #     logger.info(f"为 {stock_code} 生成了 {len(db_records)} 条标准化信号记录。")
        # else:
        #     logger.info(f"为 {stock_code} 的分析完成，无任何信号需要记录。")
        # 返回原始的DataFrame和标准化的记录列表
        return final_df, db_records
    # --------------------------------------------------------------------
    # 以下是所有策略的辅助计算方法 (Helper Methods)
    # --------------------------------------------------------------------
    def apply_take_profit_rules(self, df: pd.DataFrame, params: dict) -> pd.Series:
        """
        【V6.3 增强版】应用所有已启用的止盈规则，返回一个代表止盈类型的整数Series。
        编码: 0=无信号, 1=压力位止盈, 2=移动止盈, 3=指标止盈
        """
        # 整个函数逻辑重构，以返回类型编码而不是布尔值
        # 解释: 我们创建一个新的列来存储止盈类型，并按优先级填充它。
        #       这样可以精确地知道是哪个规则触发了信号。
        take_profit_type = pd.Series(0, index=df.index)
        # 计算每种止盈信号
        tp_signal_resistance = self._check_resistance_take_profit(df, params.get('resistance_exit', {}))
        tp_signal_trailing = self._check_trailing_stop_take_profit(df, params.get('trailing_stop_exit', {}))
        tp_signal_indicator = self._check_indicator_take_profit(df, params.get('indicator_exit', {}))
        # 按优先级（指标 > 移动 > 压力位）填充止盈类型
        # 这样如果同时触发多个，会记录最优先的那个
        take_profit_type.loc[tp_signal_resistance] = 1
        take_profit_type.loc[tp_signal_trailing] = 2
        take_profit_type.loc[tp_signal_indicator] = 3
        # 返回包含止盈类型编码的Series
        return take_profit_type
    def _check_monthly_accumulation(self, df: pd.DataFrame, params: dict) -> pd.Series:
        if not params.get('enabled', False): return pd.Series(True, index=df.index)
        ma_period = params.get('ma_period', 20)
        trix_period = params.get('trix_period', 12)
        trix_signal_period = params.get('trix_signal_period', 9)
        coppock_long_roc = params.get('coppock_long_roc', 14)
        coppock_short_roc = params.get('coppock_short_roc', 11)
        coppock_wma = params.get('coppock_wma', 10)
        ma_col = f'EMA_{ma_period}_M'
        trix_col = f'TRIX_{trix_period}_{trix_signal_period}_M'
        coppock_col = f'COPPOCK_{coppock_long_roc}_{coppock_short_roc}_{coppock_wma}_M'
        is_coppock_buy = pd.Series(False, index=df.index)
        is_trix_rising = pd.Series(False, index=df.index)
        is_near_ma = pd.Series(False, index=df.index)
        if coppock_col in df.columns: is_coppock_buy = df[coppock_col] > 0
        if trix_col in df.columns and not df[trix_col].isnull().all(): is_trix_rising = df[trix_col] > df[trix_col].shift(1)
        # 放宽 is_near_ma 的限制，允许股价在更宽的范围内被识别为吸筹
        # 原逻辑: is_near_ma = (df['close_M'] / df[ma_col]).between(0.9, 1.3)
        # 新逻辑: 允许股价在20月线上方更广阔的区间，或轻微跌破
        if ma_col in df.columns and 'close_M' in df.columns: is_near_ma = (df['close_M'] / df[ma_col]).between(0.85, 1.5)
        else: is_near_ma = pd.Series(True, index=df.index)
        accumulation_signal = (is_coppock_buy | is_trix_rising) & is_near_ma
        return accumulation_signal
    def _check_monthly_breakout(self, df: pd.DataFrame, params: dict) -> pd.Series:
        if not params.get('enabled', False): return pd.Series(True, index=df.index)
        lookback_months = params.get('lookback_months', 6)
        volume_multiplier = params.get('volume_multiplier', 1.5)
        period_high_col = f'consolidation_period_high_M_{lookback_months}'
        period_low_col = f'consolidation_period_low_M_{lookback_months}'
        avg_volume_col = f'avg_volume_M_{lookback_months}'
        required_cols = ['close_M', 'volume_M', period_high_col, period_low_col, avg_volume_col]
        if not all(col in df.columns for col in required_cols):
            return pd.Series(False, index=df.index)
        is_price_breakout = df['close_M'] > df[period_high_col]
        is_volume_breakout = df['volume_M'] > (df[avg_volume_col] * volume_multiplier)
        is_above_support = df['close_M'] > df[period_low_col]
        monthly_breakout_signal = is_price_breakout & is_volume_breakout & is_above_support
        return monthly_breakout_signal
    def _find_pullback_entry_signal(self, df: pd.DataFrame, params: dict) -> pd.Series:
        if not params.get('enabled', False): return pd.Series(0, index=df.index)
        lookback_window = params.get('lookback_window', 3)
        volume_upper_bound_ratio = params.get('volume_upper_bound_ratio', 2.0)
        monthly_breakout_params = self.params.get('monthly_breakout_params', {})
        lookback_months = monthly_breakout_params.get('lookback_months', 6)
        support_level_col = f'consolidation_period_high_M_{lookback_months}'
        if support_level_col not in df.columns:
            return pd.Series(0, index=df.index)
        breakout_days = df['signal_breakout_initiation'] == True
        pullback_signal = pd.Series(0, index=df.index)
        breakout_indices = df.index[breakout_days]
        for idx in breakout_indices:
            breakout_day_volume = df.loc[idx, 'volume_D']
            breakout_support_level = df.loc[idx, support_level_col]
            try: breakout_loc = df.index.get_loc(idx)
            except KeyError: continue
            for i in range(1, lookback_window + 1):
                current_loc = breakout_loc + i
                if current_loc >= len(df): break 
                current_idx = df.index[current_loc]
                is_not_another_breakout = df.loc[current_idx, 'signal_breakout_initiation'] == False
                is_volume_controlled = df.loc[current_idx, 'volume_D'] < breakout_day_volume * volume_upper_bound_ratio
                is_above_support = df.loc[current_idx, 'close_D'] >= breakout_support_level
                if is_not_another_breakout and is_volume_controlled and is_above_support:
                    pullback_signal.loc[current_idx] = 1
                    break
        return pullback_signal
    def _find_continuation_entry_signal(self, df: pd.DataFrame, params: dict) -> pd.Series:
        if not params.get('enabled', False): return pd.Series(0, index=df.index)
        continuation_days = params.get('continuation_days', 3)
        support_ma_period = params.get('support_ma_period', 5)
        support_ma_col = f'EMA_{support_ma_period}_D'
        if support_ma_col not in df.columns:
            return pd.Series(0, index=df.index)
        breakout_indices = df.index[df['signal_breakout_initiation'] == True]
        continuation_signal = pd.Series(0, index=df.index)
        for idx in breakout_indices:
            try: breakout_loc = df.index.get_loc(idx)
            except KeyError: continue
            is_strong_continuation = True
            if breakout_loc + continuation_days >= len(df): is_strong_continuation = False
            else:
                for i in range(1, continuation_days + 1):
                    current_loc = breakout_loc + i
                    current_idx = df.index[current_loc]
                    if df.loc[current_idx, 'close_D'] <= df.loc[current_idx, support_ma_col]:
                        is_strong_continuation = False
                        break
            if is_strong_continuation:
                entry_day_loc = breakout_loc + continuation_days
                entry_day_idx = df.index[entry_day_loc]
                continuation_signal.loc[entry_day_idx] = 1
        return continuation_signal
    def _calculate_washout_score(self, df: pd.DataFrame, params: dict) -> pd.Series:
        if not params.get('enabled', False): return pd.Series(0, index=df.index)
        washout_score = pd.Series(0, index=df.index)
        support_level = self._get_support_level(df, params)
        if support_level is None: return washout_score
        washout_intraday = (df['low_D'] < support_level) & (df['close_D'] > support_level)
        washout_interday = (df['close_D'] > support_level) & (df['close_D'].shift(1) < support_level.shift(1))
        drift_lookback = params.get('drift_lookback_period', 3)
        was_below_recently = (df['close_D'].shift(1) < support_level.shift(1)).rolling(window=drift_lookback, min_periods=1).sum() > 0
        washout_drift = (df['close_D'] > support_level) & was_below_recently
        bull_trap_params = params.get('bull_trap_params', {})
        if bull_trap_params.get('enabled', False):
            lookback = bull_trap_params.get('lookback_period', 8)
            drop_threshold = bull_trap_params.get('drop_threshold', 0.05)
            recent_peak = df['high_D'].shift(1).rolling(window=lookback).max()
            is_in_trap_zone = df['close_D'] < recent_peak * (1 - drop_threshold)
            is_recovering_from_trap = df['close_D'] > df['close_D'].shift(1)
            washout_bull_trap = is_in_trap_zone & is_recovering_from_trap
        else: washout_bull_trap = pd.Series(False, index=df.index)
        vol_con_params = params.get('volume_contraction_params', {})
        if vol_con_params.get('enabled', False):
            avg_period = vol_con_params.get('avg_period', 20)
            threshold = vol_con_params.get('threshold', 0.7)
            avg_volume = df['volume_D'].shift(1).rolling(window=avg_period).mean()
            is_volume_contracted = df['volume_D'] < avg_volume * threshold
            washout_volume_contraction = (washout_interday | washout_drift) & is_volume_contracted.shift(1).fillna(False)
        else: washout_volume_contraction = pd.Series(False, index=df.index)
        uo_div_params = params.get('uo_divergence_params', {})
        if uo_div_params.get('enabled', False):
            periods = uo_div_params.get('periods', [7, 14, 28])
            lookback = uo_div_params.get('lookback_period', 5)
            uo_col = f'UO_{periods[0]}_{periods[1]}_{periods[2]}_D'
            if uo_col in df.columns:
                low_in_lookback = df['low_D'].rolling(window=lookback).min()
                uo_at_low = df[uo_col].rolling(window=lookback).min()
                price_makes_lower_low = df['low_D'] < low_in_lookback.shift(1)
                uo_makes_higher_low = df[uo_col] > uo_at_low.shift(1)
                washout_uo_divergence = price_makes_lower_low & uo_makes_higher_low
            else: washout_uo_divergence = pd.Series(False, index=df.index)
        else: washout_uo_divergence = pd.Series(False, index=df.index)
        washout_score += washout_intraday.astype(int)
        washout_score += washout_interday.astype(int)
        washout_score += washout_drift.astype(int)
        washout_score += washout_bull_trap.astype(int)
        washout_score += washout_volume_contraction.astype(int)
        washout_score += washout_uo_divergence.astype(int)
        pre_condition = df.get('signal_monthly_accumulation', pd.Series(True, index=df.index))
        final_score = washout_score.where(pre_condition, 0)
        return final_score
    def _check_ma_rejection_signal(self, df: pd.DataFrame, params: dict) -> pd.Series:
        if not params.get('enabled', False): return pd.Series(0, index=df.index)
        ma_period = params.get('ma_period', 20)
        ma_col = f'EMA_{ma_period}_D'
        return self._check_resistance_rejection(df, ma_col, params)
    def _check_box_rejection_signal(self, df: pd.DataFrame, params: dict) -> pd.Series:
        if not params.get('enabled', False): return pd.Series(0, index=df.index)
        lookback_period = params.get('lookback_period', 60)
        resistance_col_name = f'box_top_{lookback_period}D_resistance'
        df[resistance_col_name] = df['high_D'].shift(1).rolling(window=lookback_period, min_periods=int(lookback_period * 0.8)).max()
        return self._check_resistance_rejection(df, resistance_col_name, params)
    def _get_support_level(self, df: pd.DataFrame, washout_params: dict) -> pd.Series | None:
        support_type = washout_params.get('support_type', 'MA')
        support_level = pd.Series(np.nan, index=df.index)
        if support_type == 'MA':
            ma_period = washout_params.get('support_ma_period', 20)
            ma_col = f'EMA_{ma_period}_D'
            if ma_col not in df.columns: return None
            support_level = df[ma_col]
        elif support_type == 'BOX':
            box_period = washout_params.get('box_period', 20)
            bbw_col = 'BBW_D'
            volatility_threshold = washout_params.get('box_volatility_threshold', 0.1)
            if bbw_col not in df.columns: return None
            box_bottom = df['low_D'].rolling(window=box_period).min()
            is_consolidating = df[bbw_col] < volatility_threshold
            support_level = box_bottom.where(is_consolidating, np.nan)
        if support_level.isnull().all(): return None
        support_level.ffill(inplace=True)
        return support_level
    def _check_resistance_take_profit(self, df: pd.DataFrame, params: dict) -> pd.Series:
        if not params.get('enabled', False): return pd.Series(False, index=df.index)
        lookback_period = params.get('lookback_period', 90)
        approach_threshold = params.get('approach_threshold', 0.02)
        resistance_level = df['high_D'].shift(1).rolling(window=lookback_period, min_periods=int(lookback_period*0.8)).max()
        signal = df['high_D'] >= resistance_level * (1 - approach_threshold)
        return signal & resistance_level.notna()
    def _check_trailing_stop_take_profit(self, df: pd.DataFrame, params: dict) -> pd.Series:
        if not params.get('enabled', False): return pd.Series(False, index=df.index)
        pullback_percentage = params.get('percentage_pullback', 0.10)
        lookback_period = params.get('lookback_period', 20)
        highest_close_since = df['close_D'].shift(1).rolling(window=lookback_period, min_periods=int(lookback_period*0.8)).max()
        stop_price = highest_close_since * (1 - pullback_percentage)
        signal = df['close_D'] < stop_price
        return signal & highest_close_since.notna()
    def _check_indicator_take_profit(self, df: pd.DataFrame, params: dict) -> pd.Series:
        if not params.get('enabled', False): return pd.Series(False, index=df.index)
        indicator_type = params.get('indicator_type', 'RSI')
        if indicator_type == 'RSI':
            rsi_period = params.get('rsi_period', 14)
            threshold = params.get('threshold', 80)
            rsi_col = f'RSI_{rsi_period}_D'
            if rsi_col not in df.columns: return pd.Series(False, index=df.index)
            return df[rsi_col] > threshold
        elif indicator_type == 'TRIX':
            trix_period = params.get('trix_period', 12)
            signal_period = params.get('trix_signal_period', 9)
            trix_col = f'TRIX_{trix_period}_{signal_period}_D'
            trix_signal_col = f'TRIXs_{trix_period}_{signal_period}_D'
            if not all(col in df.columns for col in [trix_col, trix_signal_col]): return pd.Series(False, index=df.index)
            return (df[trix_col].shift(1) > df[trix_signal_col].shift(1)) & (df[trix_col] < df[trix_signal_col])
        return pd.Series(False, index=df.index)
    def _check_resistance_rejection(self, df: pd.DataFrame, resistance_col: str, params: dict) -> pd.Series:
        volume_multiplier = params.get('volume_multiplier', 1.5)
        vol_ma_col = 'VOL_MA_21_D' 
        if resistance_col not in df.columns or vol_ma_col not in df.columns:
            return pd.Series(0, index=df.index)
        pattern_condition = (df['high_D'] > df[resistance_col]) & (df['close_D'] < df[resistance_col])
        upper_shadow = df['high_D'] - df[['open_D', 'close_D']].max(axis=1)
        body_size = (df['open_D'] - df['close_D']).abs().replace(0, 0.0001)
        long_upper_shadow_condition = (upper_shadow / body_size) > 1.0
        high_volume_condition = df['volume_D'] > df[vol_ma_col] * volume_multiplier
        rejection_signal = pattern_condition & long_upper_shadow_condition & high_volume_condition
        return pd.Series(np.where(rejection_signal, -1, 0), index=df.index)
