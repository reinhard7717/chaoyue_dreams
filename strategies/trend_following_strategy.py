# 文件: strategies/trend_following_strategy.py
# 版本: V62.0 - 增强调试日志版
from copy import deepcopy
import json
import logging
from decimal import Decimal
import os
from utils.data_sanitizer import sanitize_for_json
from typing import Any, Dict, List, Optional, Tuple
from scipy.signal import find_peaks
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, peak_prominences
from strategies.kline_pattern_recognizer import KlinePatternRecognizer
from utils.config_loader import load_strategy_config

logger = logging.getLogger(__name__)

class TrendFollowStrategy:
    """
    趋势跟踪策略 (V62.0 - 增强调试日志版)
    - 核心升级: 新增 _format_debug_dates 辅助函数，并全局应用于所有诊断日志，使其能输出具体发生日期，极大提升调试效率。
    """
    def __init__(self, config: dict):
        """
        【V21.1 性能优化版】
        - 核心优化: 在初始化时，一次性加载所有策略模块的参数块到实例变量中。
        - 收益: 1. 避免在策略执行过程中反复调用 _get_params_block 进行字典查找，提升微观性能。
                 2. 使后续代码可以直接通过 self.xxx_params 访问配置，更加简洁清晰。
        """
        self.unified_config = config # 修改：直接使用传入的config，而不是命名为daily_params
        
        # ▼▼▼ 一次性加载所有参数块 ▼▼▼
        self.strategy_info = self._get_params_block(self.unified_config, 'strategy_info')
        self.scoring_params = self._get_params_block(self.unified_config, 'four_layer_scoring_params')
        self.setup_scoring_matrix = self._get_params_block(self.unified_config, 'setup_scoring_matrix')
        self.exit_strategy_params = self._get_params_block(self.unified_config, 'exit_strategy_params')
        self.risk_veto_params = self._get_params_block(self.unified_config, 'risk_veto_params')
        self.debug_params = self._get_params_block(self.unified_config, 'debug_params')
        self.kline_params = self._get_params_block(self.unified_config, 'kline_pattern_params')

        self.pattern_recognizer = KlinePatternRecognizer(params=self.kline_params)
        self.signals = {}
        self.scores = {}
        self._last_score_details_df = None
        self.verbose_logging = self.debug_params.get('enabled', False) and self.debug_params.get('verbose_logging', False)
        
        self.playbook_blueprints = self._get_playbook_blueprints()
        self.risk_playbook_blueprints = self._get_risk_playbook_blueprints()
        
        print(f"--- [战术策略 TrendFollowStrategy (V91.0 风险剧本架构)] 初始化完成 ---")
        print(f"    -> 已缓存 {len(self.playbook_blueprints)} 个入场剧本蓝图。")
        print(f"    -> 已缓存 {len(self.risk_playbook_blueprints)} 个风险剧本蓝图。")
        print(f"    -> 关键参数块已预加载到实例变量中。")

    #  日志格式化辅助函数 ▼▼▼
    def _format_debug_dates(self, signal_series: pd.Series, display_limit: int = 10) -> str:
        """
        【V92.1 修复版】
        - 核心修复: 恢复了显示具体日期的能力，对调试至关重要。
        - 特性: 当日期不多时全部显示；日期过多时，显示最近的N个并附上总数。
        """
        if not isinstance(signal_series, pd.Series) or signal_series.dtype != bool:
            return ""
        
        active_dates = signal_series.index[signal_series]
        count = len(active_dates)
        
        if count == 0:
            return ""
            
        date_strings = [d.strftime('%Y-%m-%d') for d in active_dates]
        
        if count > display_limit:
            # 如果日期太多，只显示最近的N个并附上总数
            return f" -> 日期: [...{date_strings[-display_limit:]}] (共 {count} 天)"
        else:
            # 否则全部显示
            return f" -> 日期: {date_strings}"

    def _get_param_value(self, param: Any, default: Any = None) -> Any:
        """
        【V40.3 新增】健壮的参数值解析器。
        """
        if isinstance(param, dict) and 'value' in param:
            return param['value']
        if param is not None:
            return param
        return default

    def _get_params_block(self, params: dict, block_name: str, default_return: Any = None) -> dict:
        if default_return is None:
            default_return = {}
        return params.get('strategy_params', {}).get('trend_follow', {}).get(block_name, default_return)

    def _get_periods_for_timeframe(self, indicator_params: dict, timeframe: str) -> Optional[list]:
        """
        【V5.0 】根据时间周期从指标配置中智能获取正确的'periods'。
        """
        if not indicator_params:
            return None
        if 'configs' in indicator_params and isinstance(indicator_params['configs'], list):
            for config_item in indicator_params['configs']:
                if timeframe in config_item.get('apply_on', []):
                    return config_item.get('periods')
            return None
        elif 'periods' in indicator_params:
            apply_on = indicator_params.get('apply_on', [])
            if not apply_on or timeframe in apply_on:
                return indicator_params.get('periods')
            else:
                return None
        return None

    def _ensure_numeric_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        【V40.6 智能净化版】数据类型标准化引擎
        """
        converted_cols = []
        for col in df.columns:
            if df[col].dtype == 'object':
                first_valid_item = df[col].dropna().iloc[0] if not df[col].dropna().empty else None
                if isinstance(first_valid_item, Decimal):
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    converted_cols.append(col)
        if not converted_cols:
            print("      -> 所有数值列类型正常，无需转换。")
        return df

    def apply_strategy(self, df: pd.DataFrame, params: dict) -> Tuple[pd.DataFrame, Dict[str, pd.Series]]:
        """
        【V202.10 重构优化版】
        - 核心重构: 将原有的线性执行流程，按逻辑功能拆分为独立的私有方法：
                    _run_all_diagnostics: 运行所有状态诊断。
                    _calculate_all_scores: 计算所有得分（入场分、风险分）。
                    _apply_risk_veto_and_finalize: 执行风险否决并生成最终信号。
        - 收益: 主流程逻辑更清晰，代码结构更合理，便于维护和扩展。
        """
        print("======================================================================")
        print(f"====== 日期: {df.index[-1].date()} | 正在执行【战术引擎 V202.10 重构优化版】 ======")
        print("======================================================================")

        if df is None or df.empty: return pd.DataFrame(), {}
        df = self._ensure_numeric_types(df)

        # 步骤1：运行所有诊断，生成原子状态和触发事件
        df, atomic_states, trigger_events = self._run_all_diagnostics(df, params)

        # 步骤2：计算所有得分，包括入场分和风险分
        df, score_details_df, risk_details_df = self._calculate_all_scores(df, params, atomic_states, trigger_events)
        self._last_score_details_df = score_details_df
        self._last_risk_details_df = risk_details_df

        # 步骤3：执行风险否决，并生成最终信号
        df = self._apply_risk_veto_and_finalize(df, params)
        
        # 步骤4：运行持仓管理模拟
        print("--- [总指挥] 步骤4: 启动【持仓管理引擎】，模拟全程战术动作 ---")
        df = self._run_position_management_simulation(df, params)

        print(f"====== 【战术引擎 V202.10】执行完毕 ======")
        return df, atomic_states

    def _run_all_diagnostics(self, df: pd.DataFrame, params: dict) -> Tuple[pd.DataFrame, Dict, Dict]:
        """
        【新增辅助方法】运行所有诊断模块，生成原子状态和触发事件。
        """
        print("--- [总指挥] 步骤1: 运行所有诊断模块... ---")
        df = self.pattern_recognizer.identify_all(df)
        if 'close_D' in df.columns: df['pct_change_D'] = df['close_D'].pct_change()
        
        df, platform_states = self._diagnose_platform_states(df, params)
        
        atomic_states = {
            **self._diagnose_chip_states(df, params), 
            **self._diagnose_ma_states(df, params), 
            **self._diagnose_oscillator_states(df, params), 
            **self._diagnose_capital_states(df, params), 
            **self._diagnose_volatility_states(df, params), 
            **self._diagnose_box_states(df, params), 
            **self._diagnose_kline_patterns(df, params), 
            **self._diagnose_board_patterns(df, params), 
            **platform_states, 
            **self._diagnose_trend_dynamics(df, params)
        }
        
        # 定义复合原子状态
        atomic_states['CONTEXT_TREND_DETERIORATING'] = self._define_context_trend_deteriorating(df, atomic_states)
        atomic_states['PLATFORM_FAILURE'] = (df['close_D'] < df.get('PLATFORM_PRICE_STABLE', np.inf)) & (df.get('PLATFORM_PRICE_STABLE', np.inf) > 0)
        atomic_states['CHIP_DISPERSION_RISK'] = df.get('CHIP_concentration_90pct_slope_5d_D', 0) > 0
        is_extreme_winner_rate = df.get('CHIP_winner_rate_long_term_D', 0) > 98
        is_stagnating = df['high_D'].rolling(window=3).max() <= df['high_D'].shift(1)
        atomic_states['RISK_EXTREME_PROFIT_TAKING'] = is_extreme_winner_rate & is_stagnating
        atomic_states['CHIP_RAPID_CONCENTRATION'] = df.get('CHIP_concentration_90pct_slope_5d_D', 0) < -0.005
        
        trigger_events = self._define_trigger_events(df, params, atomic_states)
        is_in_squeeze_window = atomic_states.get('VOL_STATE_SQUEEZE_WINDOW', pd.Series(False, index=df.index))
        is_bb_breakout = df['close_D'] > df.get('BBU_21_2.0_D', float('inf'))
        trigger_events['VOL_BREAKOUT_FROM_SQUEEZE'] = is_bb_breakout & is_in_squeeze_window.shift(1).fillna(False)
        
        return df, atomic_states, trigger_events

    def _calculate_all_scores(self, df: pd.DataFrame, params: dict, atomic_states: Dict, trigger_events: Dict) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        【新增辅助方法】计算所有得分，包括入场分和风险分。
        """
        print("--- [总指挥] 步骤2: 计算入场分与风险分... ---")
        # 计算原始入场分
        df, score_details_df = self._calculate_entry_score(df, params, trigger_events, {}, atomic_states)
        raw_total_score = df['entry_score'].copy()
        
        # 应用最终调整
        adjusted_score, adjustment_details = self._apply_final_score_adjustments(df, raw_total_score, params, atomic_states)
        df['entry_score_raw'] = raw_total_score
        df['entry_score'] = adjusted_score
        
        final_score_details_df = pd.concat([score_details_df, adjustment_details], axis=1).fillna(0)
        
        # 计算风险分
        risk_setups = self._diagnose_risk_setups(df, params, atomic_states)
        risk_triggers = self._define_risk_triggers(df, params)
        risk_score, risk_details_df = self._calculate_risk_score(df, params, risk_setups, risk_triggers)
        df['risk_score'] = risk_score
        
        return df, final_score_details_df, risk_details_df

    def _apply_risk_veto_and_finalize(self, df: pd.DataFrame, params: dict) -> pd.DataFrame:
        """
        【新增辅助方法】执行风险否决，并生成最终的入场和出场信号。
        """
        print("--- [总指挥] 步骤3: 执行风险否决并生成最终信号... ---")
        veto_params = self.risk_veto_params
        if self._get_param_value(veto_params.get('enabled'), False):
            ratio = self._get_param_value(veto_params.get('risk_tolerance_ratio'), 0.5)
            min_risk = self._get_param_value(veto_params.get('min_absolute_risk_for_veto'), 30)
            
            print(f"    -> [裁决标准] 风险容忍系数: {ratio:.2f}, 触发否决的最低绝对风险: {min_risk}")
            df['dynamic_veto_threshold'] = df['entry_score'] * ratio
            veto_mask = (df['risk_score'] > df['dynamic_veto_threshold']) & (df['risk_score'] >= min_risk)
            
            original_buy_signals_to_veto = (df['entry_score'] > 0) & veto_mask
            if original_buy_signals_to_veto.any():
                print(f"    -> [风控裁决] 在 {original_buy_signals_to_veto.sum()} 个交易日，风险分超过了买入分可容忍的上限。")
                print(f"    -> [执行否决] 已将这些交易日的 entry_score 强制清零！")
                df.loc[veto_mask, 'entry_score'] = 0
            else:
                print("    -> [风控裁决] 所有买入信号均在风险容忍范围内，无需否决。")
        else:
            print("    -> [风控裁决] 动态否决系统被禁用，跳过此步骤。")
            
        # 计算最终出场信号码
        df['exit_signal_code'] = self._calculate_exit_signals(df, params, df['risk_score'])
        
        # 计算最终入场信号
        score_threshold = self._get_param_value(self._get_params_block(params, 'entry_scoring_params', {}).get('score_threshold'), 100)
        df['signal_entry'] = df['entry_score'] >= score_threshold
        
        return df

    # 辅助函数，用于定义 CONTEXT_TREND_DETERIORATING
    def _define_context_trend_deteriorating(self, df, atomic_states):
        default_series = pd.Series(False, index=df.index)
        is_in_divergence_window = atomic_states.get('CAPITAL_STATE_DIVERGENCE_WINDOW', default_series)
        is_long_ma_slope_negative = df.get('SLOPE_21_EMA_55_D', 0) < 0
        is_short_ma_slope_negative = df.get('SLOPE_5_EMA_13_D', 0) < 0
        is_long_ma_accel_negative = df.get('ACCEL_21_EMA_55_D', 0) < 0
        is_long_chip_slope_negative = df.get('CHIP_peak_cost_slope_55d_D', 0) < 0
        unconditional_deterioration = (is_long_ma_slope_negative & is_short_ma_slope_negative & is_long_ma_accel_negative & is_long_chip_slope_negative)
        return unconditional_deterioration & ~is_in_divergence_window

    def _quantify_risk_reasons(self, record: Dict) -> str:
        """
        【V202.14 新增】风险量化器 (Risk Quantifier)
        - 核心职责: 读取配置文件中的量化模型，将原始风险原因(如'主峰根基动摇')
                    翻译成包含 0-100 直观风险分的“仪表盘”读数。
        - 所属单位: TrendFollowStrategy (因为它拥有参数解析器 _get_params_block)
        """
        # ▼▼▼【代码修改 V202.14】: 使用 self._get_params_block 获取配置 ▼▼▼
        quantifier_params = self._get_params_block(self.daily_params, 'risk_quantifier_params', {})
        if not self._get_param_value(quantifier_params.get('enabled'), False):
            # 如果被禁用，则返回原始的、未量化的原因
            return record.get('exit_signal_reason', "") or ""
        # ▲▲▲【代码修改 V202.14】▲▲▲

        context = record.get('context_snapshot', {})
        # 从 triggered_playbooks_cn 获取最原始、最干净的原因列表
        reason_list = record.get('triggered_playbooks_cn', [])
        if not reason_list:
            return record.get('exit_signal_reason', "原因未知")

        quantified_parts = []
        
        for risk_key, config in quantifier_params.items():
            if not isinstance(config, dict): continue
            
            cn_name = config.get('cn_name')
            # 检查当前风险原因是否在需要量化的列表中
            if cn_name and cn_name in reason_list:
                metric_key = config.get('source_metric')
                raw_value = context.get(metric_key)
                
                if raw_value is None or not isinstance(raw_value, (int, float)):
                    quantified_parts.append(cn_name) # 如果没有量化数据，则只显示名称
                    continue

                direction = config.get('direction', 1)
                center = config.get('center_point', 0)
                steepness = config.get('steepness', 1)
                
                try:
                    normalized_score = 1 / (1 + np.exp(-steepness * direction * (raw_value - center)))
                    final_score = int(normalized_score * 100)
                    quantified_parts.append(f"{cn_name}({final_score}/100)")
                except (OverflowError, ValueError):
                    quantified_parts.append(cn_name) # 计算出错则只显示名称
        
        # 处理那些不在量化配置中的其他原因
        unquantified_reasons = [reason for reason in reason_list if not any(reason in qp for qp in quantified_parts)]
        
        # 将量化后的部分和未量化的部分合并
        final_parts = quantified_parts + unquantified_reasons
        return ", ".join(final_parts) if final_parts else "综合风险"

    def prepare_db_records(self, stock_code: str, result_df: pd.DataFrame, params: dict, result_timeframe: str = 'D') -> List[Dict[str, Any]]:
        """
        【V202.16 健壮性修复版】统一战报司令部
        - 核心修复: 解决了 `np.select` 在退出规则配置为空时崩溃的问题。
                    在计算 `exit_severity_level` 之前，增加了一个保护性检查。
                    如果从配置中未找到任何有效的退出规则，则安全地将严重级别设置为0，而不是尝试执行会导致错误的 `np.select`。
        - 业务逻辑: 保持不变。
        """
        required_cols = ['signal_entry', 'exit_signal_code', f'close_{result_timeframe}', 'risk_score']
        if not all(col in result_df.columns for col in required_cols):
            print(f"      -> [错误] prepare_db_records 缺少必要列: {required_cols}")
            return []
        
        df = result_df[(result_df['signal_entry'] == True) | (result_df['risk_score'] > 0)].copy()
        if df.empty: return []

        # --- 准备工作 ---
        strategy_info = self._get_params_block(params, 'strategy_info', {})
        strategy_name = self._get_param_value(strategy_info.get('name'), 'unknown_strategy')
        playbook_map = {p['name']: p.get('cn_name', p['name']) for p in self.playbook_blueprints}
        risk_playbook_map = {p['name']: p.get('cn_name', p['name']) for p in self.risk_playbook_blueprints}
        exit_cfg = params.get('exit_strategy_params', {})
        warning_thresh = exit_cfg.get('warning_threshold_params', {})
        min_warning_level = min([cfg['level'] for cfg in warning_thresh.values() if isinstance(cfg, dict)]) if warning_thresh else 30
        exit_thresh = exit_cfg.get('exit_threshold_params', {})

        # --- 向量化计算 ---
        is_exit = df['exit_signal_code'] > 0
        is_entry = df['signal_entry']
        is_warning = (df['risk_score'] >= min_warning_level) & ~is_exit & ~is_entry
        
        df['stock_code'] = stock_code
        df['timeframe'] = result_timeframe
        df['strategy_name'] = strategy_name
        df['trade_time'] = df.index
        df['close_price'] = df[f'close_{result_timeframe}'].apply(lambda x: float(x) if pd.notna(x) else None)

        df['entry_signal'] = is_entry
        df['is_risk_warning'] = is_warning
        
        df['stable_platform_price'] = np.where(
            is_entry, df['PLATFORM_PRICE_STABLE'].apply(lambda x: float(x) if pd.notna(x) else None), None
        )

        # ▼▼▼【代码修改】修复 np.select 错误的核心逻辑 ▼▼▼
        # 步骤1: 过滤出有效的退出规则，排除"说明"等非字典项，确保迭代安全
        valid_exit_rules = {name: cfg for name, cfg in exit_thresh.items() if isinstance(cfg, dict) and 'code' in cfg}
        
        # 步骤2: 增加保护，仅在存在有效规则时才执行np.select
        if valid_exit_rules:
            print(f"    -> 检测到 {len(valid_exit_rules)} 条有效退出规则，正在计算严重等级...")
            conditions = [df['exit_signal_code'] == self._get_param_value(rule.get('code')) for name, rule in valid_exit_rules.items()]
            choices = [3 if name == "CRITICAL" else 2 for name in valid_exit_rules.keys()]
            df['exit_severity_level'] = np.select(conditions, choices, default=0)
        else:
            # 如果没有定义任何退出规则，则将严重级别安全地设置为0
            print("    -> 未检测到有效退出规则，所有信号的严重等级将设为0。")
            df['exit_severity_level'] = 0
        # ▲▲▲【代码修改】▲▲▲

        def get_playbooks(timestamp, is_entry_signal):
            df_details = self._last_score_details_df if is_entry_signal else self._last_risk_details_df
            if df_details is None or timestamp not in df_details.index: return [], []
            
            details_row = df_details.loc[timestamp]
            active_items = details_row[details_row > 0].index
            
            if is_entry_signal:
                excluded = ('BASE_', 'BONUS_', 'PENALTY_', 'INDUSTRY_', 'WATCHING_SETUP', 'CHIP_PURITY_MULTIPLIER', 'VOLATILITY_SILENCE_MULTIPLIER', 'RISK_SCORE_')
                playbooks_en = [item for item in active_items if not item.startswith(excluded)]
                playbooks_cn = [playbook_map.get(p, p) for p in playbooks_en]
            else:
                playbooks_en = [item.replace('RISK_SCORE_', '') for item in active_items]
                playbooks_cn = [risk_playbook_map.get(p, p) for p in playbooks_en]
            return playbooks_en, playbooks_cn

        playbook_results = df.apply(lambda row: get_playbooks(row.name, row['signal_entry']), axis=1)
        df[['triggered_playbooks', 'triggered_playbooks_cn']] = pd.DataFrame(playbook_results.tolist(), index=df.index)

        def build_final_details(row):
            context = {
                'close': row['close_price'],
                'entry_score': float(row['entry_score']) if pd.notna(row['entry_score']) else 0.0,
                'risk_score': float(row['risk_score']) if pd.notna(row['risk_score']) else 0.0,
            }
            reason = None
            if not row['entry_signal']:
                cn_playbooks = row['triggered_playbooks_cn']
                reason_str = ", ".join(cn_playbooks) if cn_playbooks else "综合风险评分超阈值"
                record_for_quantify = {'triggered_playbooks_cn': cn_playbooks, 'context_snapshot': row.to_dict()}
                reason = self._quantify_risk_reasons(record_for_quantify)
            return reason, context

        final_details = df.apply(build_final_details, axis=1)
        df[['exit_signal_reason', 'context_snapshot']] = pd.DataFrame(final_details.tolist(), index=df.index)

        final_cols = [
            "stock_code", "trade_time", "timeframe", "strategy_name", "close_price",
            "entry_score", "stable_platform_price", "entry_signal", "is_risk_warning",
            "exit_signal_code", "exit_severity_level", "exit_signal_reason",
            "triggered_playbooks", "triggered_playbooks_cn", "context_snapshot"
        ]
        for col in final_cols:
            if col not in df.columns:
                df[col] = None if col not in ['entry_score', 'exit_signal_code', 'exit_severity_level'] else 0.0

        sanitized_df = df[final_cols].replace({np.nan: None})

        return sanitized_df.to_dict('records')

    def _create_db_record_template(self, stock_code, timestamp, timeframe, strategy_name, row) -> Dict:
        """
        【V202.6 健壮性增强版】
        - 核心修复: 对所有从DataFrame中提取的数值类型，都使用 pd.notna() 进行检查，
                    并将 NaN 值转换成数据库可接受的类型（None 或 0.0），从根源上杜绝 `ValidationError`。
        """
        # ▼▼▼【代码修改】对所有数值进行NaN检查和转换 ▼▼▼
        close_price = row.get(f'close_{timeframe}')
        entry_score = row.get('entry_score', 0.0)
        risk_score = row.get('risk_score', 0.0)
        
        return {
            "stock_code": stock_code, "trade_time": timestamp, "timeframe": timeframe,
            "strategy_name": strategy_name, 
            "close_price": float(close_price) if pd.notna(close_price) else None, # 修复点1: close_price
            "entry_score": float(entry_score) if pd.notna(entry_score) else 0.0, # 修复点2: entry_score
            "stable_platform_price": None, # 此处留空，由主流程填充并检查
            "entry_signal": False, "is_risk_warning": False,
            "exit_signal_code": 0, "exit_severity_level": 0, "exit_signal_reason": None,
            "is_pullback_setup": bool(row.get('SETUP_SCORE_PLATFORM_SUPPORT_PULLBACK', 0) > 0),
            "triggered_playbooks": [], "triggered_playbooks_cn": [], 
            "context_snapshot": {
                'close': float(close_price) if pd.notna(close_price) else None,
                'entry_score': float(entry_score) if pd.notna(entry_score) else 0.0,
                'risk_score': float(risk_score) if pd.notna(risk_score) else 0.0, # 修复点3: risk_score
            }
        }
        # ▲▲▲【代码修改】▲▲▲

    def _fill_risk_details(self, record: Dict, row: pd.Series, risk_playbook_map: Dict):
        """
        【V202.7 健壮性增强版】
        - 核心修复: 对所有添加到 context_snapshot 的数值，进行 pd.notna() 检查，确保数据清洁。
        """
        timestamp = record['trade_time']
        triggered_risks_en = []
        risk_details_df = getattr(self, '_last_risk_details_df', pd.DataFrame())
        if not risk_details_df.empty and timestamp in risk_details_df.index:
            risk_details_for_day = risk_details_df.loc[timestamp]
            active_risks = risk_details_for_day[risk_details_for_day > 0].index
            triggered_risks_en = [risk.replace('RISK_SCORE_', '') for risk in active_risks]
        
        triggered_risks_cn = [risk_playbook_map.get(risk, risk) for risk in triggered_risks_en]
        reason = ", ".join(triggered_risks_cn) if triggered_risks_cn else "综合风险评分超阈值"
        
        record['exit_signal_reason'] = reason
        record['triggered_playbooks'] = triggered_risks_en
        record['triggered_playbooks_cn'] = triggered_risks_cn
        
        context = record['context_snapshot']
        # ▼▼▼【代码修改】增加NaN检查 ▼▼▼
        context['cost_slope_5d'] = float(v) if pd.notna(v := row.get('SLOPE_5_CHIP_peak_cost_D')) else None
        context['winner_rate_slope_5d'] = float(v) if pd.notna(v := row.get('SLOPE_5_CHIP_total_winner_rate_D')) else None
        context['peak_stability_slope_5d'] = float(v) if pd.notna(v := row.get('SLOPE_5_CHIP_peak_stability_D')) else None
        context['pressure_above_slope_5d'] = float(v) if pd.notna(v := row.get('SLOPE_5_CHIP_pressure_above_D')) else None
        # ▲▲▲【代码修改】▲▲▲
        record['context_snapshot'] = sanitize_for_json(context) # sanitize_for_json 仍然保留，用于处理其他JSON不支持的类型

    def _create_db_record_template(self, stock_code, timestamp, timeframe, strategy_name, row) -> Dict:
        """【V202.6 辅助函数】创建标准化的数据库记录模板"""
        return {
            "stock_code": stock_code, "trade_time": timestamp, "timeframe": timeframe,
            "strategy_name": strategy_name, "close_price": row.get(f'close_{timeframe}'),
            "entry_score": 0.0, "stable_platform_price": None,
            "entry_signal": False, "is_risk_warning": False,
            "exit_signal_code": 0, "exit_severity_level": 0, "exit_signal_reason": None,
            "is_pullback_setup": bool(row.get('SETUP_SCORE_PLATFORM_SUPPORT_PULLBACK', 0) > 0),
            "triggered_playbooks": [], "triggered_playbooks_cn": [], 
            "context_snapshot": {
                'close': row.get(f'close_{timeframe}'),
                'entry_score': row.get('entry_score', 0.0),
                'risk_score': row.get('risk_score', 0.0),
            }
        }

    def _fill_risk_details(self, record: Dict, row: pd.Series, risk_playbook_map: Dict):
        """【V202.7 辅助函数】为卖出或预警信号填充风险详情和量化指标"""
        timestamp = record['trade_time']
        triggered_risks_en = []
        risk_details_df = getattr(self, '_last_risk_details_df', pd.DataFrame())
        if not risk_details_df.empty and timestamp in risk_details_df.index:
            risk_details_for_day = risk_details_df.loc[timestamp]
            active_risks = risk_details_for_day[risk_details_for_day > 0].index
            triggered_risks_en = [risk.replace('RISK_SCORE_', '') for risk in active_risks]
        
        triggered_risks_cn = [risk_playbook_map.get(risk, risk) for risk in triggered_risks_en]
        reason = ", ".join(triggered_risks_cn) if triggered_risks_cn else "综合风险评分超阈值"
        
        record['exit_signal_reason'] = reason
        record['triggered_playbooks'] = triggered_risks_en
        record['triggered_playbooks_cn'] = triggered_risks_cn
        
        context = record['context_snapshot']
        context['cost_slope_5d'] = row.get('SLOPE_5_CHIP_peak_cost_D')
        context['winner_rate_slope_5d'] = row.get('SLOPE_5_CHIP_total_winner_rate_D')
        context['peak_stability_slope_5d'] = row.get('SLOPE_5_CHIP_peak_stability_D')
        context['pressure_above_slope_5d'] = row.get('SLOPE_5_CHIP_pressure_above_D')
        record['context_snapshot'] = sanitize_for_json(context)

    def generate_intraday_alerts(self, daily_df: pd.DataFrame, minute_df: pd.DataFrame, params: dict) -> List[Dict]:
        """
        【V104 终局 · 执行版】
        - 核心革命: 建立一个独立的“战术执行层”，将日线策略的“战略确认”转化为分钟级别的“实时警报”。
        - 工作流程:
          1. 接收已经运行完V103日线策略的 `daily_df`。
          2. 识别出所有“冲高日”(Upthrust Day)。
          3. 在其后的“戒备日”(Alert Day)中，监控分钟线数据。
          4. 将日线级别的确认条件，“翻译”成实时价格的触发条件。
          5. 在条件满足的第一分钟，生成并返回警报。
        """
        print("\n" + "="*60)
        print("====== 开始执行【战术执行层 V104】(分钟级实时警报) ======")

        # --- 步骤0: 加载执行层参数 ---
        exec_params = self._get_params_block(params, 'intraday_execution_params', {})
        if not self._get_param_value(exec_params.get('enabled'), False):
            print("    - [信息] 战术执行层被禁用，跳过。")
            return []

        # --- 步骤1: 从日线策略中，识别出所有的“冲高日” ---
        # 我们需要重新计算V103的冲高日逻辑，以确保独立性
        p_attack = self.exit_strategy_params.get('upthrust_distribution_params', {})
        lookback_period = self._get_param_value(p_attack.get('upthrust_lookback_days'), 5)
        is_upthrust_day = daily_df['high_D'] > daily_df['high_D'].shift(1).rolling(window=lookback_period).max()
        
        upthrust_days_df = daily_df[is_upthrust_day]
        if upthrust_days_df.empty:
            print("    - [信息] 在日线数据中未发现任何“冲高日”，无需启动盘中监控。")
            return []
        
        print(f"    - [战略确认] 发现 {len(upthrust_days_df)} 个潜在的“冲高日”，将在次日启动盘中监控。")

        alerts = []
        # --- 步骤2: 遍历每一个“冲高日”，监控其后一天的分钟行情 ---
        for upthrust_date, upthrust_row in upthrust_days_df.iterrows():
            alert_date = upthrust_date + pd.Timedelta(days=1)
            
            # 获取“戒备日”当天的分钟数据
            alert_day_minute_df = minute_df[minute_df.index.date == alert_date.date()].copy()
            if alert_day_minute_df.empty:
                continue

            # --- 步骤3: 准备实时触发的阈值 ---
            # 阈值1: 戒备日当天的开盘价
            alert_day_open = alert_day_minute_df.iloc[0]['open_M']
            # 阈值2: 冲高日当天的开盘价
            upthrust_day_open = upthrust_row['open_D']

            print(f"      -> [进入戒备] 日期: {alert_date.date()} | 监控启动...")
            print(f"         - 触发阈值1 (低于今日开盘): {alert_day_open:.2f}")
            print(f"         - 触发阈值2 (低于昨日开盘): {upthrust_day_open:.2f}")

            # --- 步骤4: 在分钟线上应用“翻译”后的触发条件 ---
            triggered_minutes = alert_day_minute_df[
                (alert_day_minute_df['close_M'] < alert_day_open) &
                (alert_day_minute_df['close_M'] < upthrust_day_open)
            ]

            if not triggered_minutes.empty:
                # 获取第一次触发警报的分钟
                first_alert_minute = triggered_minutes.iloc[0]
                alert_time = first_alert_minute.name
                alert_price = first_alert_minute['close_M']
                
                alert_info = {
                    "alert_time": alert_time,
                    "alert_price": alert_price,
                    "alert_type": "INTRADAY_UPTHRUST_REJECTION",
                    "cn_name": "【盘中警报】冲高回落结构确认",
                    "reason": f"价格({alert_price:.2f})跌破今日开盘({alert_day_open:.2f})与昨日开盘({upthrust_day_open:.2f})",
                    "daily_setup_date": upthrust_date.date()
                }
                alerts.append(alert_info)
                print(f"         - [警报触发!] 时间: {alert_time.time()} | 价格: {alert_price:.2f} | 原因: {alert_info['reason']}")
        
        print("====== 【战术执行层 V104】执行完毕 ======")
        print("="*60 + "\n")
        return alerts

    # 最终得分调整层
    def _apply_final_score_adjustments(self, df: pd.DataFrame, raw_scores: pd.Series, params: dict, atomic_states: Dict[str, pd.Series]) -> Tuple[pd.Series, pd.DataFrame]:
        """
        【V119.4 自适应阈值版】最终得分调整层
        - 核心升级: 引入“自适应绝对阈值”机制。对于波动率这类指标，不再使用固定阈值，
                    而是根据其自身的长期中位数动态计算阈值（例如，低于长期中位数的50%）。
                    这使得策略能更好地适应不同股票和不同市场周期的波动特性。
        """
        print("    - [指挥棒模型 V119.4 自适应阈值版] 启动，开始对原始总分进行最终调整...")
        
        adjustment_params = self._get_params_block(params, 'final_score_adjustments', {})
        if not self._get_param_value(adjustment_params.get('enabled'), False):
            print("      -> 最终得分调整被禁用，返回原始分数。")
            return raw_scores, pd.DataFrame(index=df.index)

        total_multiplier = pd.Series(0.0, index=df.index)
        adjustment_details_df = pd.DataFrame(index=df.index)
        multiplier_rules = adjustment_params.get('multipliers', [])
        default_series = pd.Series(False, index=df.index)

        for rule in multiplier_rules:
            name = rule.get('name')
            cn_name = rule.get('cn_name', name)
            source_col = rule.get('source_column')
            condition = rule.get('condition')
            multiplier_value = rule.get('multiplier_value', 0)

            mask = pd.Series(False, index=df.index)
            if condition == 'is_lowest_percentile':
                if source_col not in df.columns:
                    print(f"      -> [警告] 指挥棒规则'{cn_name}'所需的列'{source_col}'在DataFrame中不存在，跳过。")
                    continue
                
                # 保险一：相对条件 (近期收缩)
                window = rule.get('window', 250)
                percentile = rule.get('percentile', 0.1)
                threshold_relative = df[source_col].rolling(window=window, min_periods=window//2).quantile(percentile)
                mask_relative = df[source_col] < threshold_relative
                
                # 保险二：绝对条件 (现在支持固定或自适应)
                mask_absolute = pd.Series(True, index=df.index) # 默认绝对条件为真
                
                # 检查是否有自适应阈值配置
                adaptive_config = rule.get('adaptive_threshold_config', {})
                if self._get_param_value(adaptive_config.get('enabled'), False):
                    median_window = self._get_param_value(adaptive_config.get('median_window'), 250)
                    multiplier = self._get_param_value(adaptive_config.get('multiplier'), 0.5)
                    
                    # 计算长期中位数作为波动中枢
                    long_term_median = df[source_col].rolling(window=median_window, min_periods=median_window//2).median()
                    # 计算自适应绝对阈值
                    threshold_absolute_adaptive = long_term_median * multiplier
                    
                    mask_absolute = df[source_col] < threshold_absolute_adaptive
                    print(f"      -> 规则'{cn_name}'启用自适应双重保险: 相对分位 & 自适应阈值(<长期中枢*{multiplier})")

                # 如果没有自适应配置，则检查固定阈值 (保持旧逻辑兼容性)
                elif rule.get('absolute_threshold') is not None:
                    absolute_threshold = rule.get('absolute_threshold')
                    mask_absolute = df[source_col] < absolute_threshold
                    print(f"      -> 规则'{cn_name}'启用固定双重保险: 相对分位 & 固定阈值(<{absolute_threshold})")

                # 最终的掩码是两个条件的交集
                mask = mask_relative & mask_absolute
            
            elif condition == 'is_true':
                if source_col not in atomic_states:
                    print(f"      -> [警告] 指挥棒规则'{cn_name}'所需的事件'{source_col}'在atomic_states中不存在，跳过。")
                    continue
                mask = atomic_states.get(source_col, default_series)

            # 应用乘数
            total_multiplier.loc[mask] += multiplier_value
            adjustment_details_df.loc[mask, name] = multiplier_value
            
            if mask.any():
                print(f"      -> 指挥棒规则 '{cn_name}' 已激活，在 {mask.sum()} 天提供了 {multiplier_value*100:.0f}% 的质量乘数。")

        adjusted_score = raw_scores * (1 + total_multiplier)
        print("    - [指挥棒模型 V119.4] 调整完成。")
        return adjusted_score, adjustment_details_df

    # ▼▼▼ 剧本的静态“蓝图”知识库 ▼▼▼
    def _get_playbook_blueprints(self) -> List[Dict]:
        """
        【V83.0 新增】剧本蓝图知识库
        - 职责: 定义所有剧本的静态属性（名称、家族、评分规则等）。
        - 特性: 这是一个纯粹的数据结构，不依赖任何动态数据，可以在初始化时被安全地缓存。
        """
        return [
            # --- 反转/逆势家族 (REVERSAL_CONTRARIAN) ---
            {
                'name': 'ABYSS_GAZE_S', 'cn_name': '【S级】深渊凝视', 'family': 'REVERSAL_CONTRARIAN',
                'type': 'setup', 'score': 320, 'side': 'left', 'comment': 'S级: 市场极度恐慌后的第一个功能性反转。'
            },
            {
                'name': 'CAPITULATION_PIT_REVERSAL', 'cn_name': '【动态】投降坑反转', 'family': 'REVERSAL_CONTRARIAN',
                'type': 'setup_score', 'side': 'left', 'comment': '根据坑的深度和反转K强度动态给分。', 'allow_memory': False,
                'scoring_rules': { 'min_setup_score_to_trigger': 51, 'base_score': 160, 'score_multiplier': 1.0, 'trigger_bonus': {'TRIGGER_REVERSAL_CONFIRMATION_CANDLE': 50}}
            },
            {
                'name': 'CAPITAL_DIVERGENCE_REVERSAL', 'cn_name': '【A-级】资本逆行者', 'family': 'REVERSAL_CONTRARIAN',
                'type': 'setup', 'score': 230, 'side': 'left', 'comment': 'A-级: 在“底背离”机会窗口中出现的反转确认。'
            },
            {
                'name': 'BEAR_TRAP_RALLY', 'cn_name': '【C+级】熊市反弹', 'family': 'REVERSAL_CONTRARIAN',
                'type': 'setup', 'score': 180, 'side': 'left', 'comment': 'C+级: 熊市背景下，价格首次钝化，并由长期趋势企稳信号触发。'
            },
            {
                'name': 'WASHOUT_REVERSAL_A', 'cn_name': '【A级】巨阴洗盘反转', 'family': 'REVERSAL_CONTRARIAN',
                'type': 'setup', 'score': 260, 'side': 'left', 'comment': 'A级: 极端洗盘后的拉升前兆。', 'allow_memory': False
            },
            {
                'name': 'BOTTOM_STABILIZATION_B', 'cn_name': '【B级】底部企稳', 'family': 'REVERSAL_CONTRARIAN',
                'type': 'setup', 'score': 190, 'side': 'left', 'comment': 'B级: 股价严重超卖偏离均线后，出现企稳阳线。'
            },
            # --- 趋势/动能家族 (TREND_MOMENTUM) ---
            {
                'name': 'CHIP_PLATFORM_PULLBACK', 'cn_name': '【S-级】筹码平台回踩', 'family': 'TREND_MOMENTUM',
                'type': 'setup', 'score': 330, 'side': 'right', 
                'comment': 'S-级: 股价回踩由筹码峰形成的、位于趋势线上方的稳固平台，并获得支撑。这是极高质量的“空中加油”信号。'
            },
            {
                'name': 'TREND_EMERGENCE_B_PLUS', 'cn_name': '【B+级】右侧萌芽', 'family': 'TREND_MOMENTUM',
                'type': 'setup', 'score': 210, 'comment': 'B+级: 填补战术真空。在左侧反转后，短期均线走好但尚未站上长线时的首次介入机会。'
            },
            {
                'name': 'DEEP_ACCUMULATION_BREAKOUT', 'cn_name': '【动态】潜龙出海', 'family': 'TREND_MOMENTUM',
                'type': 'setup_score', 'side': 'right', 'comment': '根据深度吸筹分数动态给分。若伴随筹码点火，则确定性更高。',
                'scoring_rules': { 'min_setup_score_to_trigger': 51, 'base_score': 200, 'score_multiplier': 1.5, 'trigger_bonus': {'CHIP_EVENT_IGNITION': 60} }
            },
            {
                'name': 'ENERGY_COMPRESSION_BREAKOUT', 'cn_name': '【动态】能量压缩突破', 'family': 'TREND_MOMENTUM',
                'type': 'precondition_score', 'side': 'right', 'comment': '根据当天前提和历史准备共同给分。若伴随筹码点火，则确定性更高。',
                'scoring_rules': { 'base_score': 150, 'min_score_to_trigger': 180, 'conditions': {'VOL_STATE_SQUEEZE_WINDOW': 50, 'CHIP_STATE_CONCENTRATION_SQUEEZE': 30, 'MA_STATE_CONVERGING': 20}, 'setup_bonus': {'ENERGY_COMPRESSION': 0.2, 'HEALTHY_MARKUP': 0.1}, 'trigger_bonus': {'CHIP_EVENT_IGNITION': 60} }
            },
            {
                'name': 'PLATFORM_SUPPORT_PULLBACK', 
                'cn_name': '【动态】多维支撑回踩', 
                'family': 'TREND_MOMENTUM',
                'type': 'setup_score', 
                'side': 'right', 
                'comment': '根据平台质量(筹码结构、趋势背景)和支撑级别(均线/筹码峰)进行动态评分。',
                'scoring_rules': {
                    'min_setup_score_to_trigger': 50,  # 要求平台质量分至少达到50
                    'base_score': 180,                 # 给予一个稳健的基础分
                    'score_multiplier': 1.2            # 允许根据平台质量分进行加成
                }
            },
            {
                'name': 'HEALTHY_BOX_BREAKOUT', 'cn_name': '【A-级】健康箱体突破', 'family': 'TREND_MOMENTUM',
                'type': 'setup', 'score': 220, 'side': 'right', 'comment': 'A-级: 在一个位于趋势均线上方的健康箱体内盘整后，发生的向上突破。'
            },
            {
                'name': 'HEALTHY_MARKUP_A', 'cn_name': '【A级】健康主升浪', 'family': 'TREND_MOMENTUM',
                'type': 'setup_score', 'side': 'right', 'comment': 'A级: 在主升浪中回踩或延续，根据主升浪健康分动态加成。',
                'scoring_rules': { 
                    'min_setup_score_to_trigger': 60, # 要求主升浪健康分至少达到60
                    'base_score': 240, 
                    'score_multiplier': 1.2, # 允许根据健康分进行加成
                    'conditions': { 'MA_STATE_DIVERGING': 20, 'OSC_STATE_MACD_BULLISH': 15, 'MA_STATE_PRICE_ABOVE_SHORT_MA': 0 } 
                }
            },
            {
                'name': 'N_SHAPE_CONTINUATION_A', 'cn_name': '【A级】N字板接力', 'family': 'TREND_MOMENTUM',
                'type': 'precondition_score', 'side': 'right', 'comment': 'A级: 强势股的经典趋势中继信号。若伴随筹码点火，则确定性更高。',
                'scoring_rules': { 'base_score': 250, 'min_score_to_trigger': 250, 'conditions': {'SETUP_SCORE_N_SHAPE_CONTINUATION_ABOVE_80': 0}, 'trigger_bonus': {'CHIP_EVENT_IGNITION': 60} }
            },
            {
                'name': 'GAP_SUPPORT_PULLBACK_B_PLUS', 'cn_name': '【B+级】缺口支撑回踩', 'family': 'TREND_MOMENTUM',
                'type': 'setup', 'score': 190, 'side': 'right', 'comment': 'B+级: 回踩前期跳空缺口获得支撑并反弹。'
            },
            # --- 特殊事件家族 (SPECIAL_EVENT) ---
            {
                'name': 'EARTH_HEAVEN_BOARD', 'cn_name': '【S+】地天板', 'family': 'SPECIAL_EVENT',
                'type': 'event_driven', 'score': 380, 'side': 'left', 'comment': '市场情绪的极致反转。'
            },
            {
                'name': 'FAULT_REBIRTH_S', 
                'cn_name': '【S级】断层新生', 
                'family': 'SPECIAL_EVENT',
                'type': 'event_driven', 
                'score': 350, 
                'side': 'left', 
                'comment': 'S级: 识别因成本断层导致的筹码结构重置，是极高价值的特殊事件信号。'
            },
        ]

    # ▼▼▼ 此方法现在是“水合”引擎 ▼▼▼
    def _get_playbook_definitions(self, df: pd.DataFrame, trigger_events: Dict[str, pd.Series], setup_scores: Dict[str, pd.Series], atomic_states: Dict[str, pd.Series]) -> List[Dict]:
        """
        【V127.0 统一交战规则版】
        - 核心重构: 不再对每个右侧剧本单独添加审查逻辑，而是在函数末尾设立一个
                    “最终审查关卡”。该关卡会强制为所有 'side: right' 的剧本的触发器
                    增加 '& ~is_trend_deteriorating' 的战略安全审查。
        - 收益: 这是一个系统性的、无死角的解决方案。它确保了未来无论增加何种类型的
                右侧剧本，都会被自动纳入战略审查体系，从根本上杜绝了“逻辑短路”和
                “审计遗漏”的问题。
        """
        print("    - [剧本水合引擎 V127.0 统一交战规则版] 启动...")
        
        # 步骤1: 深度复制蓝图，以防修改缓存的原始版本
        hydrated_playbooks = deepcopy(self.playbook_blueprints)
        
        # 步骤2: 准备所有需要用到的动态数据
        default_series = pd.Series(False, index=df.index)
        
        # 从中央情报库获取最高指令
        is_trend_deteriorating = atomic_states.get('CONTEXT_TREND_DETERIORATING', default_series)
        
        # 准备分
        score_cap_pit = setup_scores.get('SETUP_SCORE_CAPITULATION_PIT', pd.Series(0, index=df.index))
        score_deep_accum = setup_scores.get('SETUP_SCORE_DEEP_ACCUMULATION', pd.Series(0, index=df.index))
        score_nshape_cont = setup_scores.get('SETUP_SCORE_N_SHAPE_CONTINUATION', pd.Series(0, index=df.index))
        score_gap_support = setup_scores.get('SETUP_SCORE_GAP_SUPPORT_PULLBACK', pd.Series(0, index=df.index))
        score_bottoming_process = setup_scores.get('SETUP_SCORE_BOTTOMING_PROCESS', pd.Series(0, index=df.index))
        score_healthy_markup = setup_scores.get('SETUP_SCORE_HEALTHY_MARKUP', pd.Series(0, index=df.index))
        # 【新增】获取平台质量分
        score_platform_quality = setup_scores.get('SETUP_SCORE_PLATFORM_QUALITY', pd.Series(0, index=df.index))

        # 原子状态
        capital_divergence_window = atomic_states.get('CAPITAL_STATE_DIVERGENCE_WINDOW', default_series)
        setup_bottom_passivation = atomic_states.get('MA_STATE_BOTTOM_PASSIVATION', default_series)
        setup_washout_reversal = atomic_states.get('KLINE_STATE_WASHOUT_WINDOW', default_series)
        setup_healthy_box = atomic_states.get('BOX_STATE_HEALTHY_CONSOLIDATION', default_series)
        recent_reversal_context = atomic_states.get('CONTEXT_RECENT_REVERSAL_SIGNAL', default_series)
        ma_short_slope_positive = atomic_states.get('MA_STATE_SHORT_SLOPE_POSITIVE', default_series)
        is_trend_healthy = atomic_states.get('CONTEXT_OVERALL_TREND_HEALTHY', default_series)
        
        # 动态布尔条件
        atomic_states['SETUP_SCORE_N_SHAPE_CONTINUATION_ABOVE_80'] = score_nshape_cont > 80
        atomic_states['SETUP_SCORE_HEALTHY_MARKUP_ABOVE_60'] = score_healthy_markup > 60

        # --- 步骤3: 为每个剧本注入其专属的动态数据 (水合过程) ---
        for playbook in hydrated_playbooks:
            name = playbook['name']
            
            # 左侧剧本逻辑
            if name == 'ABYSS_GAZE_S':
                playbook['trigger'] = trigger_events.get('TRIGGER_DOMINANT_REVERSAL', default_series)
                playbook['setup'] = score_cap_pit > 80
            elif name == 'CAPITULATION_PIT_REVERSAL':
                playbook['setup_score_series'] = score_cap_pit
                playbook['trigger'] = (trigger_events.get('TRIGGER_REVERSAL_CONFIRMATION_CANDLE', default_series) | trigger_events.get('TRIGGER_BREAKOUT_CANDLE', default_series))
            elif name == 'CAPITAL_DIVERGENCE_REVERSAL':
                playbook['setup'] = capital_divergence_window
                playbook['trigger'] = trigger_events.get('TRIGGER_REVERSAL_CONFIRMATION_CANDLE', default_series)
            elif name == 'BEAR_TRAP_RALLY':
                playbook['setup'] = setup_bottom_passivation
                playbook['trigger'] = trigger_events.get('TRIGGER_TREND_STABILIZING', default_series)
            elif name == 'WASHOUT_REVERSAL_A':
                playbook['setup'] = setup_washout_reversal
                playbook['trigger'] = trigger_events.get('TRIGGER_BREAKOUT_CANDLE', default_series)
            elif name == 'BOTTOM_STABILIZATION_B':
                playbook['setup'] = score_bottoming_process > 50
                playbook['trigger'] = trigger_events.get('TRIGGER_BREAKOUT_CANDLE', default_series)
            
            # 右侧剧本逻辑 (只分配基础的setup和trigger，不在此处加审查)
            elif name == 'TREND_EMERGENCE_B_PLUS':
                playbook['setup'] = recent_reversal_context & ma_short_slope_positive
                playbook['trigger'] = trigger_events.get('TRIGGER_TREND_CONTINUATION_CANDLE', default_series)
            
            elif name == 'PLATFORM_SUPPORT_PULLBACK':
                playbook['setup_score_series'] = score_platform_quality
                trigger_ma_rebound = trigger_events.get('TRIGGER_PULLBACK_REBOUND', default_series)
                trigger_chip_rebound = trigger_events.get('TRIGGER_PLATFORM_PULLBACK_REBOUND', default_series)
                playbook['trigger'] = trigger_ma_rebound | trigger_chip_rebound

            elif name == 'HEALTHY_MARKUP_A':
                playbook['setup_score_series'] = score_healthy_markup
                trigger_rebound = trigger_events.get('TRIGGER_PULLBACK_REBOUND', default_series)
                trigger_continuation = trigger_events.get('TRIGGER_TREND_CONTINUATION_CANDLE', default_series)
                playbook['trigger'] = trigger_rebound | trigger_continuation
            
            elif name == 'HEALTHY_BOX_BREAKOUT':
                playbook['setup'] = setup_healthy_box
                playbook['trigger'] = trigger_events.get('BOX_EVENT_BREAKOUT', default_series)
            
            elif name == 'GAP_SUPPORT_PULLBACK_B_PLUS':
                playbook['setup'] = (score_gap_support > 60)
                playbook['trigger'] = trigger_events.get('TRIGGER_PULLBACK_REBOUND', default_series)
            
            elif name == 'CHIP_PLATFORM_PULLBACK':
                setup_platform_formed = atomic_states.get('PLATFORM_STATE_STABLE_FORMED', default_series)
                playbook['setup'] = setup_platform_formed & is_trend_healthy
                playbook['trigger'] = trigger_events.get('TRIGGER_PLATFORM_PULLBACK_REBOUND', default_series)

            elif name == 'ENERGY_COMPRESSION_BREAKOUT':
                playbook['trigger'] = trigger_events.get('TRIGGER_BREAKOUT_CANDLE', default_series)
            
            elif name == 'DEEP_ACCUMULATION_BREAKOUT':
                playbook['setup_score_series'] = score_deep_accum
                playbook['trigger'] = trigger_events.get('TRIGGER_BREAKOUT_CANDLE', default_series)
            
            elif name == 'N_SHAPE_CONTINUATION_A':
                playbook['trigger'] = trigger_events.get('TRIGGER_N_SHAPE_BREAKOUT', default_series)
            
            elif name == 'EARTH_HEAVEN_BOARD':
                playbook['trigger'] = trigger_events.get('TRIGGER_EARTH_HEAVEN_BOARD', default_series)

        # --- 步骤4: 【核心修正】统一交战规则 - 为所有右侧剧本强制增加战略审查 ---
        # print("      -> 正在执行“统一交战规则”最终审查...")
        for playbook in hydrated_playbooks:
            if playbook.get('side') == 'right':
                # 从剧本中获取已经分配好的原始触发器
                original_trigger = playbook.get('trigger', default_series)
                # 将原始触发器与“趋势未恶化”的最高指令进行“与”运算，生成最终的安全触发器
                playbook['trigger'] = original_trigger & ~is_trend_deteriorating
        print("      -> “统一交战规则”审查完毕，所有右侧进攻性操作已被置于战略监控之下。")

        # print(f"    - [剧本水合引擎 V127.0] 完成。")
        return hydrated_playbooks

    def _calculate_entry_score(
        self, 
        df: pd.DataFrame, 
        params: dict, 
        trigger_events: Dict[str, pd.Series], 
        setup_scores: Dict[str, pd.Series],
        atomic_states: Dict[str, pd.Series]
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        【V184.0 重构优化版】
        - 核心重构: 将原有的复杂计分逻辑拆分为三个独立的辅助函数：
                    _calculate_base_scores: 计算基础分（阵地分+动能分+触发分）。
                    _apply_weekly_context_modifiers: 应用周线战略背景乘数。
                    _apply_final_score_amplifier: 应用最终的“优势火力”放大器。
        - 收益: 计分流程的每一步都清晰可见，极大提高了代码的可读性和可维护性。
        """
        print("    - [计分引擎 V184.0 重构优化版] 启动...")
        
        score_details_df = pd.DataFrame(index=df.index)
        scoring_params = self.scoring_params

        # 步骤1: 计算基础分的三大组成部分 (阵地、动能、触发)
        # ▼▼▼【代码修改】现在返回三个独立的分数组件 ▼▼▼
        positional_score, dynamic_score, trigger_score, score_details_df = self._calculate_base_scores(
            df, scoring_params, atomic_states, trigger_events, score_details_df
        )
        # ▲▲▲【代码修改】▲▲▲

        # 步骤2: 应用周线战略背景修正，得到加权后的基础分
        # ▼▼▼【代码修改】将三个分数组件传入进行修正和加权 ▼▼▼
        modified_base_score, score_details_df = self._apply_weekly_context_modifiers(
            df, positional_score, dynamic_score, trigger_score, scoring_params, atomic_states, score_details_df
        )
        # ▲▲▲【代码修改】▲▲▲

        # 步骤3: 应用最终的“优势火力”放大器
        final_score, score_details_df = self._apply_final_score_amplifier(
            df, modified_base_score, scoring_params, score_details_df
        )
        
        df['entry_score'] = final_score.round(0)
        score_details_df.fillna(0, inplace=True)
        print(f"--- [计分引擎 V184.0] 计算完成。最终有 { (final_score > 0).sum() } 个交易日产生得分。 ---")
        return df, score_details_df

    def _calculate_base_scores(self, df: pd.DataFrame, scoring_params: dict, atomic_states: dict, trigger_events: dict, score_details_df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series, pd.DataFrame]:
        """
        【新增辅助方法 - 优化版】
        - 核心修改: 此方法现在只计算并返回三个最原始的分数组件（阵地分、动能分、触发分），不再进行加权。
        - 收益: 职责更单一，便于后续步骤进行更复杂的修正。
        """
        print("      -> [火力层1-3/4] 正在计算“阵地分”、“动能分”与“触发分”...")
        default_series = pd.Series(False, index=df.index)
        
        # 计算阵地分
        positional_params = scoring_params.get('positional_scoring', {})
        total_positional_score = pd.Series(0.0, index=df.index)
        for state_name, score in {**positional_params.get('positive_signals', {}), **positional_params.get('negative_signals', {})}.items():
            mask = atomic_states.get(state_name, default_series)
            total_positional_score.loc[mask] += score
            score_details_df.loc[mask, state_name] = score
            
        # 计算动能分
        dynamic_params = scoring_params.get('dynamic_scoring', {})
        total_dynamic_score = pd.Series(0.0, index=df.index)
        for state_name, score in {**dynamic_params.get('positive_signals', {}), **dynamic_params.get('negative_signals', {})}.items():
            mask = atomic_states.get(state_name, default_series)
            total_dynamic_score.loc[mask] += score
            score_details_df.loc[mask, state_name] = score

        # 计算触发分
        trigger_score_total = pd.Series(0.0, index=df.index)
        trigger_event_scores = scoring_params.get('trigger_events', {})
        for event_name, score in trigger_event_scores.items():
            if event_name == '说明': continue
            mask = trigger_events.get(event_name, default_series)
            trigger_score_total.loc[mask] += score
            score_details_df.loc[mask, event_name] = score
        
        return total_positional_score, total_dynamic_score, trigger_score_total, score_details_df

    def _apply_weekly_context_modifiers(self, df: pd.DataFrame, positional_score: pd.Series, dynamic_score: pd.Series, trigger_score: pd.Series, scoring_params: dict, atomic_states: dict, score_details_df: pd.DataFrame) -> Tuple[pd.Series, pd.DataFrame]:
        """
        【新增辅助方法 - 完整实现版】
        - 核心修复: 提供了完整的周线战略背景修正逻辑，并定义了缺失的 `default_series`。
        - 工作流程:
          1. 根据周线信号（如主升浪、底部企稳等）计算出各类分数的修正乘数。
          2. 将原始分数组件拆分为正分和负分。
          3. 对正分应用加成乘数，对负分应用豁免乘数。
          4. 将修正后的分数重新加权混合，并叠加上修正后的触发分，得到最终的修正后基础分。
        """
        print("      -> [火力层3.5/4] 正在应用“周线精确制导”...")
        # ▼▼▼【代码修改】修复点：定义 default_series ▼▼▼
        default_series = pd.Series(False, index=df.index)
        # ▲▲▲【代码修改】▲▲▲

        # 初始化各类分数的乘数，默认为1.0 (无影响)
        positional_multiplier = pd.Series(1.0, index=df.index)
        dynamic_multiplier = pd.Series(1.0, index=df.index)
        trigger_multiplier = pd.Series(1.0, index=df.index)
        penalty_immunity_multiplier = pd.Series(1.0, index=df.index) # 用于抵消负分的豁免乘数

        # 指令1: 如果周线处于“主升浪”或“经典突破”的背景下
        main_ascent_mask = df.get('state_node_main_ascent_W', default_series)
        breakout_mask = df.get('playbook_classic_breakout_W', default_series)
        momentum_context_mask = main_ascent_mask | breakout_mask
        if momentum_context_mask.any():
            print(f"        -> [周线指令] “主升浪/突破”背景激活，放大动能和触发权重。")
            dynamic_multiplier.loc[momentum_context_mask] *= 1.5
            trigger_multiplier.loc[momentum_context_mask] *= 1.8
            score_details_df.loc[momentum_context_mask, 'CONTEXT_MULT_MOMENTUM'] = 1.5
        
        # 指令2: 如果周线处于“点火期”或“底部企稳”的背景下
        ignition_mask = df.get('state_node_ignition_W', default_series)
        bottoming_mask = df.get('state_node_bottoming_W', default_series)
        reversal_context_mask = ignition_mask | bottoming_mask
        if reversal_context_mask.any():
            print(f"        -> [周线指令] “点火/筑底”背景激活，放大阵地权重并豁免部分风险。")
            positional_multiplier.loc[reversal_context_mask] *= 1.6
            penalty_immunity_multiplier.loc[reversal_context_mask] = 0.0 
            score_details_df.loc[reversal_context_mask, 'CONTEXT_MULT_POSITIONAL'] = 1.6

        # 指令3: 如果周线处于“洗盘豁免期”
        washout_immunity_mask = df.get('state_washout_immunity_W', default_series)
        if washout_immunity_mask.any():
            print(f"        -> [周线指令] “洗盘豁免”激活，大幅降低负面信号影响。")
            penalty_immunity_multiplier.loc[washout_immunity_mask] = 0.2 
            score_details_df.loc[washout_immunity_mask, 'CONTEXT_IMMUNITY_WASHOUT'] = 0.2

        # --- 应用乘数 ---
        # 分离正负分
        pos_positional = positional_score.where(positional_score > 0, 0)
        neg_positional = positional_score.where(positional_score < 0, 0)
        pos_dynamic = dynamic_score.where(dynamic_score > 0, 0)
        neg_dynamic = dynamic_score.where(dynamic_score < 0, 0)

        # 应用乘数
        adj_pos_positional = pos_positional * positional_multiplier
        adj_neg_positional = neg_positional * penalty_immunity_multiplier # 对负分应用豁免乘数
        adj_pos_dynamic = pos_dynamic * dynamic_multiplier
        adj_neg_dynamic = neg_dynamic * penalty_immunity_multiplier
        adj_trigger = trigger_score * trigger_multiplier

        # 重新合成调整后的分数组件
        adj_total_positional = adj_pos_positional + adj_neg_positional
        adj_total_dynamic = adj_pos_dynamic + adj_neg_dynamic
        
        # --- 使用调整后的分数进行混合加权 ---
        weights = scoring_params.get('hybrid_scoring_weights', {})
        weight_pos = weights.get('positional_weight', 0.4)
        weight_dyn = weights.get('dynamic_weight', 0.6)
        weighted_base_score = (adj_total_positional * weight_pos) + (adj_total_dynamic * weight_dyn)
        
        # 叠加上调整后的触发分，得到最终的修正后基础分
        modified_base_score = weighted_base_score + adj_trigger
        
        return modified_base_score, score_details_df

    def _apply_final_score_amplifier(self, df: pd.DataFrame, modified_base_score: pd.Series, scoring_params: dict, score_details_df: pd.DataFrame) -> Tuple[pd.Series, pd.DataFrame]:
        """
        【新增辅助方法】应用最终的“优势火力”放大器。
        """
        print("      -> [火力层4/4] 正在启动“优势火力”放大器...")
        amp_params = scoring_params.get('chip_dynamics_amplifier', {})
        if not amp_params.get('enabled', False):
            return modified_base_score, score_details_df

        window = self._get_param_value(amp_params.get('lookback_window'), 250)
        
        # 计算兵种1(筹码)的乘数
        conc_slope_col = 'CHIP_concentration_90pct_slope_5d_D'
        conc_multiplier = pd.Series(1.0, index=df.index)
        if conc_slope_col in df.columns:
            conc_rank = df[conc_slope_col].rolling(window=window, min_periods=window//2).rank(pct=True)
            rules = amp_params.get('concentration_slope_rules', {}).get('tiers', [])
            for rule in sorted(rules, key=lambda x: x['percentile_upper'], reverse=True):
                conc_multiplier[conc_rank <= rule['percentile_upper']] = rule['multiplier']
        
        # 计算兵种2(成本)的乘数
        cost_slope_col = 'SLOPE_5_CHIP_peak_cost_D'
        cost_multiplier = pd.Series(1.0, index=df.index)
        if cost_slope_col in df.columns:
            cost_rank = df[cost_slope_col].rolling(window=window, min_periods=window//2).rank(pct=True)
            rules = amp_params.get('cost_basis_slope_rules', {}).get('tiers', [])
            for rule in sorted(rules, key=lambda x: x['percentile_lower'], reverse=False):
                cost_multiplier[cost_rank >= rule['percentile_lower']] = rule['multiplier']
        
        # 采用“优势火力”逻辑，取最大值并应用安全限制
        final_multiplier = pd.concat([conc_multiplier, cost_multiplier], axis=1).max(axis=1)
        cap_params = amp_params.get('final_multiplier_cap', {})
        max_mult = cap_params.get('max', 3.0)
        min_mult = cap_params.get('min', 0.4)
        final_multiplier.clip(lower=min_mult, upper=max_mult, inplace=True)
        
        score_details_df['FINAL_MULTIPLIER'] = final_multiplier
        print(f"        -> 火力放大器已激活。最终乘数范围: [{final_multiplier.min():.2f}, {final_multiplier.max():.2f}]")
        
        final_score = modified_base_score * final_multiplier
        return final_score, score_details_df

    def _calculate_entry_score_legacy(
        self, 
        df: pd.DataFrame, 
        params: dict, 
        trigger_events: Dict[str, pd.Series], 
        setup_scores: Dict[str, pd.Series],
        atomic_states: Dict[str, pd.Series]
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        【V162.0 新增】这是旧版的“剧本家族竞争”计分逻辑，现在被新引擎调用以计算降权的剧本分。
        """
        default_series = pd.Series(False, index=df.index)
        context_window = self._get_param_value(
            self._get_params_block(params, 'entry_scoring_params', {}).get('context_window'), 10
        )
        p_violent = self._get_params_block(params, 'trigger_event_params', {}).get('violent_reversal_trigger', {})
        pct_change_thresh = self._get_param_value(p_violent.get('min_pct_change'), 0.04)
        vol_multiplier = self._get_param_value(p_violent.get('volume_multiplier'), 1.8)
        is_strong_rally = df['pct_change_D'] > pct_change_thresh
        is_huge_volume = df['volume_D'] > df.get('VOL_MA_21_D', 0) * vol_multiplier
        trigger_violent_reversal = is_strong_rally & is_huge_volume
        persistence_days = self._get_param_value(p_violent.get('persistence_days'), 3)
        reversal_signal_int = trigger_violent_reversal.astype(int)
        CONTEXT_VIOLENT_REVERSAL_WINDOW = (reversal_signal_int.rolling(window=persistence_days, min_periods=1).max() == 1)
        atomic_states['CONTEXT_VIOLENT_REVERSAL_WINDOW'] = CONTEXT_VIOLENT_REVERSAL_WINDOW
        playbook_definitions = self._get_playbook_definitions(df, trigger_events, setup_scores, atomic_states)
        base_scores_df = pd.DataFrame(index=df.index)
        bonus_scores_df = pd.DataFrame(index=df.index)
        for playbook in playbook_definitions:
            name = playbook['name']
            rules = playbook.get('scoring_rules', {})
            playbook_type = playbook.get('type')
            original_trigger_mask = playbook.get('trigger', default_series)
            is_core_trend_playbook = name in ['HEALTHY_MARKUP_A', 'TREND_EMERGENCE_B_PLUS']
            contextual_trigger_mask = CONTEXT_VIOLENT_REVERSAL_WINDOW if is_core_trend_playbook else default_series
            trigger_mask = original_trigger_mask | contextual_trigger_mask
            side_mask = (df['close_D'] > df.get('EMA_55_D', -1)) if playbook.get('side') == 'right' else pd.Series(True, index=df.index)
            setup_mask = pd.Series(True, index=df.index)
            if playbook_type == 'setup':
                setup_mask = playbook.get('setup', default_series)
            elif playbook_type == 'setup_score':
                min_score_req = rules.get('min_setup_score_to_trigger', 0)
                if min_score_req > 0:
                    setup_score_series = playbook.get('setup_score_series', default_series)
                    allow_memory = playbook.get('allow_memory', True)
                    if allow_memory:
                        max_score_in_context = setup_score_series.rolling(window=context_window, min_periods=1).max()
                        setup_mask = (max_score_in_context >= min_score_req) | CONTEXT_VIOLENT_REVERSAL_WINDOW
                    else:
                        setup_mask = setup_score_series >= min_score_req
            elif playbook_type == 'precondition_score':
                min_score_req = rules.get('min_score_to_trigger', 0)
                precondition_score = pd.Series(0.0, index=df.index)
                precondition_score += sum(atomic_states.get(s, default_series).astype(int) * v for s, v in rules.get('conditions', {}).items())
                precondition_score += sum(setup_scores.get(f'SETUP_SCORE_{s}', default_series).rolling(window=context_window, min_periods=1).max().fillna(0) * v for s, v in rules.get('setup_bonus', {}).items())
                setup_mask = precondition_score >= min_score_req
            valid_mask = trigger_mask & side_mask & setup_mask
            base_score = rules.get('base_score', playbook.get('score', 0))
            base_scores_df[name] = valid_mask.astype(int) * base_score
            playbook_bonus = pd.Series(0.0, index=df.index)
            if rules:
                condition_bonus = sum(atomic_states.get(s, default_series).astype(int) * v for s, v in rules.get('conditions', {}).items())
                event_bonus = sum(atomic_states.get(s, default_series).astype(int) * v for s, v in rules.get('event_conditions', {}).items())
                setup_bonus = sum(setup_scores.get(f'SETUP_SCORE_{s}', default_series).rolling(window=context_window, min_periods=1).max().fillna(0) * v for s, v in rules.get('setup_bonus', {}).items())
                trigger_bonus = sum(trigger_events.get(s, default_series).astype(int) * v for s, v in rules.get('trigger_bonus', {}).items())
                setup_multiplier_bonus = pd.Series(0.0, index=df.index)
                if playbook_type == 'setup_score':
                    setup_score_series = playbook.get('setup_score_series', default_series)
                    max_setup_in_context = setup_score_series.rolling(window=context_window, min_periods=1).max()
                    multiplier = rules.get('score_multiplier', 1.0)
                    setup_multiplier_bonus = max_setup_in_context * (multiplier - 1) if multiplier > 1 else pd.Series(0.0, index=df.index)
                playbook_bonus = condition_bonus + event_bonus + setup_bonus + trigger_bonus + setup_multiplier_bonus
            bonus_scores_df[name] = playbook_bonus * valid_mask
        playbook_to_family = {p['name']: p.get('family', 'UNCATEGORIZED') for p in playbook_definitions}
        family_to_playbooks = {}
        for p_name, f_name in playbook_to_family.items():
            if f_name not in family_to_playbooks: family_to_playbooks[f_name] = []
            family_to_playbooks[f_name].append(p_name)
        final_family_scores = pd.DataFrame(0.0, index=df.index, columns=family_to_playbooks.keys())
        score_details_df = pd.DataFrame(0.0, index=df.index, columns=base_scores_df.columns)
        for family_name, playbook_names in family_to_playbooks.items():
            family_base_scores = base_scores_df[playbook_names]
            family_bonus_scores = bonus_scores_df[playbook_names]
            winner_base_scores = family_base_scores.max(axis=1)
            family_bonus_pool = family_bonus_scores.sum(axis=1)
            final_family_scores[family_name] = winner_base_scores + family_bonus_pool
            winner_playbook_names = family_base_scores.idxmax(axis=1)
            winner_playbook_names.name = 'winner_playbook' 
            has_score_mask = final_family_scores[family_name] > 0
            if has_score_mask.any():
                df_temp = pd.concat([winner_playbook_names[has_score_mask], final_family_scores.loc[has_score_mask, family_name]], axis=1)
                def assign_score(row):
                    winner_name = row['winner_playbook']
                    if pd.notna(winner_name):
                        score_details_df.loc[row.name, winner_name] = row[family_name]
                df_temp.apply(assign_score, axis=1)
        final_score = final_family_scores.sum(axis=1)
        df['entry_score'] = final_score.round(0)
        score_details_df.fillna(0, inplace=True)
        return df, score_details_df

    # ▼▼▼ 固化“黑匣子”探针为独立模块 ▼▼▼
    def _probe_entry_score_details(
        self, 
        df: pd.DataFrame, 
        probe_dates: List[str], 
        final_score: pd.Series, 
        intermediate_masks: Dict, 
        playbook_definitions: List[Dict], 
        trigger_violent_reversal: pd.Series, 
        setup_scores: Dict[str, pd.Series], 
        context_window: int,
        atomic_states: Dict[str, pd.Series] # 修正1: 增加缺失的 atomic_states 参数
    ):
        """
        【V152.1 “探针后勤补给”修正版】
        - 核心修正: 修复了因函数签名缺少 atomic_states 参数，以及函数内部未定义
                    default_series 而导致的 NameError 崩溃问题。
        """
        print("\n" + "="*25 + " [黑匣子探针启动] " + "="*25)
        
        # 修正2: 在函数内部定义其所需的 default_series
        default_series = pd.Series(False, index=df.index)

        for probe_date_str in probe_dates:
            try:
                probe_ts = pd.to_datetime(probe_date_str).tz_localize('UTC')
                if probe_ts not in df.index:
                    print(f"\n--- [探针信息] 日期 {probe_date_str} 不在当前数据帧的索引中，跳过。 ---")
                    continue
                
                print(f"\n\n--- [探针] 正在剖析日期: {probe_date_str} ---")
                
                print("\n[全局关键变量]")
                print(f"  - 当日收盘价 (close_D): {df.loc[probe_ts, 'close_D']:.2f}")
                ema55_col = 'EMA_55_D'
                if ema55_col in df.columns:
                    print(f"  - 55日均线 ({ema55_col}): {df.loc[probe_ts, ema55_col]:.2f}")
                else:
                    print(f"  - 55日均线 ({ema55_col}): 未找到")
                print(f"  - 暴力反转信号 (trigger_violent_reversal): {trigger_violent_reversal.loc[probe_ts]}")
                
                for playbook in playbook_definitions:
                    name = playbook['name']
                    p_type = playbook.get('type')
                    masks = intermediate_masks.get(name, {})
                    
                    print(f"\n--- 剖析剧本: [{name}] (类型: {p_type}) ---")
                    print(f"  - 触发条件 (trigger_mask): {masks.get('trigger_mask', default_series).loc[probe_ts]}")
                    print(f"  - 站位条件 (side_mask): {masks.get('side_mask', default_series).loc[probe_ts]}")
                    print(f"  - 准备条件 (setup_mask): {masks.get('setup_mask', default_series).loc[probe_ts]}")
                    
                    if p_type == 'setup_score':
                        rules = playbook.get('scoring_rules', {})
                        min_req = rules.get('min_setup_score_to_trigger', 0)
                        # 此处使用 playbook.get('setup_score_series', default_series) 是安全的，因为 playbook 来自 intermediate_masks
                        score_series = masks.get('setup_score_series', default_series)
                        max_score = score_series.rolling(window=context_window, min_periods=1).max().loc[probe_ts]
                        reversal_window_active = atomic_states.get('CONTEXT_VIOLENT_REVERSAL_WINDOW', default_series).loc[probe_ts]
                        print(f"    -> 诊断(setup_score): 要求最低分>{min_req}, 近期最高分是{max_score:.2f}。豁免信号: {reversal_window_active}")

                    print(f"  - [最终决策] 剧本是否激活 (valid_mask): {masks.get('valid_mask', default_series).loc[probe_ts]}")

                print("\n[当日最终得分]")
                print(f"  - 总分 (final_score): {final_score.loc[probe_ts]:.2f}")
                print(f"--- [探针] 日期 {probe_date_str} 剖析完毕 ---")

            except Exception as e:
                print(f"\n--- [探针错误] 在处理日期 {probe_date_str} 时发生错误: {e} ---")
        print("\n" + "="*27 + " [黑匣子探针结束] " + "="*27 + "\n")

    def _calculate_exit_signals(self, df: pd.DataFrame, params: dict, risk_score: pd.Series) -> pd.Series:
        """
        【V202.11 权限重铸版】出场决策引擎
        - 核心修复: 彻底重构。此引擎现在只负责处理“真正的卖出信号”。
                    它会从配置中获取一个明确的“最低卖出风险分”，只有当风险分
                    超过此值时，才会开始匹配并分配 exit_code。
        """
        exit_strategy_params = self.exit_strategy_params
        threshold_params = exit_strategy_params.get('exit_threshold_params', {})
        
        # ▼▼▼ 设定“最低开火权限”！▼▼▼
        # 只有当风险分达到 exit_threshold_params 中定义的最低 level 时，才触发卖出。
        # 这为“风险预警”留出了明确的安全空间。
        if not threshold_params:
            return pd.Series(0, index=df.index)
        
        min_score_for_exit = min(self._get_param_value(config.get('level')) for config in threshold_params.values())
        
        # 制作一个“开火许可”面具
        fire_permission_mask = risk_score >= min_score_for_exit
        
        # 如果没有任何一天的风险达到最低卖出标准，则直接返回全0，不产生任何卖出信号
        if not fire_permission_mask.any():
            return pd.Series(0, index=df.index)

        levels = sorted(threshold_params.items(), key=lambda item: self._get_param_value(item[1].get('level')), reverse=True)
        
        conditions = []
        choices = []
        
        for level_name, config in levels:
            threshold = self._get_param_value(config.get('level'))
            exit_code = self._get_param_value(config.get('code'))
            # 条件现在必须同时满足：达到阈值 且 拥有“开火许可”
            conditions.append((risk_score >= threshold) & fire_permission_mask)
            choices.append(exit_code)
            
        exit_signal = np.select(conditions, choices, default=0)
        return pd.Series(exit_signal, index=df.index)

    def _calculate_setup_conditions(self, df: pd.DataFrame, params: dict, atomic_states: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """
        【V126.0 智能火炮版 - 质量评估】
        - 核心升级: 新增 'PLATFORM_QUALITY' 的准备分计算逻辑。
        """
        print("    - [准备状态中心 V126.0] 启动...")
        setup_scores = {}
        default_series = pd.Series(False, index=df.index)
        scoring_matrix = self.setup_scoring_matrix
        for setup_name, rules in scoring_matrix.items():
            if not self._get_param_value(rules.get('enabled'), True):
                continue
            # print(f"          -> 正在评审 '{setup_name}'...")
            
            # ▼▼▼ “投降坑” 专属评分逻辑 ▼▼▼
            if setup_name == 'CAPITULATION_PIT':
                p_cap_pit = self.setup_scoring_matrix.get('CAPITULATION_PIT', {})
                must_have_score = self._get_param_value(p_cap_pit.get('must_have_score'), 40)
                bonus_score = self._get_param_value(p_cap_pit.get('bonus_score'), 25)
                
                # 核心条件：价格必须处于负向乖离的超卖区
                must_have_conditions = atomic_states.get('OPP_STATE_NEGATIVE_DEVIATION', default_series)
                
                # 加分项：获利盘极低 + 筹码极度发散 (这才是投降的真正信号!)
                bonus_conditions_1 = atomic_states.get('CHIP_STATE_LOW_PROFIT', default_series)
                bonus_conditions_2 = atomic_states.get('CHIP_STATE_SCATTERED', default_series)
                
                base_score = must_have_conditions.astype(int) * must_have_score
                bonus_score_total = (bonus_conditions_1.astype(int) * bonus_score) + (bonus_conditions_2.astype(int) * bonus_score)
                
                final_score = (base_score + bonus_score_total).where(must_have_conditions, 0)
                setup_scores[f'SETUP_SCORE_{setup_name}'] = final_score
            #  “平台质量” 专属评分逻辑
            elif setup_name == 'PLATFORM_QUALITY':
                print("          -> 正在评审 '平台质量(PLATFORM_QUALITY)'...")
                p_quality = self.setup_scoring_matrix.get('PLATFORM_QUALITY', {})
                # 必须条件：一个稳固的筹码平台已经形成
                must_have_cond = atomic_states.get('PLATFORM_STATE_STABLE_FORMED', default_series)
                # 基础分
                base_score = must_have_cond.astype(int) * self._get_param_value(p_quality.get('base_score'), 40)
                # 加分项
                bonus_score = pd.Series(0.0, index=df.index)
                bonus_rules = p_quality.get('bonus', {})
                for state, score in bonus_rules.items():
                    state_series = atomic_states.get(state, default_series)
                    bonus_score += state_series.astype(int) * score
                # 最终质量分 = (基础分 + 加分项)，且必须满足 must_have_cond
                setup_scores[f'SETUP_SCORE_{setup_name}'] = (base_score + bonus_score).where(must_have_cond, 0)
            else:
                # 其他所有剧本使用通用评分逻辑
                current_score = pd.Series(0.0, index=df.index)
                must_have_rules = rules.get('must_have', {})
                must_have_passed = pd.Series(True, index=df.index)
                for state, score in must_have_rules.items():
                    state_series = atomic_states.get(state, default_series)
                    current_score += state_series * score
                    must_have_passed &= state_series
                any_of_rules = rules.get('any_of_must_have', {})
                any_of_passed = pd.Series(False, index=df.index)
                if any_of_rules:
                    any_of_score_component = pd.Series(0.0, index=df.index)
                    for state, score in any_of_rules.items():
                        state_series = atomic_states.get(state, default_series)
                        any_of_score_component.loc[state_series] = score
                        any_of_passed |= state_series
                    current_score += any_of_score_component
                else:
                    any_of_passed = pd.Series(True, index=df.index)
                bonus_rules = rules.get('bonus', {})
                for state, score in bonus_rules.items():
                    state_series = atomic_states.get(state, default_series)
                    current_score += state_series * score
                final_validity = must_have_passed & any_of_passed
                setup_scores[f'SETUP_SCORE_{setup_name}'] = current_score.where(final_validity, 0)

            # max_score = setup_scores.get(f'SETUP_SCORE_{setup_name}', pd.Series(0)).max()
            # print(f"            -> '{setup_name}' 评审完成，最高置信度得分: {max_score:.0f}")
        print("    - [准备状态中心 V65.0] 所有状态置信度评审完成。")
        return setup_scores

    def _define_trigger_events(self, df: pd.DataFrame, params: dict, atomic_states: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """
        【V66.1 结构为王最终版】
        - 新增 TRIGGER_PANIC_REVERSAL，专门捕捉恐慌坑后的“功能性”强反转K线。
        """
        # print("    - [触发事件中心 V66.1 结构为王最终版] 启动，开始定义所有原子化触发事件...")
        triggers = {}
        default_series = pd.Series(False, index=df.index) # 新增一个默认的空Series，用于后续代码健壮性
        trigger_params = self._get_params_block(params, 'trigger_event_params', {})
        if not self._get_param_value(trigger_params.get('enabled'), True):
            print("      -> 触发事件引擎被禁用，跳过。")
            return triggers
        vol_ma_col = 'VOL_MA_21_D'
        if 'CHIP_EVENT' in df.columns:
            triggers['TRIGGER_CHIP_IGNITION'] = (df['CHIP_EVENT'] == 'IGNITION')
            print(f"      -> '筹码点火' 事件定义完成，发现 {triggers['TRIGGER_CHIP_IGNITION'].sum()} 天。")

        p_candle = trigger_params.get('positive_candle', {})
        if self._get_param_value(p_candle.get('enabled'), True):
            is_green = df['close_D'] > df['open_D']
            body_range = (df['high_D'] - df['low_D']).replace(0, np.nan)
            body_ratio = (df['close_D'] - df['open_D']) / body_range
            is_strong_body = body_ratio > self._get_param_value(p_candle.get('min_body_ratio'), 0.6)
            triggers['TRIGGER_STRONG_POSITIVE_CANDLE'] = is_green & is_strong_body
            
            # signal = triggers.get('TRIGGER_STRONG_POSITIVE_CANDLE', default_series)
            # dates_str = self._format_debug_dates(signal)
            # print(f"      -> '强势阳线' 事件定义完成，发现 {signal.sum()} 天。{dates_str}")

        # --- 步骤1: 定义标准的“反转确认阳线” (通用级) ---
        p_reversal = trigger_params.get('reversal_confirmation_candle', {})
        if self._get_param_value(p_reversal.get('enabled'), True):
            is_green = df['close_D'] > df['open_D']
            is_strong_rally = df['pct_change_D'] > self._get_param_value(p_reversal.get('min_pct_change'), 0.03)
            is_closing_strong = df['close_D'] > (df['high_D'] + df['low_D']) / 2
            triggers['TRIGGER_REVERSAL_CONFIRMATION_CANDLE'] = is_green & is_strong_rally & is_closing_strong
            
            # signal = triggers.get('TRIGGER_REVERSAL_CONFIRMATION_CANDLE', default_series)
            # dates_str = self._format_debug_dates(signal)
            # print(f"      -> '反转确认阳线' 事件定义完成，发现 {signal.sum()} 天。{dates_str}")

        # --- 步骤2: 定义新增的“显性反转阳线” (精英级) ---
        p_dominant = trigger_params.get('dominant_reversal_candle', {}) # 允许未来在JSON中配置
        if self._get_param_value(p_dominant.get('enabled'), True):
            # 条件A: 必须首先是一个高质量的通用反转信号
            base_reversal_signal = triggers.get('TRIGGER_REVERSAL_CONFIRMATION_CANDLE', default_series)
            
            # 条件B: 今天的阳线实体，必须在力量上压制昨天的阴线实体
            today_body_size = df['close_D'] - df['open_D']
            yesterday_body_size = abs(df['close_D'].shift(1) - df['open_D'].shift(1))
            
            # 只有当昨天是阴线时，才进行此项比较
            was_yesterday_red = df['close_D'].shift(1) < df['open_D'].shift(1)
            
            # 恢复比例，默认为0.5，即要求收复昨天实体的一半
            recovery_ratio = self._get_param_value(p_dominant.get('recovery_ratio'), 0.5)
            
            is_power_recovered = today_body_size >= (yesterday_body_size * recovery_ratio)
            
            # 最终的精英信号 = 高质量反转 AND (昨天不是阴线 OR 力量已压制)
            triggers['TRIGGER_DOMINANT_REVERSAL'] = base_reversal_signal & (~was_yesterday_red | is_power_recovered)
            
            # signal = triggers.get('TRIGGER_DOMINANT_REVERSAL', default_series)
            # print(f"      -> '显性反转阳线(精英级)' 事件定义完成，发现 {signal.sum()} 天。")
            
        p_breakout = trigger_params.get('volume_spike_breakout', {})
        if self._get_param_value(p_breakout.get('enabled'), True) and vol_ma_col in df.columns:
            volume_ratio = self._get_param_value(p_breakout.get('volume_ratio'), 2.0)
            lookback = self._get_param_value(p_breakout.get('lookback_period'), 20)
            is_volume_spike = df['volume_D'] > df[vol_ma_col] * volume_ratio
            is_price_breakout = df['close_D'] > df['high_D'].shift(1).rolling(lookback).max()
            triggers['TRIGGER_VOLUME_SPIKE_BREAKOUT'] = is_volume_spike & is_price_breakout
            
            # signal = triggers.get('TRIGGER_VOLUME_SPIKE_BREAKOUT', default_series)
            # dates_str = self._format_debug_dates(signal)
            # print(f"      -> '放量突破近期高点' 事件定义完成，发现 {signal.sum()} 天。{dates_str}")
            
        p_rebound = self._get_params_block(params, 'trigger_event_params', {}).get('pullback_rebound_trigger_params', {})
        if self._get_param_value(p_rebound.get('enabled'), True):
            support_ma_period = self._get_param_value(p_rebound.get('support_ma'), 21)
            support_ma_col = f'EMA_{support_ma_period}_D'
            if support_ma_col in df.columns:
                was_touching_support = df['low_D'].shift(1) <= df[support_ma_col].shift(1)
                is_rebounded_above = df['close_D'] > df[support_ma_col]
                is_positive_day = df['close_D'] > df['open_D']
                triggers['TRIGGER_PULLBACK_REBOUND'] = was_touching_support & is_rebounded_above & is_positive_day
                
                # signal = triggers.get('TRIGGER_PULLBACK_REBOUND', default_series)
                # dates_str = self._format_debug_dates(signal)
                # print(f"      -> '回踩反弹' 触发器定义完成，发现 {signal.sum()} 天。{dates_str}")

        # ▼▼▼ 趋势延续触发器 ▼▼▼
        p_cont = trigger_params.get('trend_continuation_candle', {})
        if self._get_param_value(p_cont.get('enabled'), True):
            lookback_period = self._get_param_value(p_cont.get('lookback_period'), 8)
            is_positive_day = df['close_D'] > df['open_D']
            is_new_high = df['close_D'] >= df['high_D'].shift(1).rolling(window=lookback_period).max()
            triggers['TRIGGER_TREND_CONTINUATION_CANDLE'] = is_positive_day & is_new_high
            
            # signal = triggers.get('TRIGGER_TREND_CONTINUATION_CANDLE', default_series)
            # print(f"      -> '趋势延续确认K线' 触发器定义完成 (周期:{lookback_period})，发现 {signal.sum()} 天。{self._format_debug_dates(signal)}")

        p_nshape = self.kline_params.get('n_shape_params', {})
        if self._get_param_value(p_nshape.get('enabled'), True):
            is_positive_day = df['close_D'] > df['open_D']
            n_shape_consolidation_state = atomic_states.get('KLINE_STATE_N_SHAPE_CONSOLIDATION', pd.Series(False, index=df.index))
            consolidation_high = df['high_D'].where(n_shape_consolidation_state, np.nan).ffill()
            is_breaking_consolidation = df['close_D'] > consolidation_high.shift(1)
            is_volume_ok = df['volume_D'] > df.get(vol_ma_col, 0)
            triggers['TRIGGER_N_SHAPE_BREAKOUT'] = is_positive_day & is_breaking_consolidation & is_volume_ok
            
            # signal = triggers.get('TRIGGER_N_SHAPE_BREAKOUT', default_series)
            # dates_str = self._format_debug_dates(signal)
            # print(f"      -> 'N字形态突破' 专属事件定义完成，发现 {signal.sum()} 天。{dates_str}")

        p_cross = trigger_params.get('indicator_cross_params', {})
        if self._get_param_value(p_cross.get('enabled'), True):
            if self._get_param_value(p_cross.get('dmi_cross', {}).get('enabled'), True):
                pdi_col, mdi_col = 'PDI_14_D', 'NDI_14_D'
                if all(c in df.columns for c in [pdi_col, mdi_col]):
                    triggers['TRIGGER_DMI_CROSS'] = (df[pdi_col] > df[mdi_col]) & (df[pdi_col].shift(1) <= df[mdi_col].shift(1))
                    
                    # signal = triggers.get('TRIGGER_DMI_CROSS', default_series)
                    # dates_str = self._format_debug_dates(signal)
                    # print(f"      -> 'DMI金叉' 事件定义完成，发现 {signal.sum()} 天。{dates_str}")
                    
            macd_p = p_cross.get('macd_cross', {})
            if self._get_param_value(macd_p.get('enabled'), True):
                macd_col, signal_col = 'MACD_13_34_8_D', 'MACDs_13_34_8_D'
                if all(c in df.columns for c in [macd_col, signal_col]):
                    is_golden_cross = (df[macd_col] > df[signal_col]) & (df[macd_col].shift(1) <= df[signal_col].shift(1))
                    low_level = self._get_param_value(macd_p.get('low_level'), -0.5)
                    triggers['TRIGGER_MACD_LOW_CROSS'] = is_golden_cross & (df[macd_col] < low_level)
                    
                    # signal = triggers.get('TRIGGER_MACD_LOW_CROSS', default_series)
                    # dates_str = self._format_debug_dates(signal)
                    # print(f"      -> 'MACD低位金叉' 事件定义完成，发现 {signal.sum()} 天。{dates_str}")
                    
        box_states = self._diagnose_box_states(df, params)
        triggers['TRIGGER_BOX_BREAKOUT'] = box_states.get('BOX_EVENT_BREAKOUT', pd.Series(False, index=df.index))
        
        # signal = triggers.get('TRIGGER_BOX_BREAKOUT', default_series)
        # dates_str = self._format_debug_dates(signal)
        # print(f"      -> '箱体突破' 事件定义完成，发现 {signal.sum()} 天。{dates_str}")
        
        board_events = self._diagnose_board_patterns(df, params)
        triggers['TRIGGER_EARTH_HEAVEN_BOARD'] = board_events.get('BOARD_EVENT_EARTH_HEAVEN', pd.Series(False, index=df.index))
        
        # signal = triggers.get('TRIGGER_EARTH_HEAVEN_BOARD', default_series)
        # dates_str = self._format_debug_dates(signal)
        # print(f"      -> '地天板' 触发事件定义完成，发现 {signal.sum()} 天。{dates_str}")
        
        p_breakout_stabilize = trigger_params.get('breakout_candle', {}) # 修正变量名以示区分
        if self._get_param_value(p_breakout_stabilize.get('enabled'), True):
            boll_mid_col = 'BBM_21_2.0_D'
            required_cols = ['open_D', 'high_D', 'low_D', 'close_D', boll_mid_col]
            if all(col in df.columns for col in required_cols):
                min_body_ratio = self._get_param_value(p_breakout_stabilize.get('min_body_ratio'), 0.4)
                is_strong_positive_candle = (
                    (df['close_D'] > df['open_D']) &
                    (((df['close_D'] - df['open_D']) / (df['high_D'] - df['low_D']).replace(0, np.nan)).fillna(1.0) >= min_body_ratio)
                )
                is_breaking_boll_mid = df['close_D'] > df[boll_mid_col]
                triggers['TRIGGER_BREAKOUT_CANDLE'] = is_strong_positive_candle & is_breaking_boll_mid
                # signal = triggers.get('TRIGGER_BREAKOUT_CANDLE', default_series)
                # print(f"      -> '突破阳线(企稳型)' 触发器定义完成，发现 {signal.sum()} 天。{self._format_debug_dates(signal)}")
            else:
                print(f"      -> [警告] 缺少定义'突破阳线(企稳型)'所需的列 (如: {boll_mid_col})，跳过该触发器。")

        p_energy = trigger_params.get('energy_release', {})
        if self._get_param_value(p_energy.get('enabled'), True):
            required_cols = ['open_D', 'high_D', 'low_D', 'close_D', 'volume_D', vol_ma_col]
            if all(col in df.columns for col in required_cols):
                is_positive_day = df['close_D'] > df['open_D']
                body_range = (df['high_D'] - df['low_D']).replace(0, np.nan)
                body_ratio = (df['close_D'] - df['open_D']) / body_range
                is_strong_body = body_ratio.fillna(1.0) > self._get_param_value(p_energy.get('min_body_ratio'), 0.5)
                volume_ratio = self._get_param_value(p_energy.get('volume_ratio'), 1.5)
                is_volume_spike = df['volume_D'] > df[vol_ma_col] * volume_ratio
                triggers['TRIGGER_ENERGY_RELEASE'] = is_positive_day & is_strong_body & is_volume_spike
                # signal = triggers.get('TRIGGER_ENERGY_RELEASE', default_series)
                # print(f"      -> '能量释放(突破型)' 专属事件定义完成，发现 {signal.sum()} 天。{self._format_debug_dates(signal)}")
            else:
                print(f"      -> [警告] 缺少定义'能量释放(突破型)'所需的列 (如: {vol_ma_col})，跳过该触发器。")

        # ▼▼▼ 平台回踩触发器 ▼▼▼
        p_platform_rebound = trigger_params.get('platform_pullback_trigger_params', {})
        if self._get_param_value(p_platform_rebound.get('enabled'), True):
            platform_price_col = 'PLATFORM_PRICE_STABLE'
            if platform_price_col in df.columns:
                # 条件1: 价格回踩到平台价格附近
                proximity_ratio = self._get_param_value(p_platform_rebound.get('proximity_ratio'), 0.01) # 允许1%的误差
                is_touching_platform = df['low_D'] <= df[platform_price_col] * (1 + proximity_ratio)
                # 条件2: 收盘价重新站上平台价格
                is_closing_above = df['close_D'] > df[platform_price_col]
                # 条件3: 当天是阳线，确认反弹意图
                is_positive_day = df['close_D'] > df['open_D']
                triggers['TRIGGER_PLATFORM_PULLBACK_REBOUND'] = is_touching_platform & is_closing_above & is_positive_day
                # signal = triggers.get('TRIGGER_PLATFORM_PULLBACK_REBOUND', default_series)
                # if signal.any():
                    # print(f"      -> '筹码平台回踩反弹' 触发器定义完成，发现 {signal.sum()} 天。{self._format_debug_dates(signal)}")
            else:
                print(f"      -> [警告] 缺少 '{platform_price_col}' 列，无法定义平台回踩触发器。")

        triggers['TRIGGER_TREND_STABILIZING'] = atomic_states.get('MA_STATE_D_STABILIZING', default_series)

        for key in triggers:
            if triggers[key] is None:
                triggers[key] = pd.Series(False, index=df.index)
            else:
                triggers[key] = triggers[key].fillna(False)
        # print("    - [触发事件中心 V66.1] 所有触发事件定义完成。")
        return triggers

    def _diagnose_chip_states(self, df: pd.DataFrame, params: dict) -> Dict[str, pd.Series]:
        """
        【V204.0 军事化改编版】筹码诊断总指挥部
        - 核心重构: 此方法已重构为总指挥角色，不再处理具体逻辑。
                    它负责按顺序调用下属的各个“专业化作战单元”（辅助方法），
                    并将它们的诊断结果汇总，形成最终的筹码状态报告。
        """
        print("        -> [诊断模块 V204.0 军事化改编版] 启动...")
        states = {}
        p = self._get_params_block(params, 'chip_feature_params', {})
        if not self._get_param_value(p.get('enabled'), False):
            print("          -> 筹码诊断模块被禁用，跳过。")
            return states

        # 预处理和数据检查
        required_cols = [
            'CHIP_peak_cost_D', 'CHIP_concentration_90pct_D', 'CHIP_concentration_90pct_slope_5d_D',
            'CHIP_peak_cost_slope_21d_D', 'CHIP_peak_cost_slope_55d_D', 'CHIP_peak_cost_accel_21d_D',
            'CHIP_peak_cost_slope_8d_D', 'CHIP_winner_rate_short_term_D', 'close_D',
            'CHIP_total_winner_rate_D', 'SLOPE_5_CHIP_total_winner_rate_D', 'SLOPE_5_CHIP_peak_stability_D',
            'SLOPE_5_CHIP_peak_percent_D', 'SLOPE_5_CHIP_pressure_above_D'
        ]
        if not all(col in df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in df.columns]
            print(f"          -> [严重警告] 缺少筹码诊断所需的列: {missing}。引擎将返回空结果。")
            return states
        
        df_copy = df.copy()
        for col in required_cols:
            df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')

        # ▼▼▼【代码重构 V204.0】: 依次调用专业化作战单元 ▼▼▼
        # 1. 动态侦察单元: 负责分析斜率等动态指标
        states.update(self._diagnose_chip_dynamics_states(df_copy, p))
        
        # 2. 结构分析单元: 负责分析集中度，并内置“叛徒清除”逻辑
        states.update(self._diagnose_chip_concentration_states(df_copy, p))

        # 3. 周期研判单元: 负责判断市场宏观周期
        primary_state = self._diagnose_chip_cycle_states(df_copy, p, states)
        states.update(primary_state)

        # 4. 风险哨站: 专门识别各类筹码风险
        states.update(self._diagnose_chip_risk_states(df_copy, p, states))

        # 5. 特种事件部队: 捕捉“点火”、“断层新生”等特殊信号
        states.update(self._diagnose_chip_events(df_copy, p, primary_state['CHIP_PRIMARY_STATE']))
        # ▲▲▲【代码重构 V204.0】▲▲▲

        # 最终清理
        for key in states:
            if states[key] is None:
                states[key] = pd.Series(False, index=df_copy.index)
            else:
                states[key] = states[key].fillna(False)
        return states

    def _diagnose_chip_dynamics_states(self, df: pd.DataFrame, params: dict) -> Dict[str, pd.Series]:
        """【V204.0 新增】专业化作战单元：动态侦察单元"""
        states = {}
        # 状态1: 获利盘加速扩张 (S级进攻信号)
        winner_rate_slope_col = 'SLOPE_5_CHIP_total_winner_rate_D'
        states['CHIP_STATE_WINNER_RATE_ACCELERATING'] = (df[winner_rate_slope_col] > 0) & (df[winner_rate_slope_col] > df[winner_rate_slope_col].shift(1))
        
        # 状态2: 筹码峰正在被夯实 (A级进攻信号)
        stability_slope_col = 'SLOPE_5_CHIP_peak_stability_D'
        percent_slope_col = 'SLOPE_5_CHIP_peak_percent_D'
        states['CHIP_STATE_PEAK_CONSOLIDATING'] = (df[stability_slope_col] > 0) & (df[percent_slope_col] > 0)
        
        # 状态3: 上方套牢盘快速消化 (B级进攻信号)
        pressure_slope_col = 'SLOPE_5_CHIP_pressure_above_D'
        states['CHIP_STATE_PRESSURE_DISSOLVING'] = (df[pressure_slope_col] < 0)
        
        return states

    def _diagnose_chip_concentration_states(self, df: pd.DataFrame, params: dict) -> Dict[str, pd.Series]:
        """【V204.0 新增】专业化作战单元：结构分析单元 (内置“叛徒清除”逻辑)"""
        states = {}
        p_struct = params.get('structure_params', {})
        if not self._get_param_value(p_struct.get('enabled'), True):
            return states

        conc_col = 'CHIP_concentration_90pct_D'
        conc_slope_col = 'CHIP_concentration_90pct_slope_5d_D'
        
        # 静态条件：筹码集中度的绝对值是否低于阈值
        conc_thresh_abs = self._get_param_value(p_struct.get('high_concentration_threshold'), 0.15)
        is_low_concentration_value = df[conc_col] < conc_thresh_abs
        
        # 动态安全锁：筹码集中度的变化趋势是否健康（稳定或仍在集中）
        slope_tolerance = self._get_param_value(p_struct.get('slope_tolerance'), 0.001)
        is_trend_healthy = df[conc_slope_col] <= slope_tolerance
        
        # 新的、可靠的“高度集中”信号 = 静态条件 AND 动态安全锁
        states['CHIP_STATE_HIGHLY_CONCENTRATED'] = is_low_concentration_value & is_trend_healthy
        print(f"          -> [忠诚信号] '筹码高度集中' 已校准，激活 {states['CHIP_STATE_HIGHLY_CONCENTRATED'].sum()} 天。")

        # 全新的、最高优先级风险信号：“筹码结构崩溃”
        is_trend_dispersing = df[conc_slope_col] > slope_tolerance
        states['RISK_CHIP_STRUCTURE_COLLAPSE'] = is_low_concentration_value & is_trend_dispersing
        if states['RISK_CHIP_STRUCTURE_COLLAPSE'].any():
            print(f"          -> [!!!最高警报!!!] '筹码结构崩溃' 风险已识别，激活 {states['RISK_CHIP_STRUCTURE_COLLAPSE'].sum()} 天。")

        # 其他相关状态
        if self._get_param_value(p_struct.get('enable_relative_squeeze'), True):
            squeeze_window = self._get_param_value(p_struct.get('squeeze_window'), 120)
            squeeze_percentile = self._get_param_value(p_struct.get('squeeze_percentile'), 0.2)
            squeeze_threshold_series = df[conc_col].rolling(window=squeeze_window).quantile(squeeze_percentile)
            states['CHIP_STATE_CONCENTRATION_SQUEEZE'] = df[conc_col] < squeeze_threshold_series
        
        p_scattered = params.get('scattered_params', {})
        if self._get_param_value(p_scattered.get('enabled'), True):
            scattered_threshold_pct = self._get_param_value(p_scattered.get('threshold'), 30.0)
            states['CHIP_STATE_SCATTERED'] = df[conc_col] > (scattered_threshold_pct / 100.0)
            
        return states

    def _diagnose_chip_cycle_states(self, df: pd.DataFrame, params: dict, current_states: dict) -> Dict[str, pd.Series]:
        """【V204.0 新增】专业化作战单元：周期研判单元"""
        states = {}
        # 拉升周期
        is_markup_base = (df['CHIP_peak_cost_slope_21d_D'] > 0) & (df.get('CHIP_peak_cost_slope_55d_D', 0) > 0)
        
        # 派发周期
        p_dist = params.get('distribution_params', {})
        is_distributing = df['CHIP_concentration_90pct_slope_5d_D'] > self._get_param_value(p_dist.get('divergence_threshold'), 0.01)
        is_at_high = df['close_D'] > df['close_D'].rolling(window=55).quantile(0.8)
        is_distribution_base = is_distributing & is_at_high
        
        # 吸筹周期
        p_accum = params.get('accumulation_params', {})
        lookback_accum = self._get_param_value(p_accum.get('lookback_days'), 21)
        concentrating_days = (df['CHIP_concentration_90pct_slope_5d_D'] < 0).rolling(window=lookback_accum).sum()
        is_concentrating = concentrating_days >= (lookback_accum * self._get_param_value(p_accum.get('required_days_ratio'), 0.6))
        is_not_rising = df['CHIP_peak_cost_slope_21d_D'] <= 0
        is_accumulation_base = is_concentrating & is_not_rising
        
        conditions = [is_markup_base, is_distribution_base, is_accumulation_base]
        choices = ['MARKUP', 'DISTRIBUTION', 'ACCUMULATION']
        primary_state = pd.Series(np.select(conditions, choices, default='TRANSITION'), index=df.index)
        
        states['CHIP_PRIMARY_STATE'] = primary_state # 存储原始状态，供事件诊断使用
        states['CHIP_STATE_MARKUP'] = (primary_state == 'MARKUP')
        states['CHIP_STATE_ACCUMULATION'] = (primary_state == 'ACCUMULATION')
        states['CHIP_STATE_DISTRIBUTION'] = (primary_state == 'DISTRIBUTION')
        
        return states

    def _diagnose_chip_risk_states(self, df: pd.DataFrame, params: dict, current_states: dict) -> Dict[str, pd.Series]:
        """【V204.0 新增】专业化作战单元：风险哨站"""
        states = {}
        # 风险1: 获利盘扩张停滞
        winner_rate_slope_col = 'SLOPE_5_CHIP_total_winner_rate_D'
        states['CHIP_RISK_PROFIT_TAKING_IMMINENT'] = (df[winner_rate_slope_col] < 0) & (df[winner_rate_slope_col].shift(1) > 0)
        
        # 风险2: 力竭风险
        is_still_rising = df['CHIP_peak_cost_slope_21d_D'] > 0
        is_decelerating = df['CHIP_peak_cost_accel_21d_D'] < 0
        states['CHIP_RISK_EXHAUSTION'] = is_still_rising & is_decelerating
        
        # 风险3: 背离风险
        is_short_slope_down = df['CHIP_peak_cost_slope_8d_D'] < 0
        is_mid_slope_up = df['CHIP_peak_cost_slope_21d_D'] > 0
        is_at_high = df['close_D'] > df['close_D'].rolling(window=55).quantile(0.8)
        states['CHIP_RISK_DIVERGENCE'] = is_short_slope_down & is_mid_slope_up & is_at_high
        
        return states

    def _diagnose_chip_events(self, df: pd.DataFrame, params: dict, primary_state: pd.Series) -> Dict[str, pd.Series]:
        """【V204.0 新增】专业化作战单元：特种事件部队"""
        states = {}
        default_series = pd.Series(False, index=df.index)
        
        # 事件1: 点火事件
        p_ignite = params.get('ignition_params', {})
        is_accelerating = df['CHIP_peak_cost_accel_21d_D'] > self._get_param_value(p_ignite.get('accel_threshold'), 0.01)
        is_winner_rate_increasing = df['CHIP_winner_rate_short_term_D'] > df['CHIP_winner_rate_short_term_D'].shift(1)
        was_in_setup_state = primary_state.shift(1).isin(['ACCUMULATION', 'TRANSITION'])
        states['CHIP_EVENT_IGNITION'] = is_accelerating & is_winner_rate_increasing & was_in_setup_state
        
        # 事件2: 断层新生事件
        p_fault = params.get('fault_rebirth_params', {})
        if self._get_param_value(p_fault.get('enabled'), True):
            cost_pct_change = df['CHIP_peak_cost_D'].pct_change()
            cost_drop_threshold = self._get_param_value(p_fault.get('cost_drop_threshold'), -0.10)
            is_cost_cliff = (cost_pct_change <= cost_drop_threshold).fillna(False)
            window_days = self._get_param_value(p_fault.get('observation_window_days'), 5)
            fault_rebirth_window = self._create_persistent_state(df, entry_event=is_cost_cliff, persistence_days=window_days)
            is_re_accumulating = (df['CHIP_concentration_90pct_slope_5d_D'] < 0) & (df['CHIP_concentration_90pct_slope_5d_D'].shift(1).fillna(0) > 0)
            is_confirmed_in_window = is_re_accumulating & fault_rebirth_window
            first_confirmation_in_window = is_confirmed_in_window & ~is_confirmed_in_window.shift(1).fillna(False)
            states['CHIP_EVENT_FAULT_REBIRTH'] = is_cost_cliff | first_confirmation_in_window
        else:
            states['CHIP_EVENT_FAULT_REBIRTH'] = default_series
            
        return states

    # ▼▼▼ “平台引力”侦察模块 ▼▼▼
    def _diagnose_platform_states(self, df: pd.DataFrame, params: dict) -> Tuple[pd.DataFrame, Dict[str, pd.Series]]:
        """
        【V129.0 平台时效性与相关性版】
        - 核心重构: 彻底重写平台诊断逻辑，引入“时效性”和“相关性”双重约束。
        - 新逻辑:
          1. (稳定性) 保持原有逻辑：计算筹码成本的归一化标准差，判断成本是否稳定。
          2. (相关性) 新增约束：要求当前收盘价必须在平台成本的一定范围(如±5%)内，
             确保平台对当前价格有实际的“引力作用”。
          3. (时效性) 废除ffill()：只有在当天同时满足“稳定性”和“相关性”时，
             才认为平台有效，并记录其价格。这从根本上解决了平台“永不过期”的问题。
        - 收益: 使得“稳固筹码平台”成为一个真正稀缺、高价值的战术信号，极大地提升了其可靠性。
        """
        print("        -> [诊断模块 V129.0] 正在执行筹码平台状态诊断...")
        states = {}
        df_copy = df.copy()
        default_series = pd.Series(False, index=df_copy.index)

        p = self._get_params_block(params, 'platform_state_params', {})
        if not self._get_param_value(p.get('enabled'), True):
            print("          -> 筹码平台诊断模块被禁用，跳过。")
            df_copy['PLATFORM_PRICE_STABLE'] = np.nan
            states['PLATFORM_STATE_STABLE_FORMED'] = default_series
            return df_copy, states

        cost_col = 'CHIP_peak_cost_D'
        close_col = 'close_D'
        if cost_col not in df_copy.columns or close_col not in df_copy.columns:
            print(f"          -> [警告] 缺少列 '{cost_col}' 或 '{close_col}'，无法诊断平台状态。")
            df_copy['PLATFORM_PRICE_STABLE'] = np.nan
            states['PLATFORM_STATE_STABLE_FORMED'] = default_series
            return df_copy, states

        # --- 参数获取 ---
        stability_window = self._get_param_value(p.get('stability_window'), 10)
        stability_threshold = self._get_param_value(p.get('stability_threshold_ratio'), 0.01)
        # 新增：价格相关性阈值
        relevance_threshold = self._get_param_value(p.get('price_relevance_threshold'), 0.05) # 价格偏离平台成本5%即认为无关

        # --- 过滤一: 稳定性 (Stability) ---
        rolling_std = df_copy[cost_col].rolling(window=stability_window).std()
        rolling_mean = df_copy[cost_col].rolling(window=stability_window).mean()
        normalized_std = (rolling_std / rolling_mean).fillna(1.0)
        is_cost_stable = normalized_std < stability_threshold
        
        # --- 过滤二: 相关性 (Relevance) ---
        # 计算价格与平台成本的偏离度
        price_deviation_ratio = abs(df_copy[close_col] / df_copy[cost_col] - 1)
        is_price_relevant = price_deviation_ratio < relevance_threshold

        # --- 过滤三: 时效性 (Timeliness) ---
        # 最终的平台形成状态 = 稳定性 AND 相关性
        # 只有在当天同时满足这两个条件，才认为平台是“活的”
        is_platform_alive = is_cost_stable & is_price_relevant
        
        # 不再使用ffill()，只在平台“存活”的当天记录其价格
        df_copy['PLATFORM_PRICE_STABLE'] = df_copy[cost_col].where(is_platform_alive, np.nan)
        
        # 状态字典也基于“存活”状态
        states['PLATFORM_STATE_STABLE_FORMED'] = is_platform_alive.fillna(False)
        
        # if states['PLATFORM_STATE_STABLE_FORMED'].any():
            # print(f"          -> '稳固筹码平台'已识别 (已加入时效与相关性约束)，共持续 {states['PLATFORM_STATE_STABLE_FORMED'].sum()} 天。")

        return df_copy, states

    def _diagnose_ma_states(self, df: pd.DataFrame, params: dict) -> Dict[str, pd.Series]:
        """
        【V155.0 生命线校准版】
        - 核心修正: 采纳将军建议，将定义核心趋势状态的长期均线(long_ma)基准，
                    从过于迟钝的89日，全面校准为市场公认的55日“生命线”。
        """
        # print("        -> [诊断模块] 正在执行均线状态诊断...")
        states = {}
        default_series = pd.Series(False, index=df.index)
        p = self._get_params_block(params, 'ma_state_params', {})
        if not self._get_param_value(p.get('enabled'), False):
            print("          -> 均线诊断模块被禁用，跳过。")
            return states

        # --- 1. 核心趋势状态诊断 ---
        short_p = self._get_param_value(p.get('short_ma'), 13)
        mid_p = self._get_param_value(p.get('mid_ma'), 34)
        # 【核心修正】将长期均线基准从89日改为55日
        long_p = self._get_param_value(p.get('long_ma'), 55)
        short_ma, mid_ma, long_ma = f'EMA_{short_p}_D', f'EMA_{mid_p}_D', f'EMA_{long_p}_D'
        short_ma_slope_col = f'SLOPE_5_{short_ma}'

        if not all(c in df.columns for c in [short_ma, mid_ma, long_ma]):
            print(f"          -> [警告] 缺少核心均线列({short_ma}, {mid_ma}, {long_ma})，部分均线状态诊断跳过。")
            return states
        if short_ma_slope_col in df.columns:
            states['MA_STATE_SHORT_SLOPE_POSITIVE'] = (df[short_ma_slope_col] > 0)
        else:
            states['MA_STATE_SHORT_SLOPE_POSITIVE'] = default_series

        states['MA_STATE_PRICE_ABOVE_LONG_MA'] = df['close_D'] > df[long_ma]
        states['MA_STATE_STABLE_BULLISH'] = (df[short_ma] > df[mid_ma]) & (df[mid_ma] > df[long_ma])
        states['MA_STATE_SHORT_CROSS_MID'] = (df[short_ma] > df[mid_ma])
        states['MA_STATE_STABLE_BEARISH'] = (df[short_ma] < df[mid_ma]) & (df[mid_ma] < df[long_ma])
        states['MA_STATE_BOTTOM_PASSIVATION'] = states['MA_STATE_STABLE_BEARISH'] & (df['close_D'] > df[short_ma])

        # ... 其他逻辑保持不变 ...
        ma_spread = (df[short_ma] - df[long_ma]) / df[long_ma].replace(0, np.nan)
        ma_spread_zscore = (ma_spread - ma_spread.rolling(60).mean()) / ma_spread.rolling(60).std().replace(0, np.nan)
        states['MA_STATE_CONVERGING'] = ma_spread_zscore < self._get_param_value(p.get('converging_zscore'), -1.0)
        states['MA_STATE_DIVERGING'] = ma_spread_zscore > self._get_param_value(p.get('diverging_zscore'), 1.0)
        lookback_period = 10
        accel_d_col = f'ACCEL_{lookback_period}_{long_ma}'
        if accel_d_col in df.columns:
            states['MA_STATE_D_STABILIZING'] = (df[accel_d_col].shift(1).fillna(0) < 0) & (df[accel_d_col] >= 0)
        else:
            states['MA_STATE_D_STABILIZING'] = default_series
        simulated_w_ma_period = 105
        simulated_w_lookback = 25
        slope_w_simulated_col = f'SLOPE_{simulated_w_lookback}_EMA_{simulated_w_ma_period}_D'
        if slope_w_simulated_col in df.columns:
            states['MA_STATE_W_STABILIZING'] = (df[slope_w_simulated_col].shift(1).fillna(0) < 0) & (df[slope_w_simulated_col] >= 0)
        else:
            states['MA_STATE_W_STABILIZING'] = default_series
        key_support_mas = [21, 34, 55, 89, 144, 233]
        for ma_period in key_support_mas:
            ma_col = f'EMA_{ma_period}_D'
            if ma_col in df.columns:
                is_touching = df['low_D'] <= df[ma_col]
                is_closing_above = df['close_D'] >= df[ma_col]
                state_name = f'MA_STATE_TOUCHING_SUPPORT_{ma_period}'
                states[state_name] = is_touching & is_closing_above
            else:
                states[f'MA_STATE_TOUCHING_SUPPORT_{ma_period}'] = default_series
        p_duck = self._get_params_block(params, 'duck_neck_params', {})
        if self._get_param_value(p_duck.get('enabled'), True):
            duck_short_p = self._get_param_value(p_duck.get('short_ma'), 5)
            duck_mid_p = self._get_param_value(p_duck.get('mid_ma'), 10)
            duck_long_p = self._get_param_value(p_duck.get('long_ma'), 60)
            duck_short_ma, duck_mid_ma, duck_long_ma = f'EMA_{duck_short_p}_D', f'EMA_{duck_mid_p}_D', f'EMA_{duck_long_p}_D'
            if all(c in df.columns for c in [duck_short_ma, duck_mid_ma, duck_long_ma]):
                golden_cross_event = (df[duck_short_ma] > df[duck_mid_ma]) & (df[duck_short_ma].shift(1) <= df[duck_mid_ma].shift(1))
                break_condition = (df[duck_short_ma] < df[duck_mid_ma])
                persistence_days = self._get_param_value(p_duck.get('persistence_days'), 20)
                states['MA_STATE_DUCK_NECK_FORMING'] = self._create_persistent_state(
                    df, entry_event=golden_cross_event, persistence_days=persistence_days, break_condition=break_condition
                )
            else:
                states['MA_STATE_DUCK_NECK_FORMING'] = default_series
        for key in states:
            if states[key] is None:
                states[key] = default_series
            else:
                states[key] = states[key].fillna(False)
        return states

    # 动态惯性引擎
    def _diagnose_trend_dynamics(self, df: pd.DataFrame, params: dict) -> Dict[str, pd.Series]:
        """
        【V174.0 动态惯性引擎】
        - 核心职责: 基于趋势的“斜率”和“加速度”，生成高维度的动态原子状态。
        - 产出: 返回一个包含 DYN_... 信号的字典，供评分引擎使用。
        """
        print("        -> [诊断模块 V174.0] 正在执行动态惯性诊断...")
        dynamics_states = {}
        default_series = pd.Series(False, index=df.index)

        # --- 1. 获取核心的斜率和加速度数据 ---
        # 长期趋势的速度和加速度
        long_slope_col = 'SLOPE_55_EMA_55_D'
        long_accel_col = 'ACCEL_55_EMA_55_D'
        # 短期趋势的速度
        short_slope_col = 'SLOPE_13_EMA_13_D'

        if not all(c in df.columns for c in [long_slope_col, long_accel_col, short_slope_col]):
            print("          -> [错误] 动态惯性诊断缺少必要的斜率/加速度列，跳过。")
            return {}

        long_slope = df[long_slope_col]
        long_accel = df[long_accel_col]
        short_slope = df[short_slope_col]

        # --- 2. 定义基础布尔条件 ---
        is_long_slope_positive = long_slope > 0
        is_long_slope_negative = long_slope < 0
        is_long_accel_positive = long_accel > 0
        is_long_accel_negative = long_accel < 0
        
        is_short_slope_positive = short_slope > 0
        is_short_slope_negative = short_slope < 0
        
        # --- 3. 组合生成高维度动态状态 ---
        # 【S级进攻】健康加速: 长短期趋势同向看涨，且长期趋势在加速
        dynamics_states['DYN_TREND_HEALTHY_ACCELERATING'] = is_long_slope_positive & is_short_slope_positive & is_long_accel_positive
        
        # 【A级进攻】成熟稳定: 长短期趋势同向看涨，但长期趋势已不再加速（减速或匀速）
        dynamics_states['DYN_TREND_MATURE_STABLE'] = is_long_slope_positive & is_short_slope_positive & ~is_long_accel_positive

        # 【B级进攻】底部反转: 长期趋势仍向下，但短期趋势已率先加速向上
        dynamics_states['DYN_TREND_BOTTOM_REVERSING'] = is_long_slope_negative & is_short_slope_positive & (df[short_slope_col] > df[short_slope_col].shift(1))

        # 【S级风险】动能衰减: 长期趋势虽向上，但已开始减速，且短期趋势已逆转
        dynamics_states['DYN_TREND_WEAKENING_DECELERATING'] = is_long_slope_positive & is_long_accel_negative & is_short_slope_negative

        # 【A级风险】下跌加速: 长短期趋势同向看跌，且下跌在加速
        dynamics_states['DYN_TREND_BEARISH_ACCELERATING'] = is_long_slope_negative & is_short_slope_negative & is_long_accel_negative

        # 【B级风险】顶部背离: 价格创近期新高，但长短期斜率均在下降
        is_new_high = df['high_D'] >= df['high_D'].shift(1).rolling(window=10).max()
        is_slope_weakening = (long_slope < long_slope.shift(1)) & (short_slope < short_slope.shift(1))
        dynamics_states['DYN_TREND_TOPPING_DIVERGENCE'] = is_new_high & is_slope_weakening

        # --- 4. 打印调试信息 ---
        # for name, series in dynamics_states.items():
        #     print(f"          -> “{name}” 已定义，激活 {series.sum()} 天。")
            
        return dynamics_states

    def _diagnose_oscillator_states(self, df: pd.DataFrame, params: dict) -> Dict[str, pd.Series]:
        """【V64.5 最终融合版】震荡指标状态诊断中心"""
        # print("        -> [诊断模块] 正在执行震荡指标状态诊断...")
        states = {}
        p = self._get_params_block(params, 'oscillator_state_params', {})
        if not self._get_param_value(p.get('enabled'), False): return states
        
        # --- RSI 相关状态 ---
        rsi_col = 'RSI_13_D'
        if rsi_col in df.columns:
            states['OSC_STATE_RSI_OVERBOUGHT'] = df[rsi_col] > self._get_param_value(p.get('rsi_overbought'), 80)
            states['OSC_STATE_RSI_OVERSOLD'] = df[rsi_col] < self._get_param_value(p.get('rsi_oversold'), 25)
        
        # --- MACD 相关状态 ---
        macd_h_col = 'MACDh_13_34_8_D'
        macd_z_col = 'MACD_HIST_ZSCORE_D'
        if macd_h_col in df.columns:
            states['OSC_STATE_MACD_BULLISH'] = df[macd_h_col] > 0
        if macd_z_col in df.columns:
            is_price_higher = df['close_D'] > df['close_D'].rolling(10).max().shift(1)
            is_macd_z_lower = df[macd_z_col] < df[macd_z_col].rolling(10).max().shift(1)
            states['OSC_STATE_MACD_DIVERGENCE'] = is_price_higher & is_macd_z_lower

        # ▼▼▼ BIAS机会状态的诊断 ▼▼▼
        # ▼▼▼【代码修改 V64.7】: 使用动态分位数阈值 ▼▼▼
        p_bias = self._get_params_block(params, 'playbook_specific_params', {}).get('bias_reversal_params', {})
        if self._get_param_value(p_bias.get('enabled'), True):
            bias_col = 'BIAS_55_D'
            
            if bias_col in df.columns:
                # 获取动态阈值参数
                dynamic_threshold_params = p_bias.get('dynamic_threshold', {})
                window = self._get_param_value(dynamic_threshold_params.get('window'), 120)
                quantile = self._get_param_value(dynamic_threshold_params.get('quantile'), 0.1)

                # 计算滚动的分位数阈值
                dynamic_oversold_threshold = df[bias_col].rolling(window=window).quantile(quantile)

                # 判断当前BIAS是否低于动态阈值
                states['OPP_STATE_NEGATIVE_DEVIATION'] = df[bias_col] < dynamic_oversold_threshold
                
                # signal = states.get('OPP_STATE_NEGATIVE_DEVIATION', pd.Series(False, index=df.index))
                # dates_str = self._format_debug_dates(signal)
                # print(f"          -> '价格负向乖离' 机会状态诊断完成 (基于{bias_col}动态分位数 窗口:{window}, 分位:{quantile})，共激活 {signal.sum()} 天。{dates_str}")
            else:
                print(f"          -> [警告] 缺少列 '{bias_col}'，价格负向乖离状态无法诊断。")
        
        # print("        -> [诊断模块] 震荡指标状态诊断执行完毕。")
        return states

    def _diagnose_capital_states(self, df: pd.DataFrame, params: dict) -> Dict[str, pd.Series]:
        """【V58.0 升级版 - 集成持久化状态】资金流状态诊断"""
        states = {}
        p = self._get_params_block(params, 'capital_state_params', {})
        if not self._get_param_value(p.get('enabled'), False): return states
        mf_slope_short_col = f'SLOPE_10_net_mf_amount_D'
        mf_slope_long_col = f'SLOPE_20_net_mf_amount_D'
        if mf_slope_short_col in df.columns and mf_slope_long_col in df.columns:
            mf_slope_short = df[mf_slope_short_col]
            mf_slope_long = df[mf_slope_long_col]
            states['CAPITAL_STATE_INFLOW_CONFIRMED'] = mf_slope_short > 0
            states['CAPITAL_EVENT_SLOPE_CROSS'] = (mf_slope_short > mf_slope_long) & (mf_slope_short.shift(1) <= mf_slope_long.shift(1))
        cmf_col = 'CMF_21_D'
        if cmf_col in df.columns:
            states['CAPITAL_STATE_CMF_BULLISH'] = df[cmf_col] > self._get_param_value(p.get('cmf_bullish_threshold'), 0.05)
        price_down = df['pct_change_D'] < 0
        capital_up = df['net_mf_amount_D'] > 0
        bottom_divergence_event = price_down & capital_up
        p_context = p.get('divergence_context', {})
        persistence_days = self._get_param_value(p_context.get('persistence_days'), 15)
        trend_ma_period = self._get_param_value(p_context.get('trend_ma_period'), 55)
        trend_ma_col = f'EMA_{trend_ma_period}_D'
        if trend_ma_col in df.columns:
            break_condition = df['close_D'] > df[trend_ma_col]
            states['CAPITAL_STATE_DIVERGENCE_WINDOW'] = self._create_persistent_state(
                df, entry_event=bottom_divergence_event, persistence_days=persistence_days, break_condition=break_condition
            )
            
            # signal = states['CAPITAL_STATE_DIVERGENCE_WINDOW']
            # dates_str = self._format_debug_dates(signal)
            # print(f"          -> '资本底背离机会窗口' 持久化状态诊断完成，共激活 {signal.sum()} 天。{dates_str}")
            
        return states

    def _diagnose_volatility_states(self, df: pd.DataFrame, params: dict) -> Dict[str, pd.Series]:
        """【V58.1 升级版 - 集成持久化状态】波动率与成交量状态诊断"""
        states = {}
        p = self._get_params_block(params, 'volatility_state_params', {})
        if not self._get_param_value(p.get('enabled'), False): return states
        bbw_col = 'BBW_21_2.0_D'
        if bbw_col in df.columns:
            squeeze_threshold = df[bbw_col].rolling(60).quantile(self._get_param_value(p.get('squeeze_percentile'), 0.1))
            squeeze_event = (df[bbw_col] < squeeze_threshold) & (df[bbw_col].shift(1) >= squeeze_threshold)
            states['VOL_EVENT_SQUEEZE'] = squeeze_event
            
            # signal = states['VOL_EVENT_SQUEEZE']
            # dates_str = self._format_debug_dates(signal)
            # print(f"          -> '波动率极度压缩' 事件诊断完成，发现 {signal.sum()} 天。{dates_str}")
            
        vol_ma_col = 'VOL_MA_21_D'
        if 'volume_D' in df.columns and vol_ma_col in df.columns:
            states['VOL_STATE_SHRINKING'] = df['volume_D'] < df[vol_ma_col] * self._get_param_value(p.get('shrinking_ratio'), 0.8)
        p_context = p.get('squeeze_context', {})
        persistence_days = self._get_param_value(p_context.get('persistence_days'), 10)
        if vol_ma_col in df.columns:
            volume_break_ratio = self._get_param_value(p_context.get('volume_break_ratio'), 1.5)
            break_condition = df['volume_D'] > df[vol_ma_col] * volume_break_ratio
            states['VOL_STATE_SQUEEZE_WINDOW'] = self._create_persistent_state(
                df, entry_event=states.get('VOL_EVENT_SQUEEZE', pd.Series(False, index=df.index)),
                persistence_days=persistence_days, break_condition=break_condition
            )
            
            # signal = states['VOL_STATE_SQUEEZE_WINDOW']
            # dates_str = self._format_debug_dates(signal)
            # print(f"          -> '能量待爆发窗口' 持久化状态诊断完成，共激活 {signal.sum()} 天。{dates_str}")
            
        return states

    def _diagnose_box_states(self, df: pd.DataFrame, params: dict) -> Dict[str, pd.Series]:
        """
        【V147.0 实战化地形识别版】
        - 核心重构: 彻底废弃了过于学术化、无法识别平坦整理区的 find_peaks 算法。
        - 新逻辑:
          采用更贴近实战的“振幅比率”法。只要股价在过去N天内的振幅
          (rolling_high - rolling_low) / rolling_low 小于一个阈值(如5%)，
          就认为形成了一个有效的“战术平台”或“箱体”。
        - 收益: 能够精准识别各种形态的盘整区，特别是像06-25至06-27日那种
                极其平坦的“空中加油”平台，从根本上解决了因此类平台无法被识别
                而错失突破信号的问题。
        """
        print("        -> [诊断模块 V147.0] 正在执行箱体状态诊断(实战化地形识别版)...")
        states = {}
        default_series = pd.Series(False, index=df.index)
        box_params = self._get_params_block(params, 'dynamic_box_params', {})
        if not self._get_param_value(box_params.get('enabled'), False) or df.empty:
            print("          -> 箱体诊断模块被禁用或数据为空，跳过。")
            return states

        # --- 新逻辑参数 ---
        lookback_window = self._get_param_value(box_params.get('lookback_window'), 8)
        max_amplitude_ratio = self._get_param_value(box_params.get('max_amplitude_ratio'), 0.05) # 振幅小于5%

        # --- 步骤1: 计算滚动窗口内的高点、低点和振幅 ---
        rolling_high = df['high_D'].rolling(window=lookback_window).max()
        rolling_low = df['low_D'].rolling(window=lookback_window).min()
        
        # 计算振幅比率，分母使用rolling_low避免除以0，且更符合涨跌幅定义
        amplitude_ratio = (rolling_high - rolling_low) / rolling_low.replace(0, np.nan)

        # --- 步骤2: 识别有效的“箱体”/“平台” ---
        # 只要近期振幅足够小，就认为是一个有效的箱体
        is_valid_box = (amplitude_ratio < max_amplitude_ratio).fillna(False)
        
        # 箱体的顶部和底部就是滚动窗口的高点和低点
        box_top = rolling_high
        box_bottom = rolling_low

        # --- 步骤3: 定义突破与跌破事件 (逻辑不变，但基于新的箱体定义) ---
        was_below_top = df['close_D'].shift(1) <= box_top.shift(1)
        is_above_top = df['close_D'] > box_top
        states['BOX_EVENT_BREAKOUT'] = is_valid_box & is_above_top & was_below_top
        
        was_above_bottom = df['close_D'].shift(1) >= box_bottom.shift(1)
        is_below_bottom = df['close_D'] < box_bottom
        states['BOX_EVENT_BREAKDOWN'] = is_valid_box & is_below_bottom & was_above_bottom
        
        # --- 步骤4: 诊断“健康箱体盘整”状态 (逻辑不变) ---
        ma_params = self._get_params_block(params, 'ma_state_params', {})
        mid_ma_period = self._get_param_value(ma_params.get('mid_ma'), 55)
        mid_ma_col = f"EMA_{mid_ma_period}_D"
        
        is_in_box = (df['close_D'] < box_top) & (df['close_D'] > box_bottom)
        if mid_ma_col in df.columns:
            box_midpoint = (box_top + box_bottom) / 2
            is_box_above_ma = box_midpoint > df[mid_ma_col]
            states['BOX_STATE_HEALTHY_CONSOLIDATION'] = is_valid_box & is_in_box & is_box_above_ma
        else:
            # 如果没有参考均线，则只要在箱体内就算
            states['BOX_STATE_HEALTHY_CONSOLIDATION'] = is_valid_box & is_in_box
        
        # --- 步骤5: 清理与返回 ---
        for key in states:
            if key not in states or states[key] is None:
                states[key] = pd.Series(False, index=df.index)
            else:
                states[key] = states[key].fillna(False)
        
        return states

    def _diagnose_risk_factors(self, df: pd.DataFrame, params: dict) -> Dict[str, pd.Series]:
        """
        【V90.0 信号型离场版 - 风险诊断总指挥部】
        - 核心职责: 作为所有“原子风险因子”的诊断中心。它统一调用各个具体的风险诊断模块，
                    并将结果汇总成一个字典，供后续的风险评分引擎使用。
        - 核心升级: 正式集成了第一个高级风险信号“高位放量长上影派发”，开启了主动型、信号型离场策略的新篇章。
        """
        # 打印引擎启动信息，并使用 V90.0 版本号
        print("    - [风险诊断引擎 V90.0] 启动，开始诊断所有原子风险因子...")
        
        # 从主参数中获取出场策略的配置块
        exit_params = self.exit_strategy_params
        
        # 检查出场策略是否被禁用，如果禁用则直接返回空字典，提高效率
        if not self._get_param_value(exit_params.get('enabled'), False):
            print("      -> 出场策略被禁用，风险诊断跳过。")
            return {}
            
        # 初始化一个空字典，用于存放所有诊断出的风险因子（布尔型Series）
        risk_factors = {}
        
        # 创建一个默认的、全为False的Series，用于处理诊断模块未启用或失败的情况
        default_series = pd.Series(False, index=df.index)

        # --- 1. 价量形态风险 (Price-Volume Pattern Risks) ---
        #    - 这类风险关注K线和成交量的组合形态，通常是短期趋势反转的强烈信号。
        
        # 调用“高位放量长上影派发”诊断模块
        # 这是我们建立的第一个主动型离场信号，用于捕捉主力冲高派发的行为
        risk_factors['RISK_EVENT_UPTHRUST_DISTRIBUTION'] = self._diagnose_upthrust_distribution(df, exit_params)
        
        # --- 2. 市场结构风险 (Market Structure Risks) ---
        #    - 这类风险关注趋势线、关键支撑/阻力位的突破情况，代表着中长期趋势的改变。
        risk_factors['RISK_EVENT_STRUCTURE_BREAKDOWN'] = self._diagnose_structure_breakdown(df, exit_params)
        
        # 预留位置：未来可以添加诊断“结构性破位”的模块
        # 例如：跌破重要的上升趋势线或关键的颈线位
        # risk_factors['RISK_EVENT_STRUCTURE_BREAKDOWN'] = self._diagnose_structure_breakdown(df, params)
        
        # --- 3. 指标背离与超买风险 (Indicator Divergence & Overbought Risks) ---
        #    - 这类风险通过观察RSI, MACD, BIAS等摆动指标，判断市场是否进入情绪过热或动能衰竭的状态。
        
        # 预留位置：未来可以添加诊断“顶背离”的模块
        # risk_factors['RISK_STATE_DIVERGENCE_WINDOW'] = self._diagnose_top_divergence_window(df, params)
        # risk_factors['RISK_EVENT_TOP_DIVERGENCE'] = self._diagnose_top_divergence_event(df, params)
        
        # 预留位置：未来可以添加诊断“乖离率过高”的模块
        # risk_factors['RISK_STATE_BIAS_OVERBOUGHT'] = self._diagnose_bias_overbought(df, params)
        
        # 预留位置：未来可以添加诊断“RSI超买”的模块
        # risk_factors['RISK_STATE_RSI_OVERBOUGHT'] = self._diagnose_rsi_overbought(df, params)
        
        # --- 4. 筹码分布风险 (Chip Distribution Risks) ---
        chip_states = self._diagnose_chip_states(df, params)
        risk_factors['CHIP_RISK_EXHAUSTION'] = chip_states.get('CHIP_RISK_EXHAUSTION', pd.Series(False, index=df.index))
        risk_factors['CHIP_RISK_DIVERGENCE'] = chip_states.get('CHIP_RISK_DIVERGENCE', pd.Series(False, index=df.index))
        
        # 预留位置：未来可以添加诊断“高位筹码发散”的模块
        # risk_factors['RISK_STATE_CHIP_SCATTERING_HIGH'] = self._diagnose_chip_scattering_high(df, params)
        
        # 打印引擎结束信息
        print("    - [风险诊断引擎 V90.0] 所有风险因子诊断完成。")
        
        # 返回包含所有风险信号的字典
        return risk_factors

    def _diagnose_board_patterns(self, df: pd.DataFrame, params: dict) -> Dict[str, pd.Series]:
        """
        【V58.0 诊断模块 - 板形态诊断引擎】
        """
        # print("        -> [诊断模块] 正在执行板形态诊断...")
        states = {}
        p = self._get_params_block(params, 'board_pattern_params', {})
        if not self._get_param_value(p.get('enabled'), False):
            return states
        prev_close = df['close_D'].shift(1)
        limit_up_threshold = self._get_param_value(p.get('limit_up_threshold'), 0.098)
        limit_down_threshold = self._get_param_value(p.get('limit_down_threshold'), -0.098)
        price_buffer = self._get_param_value(p.get('price_buffer'), 0.005)
        limit_up_price = prev_close * (1 + limit_up_threshold)
        limit_down_price = prev_close * (1 + limit_down_threshold)
        is_limit_up_close = df['close_D'] >= limit_up_price * (1 - price_buffer)
        is_limit_up_high = df['high_D'] >= limit_up_price * (1 - price_buffer)
        is_limit_down_low = df['low_D'] <= limit_down_price * (1 + price_buffer)
        states['BOARD_EVENT_EARTH_HEAVEN'] = is_limit_down_low & is_limit_up_close
        
        # signal = states['BOARD_EVENT_EARTH_HEAVEN']
        # dates_str = self._format_debug_dates(signal)
        # print(f"          -> '地天板' 事件诊断完成，发现 {signal.sum()} 天。{dates_str}")
        
        is_limit_down_close = df['close_D'] <= limit_down_price * (1 + price_buffer)
        states['BOARD_EVENT_HEAVEN_EARTH'] = is_limit_up_high & is_limit_down_close
        
        # signal = states['BOARD_EVENT_HEAVEN_EARTH']
        # dates_str = self._format_debug_dates(signal)
        # print(f"          -> '天地板' 事件诊断完成，发现 {signal.sum()} 天。{dates_str}")
        
        return states

    def _diagnose_kline_patterns(self, df: pd.DataFrame, params: dict) -> Dict[str, pd.Series]:
        """
        【V60.3 状态持久化修复版】
        """
        # print("        -> [诊断模块] 正在执行K线组合形态诊断...")
        states = {}
        p = self.kline_params
        if not self._get_param_value(p.get('enabled'), False):
            return states
        p_washout = p.get('washout_params', {})
        if self._get_param_value(p_washout.get('enabled'), True):
            lookback = self._get_param_value(p_washout.get('lookback'), 60)
            pct_thresh = self._get_param_value(p_washout.get('pct_change_threshold'), -0.05)
            vol_multiplier = self._get_param_value(p_washout.get('volume_multiplier'), 1.5)
            window_days = self._get_param_value(p_washout.get('window_days'), 15)
            is_large_decline = df['pct_change_D'] <= pct_thresh
            is_high_volume = df['volume_D'] > df['VOL_MA_21_D'] * vol_multiplier
            is_near_low = df['close_D'] <= df['low_D'].rolling(window=lookback).min() * 1.05
            washout_event = is_large_decline & is_high_volume & is_near_low
            states['KLINE_EVENT_WASHOUT'] = washout_event
            
            # signal = states['KLINE_EVENT_WASHOUT']
            # dates_str = self._format_debug_dates(signal)
            # print(f"          -> '巨阴洗盘' 事件诊断完成，发现 {signal.sum()} 天。{dates_str}")
            
            counter = pd.Series(0, index=df.index)
            counter[washout_event] = window_days
            counter = counter.replace(0, np.nan).ffill().fillna(0)
            days_in_window = counter.groupby(washout_event.cumsum()).cumcount()
            states['KLINE_STATE_WASHOUT_WINDOW'] = (days_in_window < window_days) & (counter > 0)
            
            # signal = states['KLINE_STATE_WASHOUT_WINDOW']
            # dates_str = self._format_debug_dates(signal)
            # print(f"          -> '巨阴反转观察窗口' 持久化状态诊断完成，共激活 {signal.sum()} 天。{dates_str}")
            
        p_gap = p.get('gap_support_params', {})
        if self._get_param_value(p_gap.get('enabled'), True):
            gap_up_event = df['low_D'] > df['high_D'].shift(1)
            gap_support_level = df['high_D'].shift(1)
            break_condition = df['low_D'] <= gap_support_level
            persistence_days = self._get_param_value(p_gap.get('persistence_days'), 10)
            states['KLINE_STATE_GAP_SUPPORT_ACTIVE'] = self._create_persistent_state(
                df, entry_event=gap_up_event, persistence_days=persistence_days, break_condition=break_condition
            )
            
            # signal = states['KLINE_STATE_GAP_SUPPORT_ACTIVE']
            # dates_str = self._format_debug_dates(signal)
            # print(f"          -> '缺口支撑有效' 状态诊断完成，共激活 {signal.sum()} 天。{dates_str}")
            
        p_nshape = p.get('n_shape_params', {})
        if self._get_param_value(p_nshape.get('enabled'), True):
            rally_threshold = self._get_param_value(p_nshape.get('rally_threshold'), 0.097)
            consolidation_days_max = self._get_param_value(p_nshape.get('consolidation_days_max'), 3)
            n_shape_start_event = df['pct_change_D'] >= rally_threshold
            rally_open = df['open_D'].where(n_shape_start_event, np.nan)
            rally_high = df['high_D'].where(n_shape_start_event, np.nan)
            rally_open_ffill = rally_open.ffill()
            rally_high_ffill = rally_high.ffill()
            is_after_rally = n_shape_start_event.cumsum() > 0
            is_holding_above_open = df['close_D'] > rally_open_ffill
            is_not_new_rally = ~n_shape_start_event
            potential_consolidation = is_after_rally & is_holding_above_open & is_not_new_rally
            days_since_rally = potential_consolidation.groupby(n_shape_start_event.cumsum()).cumcount() + 1
            states['KLINE_STATE_N_SHAPE_CONSOLIDATION'] = potential_consolidation & (days_since_rally <= consolidation_days_max)
            states['KLINE_N_SHAPE_RALLY_HIGH'] = rally_high_ffill.where(states['KLINE_STATE_N_SHAPE_CONSOLIDATION'])
            
            # signal = states['KLINE_STATE_N_SHAPE_CONSOLIDATION']
            # dates_str = self._format_debug_dates(signal)
            # print(f"          -> 'N字形态整理期' 状态诊断完成，共激活 {signal.sum()} 天。{dates_str}")
            
        # print("        -> [诊断模块] K线组合形态诊断执行完毕。")
        return states

    # ▼▼▼ 独立的“战术预警”诊断模块 ▼▼▼
    def _check_tactical_alerts(self, row: pd.Series, params: dict) -> Tuple[int, str]:
        """
        【V202.0 动态防御版】战术预警诊断模块
        - 核心升级: 新增对动态筹码斜率的实时监控，提供更灵敏的盘中风险预警。
        """
        alert_params = self._get_params_block(params, 'alert_system_params', {})
        if not self._get_param_value(alert_params.get('enabled'), True):
            return 0, "No Alert"

        # --- 预警剧本1: 筹码发散 (静态风险) ---
        chip_dispersion_params = alert_params.get('chip_dispersion', {})
        conc_slope = getattr(row, 'CHIP_concentration_90pct_slope_5d_D', 0)
        if conc_slope > self._get_param_value(chip_dispersion_params.get('level_3_threshold'), 0.01):
            return 3, f"筹码严重发散(斜率:{conc_slope:.4f})"
        if conc_slope > self._get_param_value(chip_dispersion_params.get('level_2_threshold'), 0.005):
            return 2, f"筹码持续发散(斜率:{conc_slope:.4f})"
        if conc_slope > self._get_param_value(chip_dispersion_params.get('level_1_threshold'), 0.001):
            return 1, f"筹码初步发散(斜率:{conc_slope:.4f})"

        # ▼▼▼ 动态筹码风险预警 ▼▼▼
        # --- 预警剧本2: 获利盘蒸发 (动态风险) ---
        winner_rate_slope = getattr(row, 'SLOPE_5_CHIP_total_winner_rate_D', 0)
        if winner_rate_slope < -5.0: # 如果获利盘每天减少超过5个百分点
            return 3, f"获利盘快速蒸发(斜率:{winner_rate_slope:.2f})"
        if winner_rate_slope < -1.0: # 如果获利盘每天减少超过1个百分点
            return 2, f"获利盘开始萎缩(斜率:{winner_rate_slope:.2f})"

        # --- 预警剧本3: 主峰瓦解 (动态风险) ---
        stability_slope = getattr(row, 'SLOPE_5_CHIP_peak_stability_D', 0)
        if stability_slope < -0.1: # 稳定性斜率出现显著负值
            return 2, f"主峰稳定性瓦解(斜率:{stability_slope:.2f})"

        # --- 预警剧本4: 关键支撑失守 (结构性风险) ---
        support_break_params = alert_params.get('support_break', {})
        short_ma_col = f"EMA_{self._get_param_value(support_break_params.get('short_ma'), 13)}_D"
        mid_ma_col = f"EMA_{self._get_param_value(support_break_params.get('mid_ma'), 55)}_D"
        if getattr(row, 'close_D', 0) < getattr(row, mid_ma_col, float('inf')):
            return 3, f"失守中期生命线({mid_ma_col})"
        if getattr(row, 'close_D', 0) < getattr(row, short_ma_col, float('inf')):
            return 2, f"失守短期趋势线({short_ma_col})"

        # --- 预警剧本5: 资金流顶背离 ---
        if getattr(row, 'DYN_TREND_TOPPING_DIVERGENCE', False):
             return 2, "动态趋势呈现顶背离"

        return 0, "No Alert"

    # ▼▼▼【代码新增 V190.0】: 全新的、支持动态仓位管理的模拟引擎 ▼▼▼
    def _run_position_management_simulation(self, df: pd.DataFrame, params: dict) -> pd.DataFrame:
        """
        【V190.0 新增】战术持仓管理模拟引擎
        - 核心功能: 模拟一个完整的交易周期，包括入场、持仓、风险预警、动态减仓和最终离场。
        - 输出: 在DataFrame中增加仓位状态、预警级别、交易动作等列，用于最终分析。
        """
        print("\n" + "="*20 + " 【战术持仓管理模拟引擎 V190.0】启动 " + "="*20)
        
        # --- 1. 参数初始化 ---
        sim_params = self._get_params_block(params, 'position_management_params', {})
        if not self._get_param_value(sim_params.get('enabled'), False):
            print("    - 持仓管理模拟被禁用，跳过。")
            return df

        # 从配置中读取减仓比例
        level_2_reduction = self._get_param_value(sim_params.get('level_2_alert_reduction_pct'), 0.3)
        level_3_reduction = self._get_param_value(sim_params.get('level_3_alert_reduction_pct'), 0.5)
        
        # 初始化状态列
        df['position_size'] = 0.0
        df['alert_level'] = 0
        df['alert_reason'] = ''
        df['trade_action'] = ''

        # 初始化状态变量
        in_position = False
        position_size = 0.0
        entry_price = 0.0
        partial_exit_level_2_done = False # 标记L2减仓是否已执行，避免重复减仓

        # --- 2. 逐日迭代，模拟交易状态机 ---
        for row in df.itertuples():
            current_date = row.Index
            
            if not in_position:
                # --- 入场逻辑 ---
                if row.signal_entry:
                    in_position = True
                    position_size = 1.0
                    entry_price = row.close_D
                    partial_exit_level_2_done = False # 重置减仓标记
                    df.loc[current_date, 'trade_action'] = 'ENTRY'
            else:
                # --- 持仓期间的逻辑 ---
                
                # a. 检查硬性离场信号 (最高优先级)
                if row.exit_signal_code > 0:
                    in_position = False
                    position_size = 0.0
                    df.loc[current_date, 'trade_action'] = f'EXIT (Code:{row.exit_signal_code})'
                    df.loc[current_date, 'position_size'] = position_size
                    continue # 当天直接离场，不再执行后续预警判断

                # b. 检查战术预警信号
                alert_level, alert_reason = self._check_tactical_alerts(row, params)
                df.loc[current_date, 'alert_level'] = alert_level
                df.loc[current_date, 'alert_reason'] = alert_reason

                # c. 根据预警级别执行仓位调整
                if alert_level == 3: # 红色预警
                    if position_size > 0:
                        reduction_amount = position_size * level_3_reduction
                        position_size -= reduction_amount
                        df.loc[current_date, 'trade_action'] = f'REDUCE_L3 ({level_3_reduction:.0%})'
                
                elif alert_level == 2 and not partial_exit_level_2_done: # 橙色预警，且尚未执行过
                    if position_size > 0:
                        reduction_amount = position_size * level_2_reduction
                        position_size -= reduction_amount
                        df.loc[current_date, 'trade_action'] = f'REDUCE_L2 ({level_2_reduction:.0%})'
                        partial_exit_level_2_done = True # 标记已执行
                
                # d. 如果没有交易动作，则标记为持仓
                if df.loc[current_date, 'trade_action'] == '':
                    df.loc[current_date, 'trade_action'] = 'HOLD'

            # 更新每日的最终仓位
            df.loc[current_date, 'position_size'] = position_size
        
        print("="*25 + " 【持仓管理模拟】执行完毕 " + "="*25 + "\n")
        return df# ▼▼▼【代码新增 V190.0】: 新增独立的“战术预警”诊断模块 ▼▼▼

    def _check_tactical_alerts(self, row: pd.Series, params: dict) -> Tuple[int, str]:
        """
        【V190.0 新增】战术预警诊断模块
        - 职责: 在持仓期间，每日检查是否存在风险信号，并返回对应的预警级别和原因。
        - 返回: (alert_level, alert_reason) -> (整数, 字符串)
                 - 0: 无警报
                 - 1: 黄色预警 (观察)
                 - 2: 橙色预警 (准备减仓)
                 - 3: 红色预警 (立即行动)
        """
        alert_params = self._get_params_block(params, 'alert_system_params', {})
        if not self._get_param_value(alert_params.get('enabled'), True):
            return 0, "No Alert"

        # --- 预警剧本1: 筹码发散 ---
        chip_dispersion_params = alert_params.get('chip_dispersion', {})
        conc_slope = getattr(row, 'CHIP_concentration_90pct_slope_5d_D', 0)
        
        if conc_slope > self._get_param_value(chip_dispersion_params.get('level_3_threshold'), 0.01):
            return 3, f"筹码严重发散(斜率:{conc_slope:.4f})"
        if conc_slope > self._get_param_value(chip_dispersion_params.get('level_2_threshold'), 0.005):
            return 2, f"筹码持续发散(斜率:{conc_slope:.4f})"
        if conc_slope > self._get_param_value(chip_dispersion_params.get('level_1_threshold'), 0.001):
            return 1, f"筹码初步发散(斜率:{conc_slope:.4f})"

        # --- 预警剧本2: 关键支撑失守 ---
        support_break_params = alert_params.get('support_break', {})
        short_ma_col = f"EMA_{self._get_param_value(support_break_params.get('short_ma'), 13)}_D"
        mid_ma_col = f"EMA_{self._get_param_value(support_break_params.get('mid_ma'), 55)}_D"
        
        if getattr(row, 'close_D', 0) < getattr(row, mid_ma_col, float('inf')):
            return 3, f"失守中期生命线({mid_ma_col})"
        if getattr(row, 'close_D', 0) < getattr(row, short_ma_col, float('inf')):
            return 2, f"失守短期趋势线({short_ma_col})"

        # --- 预警剧本3: 资金流顶背离 ---
        # (此处为简化逻辑，实际可做得更复杂，如连续N日背离)
        if getattr(row, 'DYN_TREND_TOPPING_DIVERGENCE', False):
             return 2, "动态趋势呈现顶背离"

        return 0, "No Alert"

    # ▼▼▼ 风险剧本的静态“蓝图”知识库 ▼▼▼
    def _get_risk_playbook_blueprints(self) -> List[Dict]:
        """
        【V202.0 动态防御版】
        - 核心升级: 新增了三个基于动态筹码斜率的风险剧本，用于量化趋势衰竭的风险。
        """
        return [
            # --- 结构性风险 (Structure Risk) ---
            {'name': 'STRUCTURE_BREAKDOWN', 'cn_name': '【结构】关键支撑破位', 'score': 100},
            {'name': 'UPTHRUST_DISTRIBUTION', 'cn_name': '【结构】冲高派发', 'score': 80},
            # --- 动能衰竭风险 (Momentum Exhaustion) ---
            {'name': 'CHIP_EXHAUSTION', 'cn_name': '【动能】筹码成本加速衰竭', 'score': 60},
            {'name': 'CHIP_DIVERGENCE', 'cn_name': '【动能】筹码顶背离', 'score': 70},
            # ▼▼▼ 动态风险剧本 ▼▼▼
            {
                'name': 'PROFIT_EVAPORATION', 'cn_name': '【动态】获利盘蒸发', 'score': 75,
                'comment': '总获利盘斜率转负，市场赚钱效应快速消失，是强烈的离场信号。'
            },
            {
                'name': 'PEAK_WEAKENING', 'cn_name': '【动态】主峰根基动摇', 'score': 55,
                'comment': '主筹码峰的稳定性或占比开始下降，主力阵地可能在瓦解。'
            },
            {
                'name': 'RESISTANCE_BUILDING', 'cn_name': '【动态】上方压力积聚', 'score': 35,
                'comment': '上方套牢盘不减反增，表明进攻受阻，后续突破难度加大。'
            }
        ]

    # ▼▼▼ 风险剧本探针函数 ▼▼▼
    def _probe_risk_score_details(self, total_risk_score: pd.Series, risk_details_df: pd.DataFrame, params: dict):
        """
        【V93.1 探针时间限制版】
        - 核心升级: 增加了可配置的时间限制，默认只显示指定日期之后的信息，使日志更聚焦。
        """
        print("\n" + "-"*25 + " 风险剧本探针 (Risk Playbook Probe) " + "-"*24)
        
        # ▼▼▼【代码修改 V93.1】: 增加时间限制逻辑 ▼▼▼
        # 1. 从配置中获取探针的起始日期，并提供一个符合您需求的默认值
        probe_start_date_str = self._get_param_value(self.debug_params.get('probe_start_date'), '2024-12-21')
        print(f"    -> 探针时间范围: 从 {probe_start_date_str} 开始")

        # 2. 筛选出所有有风险分的日期
        key_dates = total_risk_score.index[total_risk_score > 0]
        
        # 3. 应用时间过滤器
        if probe_start_date_str:
            try:
                start_date = pd.to_datetime(probe_start_date_str, utc=True)
                key_dates = key_dates[key_dates >= start_date]
            except Exception as e:
                # 这个异常处理现在不太可能被触发，但保留它是个好习惯
                print(f"      -> [警告] 探针起始日期 '{probe_start_date_str}' 格式错误，将显示所有风险日。错误: {e}")

        if key_dates.empty:
            print(f"    -> 在 {probe_start_date_str} 之后，未发现任何已触发的风险剧本。")
        else:
            risk_playbook_cn_map = {p['name']: p.get('cn_name', p['name']) for p in self.risk_playbook_blueprints}
            
            print(f"    -> 在指定时间范围内发现 {len(key_dates)} 个风险日，详情如下:")
            for date in key_dates:
                total_score = total_risk_score.loc[date]
                print(f"    {date.strftime('%Y-%m-%d')} | 最终风险分: {total_score:.0f}")
                
                triggered_playbooks = risk_details_df.loc[date][risk_details_df.loc[date] > 0]
                for name, score in triggered_playbooks.items():
                    cn_name = risk_playbook_cn_map.get(name, name)
                    print(f"                 -> 触发: '{cn_name}', 风险分 +{score:.0f}")

    # 风险评分引擎
    def _calculate_risk_score(self, df: pd.DataFrame, params: dict, risk_setups: Dict[str, pd.Series], risk_triggers: Dict[str, pd.Series]) -> Tuple[pd.Series, pd.DataFrame]:
        """
        【V122.1 计分逻辑修复版】
        - 核心修复: 彻底重写了风险评分逻辑。现在它能够正确地遍历所有风险剧本蓝图，
                    查找对应的风险准备状态(risk_setups)，并根据蓝图中定义的'score'
                    进行累加计分。这解决了之前版本中风险信号被识别但无法计分的根本性问题。
        """
        print("    - [风险评分引擎 V122.1 计分逻辑修复版] 启动，开始量化每日风险...")
        
        risk_score = pd.Series(0.0, index=df.index)
        risk_details_df = pd.DataFrame(index=df.index)
        
        # 步骤1: 遍历所有风险剧本蓝图，并根据准备状态计分
        print("      -> 步骤1: 评估所有战术风险剧本...")
        risk_playbooks = self._get_risk_playbook_blueprints()
        
        for playbook in risk_playbooks:
            playbook_name = playbook['name']
            playbook_score = playbook.get('score', 0)
            
            # 构建对应的风险准备状态(setup)的列名
            # 例如，剧本 'PROFIT_EVAPORATION' -> 状态 'SETUP_RISK_PROFIT_EVAPORATION'
            setup_col_name = f"SETUP_RISK_{playbook_name}"
            
            # 检查这个风险准备状态是否存在且被激活
            if setup_col_name in risk_setups and risk_setups[setup_col_name].any():
                mask = risk_setups[setup_col_name]
                
                # 累加风险分
                risk_score.loc[mask] += playbook_score
                
                # 在详情报告中记录该项风险的分值
                detail_col_name = f"RISK_SCORE_{playbook_name}"
                risk_details_df.loc[mask, detail_col_name] = playbook_score

        # 步骤2: 应用战略风险放大器 (逻辑保持不变)
        print("      -> 步骤2: 启动战略风险放大器...")
        strategic_risk_params = self._get_params_block(params, 'strategic_risk_amplifier', {})
        if self._get_param_value(strategic_risk_params.get('enabled'), False):
            risk_context_col = 'filter_strategic_risk_veto_W' # 来自周线引擎的战略风险信号
            if risk_context_col in df.columns:
                mask = df[risk_context_col] == True
                penalty_points = self._get_param_value(strategic_risk_params.get('penalty_points'), 50)
                risk_score.loc[mask] += penalty_points
                risk_details_df.loc[mask, 'STRATEGIC_RISK_PENALTY'] = penalty_points
                if mask.any():
                    print(f"        -> “战略风险放大器”激活！因整体趋势恶化，在 {mask.sum()} 天的风险分上追加了 {penalty_points} 分。")

        risk_details_df.fillna(0, inplace=True)
        
        max_score = risk_score.max()
        print(f"    - [风险评分引擎 V122.1] 风险评分完成，最高风险分: {max_score if pd.notna(max_score) else 0:.0f}")
        
        return risk_score, risk_details_df

    # ▼▼▼ “风险准备状态”诊断中心 ▼▼▼
    def _diagnose_risk_setups(self, df: pd.DataFrame, params: dict, atomic_states: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """
        【V202.0 动态防御版】
        - 核心升级: 新增对动态筹码斜率风险的诊断逻辑，生成新的“风险准备状态”。
        """
        print("    - [风险前哨站 V202.0 动态防御版] 启动...")
        risk_setups = {}
        default_series = pd.Series(False, index=df.index)
        exit_params = self.exit_strategy_params
        if not self._get_param_value(exit_params.get('enabled'), False):
            print("      -> 出场策略被禁用，风险诊断跳过。")
            return {}

        # --- 1. 结构性风险诊断 (逻辑不变) ---
        risk_setups['SETUP_RISK_STRUCTURE_BREAKDOWN'] = self._diagnose_structure_breakdown(df, exit_params)
        risk_setups['SETUP_RISK_UPTHRUST_DISTRIBUTION'] = self._diagnose_upthrust_distribution(df, exit_params)

        # --- 2. 动能衰竭风险诊断 (逻辑不变) ---
        risk_setups['SETUP_RISK_CHIP_EXHAUSTION'] = atomic_states.get('CHIP_RISK_EXHAUSTION', default_series)
        risk_setups['SETUP_RISK_CHIP_DIVERGENCE'] = atomic_states.get('CHIP_RISK_DIVERGENCE', default_series)

        # ▼▼▼ 诊断新的动态风险 ▼▼▼
        # 准备所需的斜率列名
        total_winner_rate_slope_col = 'SLOPE_5_CHIP_total_winner_rate_D'
        peak_stability_slope_col = 'SLOPE_5_CHIP_peak_stability_D'
        peak_percent_slope_col = 'SLOPE_5_CHIP_peak_percent_D'
        pressure_above_slope_col = 'SLOPE_5_CHIP_pressure_above_D'
        
        required_cols = [total_winner_rate_slope_col, peak_stability_slope_col, peak_percent_slope_col, pressure_above_slope_col]
        if not all(col in df.columns for col in required_cols):
            print("      -> [警告] 缺少诊断动态风险所需的斜率列，相关诊断将跳过。")
        else:
            # 诊断1: 获利盘蒸发
            risk_setups['SETUP_RISK_PROFIT_EVAPORATION'] = (df[total_winner_rate_slope_col] < 0)
            # 诊断2: 主峰根基动摇
            risk_setups['SETUP_RISK_PEAK_WEAKENING'] = (df[peak_stability_slope_col] < 0) | (df[peak_percent_slope_col] < 0)
            # 诊断3: 上方压力积聚
            risk_setups['SETUP_RISK_RESISTANCE_BUILDING'] = (df[pressure_above_slope_col] > 0)
        # 打印诊断结果
        for name, series in risk_setups.items():
            if series.any():
                print(f"      -> 风险准备 '{name}' 已激活 {series.sum()} 天。")

        return risk_setups

    # ▼▼▼ “风险触发事件”定义中心 ▼▼▼
    def _define_risk_triggers(self, df: pd.DataFrame, params: dict) -> Dict[str, pd.Series]:
        """
        【V146.0 风险模型校准版】
        - 核心修复: 彻底重写了 'RISK_TRIGGER_BOUNCE_FAILED_CANDLE' (上攻失败) 的触发逻辑。
                    旧逻辑过于敏感，会将健康的突破后盘整误判为高风险事件。
        - 新逻辑:
          1. (上攻失败) 不仅要求创近期新高后收阴，还必须有足够长的上影线，代表明确的卖压。
          2. (趋势衰减) 定义为收盘价有效跌破短期生命线(如EMA21)，且动能指标确认走弱。
          3. (结构破位) 定义为创近期新低，并伴有下跌动能确认。
          4. (平台破位) 定义为收盘价有效跌破已形成的稳定筹码平台。
        - 收益: 大幅提升了风险模型的准确性，使其能精准识别真实风险，避免因“过度防御”
                而错失类似06-24日的有效突破信号。
        """
        print("    - [风险触发事件定义中心 V146.0 风险模型校准版] 启动...")
        triggers = {}
        exit_params = self.exit_strategy_params
        default_series = pd.Series(False, index=df.index)

        triggers['RISK_TRIGGER_ANY'] = pd.Series(True, index=df.index)

        # --- 触发器1: 平台破位 (Platform Breakdown) ---
        platform_price_col = 'PLATFORM_PRICE_STABLE' # 修正了此处可能存在的语法错误
        if platform_price_col in df.columns:
            # 价格有效跌破已形成的稳定平台价格
            triggers['RISK_TRIGGER_PLATFORM_BREAKDOWN'] = df['close_D'] < df[platform_price_col]
        else:
            triggers['RISK_TRIGGER_PLATFORM_BREAKDOWN'] = default_series

        # --- 触发器2: 上攻失败/冲高回落 (Bounce Failed) ---
        p_failed_bounce = exit_params.get('bounce_failed_candle_params', {})
        if self._get_param_value(p_failed_bounce.get('enabled'), True):
            lookback = self._get_param_value(p_failed_bounce.get('lookback_days'), 5)
            
            # 条件A: 当天创下N日新高
            is_new_high_today = df['high_D'] >= df['high_D'].shift(1).rolling(window=lookback, min_periods=min(lookback, 2)).max()
            
            # 条件B: 当天是阴线 (收盘价低于开盘价)
            is_red_candle = df['close_D'] < df['open_D']
            
            # 条件C: 具有明显的上影线 (上影线长度占总振幅的比例 > 阈值)
            body_range = (df['high_D'] - df['low_D']).replace(0, np.nan)
            # 健壮地计算上影线长度
            upper_shadow_len = df['high_D'] - np.maximum(df['open_D'], df['close_D'])
            upper_shadow_ratio = upper_shadow_len / body_range
            min_shadow_ratio = self._get_param_value(p_failed_bounce.get('min_upper_shadow_ratio'), 0.4)
            has_long_upper_shadow = upper_shadow_ratio > min_shadow_ratio

            # 最终触发器 = 创了新高 AND 是阴线 AND 有长上影线
            triggers['RISK_TRIGGER_BOUNCE_FAILED_CANDLE'] = is_new_high_today & is_red_candle & has_long_upper_shadow
        else:
            triggers['RISK_TRIGGER_BOUNCE_FAILED_CANDLE'] = default_series

        # --- 触发器3: 趋势衰减 (Trend Decay) ---
        p_decay = exit_params.get('trend_decay_params', {})
        if self._get_param_value(p_decay.get('enabled'), True):
            ma_period = self._get_param_value(p_decay.get('ma_period'), 21)
            ma_col = f'EMA_{ma_period}_D'
            macd_h_col = 'MACDh_13_34_8_D'
            if ma_col in df.columns and macd_h_col in df.columns:
                # 条件A: 收盘价有效跌破短期生命线
                is_breaking_ma = df['close_D'] < df[ma_col]
                # 条件B: MACD柱状线翻绿或持续为绿，确认动能走弱
                is_macd_weak = df[macd_h_col] < 0
                triggers['RISK_TRIGGER_TREND_DECAY'] = is_breaking_ma & is_macd_weak
            else:
                triggers['RISK_TRIGGER_TREND_DECAY'] = default_series
        
        # --- 触发器4: 真实结构破位 (True Structure Breakdown) ---
        p_breakdown = exit_params.get('true_breakdown_params', {})
        if self._get_param_value(p_breakdown.get('enabled'), True):
            lookback = self._get_param_value(p_breakdown.get('lookback_days'), 21)
            # 条件A: 创近期新低
            is_new_low = df['close_D'] <= df['low_D'].shift(1).rolling(window=lookback, min_periods=min(lookback, 2)).min()
            # 条件B: 下跌动能为负 (例如长期均线斜率为负)
            long_ma_slope_col = 'SLOPE_21_EMA_89_D'
            if long_ma_slope_col in df.columns:
                is_momentum_negative = df[long_ma_slope_col] < 0
                triggers['RISK_TRIGGER_TRUE_BREAKDOWN_CANDLE'] = is_new_low & is_momentum_negative
            else:
                triggers['RISK_TRIGGER_TRUE_BREAKDOWN_CANDLE'] = default_series
        
        # --- 触发器5: 急跌回调 (Sharp Pullback) ---
        p_pullback = exit_params.get('sharp_pullback_params', {})
        if self._get_param_value(p_pullback.get('enabled'), True):
            pct_change_thresh = self._get_param_value(p_pullback.get('pct_change_threshold'), -0.04)
            vol_multiplier = self._get_param_value(p_pullback.get('volume_multiplier'), 1.2)
            vol_ma_col = 'VOL_MA_21_D'
            if vol_ma_col in df.columns:
                # 条件A: 跌幅足够大
                is_large_decline = df['pct_change_D'] < pct_change_thresh
                # 条件B: 成交量放大
                is_volume_high = df['volume_D'] > df[vol_ma_col] * vol_multiplier
                triggers['RISK_TRIGGER_SHARP_PULLBACK_CANDLE'] = is_large_decline & is_volume_high
            else:
                triggers['RISK_TRIGGER_SHARP_PULLBACK_CANDLE'] = default_series

        # --- 清理与返回 ---
        for key in triggers:
            if key not in triggers or triggers[key] is None:
                triggers[key] = pd.Series(False, index=df.index)
            else:
                triggers[key] = triggers[key].fillna(False)
        
        print("    - [风险触发事件定义中心 V146.0] 所有风险触发器已校准并定义完成。")
        return triggers

    # ▼▼▼ “上攻乏力”风险诊断模块 ▼▼▼
    def _diagnose_upthrust_distribution(self, df: pd.DataFrame, exit_params: dict) -> pd.Series:
        """
        【V91.2 函数调用修复版】
        - 核心修复: 使用 numpy.maximum 替代错误的 pd.max，以正确计算上影线。
        """
        p = exit_params.get('upthrust_distribution_params', {})
        if not self._get_param_value(p.get('enabled'), False):
            return pd.Series(False, index=df.index)

        overextension_ma_period = self._get_param_value(p.get('overextension_ma_period'), 55)
        overextension_threshold = self._get_param_value(p.get('overextension_threshold'), 0.3)
        upper_shadow_ratio = self._get_param_value(p.get('upper_shadow_ratio'), 0.5)
        high_volume_quantile = self._get_param_value(p.get('high_volume_quantile'), 0.75)
        
        ma_col = f'EMA_{overextension_ma_period}_D'
        vol_ma_col = 'VOL_MA_21_D'
        
        required_cols = ['open_D', 'high_D', 'low_D', 'close_D', 'volume_D', ma_col, vol_ma_col]
        if not all(col in df.columns for col in required_cols):
            print("          -> [警告] 缺少诊断'高位放量长上影'所需列，跳过。")
            return pd.Series(False, index=df.index)

        is_overextended = (df['close_D'] / df[ma_col] - 1) > overextension_threshold
        total_range = (df['high_D'] - df['low_D']).replace(0, np.nan)
        
        # ▼▼▼【代码修改 V91.2】: 使用 np.maximum 替代 pd.max ▼▼▼
        upper_shadow = (df['high_D'] - np.maximum(df['open_D'], df['close_D']))
        # ▲▲▲【代码修改 V91.2】▲▲▲
        
        has_long_upper_shadow = (upper_shadow / total_range) >= upper_shadow_ratio
        volume_threshold = df['volume_D'].rolling(window=21).quantile(high_volume_quantile)
        is_high_volume = df['volume_D'] > volume_threshold
        is_weak_close = df['close_D'] < (df['high_D'] + df['low_D']) / 2
        
        signal = is_overextended & has_long_upper_shadow & is_high_volume & is_weak_close
        # print(f"          -> '高位放量长上影派发' 风险诊断完成，共激活 {signal.sum()} 天。{self._format_debug_dates(signal)}")
        return signal

    # ▼▼▼ “结构性破位”风险诊断模块 ▼▼▼
    def _diagnose_structure_breakdown(self, df: pd.DataFrame, exit_params: dict) -> pd.Series:
        """
        诊断“结构性破位”风险 (Structure Breakdown)。
        这是一个非常重要的趋势终结信号。
        """
        p = exit_params.get('structure_breakdown_params', {})
        if not self._get_param_value(p.get('enabled'), False):
            return pd.Series(False, index=df.index)

        # 1. 定义参数
        breakdown_ma_period = self._get_param_value(p.get('breakdown_ma_period'), 21)
        min_pct_change = self._get_param_value(p.get('min_pct_change'), -0.03)
        high_volume_quantile = self._get_param_value(p.get('high_volume_quantile'), 0.75)
        
        ma_col = f'EMA_{breakdown_ma_period}_D'
        
        required_cols = ['open_D', 'close_D', 'pct_change_D', 'volume_D', ma_col]
        if not all(col in df.columns for col in required_cols):
            print("          -> [警告] 缺少诊断'结构性破位'所需列，跳过。")
            return pd.Series(False, index=df.index)

        # 2. 计算各项条件
        # 条件A: 是一根有分量的阴线
        is_decisive_negative_candle = df['pct_change_D'] < min_pct_change
        
        # 条件B: 相对放量
        volume_threshold = df['volume_D'].rolling(window=21).quantile(high_volume_quantile)
        is_high_volume = df['volume_D'] > volume_threshold
        
        # 条件C: 跌破了关键均线
        is_breaking_ma = df['close_D'] < df[ma_col]
        
        # 3. 组合所有条件
        signal = is_decisive_negative_candle & is_high_volume & is_breaking_ma
        
        # print(f"          -> '结构性破位' 风险诊断完成，共激活 {signal.sum()} 天。{self._format_debug_dates(signal)}")
        return signal


    def _prepare_derived_features(self, df: pd.DataFrame, params: dict) -> pd.DataFrame:
        """
        【V58.0 资金博弈深度解析版】
        """
        print("    - [衍生特征中心 V58.0 资金博弈深度解析版] 启动...")
        p = self._get_params_block(params, 'derived_feature_params', {})
        if not self._get_param_value(p.get('enabled'), True):
            print("      -> 衍生特征引擎被禁用，跳过。")
            return df
        if 'net_mf_amount_D' not in df.columns or df['net_mf_amount_D'].isnull().all():
            if all(c in df.columns for c in ['buy_lg_amount_D', 'sell_lg_amount_D', 'buy_elg_amount_D', 'sell_elg_amount_D']):
                df['net_mf_amount_D'] = (df['buy_lg_amount_D'] + df['buy_elg_amount_D']) - (df['sell_lg_amount_D'] + df['sell_elg_amount_D'])
            else:
                df['net_mf_amount_D'] = 0
        if 'net_retail_amount_D' not in df.columns or df['net_retail_amount_D'].isnull().all():
            if all(c in df.columns for c in ['buy_sm_amount_D', 'sell_sm_amount_D']):
                df['net_retail_amount_D'] = df['buy_sm_amount_D'] - df['sell_sm_amount_D']
            else:
                df['net_retail_amount_D'] = 0
        df = self._ensure_numeric_types(df)
        print("      -> 核心资金流计算与类型净化完成。")
        if 'amount_D' in df.columns:
            df['net_mf_ratio_D'] = np.divide(df['net_mf_amount_D'], df['amount_D'], out=np.zeros_like(df['net_mf_amount_D'], dtype=float), where=df['amount_D']!=0)
            print(f"      -> 已计算 'net_mf_ratio_D' (主力净流入占比)。")
        if all(c in df.columns for c in ['buy_lg_amount_D', 'sell_lg_amount_D', 'buy_elg_amount_D', 'sell_elg_amount_D']):
            main_force_buy = df['buy_lg_amount_D'] + df['buy_elg_amount_D']
            main_force_sell = df['sell_lg_amount_D'] + df['sell_elg_amount_D']
            df['mf_buy_sell_ratio_D'] = np.divide(main_force_buy, main_force_sell, out=np.ones_like(main_force_buy, dtype=float), where=main_force_sell!=0)
            print(f"      -> 已计算 'mf_buy_sell_ratio_D' (主力多空比)。")
        capital_p = p.get('capital_flow_params', {})
        accum_period = self._get_param_value(capital_p.get('accumulation_period'), 20)
        df[f'mf_accumulation_{accum_period}_D'] = df['net_mf_amount_D'].rolling(window=accum_period).sum()
        print(f"      -> 已计算 'mf_accumulation_{accum_period}_D' ({accum_period}日主力资金累积)。")
        zscore_period = self._get_param_value(p.get('momentum_params', {}).get('zscore_period'), 20)
        rolling_mean = df['net_mf_amount_D'].rolling(window=zscore_period).mean()
        rolling_std = df['net_mf_amount_D'].rolling(window=zscore_period).std()
        df['net_mf_zscore_D'] = np.divide((df['net_mf_amount_D'] - rolling_mean), rolling_std, out=np.zeros_like(df['net_mf_amount_D'], dtype=float), where=rolling_std!=0)
        print(f"      -> 已计算 'net_mf_zscore_D' ({zscore_period}日主力资金Z-Score)。")
        df['mf_retail_divergence_D'] = np.divide(df['net_mf_amount_D'], df['net_retail_amount_D'].abs(), out=np.zeros_like(df['net_mf_amount_D'], dtype=float), where=df['net_retail_amount_D']!=0)
        print(f"      -> 已计算 'mf_retail_divergence_D' (机构散户背离度)。")
        divergence_p = p.get('divergence_params', {})
        price_up = df['pct_change_D'] > 0
        price_down = df['pct_change_D'] < 0
        capital_up = df['net_mf_amount_D'] > 0
        capital_down = df['net_mf_amount_D'] < 0
        conditions = [
            (price_up & capital_up),
            (price_down & capital_down),
            (price_up & capital_down),
            (price_down & capital_up)
        ]
        choices = [1, 1, -1, -1]
        df['capital_price_score_D'] = np.select(conditions, choices, default=0)
        trend_period = self._get_param_value(divergence_p.get('trend_period'), 10)
        df['capital_price_trend_D'] = df['capital_price_score_D'].rolling(window=trend_period).sum()
        print(f"      -> 已计算 'capital_price_trend_D' ({trend_period}日价资确认趋势)。")
        print("    - [衍生特征中心 V58.0] 所有衍生特征准备完成。")
        return df


    def _calculate_risk_states(self, df: pd.DataFrame, params: dict) -> Dict[str, pd.Series]:
        """
        【V41.12 新增】风险状态定义中心 (Risk State Center)
        """
        print("    - [风险状态中心 V41.12] 启动，开始计算所有风险状态...")
        risks = {}
        exit_params = self.exit_strategy_params
        try:
            risks['RISK_UPTHRUST_DISTRIBUTION'] = self._find_upthrust_distribution_exit(df, params)
            print(f"      -> '主力叛逃'风险状态定义完成，发现 {risks.get('RISK_UPTHRUST_DISTRIBUTION', pd.Series([])).sum()} 天。")
        except Exception as e:
            print(f"      -> [警告] 计算'主力叛逃'风险状态时出错: {e}")
        try:
            risks['RISK_STRUCTURE_BREAKDOWN'] = self._find_volume_breakdown_exit(df, params)
            print(f"      -> '结构崩溃'风险状态定义完成，发现 {risks.get('RISK_STRUCTURE_BREAKDOWN', pd.Series([])).sum()} 天。")
        except Exception as e:
            print(f"      -> [警告] 计算'结构崩溃'风险状态时出错: {e}")
        try:
            p = exit_params.get('divergence_exit_params', {})
            if self._get_param_value(p.get('enabled'), False):
                risks['RISK_TOP_DIVERGENCE'] = self._find_top_divergence_exit(df, params)
                print(f"      -> '顶背离'风险状态定义完成，发现 {risks.get('RISK_TOP_DIVERGENCE', pd.Series([])).sum()} 天。")
        except Exception as e:
            print(f"      -> [警告] 计算'顶背离'风险状态时出错: {e}")
        try:
            p = exit_params.get('indicator_exit_params', {})
            if self._get_param_value(p.get('enabled'), False):
                rsi_col = 'RSI_14_D'
                bias_col = 'BIAS_20_D'
                risks['RISK_RSI_OVERBOUGHT'] = df.get(rsi_col, pd.Series(False, index=df.index)) > self._get_param_value(p.get('rsi_threshold'), 85)
                risks['RISK_BIAS_OVERBOUGHT'] = df.get(bias_col, pd.Series(False, index=df.index)) > self._get_param_value(p.get('bias_threshold'), 20.0)
                print(f"      -> 'RSI超买'风险状态定义完成，发现 {risks.get('RISK_RSI_OVERBOUGHT', pd.Series([])).sum()} 天。")
        except Exception as e:
            print(f"      -> [警告] 计算'指标超买'风险状态时出错: {e}")
        all_needed_risks = [
            'RISK_UPTHRUST_DISTRIBUTION', 'RISK_STRUCTURE_BREAKDOWN',
            'RISK_TOP_DIVERGENCE', 'RISK_RSI_OVERBOUGHT', 'RISK_BIAS_OVERBOUGHT'
        ]
        for risk_name in all_needed_risks:
            if risk_name not in risks:
                risks[risk_name] = pd.Series(False, index=df.index)
        print("    - [风险状态中心 V41.13] 所有风险状态计算完成。")
        return risks

    def _create_persistent_state(
        self, 
        df: pd.DataFrame, 
        entry_event: pd.Series, 
        persistence_days: int, 
        break_condition: Optional[pd.Series] = None
    ) -> pd.Series:
        """
        【V58.0 新增 - 状态机生成器】
        """
        if not entry_event.any():
            return pd.Series(False, index=df.index)
        event_groups = entry_event.cumsum()
        days_since_event = df.groupby(event_groups).cumcount()
        is_within_persistence = days_since_event < persistence_days
        is_active_period = event_groups > 0
        persistent_state = is_active_period & is_within_persistence
        if break_condition is not None:
            break_points = break_condition & persistent_state
            break_groups = break_points.groupby(event_groups).transform('idxmax')
            persistent_state &= (df.index < break_groups) | (break_groups.isna())
        return persistent_state

    async def alpha_hunter_backtest(self, stock_code: str, df_full: pd.DataFrame, params: dict):
        """
        【V118.20 情报重建修复版】
        - 核心修复: 解决了因调用 apply_strategy 导致 'analysis_period' 列丢失的BUG。
        - 解决方案: 在所有计算和合并操作完成后，从原始的 analysis_df 中提取
                    'analysis_period' 列，并将其强制重新附加到 final_report_df 上，
                    确保该关键信息在最终报告中存在。
        """
        print("\n" + "="*30 + " [阿尔法猎手 V118.20 启动] " + "="*30)
        
        backtest_params = self._get_params_block(params, 'alpha_hunter_params', {})
        if not self._get_param_value(backtest_params.get('enabled'), False): return

        min_duration = self._get_param_value(backtest_params.get('min_duration_days'), 3)
        min_total_gain = self._get_param_value(backtest_params.get('min_total_gain_pct'), 15.0) / 100.0
        pre_days_lookback = self._get_param_value(backtest_params.get('pre_days_lookback'), 30)
        post_days_lookforward = 5
        activation_window = self._get_param_value(backtest_params.get('activation_window_days'), 2)
        output_dir_base = self._get_param_value(backtest_params.get('output_dir'), 'alpha_hunter_reports')
        
        print(f"    -> 目标波段标准: 持续 >= {min_duration} 天, 总涨幅 >= {min_total_gain*100:.1f}%")
        print(f"    -> 回溯窗口: {pre_days_lookback} 天 | 推演窗口: {post_days_lookforward} 天")

        print("\n--- [猎手阶段1] 正在扫描历史，寻找所有黄金上涨波段...")
        df_full['cum_min'] = df_full['close_D'].cummin()
        df_full['is_new_low'] = df_full['close_D'] <= df_full['cum_min']
        wave_starts = df_full['is_new_low'].shift(1).fillna(False) & ~df_full['is_new_low']
        start_indices = df_full.index[wave_starts]
        golden_waves = []
        for start_date in start_indices:
            wave_df = df_full.loc[start_date:]
            next_low_day = wave_df[wave_df['is_new_low']].first_valid_index()
            if next_low_day: wave_df = wave_df.loc[:next_low_day].iloc[:-1]
            if wave_df.empty: continue
            end_date = wave_df['close_D'].idxmax()
            wave_df = wave_df.loc[:end_date]
            duration = len(wave_df)
            total_gain = (wave_df.iloc[-1]['close_D'] / wave_df.iloc[0]['close_D']) - 1
            if duration >= min_duration and total_gain >= min_total_gain:
                golden_waves.append({'start_date': wave_df.index[0], 'end_date': end_date, 'duration': duration, 'total_gain': total_gain})
        
        print(f"    -> [扫描完成] 发现 {len(golden_waves)} 个原始黄金上涨波段。")
        target_tz = df_full.index.tz
        cutoff_date_naive = pd.Timestamp('2024-01-01')
        cutoff_date = cutoff_date_naive.tz_localize(target_tz) if target_tz else cutoff_date_naive
        print(f"    -> [时间过滤] 将只分析 {cutoff_date.date()} 之后启动的波段...")
        golden_waves = [wave for wave in golden_waves if wave['start_date'] >= cutoff_date]
        if not golden_waves:
            print("    -> [过滤完成] 未发现符合时间条件的黄金上涨波段。")
            return
        print(f"    -> [过滤完成] 发现 {len(golden_waves)} 个符合条件的波段，准备逐一回测...")

        success_count, missed_count = 0, 0
        for i, wave in enumerate(golden_waves):
            start_date = wave['start_date']
            print(f"\n--- [猎手阶段2] 正在回测第 {i+1}/{len(golden_waves)} 个波段: {start_date.date()} ---")
            start_loc = df_full.index.get_loc(start_date)
            
            start_slice = start_loc - pre_days_lookback
            end_slice = min(start_loc + 1 + post_days_lookforward, len(df_full))
            if start_slice < 0: continue
            
            analysis_df = df_full.iloc[start_slice:end_slice].copy()
            
            # 【修复点 A】在调用策略前，先在原始的 analysis_df 上创建好 analysis_period 列
            analysis_df['analysis_period'] = 'lookback'
            # 找到起涨点在当前切片中的位置
            start_date_in_slice_loc = analysis_df.index.get_loc(start_date)
            # 将起涨点之后的所有行标记为 'look_forward'
            if start_date_in_slice_loc + 1 < len(analysis_df):
                analysis_df.iloc[start_date_in_slice_loc + 1:, analysis_df.columns.get_loc('analysis_period')] = 'look_forward'

            try:
                result_df_raw, _ = self.apply_strategy(analysis_df, params)
            except Exception as e:
                print(f"    -> [严重错误] 策略推演时发生异常: {e}")
                continue

            result_df = result_df_raw.reindex(analysis_df.index).copy()

            is_activated = False
            activation_start = start_date - pd.Timedelta(days=activation_window)
            activation_period_df = result_df.loc[activation_start : start_date]
            
            if not activation_period_df.empty and activation_period_df['signal_entry'].any():
                success_count += 1
                activation_date = activation_period_df[activation_period_df['signal_entry']].index[0]
                print(f"    -> [捕获成功!] 策略在 {activation_date.date()} 成功激活信号。")
            else:
                missed_count += 1
                print(f"    -> [错失良机!] 未能捕获此波段。正在生成情报档案...")

                final_report_df = result_df.copy()
                if self._last_score_details_df is not None and not self._last_score_details_df.empty:
                    scoped_score_details = self._last_score_details_df.reindex(final_report_df.index).copy()
                    final_report_df = final_report_df.join(scoped_score_details, how='left')
                
                # 【修复点 B】从原始的 analysis_df 中提取 'analysis_period' 列并强制合并
                if 'analysis_period' in analysis_df.columns:
                    final_report_df['analysis_period'] = analysis_df['analysis_period']
                else:
                    # 作为备用方案，如果 analysis_df 也没有，则创建一个默认的
                    final_report_df['analysis_period'] = 'unknown'

                essential_columns_whitelist = [
                    'open_D', 'high_D', 'low_D', 'close_D', 'volume_D', 'pct_change_D',
                    'signal_entry', 'entry_score', 'risk_score', 'EMA_21_D', 'EMA_55_D', 
                    'EMA_89_D', 'MACD_13_34_8_D', 'MACDh_13_34_8_D', 'MACDs_13_34_8_D',
                    'CHIP_peak_cost_D', 'CHIP_winner_rate_short_term_D', 'CHIP_concentration_90pct_D',
                    'net_mf_amount_D', 'buy_elg_amount_D', 'sell_elg_amount_D',
                    'SLOPE_5_close_D', 'ACCEL_5_close_D', 'CHIP_concentration_90pct_slope_5d_D',
                    'analysis_period'
                ]
                
                # 动态添加所有得分详情列到白名单
                if self._last_score_details_df is not None:
                    score_detail_cols = [col for col in self._last_score_details_df.columns if col in final_report_df.columns]
                    essential_columns_whitelist.extend(score_detail_cols)

                columns_to_keep = [col for col in essential_columns_whitelist if col in final_report_df.columns]
                final_report_df = final_report_df[columns_to_keep]
                
                report_dir = os.path.join(output_dir_base, stock_code)
                try:
                    os.makedirs(report_dir, exist_ok=True)
                except OSError as e:
                    print(f"    -> [错误] 创建报告目录 '{report_dir}' 失败: {e}")
                    continue

                filename = f"{start_date.strftime('%Y-%m-%d')}.json"
                filepath = os.path.join(report_dir, filename)
                
                json_data = sanitize_for_json(final_report_df.to_dict(orient='index'))
                try:
                    with open(filepath, 'w', encoding='utf-8') as f:
                        json.dump(json_data, f, ensure_ascii=False, indent=4)
                    print(f"    -> [报告生成] 已将包含推演数据的快照 ({len(final_report_df)}行) 保存至: {filepath}")

                    interpretation_dict = await self.interpret_snapshot(filepath, params)
                    
                    base_name, _ = os.path.splitext(filename)
                    interp_filename = f"{base_name}_interpret_snapshot.json"
                    interp_filepath = os.path.join(report_dir, interp_filename)
                    with open(interp_filepath, 'w', encoding='utf-8') as f:
                        json.dump(interpretation_dict, f, ensure_ascii=False, indent=4)
                    print(f"    -> [智能解读] 已将推演诊断报告保存至: {interp_filepath}")

                except Exception as e:
                    print(f"    -> [错误] 保存或解读JSON情报档案失败: {e}")

        print("\n" + "="*30 + " [阿尔法猎手 V118.20 总结] " + "="*30)
        print(f"    - 总计分析波段数: {len(golden_waves)}")
        print(f"    - 成功捕获: {success_count} 个 | 错失良机: {missed_count} 个")
        if len(golden_waves) > 0:
            capture_rate = (success_count / len(golden_waves)) * 100
            print(f"    - 黄金波段捕获率: {capture_rate:.2f}%")
        print("="*74)

    async def interpret_snapshot(self, filepath: str, params: dict) -> dict:
        """
        【V118.18 战役推演官】
        - 核心升级 1: 修正原子状态识别逻辑，更精确。
        - 核心升级 2: 增加“战后推演”模块，分析后续5天走势。
        - 核心升级 3: 给出最终裁决：是“确认错失良机”还是“确认规避陷阱”。
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if not data:
                return {"error": "解读失败：JSON文件为空。", "source_file": filepath}

            df = pd.DataFrame.from_dict(data, orient='index')
            df.index = pd.to_datetime(df.index)
            
            # 1. 分离战前与战后数据
            df_lookback = df[df['analysis_period'] == 'lookback']
            df_look_forward = df[df['analysis_period'] == 'look_forward']
            
            if df_lookback.empty:
                return {"error": "解读失败：报告中无lookback数据。", "source_file": filepath}

            # 2. 对起涨点当天进行分析
            last_day = df_lookback.iloc[-1]
            last_date_str = last_day.name.strftime('%Y-%m-%d')

            # --- 上下文分析 ---
            ema21 = last_day.get('EMA_21_D', 0)
            ema55 = last_day.get('EMA_55_D', 0)
            close = last_day.get('close_D', 0)
            trend_context = "震荡或整理区"
            if close > ema21 > ema55: trend_context = "强势区 (收盘价 > EMA21 > EMA55)"
            elif close < ema21 < ema55: trend_context = "弱势区 (收盘价 < EMA21 < EMA55)"
            
            macd_h = last_day.get('MACDh_13_34_8_D', 0)
            momentum_context = "动能为负 (MACD绿柱)" if macd_h <= 0 else "动能为正 (MACD红柱)"

            # --- 状态诊断 (精确版) ---
            # ▼▼▼【核心修正 V118.18】精确识别原子状态 ▼▼▼
            active_states = [
                col for col, val in last_day.items() 
                if isinstance(col, str) and col.isupper() and '_D' not in col and 'score' not in col
                and isinstance(val, (int, float)) and val > 0
            ]
            # ▲▲▲【核心修正 V118.18】▲▲▲

            # --- 触发事件分析 ---
            entry_score = last_day.get('entry_score', 0)
            risk_score = last_day.get('risk_score', 0)
            scoring_params = self._get_params_block(params, 'scoring_params', {})
            min_entry_score = self._get_param_value(scoring_params.get('min_entry_score'), 0.7)

            # --- 战后推演分析 (Post-Event Analysis) ---
            post_event_analysis = {}
            if not df_look_forward.empty:
                forward_days = len(df_look_forward)
                # 后续N日内的最高价
                peak_high_forward = df_look_forward['high_D'].max()
                # 后续N日最终收盘价
                final_close_forward = df_look_forward['close_D'].iloc[-1]
                
                # 计算涨幅
                peak_gain_pct = ((peak_high_forward / close) - 1) * 100
                final_gain_pct = ((final_close_forward / close) - 1) * 100
                
                # 最终裁决
                verdict = "确认规避陷阱"
                if peak_gain_pct > 8.0: # 如果后续5日内最大涨幅超过8%，可认为是错失良机
                    verdict = "确认错失良机"
                
                post_event_analysis = {
                    "verdict": verdict,
                    "forward_days_analyzed": forward_days,
                    "peak_gain_in_period_pct": round(peak_gain_pct, 2),
                    "final_gain_at_period_end_pct": round(final_gain_pct, 2),
                    "summary": f"在后续{forward_days}天内, 股价最大涨幅达到{peak_gain_pct:.2f}%, 期末收盘涨幅为{final_gain_pct:.2f}%。"
                }
            else:
                post_event_analysis = {"verdict": "数据不足，无法推演", "summary": "报告中无后续数据。"}

            # --- 合成最终报告字典 ---
            interpretation_dict = {
                "interpretation_details": {
                    "report_date": last_date_str,
                    "source_snapshot": os.path.basename(filepath),
                    "status": "Missed Opportunity Analysis"
                },
                "pre_event_analysis": {
                    "context_summary": f"趋势: {trend_context} | 动能: {momentum_context}",
                    "active_atomic_states": active_states if active_states else "无",
                    "trigger_failure_reason": {
                        "conclusion": "未能达到最低入场阈值",
                        "final_entry_score": round(entry_score, 4),
                        "risk_score": round(risk_score, 4),
                        "required_score": min_entry_score
                    }
                },
                "post_event_analysis": post_event_analysis
            }
            return interpretation_dict

        except FileNotFoundError:
            return {"error": "解读失败：找不到文件。", "source_file": filepath}
        except Exception as e:
            return {"error": f"解读时发生未知错误: {e}", "source_file": filepath}












