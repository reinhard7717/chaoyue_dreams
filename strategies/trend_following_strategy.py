# 文件: strategies/trend_following_strategy.py
# 版本: V21.0 - 适配新架构版
import logging
from decimal import Decimal
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
    趋势跟踪策略 (V21.0 - 适配新架构版)
    - 核心修改: 移除 IndicatorService 和 asyncio 依赖，变为纯粹的同步计算类。
    - 职责定位: 接收一个包含所有时间框架数据的字典，应用复杂的战术剧本和计分系统，输出最终决策。
    """
    def _get_param_value(self, param: Any, default: Any = None) -> Any:
        """
        【V40.3 新增】健壮的参数值解析器。
        - 核心功能: 智能处理两种配置格式：
          1. 简单值: "param": 10
          2. 字典格式: "param": {"value": 10, "说明": "..."}
        - 返回值: 始终返回参数的实际值。
        """
        if isinstance(param, dict) and 'value' in param:
            return param['value']
        if param is not None:
            return param
        return default

    def __init__(self, config: dict):
        """
        【V21.1 构造函数优化版】
        - 移除 self.indicator_service 的实例化，因为本类不应负责数据获取。
        - 移除已过时的 tactical_configs 和 tactical_params 逻辑。
        - 使本类成为一个更纯粹的计算单元。
        """
        # 保存传入的完整战术配置文件
        self.daily_params = config

        # 初始化K线形态识别器
        kline_params = self._get_params_block(self.daily_params, 'kline_pattern_params')
        self.pattern_recognizer = KlinePatternRecognizer(params=kline_params)

        # 初始化用于存储中间信号和分数的字典
        self.signals = {}
        self.scores = {}
        self._last_score_details_df = None # 用于存储详细的得分表，以便后续生成报告

        # 初始化调试参数
        self.debug_params = self._get_params_block(self.daily_params, 'debug_params')
        self.verbose_logging = self.debug_params.get('enabled', False) and self.debug_params.get('verbose_logging', False)
        print("--- [战术策略 TrendFollowStrategy (V21.1 优化版)] 初始化完成。---")

    # 参数解析辅助函数
    def _get_periods_for_timeframe(self, indicator_params: dict, timeframe: str) -> Optional[list]:
        """
        【V5.0 】根据时间周期从指标配置中智能获取正确的'periods'。
        能够处理新旧两种配置格式。

        Args:
            indicator_params (dict): 单个指标的完整配置字典 (例如，params['macd'])。
            timeframe (str): 需要获取参数的时间周期，如 'D', '60', '15'。

        Returns:
            Optional[list]: 找到的周期参数列表，或None。
        """
        if not indicator_params:
            return None

        # 优先处理新的 'configs' 列表格式
        if 'configs' in indicator_params and isinstance(indicator_params['configs'], list):
            for config_item in indicator_params['configs']:
                if timeframe in config_item.get('apply_on', []):
                    return config_item.get('periods')
            # 如果在configs列表中循环后没找到，返回None
            return None
        
        # 向后兼容旧的单一格式
        elif 'periods' in indicator_params:
            # 检查旧格式的apply_on，如果存在且不匹配则返回None
            apply_on = indicator_params.get('apply_on', [])
            if not apply_on or timeframe in apply_on:
                return indicator_params.get('periods')
            else:
                return None
        
        return None

    def _ensure_numeric_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        【V40.6 智能净化版】数据类型标准化引擎
        - 核心升级: 不再依赖硬编码的列名，而是通过检查列中第一个非空元素是否为Decimal类型来智能识别需要转换的列。
        - 解决问题: 彻底根除因任何列（包括动态生成的衍生列）包含Decimal类型而导致的TypeError。
        """
        # print("    - [类型标准化引擎 V40.6 智能版] 启动，检查并转换数据类型...")
        converted_cols = []
        for col in df.columns:
            # 只检查 object 类型的列，因为 Decimal 列会被 Pandas 识别为 object
            if df[col].dtype == 'object':
                # 尝试获取第一个非空值进行类型嗅探，以提高效率
                first_valid_item = df[col].dropna().iloc[0] if not df[col].dropna().empty else None
                
                # 如果第一个有效项是Decimal，则转换整列
                if isinstance(first_valid_item, Decimal):
                    # print(f"      -> 发现列 '{col}' 包含Decimal对象，执行转换...")
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    converted_cols.append(col)

        if converted_cols:
            # print(f"      -> 已将以下 'object' 类型列智能转换为数值类型: {converted_cols}")
            pass
        else:
            print("      -> 所有数值列类型正常，无需转换。")
        
        # print("    - [类型标准化引擎 V40.6] 类型检查完成。")
        return df

    def apply_strategy(self, df: pd.DataFrame, params: dict) -> Tuple[pd.DataFrame, Dict[str, pd.Series]]:
        """
        【V45.8 终极架构版】
        - 核心架构修复: 函数签名从接收 df_dict 改为直接接收一个完全预处理好的日线DataFrame (df)。
        本模块不再关心任何其他时间周期的原始数据，职责更加纯粹。
        """
        print("\n" + "="*60)
        print(f"====== 日期: {df.index[-1].date()} | 开始执行【战术引擎 V45.8】 ======")
        
        # --- 步骤 0: 输入验证 ---
        if df is None or df.empty:
            print("    - [错误] 传入的DataFrame为空，战术引擎终止。")
            return pd.DataFrame(), {}
        
        df = self._ensure_numeric_types(df)
        self._chip_atomic_signals_cache_by_tf = {}

        timeframe_suffixes = ['_D', '_W', '_M', '_5', '_15', '_30', '_60']

        # ▼▼▼ 精准的列名重命名逻辑 (保持不变) ▼▼▼
        rename_map = {
            col: f"{col}_D" for col in df.columns 
            if not any(col.endswith(suffix) for suffix in timeframe_suffixes) 
            and not col.startswith(('VWAP_', 'BASE_', 'playbook_', 'signal_', 'kline_', 'context_', 'cond_')) # 增加对 cond_ 前缀的排除
        }
        if 'close_D' in df.columns:
            df['pct_change_D'] = df['close_D'].pct_change()
            print("    - [核心修正 V45.52] 已强制重新计算 'pct_change_D' 列。")

        if rename_map:
            df = df.rename(columns=rename_map)

        # 步骤 1: 准备衍生特征 (保持不变)
        df = self._prepare_derived_features(df)

        # 步骤 2: 计算斜率 (保持不变)
        df = self._calculate_trend_slopes(df, params)

        self.signals, self.scores = {}, {}
        df = self.pattern_recognizer.identify_all(df)

        self._analyze_dynamic_box_and_ma_trend(df, params)

        print("    - [信息] 核心计分流程开始...")
        df.loc[:, 'entry_score'], atomic_signals, score_details_df, setup_conditions = self._calculate_entry_score(df, params)
        self._last_score_details_df = score_details_df

        print("    - [信息] 正在将战术剧本触发详情合并到最终结果中...")
        if score_details_df is not None and not score_details_df.empty:
            excluded_prefixes = ('BASE_', 'BONUS_', 'PENALTY_', 'INDUSTRY_', 'DYNAMICS_SCORE', 'WATCHING_SETUP')
            for playbook_name in score_details_df.columns:
                if not playbook_name.startswith(excluded_prefixes):
                    df[f'playbook_{playbook_name}'] = score_details_df[playbook_name] > 0

        print("    - [信息] 智能出场逻辑判断开始...")
        risk_states = self._calculate_risk_states(df, params)
        df.loc[:, 'exit_signal_code'] = self._calculate_exit_signals(df, risk_states, params)

        entry_scoring_params = self._get_params_block(params, 'entry_scoring_params')
        score_threshold = self._get_param_value(entry_scoring_params.get('score_threshold'), 100)
        df.loc[:, 'signal_entry'] = df['entry_score'] >= score_threshold

        playbook_definitions_for_log = self._get_playbook_definitions(df, trigger_events={}, setup_conditions={})
        playbook_cn_name_map = {p['name']: p.get('cn_name', p['name']) for p in playbook_definitions_for_log}

        entry_signals = df[df['signal_entry']]
        print("\n---【多时间框架协同策略(V45.8 终极架构版) 逻辑链调试】---")
        
        log_start_date = pd.to_datetime('2024-07-01').date()
        filtered_entry_signals = entry_signals[entry_signals.index.date >= log_start_date]
        
        print(f"【最终买入】(得分>{score_threshold})信号总数: {len(entry_signals)} | 24年7月后信号数: {len(filtered_entry_signals)}")
        
        if not filtered_entry_signals.empty:
            for entry_date, row in filtered_entry_signals.iterrows():
                entry_score = row['entry_score']
                pct_change_val = row.get('pct_change_D', 0) * 100
                print(f"\n====== 日期: {entry_date.date()} | 收盘: {row.get('close_D', 'N/A'):.2f} | 涨跌: {pct_change_val:.2f}% ======")
                print(f"  - 核心前提 (右侧趋势): {row.get('robust_right_side_precondition', '未知')}")
                
                if self._last_score_details_df is not None and entry_date in self._last_score_details_df.index:
                    score_breakdown = self._last_score_details_df.loc[entry_date].dropna()
                    active_playbooks = [
                        k for k, v in score_breakdown.items() 
                        if v > 0 and not k.startswith(('BASE_', 'BONUS_', 'PENALTY_', 'INDUSTRY_', 'DYNAMICS_SCORE', 'WATCHING_SETUP'))
                    ]
                else:
                    active_playbooks = []

                active_setups = [
                    s.replace('SETUP_', '') for s in setup_conditions.keys()
                    if s in setup_conditions and isinstance(setup_conditions[s], pd.Series) and not setup_conditions[s].empty and entry_date in setup_conditions[s].index and setup_conditions[s].at[entry_date]
                ]
                setup_cn_name_map = {
                    'PROLONGED_COMPRESSION': '长期蓄势', 'CAPITAL_FLOW_DIVERGENCE': '资金暗流', 'CAPITAL_DIVERGENCE': '猛兽苏醒',
                    'ENERGY_COMPRESSION': '能量压缩', 'HEALTHY_PULLBACK': '健康回踩', 'DUCK_NECK_FORMING': '老鸭颈',
                    'CHIP_ACCUMULATION': '筹码吸筹', 'WASHOUT_DAY': '巨阴洗盘',
                    'SHOCK_BOTTOM': '休克谷底', 'DOJI_PAUSE': '十字星暂停',
                    'RELATIVE_STRENGTH': '相对强势', 'FORTRESS_SIEGE': '堡垒围攻',
                    'BBAND_SQUEEZE': '布林压缩', 'WINNER_RATE_WASHED_OUT': '投降坑',
                    'BIAS_EXTREME_OVERSOLD': 'BIAS超卖', 'GAP_SUPPORT': '缺口支撑',
                    'MOMENTUM_DIVERGENCE': '动能拐点',
                }
                active_setups_cn = [setup_cn_name_map.get(s, s) for s in active_setups]

                print(f"  【✔ 买入信号触发】")
                print(f"    - 总分: {entry_score:.2f} (阈值: {score_threshold})")
                print(f"    - 激活的日线剧本: {[playbook_cn_name_map.get(p, p) for p in active_playbooks]}")
                print(f"    - 成立的准备状态: {active_setups_cn if active_setups_cn else '无'}")

        print(f"====== 【战术引擎 V45.8】执行完毕 ======")
        print("="*60 + "\n")

        return df, atomic_signals

    def prepare_db_records(self, stock_code: str, result_df: pd.DataFrame, atomic_signals: Dict[str, pd.Series], params: dict, result_timeframe: str = 'D') -> List[Dict[str, Any]]:
        """
        【V42.0 修复版】
        - 核心修改: 修复了涨跌幅数据提取错误的问题，并增加了对准备状态的记录。
        """
        df_with_signals = result_df[
            (result_df['signal_entry'] == True) | (result_df.get('exit_signal_code', 0) > 0)
        ].copy()
        if df_with_signals.empty:
            return []
        
        records = []
        
        strategy_info_block = self._get_params_block(params, 'strategy_info', {})
        name_param = strategy_info_block.get('name')
        strategy_name = self._get_param_value(name_param, 'unknown_strategy')
        timeframe = result_timeframe

        # ▼▼▼【代码修改 V42.0】: 获取剧本中文名映射 ▼▼▼
        # 为了获取中文名，我们需要再次调用 _get_playbook_definitions
        # 由于不依赖于实际数据，传入空的 trigger 和 setup 即可
        playbook_definitions = self._get_playbook_definitions(result_df, {}, {})
        playbook_cn_name_map = {p['name']: p.get('cn_name', p['name']) for p in playbook_definitions}
        # ▲▲▲【代码修改 V42.0】▲▲▲

        for timestamp, row in df_with_signals.iterrows():
            triggered_playbooks_list = []
            triggered_playbooks_cn_list = [] # 新增：用于存储中文名
            
            if self._last_score_details_df is not None and timestamp in self._last_score_details_df.index:
                playbooks_with_scores = self._last_score_details_df.loc[timestamp]
                active_items = playbooks_with_scores[playbooks_with_scores > 0].index
                
                excluded_prefixes = ('BASE_', 'BONUS_', 'PENALTY_', 'INDUSTRY_', 'WATCHING_SETUP') # 移除 DYNAMICS_SCORE
                triggered_playbooks_list = [ item for item in active_items if not item.startswith(excluded_prefixes) ]
                # 使用映射表获取中文名
                triggered_playbooks_cn_list = [ playbook_cn_name_map.get(item, item) for item in triggered_playbooks_list ]

            # 查找当天成立的准备状态
            active_setups = []
            setup_cols = [col for col in row.index if col.startswith('SETUP_') and row[col] is True]
            active_setups = [s.replace('SETUP_', '') for s in setup_cols]

            context_dict = {k: v for k, v in row.items() if pd.notna(v)}
            sanitized_context = sanitize_for_json(context_dict)
            
            # ▼▼▼【代码修改 V42.0】: 修复涨跌幅提取逻辑 ▼▼▼
            pct_change = row.get('pct_change_D', row.get('pct_change', 0.0))
            # ▲▲▲【代码修改 V42.0】▲▲▲

            record = {
                "stock_code": stock_code,
                "trade_time": sanitize_for_json(timestamp),
                "timeframe": timeframe,
                "strategy_name": strategy_name,
                "close_price": sanitize_for_json(row.get('close_D')),
                "pct_change": sanitize_for_json(pct_change), # 新增：记录正确的涨跌幅
                "entry_score": sanitize_for_json(row.get('entry_score', 0.0)),
                "entry_signal": sanitize_for_json(row.get('signal_entry', False)),
                "exit_signal_code": sanitize_for_json(row.get('exit_signal_code', 0)),
                "is_right_side_trend": sanitize_for_json(row.get('robust_right_side_precondition', False)), # 新增：记录右侧前提
                "triggered_playbooks": triggered_playbooks_list,
                "triggered_playbooks_cn": triggered_playbooks_cn_list, # 新增：记录中文剧本
                "active_setups": active_setups, # 新增：记录准备状态
                "context_snapshot": sanitized_context,
            }
            # 移除旧的、不准确的字段
            record.pop("is_long_term_bullish", None)
            record.pop("is_mid_term_bullish", None)
            record.pop("is_pullback_setup", None)
            record.pop("pullback_target_price", None)

            records.append(record)
        return records

    # 趋势斜率计算中心
    def _calculate_trend_slopes(self, df: pd.DataFrame, params: dict) -> pd.DataFrame:
        """
        【V45.47 二阶斜率版】趋势斜率计算中心
        - 核心升级: 在计算完一阶斜率(SLOPE)和二阶导数(ACCEL)后，继续计算二阶导数的斜率(SLOPE_of_ACCEL)。
        """
        print("    - [斜率中心 V45.47] 开始计算关键指标的趋势斜率 (含二阶斜率)...")
        slope_params = self._get_params_block(params, 'slope_params', {})
        if not self._get_param_value(slope_params.get('enabled'), False):
            print("    - [斜率中心] 斜率计算被禁用。")
            return df

        def get_slope(y):
            y_clean = y.dropna()
            if y_clean.shape[0] < 2:
                return np.nan
            x = np.arange(len(y_clean))
            slope, _ = np.polyfit(x, y_clean.values, 1)
            return slope

        series_to_slope = self._get_param_value(slope_params.get('series_to_slope'), {})
        
        capital_lookback = 20 # 保持与特征工程中的周期一致
        capital_accum_col = f'mf_accumulation_{capital_lookback}_D'
        if capital_accum_col not in series_to_slope:
            series_to_slope[capital_accum_col] = [capital_lookback]
            print(f"      -> 动态为 '{capital_accum_col}' 添加斜率计算任务。")

        for col_name, lookbacks in series_to_slope.items():
            if col_name not in df.columns:
                print(f"    - [斜率中心] 警告: 列 '{col_name}' 不存在，跳过斜率计算。")
                continue
            
            for lookback in lookbacks:
                slope_col_name = f'SLOPE_{col_name}_{lookback}'
                df[slope_col_name] = df[col_name].rolling(window=lookback, min_periods=max(2, lookback // 2)).apply(get_slope, raw=False)
                
                accel_col_name = f'ACCEL_{col_name}_{lookback}'
                df[accel_col_name] = df[slope_col_name].rolling(window=lookback, min_periods=max(2, lookback // 2)).apply(get_slope, raw=False)
                
                jerk_col_name = f'SLOPE_{accel_col_name}_{lookback}'
                df[jerk_col_name] = df[accel_col_name].rolling(window=lookback, min_periods=max(2, lookback // 2)).apply(get_slope, raw=False)
                
        print("    - [斜率中心 V45.53] 所有斜率计算完成。")
        return df

    # ▼▼▼ 将剧本定义抽取为独立函数 ▼▼▼
    def _get_playbook_definitions(self, df: pd.DataFrame, trigger_events: Dict[str, pd.Series], setup_conditions: Dict[str, pd.Series]) -> List[Dict]:
        """
        【V42.0 新增】剧本定义中心
        - 核心功能: 集中定义所有交易剧本，并为其添加中文名，便于日志输出和调试。
        - 返回: 一个包含所有剧本定义的列表。
        """
        default_series = pd.Series(False, index=df.index)
        robust_right_side_precondition = df.get('robust_right_side_precondition', pd.Series(True, index=df.index))

        playbook_definitions = [
            # =================================================================================
            # === S级 (Tier S) 剧本: "完美风暴" (分数 > 300) ===
            # =================================================================================
            {
                'name': 'EARTH_HEAVEN_BOARD', 'cn_name': '地天板',
                'setup': True,
                'trigger': trigger_events.get('TRIGGER_EARTH_HEAVEN_BOARD', default_series),
                'score': 380, 'precondition': True,
                'comment': '地天板，市场情绪从极度恐慌到极度贪婪的日内反转，最强的V反信号。'
            },
            {
                'name': 'PERFECT_STORM', 'cn_name': '潜龙出海',
                'setup': setup_conditions.get('SETUP_PROLONGED_COMPRESSION', default_series),
                'trigger': trigger_events.get('CHIP_INSTITUTIONAL_BREAKOUT', default_series),
                'score': 350, 'precondition': True,
                'comment': '长期蓄势 + 主力点火，确定性最高的趋势启动模式。'
            },
            {
                'name': 'AWAKENED_BEAST', 'cn_name': '猛兽苏醒',
                # 准备条件: 检查“资本背离”的持久化状态是否激活
                'setup': df.get('CONTEXT_CAPITAL_DIVERGENCE_ACTIVE', default_series),
                # 触发条件: “动能背离”的核心准备状态成立
                'trigger': setup_conditions.get('SETUP_MOMENTUM_DIVERGENCE', default_series),
                'score': 330, 
                'precondition': True, # 这是一个左侧组合信号，放宽右侧前提
                'comment': '【V46.0 状态持久化版】捕捉“资本背离”状态开启后，由“动能背离”事件点燃的终极反转信号。'
            },
            {
                'name': 'INTERNAL_SET_CONTEXT_CAPITAL_DIVERGENCE', 'cn_name': '内部-设置资本背离状态',
                'setup': True, # 无需前置setup
                # 触发条件: “资本背离”准备状态成立的当天
                'trigger': setup_conditions.get('SETUP_CAPITAL_DIVERGENCE', default_series),
                'score': 0, # 不产生任何分数，仅用于触发上下文
                'precondition': True,
                'comment': '内部辅助剧本，当资本背离条件满足时，开启一个持续数天的上下文状态，用于被其他剧本捕获。'
            },
            {
                'name': 'WASH_AND_RISE', 'cn_name': '洗盘拉升',
                'setup': setup_conditions.get('SETUP_PULLBACK_WITH_MF_INFLOW', default_series),
                'trigger': trigger_events.get('STRONG_POSITIVE_CANDLE', default_series),
                'score': 310, 'precondition': robust_right_side_precondition,
                'comment': '最经典的“假跌真吸”模式，回踩的质量极高。'
            },
            # =================================================================================
            # === A+级 (Tier A+) 剧本: "高置信度" (分数 250-299) ===
            # =================================================================================
            {
                'name': 'MOMENTUM_INFLECTION_POINT', 'cn_name': '动能拐点',
                'setup': setup_conditions.get('SETUP_MOMENTUM_DIVERGENCE', default_series),
                'trigger': trigger_events.get('TRIGGER_REVERSAL_CONFIRMATION_CANDLE', default_series),
                'score': 295, 'precondition': True, # 前提放宽，因为这是左侧信号
                'comment': '捕捉下跌动能的“加加速度”达到峰值后的第一个确认阳线，是最高精度的左侧反转信号。'
            },
            {
                'name': 'GAP_SUPPORT_CONFIRMED', 'cn_name': '缺口支撑',
                'setup': setup_conditions.get('SETUP_GAP_SUPPORT', default_series),
                'trigger': trigger_events.get('STRONG_POSITIVE_CANDLE', default_series),
                'score': 290, 'precondition': robust_right_side_precondition,
                'comment': '强势上涨留下的缺口在回调中未被回补，并出现阳线确认，是趋势极强的表现。'
            },
            {
                'name': 'FIBONACCI_PULLBACK', 'cn_name': '黄金回踩',
                'setup': True,
                'trigger': trigger_events.get('TRIGGER_FIBONACCI_REBOUND', default_series),
                'score': 285, 'precondition': robust_right_side_precondition,
                'comment': '在通过多重验证的有效驱动浪后，于关键斐波那契位获得主力资金支撑的反弹。'
            },
            {
                'name': 'BREAKOUT_RETEST_GO', 'cn_name': '突破回踩',
                'setup': setup_conditions.get('SETUP_PULLBACK_POST_BREAKOUT', default_series),
                'trigger': trigger_events.get('STRONG_POSITIVE_CANDLE', default_series),
                'score': 280, 'precondition': robust_right_side_precondition,
                'comment': '最可靠的趋势延续形态之一：突破-回踩-再出发。'
            },
            {
                'name': 'REVERSAL_FIRST_PULLBACK', 'cn_name': '反转首踩',
                'setup': setup_conditions.get('SETUP_PULLBACK_POST_REVERSAL', default_series),
                'trigger': trigger_events.get('STRONG_POSITIVE_CANDLE', default_series),
                'score': 270, 'precondition': robust_right_side_precondition,
                'comment': '抓住新趋势的第一个上车点，通常有较好的盈亏比。'
            },
            {
                'name': 'OLD_DUCK_HEAD_TAKEOFF', 'cn_name': '老鸭回头',
                'setup': setup_conditions.get('SETUP_DUCK_NECK_FORMING', default_series),
                'trigger': trigger_events.get('MA_RECLAIM', default_series),
                'score': 260, 'precondition': robust_right_side_precondition,
                'comment': '经典的均线理论形态，从整理到再次发力的转折点。'
            },
            # =================================================================================
            # === A级 (Tier A) 剧本: "标准可靠" (分数 200-249) ===
            # =================================================================================
            {
                'name': 'N_SHAPE_RELAY', 'cn_name': 'N字接力',
                'setup': True,
                'trigger': trigger_events.get('TRIGGER_N_SHAPE_RELAY', default_series),
                'score': 245, 'precondition': robust_right_side_precondition,
                'comment': '经典的“N字板”接力形态，缩量回调后的再次放量突破，是极强的趋势延续信号。'
            },
            {
                'name': 'MA_ACCELERATION', 'cn_name': '均线加速',
                'setup': True,
                'trigger': trigger_events.get('TRIGGER_MA_ACCELERATION', default_series),
                'score': 235, 'precondition': robust_right_side_precondition,
                'comment': '均线、成交量、资金、趋势强度四维共振的“趋势引爆点”，动能强劲。'
            },
            {
                'name': 'ENERGY_RELEASE', 'cn_name': '能量释放',
                'setup': setup_conditions.get('SETUP_ENERGY_COMPRESSION', default_series),
                'trigger': trigger_events.get('TRIGGER_ENERGY_RELEASE', default_series),
                'score': 230, 'precondition': robust_right_side_precondition,
                'comment': '通用性强的“盘久必涨”模式。'
            },
            {
                'name': 'PULLBACK_REBOUND_CONFIRMED', 'cn_name': '回踩确认',
                'setup': setup_conditions.get('SETUP_HEALTHY_PULLBACK', default_series),
                'trigger': trigger_events.get('TRIGGER_PULLBACK_REBOUND', default_series),
                'score': 220, 'precondition': robust_right_side_precondition,
                'comment': '最基础、最常见的趋势跟踪入场点。'
            },
            {
                'name': 'V_REVERSAL_ENTRY', 'cn_name': 'V型反转',
                'setup': setup_conditions.get('SETUP_SHOCK_BOTTOM', default_series),
                'trigger': trigger_events.get('STRONG_POSITIVE_CANDLE', default_series),
                'score': 210, 'precondition': True,
                'comment': '高风险高收益的左侧交易模式，捕捉情绪拐点。'
            },
            # =================================================================================
            # === B级 (Tier B) 剧本: "机会主义" (分数 < 200) ===
            # =================================================================================
            {
                'name': 'WINNER_RATE_REVERSAL', 'cn_name': '投降坑反转',
                'setup': setup_conditions.get('SETUP_WINNER_RATE_WASHED_OUT', default_series),
                'trigger': trigger_events.get('TRIGGER_WINNER_RATE_REVERSAL', default_series),
                'score': 195, 'precondition': True,
                'comment': '“投降坑”反转，在获利盘几乎被完全洗净后出现的企稳阳线，博弈市场绝望后的情绪拐点。'
            },
            {
                'name': 'WASHOUT_REVERSAL', 'cn_name': '洗盘反包',
                'setup': setup_conditions.get('SETUP_WASHOUT_DAY', default_series),
                'trigger': trigger_events.get('TRIGGER_WASHOUT_REVERSAL', default_series),
                'score': 190, 'precondition': True,
                'comment': '博弈主力洗盘后的快速拉升。'
            },
            {
                'name': 'DOJI_CONTINUATION', 'cn_name': '十字星接力',
                'setup': setup_conditions.get('SETUP_DOJI_PAUSE', default_series),
                'trigger': trigger_events.get('TRIGGER_DOJI_BREAKOUT', default_series),
                'score': 188, 'precondition': robust_right_side_precondition,
                'comment': '上涨中继形态，十字星分歧后转为一致，是趋势延续的信号。'
            },
            {
                'name': 'MOMENTUM_BREAKOUT', 'cn_name': '动量突破',
                'setup': True,
                'trigger': trigger_events.get('TRIGGER_MOMENTUM_BREAKOUT', default_series),
                'score': 185, 'precondition': robust_right_side_precondition,
                'comment': '经典动量突破（首板）模式，捕捉趋势启动的爆发力。'
            },
            {
                'name': 'RELATIVE_STRENGTH_LEADER', 'cn_name': '相对强势',
                'setup': setup_conditions.get('SETUP_RELATIVE_STRENGTH', default_series),
                'trigger': trigger_events.get('STRONG_POSITIVE_CANDLE', default_series),
                'score': 180, 'precondition': robust_right_side_precondition,
                'comment': '捕捉市场中最强的品种，通常是下一波行情的领涨股。'
            },
            {
                'name': 'BBAND_SQUEEZE_BREAKOUT', 'cn_name': '布林压缩突破',
                'setup': setup_conditions.get('SETUP_BBAND_SQUEEZE', default_series),
                'trigger': trigger_events.get('TRIGGER_BBAND_BREAKOUT', default_series),
                'score': 175, 'precondition': robust_right_side_precondition,
                'comment': '布林带收口后的突破，经典的“盘久必涨”形态。'
            },
            {
                'name': 'FORTRESS_DEFENDED', 'cn_name': '堡垒防守',
                'setup': setup_conditions.get('SETUP_FORTRESS_SIEGE', default_series),
                'trigger': trigger_events.get('TRIGGER_FORTRESS_DEFENSE', default_series),
                'score': 170, 'precondition': robust_right_side_precondition & df.get('temp_is_fortress_valid', False),
                'comment': '基于结构支撑的防守反击模式。'
            },
            {
                'name': 'MACD_LOW_CROSS_REVERSAL', 'cn_name': 'MACD低位金叉',
                'setup': True,
                'trigger': trigger_events.get('TRIGGER_MACD_LOW_CROSS', default_series),
                'score': 165, 'precondition': True,
                'comment': 'MACD在低位形成金叉，潜在的底部反转信号。'
            },
            {
                'name': 'BIAS_REVERSAL', 'cn_name': 'BIAS超跌反弹',
                'setup': setup_conditions.get('SETUP_BIAS_EXTREME_OVERSOLD', default_series),
                'trigger': trigger_events.get('TRIGGER_BIAS_REBOUND', default_series),
                'score': 160, 'precondition': True,
                'comment': '经典的乖离率（BIAS）指标超跌反弹，捕捉技术性修复行情。'
            },
        ]
        return playbook_definitions

    def _calculate_entry_score(self, df: pd.DataFrame, params: dict) -> Tuple[pd.Series, Dict[str, pd.Series], pd.DataFrame, Dict[str, pd.Series]]:
        """
        【V45.8 终极架构版】
        - 核心重构: 彻底移除 df_dict 参数，函数现在只依赖一个被完全预处理过的日线DataFrame。
        - 依赖解耦: 所有对多时间框架数据的依赖（周线信号、分钟线VWAP）均由上游的总指挥模块处理完毕，
                    本函数只负责在日线层面进行计分，职责单一且清晰。
        """
        print("    [计分V45.8] 启动终极架构计分引擎...")
        atomic_signals = {}
        score_details_df = pd.DataFrame(index=df.index)
        scoring_params = self._get_params_block(params, 'entry_scoring_params')
        points = scoring_params.get('points', {})
        
        # --- 步骤1: 计算战略背景基础分 (逻辑不变) ---
        # ... (此部分代码保持不变) ...
        print("    [计分V45.8] 步骤1: 计算周线战略背景基础分...")
        king_signal_col = 'BASE_SIGNAL_BREAKOUT_TRIGGER'
        king_score = self._get_param_value(points.get('BREAKOUT_TRIGGER_SCORE'), 150)
        all_base_score_cols = [king_signal_col]
        if king_signal_col in df.columns and df[king_signal_col].any():
            score_details_df.loc[df[king_signal_col], king_signal_col] = king_score
        playbook_scores_map = self._get_params_block(params, 'playbook_scores', default_return={})
        playbook_cols = [col for col in df.columns if col.startswith('playbook_') and col.endswith('_W')]
        for col in playbook_cols:
            playbook_name = col.replace('playbook_', '').replace('_W', '')
            score = playbook_scores_map.get(playbook_name, 0)
            base_col_name = f"BASE_{playbook_name.upper()}"
            all_base_score_cols.append(base_col_name)
            if score > 0 and df[col].any():
                score_details_df.loc[df[col], base_col_name] = score
        strategic_accel_col = 'EVENT_STRATEGIC_ACCELERATING_W'
        if strategic_accel_col in df.columns and df[strategic_accel_col].any():
            accel_score = self._get_param_value(points.get('STRATEGIC_ACCEL_SCORE'), 100)
            score_details_df.loc[df[strategic_accel_col], 'BASE_STRATEGIC_ACCEL'] = accel_score
            all_base_score_cols.append('BASE_STRATEGIC_ACCEL')
        score_details_df.fillna(0, inplace=True)
        if king_signal_col in score_details_df.columns:
            king_signal_mask = (score_details_df[king_signal_col] > 0)
            if king_signal_mask.any():
                other_base_score_cols = [col for col in all_base_score_cols if col != king_signal_col and col in score_details_df.columns]
                score_details_df.loc[king_signal_mask, other_base_score_cols] = 0
        
        # --- 步骤2: 进行核心前提交叉验证 (逻辑不变) ---
        print("    [计分V45.8] 步骤2: 进行核心前提交叉验证...")
        validated_premises = self._validate_core_premises(df, params)

        # --- 步骤3: 调用独立的“准备”和“触发”中心 (逻辑不变) ---
        print("    [计分V45.8] 步骤3: 调用独立的准备状态和触发事件中心...")
        # ---【先】调用“触发事件中心”，获得所有触发器 ---
        print("    [计分V45.27] 步骤3: 【先】调用触发事件中心...")
        trigger_events = self._define_trigger_events(df, params)
        atomic_signals.update(trigger_events)

        # --- 【后】调用“准备状态中心”，并将触发器作为输入传入 ---
        print("    [计分V45.27] 步骤4: 【后】调用准备状态中心 (传入触发器)...")
        setup_conditions = self._calculate_setup_conditions(df, params, trigger_events) # 修改: 传入 trigger_events
        atomic_signals.update(setup_conditions)
        for setup_name, setup_signal in setup_conditions.items():
            df[setup_name] = setup_signal

        # --- 步骤4: 计算趋势动力学附加分 (逻辑不变) ---
        print("    [计分V45.8] 步骤4: 计算趋势动力学附加分...")
        dynamics_score = self._score_trend_dynamics(df, params, validated_premises)

        # --- 步骤5: 构建并评估“剧本矩阵” (逻辑不变) ---
        print("    [计分V45.8] 步骤5: 按优先级评估“剧本矩阵”...")
        playbook_definitions = self._get_playbook_definitions(df, trigger_events, setup_conditions)
        df['base_score'] = 0.0
        has_been_scored = pd.Series(False, index=df.index)
        # ▼▼▼【代码修改 V45.9】: 植入S级剧本专用探针 ▼▼▼
        # 定义探针的起始日期
        probe_start_date = pd.to_datetime('2024-07-01').tz_localize('UTC') if df.index.tz else pd.to_datetime('2024-07-01')
        
        # 筛选出需要探针的剧本
        s_tier_playbooks_to_probe = ['PERFECT_STORM', 'AWAKENED_BEAST']
        
        # 为这些剧本创建一个每日评估的DataFrame
        daily_eval_data = {}
        for playbook in playbook_definitions:
            if playbook['name'] in s_tier_playbooks_to_probe:
                # 准备状态(T-1)
                setup_signal = playbook.get('setup', pd.Series(False, index=df.index))
                if isinstance(setup_signal, bool):
                    setup_signal = pd.Series(setup_signal, index=df.index)
                
                # 触发事件(T)
                trigger_signal = playbook.get('trigger', pd.Series(False, index=df.index))
                
                # 核心前提(T)
                precondition_signal = playbook.get('precondition', pd.Series(False, index=df.index))
                if isinstance(precondition_signal, bool):
                    precondition_signal = pd.Series(precondition_signal, index=df.index)

                # 收集数据
                daily_eval_data[f"{playbook['name']}_Setup(T-1)"] = setup_signal.shift(1).fillna(False)
                daily_eval_data[f"{playbook['name']}_Trigger(T)"] = trigger_signal
                daily_eval_data[f"{playbook['name']}_Precondition(T)"] = precondition_signal

        # 将收集到的数据转换为DataFrame，并筛选目标日期
        if daily_eval_data:
            daily_eval_df = pd.DataFrame(daily_eval_data, index=df.index)
            # 筛选出2024-07-01之后，且至少有一个条件为True的日期，便于观察
            interesting_days = daily_eval_df[
                (daily_eval_df.index >= probe_start_date) & 
                (daily_eval_df.any(axis=1))
            ]
            
            # if not interesting_days.empty:
            #     print("\n" + "="*25 + " [S级剧本-每日评估探针] " + "="*25)
            #     print(f"--- (仅显示从 {probe_start_date.date()} 开始，且至少有一个条件为True的日期) ---")
            #     # 使用 to_string() 保证所有列都能显示
            #     print(interesting_days.to_string())
            #     print("="*75 + "\n")
            # else:
            #     print("\n--- [S级剧本-探针信息] 从 2024-07-01 起，未发现任何S级剧本的准备或触发条件成立。---\n")
        # ▲▲▲【代码修改 V45.9】▲▲▲
        for playbook in playbook_definitions:
            setup = playbook.get('setup', pd.Series(False, index=df.index))
            trigger = playbook.get('trigger', pd.Series(False, index=df.index))
            if isinstance(setup, bool): setup = pd.Series(setup, index=df.index)
            if playbook['name'] in ['V_REVERSAL_ENTRY', 'WASHOUT_REVERSAL', 'MOMENTUM_INFLECTION_POINT', 'AWAKENED_BEAST']:
                condition = setup & trigger
            else:
                condition = setup.shift(1).fillna(False) & trigger
            is_triggered = condition & playbook['precondition'] & ~has_been_scored
            if is_triggered.any():
                score = self._get_param_value(points.get(playbook['name']), playbook['score'])
                df.loc[is_triggered, 'base_score'] = score
                score_details_df.loc[is_triggered, playbook['name']] = score
                has_been_scored.loc[is_triggered] = True
                playbook_cn_name = playbook.get('cn_name', playbook['name'])
                print(f"    - [剧本命中] 命中剧本 '{playbook['name']} ({playbook_cn_name})'，触发 {is_triggered.sum()} 天。")

        # --- 观察分逻辑 (逻辑不变) ---
        watching_score = self._get_param_value(points.get('WATCHING_SCORE'), 50)
        is_in_any_setup = pd.concat([s for k, s in setup_conditions.items() if k.startswith('SETUP_') and isinstance(s, pd.Series) and not s.empty], axis=1).any(axis=1)
        is_watching = is_in_any_setup & ~has_been_scored
        if is_watching.any():
            df.loc[is_watching, 'base_score'] = watching_score
            score_details_df.loc[is_watching, 'WATCHING_SETUP'] = watching_score
            print(f"    - [后续跟踪] 发现 {is_watching.sum()} 天处于'观察准备'状态，赋予观察分。")

        # --- 步骤6: 融合剧本分、动力学分与环境修正项 ---
        print("    [计分V45.8] 步骤6: 融合所有得分项...")
        final_score = df['base_score'].copy()
        has_primary_score = final_score > 0
        if (has_primary_score).any():
            final_score.loc[has_primary_score] += dynamics_score.loc[has_primary_score]
            score_details_df.loc[has_primary_score, 'DYNAMICS_SCORE'] = dynamics_score.loc[has_primary_score]

        # ▼▼▼【代码修改 V45.8】: 直接使用预处理好的VWAP支撑列，不再需要df_dict或任何相关函数 ▼▼▼
        cond_vwap_support = df.get('cond_vwap_support', pd.Series(False, index=df.index))
        # ▲▲▲【代码修改 V45.8】▲▲▲
        
        if (cond_vwap_support & has_primary_score).any():
            bonus = self._get_param_value(points.get('BONUS_VWAP_SUPPORT'), 20)
            final_score.loc[cond_vwap_support & has_primary_score] += bonus
            score_details_df.loc[cond_vwap_support & has_primary_score, 'BONUS_VWAP_SUPPORT'] = bonus

        # --- 后续所有奖励、惩罚、合并、否决等逻辑都保持不变 ---
        # ... (此部分代码保持不变) ...
        high_consensus_bonus = self._get_param_value(points.get('BONUS_HIGH_CONSENSUS'), 30)
        is_high_consensus = (validated_premises.get('trend_health_score', pd.Series(0)) >= 5) & has_primary_score
        if is_high_consensus.any():
            final_score.loc[is_high_consensus] += high_consensus_bonus
            score_details_df.loc[is_high_consensus, 'BONUS_HIGH_CONSENSUS'] = high_consensus_bonus
            print(f"      -> '高共识度(健康分>=5)'奖励触发了 {is_high_consensus.sum()} 天。")
        industry_params = self._get_params_block(params, 'industry_context_params', {})
        if industry_params.get('enabled', False) and 'industry_strength_rank_D' in df.columns:
            rank_series = df['industry_strength_rank_D']
            multiplier = pd.Series(1.0, index=df.index)
            weak_rank_threshold = industry_params.get('weak_rank_threshold', 0.3)
            weak_penalty_multiplier = industry_params.get('weak_industry_penalty_multiplier', 0.7)
            strength_multiplier_factor = industry_params.get('strength_rank_multiplier', 0.5)
            top_tier_rank_threshold = industry_params.get('top_tier_rank_threshold', 0.9)
            top_tier_bonus = industry_params.get('top_tier_bonus', 30)
            is_weak = (rank_series < weak_rank_threshold) & has_primary_score
            multiplier.loc[is_weak] = weak_penalty_multiplier
            is_strong = (rank_series >= weak_rank_threshold) & has_primary_score
            multiplier.loc[is_strong] = 1 + (rank_series * strength_multiplier_factor)
            score_change = final_score * (multiplier - 1)
            final_score += score_change
            score_details_df['INDUSTRY_MULTIPLIER_ADJ'] = score_change.where(score_change != 0)
            is_top_tier = (rank_series >= top_tier_rank_threshold) & has_primary_score
            final_score.loc[is_top_tier] += top_tier_bonus
            score_details_df.loc[is_top_tier, 'INDUSTRY_TOP_TIER_BONUS'] = top_tier_bonus
        print("    [计分V45.8] 步骤7: 合并战略与战术得分...")
        base_score_from_weekly = score_details_df.filter(regex='^BASE_').sum(axis=1)
        final_score += base_score_from_weekly
        print("    [计分V45.8] 步骤X: 更新上下文状态...")
        df = self._update_contextual_states(df, score_details_df, validated_premises, params)
        print("    [计分V45.8] 步骤8: 应用最终风险否决层...")
        cond_trend_exhaustion = setup_conditions.get('RISK_TREND_EXHAUSTION', pd.Series(False, index=df.index))
        if cond_trend_exhaustion.any():
            final_score.loc[cond_trend_exhaustion] = 0
            print(f"    - [风险否决] '趋势衰竭'信号触发，否决了 {cond_trend_exhaustion.sum()} 天的买入信号。")
        cond_heaven_earth_board = self._identify_board_patterns(df, params).get('heaven_earth_board', pd.Series(False, index=df.index))
        if cond_heaven_earth_board.any():
            final_score.loc[cond_heaven_earth_board] = 0
            print(f"    - [风险否决] '天地板'信号触发，否决了 {cond_heaven_earth_board.sum()} 天的买入信号。")
        cond_volume_breakdown = self._find_volume_breakdown_exit(df, params)
        if cond_volume_breakdown.any():
            final_score.loc[cond_volume_breakdown] = 0
            print(f"    - [风险否决] '结构崩溃'信号触发，否决了 {cond_volume_breakdown.sum()} 天的买入信号。")
        kline_strong_bearish = df.get('kline_c_evening_star', pd.Series(False, index=df.index)) | \
                            df.get('kline_c_bearish_engulfing_decent', pd.Series(False, index=df.index)) | \
                            df.get('kline_c_three_black_crows', pd.Series(False, index=df.index)) | \
                            df.get('kline_c_dark_cloud_cover_decent', pd.Series(False, index=df.index))
        if kline_strong_bearish.any():
            final_score.loc[kline_strong_bearish] = 0
            print(f"    - [风险否决] '强看跌K线'信号触发，否决了 {kline_strong_bearish.sum()} 天的买入信号。")

        # 清理临时列
        df.drop(columns=['base_score', 'temp_is_fortress_valid'], inplace=True, errors='ignore')
        print("    [计分V45.8] 计分流程结束。")
        
        return final_score.round(0), atomic_signals, score_details_df.fillna(0), setup_conditions

    def _calculate_exit_signals(self, df: pd.DataFrame, risk_states: Dict[str, pd.Series], params: dict) -> pd.Series:
        """
        【V41.12 智能出场矩阵版】
        - 核心功能: 基于“风险状态”和“确认触发器”，通过“出场剧本矩阵”生成具体的退出信号。
        - 核心思想: 将所有离场逻辑剧本化、状态化、斜率化。
        """
        print("    - [出场决策中心 V41.12] 启动，开始评估所有出场剧本...")
        exit_params = self._get_params_block(params, 'exit_strategy_params', {})
        if not self._get_param_value(exit_params.get('enabled'), False):
            return pd.Series(0, index=df.index)

        exit_signal = pd.Series(0, index=df.index) # 0代表无信号
        has_exited = pd.Series(False, index=df.index) # 防止同一天被多个剧本触发

        # --- 准备通用触发器和斜率 ---
        is_bearish_candle = df['close_D'] < df['open_D'] # 阴线
        is_breaking_ma5 = df['close_D'] < df.get('EMA_5_D', df['close_D'])
        rsi_slope_col = 'SLOPE_RSI_14_D_3' # 假设已在指标计算中添加了3日斜率
        rsi_turning_down = df.get(rsi_slope_col, pd.Series(0, index=df.index)) < 0

        # --- 出场剧本0.1: 结构崩溃 (最高优先级，无条件执行) ---
        p = exit_params.get('volume_breakdown_params', {})
        if self._get_param_value(p.get('enabled'), False) and not has_exited.all():
            condition = risk_states['RISK_STRUCTURE_BREAKDOWN'] & ~has_exited
            exit_signal[condition] = self._get_param_value(p.get('exit_code'), 17)
            has_exited[condition] = True

        # --- 出场剧本0.2: 主力叛逃 (次高优先级) ---
        p = exit_params.get('upthrust_distribution_params', {})
        if self._get_param_value(p.get('enabled'), False) and not has_exited.all():
            condition = risk_states['RISK_UPTHRUST_DISTRIBUTION'] & ~has_exited
            exit_signal[condition] = self._get_param_value(p.get('exit_code'), 16)
            has_exited[condition] = True

        # --- 出场剧本1: 顶背离确认 (最高优先级) ---
        p = exit_params.get('divergence_exit_params', {})
        if self._get_param_value(p.get('enabled'), False) and not has_exited.all():
            confirmation_days = self._get_param_value(p.get('confirmation_days'), 5)
            # 风险状态: N天内曾出现顶背离
            had_recent_divergence = risk_states['RISK_TOP_DIVERGENCE'].rolling(window=confirmation_days, min_periods=1).sum() > 0
            # 触发器: 出现阴线或跌破5日线
            trigger = is_bearish_candle | is_breaking_ma5
            condition = had_recent_divergence.shift(1).fillna(False) & trigger & ~risk_states['RISK_TOP_DIVERGENCE'] & ~has_exited
            exit_signal[condition] = self._get_param_value(p.get('exit_code'), 15)
            has_exited[condition] = True

        # --- 出场剧本2: 指标超买且掉头 ---
        p = exit_params.get('indicator_exit_params', {})
        if self._get_param_value(p.get('enabled'), False) and not has_exited.all():
            # RSI剧本: 处于超买状态，且RSI斜率开始向下
            rsi_condition = risk_states['RISK_RSI_OVERBOUGHT'] & rsi_turning_down & ~has_exited
            exit_signal[rsi_condition] = self._get_param_value(p.get('exit_code_rsi'), 13)
            has_exited[rsi_condition] = True
            # BIAS剧本 (可以简化为超买即触发)
            bias_condition = risk_states['RISK_BIAS_OVERBOUGHT'] & ~has_exited
            exit_signal[bias_condition] = self._get_param_value(p.get('exit_code_bias'), 14)
            has_exited[bias_condition] = True

        # --- 出场剧本3: 移动止损 (硬规则) ---
        p = exit_params.get('trailing_stop_params', {})
        if self._get_param_value(p.get('enabled'), False) and not has_exited.all():
            lookback = self._get_param_value(p.get('lookback_period'), 20)
            pullback = self._get_param_value(p.get('percentage_pullback'), 0.10)
            highest_close_since = df['close_D'].shift(1).rolling(window=lookback).max()
            stop_price = highest_close_since * (1 - pullback)
            condition = (df['close_D'] < stop_price) & highest_close_since.notna() & ~has_exited
            exit_signal[condition] = self._get_param_value(p.get('exit_code'), 12)
            has_exited[condition] = True

        # --- 出场剧本4: 前期高点压力 (结构性止盈) ---
        p = exit_params.get('resistance_exit_params', {})
        if self._get_param_value(p.get('enabled'), False) and not has_exited.all():
            lookback = self._get_param_value(p.get('lookback_period'), 90)
            threshold = self._get_param_value(p.get('approach_threshold'), 0.01)
            resistance_level = df['high_D'].shift(1).rolling(window=lookback).max()
            # 风险状态: 接近压力位
            is_near_resistance = df['high_D'] >= resistance_level * (1 - threshold)
            # 触发器: 出现阴线
            condition = is_near_resistance & is_bearish_candle & resistance_level.notna() & ~has_exited
            exit_signal[condition] = self._get_param_value(p.get('exit_code'), 11)
            has_exited[condition] = True
        
        print(f"    - [出场决策中心 V41.13] 完成，共产生 { (exit_signal > 0).sum() } 个出场信号。")
        return exit_signal

    # ▼▼▼ “准备状态中心” (Setup Condition Center) ▼▼▼
    def _calculate_setup_conditions(self, df: pd.DataFrame, params: dict, trigger_events: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """
        【V45.0 健壮重构版】
        - 核心重构: 彻底重写S级剧本的准备状态计算逻辑，放弃复杂脆弱的形态识别，
                    改用基于滚动窗口的、更健壮的“区间状态”判断，确保信号能被稳定识别。
        """
        print("    - [准备状态中心 V45.0] 启动，开始计算所有准备状态...")
        setups = {}
        setup_params = self._get_params_block(params, 'setup_condition_params', {})
        playbook_specific_params = self._get_params_block(params, 'playbook_specific_params', {})

        # --- 1. 调用原子信号中心 (逻辑不变) ---
        chip_atomic_signals = self._define_chip_atomic_signals(df, params)
        setups['SETUP_CHIP_ACCUMULATION'] = chip_atomic_signals.get('ATOMIC_REINFORCEMENT', pd.Series(False, index=df.index))

        # --- 2. 【S级剧本重构】“潜龙出海”的准备状态 (SETUP_PROLONGED_COMPRESSION) ---
        try:
            p = playbook_specific_params.get('perfect_storm_params', {})
            if self._get_param_value(p.get('enabled'), True):
                lookback = self._get_param_value(p.get('compression_lookback'), 60)
                slope_col_name = f'SLOPE_net_mf_amount_D_{lookback}'
                required_cols = ['cost_95pct_D', 'cost_15pct_D', 'close_D', slope_col_name]
                if all(c in df.columns for c in required_cols):
                    initial_setup_signal = (
                        ((df['cost_95pct_D'] - df['cost_15pct_D']) / df['close_D'] < self._get_param_value(p.get('chip_concentration_threshold'), 0.30)) &
                        (df['close_D'] > df['cost_15pct_D']) &
                        (df[slope_col_name] > 0)
                    )
                    persistence_days = self._get_param_value(p.get('setup_persistence_days'), 5)
                    setup_timer = pd.Series(0, index=df.index, dtype=int)
                    for i in range(len(df)):
                        current_index = df.index[i]
                        if initial_setup_signal.at[current_index]:
                            setup_timer.at[current_index] = persistence_days
                        elif i > 0:
                            prev_index = df.index[i-1]
                            if setup_timer.at[prev_index] > 0:
                                setup_timer.at[current_index] = setup_timer.at[prev_index] - 1
                    break_threshold = self._get_param_value(p.get('setup_break_threshold'), 0.98)
                    is_setup_broken = df['close_D'] < (df['cost_15pct_D'] * break_threshold)
                    final_setup = (setup_timer > 0) & ~is_setup_broken
                    setups['SETUP_PROLONGED_COMPRESSION'] = final_setup
                    print(f"      -> '潜龙出海'准备状态(状态机修正版)定义完成，发现 {final_setup.sum()} 天。")
        except Exception as e:
            print(f"      -> [警告] 计算'潜龙出海'时出错: {e}")

        # --- 3. 【S级剧本重构】“猛兽苏醒”的准备状态 (SETUP_CAPITAL_FLOW_DIVERGENCE) ---
        # --- 剧本A: 动能背离 (Momentum Divergence) ---
        try:
            p = setup_params.get('momentum_divergence_params', {})
            if self._get_param_value(p.get('enabled'), True):
                lookback = self._get_param_value(p.get('lookback'), 20)
                long_period = self._get_param_value(p.get('trend_ma'), 55)
                vlong_period = self._get_param_value(p.get('regime_ma'), 144)
                
                long_ma_col = f"EMA_{long_period}_D"
                accel_col = f'ACCEL_EMA_{vlong_period}_D_{lookback}'
                jerk_col = f'SLOPE_{accel_col}_{lookback}'
                
                required_cols = ['close_D', long_ma_col, accel_col, jerk_col]
                
                if not all(col in df.columns for col in required_cols):
                    missing = [col for col in required_cols if col not in df.columns]
                    print(f"\n--- [前置检查失败] '动能背离(终极解耦)' 无法计算，缺少列: {missing} ---\n")
                else:
                    is_price_weak = df['close_D'] < df[long_ma_col]
                    is_accel_positive = df[accel_col] > 0
                    
                    jerk_t = df[jerk_col]
                    jerk_t_minus_1 = df[jerk_col].shift(1)
                    jerk_t_minus_2 = df[jerk_col].shift(2)
                    
                    is_peak_confirmed = (jerk_t_minus_1 > jerk_t_minus_2) & (jerk_t_minus_1 > jerk_t)
                    
                    # 最终的准备状态(Setup)只由核心事件决定
                    final_setup = is_peak_confirmed
                    
                    setups['SETUP_MOMENTUM_DIVERGENCE'] = final_setup
                    print(f"      -> '动能背离'(终极解耦版) 完成: 寻找“Jerk峰值已过”的事件日，发现 {final_setup.sum()} 天。")

                    # --- 聚焦探针逻辑 (诊断信息更全面) ---
                    # probe_start_date = pd.to_datetime('2024-08-01').tz_localize(df.index.tz)
                    # probe_end_date = pd.to_datetime('2024-09-30').tz_localize(df.index.tz)
                    
                    # key_dates_to_check_raw = ['2024-08-21', '2024-08-22']
                    # key_dates_as_date_obj = [pd.to_datetime(d).date() for d in key_dates_to_check_raw]
                    
                    # df_dates = df.index.date
                    # is_key_date = pd.Series(df_dates, index=df.index).isin(key_dates_as_date_obj)
                    
                    # window_mask = (df.index >= probe_start_date) & (df.index <= probe_end_date)
                    # interesting_days_mask = window_mask & (final_setup | is_key_date)
                    
                    # if interesting_days_mask.any():
                    #     probe_df = pd.DataFrame({
                    #         'Close': df['close_D'], 'PriceWeakOK': is_price_weak,
                    #         'Accel': df[accel_col], 'AccelOK': is_accel_positive,
                    #         'Jerk(T-1)': jerk_t_minus_1, 'Jerk(T)': jerk_t,
                    #         '_SETUP': final_setup
                    #     }).loc[interesting_days_mask]
                        
                    #     if not probe_df.empty:
                    #         print("\n--- [终极探针-SETUP] 诊断 '动能背离' (V45.50 终极解耦版) ---")
                    #         print(probe_df.to_string(float_format="%.6f"))
                    #         print("--- [终极探针] 诊断结束 ---\n")
        except Exception as e:
            print(f"      -> [警告] 计算'动能背离(终极解耦)'时出错: {e}")
            
        # --- 剧本B: 资本背离 (Capital Divergence) ---
        try:
            # 修正: 从 'capital_flow_divergence_params' 读取参数，而不是 'capital_divergence_params'
            p = setup_params.get('capital_flow_divergence_params', {})
            if self._get_param_value(p.get('enabled'), True):
                print("      -> '资本背离' (JSON参数驱动版) 启动...")
                # 1. 从JSON加载所有参数
                trend_ma_period = self._get_param_value(p.get('trend_ma_period'), 55)
                mf_slope_threshold = self._get_param_value(p.get('mf_slope_threshold'), 0)
                mf_accel_threshold = self._get_param_value(p.get('mf_accel_threshold'), 0)
                retail_slope_threshold = self._get_param_value(p.get('retail_slope_threshold'), 0)
                price_accel_threshold = self._get_param_value(p.get('price_accel_threshold'), 0)
                bbw_slope_threshold = self._get_param_value(p.get('bbw_slope_threshold'), 0)

                # 2. 定义所有需要的列名 (与 slope_params 中的定义匹配)
                trend_ma_col = f'EMA_{trend_ma_period}_D'
                mf_slope_col = 'SLOPE_net_mf_amount_D_10' # 使用10周期斜率
                mf_accel_col = 'ACCEL_net_mf_amount_D_10' # 使用10周期加速度
                retail_slope_col = 'SLOPE_net_retail_amount_D_20' # 使用20周期斜率
                price_accel_col = 'ACCEL_close_D_10' # 使用10周期加速度
                bbw_slope_col = 'SLOPE_BBW_21_2.0_D_10' # 使用10周期斜率

                required_cols = [
                    'close_D', trend_ma_col, mf_slope_col, mf_accel_col,
                    retail_slope_col, price_accel_col, bbw_slope_col
                ]

                # 3. 检查所有必需的列是否存在
                if not all(col in df.columns for col in required_cols):
                    missing = [col for col in required_cols if col not in df.columns]
                    print(f"      -> [警告] '资本背离'无法计算，缺少衍生列: {missing}")
                    setups['SETUP_CAPITAL_DIVERGENCE'] = pd.Series(False, index=df.index)
                else:
                    # 4. 根据JSON参数构建所有条件
                    cond_price_weak = df['close_D'] < df[trend_ma_col]
                    cond_mf_slope_improving = df[mf_slope_col] > mf_slope_threshold
                    cond_mf_accelerating = df[mf_accel_col] > mf_accel_threshold
                    cond_retail_selling = df[retail_slope_col] < retail_slope_threshold
                    cond_price_stabilizing = df[price_accel_col] > price_accel_threshold
                    cond_volatility_squeezing = df[bbw_slope_col] < bbw_slope_threshold

                    # 5. 组合所有条件形成最终的准备状态
                    final_setup = (
                        cond_price_weak &
                        cond_mf_slope_improving &
                        cond_mf_accelerating &
                        cond_retail_selling &
                        cond_price_stabilizing &
                        cond_volatility_squeezing
                    )
                    setups['SETUP_CAPITAL_DIVERGENCE'] = final_setup

                    # --- 新增: 深度诊断探针 (指定时间范围) ---
                    # 定义探针的开始和结束日期，聚焦于问题发生的时间段
                probe_start_date = pd.to_datetime('2024-08-15').tz_localize(df.index.tz)
                probe_end_date = pd.to_datetime('2024-08-25').tz_localize(df.index.tz)
                print(f"\n--- [深度探针-SETUP | {probe_start_date.date()} to {probe_end_date.date()}] 诊断 '资本背离' ---")
                
                # 根据指定的开始和结束日期筛选数据
                probe_df = df[(df.index >= probe_start_date) & (df.index <= probe_end_date)]

                # 检查probe_df是否为空，避免在没有数据时报错
                if not probe_df.empty:
                    for timestamp, row in probe_df.iterrows():
                        is_setup_ok_today = final_setup.get(timestamp, False)
                        
                        if is_setup_ok_today:
                            print(f"====== 日期: {timestamp.date()} | [✔ 成功] ======")
                            print("  - 所有条件均满足。")
                        else:
                            print(f"====== 日期: {timestamp.date()} | [✖ 失败] ======")
                            failure_reasons = []
                            # 逐一检查每个条件，并记录失败原因、阈值和实际值
                            if not cond_price_weak.get(timestamp, True):
                                reason = f"  - 价格弱势: 失败 (要求: close < EMA_55, 实际: {row['close_D']:.2f} >= {row[trend_ma_col]:.2f})"
                                failure_reasons.append(reason)
                            if not cond_mf_slope_improving.get(timestamp, True):
                                reason = f"  - 主力资金斜率改善: 失败 (要求: > {mf_slope_threshold}, 实际: {row[mf_slope_col]:.2f})"
                                failure_reasons.append(reason)
                            if not cond_mf_accelerating.get(timestamp, True):
                                reason = f"  - 主力资金流入加速: 失败 (要求: > {mf_accel_threshold}, 实际: {row[mf_accel_col]:.2f})"
                                failure_reasons.append(reason)
                            if not cond_retail_selling.get(timestamp, True):
                                reason = f"  - 散户资金流出: 失败 (要求: < {retail_slope_threshold}, 实际: {row[retail_slope_col]:.2f})"
                                failure_reasons.append(reason)
                            if not cond_price_stabilizing.get(timestamp, True):
                                reason = f"  - 价格下跌趋缓: 失败 (要求: > {price_accel_threshold}, 实际: {row[price_accel_col]:.4f})"
                                failure_reasons.append(reason)
                            if not cond_volatility_squeezing.get(timestamp, True):
                                reason = f"  - 波动率收缩: 失败 (要求: < {bbw_slope_threshold}, 实际: {row[bbw_slope_col]:.4f})"
                                failure_reasons.append(reason)
                            
                            if failure_reasons:
                                for r in failure_reasons:
                                    print(r)
                            else:
                                print("  - 状态为失败但所有子条件检查通过，可能存在数据对齐问题或逻辑边缘情况。")
                
                    print("--- [探针] 诊断结束 ---\n")
                    
                    print(f"      -> '资本背离'(JSON参数驱动版) 完成: 发现 {final_setup.sum()} 天满足所有条件。")
        except Exception as e:
            print(f"      -> [警告] 计算'资本背离(JSON参数驱动版)'时出错: {e}")

        # --- 能量压缩 ---
        try:
            p = setup_params.get('energy_compression_params', {})
            if self._get_param_value(p.get('enabled'), False):
                volatility_slope_col = 'SLOPE_BBW_21_2.0_D_10'
                volume_slope_col = 'SLOPE_VOL_MA_21_D_10'
                price_slope_col = 'SLOPE_close_D_10'
                if all(c in df.columns for c in [volatility_slope_col, volume_slope_col, price_slope_col]):
                    cond1 = df[volatility_slope_col] < self._get_param_value(p.get('volatility_slope_threshold'), -0.001)
                    cond2 = df[volume_slope_col] < self._get_param_value(p.get('volume_slope_threshold'), -1.0)
                    cond3 = df[price_slope_col].between(
                        self._get_param_value(p.get('price_slope_threshold_lower'), -0.1),
                        self._get_param_value(p.get('price_slope_threshold_upper'), 0.05)
                    )
                    setups['SETUP_ENERGY_COMPRESSION'] = cond1 & cond2 & cond3
                    print(f"      -> '能量压缩'准备状态定义完成，发现 {setups.get('SETUP_ENERGY_COMPRESSION', pd.Series([])).sum()} 天。")
        except Exception as e:
            print(f"      -> [警告] 计算'能量压缩'时出错: {e}")

        # --- 4. 健康回踩 ---
        try:
            p = setup_params.get('healthy_pullback_params', {})
            if self._get_param_value(p.get('enabled'), True):
                short_period, mid_period = self._get_param_value(p.get('short_ma'), 13), self._get_param_value(p.get('mid_ma'), 21)
                long_period, vlong_period = self._get_param_value(p.get('trend_ma'), 55), self._get_param_value(p.get('regime_ma'), 144)
                short_ma_col, mid_ma_col = f"EMA_{short_period}_D", f"EMA_{mid_period}_D"
                long_ma_col, vlong_ma_col = f"EMA_{long_period}_D", f"EMA_{vlong_period}_D"
                long_ma_slope_col, vlong_ma_slope_col = f'SLOPE_{long_ma_col}_5', f'SLOPE_{vlong_ma_col}_20'
                vol_ma_col = 'VOL_MA_21_D'
                required_cols = [short_ma_col, mid_ma_col, long_ma_col, vlong_ma_col, long_ma_slope_col, vlong_ma_slope_col, vol_ma_col]
                if all(col in df.columns for col in required_cols):
                    is_volume_shrinking, is_in_pullback_state = df['volume_D'] < df[vol_ma_col], df['close_D'] < df[short_ma_col]
                    common_conditions_ok = is_volume_shrinking & is_in_pullback_state
                    is_structure_ok = df[long_ma_col] > df[vlong_ma_col]
                    slope_epsilon = self._get_param_value(p.get('slope_epsilon'), -0.001)
                    is_momentum_ok = df[vlong_ma_slope_col] > slope_epsilon
                    is_bull_regime = is_structure_ok | is_momentum_ok
                    is_testing_lma = df['low_D'] <= df[long_ma_col] * self._get_param_value(p.get('support_buffer_long'), 1.02)
                    deep_pullback_setup = is_bull_regime & is_testing_lma & common_conditions_ok
                    is_lma_slope_ok, is_far_above_lma = df[long_ma_slope_col] > 0, df['low_D'] > df[long_ma_col]
                    is_testing_mma = df['low_D'] <= df[mid_ma_col] * self._get_param_value(p.get('support_buffer_mid'), 1.02)
                    shallow_pullback_setup = is_lma_slope_ok & is_far_above_lma & is_testing_mma & common_conditions_ok
                    setups['SETUP_HEALTHY_PULLBACK'] = deep_pullback_setup | shallow_pullback_setup
        except Exception as e:
            print(f"      -> [警告] 计算'健康回踩'时出错: {e}")

        # --- 4. 剧本联动: 突破后回踩 (Post-Breakout Pullback) ---
        try:
            p = setup_params.get('pullback_post_breakout_params', {})
            if self._get_param_value(p.get('enabled'), True):
                
                # 1. 启动器: 直接从传入的参数中获取“主力点火”事件
                ignition_trigger = trigger_events.get('CHIP_INSTITUTIONAL_BREAKOUT', pd.Series(False, index=df.index))
                
                # 2. 支撑线: 在“主力点火”当天，捕获当日的95%成本线作为支撑 (逻辑不变)
                support_level_on_ignition = df['cost_95pct_D'].where(ignition_trigger)
                
                # 3. 让支撑位在点火后持续有效N天 (逻辑不变)
                lookback_days = self._get_param_value(p.get('lookback_days'), 15)
                active_support = support_level_on_ignition.ffill(limit=lookback_days)
                
                # 4. 定义回踩状态 (逻辑不变)
                proximity_pct = self._get_param_value(p.get('proximity_pct'), 0.02)
                depth_pct = self._get_param_value(p.get('depth_pct'), 0.03)
                gravity_zone_upper = active_support * (1 + proximity_pct)
                gravity_zone_lower = active_support * (1 - depth_pct)
                is_pullback_to_support = (df['low_D'] <= gravity_zone_upper) & (df['low_D'] >= gravity_zone_lower) & active_support.notna()
                is_volume_shrinking = df['volume_D'] < df['VOL_MA_21_D']
                
                final_setup = is_pullback_to_support & is_volume_shrinking
                setups['SETUP_PULLBACK_POST_BREAKOUT'] = final_setup
                print(f"      -> '突破后回踩'(架构重构版)准备状态定义完成，发现 {final_setup.sum()} 天。")

                # --- 专属探针逻辑 (同步升级) ---
                # probe_start_date = pd.to_datetime('2024-07-01', utc=True)
                # probe_df = pd.DataFrame({
                #     'IgnitionDay': ignition_trigger,
                #     'SupportLine': active_support,
                #     'DayLow': df['low_D'],
                #     'ZoneLower': gravity_zone_lower, # 新增: 引力区下轨
                #     'ZoneUpper': gravity_zone_upper, # 新增: 引力区上轨
                #     'Price_OK': is_pullback_to_support,
                #     'Volume_OK': is_volume_shrinking,
                #     '_Setup': final_setup
                # }).loc[probe_start_date:]
                
                # interesting_days = probe_df[probe_df[['IgnitionDay', '_Setup']].any(axis=1)]
                # if not interesting_days.empty:
                #     print("\n--- [终极探针-SETUP | >24-07-01] 诊断 '突破后回踩' (精准引力区版) ---")
                #     print(interesting_days.to_string(float_format="%.2f"))
                #     print("--- [终极探针] 诊断结束 ---\n")

        except Exception as e:
            print(f"      -> [警告] 计算'突破后回踩'(架构重构版)时出错: {e}")

        # --- 5. 老鸭头-鸭颈 ---
        try:
            p = setup_params.get('duck_neck_forming_params', {})
            if self._get_param_value(p.get('enabled'), False):
                fast_ma = f"EMA_{self._get_param_value(p.get('fast_ma'), 10)}_D"
                slow_ma = f"EMA_{self._get_param_value(p.get('slow_ma'), 55)}_D"
                if all(c in df.columns for c in [fast_ma, slow_ma]):
                    cond1 = df[fast_ma] < df[slow_ma]
                    cond2 = df['volume_D'] < df['VOL_MA_21_D']
                    setups['SETUP_DUCK_NECK_FORMING'] = cond1 & cond2
                    print(f"      -> '老鸭头-鸭颈'准备状态定义完成，发现 {setups.get('SETUP_DUCK_NECK_FORMING', pd.Series([])).sum()} 天。")
        except Exception as e:
            print(f"      -> [警告] 计算'老鸭头-鸭颈'时出错: {e}")

        # --- 6. 巨阴洗盘日 ---
        try:
            p = setup_params.get('washout_day_params', {})
            if self._get_param_value(p.get('enabled'), False):
                vol_ma_period = self._get_param_value(p.get('vol_ma_period'), 21)
                vol_ma_col = f'VOL_MA_{vol_ma_period}_D'
                change_pct = df['close_D'].pct_change()
                is_big_drop = change_pct < self._get_param_value(p.get('washout_threshold'), -0.07)
                is_high_vol = df['volume_D'] > (df[vol_ma_col] * self._get_param_value(p.get('washout_volume_ratio'), 1.5))
                setups['SETUP_WASHOUT_DAY'] = (is_big_drop & is_high_vol).shift(1).fillna(False)
                print(f"      -> '巨阴洗盘日'准备状态定义完成，发现 {setups.get('SETUP_WASHOUT_DAY', pd.Series([])).sum()} 天。")
        except Exception as e:
            print(f"      -> [警告] 计算'巨阴洗盘日'时出错: {e}")

        # --- 7. V反-休克谷底 ---
        try:
            p = setup_params.get('shock_bottom_params', {})
            if self._get_param_value(p.get('enabled'), False):
                drop_days = self._get_param_value(p.get('drop_days'), 10)
                drop_pct = self._get_param_value(p.get('drop_pct'), -0.20)
                winner_rate_threshold = self._get_param_value(p.get('winner_rate_threshold'), 10.0)
                
                # 条件1: 快速大幅下跌
                high_in_period = df['high_D'].rolling(window=drop_days).max()
                current_drop_pct = (df['low_D'] / high_in_period - 1)
                is_deep_drop = current_drop_pct < drop_pct
                
                # 条件2: 下跌中主力资金是净流出的 (真摔)
                main_force_flow_during_drop = df['net_mf_amount_D'].rolling(window=drop_days).sum()
                is_main_force_fleeing = main_force_flow_during_drop < 0
                
                # 条件3: 获利盘被清洗
                is_panic_selling = df['winner_rate_D'] < winner_rate_threshold
                
                setups['SETUP_SHOCK_BOTTOM'] = is_deep_drop & is_main_force_fleeing & is_panic_selling
                print(f"      -> 'V反-休克谷底'准备状态定义完成 (V41.3 真摔版)，发现 {setups.get('SETUP_SHOCK_BOTTOM', pd.Series([])).sum()} 天。")
        except Exception as e:
            print(f"      -> [警告] 计算'V反-休克谷底'时出错: {e}")

        # --- 8. 堡垒围攻 ---
        try:
            p = setup_params.get('fortress_siege_params', {})
            if self._get_param_value(p.get('enabled'), False):
                support_col = self._get_param_value(p.get('support_col'), 'prev_high_support_D')
                if support_col in df.columns:
                    # 1. 识别有效堡垒 (这是一个状态，将作为剧本的precondition)
                    touch_buffer = self._get_param_value(p.get('fortress_touch_buffer'), 0.02)
                    is_near_support = abs(df['low_D'] - df[support_col]) / df[support_col].replace(0, np.nan) < touch_buffer
                    touch_counts = is_near_support.rolling(window=self._get_param_value(p.get('fortress_lookback'), 60)).sum()
                    df['temp_is_fortress_valid'] = touch_counts >= self._get_param_value(p.get('fortress_min_touches'), 2)
                    
                    # 2. 定义“健康围攻”准备状态
                    vol_ma_period = self._get_param_value(p.get('vol_ma_period'), 21)
                    vol_ma_col = f"VOL_MA_{vol_ma_period}_D"
                    touched_support = df['low_D'] <= df[support_col]
                    is_volume_shrinking_before = df['volume_D'].shift(1) < df[vol_ma_col].shift(1)
                    setups['SETUP_FORTRESS_SIEGE'] = touched_support & is_volume_shrinking_before
                    print(f"      -> '堡垒围攻'准备状态定义完成 (V41.3)，发现 {setups.get('SETUP_FORTRESS_SIEGE', pd.Series([])).sum()} 天。")
                else:
                    df['temp_is_fortress_valid'] = False
        except Exception as e:
            print(f"      -> [警告] 计算'堡垒围攻'时出错: {e}")
            df['temp_is_fortress_valid'] = False

        # --- 十字星暂停 (Doji Pause) ---
        try:
            p = setup_params.get('doji_pause_params', {})
            if self._get_param_value(p.get('enabled'), False):
                body_ratio_threshold = self._get_param_value(p.get('body_ratio_threshold'), 0.15)
                # 准备状态是“昨天是十字星”
                prev_body = abs(df['open_D'].shift(1) - df['close_D'].shift(1))
                prev_range = (df['high_D'].shift(1) - df['low_D'].shift(1)).replace(0, np.nan)
                is_doji_yesterday = (prev_body / prev_range) < body_ratio_threshold
                setups['SETUP_DOJI_PAUSE'] = is_doji_yesterday.fillna(False)
                print(f"      -> '十字星暂停'准备状态定义完成，发现 {setups.get('SETUP_DOJI_PAUSE', pd.Series([])).sum()} 天。")
        except Exception as e:
            print(f"      -> [警告] 计算'十字星暂停'时出错: {e}")
        
        # --- BIAS极端超卖 ---
        try:
            p = playbook_specific_params.get('bias_reversal_params', {})
            if self._get_param_value(p.get('enabled'), False):
                bias_period = self._get_param_value(p.get('bias_period'), 20)
                bias_col = f"BIAS_{bias_period}_D"
                if bias_col in df.columns:
                    oversold_threshold = self._get_param_value(p.get('oversold_threshold'), -15.0)
                    setups['SETUP_BIAS_EXTREME_OVERSOLD'] = df[bias_col] < oversold_threshold
                    print(f"      -> 'BIAS极端超卖'准备状态定义完成，发现 {setups.get('SETUP_BIAS_EXTREME_OVERSOLD', pd.Series([])).sum()} 天。")
                else:
                    print(f"      -> [警告] 缺少列 '{bias_col}'，无法计算'BIAS极端超卖'准备状态。")
        except Exception as e:
            print(f"      -> [警告] 计算'BIAS极端超卖'时出错: {e}")

        # --- 获利盘洗净 ---
        try:
            p = playbook_specific_params.get('winner_rate_reversal_params', {})
            if self._get_param_value(p.get('enabled'), False):
                winner_rate_col = 'winner_rate_D'
                if winner_rate_col in df.columns:
                    washout_threshold = self._get_param_value(p.get('washout_threshold'), 10.0)
                    # 准备状态是前一天获利盘极低
                    setups['SETUP_WINNER_RATE_WASHED_OUT'] = df[winner_rate_col].shift(1) < washout_threshold
                    print(f"      -> '获利盘洗净'准备状态定义完成，发现 {setups.get('SETUP_WINNER_RATE_WASHED_OUT', pd.Series([])).sum()} 天。")
                else:
                    print(f"      -> [警告] 缺少列 '{winner_rate_col}'，无法计算'获利盘洗净'准备状态。")
        except Exception as e:
            print(f"      -> [警告] 计算'获利盘洗净'时出错: {e}")

        # --- 布林带压缩 ---
        try:
            p = setup_params.get('bband_squeeze_params', {})
            if self._get_param_value(p.get('enabled'), False):
                bbw_col = 'BBW_21_2.0_D'
                if bbw_col in df.columns:
                    squeeze_lookback = self._get_param_value(p.get('squeeze_lookback'), 60)
                    squeeze_percentile = self._get_param_value(p.get('squeeze_percentile'), 0.10)
                    low_bbw_threshold = df[bbw_col].shift(1).rolling(window=squeeze_lookback).quantile(squeeze_percentile)
                    setups['SETUP_BBAND_SQUEEZE'] = df[bbw_col].shift(1) < low_bbw_threshold
                    print(f"      -> '布林带压缩'准备状态定义完成 (V41.4)，发现 {setups.get('SETUP_BBAND_SQUEEZE', pd.Series([])).sum()} 天。")
        except Exception as e:
            print(f"      -> [警告] 计算'布林带压缩'时出错: {e}")

        # --- 新增: 缺口支撑 (Upward Gap Support) ---
        try:
            p = playbook_specific_params.get('gap_support_params', {})
            if self._get_param_value(p.get('enabled'), False):
                gap_check_days = self._get_param_value(p.get('gap_check_days'), 10)
                gap_support_level = df['high_D'].shift(1)
                is_gap_up = df['low_D'] > gap_support_level
                active_support = pd.Series(np.nan, index=df.index)
                active_support.loc[is_gap_up] = gap_support_level[is_gap_up]
                active_support = active_support.ffill(limit=gap_check_days)
                group = (active_support.diff() != 0).cumsum()
                # 检查从缺口日之后，最低价是否一直高于支撑位
                lowest_since_gap = df.groupby(group)['low_D'].transform('cummin')
                is_supported = lowest_since_gap > active_support
                # 最终状态是：当前存在一个有效的、未被回补的缺口支撑
                setups['SETUP_GAP_SUPPORT'] = is_supported.fillna(False)
                print(f"      -> '缺口支撑'准备状态定义完成，发现 {setups.get('SETUP_GAP_SUPPORT', pd.Series([])).sum()} 天。")
        except Exception as e:
            print(f"      -> [警告] 计算'缺口支撑'时出错: {e}")

        # --- 风险：趋势衰竭 ---
        try:
            p = setup_params.get('trend_exhaustion_params', {})
            if self._get_param_value(p.get('enabled'), False):
                slope_col = 'SLOPE_EMA_55_D_10'
                accel_col = 'ACCEL_EMA_55_D_10'
                if all(c in df.columns for c in [slope_col, accel_col]):
                    is_still_rising = df[slope_col] > 0
                    is_decelerating = df[accel_col] < 0
                    setups['RISK_TREND_EXHAUSTION'] = is_still_rising & is_decelerating
                    print(f"      -> '风险-趋势衰竭'状态定义完成，发现 {setups.get('RISK_TREND_EXHAUSTION', pd.Series([])).sum()} 天。")
        except Exception as e:
            print(f"      -> [警告] 计算'趋势衰竭'时出错: {e}")

        # --- 新增: 均线回撤预备 (Pullback to MA) ---
        try:
            p = playbook_specific_params.get('pullback_ma_entry_params', {})
            if self._get_param_value(p.get('enabled'), False):
                support_ma_col = f"EMA_{self._get_param_value(p.get('support_ma'), 20)}_D"
                if support_ma_col in df.columns:
                    # 预备状态的定义：
                    # 1. 满足由健康度审计中心定义的高置信度右侧前提
                    precondition = self._validate_core_premises(df, params)['robust_right_side_precondition']
                    # 2. 收盘价在支撑均线之上
                    is_above_support = df['close_D'] > df[support_ma_col]
                    # 3. 收盘价距离支撑均线足够近
                    proximity_threshold = self._get_param_value(p.get('setup_proximity_threshold'), 0.03)
                    is_close_to_support = (df['close_D'] / df[support_ma_col] - 1) < proximity_threshold
                    # 最终的预备信号
                    setups['SETUP_PULLBACK_MA'] = precondition & is_above_support & is_close_to_support
                    # 记录目标价位，可用于后续更精细的触发器
                    df[f'target_price_{support_ma_col}'] = df[support_ma_col].where(setups['SETUP_PULLBACK_MA'])
                    print(f"      -> '均线回撤预备'准备状态定义完成，发现 {setups.get('SETUP_PULLBACK_MA', pd.Series([])).sum()} 天。")
        except Exception as e:
            print(f"      -> [警告] 计算'均线回撤预备'时出错: {e}")

        # --- 11. 填充与清洗 ---
        all_needed_setups = [
            'SETUP_PROLONGED_COMPRESSION', 'SETUP_CAPITAL_FLOW_DIVERGENCE', 'SETUP_PULLBACK_WITH_MF_INFLOW',
            'SETUP_PULLBACK_POST_BREAKOUT', 'SETUP_PULLBACK_POST_REVERSAL', 'SETUP_DUCK_NECK_FORMING',
            'SETUP_ENERGY_COMPRESSION', 'SETUP_HEALTHY_PULLBACK', 'SETUP_SHOCK_BOTTOM', 'SETUP_WASHOUT_DAY',
            'SETUP_RELATIVE_STRENGTH', 'SETUP_FORTRESS_SIEGE', 'SETUP_CHIP_ACCUMULATION', 'RISK_TREND_EXHAUSTION',
            'SETUP_DOJI_PAUSE', 'SETUP_BULLISH_FLAG_FORMING', 'SETUP_BOTTOM_DIVERGENCE', 'SETUP_BBAND_SQUEEZE',
            'SETUP_CHIP_CONCENTRATION', 'SETUP_BIAS_EXTREME_OVERSOLD', 'SETUP_WINNER_RATE_WASHED_OUT', 
            'SETUP_GAP_SUPPORT', 'SETUP_PULLBACK_MA'
        ]
        for setup_name in all_needed_setups:
            if setup_name not in setups:
                setups[setup_name] = pd.Series(False, index=df.index)
            else:
                setups[setup_name].fillna(False, inplace=True)
            
        print("    - [准备状态中心 V41.7] 所有准备状态计算完成。")
        return setups
    
    # ▼▼▼ 智能衍生特征引擎 (Intelligent Derived Feature Engine) ▼▼▼
    def _prepare_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        【V40.6 终极净化版】
        - 核心逻辑: 先进行加减法（Decimal之间运算是安全的），生成新的Decimal列，
                    然后调用【智能版】净化器一次性转换所有Decimal列，最后再进行不兼容的除法运算。
        """
        print("    - [衍生特征中心 V40.6] 启动智能衍生特征引擎...")
        
        # --- 步骤1: 主力净流入额 (智能回退逻辑) ---
        # 允许Decimal之间的加减法，这会产生一个新的Decimal列
        if 'net_mf_amount_D' not in df.columns or df['net_mf_amount_D'].isnull().all():
            print("      -> 'net_mf_amount_D' 不存在或为空，执行回退计算...")
            if all(c in df.columns for c in ['buy_lg_amount_D', 'sell_lg_amount_D', 'buy_elg_amount_D', 'sell_elg_amount_D']):
                df['net_mf_amount_D'] = (df['buy_lg_amount_D'] + df['buy_elg_amount_D']) - (df['sell_lg_amount_D'] + df['sell_elg_amount_D'])
                print("        -> 回退计算 'net_mf_amount_D' 成功。")
            else:
                print("        -> [警告] 缺少计算 'net_mf_amount_D' 所需的原始资金流列，该列将填充为0。")
                df['net_mf_amount_D'] = 0
        else:
             print("      -> 'net_mf_amount_D' (主力净流入) 已存在，使用预计算列。")

        # 【核心修改】将资本累积指标的计算提前
        # 注意：这里的lookback周期需要与后续setup中使用的保持一致。为简化，暂时硬编码。
        # 一个更优的方案是从参数文件中读取，但为聚焦当前问题，我们先用20。
        capital_lookback = 20 
        df[f'mf_accumulation_{capital_lookback}_D'] = df['net_mf_amount_D'].rolling(window=capital_lookback).sum()
        print(f"      -> 已计算 'mf_accumulation_{capital_lookback}_D' (主力资金{capital_lookback}日累积)。")

        # --- 步骤2: 散户净流入额 (智能回退逻辑) ---
        if 'net_retail_amount_D' not in df.columns or df['net_retail_amount_D'].isnull().all():
            print("      -> 'net_retail_amount_D' 不存在或为空，执行回退计算...")
            if all(c in df.columns for c in ['buy_sm_amount_D', 'sell_sm_amount_D']):
                df['net_retail_amount_D'] = df['buy_sm_amount_D'] - df['sell_sm_amount_D']
                print("        -> 回退计算 'net_retail_amount_D' 成功。")
            else:
                print("        -> [警告] 缺少计算 'net_retail_amount_D' 所需的原始资金流列，该列将填充为0。")
                df['net_retail_amount_D'] = 0
        else:
            print("      -> 'net_retail_amount_D' (散户净流入) 已存在，使用预计算列。")

        # --- 步骤3: 【核心修复点】在进行不兼容运算前，对整个DataFrame进行最终的智能净化 ---
        # 此时，所有可能引入Decimal的列（包括新创建的net_mf_amount_D）都已存在
        df = self._ensure_numeric_types(df)

        # --- 步骤4: 主力净流入额占比 (现在可以安全计算) ---
        if 'amount_D' in df.columns and 'net_mf_amount_D' in df.columns:
            # 现在这里的除法是安全的，因为所有列都已经是float64
            df['net_mf_amount_ratio_D'] = np.divide(
                df['net_mf_amount_D'], 
                df['amount_D'], 
                out=np.zeros_like(df['net_mf_amount_D'].values, dtype=float), # 使用 .values 确保类型一致
                where=df['amount_D']!=0
            )
            print("      -> 已计算 'net_mf_amount_ratio_D' (主力净流入额占比)。")
        else:
            print("      -> [警告] 缺少 'amount_D' 或 'net_mf_amount_D'，无法计算主力净流入额占比。")

        print("    - [衍生特征中心 V40.6] 衍生特征准备完成。")
        return df

    # ▼▼▼ 多维趋势健康度审计中心 (Multi-Faceted Trend Health Audit Center) ▼▼▼
    # 引入更多数据维度，对趋势健康度进行更全面的审计。
    def _validate_core_premises(self, df: pd.DataFrame, params: dict) -> Dict[str, pd.Series]:
        """
        【V40.0 主升浪优化版】
        - 核心升级: 移除对“趋势加速”的硬性要求，使其不再作为健康度评分的否决项，从而大幅提升对稳定主升浪的识别能力。
        - 审计维度 (5个核心维度):
          1. 方向 (Direction): 趋势的速度。
          2. 共识 (Consensus): 均线族排列。
          3. 资本流 (Capital Flow): 主力资金流入的趋势。
          4. 波动性 (Volatility): 趋势扩张的稳定性。
          5. 内在强度 (Internal Strength): 趋势的强度与持续性 (ADX & RSI)。
        - 备注: “趋势加速” (Momentum) 被降级为纯粹的动态加分项，在 `_score_trend_dynamics` 中处理。
        - 输出: 一个包含0-5分制的“趋势健康度得分”和关键前提条件的字典。
        """
        print("    - [前提验证中心 V40.0 主升浪优化版] 开始启动'多维趋势健康度审计'...")
        premises = {}
        
        # --- 准备参数和列名 ---
        ma_params = self._get_params_block(params, 'mid_term_trend_params', {})
        short_ma = self._get_param_value(ma_params.get('short_ma'), 21)
        mid_ma = self._get_param_value(ma_params.get('mid_ma'), 55)
        slow_ma = self._get_param_value(ma_params.get('slow_ma'), 89)
        
        short_ma_col = f"EMA_{short_ma}_D"
        mid_ma_col = f"EMA_{mid_ma}_D"
        slow_ma_col = f"EMA_{slow_ma}_D"
        
        mid_ma_slope_col = f'SLOPE_{mid_ma_col}_10'
        mid_ma_accel_col = f'ACCEL_{mid_ma_col}_10'
        
        mf_slope_col = 'SLOPE_net_mf_amount_D_10'
        cmf_col = 'CMF_21_D'
        cmf_slope_col = 'SLOPE_CMF_21_D_10'
        bbw_slope_col = 'SLOPE_BBW_21_2.0_D_10'
        adx_col = 'ADX_14_D'
        rsi_col = 'RSI_13_D'

        volatility_params = ma_params.get('volatility_params', {})
        min_bbw_slope = self._get_param_value(volatility_params.get('min_bbw_slope'), 0)
        max_bbw_slope = self._get_param_value(volatility_params.get('max_bbw_slope'), 0.01)

        strength_params = ma_params.get('internal_strength_params', {})
        adx_threshold = self._get_param_value(strength_params.get('adx_threshold'), 20)
        rsi_exhaustion_threshold = self._get_param_value(strength_params.get('rsi_exhaustion_threshold'), 80)
        
        # 修改: 将默认健康分阈值从4(满分6)调整为3(满分5)，逻辑更合理
        health_score_threshold = self._get_param_value(ma_params.get('health_score_threshold'), 3)

        required_cols = [
            short_ma_col, mid_ma_col, slow_ma_col, mid_ma_slope_col, mid_ma_accel_col,
            mf_slope_col, bbw_slope_col, adx_col, rsi_col, cmf_col, cmf_slope_col
        ]

        # 检查缺失列的逻辑保持不变
        if not all(c in df.columns for c in required_cols):
            missing = [col for col in required_cols if col not in df.columns]
            print(f"    - [前提验证中心-警告]: 缺少审计所需的列: {missing}，前提验证将受限。请检查指标和斜率计算配置。")
            # ... (返回空前提的逻辑不变)
            return {
                'robust_right_side_precondition': pd.Series(False, index=df.index),
                'trend_health_score': pd.Series(0, index=df.index),
                'is_trend_accelerating': pd.Series(False, index=df.index)
            }

        # --- 维度1: 方向 (Direction) - 趋势速度为正？ ---
        dimension_direction = df[mid_ma_slope_col] > 0
        print(f"      -> 维度1 (方向): {dimension_direction.sum()} 天满足。")

        # --- 维度2: 动量 (Momentum) - 趋势是否在加速？ (仅计算，不计入核心健康分) ---
        dimension_momentum = df[mid_ma_accel_col] > 0
        print(f"      -> (参考维度) 动量: {dimension_momentum.sum()} 天满足。 (注: 不计入核心健康分)")

        # --- 维度3: 共识 (Consensus) - 均线族是否呈多头排列？ ---
        dimension_consensus = (df[short_ma_col] > df[mid_ma_col]) & (df[mid_ma_col] > df[slow_ma_col])
        print(f"      -> 维度2 (共识): {dimension_consensus.sum()} 天满足。")

        # --- 维度4: 资本流 (Capital Flow) - 主力资金是否持续流入？ ---
        is_mf_inflow_improving = df[mf_slope_col] > 0
        is_cmf_inflow_improving = df[cmf_slope_col] > 0
        is_cmf_positive = df[cmf_col] > 0
        dimension_capital = is_mf_inflow_improving & (is_cmf_positive | is_cmf_inflow_improving)
        print(f"      -> 维度3 (资本流): {dimension_capital.sum()} 天满足。")

        # --- 维度5: 波动性 (Volatility) - 趋势扩张是否稳定？ ---
        dimension_volatility = (df[bbw_slope_col] > min_bbw_slope) & (df[bbw_slope_col] < max_bbw_slope)
        print(f"      -> 维度4 (波动性): {dimension_volatility.sum()} 天满足。")

        # --- 维度6: 内在强度 (Internal Strength) - 趋势是否强劲且未衰竭？ ---
        is_trend_strong = df[adx_col] > adx_threshold
        is_not_exhausted = df[rsi_col] < rsi_exhaustion_threshold
        dimension_strength = is_trend_strong & is_not_exhausted
        print(f"      -> 维度5 (内在强度): {dimension_strength.sum()} 天满足。")

        # --- 形成健康度评分 (0-5分) ---
        # 修改: 移除了 dimension_momentum.astype(int)
        trend_health_score = (
            dimension_direction.astype(int) +
            dimension_consensus.astype(int) +
            dimension_capital.astype(int) +
            dimension_volatility.astype(int) +
            dimension_strength.astype(int)
        )

        # --- 定义最终的前提条件 ---
        robust_precondition = (trend_health_score >= health_score_threshold)
        # 将前提条件写入df，以便调试时查看
        df['robust_right_side_precondition'] = robust_precondition

        premises['robust_right_side_precondition'] = robust_precondition
        premises['trend_health_score'] = trend_health_score
        premises['is_trend_accelerating'] = dimension_momentum # 仍然传递给下游用于加分

        print(f"      -> [审计完成] '高置信度'右侧前提 (健康分>={health_score_threshold}/5) 触发了 {robust_precondition.sum()} 天。")

        premises['evidence_strength'] = trend_health_score 

        return premises
    
    # ▼▼▼ 动态健康度计分引擎 (Dynamic Health Scoring Engine) ▼▼▼
    def _score_trend_dynamics(self, df: pd.DataFrame, params: dict, validated_premises: Dict[str, pd.Series]) -> pd.Series:
        """
        【V40.1 终极健壮版】
        - 核心修复: 对从 `entry_scoring_params.points` 中读取的所有分数值，全面应用 `_get_param_value` 解析器。
        - 效果: 彻底解决了因计分参数采用 `{'value':...}` 结构而导致的 TypeError，使整个计分流程对配置格式完全免疫。
        """
        print("    - [动力学评分 V40.1] 开始基于'趋势健康度'进行动态计分...")
        
        # --- 准备参数和数据 ---
        scoring_params = self._get_params_block(params, 'entry_scoring_params', {})
        points = scoring_params.get('points', {})
        
        trend_params = self._get_params_block(params, 'mid_term_trend_params', {})
        health_score_threshold = self._get_param_value(trend_params.get('health_score_threshold'), 4)

        # 从前提验证结果中获取核心数据
        trend_health_score = validated_premises.get('trend_health_score', pd.Series(0, index=df.index))
        is_trend_accelerating = validated_premises.get('is_trend_accelerating', pd.Series(False, index=df.index))

        # 初始化动态得分Series
        dynamics_score = pd.Series(0, index=df.index, dtype=float)
        
        # --- 1. 等级化健康度奖励 (Graded Health Bonus) ---
        points_per_level = self._get_param_value(points.get('POINTS_PER_HEALTH_LEVEL'), 10)
        
        score_above_threshold = (trend_health_score - health_score_threshold).clip(lower=0)
        graded_bonus = score_above_threshold * points_per_level
        
        dynamics_score += graded_bonus
        
        if graded_bonus.sum() > 0:
            print(f"      -> '等级化健康度'奖励已应用。最高奖励: {graded_bonus.max()}分。")

        # --- 2. 完美健康度王牌奖励 (Perfect Health Ace Bonus) ---
        perfect_health_bonus = self._get_param_value(points.get('BONUS_PERFECT_HEALTH'), 50)
        is_perfect_health = (trend_health_score == 6)
        
        if is_perfect_health.any():
            dynamics_score.loc[is_perfect_health] += perfect_health_bonus
            print(f"      -> '完美健康度(6/6)'王牌奖励触发了 {is_perfect_health.sum()} 天，奖励 {perfect_health_bonus} 分。")

        # --- 3. 趋势加速奖励 (Trend Acceleration Bonus) ---
        accel_bonus = self._get_param_value(points.get('BONUS_TREND_ACCELERATING'), 25)

        if is_trend_accelerating.any():
            dynamics_score.loc[is_trend_accelerating] += accel_bonus
            print(f"      -> '趋势加速'奖励触发了 {is_trend_accelerating.sum()} 天，奖励 {accel_bonus} 分。")
            
        return dynamics_score

    # ▼▼▼ 健康度感知上下文引擎 (Health-Aware Context Engine) ▼▼▼
    def _update_contextual_states(self, df: pd.DataFrame, score_details_df: pd.DataFrame, validated_premises: Dict[str, pd.Series], params: dict) -> pd.DataFrame:
        """
        【V39.4 健康度感知上下文引擎版】
        - 核心升级:
        1. 动态持续时间: 上下文的倒计时天数由触发当日的 trend_health_score 决定。
        2. 动态优先级: 上下文的优先级也由 trend_health_score 决定。
        - 机制:
          - 配置文件中为每个剧本定义多个“质量等级”(tiers)。
          - 触发时，根据健康分匹配最高等级，获取对应的持续时间和优先级。
        """
        print("      - [上下文中心 V39.4] 正在启动'健康度感知上下文引擎'...")

        # --- 步骤 1: 从配置加载上下文定义 ---
        context_params = self._get_params_block(params, 'context_setting_params', {})
        context_definitions = context_params.get('definitions', {})

        # ▼▼▼【代码修改 V39.5】: 动态注入“资本背离”上下文的定义，并为其增加失效条件 ▼▼▼
        if 'INTERNAL_SET_CONTEXT_CAPITAL_DIVERGENCE' not in context_definitions:
            context_definitions['INTERNAL_SET_CONTEXT_CAPITAL_DIVERGENCE'] = {
                'base_name': 'CAPITAL_DIVERGENCE_ACTIVE',
                'tiers': [{'min_score': 0, 'priority': 5, 'duration': 3}],
                # 新增：定义状态的失效条件
                'invalidation': {
                    'enabled': True,
                    # 当状态开启时，记录当时的最低价作为“防守底线”
                    'anchor_metric': 'low_D', 
                    # 如果后续收盘价跌破了“防守底线”，则状态失效
                    'check_metric': 'close_D',
                    'condition': 'less' # check_metric < anchor_metric
                }
            }
            print("        -> [动态注入] 已为'资本背离'状态添加上下文定义(含失效条件)。")
        
        # 从前提验证结果中获取健康度评分
        trend_health_score = validated_premises.get('trend_health_score', pd.Series(0, index=df.index))

        # --- 步骤 2: 初始化或衰减所有计时器 ---
        print("        -> 步骤2.1: 初始化/衰减所有上下文计时器...")
        all_timer_cols = []
        # 从定义中动态收集所有可能的上下文基础名
        all_base_names = {v['base_name'] for k, v in context_definitions.items()}
        
        for base_name in all_base_names:
            timer_col = f"CONTEXT_{base_name}_TIMER"
            all_timer_cols.append(timer_col)
            if timer_col in df.columns:
                df[timer_col] = (df[timer_col].shift(1) - 1).fillna(0).clip(lower=0)
            else:
                df[timer_col] = 0

        # --- 步骤 3: 根据当日触发的剧本，【动态地】重置计时器和优先级 ---
        print("        -> 步骤2.2: 检查当日剧本触发，动态重置计时器和优先级...")
        # 创建用于存储当天动态优先级和持续时间的临时列
        df['temp_priority'] = 0
        df['temp_duration'] = 0
        # ▼▼▼【代码修改 V39.5】: 初始化锚点价格列 ▼▼▼
        for playbook_name, definition in context_definitions.items():
            if definition.get('invalidation', {}).get('enabled', False):
                anchor_col = f"CONTEXT_{definition['base_name']}_ANCHOR"
                if anchor_col not in df.columns:
                    df[anchor_col] = np.nan

        for playbook_name, definition in context_definitions.items():
            if playbook_name not in score_details_df.columns:
                continue
            
            triggered_mask = score_details_df[playbook_name] > 0
            if not triggered_mask.any():
                continue

            base_name = definition['base_name']
            tiers = sorted(definition['tiers'], key=lambda x: x['min_score'], reverse=True) # 按min_score从高到低排序
            
            # 对每个触发日，根据其健康分查找对应的等级
            for idx in df.index[triggered_mask]:
                score_at_trigger = trend_health_score.at[idx]
                
                # 匹配最高可用等级
                matched_tier = tiers[0] # 默认最高级
                for tier in tiers:
                    if score_at_trigger >= tier['min_score']:
                        matched_tier = tier
                        break # 找到第一个满足的最高等级就跳出
                
                current_priority = df.at[idx, 'temp_priority']
                if matched_tier['priority'] > current_priority:
                    df.at[idx, 'temp_priority'] = matched_tier['priority']
                    df.at[idx, 'temp_duration'] = matched_tier['duration']
                    df.at[idx, 'CONTEXT_ACTIVE_NAME'] = base_name
                    
                    if definition.get('invalidation', {}).get('enabled', False):
                        anchor_metric = definition['invalidation']['anchor_metric']
                        anchor_col = f"CONTEXT_{base_name}_ANCHOR"
                        if anchor_metric in df.columns:
                            df.at[idx, anchor_col] = df.at[idx, anchor_metric]
                            # ▼▼▼【调试探针 1: 状态开启与锚点记录】▼▼▼
                            print(f"          - [上下文引擎探针] 日期 {idx.date()}: 剧本 '{playbook_name}' 触发，开启 '{base_name}' 状态，持续 {matched_tier['duration']} 天。记录锚点价格 ({anchor_metric}): {df.at[idx, anchor_col]:.2f}")

        # --- 步骤 4: 更新计时器和主导上下文状态 ---
        print("        -> 步骤2.3: 更新主导上下文状态...")
        # 初始化主导上下文列
        if 'CONTEXT_ACTIVE_NAME' not in df.columns:
            df['CONTEXT_ACTIVE_NAME'] = 'NONE'
        
        # 状态继承：如果今天没有新事件，则继承昨天的状态
        df['CONTEXT_ACTIVE_NAME'] = df['CONTEXT_ACTIVE_NAME'].where(df['temp_priority'] > 0, df['CONTEXT_ACTIVE_NAME'].shift(1).fillna('NONE'))
        # ▼▼▼【代码修改 V39.5】: 继承锚点价格 ▼▼▼
        for playbook_name, definition in context_definitions.items():
             if definition.get('invalidation', {}).get('enabled', False):
                base_name = definition['base_name']
                anchor_col = f"CONTEXT_{base_name}_ANCHOR"
                # 如果今天没有新设置锚点，就继承昨天的
                df[anchor_col] = df[anchor_col].fillna(method='ffill')
        
        # 更新所有计时器
        for base_name in all_base_names:
            timer_col = f"CONTEXT_{base_name}_TIMER"
            # 如果今天的主导叙事是这个，就用动态计算出的duration重置计时器
            reset_mask = (df['CONTEXT_ACTIVE_NAME'] == base_name) & (df['temp_duration'] > 0)
            if reset_mask.any():
                df.loc[reset_mask, timer_col] = df.loc[reset_mask, 'temp_duration']
        # ▼▼▼【代码修改 V39.5-Debug】: 应用状态失效逻辑并增加详细探针 ▼▼▼
        for playbook_name, definition in context_definitions.items():
            invalidation_params = definition.get('invalidation', {})
            if invalidation_params.get('enabled', False):
                base_name = definition['base_name']
                timer_col = f"CONTEXT_{base_name}_TIMER"
                anchor_col = f"CONTEXT_{base_name}_ANCHOR"
                check_metric_col = invalidation_params['check_metric']
                
                if all(c in df.columns for c in [timer_col, anchor_col, check_metric_col]):
                    is_active_yesterday = df[timer_col].shift(1) > 0
                    is_invalidated = df[check_metric_col] < df[anchor_col]
                    invalidation_mask = is_active_yesterday & is_invalidated
                    
                    # 循环检查每一天，提供详细日志
                    for idx in df.index[invalidation_mask]:
                        anchor_val = df.at[idx, anchor_col]
                        check_val = df.at[idx, check_metric_col]
                        # ▼▼▼【调试探针 2: 状态失效检查】▼▼▼
                        print(f"          - [上下文引擎探针] 日期 {idx.date()}: 检查 '{base_name}' 失效条件... 检查价({check_metric_col}): {check_val:.2f} vs 锚点价({anchor_col}): {anchor_val:.2f}. 条件: {check_val < anchor_val} -> 状态失效!")
                        # ▲▲▲【调试探针结束】▲▲▲

                    if invalidation_mask.any():
                        df.loc[invalidation_mask, timer_col] = 0
                        df.loc[invalidation_mask, anchor_col] = np.nan
                        df.loc[invalidation_mask & (df['CONTEXT_ACTIVE_NAME'] == base_name), 'CONTEXT_ACTIVE_NAME'] = 'NONE'
                        # ▼▼▼【调试探针 3: 状态失效执行】▼▼▼
                        print(f"          - [上下文引擎探针] [!!] '{base_name}' 状态因跌破锚点而在 {invalidation_mask.sum()} 天被强制终止。")
                        # ▲▲▲【调试探针结束】▲▲▲

        # --- 步骤 5: 更新主导上下文的剩余时间和最终的原子布尔状态 ---
        print("        -> 步骤2.4: 更新主导上下文计时器和原子状态...")
        df['CONTEXT_ACTIVE_TIMER'] = 0
        for base_name in all_base_names:
            timer_col = f"CONTEXT_{base_name}_TIMER"
            bool_col = f"CONTEXT_{base_name}"
            
            mask = df['CONTEXT_ACTIVE_NAME'] == base_name
            if mask.any():
                df.loc[mask, 'CONTEXT_ACTIVE_TIMER'] = df.loc[mask, timer_col]
                
            df[bool_col] = df[timer_col] > 0

            # ▼▼▼【调试探针 4: 每日最终状态】▼▼▼
            active_today = df[bool_col]
            if active_today.any():
                for idx in df.index[active_today]:
                     print(f"          - [上下文引擎探针] 日期 {idx.date()}: '{base_name}' 最终状态为 Active, 计时器剩余: {df.at[idx, timer_col]:.0f}")
            # ▲▲▲【调试探针结束】▲▲▲

        # 清理临时列
        df.drop(columns=['temp_priority', 'temp_duration'], inplace=True)

        active_days = (df['CONTEXT_ACTIVE_NAME'] != 'NONE').sum()
        if active_days > 0:
            last_active_day = df[df['CONTEXT_ACTIVE_NAME'] != 'NONE'].index[-1]
            last_active_context = df.loc[last_active_day, 'CONTEXT_ACTIVE_NAME']
            last_active_timer = df.loc[last_active_day, 'CONTEXT_ACTIVE_TIMER']
            print(f"      - [上下文中心 V39.4] 状态机更新完成。共发现 {active_days} 个处于上下文状态的交易日。")
            print(f"        -> 最近的上下文: 日期={last_active_day.date()}, 主导叙事='{last_active_context}', 剩余时间={last_active_timer:.0f}天。")
        else:
            print("      - [上下文中心 V39.4] 状态机更新完成。未发现任何处于上下文状态的交易日。")

        return df

    # ▼▼▼ 触发事件融合中心 (Trigger Event Fusion Center) ▼▼▼
    def _define_trigger_events(self, df: pd.DataFrame, params: dict) -> Dict[str, pd.Series]:
        """
        【V41.8 精炼增强版】
        - 核心升级: 在V41.5的基础上，进行最终的精炼与增强。
          1. 命名一致性修正: 完成所有触发器的`TRIGGER_`前缀统一。
          2. 经典模式补全: 重新引入经典的“通用放量突破”触发器。
          3. 信号质量增强: 为“回踩反弹”触发器增加成交量确认，过滤弱反弹。
          4. 逻辑注释强化: 阐明“机构突破”与“游资突破”的区分逻辑。
        """
        print("    - [触发事件中心 V41.8] 启动，开始定义所有原子化触发事件...")
        triggers = {}
        trigger_params = self._get_params_block(params, 'trigger_event_params', {})
        playbook_specific_params = self._get_params_block(params, 'playbook_specific_params', {})
        
        # --- 步骤1: 获取底层的、原子的“结构”信号 ---
        try:
            chip_atomic_signals = self._define_chip_atomic_signals(df, params)
            is_breakout_structure = chip_atomic_signals.get('ATOMIC_COST_BREAKTHROUGH', pd.Series(False, index=df.index))
            is_acceleration_structure = (
                chip_atomic_signals.get('ATOMIC_HURDLE_CLEAR', pd.Series(False, index=df.index)) |
                chip_atomic_signals.get('ATOMIC_PRESSURE_RELEASE', pd.Series(False, index=df.index))
            )
            print("      -> '底层筹码结构'信号已加载。")
        except Exception as e:
            print(f"    - [错误] 底层筹码信号计算失败: {e}，将使用空信号继续。")
            is_breakout_structure = pd.Series(False, index=df.index)
            is_acceleration_structure = pd.Series(False, index=df.index)

        # --- 步骤2: 计算“力量”信号 (资金行为的确认) ---
        print("      -> 正在计算'资金力量'信号 (强制日线数据源)...")
        
        # 定义需要检查的日线资金流列名
        mf_col = 'net_mf_amount_D'
        elg_buy_col = 'buy_elg_amount_D'
        elg_sell_col = 'sell_elg_amount_D'
        vol_ma_period = self._get_param_value(trigger_params.get('vol_ma_period'), 21)
        vol_ma_col = f"VOL_MA_{vol_ma_period}_D"
        volume_col = 'volume_D'
        
        required_force_cols = [mf_col, elg_buy_col, elg_sell_col, vol_ma_col, volume_col]
        
        # 严格检查所有需要的列是否存在于日线df中
        if not all(c in df.columns for c in required_force_cols):
            missing_cols = [c for c in required_force_cols if c not in df.columns]
            print(f"      -> [严重警告] 缺少计算'资金力量'所需的日线列: {missing_cols}。所有依赖此逻辑的触发器将不会被激活。")
            is_institutional_buying = pd.Series(False, index=df.index)
            is_hot_money_blitz = pd.Series(False, index=df.index)
        else:
            # 所有计算都明确使用带 '_D' 后缀的列
            is_institutional_buying = df[mf_col] > 0
            
            elg_net_buy = df[elg_buy_col] - df[elg_sell_col]
            net_mf_amount = df[mf_col]
            
            volume_spike_ratio = self._get_param_value(trigger_params.get('volume_spike_ratio'), 2.0)
            is_volume_spike = df[volume_col] > (df[vol_ma_col] * volume_spike_ratio)
            
            # “游资闪击”的定义
            is_hot_money_blitz = is_institutional_buying & (elg_net_buy > net_mf_amount * 0.7) & is_volume_spike

        # --- 步骤3: 融合“结构”与“力量”，生成高阶触发器 (逻辑不变) ---
        # --- 步骤3.1: 定义“主力点火”触发器 ---
        triggers['CHIP_INSTITUTIONAL_BREAKOUT'] = is_breakout_structure & is_institutional_buying & ~is_hot_money_blitz
        
        # --- 步骤3.2: “主力点火”专属诊断探针 ---
        # probe_start_date = pd.to_datetime('2024-07-01', utc=True)
        # probe_df = pd.DataFrame({
        #     'Struct_OK': is_breakout_structure,
        #     'MF_OK': is_institutional_buying,
        #     'Not_HotMoney': ~is_hot_money_blitz,
        #     '_Trigger': triggers['CHIP_INSTITUTIONAL_BREAKOUT']
        # }).loc[probe_start_date:]

        # interesting_days = probe_df[probe_df.any(axis=1)]
        # if not interesting_days.empty:
        #     print("\n--- [终极探针-TRIGGER | >24-07-01] 诊断 '主力点火' (CHIP_INSTITUTIONAL_BREAKOUT) ---")
        #     print(interesting_days.to_string())
        #     print("--- [终极探针] 诊断结束 ---\n")

        # “游资突破”：定义为由“游资闪击”行为主导的突破，通常更具爆发性，但波动也可能更大。
        triggers['CHIP_HOT_MONEY_BREAKOUT'] = is_breakout_structure & is_hot_money_blitz
        # “确认加速”：定义为股价越过关键筹码压力位（85%或95%成本线），并得到机构资金的确认，是趋势加速的强烈信号。
        triggers['CHIP_CONFIRMED_ACCELERATION'] = is_acceleration_structure & is_institutional_buying
        print(f"      -> 高阶触发器生成: 机构突破({triggers['CHIP_INSTITUTIONAL_BREAKOUT'].sum()}), 游资突破({triggers['CHIP_HOT_MONEY_BREAKOUT'].sum()}), 确认加速({triggers['CHIP_CONFIRMED_ACCELERATION'].sum()})")

        # --- 步骤4: 定义专用的、高规格的剧本触发器 ---
        # “能量释放”专用触发器
        try:
            p = trigger_params.get('energy_release_trigger_params', {})
            if self._get_param_value(p.get('enabled'), False):
                is_strong_candle = df['close_D'] > df['open_D']
                is_pct_change_valid = df['close_D'].pct_change() > self._get_param_value(p.get('breakout_pct_change'), 0.03)
                is_breakout_volume = df['volume_D'] > df[vol_ma_col] * self._get_param_value(p.get('breakout_volume_ratio'), 1.8)
                is_main_force_driving = df.get('net_mf_amount_D', pd.Series(0, index=df.index)) > 0
                is_ignition_action = is_strong_candle & is_pct_change_valid & is_breakout_volume & is_main_force_driving
                support_ma_col = f"EMA_{self._get_param_value(p.get('support_ma'), 21)}_D"
                has_trend_support = (df[support_ma_col] > df[support_ma_col].shift(1)) & (df['close_D'] > df[support_ma_col]) if support_ma_col in df.columns else pd.Series(True, index=df.index)
                dynamic_ceiling = df['high_D'].shift(1).rolling(window=self._get_param_value(p.get('release_window'), 5)).max()
                triggers['TRIGGER_ENERGY_RELEASE'] = is_ignition_action & has_trend_support & (df['close_D'] > dynamic_ceiling)
                print(f"      -> '能量释放专用' 触发器定义完成，发现 {triggers.get('TRIGGER_ENERGY_RELEASE', pd.Series([])).sum()} 天。")
        except Exception as e:
            print(f"      -> [警告] 计算'能量释放专用'触发器时出错: {e}")

        # “回踩反弹”专用触发器
        try:
            p = trigger_params.get('pullback_rebound_trigger_params', {})
            if self._get_param_value(p.get('enabled'), False):
                support_ma_period = self._get_param_value(p.get('support_ma'), 21)
                support_ma_col = f"EMA_{support_ma_period}_D"
                if support_ma_col in df.columns:
                    dipped_and_recovered = (df['low_D'] <= df[support_ma_col]) & (df['close_D'] > df[support_ma_col])
                    is_green_candle = df['close_D'] > df['open_D']
                    is_closing_high = df['close_D'] > (df['high_D'] + df['low_D']) / 2
                    has_lower_shadow = (df['open_D'] - df['low_D']) > 0.01
                    is_strong_reversal_candle = is_green_candle & is_closing_high & has_lower_shadow
                    is_main_force_buying = df.get('net_mf_amount_D', pd.Series(0, index=df.index)) > 0
                    # ▼▼▼【代码修改 V41.8】: 增加成交量确认条件，过滤无量弱反弹 ▼▼▼
                    min_rebound_volume_ratio = self._get_param_value(p.get('min_rebound_volume_ratio'), 0.8)
                    is_volume_confirmed = df['volume_D'] > (df.get(vol_ma_col, df['volume_D']) * min_rebound_volume_ratio)
                    triggers['TRIGGER_PULLBACK_REBOUND'] = dipped_and_recovered & is_strong_reversal_candle & is_main_force_buying & is_volume_confirmed
                    # ▲▲▲【代码修改 V41.8】▲▲▲
                    print(f"      -> '回踩反弹专用' 触发器定义完成 (V41.8 量能增强版)，发现 {triggers.get('TRIGGER_PULLBACK_REBOUND', pd.Series([])).sum()} 天。")
        except Exception as e:
            print(f"      -> [警告] 计算'回踩反弹专用'触发器时出错: {e}")

        # “V型反转”专用触发器
        try:
            p = self._get_params_block(params, 'v_recovery_params', {})
            if self._get_param_value(p.get('enabled'), True) and 'net_mf_amount_D' in df.columns:
                is_strong_rally = (df['close_D'] / df['close_D'].shift(1) - 1) > self._get_param_value(p.get('reversal_rally_pct'), 0.05)
                is_strong_candle = (df['close_D'] - df['open_D']) > (df['high_D'] - df['low_D']) * 0.6
                avg_amount_in_drop = df['amount_D'].shift(1).rolling(window=self._get_param_value(p.get('drop_days'), 10)).mean()
                is_huge_mf_inflow = df['net_mf_amount_D'].fillna(0) > (avg_amount_in_drop * 0.1)
                triggers['TRIGGER_V_RECOVERY'] = is_strong_rally & is_strong_candle & is_huge_mf_inflow
                print(f"      -> 'V型反转修复' 触发器定义完成，发现 {triggers.get('TRIGGER_V_RECOVERY', pd.Series([])).sum()} 天。")
        except Exception as e:
            print(f"      -> [警告] 计算'V型反转修复'触发器时出错: {e}")
            
        # “巨阴洗盘反包”专用触发器
        try:
            p = self._get_params_block(params, 'washout_reversal_params', {})
            if self._get_param_value(p.get('enabled'), True):
                is_reversal_cover = df['close_D'] > df['close_D'].shift(1)
                reversal_rally = (df['close_D'] - df['open_D']) / df['open_D'].replace(0, np.nan)
                is_strong_reversal_candle = reversal_rally > self._get_param_value(p.get('reversal_rally_threshold'), 0.03)
                triggers['TRIGGER_WASHOUT_REVERSAL'] = is_reversal_cover & is_strong_reversal_candle
                print(f"      -> '巨阴洗盘反包' 触发器定义完成，发现 {triggers.get('TRIGGER_WASHOUT_REVERSAL', pd.Series([])).sum()} 天。")
        except Exception as e:
            print(f"      -> [警告] 计算'巨阴洗盘反包'触发器时出错: {e}")

        # --- 新增: 十字星突破 (Doji Breakout) ---
        try:
            p = trigger_params.get('doji_breakout_params', {})
            if self._get_param_value(p.get('enabled'), True):
                confirmation_vol_ratio = self._get_param_value(p.get('confirmation_volume_ratio'), 1.2)
                # 条件1: 价格确认 (收盘突破昨日高点)
                is_price_confirmation = df['close_D'] > df['high_D'].shift(1)
                # 条件2: 成交量确认 (今日成交量 > 昨日成交量 * 倍数)
                is_volume_confirmation = df['volume_D'] > (df['volume_D'].shift(1) * confirmation_vol_ratio)
                triggers['TRIGGER_DOJI_BREAKOUT'] = is_price_confirmation & is_volume_confirmation
                print(f"      -> '十字星突破' 触发器定义完成，发现 {triggers.get('TRIGGER_DOJI_BREAKOUT', pd.Series([])).sum()} 天。")
        except Exception as e:
            print(f"      -> [警告] 计算'十字星突破'触发器时出错: {e}")

        # --- 新增: 地天板反转 (Earth-Heaven Board) ---
        try:
            # 复用已有的板形态识别模块
            board_patterns = self._identify_board_patterns(df, params)
            if board_patterns:
                 triggers['TRIGGER_EARTH_HEAVEN_BOARD'] = board_patterns.get('earth_heaven_board', pd.Series(False, index=df.index))
                 print(f"      -> '地天板反转' 触发器定义完成，发现 {triggers.get('TRIGGER_EARTH_HEAVEN_BOARD', pd.Series([])).sum()} 天。")
        except Exception as e:
            print(f"      -> [警告] 计算'地天板反转'触发器时出错: {e}")

        # “堡垒防卫”专用触发器
        try:
            p = self._get_params_block(params, 'fortress_defense_params', {})
            if self._get_param_value(p.get('enabled'), True):
                is_green_candle = df['close_D'] > df['open_D']
                is_closing_high = df['close_D'] > (df['high_D'] + df['low_D']) / 2
                is_strong_reversal_candle = is_green_candle & is_closing_high
                is_main_force_buying = df.get('net_mf_amount_D', pd.Series(0, index=df.index)) > 0
                triggers['TRIGGER_FORTRESS_DEFENSE'] = is_strong_reversal_candle & is_main_force_buying
                print(f"      -> '堡垒防卫' 触发器定义完成，发现 {triggers.get('TRIGGER_FORTRESS_DEFENSE', pd.Series([])).sum()} 天。")
        except Exception as e:
            print(f"      -> [警告] 计算'堡垒防卫'触发器时出错: {e}")

        # --- 步骤5: 定义通用的、经典的触发器 ---
        # 动量突破 (首板)
        try:
            p = trigger_params.get('momentum_breakout_params', {})
            if self._get_param_value(p.get('enabled'), False):
                price_increase_ratio = (df['close_D'] - df['open_D']) / df['open_D'].replace(0, np.nan)
                is_strong_candle = price_increase_ratio > self._get_param_value(p.get('rally_threshold'), 0.05)
                vol_ma_col_mom = f"VOL_MA_{self._get_param_value(p.get('vol_ma_period'), 21)}_D"
                is_volume_surge = df['volume_D'] > df[vol_ma_col_mom] * self._get_param_value(p.get('volume_ratio'), 2.0)
                triggers['TRIGGER_MOMENTUM_BREAKOUT'] = is_strong_candle & is_volume_surge
                print(f"      -> '动量突破(首板)' 触发器定义完成，发现 {triggers.get('TRIGGER_MOMENTUM_BREAKOUT', pd.Series([])).sum()} 天。")
        except Exception as e:
            print(f"      -> [警告] 计算'动量突破'时出错: {e}")

        # 通用放量突破 (突破近期高点)
        try:
            p = trigger_params.get('volume_spike_breakout', {})
            if self._get_param_value(p.get('enabled'), True):
                vol_ma_period_gen = self._get_param_value(p.get('vol_ma_period'), 21)
                vol_ma_col_gen = f"VOL_MA_{vol_ma_period_gen}_D"
                if vol_ma_col_gen in df.columns:
                    volume_ratio_gen = self._get_param_value(p.get('volume_ratio'), 2.0)
                    lookback_gen = self._get_param_value(p.get('lookback_period'), 10)
                    is_volume_spike_gen = df['volume_D'] > df[vol_ma_col_gen] * volume_ratio_gen
                    is_price_breakout_gen = df['close_D'] > df['high_D'].shift(1).rolling(lookback_gen).max()
                    triggers['TRIGGER_VOLUME_SPIKE_BREAKOUT'] = is_volume_spike_gen & is_price_breakout_gen
                    print(f"      -> '通用放量突破' 触发器定义完成，发现 {triggers.get('TRIGGER_VOLUME_SPIKE_BREAKOUT', pd.Series([])).sum()} 天。")
        except Exception as e:
            print(f"      -> [警告] 计算'通用放量突破'触发器时出错: {e}")

        # 布林带突破 (中轨)
        try:
            p = trigger_params.get('bband_breakout_params', {})
            if self._get_param_value(p.get('enabled'), False):
                bbm_col = 'BBM_21_2.0_D'
                if bbm_col in df.columns:
                    triggers['TRIGGER_BBAND_BREAKOUT'] = (df['close_D'] > df[bbm_col]) & (df['close_D'].shift(1) <= df[bbm_col].shift(1))
                    print(f"      -> '布林带突破' 触发器定义完成，发现 {triggers.get('TRIGGER_BBAND_BREAKOUT', pd.Series([])).sum()} 天。")
        except Exception as e:
            print(f"      -> [警告] 计算'布林带突破'时出错: {e}")

        # 指标交叉
        try:
            p = trigger_params.get('indicator_cross_params', {})
            if self._get_param_value(p.get('enabled'), False):
                if self._get_param_value(p.get('dmi_cross', {}).get('enabled'), False):
                    pdi_col, mdi_col = 'PDI_14_D', 'NDI_14_D'
                    if all(c in df.columns for c in [pdi_col, mdi_col]):
                        triggers['TRIGGER_DMI_CROSS'] = (df[pdi_col] > df[mdi_col]) & (df[pdi_col].shift(1) <= df[mdi_col].shift(1))
                        print(f"      -> 'DMI金叉' 触发器定义完成，发现 {triggers.get('TRIGGER_DMI_CROSS', pd.Series([])).sum()} 天。")
                macd_p = p.get('macd_cross', {})
                if self._get_param_value(macd_p.get('enabled'), False):
                    macd_col, signal_col = 'MACD_13_34_8_D', 'MACDs_13_34_8_D'
                    if all(c in df.columns for c in [macd_col, signal_col]):
                        is_golden_cross = (df[macd_col] > df[signal_col]) & (df[macd_col].shift(1) <= df[signal_col].shift(1))
                        low_level = self._get_param_value(macd_p.get('low_level'), -0.5)
                        high_level = self._get_param_value(macd_p.get('high_level'), 0.5)
                        triggers['TRIGGER_MACD_LOW_CROSS'] = is_golden_cross & (df[macd_col] < low_level)
                        triggers['TRIGGER_MACD_ZERO_CROSS'] = is_golden_cross & (df[macd_col].between(low_level, high_level))
                        print(f"      -> 'MACD金叉' 触发器定义完成，发现低位 {triggers.get('TRIGGER_MACD_LOW_CROSS', pd.Series([])).sum()} 天, 零轴 {triggers.get('TRIGGER_MACD_ZERO_CROSS', pd.Series([])).sum()} 天。")
        except Exception as e:
            print(f"      -> [警告] 计算'指标交叉'时出错: {e}")
        # --- “反转确认扳机” ---
        try:
            p = trigger_params.get('reversal_confirmation_candle', {})
            if self._get_param_value(p.get('enabled'), True):
                is_green = df['close_D'] > df['open_D']
                # 条件2: 涨幅足够大
                min_pct_change = self._get_param_value(p.get('min_pct_change'), 0.03) # 至少上涨3%
                is_strong_rally = df['pct_change_D'] > min_pct_change
                # 条件3: 收盘价在当天振幅的一半以上，代表收盘强势
                is_closing_strong = df['close_D'] > (df['high_D'] + df['low_D']) / 2
                
                triggers['TRIGGER_REVERSAL_CONFIRMATION_CANDLE'] = is_green & is_strong_rally & is_closing_strong
                print(f"      -> '反转确认阳线' 专属触发器定义完成，发现 {triggers.get('TRIGGER_REVERSAL_CONFIRMATION_CANDLE', pd.Series([])).sum()} 天。")
        except Exception as e:
            print(f"      -> [警告] 计算'反转确认阳线'触发器时出错: {e}")

        # 强势阳线 (通用)
        try:
            p = trigger_params.get('positive_candle', {})
            if self._get_param_value(p.get('enabled'), True):
                is_green = df['close_D'] > df['open_D']
                body_range = (df['high_D'] - df['low_D']).replace(0, np.nan)
                body_ratio = (df['close_D'] - df['open_D']) / body_range
                is_strong_body = body_ratio > self._get_param_value(p.get('min_body_ratio'), 0.6)
                triggers['TRIGGER_STRONG_POSITIVE_CANDLE'] = is_green & is_strong_body
                print(f"      -> '强势阳线' 触发器定义完成，发现 {triggers.get('TRIGGER_STRONG_POSITIVE_CANDLE', pd.Series([])).sum()} 天。")
        except Exception as e:
            print(f"      -> [警告] 计算'强势阳线'触发器时出错: {e}")

        # 收复关键均线 (通用)
        try:
            p = trigger_params.get('ma_reclaim', {})
            if self._get_param_value(p.get('enabled'), True):
                ma_col = f"EMA_{self._get_param_value(p.get('ma_period'), 55)}_D"
                if ma_col in df.columns:
                    triggers['TRIGGER_MA_RECLAIM'] = (df['close_D'] > df[ma_col]) & (df['close_D'].shift(1) <= df[ma_col].shift(1))
                    print(f"      -> '收复关键均线' 触发器定义完成，发现 {triggers.get('TRIGGER_MA_RECLAIM', pd.Series([])).sum()} 天。")
        except Exception as e:
            print(f"      -> [警告] 计算'收复均线'触发器时出错: {e}")

        print("      -> [V41.9] 开始集成高级剧本触发器...")

        # --- 剧本触发器 1: BIAS超跌反弹 ---
        try:
            p = playbook_specific_params.get('bias_reversal_params', {})
            if self._get_param_value(p.get('enabled'), False):
                bias_period = self._get_param_value(p.get('bias_period'), 20)
                bias_col = f"BIAS_{bias_period}_D"
                if bias_col in df.columns:
                    triggers['TRIGGER_BIAS_REBOUND'] = df[bias_col] > df[bias_col].shift(1)
                    print(f"      -> 'BIAS超跌反弹' 触发器定义完成，发现 {triggers.get('TRIGGER_BIAS_REBOUND', pd.Series([])).sum()} 天。")
        except Exception as e:
            print(f"      -> [警告] 计算'BIAS超跌反弹'触发器时出错: {e}")

        # --- 剧本触发器 2: 获利盘洗净后反转K线 ---
        try:
            p = playbook_specific_params.get('winner_rate_reversal_params', {})
            if self._get_param_value(p.get('enabled'), False):
                triggers['TRIGGER_WINNER_RATE_REVERSAL'] = df['close_D'] > df['open_D']
                print(f"      -> '获利盘洗净反转' 触发器定义完成，发现 {triggers.get('TRIGGER_WINNER_RATE_REVERSAL', pd.Series([])).sum()} 天。")
        except Exception as e:
            print(f"      -> [警告] 计算'获利盘洗净反转'触发器时出错: {e}")

        # --- 剧本触发器 3: N字板接力 (完整逻辑封装) ---
        try:
            p = playbook_specific_params.get('n_shape_relay_params', {})
            if self._get_param_value(p.get('enabled'), False):
                rally_threshold = self._get_param_value(p.get('rally_threshold'), 0.097)
                consolidation_days_max = self._get_param_value(p.get('consolidation_days_max'), 3)
                hold_above_key = self._get_param_value(p.get('hold_above_key'), 'open_D')
                body_rally = (df['close_D'] - df['open_D']) / df['open_D'].replace(0, np.nan)
                is_strong_body = body_rally > self._get_param_value(p.get('body_rally_threshold'), 0.05)
                is_limit_up_rally = df['close_D'].pct_change() > rally_threshold
                is_rally_day = is_strong_body | is_limit_up_rally
                final_signal = pd.Series(False, index=df.index)
                for days in range(1, consolidation_days_max + 1):
                    is_second_rally = is_rally_day
                    is_first_rally = is_rally_day.shift(days + 1)
                    first_rally_high = df['high_D'].shift(days + 1)
                    first_rally_support = df[hold_above_key].shift(days + 1)
                    first_rally_volume = df['volume_D'].shift(days + 1)
                    consolidation_lows = df['low_D'].shift(1).rolling(window=days).min()
                    consolidation_volumes = df['volume_D'].shift(1).rolling(window=days).max()
                    is_supported = consolidation_lows > first_rally_support
                    is_volume_shrunk = consolidation_volumes < first_rally_volume
                    is_breakout = df['close_D'] > first_rally_high
                    current_signal = is_first_rally & is_supported & is_volume_shrunk & is_second_rally & is_breakout
                    final_signal |= current_signal.fillna(False)
                triggers['TRIGGER_N_SHAPE_RELAY'] = final_signal
                print(f"      -> 'N字板接力' 触发器定义完成，发现 {triggers.get('TRIGGER_N_SHAPE_RELAY', pd.Series([])).sum()} 天。")
        except Exception as e:
            print(f"      -> [警告] 计算'N字板接力'触发器时出错: {e}")

        # --- 剧本触发器 4: 斐波那契引力区 (完整逻辑封装) ---
        try:
            p = playbook_specific_params.get('fibonacci_pullback_params', {})
            if self._get_param_value(p.get('enabled'), False):
                peak_distance = self._get_param_value(p.get('peak_distance'), 10)
                peak_prominence = self._get_param_value(p.get('peak_prominence'), 0.05)
                impulse_min_rise_pct = self._get_param_value(p.get('impulse_min_rise_pct'), 0.25)
                impulse_min_duration = self._get_param_value(p.get('impulse_min_duration'), 10)
                impulse_min_adx = self._get_param_value(p.get('impulse_min_adx'), 20)
                fib_levels = self._get_param_value(p.get('retracement_levels'), [0.382, 0.5, 0.618])
                pullback_buffer = self._get_param_value(p.get('pullback_buffer'), 0.015)
                adx_col, vol_ma_col, mf_col = 'ADX_14_D', 'VOL_MA_21_D', 'net_mf_amount_D'
                required_cols = ['high_D', 'low_D', 'close_D', 'open_D', 'volume_D', adx_col, vol_ma_col, mf_col]
                if all(c in df.columns for c in required_cols):
                    price_range = df['high_D'].max() - df['low_D'].min()
                    final_signal = pd.Series(False, index=df.index)
                    if price_range > 0:
                        absolute_prominence = price_range * peak_prominence
                        swing_high_indices, _ = find_peaks(df['high_D'], distance=peak_distance, prominence=absolute_prominence)
                        swing_low_indices, _ = find_peaks(-df['low_D'], distance=peak_distance, prominence=absolute_prominence)
                        if len(swing_high_indices) > 0 and len(swing_low_indices) > 0:
                            for high_idx in swing_high_indices:
                                possible_lows = swing_low_indices[swing_low_indices < high_idx]
                                if len(possible_lows) == 0: continue
                                low_idx = possible_lows[-1]
                                impulse_wave_df = df.iloc[low_idx:high_idx+1]
                                if not (len(impulse_wave_df) >= impulse_min_duration and ((impulse_wave_df['high_D'].max() / impulse_wave_df['low_D'].min()) - 1) >= impulse_min_rise_pct and impulse_wave_df[adx_col].mean() >= impulse_min_adx):
                                    continue
                                swing_high_price, swing_low_price = df['high_D'].iloc[high_idx], df['low_D'].iloc[low_idx]
                                fib_support_levels = {lvl: swing_high_price - (swing_high_price - swing_low_price) * lvl for lvl in fib_levels}
                                search_period_df = df.iloc[high_idx + 1 : high_idx + 1 + 60]
                                for date, row in search_period_df.iterrows():
                                    for level, support_price in fib_support_levels.items():
                                        if row['low_D'] <= support_price * (1 + pullback_buffer) and row['low_D'] >= support_price * (1 - pullback_buffer):
                                            if row['volume_D'] < row[vol_ma_col] and row['close_D'] > row['open_D'] and row['close_D'] > support_price and row[mf_col] > 0:
                                                final_signal.loc[date] = True
                                                break
                                    if final_signal.loc[date]: break
                    triggers['TRIGGER_FIBONACCI_REBOUND'] = final_signal
                    print(f"      -> '斐波那契引力区' 触发器定义完成，发现 {triggers.get('TRIGGER_FIBONACCI_REBOUND', pd.Series([])).sum()} 天。")
        except Exception as e:
            print(f"      -> [警告] 计算'斐波那契引力区'触发器时出错: {e}")

        # --- 剧本触发器 5: 趋势引爆 (完整逻辑封装) ---
        try:
            p = playbook_specific_params.get('ma_acceleration_params', {})
            if self._get_param_value(p.get('enabled'), False):
                ma_period = self._get_param_value(p.get('ma_period'), 21)
                volume_surge_ratio = self._get_param_value(p.get('volume_surge_ratio'), 1.5)
                min_adx_level = self._get_param_value(p.get('min_adx_level'), 20)
                ema_col, vol_ma_col, adx_col, mf_col = f'EMA_{ma_period}_D', f'VOL_MA_21_D', f'ADX_14_D', f'net_mf_amount_D'
                slope_col, accel_col = f'{ema_col}_slope_for_accel', f'{ema_col}_acceleration'
                required_cols = [ema_col, vol_ma_col, adx_col, mf_col, 'volume_D']
                if all(c in df.columns for c in required_cols):
                    df[slope_col] = df[ema_col].diff(1)
                    df[accel_col] = df[slope_col].diff(1)
                    is_ma_accelerating = (df[slope_col] > 0) & (df[accel_col] > 0)
                    is_volume_surged = df['volume_D'] > (df[vol_ma_col] * volume_surge_ratio)
                    is_main_force_driving = df[mf_col] > 0
                    is_in_trend = df[adx_col] > min_adx_level
                    triggers['TRIGGER_MA_ACCELERATION'] = is_ma_accelerating & is_volume_surged & is_main_force_driving & is_in_trend
                    df.drop(columns=[slope_col, accel_col], inplace=True, errors='ignore')
                    print(f"      -> '趋势引爆' 触发器定义完成，发现 {triggers.get('TRIGGER_MA_ACCELERATION', pd.Series([])).sum()} 天。")
        except Exception as e:
            print(f"      -> [警告] 计算'趋势引爆'触发器时出错: {e}")

        # --- 步骤6: 最终清洗 ---
        for key in list(triggers.keys()):
            if triggers[key] is None:
                triggers[key] = pd.Series(False, index=df.index)
            else:
                triggers[key] = triggers[key].fillna(False)

        print("    - [触发事件中心 V41.8] 所有原子化触发事件定义完成。")
        return triggers

    # ▼▼▼ “原子筹码信号提供器” (Atomic Chip Signal Provider) ▼▼▼
    def _define_chip_atomic_signals(self, df: pd.DataFrame, params: dict) -> Dict[str, pd.Series]:
        """
        【V45.5 时间框架感知缓存版】
        - 核心功能: 融合所有分散的筹码剧本逻辑，成为计算底层筹码信号的唯一“真理之源”。
        - 根本性修复: 引入按时间框架区分的缓存机制 (self._chip_atomic_signals_cache_by_tf)，
                      彻底解决因缓存污染导致日线计算返回周线结果的根本问题。
        - 增强诊断: 在函数入口增加“数据完整性探针”，验证传入df的索引是否为日线。
        """
        # --- 步骤 0: 数据完整性探针 (治本的第一步：验证输入) ---
        print("\n--- [探针-CHIP_INPUT | V45.5] 正在检查 _define_chip_atomic_signals 的输入DataFrame ---")
        if df.empty:
            print("  -> 输入的DataFrame为空，无法进行诊断和计算。")
            return {}
        try:
            # 尝试推断索引频率
            # freq = pd.infer_freq(df.index)
            # print(f"  -> 输入 df 的推断索引频率: {freq}")
            # print(f"  -> 输入 df 的总行数: {len(df)}")
            # print(f"  -> 输入 df 的起始日期: {df.index[0].date()}")
            # print(f"  -> 输入 df 的结束日期: {df.index[-1].date()}")
            # 从列名中推断时间框架，作为缓存的key
            tf_key = 'D' # 默认为日线
            for col in df.columns:
                if col.endswith('_W'):
                    tf_key = 'W'
                    break
                elif col.endswith('_M'):
                    tf_key = 'M'
                    break
            # print(f"  -> 从列名推断的时间框架(缓存Key): '{tf_key}'")
        except Exception as e:
            # print(f"  -> [警告] 诊断输入df时发生错误: {e}")
            tf_key = 'D' # 出错时默认
        # print("--- [探针-CHIP_INPUT] 检查结束 ---\n")

        # --- 步骤 1: 检查“时间框架感知”的缓存 ---
        if hasattr(self, '_chip_atomic_signals_cache_by_tf') and tf_key in self._chip_atomic_signals_cache_by_tf:
            print(f"      -> [筹码原子信号中心 V45.5] 从缓存加载 '{tf_key}' 框架的信号。")
            return self._chip_atomic_signals_cache_by_tf[tf_key]
        
        print(f"      -> [筹码原子信号中心 V45.5] 启动，为 '{tf_key}' 框架计算所有底层筹码信号...")
        signals = {}
        chip_params = self._get_params_block(params, 'chip_atomic_signal_params', {})
        if not self._get_param_value(chip_params.get('enabled'), False):
            print("        -> [警告] 原子化筹码信号计算被禁用，返回空信号。")
            if hasattr(self, '_chip_atomic_signals_cache_by_tf'):
                self._chip_atomic_signals_cache_by_tf[tf_key] = {} # 缓存空结果
            return {}

        # --- 步骤 2: 信号计算 (逻辑保持不变，但现在是安全的) ---
        # ... (此处的计算逻辑与您之前的版本完全相同，无需修改) ...
        # --- 1. 成本区增强 (ATOMIC_REINFORCEMENT) ---
        try:
            p = chip_params.get('reinforcement_params', {})
            if self._get_param_value(p.get('enabled'), False):
                required_cols = ['weight_avg_D', 'close_D', 'volume_D', 'VOL_MA_21_D', 'net_mf_amount_D']
                if all(c in df.columns for c in required_cols):
                    proximity_threshold = self._get_param_value(p.get('proximity_threshold'), 0.03)
                    volume_ratio = self._get_param_value(p.get('volume_ratio'), 1.5)
                    sideways_threshold = self._get_param_value(p.get('sideways_threshold'), 0.015)
                    is_price_near_cost = abs(df['close_D'] - df['weight_avg_D']) / df['weight_avg_D'].replace(0, np.nan) <= proximity_threshold
                    is_volume_surged = df['volume_D'] > (df['VOL_MA_21_D'] * volume_ratio)
                    is_sideways = df['close_D'].pct_change().abs() < sideways_threshold
                    is_main_force_inflow = df['net_mf_amount_D'] > 0
                    signals['ATOMIC_REINFORCEMENT'] = is_price_near_cost & is_volume_surged & is_sideways & is_main_force_inflow
                    print(f"        -> '成本区增强'信号计算完成，发现 {signals.get('ATOMIC_REINFORCEMENT', pd.Series([])).sum()} 天。")
        except Exception as e:
            print(f"        -> [警告] 计算'成本区增强'信号时出错: {e}")

        # --- 2. 筹码高度集中状态 (ATOMIC_CONCENTRATED_STATE) & 集中后突破 (ATOMIC_CONCENTRATION_BREAKOUT) ---
        try:
            p = chip_params.get('concentration_params', {})
            if self._get_param_value(p.get('enabled'), False):
                percentiles = self._get_param_value(p.get('concentration_percentiles'), [15, 85])
                lower_pct, upper_pct = min(percentiles), max(percentiles)
                cost_lower_col, cost_upper_col = f'cost_{lower_pct}pct_D', f'cost_{upper_pct}pct_D'
                required_cols = [cost_lower_col, cost_upper_col, 'weight_avg_D', 'close_D']
                if all(c in df.columns for c in required_cols):
                    concentration_ratio = (df[cost_upper_col] - df[cost_lower_col]) / df['weight_avg_D'].replace(0, np.nan)
                    signals['ATOMIC_CONCENTRATED_STATE'] = concentration_ratio < self._get_param_value(p.get('concentration_threshold'), 0.10)
                    was_below_upper_cost = df['close_D'].shift(1) <= df[cost_upper_col].shift(1)
                    is_breakout = df['close_D'] > df[cost_upper_col]
                    signals['ATOMIC_CONCENTRATION_BREAKOUT'] = is_breakout & was_below_upper_cost
                    print(f"        -> '筹码集中状态'计算完成，发现 {signals.get('ATOMIC_CONCENTRATED_STATE', pd.Series([])).sum()} 天。")
                    print(f"        -> '筹码集中后突破'信号计算完成，发现 {signals.get('ATOMIC_CONCENTRATION_BREAKOUT', pd.Series([])).sum()} 天。")
        except Exception as e:
            print(f"        -> [警告] 计算'筹码集中'相关信号时出错: {e}")

        # --- 3. 上穿市场平均成本 (ATOMIC_COST_BREAKTHROUGH) ---
        try:
            p = chip_params.get('cost_breakthrough_params', {})
            if self._get_param_value(p.get('enabled'), False) and all(c in df.columns for c in ['close_D', 'weight_avg_D']):
                signals['ATOMIC_COST_BREAKTHROUGH'] = (df['close_D'] > df['weight_avg_D']) & (df['close_D'].shift(1) <= df['weight_avg_D'].shift(1))
                print(f"        -> '上穿平均成本'信号计算完成，发现 {signals.get('ATOMIC_COST_BREAKTHROUGH', pd.Series([])).sum()} 天。")
        except Exception as e:
            print(f"        -> [警告] 计算'上穿平均成本'信号时出错: {e}")

        # --- 4. 突破95%套牢盘 (ATOMIC_PRESSURE_RELEASE) ---
        try:
            p = chip_params.get('pressure_release_params', {})
            if self._get_param_value(p.get('enabled'), False) and 'cost_95pct_D' in df.columns:
                signals['ATOMIC_PRESSURE_RELEASE'] = (df['close_D'] > df['cost_95pct_D']) & (df['close_D'].shift(1) <= df['cost_95pct_D'].shift(1))
                print(f"        -> '突破95%套牢盘'信号计算完成，发现 {signals.get('ATOMIC_PRESSURE_RELEASE', pd.Series([])).sum()} 天。")
        except Exception as e:
            print(f"        -> [警告] 计算'突破95%套牢盘'信号时出错: {e}")

        # --- 5. 突破85%套牢盘 (ATOMIC_HURDLE_CLEAR) ---
        try:
            p = chip_params.get('hurdle_clear_params', {})
            if self._get_param_value(p.get('enabled'), False) and 'cost_85pct_D' in df.columns:
                signals['ATOMIC_HURDLE_CLEAR'] = (df['close_D'] > df['cost_85pct_D']) & (df['close_D'].shift(1) <= df['cost_85pct_D'].shift(1))
                print(f"        -> '突破85%套牢盘'信号计算完成，发现 {signals.get('ATOMIC_HURDLE_CLEAR', pd.Series([])).sum()} 天。")
        except Exception as e:
            print(f"        -> [警告] 计算'突破85%套牢盘'信号时出错: {e}")

        # --- 步骤 3: 填充、清洗并存入“时间框架感知”的缓存 ---
        all_atomic_signals = [
            'ATOMIC_REINFORCEMENT', 'ATOMIC_CONCENTRATED_STATE', 'ATOMIC_CONCENTRATION_BREAKOUT',
            'ATOMIC_COST_BREAKTHROUGH', 'ATOMIC_PRESSURE_RELEASE', 'ATOMIC_HURDLE_CLEAR'
        ]
        for sig_name in all_atomic_signals:
            if sig_name not in signals:
                signals[sig_name] = pd.Series(False, index=df.index)
            else:
                signals[sig_name].fillna(False, inplace=True)
        
        if hasattr(self, '_chip_atomic_signals_cache_by_tf'):
            self._chip_atomic_signals_cache_by_tf[tf_key] = signals # 按推断出的时间框架key缓存结果
        
        print(f"      -> [筹码原子信号中心 V45.5] 为 '{tf_key}' 框架计算完成并已缓存。")
        return signals

    # ▼▼▼ 风险状态定义中心 ▼▼▼
    def _calculate_risk_states(self, df: pd.DataFrame, params: dict) -> Dict[str, pd.Series]:
        """
        【V41.12 新增】风险状态定义中心 (Risk State Center)
        - 核心功能: 集中计算所有用于“出场剧本”的风险状态。
        """
        print("    - [风险状态中心 V41.12] 启动，开始计算所有风险状态...")
        risks = {}
        exit_params = self._get_params_block(params, 'exit_strategy_params', {})

        # --- 风险状态: 主力叛逃 (高位派发) ---
        try:
            # 此处直接调用您提供的函数，因为它已经是一个完整的剧本逻辑
            risks['RISK_UPTHRUST_DISTRIBUTION'] = self._find_upthrust_distribution_exit(df, params)
            print(f"      -> '主力叛逃'风险状态定义完成，发现 {risks.get('RISK_UPTHRUST_DISTRIBUTION', pd.Series([])).sum()} 天。")
        except Exception as e:
            print(f"      -> [警告] 计算'主力叛逃'风险状态时出错: {e}")

        # --- 风险状态: 结构崩溃 (放量破位) ---
        try:
            # 此处直接调用您提供的函数
            risks['RISK_STRUCTURE_BREAKDOWN'] = self._find_volume_breakdown_exit(df, params)
            print(f"      -> '结构崩溃'风险状态定义完成，发现 {risks.get('RISK_STRUCTURE_BREAKDOWN', pd.Series([])).sum()} 天。")
        except Exception as e:
            print(f"      -> [警告] 计算'结构崩溃'风险状态时出错: {e}")
        
        # --- 风险状态1: 顶背离 (Top Divergence) ---
        try:
            p = exit_params.get('divergence_exit_params', {})
            if self._get_param_value(p.get('enabled'), False):
                # 复用您提供的 _find_top_divergence_exit 函数逻辑
                risks['RISK_TOP_DIVERGENCE'] = self._find_top_divergence_exit(df, params)
                print(f"      -> '顶背离'风险状态定义完成，发现 {risks.get('RISK_TOP_DIVERGENCE', pd.Series([])).sum()} 天。")
        except Exception as e:
            print(f"      -> [警告] 计算'顶背离'风险状态时出错: {e}")

        # --- 风险状态2: 指标超买 (Indicator Overbought) ---
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

        # --- 清洗与填充 ---
        all_needed_risks = [
            'RISK_UPTHRUST_DISTRIBUTION', 'RISK_STRUCTURE_BREAKDOWN', # 新增
            'RISK_TOP_DIVERGENCE', 'RISK_RSI_OVERBOUGHT', 'RISK_BIAS_OVERBOUGHT'
        ]
        for risk_name in all_needed_risks:
            if risk_name not in risks:
                risks[risk_name] = pd.Series(False, index=df.index)

        print("    - [风险状态中心 V41.13] 所有风险状态计算完成。")
        return risks

    def _get_params_block(self, params: dict, block_name: str, default_return: Any = None) -> dict:
        # 修改行: 增加一个默认返回值，使调用更安全
        if default_return is None:
            default_return = {}
        return params.get('strategy_params', {}).get('trend_follow', {}).get(block_name, default_return)

    def _analyze_dynamic_box_and_ma_trend(self, df: pd.DataFrame, params: dict):
        """
        【V40.0 健壮参数版】
        - 核心升级: 使用 `_get_param_value` 辅助函数来解析均线周期参数。
        - 效果: 适配新配置格式，确保均线趋势判断逻辑的健壮性。
        """
        box_params = self._get_params_block(params, 'dynamic_box_params')

        ma_params = self._get_params_block(params, 'mid_term_trend_params')
        mid_ma = self._get_param_value(ma_params.get('mid_ma'), 55)
        slow_ma = self._get_param_value(ma_params.get('slow_ma'), 89)

        mid_ma_col = f"EMA_{mid_ma}_D"
        slow_ma_col = f"EMA_{slow_ma}_D"

        if df.empty or not box_params.get('enabled', False):
            if self.verbose_logging:
                print("调试信息: 动态箱体被禁用或DataFrame为空，执行经典中期趋势判断。")
            if mid_ma_col in df.columns and slow_ma_col in df.columns:
                is_classic_uptrend = (df[mid_ma_col] > df[slow_ma_col]) & (df['close_D'] > df[mid_ma_col])
            else:
                is_classic_uptrend = pd.Series(False, index=df.index)
            df['context_mid_term_bullish'] = is_classic_uptrend
            self.signals['dynamic_box_breakout'] = pd.Series(False, index=df.index)
            self.signals['dynamic_box_breakdown'] = pd.Series(False, index=df.index)
            return

        # 1. 计算动态的 prominence 序列 (Series)，而不是标量
        prominence_base = df['close_D']
        peak_prominence_series = prominence_base * box_params.get('peak_prominence', 0.02)
        trough_prominence_series = prominence_base * box_params.get('trough_prominence', 0.02)

         # --- 步骤 2: 【核心修改】手动应用动态 Prominence ---
        # 2.1: 找出所有候选波峰，不使用 prominence 参数
        candidate_peak_indices, _ = find_peaks(df['close_D'], distance=box_params.get('peak_distance', 10))
        
        final_peak_indices = []
        if len(candidate_peak_indices) > 0:
            # 2.2: 计算这些候选波峰的实际 prominence
            actual_prominences, _, _ = peak_prominences(df['close_D'], candidate_peak_indices)
            # 2.3: 从我们的动态阈值序列中，提取出这些候选波峰对应的阈值
            custom_thresholds = peak_prominence_series.iloc[candidate_peak_indices]
            # 2.4: 手动比较，筛选出有效的波峰
            valid_peaks_mask = actual_prominences >= custom_thresholds.values
            final_peak_indices = candidate_peak_indices[valid_peaks_mask]

        # 对波谷执行同样的操作
        candidate_trough_indices, _ = find_peaks(-df['close_D'], distance=box_params.get('peak_distance', 10))
        
        final_trough_indices = []
        if len(candidate_trough_indices) > 0:
            actual_prominences, _, _ = peak_prominences(-df['close_D'], candidate_trough_indices)
            custom_thresholds = trough_prominence_series.iloc[candidate_trough_indices]
            valid_troughs_mask = actual_prominences >= custom_thresholds.values
            final_trough_indices = candidate_trough_indices[valid_troughs_mask]

        # --- 步骤 3: 后续逻辑使用最终筛选出的索引 (final_peak_indices, final_trough_indices) ---
        df['last_peak_price'] = np.nan
        if len(final_peak_indices) > 0:
            # 使用 .iloc 和 .columns.get_loc 是最稳健的赋值方式
            df.iloc[final_peak_indices, df.columns.get_loc('last_peak_price')] = df['close_D'].iloc[final_peak_indices]
        df['last_peak_price'].ffill(inplace=True)

        df['last_trough_price'] = np.nan
        if len(final_trough_indices) > 0:
            df.iloc[final_trough_indices, df.columns.get_loc('last_trough_price')] = df['close_D'].iloc[final_trough_indices]
        df['last_trough_price'].ffill(inplace=True)

        # 4. 定义每一天的动态箱体
        box_top = df['last_peak_price']
        box_bottom = df['last_trough_price']
        
        box_height = box_top - box_bottom
        box_midpoint = (box_top + box_bottom) / 2
        box_width_ratio = np.divide(box_height, box_midpoint, out=np.full_like(box_height, np.inf), where=box_midpoint!=0)
        is_valid_box = (box_height > 0) & (box_width_ratio < box_params.get('max_width_ratio', 0.25))

        # 5. 【协同判断】重新定义中期趋势 (context_mid_term_bullish)
        is_classic_uptrend = pd.Series(False, index=df.index)
        if mid_ma_col in df.columns and slow_ma_col in df.columns:
            is_classic_uptrend = (df[mid_ma_col] > df[slow_ma_col]) & (df['close_D'] > df[mid_ma_col])
        
        is_in_box = (df['close_D'] < box_top) & (df['close_D'] > box_bottom)
        is_box_above_ma = box_midpoint > df.get(mid_ma_col, pd.Series(np.inf, index=df.index))
        is_healthy_consolidation = is_valid_box & is_in_box & is_box_above_ma
        
        df['context_mid_term_bullish'] = is_classic_uptrend | is_healthy_consolidation

        # 6. 【信号生成】生成动态箱体的突破和跌破信号
        was_below_top = df['close_D'].shift(1) <= box_top.shift(1)
        breakout_signal = is_valid_box & (df['close_D'] > box_top) & was_below_top
        
        was_above_bottom = df['close_D'].shift(1) >= box_bottom.shift(1)
        breakdown_signal = is_valid_box & (df['close_D'] < box_bottom) & was_above_bottom

        # 7. 将信号存入实例变量，供买卖逻辑使用
        self.signals['dynamic_box_breakout'] = breakout_signal
        self.signals['dynamic_box_breakdown'] = breakdown_signal
        
        # 8. 清理临时列
        df.drop(columns=['last_peak_price', 'last_trough_price'], inplace=True)

    def _identify_board_patterns(self, df: pd.DataFrame, params: dict) -> Dict[str, pd.Series]:
        """
        【模块】【V2.1 容错增强版】识别A股特色的各种“板”形态。
        - 核心修正: 增加了价格判断的容错缓冲，以应对真实市场中的微小价格波动和近似计算误差。
        """
        params = self._get_params_block(params, 'board_pattern_params')
        if not params.get('enabled', False):
            return {} # 如果禁用，返回空字典

        # --- 准备基础数据 ---
        prev_close = df['close_D'].shift(1)
        limit_up_threshold = params.get('limit_up_threshold', 0.098)
        limit_down_threshold = params.get('limit_down_threshold', -0.098)
        high_turnover_rate = params.get('high_turnover_rate', 7.0)
        # ▼▼▼增加容错缓冲 ▼▼▼
        price_buffer = params.get('price_buffer', 0.005) # 0.5%的价格缓冲

        # --- 计算涨跌停价格（近似）---
        limit_up_price = prev_close * (1 + limit_up_threshold)
        limit_down_price = prev_close * (1 + limit_down_threshold)

        # --- 识别涨跌停状态 (带缓冲) ---
        # ▼▼▼在价格比较中应用缓冲 ▼▼▼
        is_limit_up = df['close_D'] >= limit_up_price * (1 - price_buffer)
        is_limit_down = df['close_D'] <= limit_down_price * (1 + price_buffer)
        is_limit_up_high = df['high_D'] >= limit_up_price * (1 - price_buffer)
        is_limit_down_low = df['low_D'] <= limit_down_price * (1 + price_buffer)

        # 1. 一字板 (Unbroken Board)
        is_one_word_shape = (df['open_D'] == df['high_D']) & (df['high_D'] == df['low_D']) & (df['low_D'] == df['close_D'])
        is_one_word_limit_up = is_limit_up & is_one_word_shape
        
        # 2. 换手板 (Turnover Board)
        # ▼▼▼放宽收盘价等于最高价的条件 ▼▼▼
        is_limit_up_close = (df['close_D'] >= df['high_D'] * (1 - price_buffer)) & is_limit_up
        is_opened_during_day = df['open_D'] < df['high_D'] # 盘中开过板
        is_high_turnover = df.get('turnover_rate_D', pd.Series(0, index=df.index)) > high_turnover_rate
        is_turnover_board = is_limit_up_close & is_opened_during_day & is_high_turnover

        # 3. 天地板 (Heaven-Earth Board) - 强力卖出/风险信号
        is_limit_down_close = (df['close_D'] <= df['low_D'] * (1 + price_buffer)) & is_limit_down
        is_heaven_earth_board = is_limit_up_high & is_limit_down_close

        # 4. 地天板 (Earth-Heaven Board) - 强力买入/反转信号
        # is_limit_up_close 已在上面定义
        is_earth_heaven_board = is_limit_down_low & is_limit_up_close
        
        if self.verbose_logging:
            print(f"    [调试-板形态V2.1]: 地天板触发: {is_earth_heaven_board.sum()} 天, 天地板触发: {is_heaven_earth_board.sum()} 天")

        return {
            'one_word_limit_up': is_one_word_limit_up,
            'turnover_board': is_turnover_board,
            'heaven_earth_board': is_heaven_earth_board, # 风险信号
            'earth_heaven_board': is_earth_heaven_board, # 机会信号
        }

    def _find_volume_breakdown_exit(self, df: pd.DataFrame, params: dict) -> pd.Series:
        """
        【卖出剧本】【V2.0 主力确认与筹码崩溃版】“结构崩溃” - 识别无法挽回的破位。
        - 核心升级: 从“跌破一根线”升级为“支撑体系的崩溃”，并由主力行为确认。
        - 策略逻辑:
          1. 结构崩溃: 股价必须同时跌破动态支撑(MA)和静态支撑(市场平均成本)。
          2. 主力砸盘: 破位必须伴随放量，且主力资金呈“大幅净流出”状态。
          3. 买方放弃: K线形态为坚决的、实体较大的阴线，表明买方毫无抵抗。
        """
        params = self._get_params_block(params, 'volume_breakdown_params')
        if not params.get('enabled', False):
            return pd.Series(False, index=df.index)

        # --- 从配置中读取或使用默认值 ---
        support_ma_period = params.get('support_ma_period', 55) # 使用55日线作为关键支撑
        volume_surge_ratio = params.get('volume_surge_ratio', 1.8)
        main_force_dump_ratio = params.get('main_force_dump_ratio', 0.1) # 主力净卖出额 > 前20日均成交额的10%
        
        # --- 准备所需列名 ---
        support_ma_col = f"EMA_{support_ma_period}_D"
        vol_ma_col = f"VOL_MA_{params.get('vol_ma_period', 21)}_D"
        cost_avg_col = 'weight_avg_D' # 市场平均成本
        required_cols = [
            'open_D', 'high_D', 'low_D', 'close_D', 'volume_D', 'amount_D',
            support_ma_col, vol_ma_col, cost_avg_col, 'net_main_force_amount_D'
        ]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            if self.verbose_logging:
                print(f"    [调试-结构崩溃-警告]: 缺少必需列: {missing_cols}，剧本跳过。")
            return pd.Series(False, index=df.index)

        # 1. 结构 - 识别支撑体系是否被击穿
        is_ma_broken = (df['close_D'] < df[support_ma_col]) & (df['close_D'].shift(1) >= df[support_ma_col].shift(1))
        is_cost_line_broken = df['close_D'] < df[cost_avg_col]
        is_structure_broken = is_ma_broken & is_cost_line_broken

        # 2. 力量 - 识别主力是否在主动、大力度砸盘
        is_volume_surge = df['volume_D'] > df[vol_ma_col] * volume_surge_ratio
        avg_amount_20d = df['amount_D'].rolling(20).mean()
        is_main_force_dumping = df['net_main_force_amount_D'] < -(avg_amount_20d * main_force_dump_ratio)
        is_forceful_breakdown = is_volume_surge & is_main_force_dumping

        # 3. 后果 - 识别买方是否放弃抵抗
        is_bearish_candle = df['close_D'] < df['open_D']
        # 阴线实体必须足够大，占据当天振幅的60%以上
        is_conviction_candle = (df['open_D'] - df['close_D']) > (df['high_D'] - df['low_D']) * 0.6
        is_buyer_absent = is_bearish_candle & is_conviction_candle

        final_signal = is_structure_broken & is_forceful_breakdown & is_buyer_absent

        if self.verbose_logging:
            print(f"    [调试-结构崩溃V2.0]: 结构崩溃: {is_structure_broken.sum()} | "
                  f"主力砸盘: {is_forceful_breakdown.sum()} | "
                  f"买方放弃: {is_buyer_absent.sum()} | "
                  f"最终信号: {final_signal.sum()}")

        return final_signal.fillna(False)

    def _find_upthrust_distribution_exit(self, df: pd.DataFrame, params: dict) -> pd.Series:
        """
        【卖出剧本】【V2.0 主力行为版】“主力叛逃” - 识别高位派发的真实意图。
        - 核心升级: 从价量形态升级为对“价格行为”与“资金行为”背离的捕捉。
        - 策略逻辑:
          1. 舞台(风险区): 股价大幅偏离中长期均线，进入“超涨”状态。
          2. 动作(虚假繁荣): 出现经典的高成交量、长上影线K线。
          3. 动机(资金叛逃): 当日主力资金必须是净流出，这是确认派发的核心证据。
        """
        params = self._get_params_block(params, 'upthrust_distribution_params')
        if not params.get('enabled', False):
            return pd.Series(False, index=df.index)

        # --- 从配置中读取或使用默认值 ---
        lookback = params.get('lookback_period', 30)
        upper_shadow_ratio = params.get('upper_shadow_ratio', 0.6)
        high_vol_quantile = params.get('high_volume_quantile', 0.85) # 提高成交量要求
        overextension_ma_period = params.get('overextension_ma_period', 55) # 用于判断超涨的均线
        overextension_threshold = params.get('overextension_threshold', 0.3) # 股价超过均线30%视为超涨

        # --- 准备所需列名 ---
        overextension_ma_col = f"EMA_{overextension_ma_period}_D"
        required_cols = [
            'open_D', 'high_D', 'low_D', 'close_D', 'volume_D', overextension_ma_col,
            'net_main_force_amount_D' # 依赖预先计算的主力净流入
        ]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            if self.verbose_logging:
                print(f"    [调试-主力叛逃-警告]: 缺少必需列: {missing_cols}，剧本跳过。")
            return pd.Series(False, index=df.index)

        # 1. 舞台 - 识别是否处于“超涨”风险区
        is_overextended = (df['close_D'] / df[overextension_ma_col] - 1) > overextension_threshold

        # 2. 动作 - 识别经典的长上影线 + 高成交量
        total_range = (df['high_D'] - df['low_D']).replace(0, np.nan)
        # 上影线定义更严格：必须是阳线实体上方或阴线实体上方的部分
        upper_shadow = (df['high_D'] - np.maximum(df['open_D'], df['close_D'])) / total_range
        has_long_upper_shadow = upper_shadow > upper_shadow_ratio
        is_high_volume = df['volume_D'] > df['volume_D'].rolling(lookback).quantile(high_vol_quantile)
        is_upthrust_action = has_long_upper_shadow & is_high_volume

        # 3. 动机 - 识别主力资金是否在净卖出
        is_main_force_selling = df['net_main_force_amount_D'] < 0

        final_signal = is_overextended & is_upthrust_action & is_main_force_selling

        if self.verbose_logging:
            print(f"    [调试-主力叛逃V2.0]: 超涨天数: {is_overextended.sum()} | "
                  f"派发动作: {is_upthrust_action.sum()} | "
                  f"主力净卖出: {is_main_force_selling.sum()} | "
                  f"最终信号: {final_signal.sum()}")

        return final_signal.fillna(False)









