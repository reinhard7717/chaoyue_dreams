# 文件: strategies/trend_following_strategy.py
# 版本: V21.0 - 适配新架构版
import logging
from decimal import Decimal
from utils.data_sanitizer import sanitize_for_json
from typing import Any, Dict, List, Optional, Tuple
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
        print("    - [类型标准化引擎 V40.6 智能版] 启动，检查并转换数据类型...")
        converted_cols = []
        for col in df.columns:
            # 只检查 object 类型的列，因为 Decimal 列会被 Pandas 识别为 object
            if df[col].dtype == 'object':
                # 尝试获取第一个非空值进行类型嗅探，以提高效率
                first_valid_item = df[col].dropna().iloc[0] if not df[col].dropna().empty else None
                
                # 如果第一个有效项是Decimal，则转换整列
                if isinstance(first_valid_item, Decimal):
                    print(f"      -> 发现列 '{col}' 包含Decimal对象，执行转换...")
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    converted_cols.append(col)

        if converted_cols:
            print(f"      -> 已将以下 'object' 类型列智能转换为数值类型: {converted_cols}")
        else:
            print("      -> 所有数值列类型正常，无需转换。")
        
        print("    - [类型标准化引擎 V40.6] 类型检查完成。")
        return df

    def apply_strategy(self, df_dict: Dict[str, pd.DataFrame], params: dict) -> Tuple[pd.DataFrame, Dict[str, pd.Series]]:
        """
        【V40.2 终极健壮版】
        - 核心修复: 对 `score_threshold` 参数应用 `_get_param_value` 解析器。
        - 效果: 确保了在进行最终信号判断时，分数阈值是一个正确的数值，而不是字典。
        """
        # print("\n--- [战术策略层 apply_strategy V22.2] 开始执行 ---") # 日常运行时可注释
        df = df_dict.get('D')
        if df is None or df.empty:
            return pd.DataFrame(), {}
        
        df = self._ensure_numeric_types(df)

        timeframe_suffixes = ['_D', '_W', '_M', '_5', '_15', '_30', '_60']

        # ▼▼▼ 精准的列名重命名逻辑 ▼▼▼
        rename_map = {
            col: f"{col}_D" for col in df.columns 
            if not any(col.endswith(suffix) for suffix in timeframe_suffixes) 
            and not col.startswith(('VWAP_', 'BASE_', 'playbook_', 'signal_', 'kline_', 'context_'))
        }

        if rename_map:
            df = df.rename(columns=rename_map)

        # 步骤 1: 准备所有衍生特征 (如主力/散户净流入)
        df = self._prepare_derived_features(df)

        # 步骤 2: 基于准备好的特征，计算所有斜率
        df = self._calculate_trend_slopes(df, params)

        self.signals, self.scores = {}, {}
        df = self.pattern_recognizer.identify_all(df)
        df.loc[:, 'signal_top_divergence'] = self._find_top_divergence_exit(df, params)
        self._analyze_dynamic_box_and_ma_trend(df, params)

        print("    - [信息] 核心计分流程开始...")
        df.loc[:, 'entry_score'], atomic_signals, score_details_df = self._calculate_entry_score(df, df_dict, params)
        self._last_score_details_df = score_details_df

        entry_scoring_params = self._get_params_block(params, 'entry_scoring_params')
        score_threshold = self._get_param_value(entry_scoring_params.get('score_threshold'), 100)

        df.loc[:, 'signal_entry'] = df['entry_score'] >= score_threshold

        print("    - [信息] 止盈逻辑判断开始...")
        df.loc[:, 'take_profit_signal'] = self._apply_take_profit_rules(df, df['signal_entry'], df['signal_top_divergence'], params)

        print("\n---【多时间框架协同策略(V22.2 融合版) 逻辑链调试】---")
        entry_signals = df[df['signal_entry']]
        print(f"【最终买入】(得分>{score_threshold})信号总数: {len(entry_signals)}")
        if not entry_signals.empty:
            last_entry_date = entry_signals.index[-1]
            last_entry_score = df.loc[last_entry_date, 'entry_score']
            print(f"  - 最近一次买入: {last_entry_date.date()} (当日总分: {last_entry_score:.2f})")
            
            print(f"  --- 得分小票 for {last_entry_date.date()} ---")
            score_breakdown = score_details_df.loc[last_entry_date].dropna()
            base_score_items = {k: v for k, v in score_breakdown.items() if k.startswith('BASE_') and v > 0}
            tactical_score_items = {k: v for k, v in score_breakdown.items() if not k.startswith('BASE_') and v != 0}
            
            base_total = sum(base_score_items.values())
            print(f"    [战略基础分: {base_total:.2f} pts]")
            if not base_score_items:
                print("      - (无)")
            else:
                for item, score in base_score_items.items():
                    print(f"      - {item}: {score:.2f} pts")

            tactical_total = sum(tactical_score_items.values())
            print(f"    [战术叠加分: {tactical_total:.2f} pts]")
            if not tactical_score_items:
                print("      - (无)")
            else:
                for item, score in tactical_score_items.items():
                    print(f"      - {item}: {score:.2f} pts")
            
            print(f"    ------------------------------------")
            print(f"    [核算总分: {base_total + tactical_total:.2f} pts]")

        return df, atomic_signals

    def prepare_db_records(self, stock_code: str, result_df: pd.DataFrame, atomic_signals: Dict[str, pd.Series], params: dict, result_timeframe: str = 'D') -> List[Dict[str, Any]]:
        """
        【V40.8 终极严谨版】将策略分析结果DataFrame转换为用于数据库存储的字典列表。
        - 核心修复: 采用更严谨的逻辑解析 strategy_name，并使用 'unknown_strategy' 作为清晰的默认值。
        - 解决问题: 彻底解决了 "unhashable type: 'dict'" TypeError，并提高了代码的健壮性和可维护性。
        """
        df_with_signals = result_df[
            (result_df['signal_entry'] == True) | (result_df['take_profit_signal'] > 0)
        ].copy()
        if df_with_signals.empty:
            return []
        
        records = []
        
        # 1. 安全地获取 'strategy_info' 配置块，如果不存在则返回空字典
        strategy_info_block = self._get_params_block(params, 'strategy_info', {})
        
        # 2. 从配置块中获取 'name' 参数，这可能是一个值，也可能是一个字典，如果不存在则为None
        name_param = strategy_info_block.get('name')
        
        # 3. 使用 _get_param_value 最终解析出字符串值，并提供一个清晰的默认值 'unknown_strategy'
        strategy_name = self._get_param_value(name_param, 'unknown_strategy')
        
        timeframe = result_timeframe

        for timestamp, row in df_with_signals.iterrows():
            triggered_playbooks_list = []
            if self._last_score_details_df is not None and timestamp in self._last_score_details_df.index:
                playbooks_with_scores = self._last_score_details_df.loc[timestamp]
                active_items = playbooks_with_scores[playbooks_with_scores > 0].index
                
                excluded_prefixes = ('BASE_', 'BONUS_', 'PENALTY_', 'INDUSTRY_')
                triggered_playbooks_list = [ item for item in active_items if not item.startswith(excluded_prefixes) ]

            is_setup_day = 'PULLBACK_SETUP' in triggered_playbooks_list
            context_dict = {k: v for k, v in row.items() if pd.notna(v)}
            sanitized_context = sanitize_for_json(context_dict)
            
            record = {
                "stock_code": stock_code,
                "trade_time": sanitize_for_json(timestamp),
                "timeframe": timeframe,
                "strategy_name": strategy_name, # 现在这里保证是字符串
                "close_price": sanitize_for_json(row.get('close_D')),
                "entry_score": sanitize_for_json(row.get('entry_score', 0.0)),
                "entry_signal": sanitize_for_json(row.get('signal_entry', False)),
                "exit_signal_code": sanitize_for_json(row.get('take_profit_signal', 0)),
                "is_long_term_bullish": sanitize_for_json(row.get('context_long_term_bullish', False)),
                "is_mid_term_bullish": sanitize_for_json(row.get('context_mid_term_bullish', False)),
                "is_pullback_setup": is_setup_day,
                "pullback_target_price": sanitize_for_json(row.get('pullback_target_price')),
                "triggered_playbooks": triggered_playbooks_list,
                "context_snapshot": sanitized_context,
            }
            records.append(record)
        return records

    def _calculate_trend_slopes(self, df: pd.DataFrame, params: dict) -> pd.DataFrame:
        """
        【V40.3 修复版】趋势斜率计算中心
        - 核心修复: 使用 `_get_param_value` 解析器来获取 `series_to_slope` 参数，确保能正确处理 `{"value": {...}}` 结构。
        """
        print("    - [斜率中心] 开始计算关键指标的趋势斜率 (V29.0 动力学版)...")
        slope_params = self._get_params_block(params, 'slope_params', {})
        if not self._get_param_value(slope_params.get('enabled'), False): # 修改: 应用解析器
            print("    - [斜率中心] 斜率计算被禁用。")
            return df

        def get_slope(y):
            y_clean = y.dropna()
            if y_clean.shape[0] < 2:
                return np.nan
            x = np.arange(len(y_clean))
            slope, _ = np.polyfit(x, y_clean.values, 1)
            return slope

        series_to_slope = self._get_param_value(slope_params.get('series_to_slope'), {}) # 修改: 应用解析器
        
        for col_name, lookbacks in series_to_slope.items():
            if col_name not in df.columns:
                print(f"    - [斜率中心] 警告: 列 '{col_name}' 不存在，跳过斜率计算。")
                continue
            
            for lookback in lookbacks:
                slope_col_name = f'SLOPE_{col_name}_{lookback}'
                df[slope_col_name] = df[col_name].rolling(window=lookback, min_periods=max(2, lookback // 2)).apply(get_slope, raw=False)
                
                accel_col_name = f'ACCEL_{col_name}_{lookback}'
                df[accel_col_name] = df[slope_col_name].rolling(window=lookback, min_periods=max(2, lookback // 2)).apply(get_slope, raw=False)
                
        print("    - [斜率中心] 所有斜率计算完成。")
        return df

    def _calculate_entry_score(self, df: pd.DataFrame, df_dict: Dict[str, pd.DataFrame], params: dict) -> Tuple[pd.Series, Dict[str, pd.Series], pd.DataFrame]:
        """
        【V39.7 一致性重构版】
        - 核心修改: 统一将“准备状态”和“触发事件”作为字典处理，消除了数据结构的不一致性，提升了代码的可读性和优雅性。
        """
        atomic_signals = {} # 初始化原子信号字典
        score_details_df = pd.DataFrame(index=df.index)
        scoring_params = self._get_params_block(params, 'entry_scoring_params')
        points = scoring_params.get('points', {})
        
        # --- 步骤1: 计算并记录战略背景基础分 (逻辑不变) ---
        print("    [调试-计分V34.0] 步骤1: 计算周线战略背景基础分...")
        king_signal_col = 'BASE_SIGNAL_BREAKOUT_TRIGGER'
        king_score = points.get('BREAKOUT_TRIGGER_SCORE', 150)
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
            accel_score = points.get('STRATEGIC_ACCEL_SCORE', 100)
            score_details_df.loc[df[strategic_accel_col], 'BASE_STRATEGIC_ACCEL'] = accel_score
            all_base_score_cols.append('BASE_STRATEGIC_ACCEL')

        score_details_df.fillna(0, inplace=True)
        if king_signal_col in score_details_df.columns:
            king_signal_mask = (score_details_df[king_signal_col] > 0)
            if king_signal_mask.any():
                other_base_score_cols = [col for col in all_base_score_cols if col != king_signal_col and col in score_details_df.columns]
                score_details_df.loc[king_signal_mask, other_base_score_cols] = 0
        
        # --- 步骤2: 进行核心前提交叉验证 (逻辑不变) ---
        print("    [调试-计分V34.0] 步骤2: 进行核心前提交叉验证...")
        validated_premises = self._validate_core_premises(df, params)
        robust_right_side_precondition = validated_premises['robust_right_side_precondition']

        # --- 步骤3: 计算所有“准备状态”和“触发事件” ---
        print("    [计分V37.0] 步骤3: 计算所有准备状态和触发事件...")
        # ▼▼▼ 统一使用字典作为容器 ▼▼▼
        # 将变量名从 setup_conditions_df 改为 setup_conditions，以准确反映其字典类型
        setup_conditions = self._calculate_setup_conditions(df, params)
        trigger_events = self._define_trigger_events(df, params)
        
        # 直接更新字典，不再需要 .to_dict('series') 转换
        atomic_signals.update(setup_conditions)
        atomic_signals.update(trigger_events)

        # --- 步骤4: 计算趋势动力学附加分 (逻辑不变) ---
        print("    [调试-计分V34.0] 步骤4: 计算趋势动力学附加分...")
        dynamics_score = self._score_trend_dynamics(df, params, validated_premises)

        # --- 步骤5: 【核心重构】构建并评估“剧本矩阵” ---
        print("    [计分V37.0] 步骤5: 按优先级评估“剧本矩阵”...")
        
        # ▼▼▼ 使用统一的字典 `setup_conditions` 来获取准备状态 ▼▼▼
        playbook_definitions = [
            # =================================================================================
            # === S级 (Tier S) 剧本: "完美风暴" (分数 > 300) ===
            # =================================================================================
            {
                'name': 'PERFECT_STORM',
                'setup': setup_conditions.get('SETUP_PROLONGED_COMPRESSION'), # 修改: setup_conditions_df -> setup_conditions
                'trigger': trigger_events.get('CHIP_INSTITUTIONAL_BREAKOUT'),
                'score': 350,
                'precondition': robust_right_side_precondition,
                'comment': '长期蓄势 + 主力点火，确定性最高的趋势启动模式。'
            },
            {
                'name': 'AWAKENED_BEAST',
                'setup': setup_conditions.get('SETUP_CAPITAL_FLOW_DIVERGENCE'), # 修改: setup_conditions_df -> setup_conditions
                'trigger': trigger_events.get('CHIP_INSTITUTIONAL_BREAKOUT'),
                'score': 330,
                'precondition': True,
                'comment': '底部价量背离后的强力反转信号。'
            },
            {
                'name': 'WASH_AND_RISE',
                'setup': setup_conditions.get('SETUP_PULLBACK_WITH_MF_INFLOW'), # 修改: setup_conditions_df -> setup_conditions
                'trigger': trigger_events.get('STRONG_POSITIVE_CANDLE'),
                'score': 310,
                'precondition': robust_right_side_precondition,
                'comment': '最经典的“假跌真吸”模式，回踩的质量极高。'
            },

            # =================================================================================
            # === A+级 (Tier A+) 剧本: "高置信度" (分数 250-299) ===
            # =================================================================================
            {
                'name': 'BREAKOUT_RETEST_GO',
                'setup': setup_conditions.get('SETUP_PULLBACK_POST_BREAKOUT'), # 修改: setup_conditions_df -> setup_conditions
                'trigger': trigger_events.get('STRONG_POSITIVE_CANDLE'),
                'score': 280,
                'precondition': robust_right_side_precondition,
                'comment': '最可靠的趋势延续形态之一：突破-回踩-再出发。'
            },
            {
                'name': 'REVERSAL_FIRST_PULLBACK',
                'setup': setup_conditions.get('SETUP_PULLBACK_POST_REVERSAL'), # 修改: setup_conditions_df -> setup_conditions
                'trigger': trigger_events.get('STRONG_POSITIVE_CANDLE'),
                'score': 270,
                'precondition': robust_right_side_precondition,
                'comment': '抓住新趋势的第一个上车点，通常有较好的盈亏比。'
            },
            {
                'name': 'OLD_DUCK_HEAD_TAKEOFF',
                'setup': setup_conditions.get('SETUP_DUCK_NECK_FORMING'), # 修改: setup_conditions_df -> setup_conditions
                'trigger': trigger_events.get('MA_RECLAIM'),
                'score': 260,
                'precondition': robust_right_side_precondition,
                'comment': '经典的均线理论形态，从整理到再次发力的转折点。'
            },

            # =================================================================================
            # === A级 (Tier A) 剧本: "标准可靠" (分数 200-249) ===
            # =================================================================================
            {
                'name': 'ENERGY_RELEASE',
                'setup': setup_conditions.get('SETUP_ENERGY_COMPRESSION'), # 修改: setup_conditions_df -> setup_conditions
                'trigger': trigger_events.get('VOLUME_SPIKE_BREAKOUT'),
                'score': 230,
                'precondition': robust_right_side_precondition,
                'comment': '通用性强的“盘久必涨”模式。'
            },
            {
                'name': 'PULLBACK_REBOUND_CONFIRMED',
                'setup': setup_conditions.get('SETUP_HEALTHY_PULLBACK'), # 修改: setup_conditions_df -> setup_conditions
                'trigger': trigger_events.get('STRONG_POSITIVE_CANDLE'),
                'score': 220,
                'precondition': robust_right_side_precondition,
                'comment': '最基础、最常见的趋势跟踪入场点。'
            },
            {
                'name': 'V_REVERSAL_ENTRY',
                'setup': setup_conditions.get('SETUP_SHOCK_BOTTOM'),  # 修改: setup_conditions_df -> setup_conditions
                'trigger': trigger_events.get('TRIGGER_V_RECOVERY'),
                'score': 210,
                'precondition': True,
                'comment': '高风险高收益的左侧交易模式，捕捉情绪拐点。'
            },

            # =================================================================================
            # === B级 (Tier B) 剧本: "机会主义" (分数 < 200) ===
            # =================================================================================
            {
                'name': 'WASHOUT_REVERSAL',
                'setup': setup_conditions.get('SETUP_WASHOUT_DAY'),       # 修改: setup_conditions_df -> setup_conditions
                'trigger': trigger_events.get('TRIGGER_WASHOUT_REVERSAL'),
                'score': 190,
                'precondition': True,
                'comment': '博弈主力洗盘后的快速拉升。'
            },
            {
                'name': 'RELATIVE_STRENGTH_LEADER',
                'setup': setup_conditions.get('SETUP_RELATIVE_STRENGTH'), # 修改: setup_conditions_df -> setup_conditions
                'trigger': trigger_events.get('STRONG_POSITIVE_CANDLE'),
                'score': 180,
                'precondition': robust_right_side_precondition,
                'comment': '捕捉市场中最强的品种，通常是下一波行情的领涨股。'
            },
            {
                'name': 'FORTRESS_DEFENDED',
                'setup': setup_conditions.get('SETUP_FORTRESS_SIEGE'), # 修改: setup_conditions_df -> setup_conditions
                'trigger': trigger_events.get('TRIGGER_FORTRESS_DEFENSE'),
                'score': 170,
                'precondition': robust_right_side_precondition,
                'comment': '基于结构支撑的防守反击模式。'
            },
        ]

        df['base_score'] = 0.0
        has_been_scored = pd.Series(False, index=df.index)

        for playbook in playbook_definitions:
            setup = playbook.get('setup', pd.Series(False, index=df.index))
            trigger = playbook.get('trigger', pd.Series(False, index=df.index))
            
            if setup is None or trigger is None:
                print(f"    - [剧本警告] 剧本 '{playbook['name']}' 的 setup 或 trigger 未定义，跳过。")
                continue
                
            condition = setup.shift(1).fillna(False) & trigger
            is_triggered = condition & playbook['precondition'] & ~has_been_scored
            
            if is_triggered.any():
                score = points.get(playbook['name'], {}).get('score', playbook['score'])
                df.loc[is_triggered, 'base_score'] = score
                score_details_df.loc[is_triggered, playbook['name']] = score
                has_been_scored.loc[is_triggered] = True
                print(f"    - [剧本命中] 命中剧本 '{playbook['name']}'，触发 {is_triggered.sum()} 天。")

        # 【观察分逻辑】
        watching_score = points.get('WATCHING_SCORE', 50)
        # ▼▼▼ 使用统一的字典 `setup_conditions` 来获取准备状态 ▼▼▼
        is_in_any_setup = (
            setup_conditions.get('SETUP_ENERGY_COMPRESSION', pd.Series(False, index=df.index)) |
            setup_conditions.get('SETUP_CAPITAL_FLOW_DIVERGENCE', pd.Series(False, index=df.index)) |
            setup_conditions.get('SETUP_HEALTHY_PULLBACK', pd.Series(False, index=df.index)) |
            setup_conditions.get('SETUP_DUCK_NECK_FORMING', pd.Series(False, index=df.index)) |
            setup_conditions.get('SETUP_CHIP_ACCUMULATION', pd.Series(False, index=df.index)) |
            setup_conditions.get('SETUP_WASHOUT_DAY', pd.Series(False, index=df.index)) |
            setup_conditions.get('SETUP_SHOCK_BOTTOM', pd.Series(False, index=df.index)) |
            setup_conditions.get('SETUP_DOJI_PAUSE', pd.Series(False, index=df.index)) |
            setup_conditions.get('SETUP_BULLISH_FLAG_FORMING', pd.Series(False, index=df.index)) |
            setup_conditions.get('SETUP_RELATIVE_STRENGTH', pd.Series(False, index=df.index)) |
            setup_conditions.get('SETUP_FORTRESS_SIEGE', pd.Series(False, index=df.index)) |
            setup_conditions.get('SETUP_BOTTOM_DIVERGENCE', pd.Series(False, index=df.index)) |
            setup_conditions.get('SETUP_BBAND_SQUEEZE', pd.Series(False, index=df.index))
        )
        is_watching = is_in_any_setup & ~has_been_scored
        if is_watching.any():
            df.loc[is_watching, 'base_score'] = watching_score
            score_details_df.loc[is_watching, 'WATCHING_SETUP'] = watching_score
            print(f"    - [后续跟踪] 发现 {is_watching.sum()} 天处于'观察准备'状态，赋予观察分。")

        # --- 步骤6: 融合剧本分、动力学分与环境修正项 (逻辑不变) ---
        print("    [调试-计分V34.0] 步骤6: 融合所有得分项...")
        final_score = df['base_score'].copy()
        has_primary_score = final_score > 0
        if (has_primary_score).any():
            final_score.loc[has_primary_score] += dynamics_score.loc[has_primary_score]
            score_details_df.loc[has_primary_score, 'DYNAMICS_SCORE'] = dynamics_score.loc[has_primary_score]

        cond_vwap_support = self._check_vwap_confirmation(df_dict, params)
        if (cond_vwap_support & has_primary_score).any():
            bonus = points.get('BONUS_VWAP_SUPPORT', 20)
            final_score.loc[cond_vwap_support & has_primary_score] += bonus
            score_details_df.loc[cond_vwap_support & has_primary_score, 'BONUS_VWAP_SUPPORT'] = bonus

        high_consensus_bonus = points.get('BONUS_HIGH_CONSENSUS', 30)
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
        
        # --- 步骤7: 合并战略与战术得分 (逻辑不变) ---
        print("    [调试-计分V34.0] 步骤7: 合并战略与战术得分...")
        base_score_from_weekly = score_details_df.filter(regex='^BASE_').sum(axis=1)
        final_score += base_score_from_weekly

        # --- 更新上下文状态 (逻辑不变) ---
        print("    [计分V39.4] 步骤X: 更新上下文状态...")
        df = self._update_contextual_states(df, score_details_df, validated_premises, params)

        # --- 步骤8: 最终风险否决层 ---
        print("    [调试-计分V34.0] 步骤8: 应用最终风险否决层...")
        # ▼▼▼【代码修改 V39.7】: 使用统一的字典 `setup_conditions` 来获取风险信号 ▼▼▼
        cond_trend_exhaustion = setup_conditions.get('RISK_TREND_EXHAUSTION', pd.Series(False, index=df.index))
        # ▲▲▲【代码修改 V39.7】▲▲▲
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
        df.drop(columns=['base_score'], inplace=True, errors='ignore')
        print("    [计分V37.0] 计分流程结束。")
        
        return final_score.round(0), atomic_signals, score_details_df.fillna(0)

    # ▼▼▼ “准备状态中心” (Setup Condition Center) ▼▼▼
    def _calculate_setup_conditions(self, df: pd.DataFrame, params: dict) -> Dict[str, pd.Series]:
        """
        【V40.3 完整实现版】
        - 核心升级: 根据配置文件，完整实现了所有交易剧本所需的“准备状态”(Setup)的计算。
        - 健壮性: 全面应用 `_get_param_value` 解析器，并为每个计算块增加 try-except 保护。
        """
        print("    - [准备状态中心 V40.3] 启动，开始计算所有准备状态...")
        setups = {}
        setup_params = self._get_params_block(params, 'setup_condition_params', {})

        # --- 1. 获取底层的、原子的筹码信号 ---
        try:
            chip_atomic_signals = self._define_chip_atomic_signals(df, params)
            setups['SETUP_CHIP_ACCUMULATION'] = chip_atomic_signals.get('ATOMIC_REINFORCEMENT', pd.Series(False, index=df.index))
            print(f"      -> '筹码吸筹'准备状态定义完成，发现 {setups['SETUP_CHIP_ACCUMULATION'].sum()} 天。")
        except Exception as e:
            print(f"      -> [警告] 计算'筹码吸筹'时出错: {e}")

        # --- 2. 能量压缩 (潜龙在渊) ---
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

        # --- 3. 资金暗流 (价量背离) ---
        try:
            p = setup_params.get('capital_flow_divergence_params', {})
            if self._get_param_value(p.get('enabled'), False):
                ma_period = self._get_param_value(p.get('trend_ma_period'), 55)
                ma_col = f'EMA_{ma_period}_D'
                mf_slope_col = 'SLOPE_net_mf_amount_D_10'
                if all(c in df.columns for c in [ma_col, mf_slope_col]):
                    cond1 = df['close_D'] < df[ma_col] # 价格弱势
                    cond2 = df[mf_slope_col] > self._get_param_value(p.get('mf_slope_threshold'), 0) # 主力流入
                    setups['SETUP_CAPITAL_FLOW_DIVERGENCE'] = cond1 & cond2
                    print(f"      -> '资金暗流'准备状态定义完成，发现 {setups.get('SETUP_CAPITAL_FLOW_DIVERGENCE', pd.Series([])).sum()} 天。")
        except Exception as e:
            print(f"      -> [警告] 计算'资金暗流'时出错: {e}")

        # --- 4. 健康回踩 ---
        try:
            p = setup_params.get('healthy_pullback_params', {})
            if self._get_param_value(p.get('enabled'), False):
                ma_period = self._get_param_value(p.get('support_ma'), 21)
                ma_col = f'EMA_{ma_period}_D'
                if ma_col in df.columns:
                    is_shrinking_vol = df['volume_D'] < df[f'VOL_MA_{ma_period}_D']
                    is_pullback = (df['close_D'] < df['close_D'].shift(1)) & (df['low_D'] < df[ma_col])
                    setups['SETUP_HEALTHY_PULLBACK'] = is_pullback & is_shrinking_vol
                    print(f"      -> '健康回踩'准备状态定义完成，发现 {setups.get('SETUP_HEALTHY_PULLBACK', pd.Series([])).sum()} 天。")
        except Exception as e:
            print(f"      -> [警告] 计算'健康回踩'时出错: {e}")

        # --- 5. 老鸭头-鸭颈 ---
        try:
            p = setup_params.get('duck_neck_forming_params', {})
            if self._get_param_value(p.get('enabled'), False):
                fast_ma = f"EMA_{self._get_param_value(p.get('fast_ma'), 10)}_D"
                slow_ma = f"EMA_{self._get_param_value(p.get('slow_ma'), 55)}_D"
                if all(c in df.columns for c in [fast_ma, slow_ma]):
                    cond1 = df[fast_ma] < df[slow_ma] # 快线在慢线之下
                    cond2 = df['volume_D'] < df['VOL_MA_21_D'] # 缩量
                    setups['SETUP_DUCK_NECK_FORMING'] = cond1 & cond2
                    print(f"      -> '老鸭头-鸭颈'准备状态定义完成，发现 {setups.get('SETUP_DUCK_NECK_FORMING', pd.Series([])).sum()} 天。")
        except Exception as e:
            print(f"      -> [警告] 计算'老鸭头-鸭颈'时出错: {e}")

        # --- 6. 巨阴洗盘日 ---
        try:
            p = setup_params.get('washout_day_params', {})
            if self._get_param_value(p.get('enabled'), False):
                vol_ma_period = self._get_param_value(p.get('vol_ma_period'), 20)
                vol_ma_col = f'VOL_MA_{vol_ma_period}_D'
                change_pct = df['close_D'].pct_change()
                is_big_drop = change_pct < self._get_param_value(p.get('washout_threshold'), -0.07)
                is_high_vol = df['volume_D'] > (df[vol_ma_col] * self._get_param_value(p.get('washout_volume_ratio'), 1.5))
                setups['SETUP_WASHOUT_DAY'] = (is_big_drop & is_high_vol).shift(1).fillna(False) # 准备状态是基于前一天的
                print(f"      -> '巨阴洗盘日'准备状态定义完成，发现 {setups.get('SETUP_WASHOUT_DAY', pd.Series([])).sum()} 天。")
        except Exception as e:
            print(f"      -> [警告] 计算'巨阴洗盘日'时出错: {e}")

        # --- 7. V反-休克谷底 ---
        try:
            p = setup_params.get('shock_bottom_params', {})
            if self._get_param_value(p.get('enabled'), False):
                drop_days = self._get_param_value(p.get('drop_days'), 10)
                drop_pct = self._get_param_value(p.get('drop_pct'), -0.25)
                winner_rate_threshold = self._get_param_value(p.get('winner_rate_threshold'), 10.0)
                
                highest_in_period = df['high_D'].rolling(window=drop_days).max()
                is_deep_drop = (df['low_D'] / highest_in_period - 1) < drop_pct
                is_panic_selling = df['winner_rate_D'] < winner_rate_threshold
                setups['SETUP_SHOCK_BOTTOM'] = is_deep_drop & is_panic_selling
                print(f"      -> 'V反-休克谷底'准备状态定义完成，发现 {setups.get('SETUP_SHOCK_BOTTOM', pd.Series([])).sum()} 天。")
        except Exception as e:
            print(f"      -> [警告] 计算'V反-休克谷底'时出错: {e}")

        # --- 8. 风险：趋势衰竭 ---
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

        # --- 9. 填充所有未计算的Setups为False，确保剧本矩阵安全 ---
        all_needed_setups = [
            'SETUP_PROLONGED_COMPRESSION', 'SETUP_CAPITAL_FLOW_DIVERGENCE', 'SETUP_PULLBACK_WITH_MF_INFLOW',
            'SETUP_PULLBACK_POST_BREAKOUT', 'SETUP_PULLBACK_POST_REVERSAL', 'SETUP_DUCK_NECK_FORMING',
            'SETUP_ENERGY_COMPRESSION', 'SETUP_HEALTHY_PULLBACK', 'SETUP_SHOCK_BOTTOM', 'SETUP_WASHOUT_DAY',
            'SETUP_RELATIVE_STRENGTH', 'SETUP_FORTRESS_SIEGE', 'SETUP_CHIP_ACCUMULATION', 'RISK_TREND_EXHAUSTION',
            'SETUP_DOJI_PAUSE', 'SETUP_BULLISH_FLAG_FORMING', 'SETUP_BOTTOM_DIVERGENCE', 'SETUP_BBAND_SQUEEZE'
        ]
        for setup_name in all_needed_setups:
            if setup_name not in setups:
                setups[setup_name] = pd.Series(False, index=df.index)

        # --- 10. 最终清洗 ---
        for key in list(setups.keys()):
            setups[key] = setups[key].fillna(False)
            
        print("    - [准备状态中心 V40.3] 所有准备状态计算完成。")
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
        【V39.2 多维趋势健康度审计中心版】
        - 核心升级: 引入资金流、波动率、内部强度指标，将诊断维度从4个扩展到6个。
        - 审计维度:
          1. 方向 (Direction): 趋势的速度。
          2. 动量 (Momentum): 趋势的加速度。
          3. 共识 (Consensus): 均线族排列。
          4. 资本流 (Capital Flow): 主力资金流入的趋势。
          5. 波动性 (Volatility): 趋势扩张的稳定性。
          6. 内在强度 (Internal Strength): 趋势的强度与持续性 (ADX & RSI)。
        - 输出: 一个包含0-6分制的“趋势健康度得分”和关键前提条件的字典。
        """
        print("    - [前提验证中心 V39.2] 开始启动'多维趋势健康度审计'...")
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
        bbw_slope_col = 'SLOPE_BBW_21_2.0_D_10'
        adx_col = 'ADX_14_D'
        rsi_col = 'RSI_13_D'

        # 解析嵌套的参数块
        volatility_params = ma_params.get('volatility_params', {})
        min_bbw_slope = self._get_param_value(volatility_params.get('min_bbw_slope'), 0)
        max_bbw_slope = self._get_param_value(volatility_params.get('max_bbw_slope'), 0.01)

        strength_params = ma_params.get('internal_strength_params', {})
        adx_threshold = self._get_param_value(strength_params.get('adx_threshold'), 20)
        rsi_exhaustion_threshold = self._get_param_value(strength_params.get('rsi_exhaustion_threshold'), 80)
        
        health_score_threshold = self._get_param_value(ma_params.get('health_score_threshold'), 4)

        required_cols = [
            short_ma_col, mid_ma_col, slow_ma_col, mid_ma_slope_col, mid_ma_accel_col,
            mf_slope_col, bbw_slope_col, adx_col, rsi_col
        ]

        if not all(c in df.columns for c in required_cols):
            missing = [col for col in required_cols if col not in df.columns]
            print(f"    - [前提验证中心-警告]: 缺少审计所需的列: {missing}，前提验证将受限。请检查指标和斜率计算配置。")
            return {
                'robust_right_side_precondition': pd.Series(False, index=df.index),
                'trend_health_score': pd.Series(0, index=df.index),
                'is_trend_accelerating': pd.Series(False, index=df.index)
            }

        # --- 维度1: 方向 (Direction) - 趋势速度为正？ ---
        dimension_direction = df[mid_ma_slope_col] > 0
        print(f"      -> 维度1 (方向): {dimension_direction.sum()} 天满足。")

        # --- 维度2: 动量 (Momentum) - 趋势是否在加速？ ---
        dimension_momentum = df[mid_ma_accel_col] > 0
        print(f"      -> 维度2 (动量): {dimension_momentum.sum()} 天满足。")

        # --- 维度3: 共识 (Consensus) - 均线族是否呈多头排列？ ---
        dimension_consensus = (df[short_ma_col] > df[mid_ma_col]) & (df[mid_ma_col] > df[slow_ma_col])
        print(f"      -> 维度3 (共识): {dimension_consensus.sum()} 天满足。")

        # --- 维度4: 资本流 (Capital Flow) - 主力资金是否持续流入？ ---
        dimension_capital = df[mf_slope_col] > 0
        print(f"      -> 维度4 (资本流): {dimension_capital.sum()} 天满足。")

        # --- 维度5: 波动性 (Volatility) - 趋势扩张是否稳定？ ---
        dimension_volatility = (df[bbw_slope_col] > min_bbw_slope) & (df[bbw_slope_col] < max_bbw_slope)
        print(f"      -> 维度5 (波动性): {dimension_volatility.sum()} 天满足。")

        # --- 维度6: 内在强度 (Internal Strength) - 趋势是否强劲且未衰竭？ ---
        is_trend_strong = df[adx_col] > adx_threshold
        is_not_exhausted = df[rsi_col] < rsi_exhaustion_threshold
        dimension_strength = is_trend_strong & is_not_exhausted
        print(f"      -> 维度6 (内在强度): {dimension_strength.sum()} 天满足。")

        # --- 形成健康度评分 (0-6分) ---
        trend_health_score = (
            dimension_direction.astype(int) +
            dimension_momentum.astype(int) +
            dimension_consensus.astype(int) +
            dimension_capital.astype(int) +
            dimension_volatility.astype(int) +
            dimension_strength.astype(int)
        )

        # --- 定义最终的前提条件 ---
        robust_precondition = (trend_health_score >= health_score_threshold)

        premises['robust_right_side_precondition'] = robust_precondition
        premises['trend_health_score'] = trend_health_score
        premises['is_trend_accelerating'] = dimension_momentum

        print(f"      -> [审计完成] '高置信度'右侧前提 (健康分>={health_score_threshold}) 触发了 {robust_precondition.sum()} 天。")

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
                # 只有当新事件的优先级更高时，才覆盖
                if matched_tier['priority'] > current_priority:
                    df.at[idx, 'temp_priority'] = matched_tier['priority']
                    df.at[idx, 'temp_duration'] = matched_tier['duration']
                    # 将重置信息直接存入主导上下文名称，简化后续逻辑
                    df.at[idx, 'CONTEXT_ACTIVE_NAME'] = base_name
                    print(f"          - 日期 {idx.date()}: 剧本 '{playbook_name}' 命中，健康分 {score_at_trigger:.0f}，匹配等级(P{matched_tier['priority']}, D{matched_tier['duration']})，设置主导叙事为 '{base_name}'。")

        # --- 步骤 4: 更新计时器和主导上下文状态 ---
        print("        -> 步骤2.3: 更新主导上下文状态...")
        # 初始化主导上下文列
        if 'CONTEXT_ACTIVE_NAME' not in df.columns:
            df['CONTEXT_ACTIVE_NAME'] = 'NONE'
        
        # 状态继承：如果今天没有新事件，则继承昨天的状态
        df['CONTEXT_ACTIVE_NAME'] = df['CONTEXT_ACTIVE_NAME'].where(df['temp_priority'] > 0, df['CONTEXT_ACTIVE_NAME'].shift(1).fillna('NONE'))
        
        # 更新所有计时器
        for base_name in all_base_names:
            timer_col = f"CONTEXT_{base_name}_TIMER"
            # 如果今天的主导叙事是这个，就用动态计算出的duration重置计时器
            reset_mask = (df['CONTEXT_ACTIVE_NAME'] == base_name) & (df['temp_duration'] > 0)
            if reset_mask.any():
                df.loc[reset_mask, timer_col] = df.loc[reset_mask, 'temp_duration']

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
        【V40.3 修复版】
        - 核心修复: 全面应用 `_get_param_value` 解析器，解决因参数结构导致的 TypeError。
        """
        print("    - [触发事件中心 V40.3] 启动，开始定义所有原子化触发事件...")
        triggers = {}
        trigger_params = self._get_params_block(params, 'trigger_event_params', {})
        
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
        print("      -> 正在计算'资金力量'信号...")
        mf_ratio = df.get('net_mf_amount_ratio_D', pd.Series(0, index=df.index))
        
        # 核心修复: 使用解析器获取正确的阈值
        institutional_buying_threshold = self._get_param_value(trigger_params.get('institutional_buying_ratio'), 0.05)
        is_institutional_buying = mf_ratio > institutional_buying_threshold
        
        elg_net_buy = df.get('buy_elg_amount_D', 0) - df.get('sell_elg_amount_D', 0)
        net_mf_amount = df.get('net_mf_amount_D', 0)
        vol_ma_period = self._get_param_value(trigger_params.get('vol_ma_period'), 21)
        vol_ma_col = f"VOL_MA_{vol_ma_period}_D"
        volume_spike_ratio = self._get_param_value(trigger_params.get('volume_spike_ratio'), 2.0)
        is_volume_spike = df['volume_D'] > (df.get(vol_ma_col, df['volume_D']) * volume_spike_ratio)
        is_hot_money_blitz = is_institutional_buying & (elg_net_buy > net_mf_amount * 0.7) & is_volume_spike
        
        # --- 步骤3: 【核心】融合“结构”与“力量”，生成高阶触发器 ---
        print("      -> 正在融合'结构'与'力量'生成高阶触发器...")
        triggers['CHIP_INSTITUTIONAL_BREAKOUT'] = is_breakout_structure & is_institutional_buying & ~is_hot_money_blitz
        triggers['CHIP_HOT_MONEY_BREAKOUT'] = is_breakout_structure & is_hot_money_blitz
        triggers['CHIP_CONFIRMED_ACCELERATION'] = is_acceleration_structure & is_institutional_buying
        print(f"      -> 高阶触发器生成: 机构突破({triggers['CHIP_INSTITUTIONAL_BREAKOUT'].sum()}), 游资突破({triggers['CHIP_HOT_MONEY_BREAKOUT'].sum()}), 确认加速({triggers['CHIP_CONFIRMED_ACCELERATION'].sum()})")

        # --- 步骤4: 定义通用价量与形态触发器 (逻辑保持，代码更健壮) ---
        # 强势阳线 (通用触发器)
        try:
            p = trigger_params.get('positive_candle', {})
            if self._get_param_value(p.get('enabled'), True):
                is_green = df['close_D'] > df['open_D']
                body_range = (df['high_D'] - df['low_D']).replace(0, np.nan)
                body_ratio = (df['close_D'] - df['open_D']) / body_range
                # 核心修复: 使用解析器获取正确的阈值
                min_body_ratio = self._get_param_value(p.get('min_body_ratio'), 0.6)
                is_strong_body = body_ratio > min_body_ratio
                triggers['STRONG_POSITIVE_CANDLE'] = is_green & is_strong_body
                print(f"      -> '强势阳线' 触发器定义完成，发现 {triggers.get('STRONG_POSITIVE_CANDLE', pd.Series([])).sum()} 天。")
        except Exception as e:
            print(f"      -> [警告] 计算'强势阳线'触发器时出错: {e}")

        # 通用放量突破 (通用触发器)
        try:
            p = trigger_params.get('volume_spike_breakout', {})
            if self._get_param_value(p.get('enabled'), True):
                vol_ma_period = self._get_param_value(p.get('vol_ma_period'), 21)
                vol_ma_col = f"VOL_MA_{vol_ma_period}_D"
                if vol_ma_col in df.columns:
                    volume_ratio = self._get_param_value(p.get('volume_ratio'), 2.0)
                    lookback = self._get_param_value(p.get('lookback_period'), 10)
                    is_volume_spike_gen = df['volume_D'] > df[vol_ma_col] * volume_ratio
                    is_price_breakout = df['close_D'] > df['high_D'].shift(1).rolling(lookback).max()
                    triggers['VOLUME_SPIKE_BREAKOUT'] = is_volume_spike_gen & is_price_breakout
                    print(f"      -> '放量突破' 触发器定义完成，发现 {triggers.get('VOLUME_SPIKE_BREAKOUT', pd.Series([])).sum()} 天。")
        except Exception as e:
            print(f"      -> [警告] 计算'放量突破'触发器时出错: {e}")

        # 收复关键均线 (特定场景触发器)
        try:
            p = trigger_params.get('ma_reclaim', {})
            if self._get_param_value(p.get('enabled'), True):
                ma_period = self._get_param_value(p.get('ma_period'), 21)
                ma_col = f"EMA_{ma_period}_D"
                if ma_col in df.columns:
                    triggers['MA_RECLAIM'] = (df['close_D'] > df[ma_col]) & (df['close_D'].shift(1) <= df[ma_col].shift(1))
                    print(f"      -> '收复关键均线' 触发器定义完成，发现 {triggers.get('MA_RECLAIM', pd.Series([])).sum()} 天。")
        except Exception as e:
            print(f"      -> [警告] 计算'收复均线'触发器时出错: {e}")

        # “V型反转”触发器 (TRIGGER_V_RECOVERY)
        try:
            p = self._get_params_block(params, 'v_reversal_entry_params', {}) # 注意这里参数块不同
            if self._get_param_value(p.get('enabled'), True) and 'net_mf_amount_D' in df.columns:
                reversal_rally_pct = self._get_param_value(p.get('reversal_rally_pct'), 0.05)
                is_strong_rally = (df['close_D'] / df['close_D'].shift(1) - 1) > reversal_rally_pct
                is_strong_candle = (df['close_D'] - df['open_D']) > (df['high_D'] - df['low_D']) * 0.6
                drop_days = self._get_param_value(p.get('drop_days'), 10)
                avg_amount_in_drop = df['amount_D'].shift(1).rolling(window=drop_days).mean()
                is_huge_mf_inflow = df['net_mf_amount_D'].fillna(0) > (avg_amount_in_drop * 0.1)
                triggers['TRIGGER_V_RECOVERY'] = is_strong_rally & is_strong_candle & is_huge_mf_inflow
                print(f"      -> 'V型反转修复' 触发器定义完成，发现 {triggers.get('TRIGGER_V_RECOVERY', pd.Series([])).sum()} 天。")
        except Exception as e:
            print(f"      -> [警告] 计算'V型反转修复'触发器时出错: {e}")
            
        # “巨阴洗盘”触发器 (TRIGGER_WASHOUT_REVERSAL)
        try:
            p = self._get_params_block(params, 'washout_reversal_params', {})
            if self._get_param_value(p.get('enabled'), True):
                is_reversal_cover = df['close_D'] > df['close_D'].shift(1)
                reversal_rally = (df['close_D'] - df['open_D']) / df['open_D'].replace(0, np.nan)
                reversal_rally_threshold = self._get_param_value(p.get('reversal_rally_threshold'), 0.03)
                is_strong_reversal_candle = reversal_rally > reversal_rally_threshold
                triggers['TRIGGER_WASHOUT_REVERSAL'] = is_reversal_cover & is_strong_reversal_candle
                print(f"      -> '巨阴洗盘反包' 触发器定义完成，发现 {triggers.get('TRIGGER_WASHOUT_REVERSAL', pd.Series([])).sum()} 天。")
        except Exception as e:
            print(f"      -> [警告] 计算'巨阴洗盘反包'触发器时出错: {e}")

        # “堡垒防卫”触发器 (TRIGGER_FORTRESS_DEFENSE)
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

        # --- 步骤5: 最终清洗 ---
        for key in list(triggers.keys()):
            if triggers[key] is None:
                triggers[key] = pd.Series(False, index=df.index)
            else:
                triggers[key] = triggers[key].fillna(False)

        print("    - [触发事件中心 V40.3] 所有原子化触发事件定义完成。")
        return triggers

    # ▼▼▼ “原子筹码信号提供器” (Atomic Chip Signal Provider) ▼▼▼
    def _define_chip_atomic_signals(self, df: pd.DataFrame, params: dict) -> Dict[str, pd.Series]:
        """
        【V39.7 原子信号提供器版】
        - 核心职责: 仅计算并返回底层的、原子的筹码结构信号。
        - 不再负责定义“准备状态”或“触发器”，只提供最纯粹的结构信号作为原材料。
        """
        print("    - [原子筹码信号 V39.7] 启动，开始计算底层筹码结构信号...")
        signals = {}
        chip_params = self._get_params_block(params, 'chip_signal_params', {})
        
        # --- 信号1: 成本区强化 (用于准备状态) ---
        reinforce_params = chip_params.get('reinforcement', {})
        if reinforce_params.get('enabled', True):
            cost_dist_narrow = df['cost_85pct_D'] / df['cost_15pct_D'] < reinforce_params.get('max_cost_dist_ratio', 1.25)
            winner_rate_stable = df['winner_rate_D'].between(
                reinforce_params.get('min_winner_rate', 60), 
                reinforce_params.get('max_winner_rate', 95)
            )
            signals['ATOMIC_REINFORCEMENT'] = cost_dist_narrow & winner_rate_stable
        
        # --- 信号2: 集中度突破 (用于准备状态) ---
        conc_params = chip_params.get('concentration_break', {})
        if conc_params.get('enabled', True):
            cost_95_prev = df['cost_95pct_D'].shift(1)
            signals['ATOMIC_CONCENTRATION_BREAK'] = (df['close_D'] > cost_95_prev) & (df['close_D'].shift(1) <= cost_95_prev)

        # --- 信号3: 成本区突破 (用于触发事件) ---
        break_params = chip_params.get('cost_breakthrough', {})
        if break_params.get('enabled', True):
            signals['ATOMIC_COST_BREAKTHROUGH'] = df['close_D'] > df['cost_95pct_D']

        # --- 信号4: 跨越障碍 (用于触发事件) ---
        hurdle_params = chip_params.get('hurdle_clear', {})
        if hurdle_params.get('enabled', True):
            signals['ATOMIC_HURDLE_CLEAR'] = df['close_D'] > df['weight_avg_D']

        # --- 信号5: 压力释放 (用于触发事件) ---
        release_params = chip_params.get('pressure_release', {})
        if release_params.get('enabled', True):
            signals['ATOMIC_PRESSURE_RELEASE'] = df['winner_rate_D'] > release_params.get('min_winner_rate', 90)

        # --- 清洗 ---
        for key in list(signals.keys()):
            if key in signals and signals[key] is not None:
                signals[key] = signals[key].fillna(False)
            else:
                signals[key] = pd.Series(False, index=df.index)
        
        print("    - [原子筹码信号 V39.7] 底层信号计算完成。")
        return signals
    
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

    def _check_cmf_confirmation(self, df: pd.DataFrame, params: dict) -> pd.Series:
        """
        【状态确认】检查CMF资金流指标是否为正，以确认买入信号。
        这不是一个直接的买入信号，而是一个用于给其他信号加分或加权的确认状态。
        返回一个布尔序列，表示当天CMF是否处于“资金流入”的确认状态。
        """
        # 从配置中获取参数
        params = self._get_params_block(params, 'cmf_params')
        if not params.get('enabled', True): # 默认启用
            return pd.Series(False, index=df.index)

        # 动态构建CMF列名
        period = params.get('cmf_period', 21) # 默认值也更新为21
        cmf_col = f'CMF_{period}_D'

        # 检查必需的列是否存在，确保健壮性
        if cmf_col not in df.columns:
            # 仅在首次或需要时打印警告，避免刷屏
            if not hasattr(self, '_warned_missing_cmf_col'):
                logger.warning(f"缺少CMF列 '{cmf_col}'，CMF确认功能将不生效。请检查指标计算配置。")
                print(f"columns: {df.columns.to_list()}")
                # 使用一个实例属性来确保警告只打印一次
                self._warned_missing_cmf_col = True
            return pd.Series(False, index=df.index)

        # 获取确认阈值，通常为0
        threshold = params.get('threshold', 0)
        
        # CMF大于阈值（通常为0）表示资金流入，是有效的确认信号
        is_confirmed = df[cmf_col] > threshold
        if self.verbose_logging:
            print(f"调试信息: CMF确认信号触发 {is_confirmed.sum()} 次 (CMF > {threshold})")
        
        return is_confirmed

    def _check_upward_gap_support(self, df: pd.DataFrame, params: dict) -> pd.Series:
        """
        【状态确认】检查近期是否存在未回补的向上跳空缺口。
        这不是一个直接的买入信号，而是一个强烈的趋势确认状态，用于给其他信号加分。
        返回一个布尔序列，表示当天是否处于“缺口支撑”的强势状态下。
        """
        params = self._get_params_block(params, 'gap_support_params')
        if not params.get('enabled', False):
            return pd.Series(False, index=df.index)

        gap_check_days = params.get('gap_check_days', 5)
        
        # 1. 识别所有向上跳空缺口，并记录缺口上沿（支撑位）
        gap_support_level = df['high_D'].shift(1)
        is_gap_up = df['low_D'] > gap_support_level
        
        # 2. 创建一个只在缺口日有值的支撑位序列
        active_support = pd.Series(np.nan, index=df.index)
        active_support.loc[is_gap_up] = gap_support_level[is_gap_up]
        
        # 3. 向前填充支撑位，模拟支撑的持续性，但有时间限制
        active_support = active_support.ffill(limit=gap_check_days)
        
        # 4. 检查从缺口日之后，价格是否始终维持在支撑位之上
        # 使用一个辅助列来判断是否在检查周期内
        is_in_check_period = active_support.notna() & (active_support.diff() != 0).cumsum().ffill().gt(0)
        
        # 检查从缺口日到当前，最低价是否一直高于支撑位
        # 使用 groupby 和 cummin 实现此逻辑
        group = (active_support.diff() != 0).cumsum()
        lowest_since_gap = df.groupby(group)['low_D'].transform('cummin')
        
        is_supported = lowest_since_gap > active_support
        
        return is_supported & is_in_check_period.fillna(False)

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

    # 【模块】主力筹码运作行为识别

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

    def _find_bottom_divergence_entry(self, df: pd.DataFrame, params: dict) -> pd.Series:
        """
        【V4.1 增强调试版】识别“复合底背离”买入信号。
        - 新增: 详细的print语句，用于追踪数据列、参数、中间计算结果，快速定位信号为0的问题。
        """
        # 步骤1: 获取并打印参数
        div_params = self._get_params_block(params, 'bottom_divergence_params')
        if not div_params.get('enabled', False):
            return pd.Series(False, index=df.index)

        # ▼▼▼ 增加详细的调试打印 ▼▼▼
        if self.verbose_logging:
            print("\n--- [调试-复合底背离-启动] ---")

        # 从配置中读取参数
        distance = div_params.get('distance', 5)
        price_prominence = div_params.get('prominence_price', 0.02)
        rsi_col = div_params.get('rsi_col', 'RSI_14_D')
        prominence_rsi = div_params.get('prominence_rsi', 7.5)
        macd_hist_col = div_params.get('macd_hist_col', 'MACD_HIST_ZSCORE_D')
        prominence_macd = div_params.get('prominence_macd_zscore', 0.75)
        confirmation_days = div_params.get('confirmation_days', 3)

        if self.verbose_logging:
            print(f"    [参数检查] distance: {distance}, price_prominence: {price_prominence}")
            print(f"    [参数检查] rsi_col: '{rsi_col}', prominence_rsi: {prominence_rsi}")
            print(f"    [参数检查] macd_hist_col: '{macd_hist_col}', prominence_macd: {prominence_macd}")
            print(f"    [参数检查] confirmation_days: {confirmation_days}")

        # 步骤2: 检查数据列是否存在
        base_cols = ['low_D', 'close_D', 'open_D']
        rsi_col_found = rsi_col in df.columns
        macd_col_found = macd_hist_col in df.columns

        if self.verbose_logging:
            print(f"    [数据列检查] 核心价格列: {'存在' if all(c in df.columns for c in base_cols) else '缺失!'}")
            print(f"    [数据列检查] RSI列 '{rsi_col}': {'存在' if rsi_col_found else '!!! 不存在 !!!'}")
            print(f"    [数据列检查] MACD列 '{macd_hist_col}': {'存在' if macd_col_found else '!!! 不存在 !!!'}")

        if not all(c in df.columns for c in base_cols):
            if self.verbose_logging: print("    [调试终止] 缺少核心价格列，无法执行。")
            return pd.Series(False, index=df.index)
        
        if not rsi_col_found and not macd_col_found:
            if self.verbose_logging: print("    [调试终止] RSI和MACD列均不存在，无法执行。")
            return pd.Series(False, index=df.index)

        # 步骤3: 调用核心算法，计算所有“背离事件”
        divergence_events = pd.Series(False, index=df.index)

        # --- 计算RSI底背离 ---
        if rsi_col_found:
            rsi_trough_params = {"distance": distance, "prominence": price_prominence, "prominence_indicator": prominence_rsi}
            _, rsi_bottom_div = self._find_divergence(df['low_D'], df[rsi_col], {}, rsi_trough_params)
            divergence_events |= rsi_bottom_div
            if self.verbose_logging:
                # ▼▼▼ 增加详细的调试打印 ▼▼▼
                print(f"    [RSI背离检测] 发现 {rsi_bottom_div.sum()} 个RSI背离事件。")

        # --- 计算MACD底背离 ---
        if macd_col_found:
            macd_trough_params = {"distance": distance, "prominence": price_prominence, "prominence_indicator": prominence_macd}
            _, macd_bottom_div = self._find_divergence(df['low_D'], df[macd_hist_col], {}, macd_trough_params)
            divergence_events |= macd_bottom_div
            if self.verbose_logging:
                # ▼▼▼ 增加详细的调试打印 ▼▼▼
                print(f"    [MACD背离检测] 发现 {macd_bottom_div.sum()} 个MACD背离事件。")

        # 步骤4: 应用“因果律”确认逻辑
        total_events = divergence_events.sum()
        if self.verbose_logging:
            # ▼▼▼ 增加详细的调试打印 ▼▼▼
            print(f"    [事件汇总] 总计发现 {total_events} 个复合背离事件。")

        if total_events == 0:
            if self.verbose_logging:
                print("    [调试终止] 未发现任何背离事件，无法生成最终信号。请检查参数是否过于严格或数据本身无此类形态。")
                print("--- [调试-复合底背离-结束] ---\n")
            return pd.Series(False, index=df.index)

        is_green_candle = df['close_D'] > df['open_D']
        has_recent_divergence = divergence_events.rolling(window=confirmation_days, min_periods=1).sum() > 0
        final_signal = is_green_candle & has_recent_divergence.shift(1).fillna(False) & ~divergence_events
        
        # 步骤5: 打印最终结果并返回
        if self.verbose_logging:
            print(f"    [信号确认] 发现 {is_green_candle.sum()} 个阳线。")
            print(f"    [信号确认] 经过'因果律'过滤后，最终生成 {final_signal.sum()} 个买入信号。")
            print("--- [调试-复合底背离-结束] ---\n")
              
        return final_signal.fillna(False)

    # 寻找BIAS超跌反弹买点
    def _find_bias_reversal_entry(self, df: pd.DataFrame, params: dict) -> pd.Series:
        """【V2.10 】寻找BIAS极端超跌后的反弹买点。"""
        params = self._get_params_block(params, 'bias_reversal_entry_params')
        if not params.get('enabled', False): return pd.Series(False, index=df.index)

        bias_col = f"BIAS_{params.get('bias_period', 20)}_D"
        if bias_col not in df.columns: return pd.Series(False, index=df.index)

        # 条件1: BIAS值低于极端阈值，表明严重超卖
        is_extreme_oversold = df[bias_col] < params.get('oversold_threshold', -15.0)
        
        # 条件2: BIAS开始回升（今日BIAS > 昨日BIAS），出现反弹迹象
        is_rebounding = df[bias_col] > df[bias_col].shift(1)

        # 信号：前一日极端超卖，今日开始反弹
        signal = is_extreme_oversold.shift(1).fillna(False) & is_rebounding
        return signal

    # 通用的背离检测逻辑
    def _find_divergence(self, price_series: pd.Series, indicator_series: pd.Series, 
                         find_peaks_params: dict, find_troughs_params: dict) -> Tuple[pd.Series, pd.Series]:
        """
        【V4.0 算法修复版】通用背离检测函数。
        - 核心修复: 解决了prominence参数的“相对/绝对”值混用问题。
        - 动态计算: 能根据价格序列的实际范围，将配置文件中的相对prominence(如0.02)转换为绝对值。
        - 参数兼容: 允许为指标(indicator)单独指定一个绝对的prominence值(prominence_indicator)。
        - 向量化保留: 完全保留了V3.1版本中高效的ffill向量化核心逻辑。
        """
        # 步骤1: 创建一个临时的DataFrame，避免污染原始df
        temp_df = pd.DataFrame({
            'price': price_series,
            'indicator': indicator_series
        })

        # 步骤2: 【核心修复】动态计算并准备 find_peaks 所需的参数
        # ----------------------------------------------------------------
        # 复制参数字典，避免修改传入的原始字典
        peak_params = find_peaks_params.copy()
        trough_params = find_troughs_params.copy()

        # 计算价格序列的波动范围，用于将相对prominence转换为绝对值
        price_range = temp_df['price'].max() - temp_df['price'].min()
        if price_range == 0: price_range = 0.01 # 避免在价格不变时除以零

        # --- 准备价格查找参数 ---
        price_peak_params = peak_params.copy()
        if 'prominence' in price_peak_params:
            # 将价格的相对prominence转换为绝对值
            relative_prom = price_peak_params['prominence']
            price_peak_params['prominence'] = price_range * relative_prom
            if self.verbose_logging:
                print(f"    [调试-背离检测] 价格波峰相对prominence: {relative_prom}, 计算后绝对值: {price_peak_params['prominence']:.4f}")
        price_peak_params.pop('prominence_indicator', None) # 使用 .pop() 安全地移除，如果键不存在也不会报错

        price_trough_params = trough_params.copy()
        if 'prominence' in price_trough_params:
            # 将价格的相对prominence转换为绝对值
            relative_prom = price_trough_params['prominence']
            price_trough_params['prominence'] = price_range * relative_prom
            if self.verbose_logging:
                print(f"    [调试-背离检测] 价格波谷相对prominence: {relative_prom}, 计算后绝对值: {price_trough_params['prominence']:.4f}")
        price_trough_params.pop('prominence_indicator', None) # 使用 .pop() 安全地移除

        # --- 准备指标查找参数 ---
        indicator_peak_params = peak_params.copy()
        if 'prominence_indicator' in indicator_peak_params:
            # 如果为指标指定了独立的prominence，则使用它
            indicator_peak_params['prominence'] = indicator_peak_params.pop('prominence_indicator')
        
        indicator_trough_params = trough_params.copy()
        if 'prominence_indicator' in indicator_trough_params:
            # 如果为指标指定了独立的prominence，则使用它
            indicator_trough_params['prominence'] = indicator_trough_params.pop('prominence_indicator')
        # ----------------------------------------------------------------

        # 步骤3: 使用修复后的参数执行波峰/波谷查找
        # --- 顶背离计算 ---
        price_peaks, _ = find_peaks(temp_df['price'], **price_peak_params)
        indicator_peaks, _ = find_peaks(temp_df['indicator'], **indicator_peak_params)
        
        temp_df['price_at_peak'] = np.nan
        temp_df.iloc[price_peaks, temp_df.columns.get_loc('price_at_peak')] = temp_df['price'].iloc[price_peaks]
        temp_df['indicator_at_peak'] = np.nan
        temp_df.iloc[indicator_peaks, temp_df.columns.get_loc('indicator_at_peak')] = temp_df['indicator'].iloc[indicator_peaks]

        temp_df['last_price_peak'] = temp_df['price_at_peak'].ffill()
        temp_df['last_indicator_peak'] = temp_df['indicator_at_peak'].ffill()

        price_higher_high = temp_df['price_at_peak'] > temp_df['last_price_peak'].shift(1)
        indicator_lower_high = temp_df['indicator_at_peak'] < temp_df['last_indicator_peak'].shift(1)
        top_divergence_signal = temp_df['price_at_peak'].notna() & temp_df['indicator_at_peak'].notna() & price_higher_high & indicator_lower_high

        # --- 底背离计算 ---
        price_troughs, _ = find_peaks(-temp_df['price'], **price_trough_params)
        indicator_troughs, _ = find_peaks(-temp_df['indicator'], **indicator_trough_params)

        temp_df['price_at_trough'] = np.nan
        temp_df.iloc[price_troughs, temp_df.columns.get_loc('price_at_trough')] = temp_df['price'].iloc[price_troughs]
        temp_df['indicator_at_trough'] = np.nan
        temp_df.iloc[indicator_troughs, temp_df.columns.get_loc('indicator_at_trough')] = temp_df['indicator'].iloc[indicator_troughs]

        temp_df['last_price_trough'] = temp_df['price_at_trough'].ffill()
        temp_df['last_indicator_trough'] = temp_df['indicator_at_trough'].ffill()

        price_lower_low = temp_df['price_at_trough'] < temp_df['last_price_trough'].shift(1)
        indicator_higher_low = temp_df['indicator_at_trough'] > temp_df['last_indicator_trough'].shift(1)
        bottom_divergence_signal = temp_df['price_at_trough'].notna() & temp_df['indicator_at_trough'].notna() & price_lower_low & indicator_higher_low
        
        # 步骤4: 返回结果，并确保索引与输入一致
        return top_divergence_signal.set_axis(price_series.index), bottom_divergence_signal.set_axis(price_series.index)
    
    # 寻找复合顶背离卖点
    def _find_top_divergence_exit(self, df: pd.DataFrame, params: dict) -> pd.Series:
        """【V5.0 参数适配版】调用新的辅助函数获取正确的指标周期。"""
        # 简化此函数，使其专注于调用和组合结果
        divergence_params = self._get_params_block(params, 'divergence_params')
        if not divergence_params.get('enabled', False): return pd.Series(False, index=df.index)

        # 动态获取指标列名 
        fe_params = params.get('feature_engineering_params', {})
        indicators_config = fe_params.get('indicators', {})
        macd_config = indicators_config.get('macd', {})
        macd_periods = self._get_periods_for_timeframe(macd_config, 'D')
        if not macd_periods:
            logger.warning("日线MACD周期未配置，跳过顶背离检测。")
            return pd.Series(False, index=df.index)
        fast, slow, signal_line = macd_periods
        macd_hist_col = f"MACDh_{fast}_{slow}_{signal_line}_D"
        
        rsi_config = indicators_config.get('rsi', {})
        rsi_periods_list = self._get_periods_for_timeframe(rsi_config, 'D')
        if not rsi_periods_list:
            logger.warning("日线RSI周期未配置，跳过顶背离检测。")
            return pd.Series(False, index=df.index)
        rsi_period = rsi_periods_list[0]
        rsi_col = f"RSI_{rsi_period}_D"

        if macd_hist_col not in df.columns or rsi_col not in df.columns:
            return pd.Series(False, index=df.index)

        # 定义find_peaks的参数
        find_peaks_params = {
            "distance": params.get('distance', 5),
            "prominence": params.get('prominence_top', 0.1)
        }
        # 底背离参数在此处不需要，传入空字典
        find_troughs_params = {}

        # 调用通用的向量化背离函数
        macd_top_div, _ = self._find_divergence(df['high_D'], df[macd_hist_col], find_peaks_params, find_troughs_params)
        rsi_top_div, _ = self._find_divergence(df['high_D'], df[rsi_col], find_peaks_params, find_troughs_params)

        # 组合信号
        return macd_top_div | rsi_top_div

    def _find_n_shape_relay_entry(self, df: pd.DataFrame, precondition: pd.Series, params: dict) -> pd.Series:
        """
        【剧本】【V1.2 涨停识别修正版】识别“N字板”接力。
        修正了对“涨停/大阳线”的识别逻辑，使其能正确捕捉包括一字板、T字板在内的所有强势形态。
        """
        params = self._get_params_block(params, 'n_shape_relay_params')
        if not params.get('enabled', False):
            return pd.Series(False, index=df.index)

        rally_threshold = params.get('rally_threshold', 0.097)
        consolidation_days_max = params.get('consolidation_days_max', 3)
        hold_above_key = params.get('hold_above_key', 'open_D')

        # 采用更鲁棒的“拉升日”识别逻辑，并保持向量化实现

        # 步骤1: 识别所有可能的“拉升日” (启动板和接力板)
        # 条件a: K线实体涨幅够大 (捕捉大阳线)
        body_rally = (df['close_D'] - df['open_D']) / df['open_D'].replace(0, np.nan)
        is_strong_body = body_rally > params.get('body_rally_threshold', 0.05) # 参数，默认为5%
        
        # 条件b: 日涨幅够大 (捕捉各类涨停板)
        close_to_close_rally = df['close_D'].pct_change()
        is_limit_up_rally = close_to_close_rally > rally_threshold

        # 最终的“拉升日”定义：满足任一条件即可
        is_rally_day = is_strong_body | is_limit_up_rally

        final_signal = pd.Series(False, index=df.index)

        # 步骤2: 遍历所有可能的整理周期 (1天到consolidation_days_max天)
        for days in range(1, consolidation_days_max + 1):
            # 假设今天(T)是接力板，那么启动板就在 T-days-1 日
            is_second_rally = is_rally_day
            is_first_rally = is_rally_day.shift(days + 1)

            # 获取启动板和整理期的相关数据
            first_rally_high = df['high_D'].shift(days + 1)
            first_rally_support = df[hold_above_key].shift(days + 1)
            first_rally_volume = df['volume_D'].shift(days + 1)

            # 步骤3: 验证整理期 (T-days 到 T-1)
            consolidation_lows = df['low_D'].shift(1).rolling(window=days).min()
            consolidation_volumes = df['volume_D'].shift(1).rolling(window=days).max()

            is_supported = consolidation_lows > first_rally_support
            is_volume_shrunk = consolidation_volumes < first_rally_volume
            
            # 步骤4: 验证接力板(今天)是否突破了启动板的高点
            is_breakout = df['close_D'] > first_rally_high

            # 步骤5: 组合所有条件
            current_signal = (
                is_first_rally & 
                is_supported & 
                is_volume_shrunk & 
                is_second_rally & 
                is_breakout
            )
            
            final_signal |= current_signal.fillna(False)

        return final_signal & precondition

    # 专门用于识别“回撤预备”信号
    def _find_pullback_setup(self, df: pd.DataFrame, precondition: pd.Series, params: dict) -> Tuple[pd.Series, pd.Series]:
        """
        【剧本】识别“回撤预备”信号，用于次日盘中监控。
        返回:
            - is_setup (pd.Series): 当天是否处于回撤预备状态的布尔序列。
            - target_price (pd.Series): 对应的目标回撤价位序列。
        """
        params = self._get_params_block(params, 'pullback_ma_entry_params')
        if not params.get('enabled', False):
            return pd.Series(False, index=df.index), pd.Series(np.nan, index=df.index)

        support_ma_col = f"EMA_{params.get('support_ma', 20)}_D"
        if support_ma_col not in df.columns:
            return pd.Series(False, index=df.index), pd.Series(np.nan, index=df.index)

        # 预备状态的定义：
        # 1. 收盘价在支撑均线之上
        is_above_support = df['close_D'] > df[support_ma_col]
        
        # 2. 收盘价距离支撑均线足够近
        proximity_threshold = params.get('setup_proximity_threshold', 0.03) # 配置项，例如3%
        is_close_to_support = (df['close_D'] / df[support_ma_col] - 1) < proximity_threshold

        # 最终的预备信号：满足大趋势 + 在支撑之上 + 距离支撑足够近
        is_setup = precondition & is_above_support & is_close_to_support
        
        # 目标价位就是当天的支撑均线价格，只在预备日有效
        target_price = df[support_ma_col].where(is_setup)

        return is_setup, target_price

    def _find_chip_cost_breakthrough(self, df: pd.DataFrame, precondition: pd.Series, params: dict) -> pd.Series:
        """【新剧本】【V2.1 列名修复版】筹码成本突破：股价上穿市场平均成本。"""
        params = self._get_params_block(params, 'chip_cost_breakthrough_params')
        close_col = 'close_D'
        weight_avg_col = 'weight_avg_D'
        if not params.get('enabled', False) or close_col not in df.columns or weight_avg_col not in df.columns:
            return pd.Series(False, index=df.index)
        
        signal = (df[close_col] > df[weight_avg_col]) & (df[close_col].shift(1) <= df[weight_avg_col].shift(1))

        return signal & precondition

    def _find_chip_pressure_release(self, df: pd.DataFrame, precondition: pd.Series, params: dict) -> pd.Series:
        """【新剧本】【V2.1 列名修复版】筹码压力释放：股价突破95%的套牢盘成本线。"""
        params = self._get_params_block(params, 'chip_pressure_release_params')
        close_col = 'close_D'
        cost_95pct_col = 'cost_95pct_D'
        if not params.get('enabled', False) or close_col not in df.columns or cost_95pct_col not in df.columns:
            return pd.Series(False, index=df.index)
            
        signal = df[close_col] > df[cost_95pct_col]

        return signal & precondition

    # ▼▼▼ 突破85%成本线 ▼▼▼
    def _find_chip_hurdle_clear_entry(self, df: pd.DataFrame, precondition: pd.Series, params: dict) -> pd.Series:
        """【新增剧本】筹码关口扫清：股价突破85%的套牢盘成本线，作为趋势确认信号。"""
        # 注意：这里我们复用 pressure_release 的参数块，或者你可以为其新建一个参数块
        params = self._get_params_block(params, 'chip_hurdle_clear_params') # 假设复用参数
        close_col = 'close_D'
        cost_85pct_col = 'cost_85pct_D'
        if not params.get('enabled', False) or close_col not in df.columns or cost_85pct_col not in df.columns:
            return pd.Series(False, index=df.index)
        
        signal = (df[close_col] > df[cost_85pct_col]) & (df[close_col].shift(1) <= df[cost_85pct_col].shift(1))

        return signal & precondition

    # ▼▼▼ “获利盘洗净反转”剧本 ▼▼▼
    def _find_winner_rate_reversal_entry(self, df: pd.DataFrame, params: dict) -> pd.Series:
        """
        【全新剧本】【V1.1 列名修复版】识别“获利盘洗净反转”信号 (或称“投降坑”反转)。
        当获利盘比例极低（市场绝望），随后出现企稳阳线时，视为底部反转信号。
        """
        # 从主配置中获取本剧本的参数块
        params = self._get_params_block(params, 'winner_rate_reversal_params')
        if not params.get('enabled', False):
            return pd.Series(False, index=df.index)
        # 检查必需的CYQ数据列是否存在
        winner_rate_col = 'winner_rate_D'
        close_col = 'close_D'
        open_col = 'open_D'
        required_cols = [winner_rate_col, close_col, open_col]
        if not all(col in df.columns and df[col].notna().any() for col in required_cols):
            if self.verbose_logging:
                print(f"    [调试-获利盘洗净反转]: 跳过，缺少必需的CYQ列: {required_cols}")
            return pd.Series(False, index=df.index)
        # 1. 条件A: 前一日获利盘比例极低（市场已完成“投降式”抛售）
        # 注意：winner_rate的单位是%，所以阈值也要是%
        was_washed_out = df[winner_rate_col].shift(1) < params.get('washout_threshold', 10.0)
        # 2. 条件B: 当日为企稳阳线
        is_reversal_candle = df['close_D'] > df['open_D']
        # 这个信号是典型的左侧交易信号，可以不依赖于严格的上升趋势前提(precondition)
        final_signal = was_washed_out & is_reversal_candle

        if self.verbose_logging and final_signal.any():
            print(f"    [调试-获利盘洗净反转]: 剧本触发 {final_signal.sum()} 次。")
        return final_signal.fillna(False)

    def _find_fibonacci_pullback_entry(self, df: pd.DataFrame, precondition: pd.Series, params: dict) -> pd.Series:
        """
        【剧本】【V2.0 动态行为版】“斐波那契引力区” - 动态识别并验证高质量的回踩。
        - 核心修正: 不再依赖外部预计算的斐波那契列，改为在函数内部动态寻找波峰波谷并计算。
        - 策略深化:
          1. 驱动浪验证: 只对“高涨幅、强趋势(ADX)”的有效驱动浪进行分析。
          2. 回踩质量验证: 要求回踩过程必须“缩量”，表明卖压衰竭。
          3. 主力行为确认: 在斐波那契支撑位企稳反弹的当天，必须由“主力资金净流入”确认。
        """
        fib_params = self._get_params_block(params, 'fibonacci_pullback_params')
        if not fib_params.get('enabled', False):
            return pd.Series(False, index=df.index)

        if self.verbose_logging:
            print("\n--- [调试-斐波那契V2.0-启动] ---")

        # --- 从配置中读取或使用默认值 ---
        peak_distance = fib_params.get('peak_distance', 10)
        peak_prominence = fib_params.get('peak_prominence', 0.05) # 波峰/谷的显著性(5%)
        impulse_min_rise_pct = fib_params.get('impulse_min_rise_pct', 0.25) # 驱动浪最小涨幅25%
        impulse_min_duration = fib_params.get('impulse_min_duration', 10) # 驱动浪最短持续10天
        impulse_min_adx = fib_params.get('impulse_min_adx', 20) # 驱动浪期间ADX均值需大于20
        fib_levels = fib_params.get('retracement_levels', [0.382, 0.5, 0.618])
        pullback_buffer = fib_params.get('pullback_buffer', 0.015) # 1.5%的缓冲带

        # --- 准备所需列名 ---
        adx_col = 'ADX_14_D'
        vol_ma_col = 'VOL_MA_21_D'
        required_cols = [
            'high_D', 'low_D', 'close_D', 'open_D', 'volume_D', adx_col, vol_ma_col, 'net_main_force_amount_D'
        ]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            if self.verbose_logging:
                print(f"    [调试-斐波那契-警告]: 缺少必需列: {missing_cols}，剧本跳过。")
            return pd.Series(False, index=df.index)

        # --- 步骤1: 动态识别所有显著的波峰和波谷 ---
        price_range = df['high_D'].max() - df['low_D'].min()
        if price_range == 0: return pd.Series(False, index=df.index) # 价格无波动则跳过
        
        # find_peaks 需要绝对显著性值，我们将其从相对值转换
        absolute_prominence = price_range * peak_prominence
        
        swing_high_indices, _ = find_peaks(df['high_D'], distance=peak_distance, prominence=absolute_prominence)
        swing_low_indices, _ = find_peaks(-df['low_D'], distance=peak_distance, prominence=absolute_prominence)

        if self.verbose_logging:
            print(f"    [调试] 发现 {len(swing_high_indices)} 个显著波峰, {len(swing_low_indices)} 个显著波谷。")

        if len(swing_high_indices) == 0 or len(swing_low_indices) == 0:
            return pd.Series(False, index=df.index)

        final_signal = pd.Series(False, index=df.index)

        # --- 步骤2: 遍历每个波峰，寻找并验证其前的驱动浪 ---
        for high_idx in swing_high_indices:
            # 找到此波峰之前最近的那个波谷
            possible_lows = swing_low_indices[swing_low_indices < high_idx]
            if len(possible_lows) == 0:
                continue
            low_idx = possible_lows[-1]

            # --- 步骤2a: 验证驱动浪 (从 low_idx 到 high_idx) 的质量 ---
            impulse_wave_df = df.iloc[low_idx:high_idx+1]
            duration = len(impulse_wave_df)
            rise_pct = (impulse_wave_df['high_D'].max() / impulse_wave_df['low_D'].min()) - 1
            avg_adx = impulse_wave_df[adx_col].mean()

            if not (duration >= impulse_min_duration and rise_pct >= impulse_min_rise_pct and avg_adx >= impulse_min_adx):
                continue # 如果驱动浪质量不达标，则跳过这个波峰

            if self.verbose_logging:
                start_date, end_date = df.index[low_idx].date(), df.index[high_idx].date()
                print(f"\n    [调试] 发现有效驱动浪: {start_date} -> {end_date} (涨幅: {rise_pct:.2%}, 持续: {duration}天, ADX: {avg_adx:.1f})")

            # --- 步骤2b: 计算此驱动浪的斐波那契回撤水平 ---
            swing_high_price = df['high_D'].iloc[high_idx]
            swing_low_price = df['low_D'].iloc[low_idx]
            retracement_range = swing_high_price - swing_low_price
            
            fib_support_levels = {
                level: swing_high_price - retracement_range * level for level in fib_levels
            }
            if self.verbose_logging:
                print(f"      -> 斐波那契支撑位: " + ", ".join([f"{lvl*100:.1f}%={price:.2f}" for lvl, price in fib_support_levels.items()]))

            # --- 步骤3: 在驱动浪之后寻找符合条件的回踩买点 ---
            # 只在驱动浪结束后的特定周期内寻找回踩，避免过时
            search_period_df = df.iloc[high_idx + 1 : high_idx + 1 + 60] # 最多往后找60天
            for date, row in search_period_df.iterrows():
                for level, support_price in fib_support_levels.items():
                    # 条件1: 价格触及斐波那契支撑位 (带缓冲)
                    if row['low_D'] <= support_price * (1 + pullback_buffer) and row['low_D'] >= support_price * (1 - pullback_buffer):
                        # 条件2: 回踩必须是缩量的
                        is_volume_shrinking = row['volume_D'] < row[vol_ma_col]
                        # 条件3: 企稳K线必须是阳线，且收盘价高于支撑位
                        is_reversal_candle = row['close_D'] > row['open_D'] and row['close_D'] > support_price
                        # 条件4: 主力资金必须净流入
                        is_main_force_buying = row['net_main_force_amount_D'] > 0

                        if is_volume_shrinking and is_reversal_candle and is_main_force_buying:
                            if self.verbose_logging:
                                print(f"      ★★ 信号触发! ★★ 日期: {date.date()}, 在 {level*100:.1f}% ({support_price:.2f}) 水平获得支撑。")
                                print(f"          -> 缩量: {is_volume_shrinking}, K线确认: {is_reversal_candle}, 主力买入: {is_main_force_buying}")
                            final_signal.loc[date] = True
                            # 找到一个有效回踩后，就跳出对此驱动浪的后续搜索
                            break 
                if final_signal.loc[date]:
                    break

        if self.verbose_logging:
            print(f"--- [调试-斐波那契V2.0-结束] 最终信号总数: {final_signal.sum()} ---")

        return final_signal.fillna(False) & precondition

    # ▼▼▼“均线加速上涨”剧本方法 ▼▼▼
    def _find_ma_acceleration_entry(self, df: pd.DataFrame, precondition: pd.Series, params: dict) -> pd.Series:
        """
        【剧本】【V2.0 动能共振版】“趋势引爆” - 识别多维度动能共振的爆发点。
        - 核心升级: 从单一的均线数学形态，深化为对价格、成交量、资金、趋势强度四维共振的捕捉。
        - 策略逻辑:
          1. 趋势引擎 (均线加速): 保留均线斜率和加速度为正的核心逻辑。
          2. 燃料供应 (成交量确认): 要求加速日成交量必须显著放大。
          3. 驾驶员 (主力资金确认): 要求加速日必须由主力资金净流入驱动。
          4. 路况检查 (趋势强度确认): 要求ADX指标显示当前处于明确的趋势行情中。
        """
        # 从JSON获取此剧本的专属配置
        params = self._get_params_block(params, 'ma_acceleration_playbook')
        if not params.get('enabled', False):
            return pd.Series(False, index=df.index)

        # --- 从配置中读取或使用默认值 ---
        ma_period = params.get('ma_period', 21)
        timeframe = params.get('timeframe', 'D')
        volume_surge_ratio = params.get('volume_surge_ratio', 1.5) # 成交量放大倍数
        min_adx_level = params.get('min_adx_level', 20) # ADX最低阈值

        # --- 准备所需列名 ---
        ema_col = f'EMA_{ma_period}_{timeframe}'
        close_col = f'close_{timeframe}'
        vol_ma_col = f'VOL_MA_21_{timeframe}' # 使用21日成交量均线
        adx_col = f'ADX_14_{timeframe}' # 使用14日ADX
        net_mf_col = f'net_main_force_amount_{timeframe}' # 主力净流入

        # 使用独立后缀避免与其他剧本的列名冲突
        slope_col = f'{ema_col}_slope_for_accel'
        accel_col = f'{ema_col}_acceleration'

        # --- 检查所有依赖列是否存在 ---
        required_cols = [ema_col, close_col, vol_ma_col, adx_col, net_mf_col, 'volume_D']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            if self.verbose_logging:
                print(f"    [调试-均线加速V2.0-警告]: 缺少必需列: {missing_cols}，剧本跳过。")
            return pd.Series(False, index=df.index)

        # --- 核心计算逻辑 ---
        # 1. 趋势引擎 (均线加速) - 保留原有逻辑
        df[slope_col] = df[ema_col].diff(1)
        df[accel_col] = df[slope_col].diff(1)
        is_ma_accelerating = (df[slope_col] > 0) & (df[accel_col] > 0)

        # 2. 燃料供应 (成交量确认)
        is_volume_surged = df['volume_D'] > (df[vol_ma_col] * volume_surge_ratio)

        # 3. 驾驶员 (主力资金确认)
        is_main_force_driving = df[net_mf_col] > 0

        # 4. 路况检查 (趋势强度确认)
        is_in_trend = df[adx_col] > min_adx_level
        
        # --- 组合最终信号：四维共振 ---
        final_signal = (
            is_ma_accelerating &
            is_volume_surged &
            is_main_force_driving &
            is_in_trend
        )

        # 最终信号必须满足外部传入的严格前提条件 (例如，周线多头)
        final_signal &= precondition

        if self.verbose_logging:
            print(f"    [调试-均线加速V2.0]: 前提满足: {precondition.sum()} | "
                  f"均线加速: {is_ma_accelerating.sum()} | "
                  f"放量: {is_volume_surged.sum()} | "
                  f"主力买入: {is_main_force_driving.sum()} | "
                  f"趋势确认(ADX): {is_in_trend.sum()} | "
                  f"最终信号: {final_signal.sum()}")

        # 清理临时列，保持DataFrame干净
        df.drop(columns=[slope_col, accel_col], inplace=True, errors='ignore')

        return final_signal.fillna(False)

    # 引入状态机，重构止盈逻辑以支持连续交易模拟
    def _apply_take_profit_rules(self, df: pd.DataFrame, entry_signals: pd.Series, top_divergence_signals: pd.Series, params: dict) -> pd.Series:
        """
        【V22.0 调试增强版】
        """
        final_tp_signal = pd.Series(0, index=df.index, dtype=int)
        tp_params = self._get_params_block(params, 'take_profit_params')
        
        print("    - [止盈-调试] 步骤1: 计算所有潜在止盈条件...")
        tp_signal_resistance = self._check_resistance_take_profit(df, tp_params.get('resistance_exit', {}))
        tp_signal_trailing = self._check_trailing_stop_take_profit(df, tp_params.get('trailing_stop_exit', {}))
        tp_signal_indicator = self._check_indicator_take_profit(df, tp_params.get('indicator_exit', {}))
        board_patterns = self._identify_board_patterns(df, params)
        tp_signal_heaven_earth = board_patterns.get('heaven_earth_board', pd.Series(False, index=df.index))
        tp_signal_upthrust = self._find_upthrust_distribution_exit(df, params)
        tp_signal_breakdown = self._find_volume_breakdown_exit(df, params)
        box_params = self._get_params_block(params, 'dynamic_box_params')
        tp_signal_box_breakdown = pd.Series(False, index=df.index)
        if box_params.get('breakdown_sell_enabled', True):
            tp_signal_box_breakdown = self.signals.get('dynamic_box_breakdown', pd.Series(False, index=df.index))

        fib_params = self._get_params_block(params, 'fibonacci_analysis_params')
        tp_signal_fib_extension = pd.Series(False, index=df.index)
        if fib_params.get('enabled', False):
            ext_levels = fib_params.get('extension_levels', [1.618])
            if ext_levels:
                first_target_level = ext_levels[0]
                col_name = f'fib_ext_{str(first_target_level).replace(".", "")}_D'
                if col_name in df.columns:
                    target_price = df[col_name]
                    hit_target = (df['high_D'] >= target_price) & target_price.notna()
                    tp_signal_fib_extension[hit_target] = True
        
        # ▼▼▼ 增加详细的止盈条件触发统计 ▼▼▼
        print("    - [止盈-调试] 原始止盈条件触发统计:")
        print(f"      - 阻力位(1): {tp_signal_resistance.sum()} 次")
        print(f"      - 移动止损(2): {tp_signal_trailing.sum()} 次")
        print(f"      - 指标超买(3/5): {(tp_signal_indicator > 0).sum()} 次")
        print(f"      - 顶背离(4): {top_divergence_signals.sum()} 次")
        print(f"      - 天地板(6): {tp_signal_heaven_earth.sum()} 次")
        print(f"      - 高位派发(7): {tp_signal_upthrust.sum()} 次")
        print(f"      - 放量破位(8): {tp_signal_breakdown.sum()} 次")
        print(f"      - 箱体跌破(9): {tp_signal_box_breakdown.sum()} 次")
        print(f"      - 斐波那契扩展(10): {tp_signal_fib_extension.sum()} 次")

        any_tp_signal = (
            tp_signal_fib_extension | tp_signal_breakdown | tp_signal_heaven_earth |
            tp_signal_upthrust | tp_signal_box_breakdown | top_divergence_signals |
            (tp_signal_indicator > 0) | tp_signal_trailing | tp_signal_resistance
        )

        print("    - [止盈-调试] 步骤2: 使用向量化状态机判断持仓...")
        is_new_entry = entry_signals & ~any_tp_signal.shift(1).fillna(False)
        is_exit = any_tp_signal
        trade_block_id = is_new_entry.cumsum()
        first_exit_in_block = is_exit & (is_exit.groupby(trade_block_id).cumcount() == 0)
        reset_points = first_exit_in_block.cumsum()
        active_trade_block_id = trade_block_id - reset_points.shift(1).fillna(0)
        is_holding_vectorized = active_trade_block_id > 0
        print(f"    - [止盈-调试] 识别出 {is_new_entry.sum()} 个交易区块，最终持仓天数共 {is_holding_vectorized.sum()} 天。")

        print("    - [止盈-调试] 步骤3: 应用优先级规则生成最终止盈信号...")
        final_tp_signal.loc[is_holding_vectorized & tp_signal_fib_extension] = 10
        final_tp_signal.loc[(final_tp_signal == 0) & is_holding_vectorized & tp_signal_breakdown] = 8
        final_tp_signal.loc[(final_tp_signal == 0) & is_holding_vectorized & tp_signal_heaven_earth] = 6
        final_tp_signal.loc[(final_tp_signal == 0) & is_holding_vectorized & tp_signal_upthrust] = 7
        final_tp_signal.loc[(final_tp_signal == 0) & is_holding_vectorized & tp_signal_box_breakdown] = 9
        final_tp_signal.loc[(final_tp_signal == 0) & is_holding_vectorized & top_divergence_signals] = 4
        indicator_tp_mask = (final_tp_signal == 0) & is_holding_vectorized & (tp_signal_indicator > 0)
        final_tp_signal.loc[indicator_tp_mask] = tp_signal_indicator[indicator_tp_mask]
        final_tp_signal.loc[(final_tp_signal == 0) & is_holding_vectorized & tp_signal_trailing] = 2
        final_tp_signal.loc[(final_tp_signal == 0) & is_holding_vectorized & tp_signal_resistance] = 1

        print(f"    - [止盈-调试] 止盈逻辑判断完成，共产生 {(final_tp_signal > 0).sum()} 个最终止盈信号。")
        return final_tp_signal

    def _check_resistance_take_profit(self, df: pd.DataFrame, params: dict) -> pd.Series:
        if not params.get('enabled', False): return pd.Series(False, index=df.index)
        resistance_level = df['high_D'].shift(1).rolling(window=params.get('lookback_period', 90)).max()
        signal = df['high_D'] >= resistance_level * (1 - params.get('approach_threshold', 0.02))
        return signal & resistance_level.notna()

    def _check_trailing_stop_take_profit(self, df: pd.DataFrame, params: dict) -> pd.Series:
        if not params.get('enabled', False): return pd.Series(False, index=df.index)
        highest_close_since = df['close_D'].shift(1).rolling(window=params.get('lookback_period', 20)).max()
        stop_price = highest_close_since * (1 - params.get('percentage_pullback', 0.10))
        signal = df['close_D'] < stop_price
        return signal & highest_close_since.notna()

    def _check_indicator_take_profit(self, df: pd.DataFrame, params: dict) -> pd.Series:
        """【V2.10 优化】支持RSI和BIAS两种指标止盈。"""
        if not params.get('enabled', False): return pd.Series(0, index=df.index)
        
        # 初始化返回的信号序列，0表示无信号
        final_signal = pd.Series(0, index=df.index)

        # 检查RSI止盈
        if params.get('use_rsi_exit', False):
            rsi_col = f"RSI_{params.get('rsi_period', 14)}_D"
            if rsi_col in df.columns:
                rsi_signal = df[rsi_col] > params.get('rsi_threshold', 80)
                final_signal[rsi_signal] = 3 # 3 代表RSI超买止盈
        
        # 检查BIAS止盈
        if params.get('use_bias_exit', False):
            bias_col = f"BIAS_{params.get('bias_period', 20)}_D"
            if bias_col in df.columns:
                bias_signal = df[bias_col] > params.get('bias_threshold', 20.0)
                # BIAS信号优先级更高，可以覆盖RSI信号
                final_signal[bias_signal] = 5 # 5 代表BIAS超买止盈

        return final_signal

    def _check_fund_flow_confirmation(self, df: pd.DataFrame, params: dict) -> pd.Series:
        """【新状态】资金流确认：当天主力资金为净流入。"""
        params = self._get_params_block(params, 'fund_flow_confirm_params')
        if not params.get('enabled', False) or 'fund_buy_lg_amount_D' not in df.columns:
            return pd.Series(False, index=df.index)
        
        threshold = params.get('threshold', 0) # 阈值，可以设置为0，代表只要是净流入就算
        # 状态：当日大单净额 > 阈值，并且5日主力净额也是正的，代表趋势性流入
        is_confirmed = (df['fund_buy_lg_amount_D'] > threshold) & (df.get('fund_net_d5_amount_D', 0) > 0)
        return is_confirmed

    def _check_vwap_confirmation(self, df_dict: Dict[str, pd.DataFrame], params: dict) -> pd.Series:
        """
        【V2.5 最终修复版】检查VWAP支撑确认信号。
        """
        # print("    [VWAP确认] 正在执行分钟线VWAP确认逻辑 (V2.5 最终修复版)...") # 可以注释掉常规日志
        vwap_params = self._get_params_block(params, 'vwap_confirmation_params')
        if not vwap_params.get('enabled', False):
            if 'D' in df_dict and not df_dict['D'].empty:
                return pd.Series(False, index=df_dict['D'].index)
            return pd.Series([])

        tf = vwap_params.get('timeframe', '5')
        buffer = vwap_params.get('confirmation_buffer', 0.001)

        df_minute = df_dict.get(tf)
        df_daily = df_dict.get('D')

        if df_minute is None or df_minute.empty or df_daily is None or df_daily.empty:
             return pd.Series(False, index=df_daily.index if df_daily is not None else None)

        # 检查VWAP列是否存在
        vwap_col = f'VWAP_{tf}'
        if vwap_col not in df_minute.columns: vwap_col = 'VWAP_D' # 兼容旧列名
        if vwap_col not in df_minute.columns: return pd.Series(False, index=df_daily.index)

        # 核心逻辑：检查分钟线是否在VWAP之上，然后按天聚合
        # .normalize() 对UTC时区索引同样有效，会将其归一化到当天的零点
        is_above_vwap = df_minute['close'] > df_minute[vwap_col] * (1 + buffer)
        daily_confirmation = is_above_vwap.groupby(is_above_vwap.index.normalize()).any()
        
        # 将聚合后的日线信号映射回原始的日线DataFrame
        final_signal = pd.Series(False, index=df_daily.index)
        
        if not daily_confirmation.empty:
            # 使用 .reindex() 是一种更健壮和简洁的映射方式
            # 它会根据 df_daily 的归一化日期索引，从 daily_confirmation 中查找对应的值
            # 找不到的日期会自动填充为 fill_value (默认为NaN，我们用 .fillna(False) 处理)
            normalized_daily_index = df_daily.index.normalize()
            final_signal = daily_confirmation.reindex(normalized_daily_index).fillna(False)
            # reindex 后索引会变成归一化的日期，需要把它重置回原始的日线索引
            final_signal.index = df_daily.index
        # print(f"    [VWAP确认] 完成，共产生 {final_signal.sum()} 个VWAP支撑信号。") # 可以注释掉常规日志
        return final_signal




