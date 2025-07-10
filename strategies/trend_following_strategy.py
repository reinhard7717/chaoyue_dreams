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

    def _get_params_block(self, params: dict, block_name: str, default_return: Any = None) -> dict:
        # 修改行: 增加一个默认返回值，使调用更安全
        if default_return is None:
            default_return = {}
        return params.get('strategy_params', {}).get('trend_follow', {}).get(block_name, default_return)

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
        【V57.0 总指挥版】
        - 核心架构: 严格遵循“数据流驱动”思想，按顺序调用三大核心引擎，完成信号生成。
          1. 数据预处理 (Preprocessing)
          2. 状态评审 (Setup Scoring)
          3. 事件定义 (Trigger Definition)
          4. 计分总装 (Final Scoring)
          5. 出场决策 (Exit Logic)
        """
        print("\n" + "="*60)
        print(f"====== 日期: {df.index[-1].date()} | 开始执行【战术引擎 V57.0 总指挥版】 ======")
        
        # --- 步骤 0: 输入验证与数据预处理 ---
        if df is None or df.empty:
            print("    - [错误] 传入的DataFrame为空，战术引擎终止。")
            return pd.DataFrame(), {}
        
        # 确保所有列都是数值类型，避免后续计算出错
        df = self._ensure_numeric_types(df)
        
        # 统一列名后缀，例如 'close' -> 'close_D'
        timeframe_suffixes = ['_D', '_W', '_M', '_5', '_15', '_30', '_60']
        rename_map = {
            col: f"{col}_D" for col in df.columns 
            if not any(col.endswith(suffix) for suffix in timeframe_suffixes) 
            and not col.startswith(('VWAP_', 'BASE_', 'playbook_', 'signal_', 'kline_', 'context_', 'cond_'))
        }
        if rename_map:
            df = df.rename(columns=rename_map)
        
        # 确保核心列存在
        if 'close_D' in df.columns:
            df['pct_change_D'] = df['close_D'].pct_change()
        
        # --- 步骤 1: 核心数据引擎 ---
        print("--- [总指挥] 步骤1: 核心数据引擎启动 ---")
        # 1.1 计算基础衍生特征
        df = self._prepare_derived_features(df, params)
        # 1.2 计算所有需要的斜率和加速度
        df = self._calculate_trend_slopes(df, params)
        # 1.3 (可选) 运行独立的K线形态识别
        df = self.pattern_recognizer.identify_all(df)

        # --- 步骤 2: 准备状态评审 (米其林评审官) ---
        print("--- [总指挥] 步骤2: 准备状态评审引擎启动 ---")
        # 输入是处理好的df，输出是带有置信度分数的字典
        setup_scores = self._calculate_setup_conditions(df, params, {}, {}) # 后两个参数在新架构下已无用

        # --- 步骤 3: 触发事件定义 (事件定义中心) ---
        print("--- [总指挥] 步骤3: 触发事件定义引擎启动 ---")
        # 输入是处理好的df，输出是布尔信号字典
        trigger_events = self._define_trigger_events(df, params, {}) # 最后一个参数在新架构下已无用

        # --- 步骤 4: 计分总装 (最终计分引擎) ---
        print("--- [总指挥] 步骤4: 最终计分引擎启动 ---")
        # 输入是 setup_scores 和 trigger_events，输出是最终得分
        df, score_details_df = self._calculate_entry_score(df, params, trigger_events, setup_scores)
        self._last_score_details_df = score_details_df # 保存得分详情用于后续分析

        # --- 步骤 5: 出场决策  ---
        print("--- [总指挥] 步骤5: 智能风险评审与出场决策引擎启动 ---")
        # 5.1 诊断所有独立的风险因子
        risk_factors = self._diagnose_risk_factors(df, params)
        # 5.2 根据风险矩阵，计算每日的综合风险分数
        risk_score = self._calculate_risk_score(df, params, risk_factors)
        # 5.3 根据风险分数和阈值，生成最终的出场代码
        df['exit_signal_code'] = self._calculate_exit_signals(df, params, risk_score)

        # --- 步骤 6: 最终信号合成与日志输出 ---
        print("--- [总指挥] 步骤6: 最终信号合成与日志输出 ---")
        entry_scoring_params = self._get_params_block(params, 'entry_scoring_params')
        score_threshold = self._get_param_value(entry_scoring_params.get('score_threshold'), 100)
        df['signal_entry'] = df['entry_score'] >= score_threshold

        # 日志输出逻辑 (保持不变或根据新结构优化)
        # ... (省略日志输出代码) ...

        print(f"====== 【战术引擎 V57.0】执行完毕 ======")
        print("="*60 + "\n")

        # 返回最终的df和空的atomic_signals（因为新架构不再需要它）
        return df, {}

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
        【V58.0 归一化与动态任务版】
        - 核心升级:
          1. 引入归一化斜率 (Normalized Slope): 将斜率除以其自身的滚动标准差，使其数值可跨指标比较。
          2. 动态任务注入: 自动发现上游模块生成的新特征（如 'net_mf_zscore_D'）并加入计算任务。
          3. 命名规范化: 统一输出列名格式，如 `SLOPE_10_close_D`, `ACCEL_10_close_D`, `SLOPE_NORM_10_close_D`。
        """
        print("    - [斜率中心 V58.0 归一化与动态任务版] 启动...")
        slope_params = self._get_params_block(params, 'slope_params', {})
        if not self._get_param_value(slope_params.get('enabled'), False):
            print("      -> 斜率计算被禁用，跳过。")
            return df

        # --- 步骤 1: 准备斜率计算任务列表 ---
        # 从配置中获取基础任务
        series_to_slope = self._get_param_value(slope_params.get('series_to_slope'), {})
        
        # 【动态任务注入】自动发现上游模块生成的新特征并加入任务
        auto_detect_patterns = self._get_param_value(slope_params.get('auto_detect_patterns'), [])
        auto_detect_lookbacks = self._get_param_value(slope_params.get('auto_detect_default_lookbacks'), [10, 20])
        
        for pattern in auto_detect_patterns:
            for col in df.columns:
                if pattern in col and col not in series_to_slope:
                    series_to_slope[col] = auto_detect_lookbacks
                    print(f"      -> [动态注入] 发现新特征 '{col}'，已加入斜率计算任务。")

        # --- 步骤 2: 遍历任务列表，执行计算 ---
        for col_name, lookbacks in series_to_slope.items():
            if col_name not in df.columns:
                print(f"      -> [警告] 列 '{col_name}' 不存在，跳过斜率计算。")
                continue
            
            # 确保源数据是浮点数类型
            source_series = df[col_name].astype(float)

            for lookback in lookbacks:
                min_p = max(2, lookback // 2)
                
                # --- 2.1 计算一阶斜率 (速度) ---
                slope_col_name = f'SLOPE_{lookback}_{col_name}'
                linreg_result = df.ta.linreg(close=source_series, length=lookback, min_periods=min_p, slope=True, intercept=False, r=False)
                
                # 智能处理 pandas_ta 的返回值 (保持健壮性)
                if isinstance(linreg_result, pd.DataFrame):
                    slope_series = linreg_result.iloc[:, 0] # 取第一列作为斜率
                elif isinstance(linreg_result, pd.Series):
                    slope_series = linreg_result
                else:
                    slope_series = pd.Series(np.nan, index=df.index)
                
                df[slope_col_name] = slope_series.fillna(0)

                # --- 2.2 计算二阶斜率 (加速度) ---
                accel_col_name = f'ACCEL_{lookback}_{col_name}'
                if not df[slope_col_name].dropna().empty:
                    accel_linreg_result = df.ta.linreg(close=df[slope_col_name], length=lookback, min_periods=min_p, slope=True, intercept=False, r=False)
                    if isinstance(accel_linreg_result, pd.DataFrame):
                        accel_series = accel_linreg_result.iloc[:, 0]
                    elif isinstance(accel_linreg_result, pd.Series):
                        accel_series = accel_linreg_result
                    else:
                        accel_series = pd.Series(np.nan, index=df.index)
                    df[accel_col_name] = accel_series.fillna(0)
                else:
                    df[accel_col_name] = np.nan

                # --- 2.3 【新增】计算归一化斜率 (相对强度) ---
                norm_slope_col_name = f'SLOPE_NORM_{lookback}_{col_name}'
                # 计算斜率自身的滚动标准差，用于归一化
                slope_std = df[slope_col_name].rolling(window=lookback * 2).std()
                # 归一化斜率 = 原始斜率 / 滚动标准差
                df[norm_slope_col_name] = np.divide(df[slope_col_name], slope_std, out=np.zeros_like(df[slope_col_name], dtype=float), where=slope_std!=0)
                
            print(f"        -> 完成对 '{col_name}' 的所有斜率计算 (周期: {lookbacks})。")

        print("    - [斜率中心 V58.0] 所有斜率计算完成。")
        return df

    # ▼▼▼ 剧本定义中心 (Playbook Definition Center) ▼▼▼
    def _get_playbook_definitions(self, df: pd.DataFrame, trigger_events: Dict[str, pd.Series], setup_conditions: Dict[str, pd.Series]) -> List[Dict]:
        """
        【V57.0 动态分级版 - 菜单设计师】
        - 核心思想: 废除布尔型setup，完全基于上游传入的“置信度分数”来动态定义和分级剧本。
        - 职责:
          1. 获取各个准备状态的置信度分数。
          2. 为同一个交易概念（如潜龙出海）定义不同分数区间的剧本（S+/S/S-）。
          3. 应用全局风险过滤。
        """
        print("    - [剧本定义中心 V57.0 动态分级版] 启动...")
        
        # --- 步骤 1: 初始化和获取上下文 ---
        # 创建一个全为False的Series作为默认值，防止KeyError
        default_series = pd.Series(False, index=df.index)
        # 获取右侧交易的核心前提条件
        robust_right_side_precondition = df.get('robust_right_side_precondition', pd.Series(True, index=df.index))
        
        # --- 步骤 2: 获取所有准备状态的“置信度分数” ---
        # setup_conditions 现在是一个包含分数的字典, e.g., {'SETUP_SCORE_DEEP_ACCUMULATION': pd.Series([...])}
        score_deep_accum = setup_conditions.get('SETUP_SCORE_DEEP_ACCUMULATION', pd.Series(0, index=df.index))
        score_cap_pit = setup_conditions.get('SETUP_SCORE_CAPITULATION_PIT', pd.Series(0, index=df.index))
        score_healthy_markup = setup_conditions.get('SETUP_SCORE_HEALTHY_MARKUP', pd.Series(0, index=df.index))
        score_energy_comp = setup_conditions.get('SETUP_SCORE_ENERGY_COMPRESSION', pd.Series(0, index=df.index))
        score_washout_reversal = setup_conditions.get('SETUP_SCORE_WASHOUT_REVERSAL', pd.Series(0, index=df.index))
        score_nshape_cont = setup_conditions.get('SETUP_SCORE_N_SHAPE_CONTINUATION', pd.Series(0, index=df.index))
        score_gap_support = setup_conditions.get('SETUP_SCORE_GAP_SUPPORT_PULLBACK', pd.Series(0, index=df.index))
        
        # 获取全局风险分数，用于一票否决
        # 假设派发期的分数是一个负值，只要大于0就认为有风险
        # 或者，我们可以直接使用布尔型的 'SETUP_DISTRIBUTION_RISK'
        is_in_distribution_risk = setup_conditions.get('SETUP_DISTRIBUTION_RISK', pd.Series(False, index=df.index))
        
        # --- 步骤 3: 定义所有交易剧本 (Playbooks) ---
        playbook_definitions = [
            # =================================================================================
            # === 概念 1: 潜龙出海 (Perfect Storm) - 基于深度吸筹分数动态分级 ===
            # =================================================================================
            {
                'name': 'PERFECT_STORM_S_PLUS',
                'cn_name': '【S+级】潜龙出海',
                # 准备条件: 深度吸筹置信度 > 120 (例如: 必须项全满足 + 至少两个顶级加分项)
                'setup': score_deep_accum > 120,
                'trigger': trigger_events.get('CHIP_EVENT_IGNITION', default_series),
                'score': 400, # 给予最高的基础分
                'precondition': True, # 左侧转右侧的临界点，放宽前提
                'comment': 'S+级: 筹码、均线、资金、波动率四维共振后的首次点火，确定性极高。'
            },
            {
                'name': 'PERFECT_STORM_S',
                'cn_name': '【S级】潜龙出海',
                # 准备条件: 80 < 置信度 <= 120 (例如: 必须项全满足 + 一个关键加分项)
                'setup': (score_deep_accum > 80) & (score_deep_accum <= 120),
                'trigger': trigger_events.get('CHIP_EVENT_IGNITION', default_series),
                'score': 350,
                'precondition': True,
                'comment': 'S级: 核心条件具备，多重验证下的标准启动信号。'
            },
            {
                'name': 'PERFECT_STORM_A_PLUS',
                'cn_name': '【A+级】潜龙出海',
                # 准备条件: 50 < 置信度 <= 80 (例如: 仅满足必须项)
                'setup': (score_deep_accum > 50) & (score_deep_accum <= 80),
                'trigger': trigger_events.get('CHIP_EVENT_IGNITION', default_series),
                'score': 280,
                'precondition': True,
                'comment': 'A+级: 满足深度吸筹的核心定义，但缺乏额外共振确认，值得关注。'
            },

            # =================================================================================
            # === 概念 2: 投降坑反转 (Capitulation Reversal) - 基于投降坑分数动态分级 ===
            # =================================================================================
            {
                'name': 'PIT_REVERSAL_A_PLUS',
                'cn_name': '【A+级】投降坑反转',
                # 准备条件: 投降坑置信度 > 80 (例如: 必须项满足 + 动能企稳加分)
                'setup': score_cap_pit > 80,
                'trigger': trigger_events.get('TRIGGER_REVERSAL_CONFIRMATION_CANDLE', default_series),
                'score': 290,
                'precondition': True, # 左侧信号，放宽前提
                'comment': 'A+级: 在市场恐慌形成的“投降坑”中，出现日线级别动能企稳后的反转K线，可靠性高。'
            },
            {
                'name': 'PIT_REVERSAL_A',
                'cn_name': '【A级】投降坑反转',
                # 准备条件: 50 < 置信度 <= 80 (例如: 仅满足必须项)
                'setup': (score_cap_pit > 50) & (score_cap_pit <= 80),
                'trigger': trigger_events.get('TRIGGER_REVERSAL_CONFIRMATION_CANDLE', default_series),
                'score': 220,
                'precondition': True,
                'comment': 'A级: 出现投降坑，但动能尚未完全企稳，属于高风险高赔率的左侧博弈。'
            },

            # =================================================================================
            # === 概念 3: 趋势中继 (Trend Continuation) - 基于健康主升浪分数动态分级 ===
            # =================================================================================
            {
                'name': 'TREND_CONTINUATION_A',
                'cn_name': '【A级】趋势中继',
                # 准备条件: 健康主升浪置信度 > 80 (例如: 必须项全满足)
                'setup': score_healthy_markup > 80,
                'trigger': trigger_events.get('TRIGGER_PULLBACK_REBOUND', default_series),
                'score': 240,
                'precondition': robust_right_side_precondition,
                'comment': 'A级: 在通过多维度验证的健康主升浪中，出现的回踩反弹，是可靠的加仓或上车点。'
            },
            {
                'name': 'TREND_CONTINUATION_B_PLUS',
                'cn_name': '【B+级】趋势中继',
                # 准备条件: 50 < 置信度 <= 80 (例如: 满足部分必须项，但可能缺少资金确认)
                'setup': (score_healthy_markup > 50) & (score_healthy_markup <= 80),
                'trigger': trigger_events.get('TRIGGER_PULLBACK_REBOUND', default_series),
                'score': 180,
                'precondition': robust_right_side_precondition,
                'comment': 'B+级: 趋势尚可，但某些维度存在瑕疵，属于机会主义的趋势跟踪。'
            },

            # =================================================================================
            # === 概念 4: 能量释放 (Energy Release) - 基于能量压缩分数 ===
            # =================================================================================
            {
                'name': 'ENERGY_RELEASE_A',
                'cn_name': '【A级】能量释放',
                # 准备条件: 能量压缩置信度 > 60
                'setup': score_energy_comp > 60,
                'trigger': trigger_events.get('TRIGGER_ENERGY_RELEASE', default_series),
                'score': 230,
                'precondition': robust_right_side_precondition,
                'comment': 'A级: 在波动率和筹码双重压缩后的能量释放，突破成功率较高。'
            },

            # =================================================================================
            # === 【新增】概念 5: 巨阴洗盘反转 (Washout Reversal) ===
            # =================================================================================
            {
                'name': 'WASHOUT_REVERSAL_A',
                'cn_name': '【A级】巨阴洗盘反转',
                'setup': score_washout_reversal > 70, # 要求满足必须项和至少一个加分项
                'trigger': trigger_events.get('TRIGGER_REVERSAL_CONFIRMATION_CANDLE', default_series),
                'score': 260,
                'precondition': True, # 左侧信号，放宽前提
                'comment': 'A级: 在恐慌性的巨量阴线后，出现企稳反转信号，通常是主力极端洗盘后的拉升前兆。'
            },

            # =================================================================================
            # === 【新增】概念 6: N字板接力 (N-Shape Continuation) ===
            # =================================================================================
            {
                'name': 'N_SHAPE_CONTINUATION_A',
                'cn_name': '【A级】N字板接力',
                'setup': score_nshape_cont > 80,
                'trigger': trigger_events.get('TRIGGER_VOLUME_SPIKE_BREAKOUT', default_series), # 使用放量突破作为触发器
                'score': 250,
                'precondition': robust_right_side_precondition,
                'comment': 'A级: 强势股在涨停或大阳线后，经过短暂、强势的整理，再次放量突破，是经典的趋势中继信号。'
            },

            # =================================================================================
            # === 【新增】概念 7: 缺口支撑回踩 (Gap Support Pullback) ===
            # =================================================================================
            {
                'name': 'GAP_SUPPORT_PULLBACK_B_PLUS',
                'cn_name': '【B+级】缺口支撑回踩',
                'setup': score_gap_support > 60,
                'trigger': trigger_events.get('TRIGGER_PULLBACK_REBOUND', default_series), # 使用回踩反弹作为触发器
                'score': 190,
                'precondition': robust_right_side_precondition,
                'comment': 'B+级: 股价回踩到前期跳空缺口获得支撑并反弹，是可靠的右侧交易机会。'
            },
            
            # =================================================================================
            # === 独立剧本: 不依赖于复合状态，但仍受全局风控影响 ===
            # =================================================================================
            {
                'name': 'EARTH_HEAVEN_BOARD', 'cn_name': '【S+】地天板',
                'setup': True, # 纯事件驱动，不依赖任何复合状态
                'trigger': trigger_events.get('TRIGGER_EARTH_HEAVEN_BOARD', default_series),
                'score': 380, 'precondition': True,
                'comment': '市场情绪的极致反转，拥有最高优先级。'
            },
        ]

        # --- 步骤 4: 全局风险过滤 ---
        # 这是一个强大的风控层，确保任何剧本都不会在明确的派发风险区触发
        final_playbook_list = []
        
        for playbook in playbook_definitions:
            # 复制一份以避免修改原始列表
            modified_playbook = playbook.copy()
            
            # 获取原始的setup条件
            original_setup = modified_playbook.get('setup', True)
            
            # 将全局风控条件与原始setup条件用“与”逻辑连接
            # 只有在“非派发风险区”时，原始的setup才可能为True
            modified_playbook['setup'] = original_setup & ~is_in_distribution_risk
            
            final_playbook_list.append(modified_playbook)
            
        print(f"    - [剧本定义中心 V57.0] 完成，共定义 {len(final_playbook_list)} 个动态分级剧本。")
        return final_playbook_list

    # ▼▼▼ 最终计分引擎 (Final Scoring Engine) ▼▼▼
    def _calculate_entry_score(
        self, 
        df: pd.DataFrame, 
        params: dict, 
        trigger_events: Dict[str, pd.Series], 
        # ▼▼▼【代码修改】: 输入从 setup_conditions 变为 setup_scores ▼▼▼
        setup_scores: Dict[str, pd.Series]
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        【V57.0 最终计分引擎】
        - 核心思想: 作为策略的总装配线，将“准备状态的置信度分数”与“触发事件”组合成最终的交易信号和得分。
        - 职责:
          1. 调用 _get_playbook_definitions 获取所有剧本的定义。
          2. 遍历剧本，检查“准备状态分数”是否达到剧本要求的最低门槛。
          3. 如果“准备状态”和“触发事件”都满足，则计算最终得分，得分可以与准备状态的质量挂钩。
        """
        print("    - [计分引擎 V57.0] 启动，开始装配剧本并计算最终得分...")
        
        # --- 步骤 1: 初始化 ---
        final_score = pd.Series(0.0, index=df.index)
        # score_details_df 用于记录每个剧本的贡献分数，便于调试
        score_details_df = pd.DataFrame(index=df.index)

        # --- 步骤 2: 获取所有剧本定义 ---
        # _get_playbook_definitions 现在是剧本的“蓝图中心”
        playbook_definitions = self._get_playbook_definitions(df, trigger_events, setup_scores)
        
        # --- 步骤 3: 遍历所有剧本蓝图，进行装配和计分 ---
        for playbook in playbook_definitions:
            name = playbook['name']
            cn_name = playbook.get('cn_name', name)
            
            # --- 3.1: 获取剧本的“准备状态”要求 ---
            # 准备状态的来源，例如 'SETUP_SCORE_DEEP_ACCUMULATION'
            setup_score_source = playbook.get('setup_score_source')
            # 剧本要求的最低置信度分数
            min_setup_score = playbook.get('min_setup_score', 0)
            
            # --- 3.2: 获取剧本的“触发事件” ---
            trigger_signal = playbook.get('trigger', pd.Series(False, index=df.index))

            # 如果没有定义准备状态来源或触发器，则跳过此剧本
            if setup_score_source is None or not trigger_signal.any():
                continue

            # 从传入的 setup_scores 字典中获取对应的分数Series
            setup_score_series = setup_scores.get(setup_score_source, pd.Series(0, index=df.index))
            
            # --- 3.3: 判断剧本是否成立 ---
           # 直接使用 playbook 中已经计算好的 setup 布尔信号
            is_setup_valid = playbook.get('setup', pd.Series(False, index=df.index))
            trigger_signal = playbook.get('trigger', pd.Series(False, index=df.index))
            
            # 最终剧本信号 = 准备状态成立 AND 触发事件发生
            playbook_signal = is_setup_valid & trigger_signal
            
            if playbook_signal.any():
                # --- 3.4: 计算剧本得分 ---
                # 剧本的基础分
                base_score = playbook.get('score', 0)
                # 准备状态的质量加成：分数越高，加成越多
                quality_bonus = setup_score_series * playbook.get('quality_multiplier', 0.1)
                
                # 计算该剧本在触发日的总得分
                current_playbook_score = base_score + quality_bonus
                
                # 将该剧本的得分累加到最终总分上
                final_score.loc[playbook_signal] += current_playbook_score.loc[playbook_signal]
                
                # 记录得分详情
                score_details_df.loc[playbook_signal, name] = current_playbook_score.loc[playbook_signal]
                print(f"      -> 剧本 '{cn_name}' 触发了 {playbook_signal.sum()} 天，贡献分数约: {base_score:.0f} + 质量加成")

        # --- 步骤 4: 最终处理和返回 ---
        df['entry_score'] = final_score.round(0)
        score_details_df.fillna(0, inplace=True)
        
        print(f"--- [计分引擎 V57.0] 计算完成。最终有 { (final_score > 0).sum() } 个交易日产生得分。 ---")
        
        # 返回更新后的df和得分详情，注意函数签名和返回值的变化
        return df, score_details_df

    # 风险评分引擎
    def _calculate_risk_score(self, df: pd.DataFrame, params: dict, risk_factors: Dict[str, pd.Series]) -> pd.Series:
        """
        【V57.0 风险评分引擎】
        - 核心功能: 基于“风险置信度矩阵”，将所有原子风险因子累加成一个量化的“每日风险分数”。
        - 思想: 风险不是开关，而是温度计。分数越高，风险越大。
        - 输出: 一个代表每日风险程度的数值型 pd.Series。
        """
        print("    - [风险评分引擎 V57.0] 启动，开始量化每日风险...")
        
        # 初始化风险分数为0
        risk_score = pd.Series(0.0, index=df.index)
        
        # 从配置中加载风险评分矩阵
        risk_matrix = self._get_params_block(params, 'exit_risk_scoring_matrix', {})
        if not risk_matrix:
            print("      -> [警告] 未找到风险评分矩阵，风险分数为0。")
            return risk_score

        # 遍历评分矩阵，累加风险分数
        for factor_name, points in risk_matrix.items():
            # 获取对应的风险因子布尔信号
            factor_signal = risk_factors.get(factor_name, pd.Series(False, index=df.index))
            
            # 如果风险因子被触发，则加上对应的分数
            if factor_signal.any():
                score_to_add = self._get_param_value(points, 0)
                risk_score.loc[factor_signal] += score_to_add
                print(f"      -> 风险因子 '{factor_name}' 触发，风险分 +{score_to_add}")

        df['risk_score'] = risk_score # 将风险分存入df，便于调试
        print(f"    - [风险评分引擎 V57.0] 风险评分完成，最高风险分: {risk_score.max():.0f}")
        return risk_score

    # 出场决策引擎
    def _calculate_exit_signals(self, df: pd.DataFrame, params: dict, risk_score: pd.Series) -> pd.Series:
        """
        【V57.0 出场决策引擎】
        - 核心功能: 基于上游传入的“风险分数”，与配置的阈值进行比较，做出最终的出场决策。
        - 思想: 决策与计算分离。本函数只负责决策。
        - 输出: 最终的出场信号代码 (整数)。
        """
        print("    - [出场决策引擎 V57.0] 启动，开始根据风险分做出决策...")
        
        # 获取出场阈值参数
        threshold_params = self._get_params_block(params, 'exit_threshold_params', {})
        if not threshold_params:
            return pd.Series(0, index=df.index)

        # 定义风险等级和对应的退出代码
        # 从高到低排列，确保高优先级的风险被首先判断
        levels = sorted(threshold_params.items(), key=lambda item: self._get_param_value(item[1].get('level')), reverse=True)
        
        conditions = []
        choices = []
        
        for level_name, config in levels:
            threshold = self._get_param_value(config.get('level'))
            exit_code = self._get_param_value(config.get('code'))
            conditions.append(risk_score >= threshold)
            choices.append(exit_code)
            print(f"      -> 定义决策规则: 风险分 >= {threshold}，则出场，代码: {exit_code}")

        # 使用 np.select 实现高效的、带优先级的条件判断
        exit_signal = np.select(conditions, choices, default=0)
        
        final_exit_signal = pd.Series(exit_signal, index=df.index)
        
        print(f"    - [出场决策引擎 V57.0] 决策完成，共产生 { (final_exit_signal > 0).sum() } 个出场信号。")
        return final_exit_signal

    # ▼▼▼ 准备状态中心 (Setup Condition Center) ▼▼▼
    def _calculate_setup_conditions(self, df: pd.DataFrame, params: dict, trigger_events: Dict[str, pd.Series], chip_atomic_signals: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """
        【V57.0 米其林评审版 - 置信度矩阵引擎】
        - 核心思想: 废除简单的布尔型SETUP，引入“置信度矩阵”对每个准备状态进行动态评分和分级。
        - 职责:
          1. 调用各个独立的诊断模块，获取所有维度的“原子状态”。
          2. 根据配置文件中的“置信度矩阵”，对每个准备状态进行量化评分。
          3. 输出以分数形式存在的准备状态 (e.g., 'SETUP_SCORE_DEEP_ACCUMULATION')。
        """
        print("    - [准备状态中心 V57.0 米其林评审版] 启动...")
        
        # --- 步骤 1: 调用各个独立的诊断模块，获取所有原子状态（食材） ---
        print("        -> 步骤1: 调用各诊断模块，生成所有原子状态...")
        
        # 1.1 筹码状态诊断
        chip_states = self._diagnose_chip_states(df, params)
        
        # 1.2 均线结构与动能状态诊断
        ma_states = self._diagnose_ma_states(df, params)
        
        # 1.3 震荡与超买超卖状态诊断
        oscillator_states = self._diagnose_oscillator_states(df, params)
        
        # 1.4 资金流状态诊断
        capital_states = self._diagnose_capital_states(df, params)

        # 1.5 波动率与成交量状态诊断
        volatility_states = self._diagnose_volatility_states(df, params)

        # 1.6 箱体诊断
        box_states = self._diagnose_box_states(df, params)

        # 1.7 K线组合形态诊断 ▼▼▼
        kline_patterns = self._diagnose_kline_patterns(df, params)

        # 将所有原子状态合并到一个字典中，方便后续调用
        atomic_states = {
            **chip_states, **ma_states, **oscillator_states, 
            **capital_states, **volatility_states, **box_states,
            **kline_patterns
        }
        print("        -> 原子状态生成完毕。")

        # --- 步骤 2: 【核心】定义置信度矩阵并计算分数 ---
        print("        -> 步骤2: 定义置信度矩阵并计算各状态分数...")
        
        # 从配置文件加载评分标准
        scoring_matrix = self._get_params_block(params, 'setup_scoring_matrix', {})
        
        # 初始化一个字典来存储最终的SETUP分数
        setup_scores = {}

        # 遍历矩阵中的每一个准备状态定义
        for setup_name, matrix in scoring_matrix.items():
            print(f"          -> 正在评审 '{setup_name}'...")
            
            # 初始化当前状态的总分Series
            total_score = pd.Series(0.0, index=df.index)
            
            # --- 2.1 检查“必须项” ---
            must_have_conditions = matrix.get('must_have', {})
            # 初始化一个“资格”信号，默认为True，任何一个必须项不满足则变为False
            is_qualified = pd.Series(True, index=df.index)
            
            for condition_name, score in must_have_conditions.items():
                # 从原子状态字典中获取对应的布尔信号
                condition_signal = atomic_states.get(condition_name, pd.Series(False, index=df.index))
                # 更新资格信号：只有之前合格且当前条件也满足的才继续合格
                is_qualified &= condition_signal
                # 如果条件满足，加上基础分
                total_score.loc[condition_signal] += self._get_param_value(score, 0)

            # --- 2.2 检查“加分/减分项” ---
            bonus_conditions = matrix.get('bonus', {})
            for condition_name, score in bonus_conditions.items():
                condition_signal = atomic_states.get(condition_name, pd.Series(False, index=df.index))
                # 如果条件满足，直接加上奖励分（或减去惩罚分）
                total_score.loc[condition_signal] += self._get_param_value(score, 0)
            
            # --- 2.3 应用资格过滤 ---
            # 只有满足所有“必须项”的日期，其分数才有效，否则清零
            final_score = total_score.where(is_qualified, 0)
            
            # 将最终的置信度分数存入结果字典，键名格式为 SETUP_SCORE_XXX
            setup_scores[f'SETUP_SCORE_{setup_name}'] = final_score
            print(f"            -> '{setup_name}' 评审完成，最高置信度得分: {final_score.max():.0f}")

        # --- 步骤 3: 返回分数形式的准备状态 ---
        print("    - [准备状态中心 V57.0] 所有状态置信度评审完成。")
        return setup_scores

    # 筹码状态诊断引擎
    def _diagnose_chip_states(self, df: pd.DataFrame, params: dict) -> Dict[str, pd.Series]:
        """
        【V56.0 辅助模块 - 筹码状态诊断引擎】
        - 职责: 融合新旧两套筹码数据，深度诊断市场所处的筹码博弈阶段，并输出一系列结构化的“筹码原子状态”。
        - 输入:
          - df: 包含所有必需筹码指标的DataFrame。
          - params: 策略配置文件。
        - 输出:
          - 一个字典，键为状态名称 (如 'CHIP_STATE_ACCUMULATION')，值为对应的布尔型 pd.Series。
        """
        print("        -> [诊断模块] 正在执行筹码状态诊断...")
        
        # 初始化状态字典，用于存储所有诊断结果
        states = {}
        
        # --- 步骤 0: 获取参数并校验数据 ---
        # 从总配置中获取本模块专属的参数块
        p = self._get_params_block(params, 'chip_feature_params', {})
        if not self._get_param_value(p.get('enabled'), False):
            print("          -> 筹码诊断模块被禁用，跳过。")
            return states

        # 定义并检查所有必需的列，确保数据完整性
        required_cols = {
            # 新筹码数据 (动态演化)
            'dynamic_concentration_slope': 'CHIP_concentration_90pct_slope_5d',
            'dynamic_winner_rate_short': 'CHIP_winner_rate_short_term',
            'dynamic_winner_rate_long': 'CHIP_winner_rate_long_term',
            'dynamic_slope_8d': 'CHIP_peak_cost_slope_8d',
            'dynamic_slope_21d': 'CHIP_peak_cost_slope_21d',
            'dynamic_accel_21d': 'CHIP_peak_cost_accel_21d',
            # 基础数据
            'base_close': 'close_D'
        }
        
        # 检查所有必需的列是否存在
        if not all(col in df.columns for col in required_cols.values()):
            missing = [k for k, v in required_cols.items() if v not in df.columns]
            print(f"          -> [严重警告] 缺少筹码诊断所需的列: {missing}。引擎将返回空结果。")
            return states

        # --- 步骤 1: 诊断核心的“主状态” (Primary State) ---
        # 主状态是互斥的，一个时间点只能有一个主状态（吸筹、拉升、派发、过渡）
        
        # 1.1 定义“拉升期”条件 (最强特征，优先判断)
        # 定义: 中长期（21日）和超长期（55日，假设已计算）成本峰斜率均向上
        is_markup_base = (df[required_cols['dynamic_slope_21d']] > 0) & \
                         (df.get('CHIP_peak_cost_slope_55d', 0) > 0) # 使用.get安全获取，若无则默认为0

        # 1.2 定义“派发期”条件
        p_dist = p.get('distribution_params', {})
        # 条件a: 筹码在发散（集中度斜率为正）
        is_distributing = df[required_cols['dynamic_concentration_slope']] > self._get_param_value(p_dist.get('divergence_threshold'), 0.01)
        # 条件b: 股价处于近期高位
        is_at_high = df[required_cols['base_close']] > df[required_cols['base_close']].rolling(window=55).quantile(0.8)
        is_distribution_base = is_distributing & is_at_high

        # 1.3 定义“吸筹期”条件
        p_accum = p.get('accumulation_params', {})
        lookback_accum = self._get_param_value(p_accum.get('lookback_days'), 21)
        # 条件a: 在过去N天中，大部分时间筹码都在集中
        concentrating_days = (df[required_cols['dynamic_concentration_slope']] < 0).rolling(window=lookback_accum).sum()
        is_concentrating = concentrating_days >= (lookback_accum * self._get_param_value(p_accum.get('required_days_ratio'), 0.6))
        # 条件b: 中期成本峰未进入明确的上升趋势（允许横盘或下跌）
        is_not_rising = df[required_cols['dynamic_slope_21d']] <= 0
        is_accumulation_base = is_concentrating & is_not_rising

        # 1.4 使用 np.select 根据优先级应用主状态
        # 优先级: 拉升 > 派发 > 吸筹 > 过渡
        conditions = [is_markup_base, is_distribution_base, is_accumulation_base]
        choices = ['MARKUP', 'DISTRIBUTION', 'ACCUMULATION']
        primary_state = pd.Series(np.select(conditions, choices, default='TRANSITION'), index=df.index)
        
        # 将主状态存入结果字典
        states['CHIP_STATE_ACCUMULATION'] = (primary_state == 'ACCUMULATION')
        states['CHIP_STATE_MARKUP'] = (primary_state == 'MARKUP')
        states['CHIP_STATE_DISTRIBUTION'] = (primary_state == 'DISTRIBUTION')
        print("          -> 主状态诊断完成 (吸筹/拉升/派发/过渡)。")

        # --- 步骤 2: 诊断附加的“子状态/标签” (Sub-States / Tags) ---
        # 子状态可以共存，为主状态提供更精细的描述或风险提示

        # 2.1 “深度”标签 (DEEP) - 形容吸筹或派发的程度
        # 定义: 筹码持续集中的天数比例超过一个更高的阈值
        is_deep = concentrating_days >= (lookback_accum * self._get_param_value(p_accum.get('deep_ratio'), 0.85))
        states['CHIP_STATE_ACCUMULATION_DEEP'] = states['CHIP_STATE_ACCUMULATION'] & is_deep
        print(f"          -> “深度吸筹”子状态诊断完成，发现 {states['CHIP_STATE_ACCUMULATION_DEEP'].sum()} 天。")

        # 2.2 “投降坑”机会标签 (OPPORTUNITY_PIT)
        p_capit = p.get('capitulation_params', {})
        # 预计算总获利盘
        total_winner_rate = df[required_cols['dynamic_winner_rate_short']] + df[required_cols['dynamic_winner_rate_long']]
        # 定义: 总获利盘极低，且发生在吸筹期
        is_washed_out = total_winner_rate < self._get_param_value(p_capit.get('winner_rate_threshold'), 8.0)
        states['CHIP_STATE_PIT_OPPORTUNITY'] = is_washed_out & states['CHIP_STATE_ACCUMULATION']
        print(f"          -> “投降坑机会”标签诊断完成，发现 {states['CHIP_STATE_PIT_OPPORTUNITY'].sum()} 天。")

        # 2.3 “趋势衰竭”风险标签 (RISK_EXHAUSTION)
        # 定义: 中期成本仍在上升，但上升的加速度已经转为负值（上涨乏力）
        is_still_rising = df[required_cols['dynamic_slope_21d']] > 0
        is_decelerating = df[required_cols['dynamic_accel_21d']] < 0
        states['CHIP_RISK_EXHAUSTION'] = is_still_rising & is_decelerating
        print(f"          -> “趋势衰竭”风险标签诊断完成，发现 {states['CHIP_RISK_EXHAUSTION'].sum()} 天。")

        # 2.4 “斜率背离”风险标签 (RISK_DIVERGENCE)
        # 定义: 短期成本峰斜率向下，但中期仍在向上，且股价处于高位
        is_short_slope_down = df[required_cols['dynamic_slope_8d']] < 0
        is_mid_slope_up = df[required_cols['dynamic_slope_21d']] > 0
        states['CHIP_RISK_DIVERGENCE'] = is_short_slope_down & is_mid_slope_up & is_at_high
        print(f"          -> “斜率背离”风险标签诊断完成，发现 {states['CHIP_RISK_DIVERGENCE'].sum()} 天。")

        # --- 步骤 3: 诊断关键的“事件” (Events) ---
        # 事件是瞬时的，代表一个重要的转折点
        
        # 3.1 “点火”事件 (EVENT_IGNITION)
        # 定义: 股价上穿市场平均成本，同时伴随着成本峰的加速上移，且发生在吸筹或过渡期之后
        p_ignite = p.get('ignition_params', {}) # 假设为点火事件增加专属参数
        # 从旧筹码数据中获取突破信号
        is_breakout = (df['close_D'] > df['weight_avg_D']) & (df['close_D'].shift(1) <= df['weight_avg_D'].shift(1))
        # 从新筹码数据中获取加速确认
        is_accelerating = df.get('CHIP_peak_cost_accel_5d', 0) > self._get_param_value(p_ignite.get('accel_threshold'), 0.01)
        # 要求前一天处于适合点火的状态
        was_in_setup_state = primary_state.shift(1).isin(['ACCUMULATION', 'TRANSITION'])
        states['CHIP_EVENT_IGNITION'] = is_breakout & is_accelerating & was_in_setup_state
        print(f"          -> “点火事件”诊断完成，发现 {states['CHIP_EVENT_IGNITION'].sum()} 天。")

        # --- 步骤 4: 清理与返回 ---
        # 确保所有返回的 Series 都已填充 NaN，避免下游出错
        for key in states:
            if states[key] is None:
                states[key] = pd.Series(False, index=df.index)
            else:
                states[key] = states[key].fillna(False)

        print("        -> [诊断模块] 筹码状态诊断执行完毕。")
        return states

    # 均线结构与动能状态诊断
    def _diagnose_ma_states(self, df: pd.DataFrame, params: dict) -> Dict[str, pd.Series]:
        """【V57.0 辅助模块】均线结构与动能状态诊断"""
        states = {}
        p = self._get_params_block(params, 'ma_state_params', {})
        if not self._get_param_value(p.get('enabled'), False): return states

        # 定义均线周期
        short_p = self._get_param_value(p.get('short_ma'), 13)
        mid_p = self._get_param_value(p.get('mid_ma'), 34)
        long_p = self._get_param_value(p.get('long_ma'), 89)
        
        short_ma, mid_ma, long_ma = f'EMA_{short_p}_D', f'EMA_{mid_p}_D', f'EMA_{long_p}_D'
        
        # 检查列是否存在
        if not all(c in df.columns for c in [short_ma, mid_ma, long_ma]):
            print(f"          -> [警告] 缺少均线列，均线状态诊断跳过。")
            return states

        # 均线结构状态
        states['MA_STATE_STABLE_BULLISH'] = (df[short_ma] > df[mid_ma]) & (df[mid_ma] > df[long_ma])
        states['MA_STATE_STABLE_BEARISH'] = (df[short_ma] < df[mid_ma]) & (df[mid_ma] < df[long_ma])
        
        # 计算均线间距的Z-score来判断收敛
        ma_spread = (df[short_ma] - df[long_ma]) / df[long_ma].replace(0, np.nan)
        ma_spread_zscore = (ma_spread - ma_spread.rolling(60).mean()) / ma_spread.rolling(60).std().replace(0, np.nan)
        states['MA_STATE_CONVERGING'] = ma_spread_zscore < self._get_param_value(p.get('converging_zscore'), -1.0)
        states['MA_STATE_DIVERGING'] = ma_spread_zscore > self._get_param_value(p.get('diverging_zscore'), 1.0)
        
        # 底部钝化状态: 空头排列下，价格首次站上短期均线
        states['MA_STATE_BOTTOM_PASSIVATION'] = states['MA_STATE_STABLE_BEARISH'] & (df['close_D'] > df[short_ma])

        # 日线动能状态
        lookback_period = 10 # 定义用于动能判断的回看周期
        accel_d_col = f'ACCEL_{lookback_period}_{long_ma}' # 正确的列名格式
        
        # 使用安全的方式检查列是否存在
        if accel_d_col in df.columns:
            states['MA_STATE_D_STABILIZING'] = (df[accel_d_col].shift(1).fillna(0) < 0) & (df[accel_d_col] >= 0)
        else:
            # 如果列不存在，则状态为False
            states['MA_STATE_D_STABILIZING'] = pd.Series(False, index=df.index)
            print(f"          -> [警告] 缺少日线加速度列 '{accel_d_col}'，'MA_STATE_D_STABILIZING' 无法计算。")

        # 周线动能状态
        lookback_period_w = 5 # 假设周线使用5周回看
        slope_w_col = f'SLOPE_{lookback_period_w}_EMA_21_W' # 正确的列名格式
        
        if slope_w_col in df.columns:
            states['MA_STATE_W_STABILIZING'] = (df[slope_w_col].shift(1).fillna(0) < 0) & (df[slope_w_col] >= 0)
        else:
            states['MA_STATE_W_STABILIZING'] = pd.Series(False, index=df.index)
            print(f"          -> [警告] 缺少周线斜率列 '{slope_w_col}'，'MA_STATE_W_STABILIZING' 无法计算。")

        # ---  老鸭头形态诊断 ---
        p_duck = self._get_params_block(params, 'duck_neck_params', {})
        if self._get_param_value(p_duck.get('enabled'), True):
            short_ma_p = self._get_param_value(p_duck.get('short_ma'), 5)
            mid_ma_p = self._get_param_value(p_duck.get('mid_ma'), 10)
            long_ma_p = self._get_param_value(p_duck.get('long_ma'), 60)
            
            short_ma = f'EMA_{short_ma_p}_D'
            mid_ma = f'EMA_{mid_ma_p}_D'
            long_ma = f'EMA_{long_ma_p}_D'

            if all(c in df.columns for c in [short_ma, mid_ma, long_ma]):
                # 定义“金叉”事件：5日线上穿10日线
                golden_cross_event = (df[short_ma] > df[mid_ma]) & (df[short_ma].shift(1) <= df[mid_ma].shift(1))
                
                # 定义破坏条件：5日线重新死叉10日线，形态失败
                break_condition = (df[short_ma] < df[mid_ma])
                
                persistence_days = self._get_param_value(p_duck.get('persistence_days'), 20)
                
                # 创建“鸭颈形成中”的持久化状态
                states['MA_STATE_DUCK_NECK_FORMING'] = self._create_persistent_state(
                    df,
                    entry_event=golden_cross_event,
                    persistence_days=persistence_days,
                    break_condition=break_condition
                )
                print(f"          -> '老鸭颈形成中' 持久化状态诊断完成，共激活 {states['MA_STATE_DUCK_NECK_FORMING'].sum()} 天。")

        
        return states

    # 震荡指标与超买超卖状态诊断
    def _diagnose_oscillator_states(self, df: pd.DataFrame, params: dict) -> Dict[str, pd.Series]:
        """【V57.0 辅助模块】震荡指标与超买超卖状态诊断"""
        states = {}
        p = self._get_params_block(params, 'oscillator_state_params', {})
        if not self._get_param_value(p.get('enabled'), False): return states

        # RSI 状态
        rsi_col = 'RSI_13_D'
        if rsi_col in df.columns:
            states['OSC_STATE_RSI_OVERBOUGHT'] = df[rsi_col] > self._get_param_value(p.get('rsi_overbought'), 80)
            states['OSC_STATE_RSI_OVERSOLD'] = df[rsi_col] < self._get_param_value(p.get('rsi_oversold'), 25)

        # MACD 状态
        macd_h_col = 'MACDh_13_34_8_D'
        macd_z_col = 'MACD_HIST_ZSCORE_D'
        if macd_h_col in df.columns:
            states['OSC_STATE_MACD_BULLISH'] = df[macd_h_col] > 0
        if macd_z_col in df.columns:
            # 简化的背离判断：价格创新高，但Z-score没有
            is_price_higher = df['close_D'] > df['close_D'].rolling(10).max().shift(1)
            is_macd_z_lower = df[macd_z_col] < df[macd_z_col].rolling(10).max().shift(1)
            states['OSC_STATE_MACD_DIVERGENCE'] = is_price_higher & is_macd_z_lower

        return states

    # 资金流状态诊断
    def _diagnose_capital_states(self, df: pd.DataFrame, params: dict) -> Dict[str, pd.Series]:
        """【V58.0 升级版 - 集成持久化状态】资金流状态诊断"""
        states = {}
        p = self._get_params_block(params, 'capital_state_params', {})
        if not self._get_param_value(p.get('enabled'), False): return states

        # --- 1. 瞬时状态诊断 (与之前相同) ---
        mf_slope_short_col = f'SLOPE_10_net_mf_amount_D' # 使用新命名规范
        mf_slope_long_col = f'SLOPE_20_net_mf_amount_D'
        if mf_slope_short_col in df.columns and mf_slope_long_col in df.columns:
            mf_slope_short = df[mf_slope_short_col]
            mf_slope_long = df[mf_slope_long_col]
            states['CAPITAL_STATE_INFLOW_CONFIRMED'] = mf_slope_short > 0
            # 定义一个瞬时的“资金斜率金叉”事件
            states['CAPITAL_EVENT_SLOPE_CROSS'] = (mf_slope_short > mf_slope_long) & (mf_slope_short.shift(1) <= mf_slope_long.shift(1))
        
        cmf_col = 'CMF_21_D'
        if cmf_col in df.columns:
            states['CAPITAL_STATE_CMF_BULLISH'] = df[cmf_col] > self._get_param_value(p.get('cmf_bullish_threshold'), 0.05)

        # --- 2. 【新增】持久化状态诊断 ---
        # 2.1 定义“价跌资增”的底背离事件
        price_down = df['pct_change_D'] < 0
        capital_up = df['net_mf_amount_D'] > 0
        bottom_divergence_event = price_down & capital_up
        
        # 2.2 调用状态机生成器，创建“资本底背离机会窗口”
        p_context = p.get('divergence_context', {})
        persistence_days = self._get_param_value(p_context.get('persistence_days'), 15)
        trend_ma_period = self._get_param_value(p_context.get('trend_ma_period'), 55)
        trend_ma_col = f'EMA_{trend_ma_period}_D'
        
        if trend_ma_col in df.columns:
            # 定义破坏条件：一旦价格站上关键趋势线，则机会窗口关闭
            break_condition = df['close_D'] > df[trend_ma_col]
            
            # 调用新工具生成持久化状态
            states['CAPITAL_STATE_DIVERGENCE_WINDOW'] = self._create_persistent_state(
                df,
                entry_event=bottom_divergence_event,
                persistence_days=persistence_days,
                break_condition=break_condition
            )
            print(f"          -> '资本底背离机会窗口' 持久化状态诊断完成，共激活 {states['CAPITAL_STATE_DIVERGENCE_WINDOW'].sum()} 天。")

        return states

    # 动率与成交量状态诊断
    def _diagnose_volatility_states(self, df: pd.DataFrame, params: dict) -> Dict[str, pd.Series]:
        """【V58.1 升级版 - 集成持久化状态】波动率与成交量状态诊断"""
        states = {}
        p = self._get_params_block(params, 'volatility_state_params', {})
        if not self._get_param_value(p.get('enabled'), False): return states

        # --- 1. 瞬时状态诊断 ---
        bbw_col = 'BBW_21_2.0_D'
        if bbw_col in df.columns:
            # 定义“极度压缩”事件：布林带宽度进入历史最低的10%分位
            squeeze_threshold = df[bbw_col].rolling(60).quantile(self._get_param_value(p.get('squeeze_percentile'), 0.1))
            # Squeeze事件：当天宽度小于阈值，且前一天大于等于阈值
            squeeze_event = (df[bbw_col] < squeeze_threshold) & (df[bbw_col].shift(1) >= squeeze_threshold)
            states['VOL_EVENT_SQUEEZE'] = squeeze_event
            print(f"          -> '波动率极度压缩' 事件诊断完成，发现 {states['VOL_EVENT_SQUEEZE'].sum()} 天。")

        vol_ma_col = 'VOL_MA_21_D'
        if 'volume_D' in df.columns and vol_ma_col in df.columns:
            states['VOL_STATE_SHRINKING'] = df['volume_D'] < df[vol_ma_col] * self._get_param_value(p.get('shrinking_ratio'), 0.8)

        # --- 2. 【新增】持久化状态诊断 ---
        # 从“极度压缩”事件，生成一个“能量待爆发窗口”
        p_context = p.get('squeeze_context', {})
        persistence_days = self._get_param_value(p_context.get('persistence_days'), 10)
        
        # 定义破坏条件：成交量显著放大，意味着能量已经释放，窗口关闭
        if vol_ma_col in df.columns:
            volume_break_ratio = self._get_param_value(p_context.get('volume_break_ratio'), 1.5)
            break_condition = df['volume_D'] > df[vol_ma_col] * volume_break_ratio
            
            states['VOL_STATE_SQUEEZE_WINDOW'] = self._create_persistent_state(
                df,
                entry_event=states.get('VOL_EVENT_SQUEEZE', pd.Series(False, index=df.index)),
                persistence_days=persistence_days,
                break_condition=break_condition
            )
            print(f"          -> '能量待爆发窗口' 持久化状态诊断完成，共激活 {states['VOL_STATE_SQUEEZE_WINDOW'].sum()} 天。")

        return states

    # 箱体状态诊断引擎
    def _diagnose_box_states(self, df: pd.DataFrame, params: dict) -> Dict[str, pd.Series]:
        """
        【V57.0 诊断模块 - 箱体状态诊断引擎】
        - 核心功能: 识别动态的盘整箱体，并将其状态（如盘整、突破、跌破）转化为标准的原子状态。
        - 职责:
          1. 使用 find_peaks 算法识别重要的波峰和波谷。
          2. 定义每一天的动态箱体（上轨、下轨）。
          3. 诊断当前价格处于箱体的何种状态。
        - 输出: 一个包含所有箱体原子状态的字典。
        """
        print("        -> [诊断模块] 正在执行箱体状态诊断...")
        
        # 初始化状态字典
        states = {}
        # 获取本模块的参数
        box_params = self._get_params_block(params, 'dynamic_box_params', {})
        
        # 如果模块被禁用或数据不足，则返回空字典
        if not self._get_param_value(box_params.get('enabled'), False) or df.empty:
            print("          -> 箱体诊断模块被禁用或数据为空，跳过。")
            return states

        # --- 步骤 1: 识别重要的波峰和波谷 ---
        # 使用 scipy.signal.find_peaks 寻找高点和低点
        # prominence (突起程度) 是一个关键参数，用于过滤掉不重要的微小波动
        peak_distance = self._get_param_value(box_params.get('peak_distance'), 10)
        peak_prominence = self._get_param_value(box_params.get('peak_prominence'), 0.02)
        
        # 计算绝对突起程度，因为它比相对值更稳定
        price_range = df['high_D'].max() - df['low_D'].min()
        absolute_prominence = price_range * peak_prominence if price_range > 0 else 0.01

        # 寻找波峰（高点）
        peak_indices, _ = find_peaks(df['close_D'], distance=peak_distance, prominence=absolute_prominence)
        # 寻找波谷（低点），注意要对价格取反
        trough_indices, _ = find_peaks(-df['close_D'], distance=peak_distance, prominence=absolute_prominence)

        # --- 步骤 2: 构建动态箱体的上轨和下轨 ---
        # 创建一个临时列来存储最近的波峰价格
        last_peak_price = pd.Series(np.nan, index=df.index)
        if len(peak_indices) > 0:
            last_peak_price.iloc[peak_indices] = df['close_D'].iloc[peak_indices]
        last_peak_price.ffill(inplace=True) # 向前填充，使得每一天都知道它之前的最后一个波峰价

        # 创建一个临时列来存储最近的波谷价格
        last_trough_price = pd.Series(np.nan, index=df.index)
        if len(trough_indices) > 0:
            last_trough_price.iloc[trough_indices] = df['close_D'].iloc[trough_indices]
        last_trough_price.ffill(inplace=True) # 向前填充

        # 定义箱体上轨和下轨
        box_top = last_peak_price
        box_bottom = last_trough_price

        # --- 步骤 3: 诊断并生成原子状态 ---
        # 3.1 定义一个有效的箱体
        is_valid_box = (box_top.notna()) & (box_bottom.notna()) & (box_top > box_bottom)
        
        # 3.2 原子状态: 箱体向上突破 (BOX_EVENT_BREAKOUT)
        # 定义: 昨天收盘在箱体内或箱体下，今天收盘在箱体上
        was_below_top = df['close_D'].shift(1) <= box_top.shift(1)
        is_above_top = df['close_D'] > box_top
        states['BOX_EVENT_BREAKOUT'] = is_valid_box & is_above_top & was_below_top
        print(f"          -> '箱体向上突破' 事件诊断完成，发现 {states['BOX_EVENT_BREAKOUT'].sum()} 天。")

        # 3.3 原子状态: 箱体向下突破 (BOX_EVENT_BREAKDOWN)
        # 定义: 昨天收盘在箱体内或箱体上，今天收盘在箱体下
        was_above_bottom = df['close_D'].shift(1) >= box_bottom.shift(1)
        is_below_bottom = df['close_D'] < box_bottom
        states['BOX_EVENT_BREAKDOWN'] = is_valid_box & is_below_bottom & was_above_bottom
        print(f"          -> '箱体向下突破' 事件诊断完成，发现 {states['BOX_EVENT_BREAKDOWN'].sum()} 天。")

        # 3.4 原子状态: 健康的箱体盘整 (BOX_STATE_HEALTHY_CONSOLIDATION)
        # 定义: 处于一个有效的箱体内，并且箱体整体位于关键均线上方
        ma_params = self._get_params_block(params, 'ma_state_params', {})
        mid_ma_period = self._get_param_value(ma_params.get('mid_ma'), 55)
        mid_ma_col = f"EMA_{mid_ma_period}_D"
        
        is_in_box = (df['close_D'] < box_top) & (df['close_D'] > box_bottom)
        if mid_ma_col in df.columns:
            # 箱体中轴线高于中期均线，代表盘整是强势的
            box_midpoint = (box_top + box_bottom) / 2
            is_box_above_ma = box_midpoint > df[mid_ma_col]
            states['BOX_STATE_HEALTHY_CONSOLIDATION'] = is_valid_box & is_in_box & is_box_above_ma
        else:
            # 如果没有均线数据，则只判断是否在箱体内
            states['BOX_STATE_HEALTHY_CONSOLIDATION'] = is_valid_box & is_in_box
        print(f"          -> '健康箱体盘整' 状态诊断完成，发现 {states['BOX_STATE_HEALTHY_CONSOLIDATION'].sum()} 天。")

        # --- 步骤 4: 清理与返回 ---
        # 确保所有返回的 Series 都已填充 NaN，避免下游出错
        for key in states:
            if states[key] is None:
                states[key] = pd.Series(False, index=df.index)
            else:
                states[key] = states[key].fillna(False)

        print("        -> [诊断模块] 箱体状态诊断执行完毕。")
        return states

    # 风险因子诊断引擎
    def _diagnose_risk_factors(self, df: pd.DataFrame, params: dict) -> Dict[str, pd.Series]:
        """
        【V59.1 诊断模块 - 风险因子全景诊断版】
        - 核心功能: 集中诊断所有可能导致离场的、独立的“原子风险因子”，包括瞬时事件和持久化状态。
        - 职责: 将复杂的价量形态和指标状态，转化为一系列标准的、可用于风险评分的布尔型风险标签。
        - 输出: 一个包含所有原子风险因子的字典 (e.g., {'RISK_EVENT_...': pd.Series, 'RISK_STATE_...': pd.Series})。
        """
        print("    - [风险诊断引擎 V59.1] 启动，开始诊断所有原子风险因子...")
        # 初始化一个字典来存储所有风险因子
        risks = {}
        # 获取出场策略的专属参数块
        exit_params = self._get_params_block(params, 'exit_strategy_params', {})
        if not self._get_param_value(exit_params.get('enabled'), False):
            print("      -> 出场策略被禁用，风险诊断跳过。")
            return risks

        # --- 风险因子1: 主力叛逃 (高位派发) - 瞬时事件 ---
        p = exit_params.get('upthrust_distribution_params', {})
        if self._get_param_value(p.get('enabled'), True):
            overextension_ma_col = f"EMA_{self._get_param_value(p.get('overextension_ma_period'), 55)}_D"
            if all(c in df.columns for c in [overextension_ma_col, 'net_mf_amount_D']):
                is_overextended = (df['close_D'] / df[overextension_ma_col] - 1) > self._get_param_value(p.get('overextension_threshold'), 0.3)
                total_range = (df['high_D'] - df['low_D']).replace(0, np.nan)
                upper_shadow = (df['high_D'] - np.maximum(df['open_D'], df['close_D'])) / total_range
                has_long_upper_shadow = upper_shadow > self._get_param_value(p.get('upper_shadow_ratio'), 0.6)
                is_high_volume = df['volume_D'] > df['volume_D'].rolling(30).quantile(self._get_param_value(p.get('high_volume_quantile'), 0.85))
                is_main_force_selling = df['net_mf_amount_D'] < 0
                risks['RISK_EVENT_UPTHRUST_DISTRIBUTION'] = is_overextended & has_long_upper_shadow & is_high_volume & is_main_force_selling
                print(f"      -> '主力叛逃' 风险事件诊断完成，发现 {risks.get('RISK_EVENT_UPTHRUST_DISTRIBUTION', pd.Series([])).sum()} 天。")

        # --- 风险因子2: 结构崩溃 (放量破位) - 瞬时事件 ---
        p = exit_params.get('volume_breakdown_params', {})
        if self._get_param_value(p.get('enabled'), True):
            support_ma_col = f"EMA_{self._get_param_value(p.get('support_ma_period'), 55)}_D"
            vol_ma_col = 'VOL_MA_21_D'
            if all(c in df.columns for c in [support_ma_col, vol_ma_col, 'net_mf_amount_D', 'amount_D']):
                is_ma_broken = (df['close_D'] < df[support_ma_col]) & (df['close_D'].shift(1) >= df[support_ma_col].shift(1))
                is_volume_surge = df['volume_D'] > df[vol_ma_col] * self._get_param_value(p.get('volume_surge_ratio'), 1.8)
                avg_amount_20d = df['amount_D'].rolling(20).mean()
                is_main_force_dumping = df['net_mf_amount_D'] < -(avg_amount_20d * self._get_param_value(p.get('main_force_dump_ratio'), 0.1))
                # 增加坚决阴线确认
                is_conviction_candle = (df['open_D'] - df['close_D']) > (df['high_D'] - df['low_D']) * 0.6
                risks['RISK_EVENT_STRUCTURE_BREAKDOWN'] = is_ma_broken & is_volume_surge & is_main_force_dumping & is_conviction_candle
                print(f"      -> '结构崩溃' 风险事件诊断完成，发现 {risks.get('RISK_EVENT_STRUCTURE_BREAKDOWN', pd.Series([])).sum()} 天。")

        # --- 风险因子3: 顶背离 (瞬时事件 + 持久化状态) ---
        p_div = exit_params.get('divergence_exit_params', {})
        if self._get_param_value(p_div.get('enabled'), True):
            lookback = self._get_param_value(p_div.get('lookback_period'), 20)
            price_is_new_high = df['close_D'] == df['close_D'].rolling(lookback).max()
            rsi_not_new_high = df['RSI_13_D'] < df['RSI_13_D'].rolling(lookback).max().shift(1)
            macd_z_not_new_high = df.get('MACD_HIST_ZSCORE_D', pd.Series(0, index=df.index)) < df.get('MACD_HIST_ZSCORE_D', pd.Series(0, index=df.index)).rolling(lookback).max().shift(1)
            top_divergence_event = price_is_new_high & (rsi_not_new_high | macd_z_not_new_high)
            risks['RISK_EVENT_TOP_DIVERGENCE'] = top_divergence_event
            print(f"      -> '顶背离' 风险事件诊断完成，发现 {risks.get('RISK_EVENT_TOP_DIVERGENCE', pd.Series([])).sum()} 天。")

            p_context = exit_params.get('divergence_context', {})
            persistence_days = self._get_param_value(p_context.get('persistence_days'), 5)
            break_ma_period = self._get_param_value(p_context.get('break_ma_period'), 10)
            break_ma_col = f'EMA_{break_ma_period}_D'
            if break_ma_col in df.columns:
                break_condition = df['close_D'] < df[break_ma_col]
                risks['RISK_STATE_DIVERGENCE_WINDOW'] = self._create_persistent_state(
                    df, entry_event=top_divergence_event, persistence_days=persistence_days, break_condition=break_condition
                )
                print(f"      -> '顶背离高危窗口' 持久化风险状态诊断完成，共激活 {risks.get('RISK_STATE_DIVERGENCE_WINDOW', pd.Series([])).sum()} 天。")

        # --- 风险因子4: 指标超买 (常规风险状态) ---
        p = exit_params.get('indicator_exit_params', {})
        if self._get_param_value(p.get('enabled'), True):
            risks['RISK_STATE_RSI_OVERBOUGHT'] = df.get('RSI_13_D', 50) > self._get_param_value(p.get('rsi_threshold'), 85)
            risks['RISK_STATE_BIAS_OVERBOUGHT'] = df.get('BIAS_20_D', 0) > self._get_param_value(p.get('bias_threshold'), 20.0)
            print(f"      -> '指标超买' (RSI/BIAS) 风险状态诊断完成。")
            
        # --- 风险因子5: 天地板 (极端风险事件) ---
        board_events = self._diagnose_board_patterns(df, params)
        risks['RISK_EVENT_HEAVEN_EARTH_BOARD'] = board_events.get('BOARD_EVENT_HEAVEN_EARTH', pd.Series(False, index=df.index))
        print(f"      -> '天地板' 极端风险事件诊断完成，发现 {risks.get('RISK_EVENT_HEAVEN_EARTH_BOARD', pd.Series([])).sum()} 天。")

        # --- 清理与返回 ---
        for key in risks:
            risks[key] = risks[key].fillna(False)
        
        print("    - [风险诊断引擎 V59.1] 所有风险因子诊断完成。")
        return risks

    # 板形态诊断引擎
    def _diagnose_board_patterns(self, df: pd.DataFrame, params: dict) -> Dict[str, pd.Series]:
        """
        【V58.0 诊断模块 - 板形态诊断引擎】
        - 核心功能: 识别A股特色的各种“板”形态，并将其转化为标准的原子事件。
        - 输出: 一个包含所有板形态事件的字典。
        """
        print("        -> [诊断模块] 正在执行板形态诊断...")
        states = {}
        p = self._get_params_block(params, 'board_pattern_params', {})
        if not self._get_param_value(p.get('enabled'), False):
            return states

        # --- 准备基础数据 ---
        prev_close = df['close_D'].shift(1)
        limit_up_threshold = self._get_param_value(p.get('limit_up_threshold'), 0.098)
        limit_down_threshold = self._get_param_value(p.get('limit_down_threshold'), -0.098)
        price_buffer = self._get_param_value(p.get('price_buffer'), 0.005)

        # --- 计算涨跌停价格（近似）---
        limit_up_price = prev_close * (1 + limit_up_threshold)
        limit_down_price = prev_close * (1 + limit_down_threshold)

        # --- 识别涨跌停状态 (带缓冲) ---
        is_limit_up_close = df['close_D'] >= limit_up_price * (1 - price_buffer)
        is_limit_up_high = df['high_D'] >= limit_up_price * (1 - price_buffer)
        is_limit_down_low = df['low_D'] <= limit_down_price * (1 + price_buffer)

        # --- 诊断事件 ---
        # 地天板 (强力买入/反转信号)
        states['BOARD_EVENT_EARTH_HEAVEN'] = is_limit_down_low & is_limit_up_close
        print(f"          -> '地天板' 事件诊断完成，发现 {states['BOARD_EVENT_EARTH_HEAVEN'].sum()} 天。")
        
        # 天地板 (强力卖出/风险信号)
        is_limit_down_close = df['close_D'] <= limit_down_price * (1 + price_buffer)
        states['BOARD_EVENT_HEAVEN_EARTH'] = is_limit_up_high & is_limit_down_close
        print(f"          -> '天地板' 事件诊断完成，发现 {states['BOARD_EVENT_HEAVEN_EARTH'].sum()} 天。")

        return states

    # K线组合形态诊断引擎
    def _diagnose_kline_patterns(self, df: pd.DataFrame, params: dict) -> Dict[str, pd.Series]:
        """
        【V59.0 诊断模块 - K线组合形态诊断引擎】
        - 核心功能: 识别并量化经典的、跨越多日的K线组合形态。
        - 职责: 将旧框架中的硬编码剧本（如巨阴洗盘、N字板、缺口支撑）转化为标准的原子状态。
        - 工具: 大量使用 _create_persistent_state 状态机生成器。
        """
        print("        -> [诊断模块] 正在执行K线组合形态诊断...")
        states = {}
        p = self._get_params_block(params, 'kline_pattern_params', {})
        if not self._get_param_value(p.get('enabled'), False):
            return states

        # --- 1. 诊断“巨阴洗盘”事件 (Washout Candle) ---
        p_wash = p.get('washout_params', {})
        if self._get_param_value(p_wash.get('enabled'), True):
            washout_threshold = self._get_param_value(p_wash.get('washout_threshold'), -0.07)
            washout_vol_ratio = self._get_param_value(p_wash.get('washout_volume_ratio'), 1.5)
            vol_ma_col = 'VOL_MA_21_D'
            if vol_ma_col in df.columns:
                is_big_down_day = df['pct_change_D'] < washout_threshold
                is_huge_volume = df['volume_D'] > df[vol_ma_col] * washout_vol_ratio
                states['KLINE_EVENT_WASHOUT'] = is_big_down_day & is_huge_volume
                print(f"          -> '巨阴洗盘' 事件诊断完成，发现 {states['KLINE_EVENT_WASHOUT'].sum()} 天。")

        # --- 2. 诊断“缺口支撑”状态 (Gap Support) ---
        p_gap = p.get('gap_support_params', {})
        if self._get_param_value(p_gap.get('enabled'), True):
            # 事件：向上跳空缺口
            gap_up_event = df['low_D'] > df['high_D'].shift(1)
            # 破坏条件：缺口被回补
            gap_support_level = df['high_D'].shift(1)
            break_condition = df['low_D'] <= gap_support_level
            
            persistence_days = self._get_param_value(p_gap.get('persistence_days'), 10)
            states['KLINE_STATE_GAP_SUPPORT_ACTIVE'] = self._create_persistent_state(
                df,
                entry_event=gap_up_event,
                persistence_days=persistence_days,
                break_condition=break_condition
            )
            print(f"          -> '缺口支撑有效' 状态诊断完成，共激活 {states['KLINE_STATE_GAP_SUPPORT_ACTIVE'].sum()} 天。")

        # --- 3. 诊断“N字形态-整理期”状态 (N-Shape Consolidation) ---
        p_nshape = p.get('n_shape_params', {})
        if self._get_param_value(p_nshape.get('enabled'), True):
            # 事件：一根涨停或大阳线作为N字的第一笔（旗杆）
            rally_threshold = self._get_param_value(p_nshape.get('rally_threshold'), 0.097)
            n_shape_start_event = df['pct_change_D'] >= rally_threshold
            
            # 破坏条件：跌破启动阳线的开盘价
            start_day_open = df['open_D'].where(n_shape_start_event, np.nan).ffill()
            break_condition = df['close_D'] < start_day_open
            
            persistence_days = self._get_param_value(p_nshape.get('consolidation_days_max'), 3)
            # N字整理期从启动日的下一天开始，所以要对event进行shift
            states['KLINE_STATE_N_SHAPE_CONSOLIDATION'] = self._create_persistent_state(
                df,
                entry_event=n_shape_start_event.shift(1).fillna(False),
                persistence_days=persistence_days,
                break_condition=break_condition
            )
            print(f"          -> 'N字形态整理期' 状态诊断完成，共激活 {states['KLINE_STATE_N_SHAPE_CONSOLIDATION'].sum()} 天。")

        print("        -> [诊断模块] K线组合形态诊断执行完毕。")
        return states


    # ▼▼▼ 智能衍生特征引擎 (Intelligent Derived Feature Engine) ▼▼▼
    def _prepare_derived_features(self, df: pd.DataFrame, params: dict) -> pd.DataFrame:
        """
        【V58.0 资金博弈深度解析版】
        - 核心升级: 从简单的资金流计算，升级为对资金博弈、动能和价资关系的深度量化。
        - 新增特征:
          1. 资金动能 (Z-Score): 衡量当日主力资金的统计显著性。
          2. 机构散户背离度: 量化主力与散户的行为差异。
          3. 价资确认度: 判断价格趋势是否得到资金的确认。
          4. 主力多空比: 衡量主力内部的买卖意愿强度。
        """
        print("    - [衍生特征中心 V58.0 资金博弈深度解析版] 启动...")
        
        # --- 步骤 0: 获取参数 ---
        p = self._get_params_block(params, 'derived_feature_params', {})
        if not self._get_param_value(p.get('enabled'), True):
            print("      -> 衍生特征引擎被禁用，跳过。")
            return df

        # --- 步骤 1: 核心资金流计算 (保持稳健) ---
        # 1.1 主力净流入额 (智能回退逻辑)
        if 'net_mf_amount_D' not in df.columns or df['net_mf_amount_D'].isnull().all():
            if all(c in df.columns for c in ['buy_lg_amount_D', 'sell_lg_amount_D', 'buy_elg_amount_D', 'sell_elg_amount_D']):
                df['net_mf_amount_D'] = (df['buy_lg_amount_D'] + df['buy_elg_amount_D']) - (df['sell_lg_amount_D'] + df['sell_elg_amount_D'])
            else:
                df['net_mf_amount_D'] = 0
        
        # 1.2 散户净流入额 (智能回退逻辑)
        if 'net_retail_amount_D' not in df.columns or df['net_retail_amount_D'].isnull().all():
            if all(c in df.columns for c in ['buy_sm_amount_D', 'sell_sm_amount_D']):
                df['net_retail_amount_D'] = df['buy_sm_amount_D'] - df['sell_sm_amount_D']
            else:
                df['net_retail_amount_D'] = 0

        # --- 步骤 2: 【关键】在进行任何除法运算前，进行最终的智能净化 ---
        df = self._ensure_numeric_types(df)
        print("      -> 核心资金流计算与类型净化完成。")

        # --- 步骤 3: 计算标准化的比率型特征 ---
        # 3.1 主力净流入额占比 (衡量主力资金对市场总成交的影响力)
        if 'amount_D' in df.columns:
            df['net_mf_ratio_D'] = np.divide(df['net_mf_amount_D'], df['amount_D'], out=np.zeros_like(df['net_mf_amount_D'], dtype=float), where=df['amount_D']!=0)
            print(f"      -> 已计算 'net_mf_ratio_D' (主力净流入占比)。")
        
        # 3.2 【新增】主力多空比 (衡量主力内部买卖意愿)
        if all(c in df.columns for c in ['buy_lg_amount_D', 'sell_lg_amount_D', 'buy_elg_amount_D', 'sell_elg_amount_D']):
            main_force_buy = df['buy_lg_amount_D'] + df['buy_elg_amount_D']
            main_force_sell = df['sell_lg_amount_D'] + df['sell_elg_amount_D']
            df['mf_buy_sell_ratio_D'] = np.divide(main_force_buy, main_force_sell, out=np.ones_like(main_force_buy, dtype=float), where=main_force_sell!=0)
            print(f"      -> 已计算 'mf_buy_sell_ratio_D' (主力多空比)。")

        # --- 步骤 4: 计算动能与博弈特征 ---
        capital_p = p.get('capital_flow_params', {})
        # 4.1 主力资金累积 (衡量中期趋势)
        accum_period = self._get_param_value(capital_p.get('accumulation_period'), 20)
        df[f'mf_accumulation_{accum_period}_D'] = df['net_mf_amount_D'].rolling(window=accum_period).sum()
        print(f"      -> 已计算 'mf_accumulation_{accum_period}_D' ({accum_period}日主力资金累积)。")

        # 4.2 【新增】主力资金Z-Score (衡量短期资金异动)
        zscore_period = self._get_param_value(p.get('momentum_params', {}).get('zscore_period'), 20)
        rolling_mean = df['net_mf_amount_D'].rolling(window=zscore_period).mean()
        rolling_std = df['net_mf_amount_D'].rolling(window=zscore_period).std()
        df['net_mf_zscore_D'] = np.divide((df['net_mf_amount_D'] - rolling_mean), rolling_std, out=np.zeros_like(df['net_mf_amount_D'], dtype=float), where=rolling_std!=0)
        print(f"      -> 已计算 'net_mf_zscore_D' ({zscore_period}日主力资金Z-Score)。")

        # 4.3 【新增】机构散户背离度 (衡量市场共识)
        # 定义：主力净流入 / abs(散户净流入)。绝对值越大，背离越严重。正数代表主力和散户对冲，负数代表同向。
        df['mf_retail_divergence_D'] = np.divide(df['net_mf_amount_D'], df['net_retail_amount_D'].abs(), out=np.zeros_like(df['net_mf_amount_D'], dtype=float), where=df['net_retail_amount_D']!=0)
        print(f"      -> 已计算 'mf_retail_divergence_D' (机构散户背离度)。")

        # --- 步骤 5: 【新增】计算价资关系特征 ---
        divergence_p = p.get('divergence_params', {})
        # 5.1 计算每日的“价资确认分数”
        price_up = df['pct_change_D'] > 0
        price_down = df['pct_change_D'] < 0
        capital_up = df['net_mf_amount_D'] > 0
        capital_down = df['net_mf_amount_D'] < 0
        
        conditions = [
            (price_up & capital_up),      # 价涨资增 (确认)
            (price_down & capital_down),  # 价跌资减 (确认)
            (price_up & capital_down),    # 价涨资减 (顶背离风险)
            (price_down & capital_up)     # 价跌资增 (底背离机会)
        ]
        choices = [1, 1, -1, -1]
        df['capital_price_score_D'] = np.select(conditions, choices, default=0)
        
        # 5.2 计算“价资确认趋势”
        trend_period = self._get_param_value(divergence_p.get('trend_period'), 10)
        df['capital_price_trend_D'] = df['capital_price_score_D'].rolling(window=trend_period).sum()
        print(f"      -> 已计算 'capital_price_trend_D' ({trend_period}日价资确认趋势)。")

        print("    - [衍生特征中心 V58.0] 所有衍生特征准备完成。")
        return df

    # ▼▼▼ 触发事件引擎 (Trigger Event Engine) ▼▼▼
    def _define_trigger_events(self, df: pd.DataFrame, params: dict, chip_atomic_signals: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """
        【V57.0 触发事件引擎】
        - 核心思想: 集中定义所有瞬时的“点火”信号。每个信号都代表一个明确的、可识别的交易动作。
        - 职责:
          1. 从配置文件中读取各类触发事件的参数。
          2. 基于量价、K线形态、指标交叉等信息，计算出一系列布尔型的触发事件信号。
          3. 输出一个包含所有触发事件的字典 (e.g., {'TRIGGER_CHIP_IGNITION': pd.Series(...), ...})。
        """
        print("    - [触发事件中心 V57.0] 启动，开始定义所有原子化触发事件...")
        
        # 初始化一个字典来存储所有的触发事件
        triggers = {}
        # 获取本模块专属的参数块
        trigger_params = self._get_params_block(params, 'trigger_event_params', {})
        if not self._get_param_value(trigger_params.get('enabled'), True):
            print("      -> 触发事件引擎被禁用，跳过。")
            return triggers

        # --- 准备通用数据和参数 ---
        vol_ma_col = 'VOL_MA_21_D' # 通用成交量均线

        # --- 事件组1: 基于筹码的事件 ---
        # 1.1 “筹码点火”事件 (最强力的启动信号)
        # 定义: 股价上穿市场平均成本，同时伴随着成本峰的加速上移，且发生在吸筹或过渡期之后
        # 注意: 此事件的计算依赖于上游的筹码诊断结果，我们假设它已存在于df中
        if 'CHIP_EVENT' in df.columns:
            triggers['TRIGGER_CHIP_IGNITION'] = (df['CHIP_EVENT'] == 'IGNITION')
            print(f"      -> '筹码点火' 事件定义完成，发现 {triggers['TRIGGER_CHIP_IGNITION'].sum()} 天。")

        # --- 事件组2: 基于K线和量价的事件 ---
        # 2.1 “强势阳线”事件 (通用看涨信号)
        p_candle = trigger_params.get('positive_candle', {})
        if self._get_param_value(p_candle.get('enabled'), True):
            is_green = df['close_D'] > df['open_D']
            body_range = (df['high_D'] - df['low_D']).replace(0, np.nan)
            body_ratio = (df['close_D'] - df['open_D']) / body_range
            is_strong_body = body_ratio > self._get_param_value(p_candle.get('min_body_ratio'), 0.6)
            triggers['TRIGGER_STRONG_POSITIVE_CANDLE'] = is_green & is_strong_body
            print(f"      -> '强势阳线' 事件定义完成，发现 {triggers.get('TRIGGER_STRONG_POSITIVE_CANDLE', pd.Series([])).sum()} 天。")

        # 2.2 “反转确认阳线”事件 (专用于底部反转)
        p_reversal = trigger_params.get('reversal_confirmation_candle', {})
        if self._get_param_value(p_reversal.get('enabled'), True):
            is_green = df['close_D'] > df['open_D']
            is_strong_rally = df['pct_change_D'] > self._get_param_value(p_reversal.get('min_pct_change'), 0.03)
            is_closing_strong = df['close_D'] > (df['high_D'] + df['low_D']) / 2
            triggers['TRIGGER_REVERSAL_CONFIRMATION_CANDLE'] = is_green & is_strong_rally & is_closing_strong
            print(f"      -> '反转确认阳线' 事件定义完成，发现 {triggers.get('TRIGGER_REVERSAL_CONFIRMATION_CANDLE', pd.Series([])).sum()} 天。")

        # 2.3 “放量突破近期高点”事件
        p_breakout = trigger_params.get('volume_spike_breakout', {})
        if self._get_param_value(p_breakout.get('enabled'), True) and vol_ma_col in df.columns:
            volume_ratio = self._get_param_value(p_breakout.get('volume_ratio'), 2.0)
            lookback = self._get_param_value(p_breakout.get('lookback_period'), 20)
            is_volume_spike = df['volume_D'] > df[vol_ma_col] * volume_ratio
            is_price_breakout = df['close_D'] > df['high_D'].shift(1).rolling(lookback).max()
            triggers['TRIGGER_VOLUME_SPIKE_BREAKOUT'] = is_volume_spike & is_price_breakout
            print(f"      -> '放量突破近期高点' 事件定义完成，发现 {triggers.get('TRIGGER_VOLUME_SPIKE_BREAKOUT', pd.Series([])).sum()} 天。")

        # 2.4 “回踩反弹”事件
        p_pullback = trigger_params.get('pullback_rebound_trigger_params', {})
        if self._get_param_value(p_pullback.get('enabled'), True):
            support_ma_period = self._get_param_value(p_pullback.get('support_ma'), 21)
            support_ma_col = f"EMA_{support_ma_period}_D"
            if support_ma_col in df.columns:
                dipped_and_recovered = (df['low_D'] <= df[support_ma_col]) & (df['close_D'] > df[support_ma_col])
                is_green_candle = df['close_D'] > df['open_D']
                has_lower_shadow = (df['open_D'] - df['low_D']) > (df['high_D'] - df['low_D']) * 0.1 # 下影线至少占10%
                is_volume_confirmed = df['volume_D'] > df.get(vol_ma_col, df['volume_D']) * self._get_param_value(p_pullback.get('min_rebound_volume_ratio'), 0.8)
                triggers['TRIGGER_PULLBACK_REBOUND'] = dipped_and_recovered & is_green_candle & has_lower_shadow & is_volume_confirmed
                print(f"      -> '回踩反弹' 事件定义完成，发现 {triggers.get('TRIGGER_PULLBACK_REBOUND', pd.Series([])).sum()} 天。")

        # --- 事件组3: 基于指标交叉的事件 ---
        p_cross = trigger_params.get('indicator_cross_params', {})
        if self._get_param_value(p_cross.get('enabled'), True):
            # 3.1 DMI金叉
            if self._get_param_value(p_cross.get('dmi_cross', {}).get('enabled'), True):
                pdi_col, mdi_col = 'PDI_14_D', 'NDI_14_D'
                if all(c in df.columns for c in [pdi_col, mdi_col]):
                    triggers['TRIGGER_DMI_CROSS'] = (df[pdi_col] > df[mdi_col]) & (df[pdi_col].shift(1) <= df[mdi_col].shift(1))
                    print(f"      -> 'DMI金叉' 事件定义完成，发现 {triggers.get('TRIGGER_DMI_CROSS', pd.Series([])).sum()} 天。")
            
            # 3.2 MACD金叉 (低位)
            macd_p = p_cross.get('macd_cross', {})
            if self._get_param_value(macd_p.get('enabled'), True):
                macd_col, signal_col = 'MACD_13_34_8_D', 'MACDs_13_34_8_D'
                if all(c in df.columns for c in [macd_col, signal_col]):
                    is_golden_cross = (df[macd_col] > df[signal_col]) & (df[macd_col].shift(1) <= df[signal_col].shift(1))
                    low_level = self._get_param_value(macd_p.get('low_level'), -0.5)
                    triggers['TRIGGER_MACD_LOW_CROSS'] = is_golden_cross & (df[macd_col] < low_level)
                    print(f"      -> 'MACD低位金叉' 事件定义完成，发现 {triggers.get('TRIGGER_MACD_LOW_CROSS', pd.Series([])).sum()} 天。")

        # --- 事件组4: 基于结构突破的事件 ---
        box_states = self._diagnose_box_states(df, params)
        triggers['TRIGGER_BOX_BREAKOUT'] = box_states.get('BOX_EVENT_BREAKOUT', pd.Series(False, index=df.index))
        print(f"      -> '箱体突破' 事件定义完成，发现 {triggers.get('TRIGGER_BOX_BREAKOUT', pd.Series([])).sum()} 天。")

        # --- 事件组5: 极端情绪事件 ---
        # 调用板形态诊断模块
        board_events = self._diagnose_board_patterns(df, params)
        
        # 将“地天板”作为一个顶级的买入触发器
        triggers['TRIGGER_EARTH_HEAVEN_BOARD'] = board_events.get('BOARD_EVENT_EARTH_HEAVEN', pd.Series(False, index=df.index))
        print(f"      -> '地天板' 触发事件定义完成，发现 {triggers.get('TRIGGER_EARTH_HEAVEN_BOARD', pd.Series([])).sum()} 天。")

        # --- 步骤4: 最终清洗与返回 ---
        # 确保所有返回的 Series 都已填充 NaN，避免下游出错
        for key in triggers:
            if triggers[key] is None:
                triggers[key] = pd.Series(False, index=df.index)
            else:
                triggers[key] = triggers[key].fillna(False)

        print("    - [触发事件中心 V57.0] 所有触发事件定义完成。")
        return triggers

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

    # 状态机生成器
    def _create_persistent_state(
        self, 
        df: pd.DataFrame, 
        entry_event: pd.Series, 
        persistence_days: int, 
        break_condition: Optional[pd.Series] = None
    ) -> pd.Series:
        """
        【V58.0 新增 - 状态机生成器】
        - 核心功能: 将一个瞬时的布尔型“进入事件”，转化为一个持续N天的“持久化状态”。
        - 解决方案: 利用向量化操作，高效地创建状态窗口。
        - 参数:
          - entry_event: 启动状态的布尔信号 (True代表启动)。
          - persistence_days: 状态的最大持续天数。
          - break_condition: 可选的、提前终止状态的布尔信号 (True代表终止)。
        - 返回: 一个代表持久化状态是否激活的布尔型 pd.Series。
        """
        # 如果没有进入事件，直接返回全False
        if not entry_event.any():
            return pd.Series(False, index=df.index)

        # 1. 使用 cumsum() 为每个由 entry_event 触发的周期创建一个唯一的组ID
        # 当 entry_event 为 True 时，组ID会+1，从而开启一个新组
        event_groups = entry_event.cumsum()
        
        # 2. 在每个组内，计算从事件发生到现在的天数
        # 使用 transform 可以保持原始的DataFrame索引
        days_since_event = df.groupby(event_groups).cumcount()
        
        # 3. 计算状态是否在有效期内
        is_within_persistence = days_since_event < persistence_days
        
        # 4. 筛选出真正由事件触发的周期 (即组ID > 0 的部分)
        is_active_period = event_groups > 0
        
        # 5. 组合基础状态：必须是活跃周期且在持续天数内
        persistent_state = is_active_period & is_within_persistence
        
        # 6. 应用可选的“破坏条件”
        if break_condition is not None:
            # 如果破坏条件为True，则状态必须终止。
            # 我们需要找到每个组内首次出现破坏条件的位置，并将其后的状态置为False。
            # has_broken_in_group = break_condition.groupby(event_groups).cumsum() > 0
            # persistent_state &= ~has_broken_in_group
            
            # 更精确的实现：找到每个组内第一个破坏点
            break_points = break_condition & persistent_state
            break_groups = break_points.groupby(event_groups).transform('idxmax')
            # 如果当前索引大于等于组内的破坏点索引，则状态失效
            persistent_state &= (df.index < break_groups) | (break_groups.isna())

        return persistent_state








