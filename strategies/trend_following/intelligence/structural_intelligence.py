# 文件: strategies/trend_following/intelligence/structural_intelligence.py
# 结构情报模块 (均线, 箱体, 平台, 趋势)
import pandas as pd
import numpy as np
from typing import Dict, Tuple
from strategies.trend_following.utils import get_params_block, get_param_value, create_persistent_state

class StructuralIntelligence:
    def __init__(self, strategy_instance, dynamic_thresholds: Dict):
        """
        初始化结构情报模块。
        :param strategy_instance: 策略主实例的引用。
        :param dynamic_thresholds: 动态阈值字典。
        """
        self.strategy = strategy_instance
        self.dynamic_thresholds = dynamic_thresholds

    def diagnose_ma_states(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V7.0 双核诊断版】均线状态诊断引擎
        - 核心职责: 构建一个同时输出“分级布尔信号”和“连续数值评分”的双核诊断系统。
        - 核心升级 (本次重构):
          - [信号体系] 新增基于多维交叉验证的、对称的B/A/S三级信号：
            1. 上升/下跌共振 (多时间周期斜率/加速度交叉验证)
            2. 底部/顶部反转 (同周期静态/斜率/加速度交叉验证)
          - [数值化保留] 保留并优化了V600版的优秀数值化评分体系，作为趋势健康度的量化指标。
          - [数据驱动] 严格基于军械库清单进行诊断，确保逻辑的健壮性。
        """
        print("        -> [诊断模块 V7.0 双核诊断版] 启动...") # 更新版本号和打印信息
        states = {}
        p = get_params_block(self.strategy, 'multi_dim_ma_params')
        if not get_param_value(p.get('enabled'), True): return {}

        # --- 军备检查 (Arsenal Check) ---
        ma_periods = get_param_value(p.get('ma_periods'), [5, 13, 21, 55])
        # 为了逻辑清晰，我们选取与均线周期匹配的斜率和加速度周期进行分析
        ema_cols = {p: f'EMA_{p}_D' for p in ma_periods}
        slope_cols = {p: f'SLOPE_{p}_EMA_{p}_D' if p > 5 else f'SLOPE_5_EMA_{p}_D' for p in ma_periods}
        accel_cols = {p: f'ACCEL_{p}_EMA_{p}_D' if p > 5 else f'ACCEL_5_EMA_{p}_D' for p in ma_periods}
        
        # 动态构建所需列清单
        required_cols = list(ema_cols.values())
        # 确保配置文件中的斜率和加速度列存在
        slope_params = get_params_block(self.strategy, 'slope_params')
        accel_params = get_params_block(self.strategy, 'accel_params')
        
        # 智能检查列是否存在，而不是写死
        for p in ma_periods:
            # 检查斜率列
            found_slope = False
            for lookback in [p, 21, 13, 5]: # 优先匹配同周期，再匹配常用周期
                col_name = f'SLOPE_{lookback}_EMA_{p}_D'
                if col_name in df.columns:
                    slope_cols[p] = col_name
                    required_cols.append(col_name)
                    found_slope = True
                    break
            # 检查加速度列
            found_accel = False
            for lookback in [p, 21, 13, 5]: # 优先匹配同周期，再匹配常用周期
                col_name = f'ACCEL_{lookback}_EMA_{p}_D'
                if col_name in df.columns:
                    accel_cols[p] = col_name
                    required_cols.append(col_name)
                    found_accel = True
                    break

        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"          -> [错误] 均线诊断引擎缺少必要列: {missing_cols}，跳过。")
            return {}

        # === Part 1: 基础条件定义 (Fundamental Conditions) ===
        # 将核心指标的各个维度状态布尔化，作为信号合成的积木
        is_slope_up = {p: df[slope_cols[p]] > 0 for p in ma_periods}
        is_slope_down = {p: df[slope_cols[p]] < 0 for p in ma_periods}
        is_accel_up = {p: df[accel_cols[p]] > 0 for p in ma_periods}
        is_accel_down = {p: df[accel_cols[p]] < 0 for p in ma_periods}

        # === Part 2: 共振信号合成 (多时间周期交叉验证) ===
        # 2.1 上升共振 (Bullish Resonance)
        bullish_slope_count = sum(is_slope_up.values())
        bullish_accel_count = sum(is_accel_up.values())
        # B级: 超过半数均线趋势向上 (战术性看多)
        states['OPP_MA_BULLISH_RESONANCE_B'] = (bullish_slope_count >= len(ma_periods) * 0.75)
        # A级: 所有均线趋势一致向上 (战略性看多)
        states['OPP_MA_BULLISH_RESONANCE_A'] = (bullish_slope_count == len(ma_periods))
        # S级: A级基础上，超过半数均线在加速上涨 (全局性主升浪)
        states['OPP_MA_BULLISH_RESONANCE_S'] = states['OPP_MA_BULLISH_RESONANCE_A'] & (bullish_accel_count >= len(ma_periods) * 0.75)

        # 2.2 下跌共振 (Bearish Resonance)
        bearish_slope_count = sum(is_slope_down.values())
        bearish_accel_count = sum(is_accel_down.values())
        # B级: 超过半数均线趋势向下
        states['RISK_MA_BEARISH_RESONANCE_B'] = (bearish_slope_count >= len(ma_periods) * 0.75)
        # A级: 所有均线趋势一致向下
        states['RISK_MA_BEARISH_RESONANCE_A'] = (bearish_slope_count == len(ma_periods))
        # S级: A级基础上，超过半数均线在加速下跌
        states['RISK_MA_BEARISH_RESONANCE_S'] = states['RISK_MA_BEARISH_RESONANCE_A'] & (bearish_accel_count >= len(ma_periods) * 0.75)

        # === Part 3: 反转信号合成 (同周期多维度交叉验证) ===
        # 3.1 底部反转 (Bottom Reversal)
        setup_bottom = is_slope_down[55] # 环境：长期趋势向下
        # B级(试探性): 环境成立 + 短期趋势(5日)下跌减速
        trigger_b_bottom = is_accel_up[5]
        states['OPP_MA_BOTTOM_REVERSAL_B'] = setup_bottom & trigger_b_bottom
        # A级(确认级): B级基础上 + 短期趋势(5日)已确认反转(斜率转正)
        trigger_a_bottom = trigger_b_bottom & is_slope_up[5]
        states['OPP_MA_BOTTOM_REVERSAL_A'] = setup_bottom & trigger_a_bottom
        # S级(强力级): A级基础上 + 中期趋势(13日或21日)也开始减速或反转
        trigger_s_bottom = trigger_a_bottom & (is_accel_up[13] | is_slope_up[13])
        states['OPP_MA_BOTTOM_REVERSAL_S'] = setup_bottom & trigger_s_bottom

        # 3.2 顶部反转 (Top Reversal)
        setup_top = is_slope_up[55] # 环境：长期趋势向上
        # B级(预警性): 环境成立 + 短期趋势(5日)上涨减速
        trigger_b_top = is_accel_down[5]
        states['RISK_MA_TOP_REVERSAL_B'] = setup_top & trigger_b_top
        # A级(确认级): B级基础上 + 短期趋势(5日)已确认反转(斜率转负)
        trigger_a_top = trigger_b_top & is_slope_down[5]
        states['RISK_MA_TOP_REVERSAL_A'] = setup_top & trigger_a_top
        # S级(强力级): A级基础上 + 中期趋势(13日或21日)也开始减速或反转
        trigger_s_top = trigger_a_top & (is_accel_down[13] | is_slope_down[13])
        states['RISK_MA_TOP_REVERSAL_S'] = setup_top & trigger_s_top

        # === Part 4: [保留并优化] 数值化评分体系 ===
        # 4.1 静态排列评分 (向量化优化)
        short_ma_cols_list = [ema_cols[p] for p in ma_periods[:-1]]
        long_ma_cols_list = [ema_cols[p] for p in ma_periods[1:]]
        alignment_sum = np.sum(df[short_ma_cols_list].values > df[long_ma_cols_list].values, axis=1)
        static_alignment_score = alignment_sum / (len(ma_periods) - 1)
        states['SCORE_MA_STATIC_ALIGNMENT'] = pd.Series(static_alignment_score, index=df.index, dtype=np.float32)

        # 4.2 动态斜率与加速度评分 (归一化处理)
        norm_window = get_param_value(p.get('norm_window'), 120)
        min_periods = max(1, norm_window // 5)
        slope_series_list = [
            df[slope_cols[p]].rolling(window=norm_window, min_periods=min_periods).rank(pct=True).fillna(0.5)
            for p in ma_periods
        ]
        accel_series_list = [
            df[accel_cols[p]].rolling(window=norm_window, min_periods=min_periods).rank(pct=True).fillna(0.5)
            for p in ma_periods
        ]

        # 4.3 合成高级数值信号 (NumPy优化)
        slope_values = np.mean(np.array([s.values for s in slope_series_list]), axis=0)
        states['SCORE_MA_DYN_RESONANCE'] = pd.Series(slope_values, index=df.index, dtype=np.float32)
        accel_values = np.mean(np.array([s.values for s in accel_series_list]), axis=0)
        states['SCORE_MA_ACCEL_RESONANCE'] = pd.Series(accel_values, index=df.index, dtype=np.float32)
        long_term_strength = slope_series_list[-1] # 使用最后一个周期的斜率代表长期强度
        short_term_weakness = 1 - slope_series_list[0] # 使用第一个周期的斜率代表短期弱势
        states['SCORE_MA_DIVERGENCE_RISK'] = (long_term_strength * short_term_weakness).fillna(0.5).astype(np.float32)

        # 4.4 融合生成最终的均线健康总分
        states['SCORE_MA_HEALTH'] = (
            states['SCORE_MA_STATIC_ALIGNMENT'] *
            states['SCORE_MA_DYN_RESONANCE'] *
            states['SCORE_MA_ACCEL_RESONANCE']
        ).astype(np.float32)
        
        print("        -> [诊断模块 V7.0 双核诊断版] 分析完毕。") # 更新打印信息
        return states

    def diagnose_box_states_scores(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V4.0 战备-点火分离式诊断版】箱体突破质量评分引擎
        - 核心重构: 引入“战备质量(Setup)”与“点火质量(Trigger)”分离式评估体系。
        - 核心逻辑: 最终突破分 = (战备质量分) * (点火质量分)，实现优中选优。
        - 战备质量分 (Setup Score):
          - 评估箱体形成过程的“微观环境”。
          - 融合了“波动率压缩程度”与“箱体内资金流向”两个维度。
        - 点火质量分 (Trigger Score):
          - 评估突破瞬间的“宏观确认强度”。
          - 融合了“动态成交量爆发”、“K线实体强度”与“上游多维共振信号”。
        - 新增信号 (数值型):
          - SCORE_BOX_SETUP_QUALITY: 箱体战备质量分 (0-1)。
          - SCORE_BOX_BREAKOUT_S/A/B: S/A/B三级箱体向上突破质量分。
          - SCORE_BOX_BREAKDOWN_S/A/B: S/A/B三级箱体向下突破强度分。
        """
        print("        -> [箱体诊断模块 V4.0 战备-点火分离式诊断版] 启动...") # 更新版本号和打印信息
        states = {}
        p = get_params_block(self.strategy, 'box_state_params')
        if not get_param_value(p.get('enabled'), True) or df.empty:
            return states

        # --- 1. 军备检查与参数获取 ---
        atomic = self.strategy.atomic_states
        required_cols = [
            'high_D', 'low_D', 'close_D', 'open_D', 'volume_D',
            'BBW_21_2.0_D', 'SLOPE_5_CMF_21_D', 'SLOPE_5_volume_D', 'ACCEL_5_volume_D'
        ]
        # 依赖上游S/A/B级共振信号
        required_signals = [
            'OPP_MA_BULLISH_RESONANCE_S', 'OPP_MA_BULLISH_RESONANCE_A', 'OPP_MA_BULLISH_RESONANCE_B',
            'RISK_MA_BEARISH_RESONANCE_S', 'RISK_MA_BEARISH_RESONANCE_A', 'RISK_MA_BEARISH_RESONANCE_B'
        ]
        missing_cols = [col for col in required_cols if col not in df.columns]
        missing_signals = [s for s in required_signals if s not in atomic]
        if missing_cols or missing_signals:
            print(f"          -> [警告] 箱体诊断缺少关键数据: 列{missing_cols}, 信号{missing_signals}。模块已跳过。")
            return {}
        
        high_d, low_d, close_d, open_d, volume_d = df['high_D'], df['low_D'], df['close_D'], df['open_D'], df['volume_D']
        lookback_window = get_param_value(p.get('lookback_window'), 8)
        norm_window = get_param_value(p.get('norm_window'), 60)
        min_periods = max(1, norm_window // 5)

        # --- 2. 计算基础箱体和布尔突破/跌破事件 ---
        box_top = high_d.rolling(window=lookback_window).max()
        box_bottom = low_d.rolling(window=lookback_window).min()
        amplitude_ratio = (box_top - box_bottom) / box_bottom.replace(0, np.nan)
        is_valid_box = (amplitude_ratio < get_param_value(p.get('max_amplitude_ratio'), 0.05)).fillna(False)
        
        df['box_top_D'] = box_top # 兼容下游模块
        df['box_bottom_D'] = box_bottom # 兼容下游模块

        is_breakout = is_valid_box & (close_d > box_top.shift(1)) & (close_d.shift(1) <= box_top.shift(1))
        is_breakdown = is_valid_box & (close_d < box_bottom.shift(1)) & (close_d.shift(1) >= box_bottom.shift(1))

        # --- 3. 战备质量评分 (Setup Score) ---
        # 3.1 波动率压缩分 (越低越好)
        vol_compression_score = 1 - df['BBW_21_2.0_D'].rolling(window=norm_window, min_periods=min_periods).rank(pct=True).fillna(0.5)
        # 3.2 箱体内资金流入趋势分 (越高越好)
        capital_inflow_score = df['SLOPE_5_CMF_21_D'].rolling(window=lookback_window).mean().rolling(window=norm_window, min_periods=min_periods).rank(pct=True).fillna(0.5)
        # 3.3 融合为战备总分
        setup_quality_score = (vol_compression_score * capital_inflow_score).where(is_valid_box, 0.0)
        states['SCORE_BOX_SETUP_QUALITY'] = setup_quality_score.astype(np.float32)

        # --- 4. 点火质量评分 (Trigger Score) ---
        # 4.1 动态成交量爆发分
        vol_slope_score = df['SLOPE_5_volume_D'].rolling(window=norm_window, min_periods=min_periods).rank(pct=True).fillna(0.5)
        vol_accel_score = df['ACCEL_5_volume_D'].rolling(window=norm_window, min_periods=min_periods).rank(pct=True).fillna(0.5)
        volume_thrust_score = vol_slope_score * vol_accel_score
        # 4.2 K线实体强度分
        candle_range = (high_d - low_d).replace(0, np.nan)
        breakout_candle_score = ((close_d - open_d) / candle_range).clip(0, 1).fillna(0.0)
        breakdown_candle_score = ((open_d - close_d) / candle_range).clip(0, 1).fillna(0.0)
        # 4.3 宏观共振确认分
        bullish_resonance_s = atomic.get('OPP_MA_BULLISH_RESONANCE_S', pd.Series(False, index=df.index)).astype(float)
        bullish_resonance_a = atomic.get('OPP_MA_BULLISH_RESONANCE_A', pd.Series(False, index=df.index)).astype(float)
        bullish_resonance_b = atomic.get('OPP_MA_BULLISH_RESONANCE_B', pd.Series(False, index=df.index)).astype(float)
        bearish_resonance_s = atomic.get('RISK_MA_BEARISH_RESONANCE_S', pd.Series(False, index=df.index)).astype(float)
        bearish_resonance_a = atomic.get('RISK_MA_BEARISH_RESONANCE_A', pd.Series(False, index=df.index)).astype(float)
        bearish_resonance_b = atomic.get('RISK_MA_BEARISH_RESONANCE_B', pd.Series(False, index=df.index)).astype(float)

        # --- 5. 融合生成最终质量分与兼容性信号 ---
        # 5.1 向上突破
        breakout_trigger_base_score = volume_thrust_score * breakout_candle_score
        states['SCORE_BOX_BREAKOUT_S'] = (is_breakout * setup_quality_score.shift(1) * breakout_trigger_base_score * bullish_resonance_s).fillna(0.0).astype(np.float32)
        states['SCORE_BOX_BREAKOUT_A'] = (is_breakout * setup_quality_score.shift(1) * breakout_trigger_base_score * bullish_resonance_a).fillna(0.0).astype(np.float32)
        states['SCORE_BOX_BREAKOUT_B'] = (is_breakout * setup_quality_score.shift(1) * breakout_trigger_base_score * bullish_resonance_b).fillna(0.0).astype(np.float32)

        # 5.2 向下突破
        breakdown_trigger_base_score = volume_thrust_score * breakdown_candle_score
        states['SCORE_BOX_BREAKDOWN_S'] = (is_breakdown * setup_quality_score.shift(1) * breakdown_trigger_base_score * bearish_resonance_s).fillna(0.0).astype(np.float32)
        states['SCORE_BOX_BREAKDOWN_A'] = (is_breakdown * setup_quality_score.shift(1) * breakdown_trigger_base_score * bearish_resonance_a).fillna(0.0).astype(np.float32)
        states['SCORE_BOX_BREAKDOWN_B'] = (is_breakdown * setup_quality_score.shift(1) * breakdown_trigger_base_score * bearish_resonance_b).fillna(0.0).astype(np.float32)

        # 5.3 兼容旧版信号
        states['BOX_EVENT_BREAKOUT'] = is_breakout.fillna(False)
        states['BOX_EVENT_BREAKDOWN'] = is_breakdown.fillna(False)
        
        print("        -> [箱体诊断模块 V4.0 战备-点火分离式诊断版] 诊断完毕。") # 更新打印信息
        return states

    def diagnose_platform_states_scores(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, pd.Series]]:
        """
        【V3.0 动态力学诊断版】平台诊断与风险评分引擎
        - 核心重构: 从评估“静态稳定性”升级为诊断“动态蓄势状态”。
        - 核心逻辑: 平台总质量分 = f(稳定性, 成本动能, 成本加速度, 宏观环境)。
        - 新增信号 (数值型):
          - SCORE_PLATFORM_COST_MOMENTUM: 成本动能分，量化平台重心的上移趋势。
          - SCORE_PLATFORM_COST_ACCEL: 成本加速分，捕捉平台启动的早期信号。
          - SCORE_PLATFORM_QUALITY_B/A/S: B/A/S三级平台总质量分。
        - 风险升级: 破位风险分融合了“平台质量”与“破位强度”，评估更精准。
        """
        # print("        -> [诊断模块 V3.0 动态力学诊断版] 启动...") # 更新版本号和打印信息
        states = {}
        p = get_params_block(self.strategy, 'platform_state_params')
        if not get_param_value(p.get('enabled'), True): return df, {}
        
        # --- 1. 军备检查 (Arsenal Check) ---
        atomic = self.strategy.atomic_states
        cost_periods = get_param_value(p.get('cost_dynamic_periods'), [5, 21, 55])
        required_cols = ['peak_cost_D', 'close_D', 'open_D', 'high_D', 'low_D', 'volume_D']
        for period in cost_periods:
            required_cols.extend([
                f'SLOPE_{period}_peak_cost_D',
                f'ACCEL_{period if period > 5 else 5}_peak_cost_D' # 确保有对应的加速度列
            ])
        required_signals = ['SCORE_MA_HEALTH'] # 依赖上游均线健康分
        
        missing_cols = [col for col in required_cols if col not in df.columns]
        missing_signals = [s for s in required_signals if s not in atomic]
        if missing_cols or missing_signals:
            print(f"          -> [警告] 平台诊断缺少关键数据: 列{missing_cols}, 信号{missing_signals}。模块已跳过。")
            # 返回安全默认值
            df['PLATFORM_PRICE_STABLE'] = np.nan
            states['PLATFORM_STATE_STABLE_FORMED'] = pd.Series(False, index=df.index)
            states['SCORE_RISK_PLATFORM_BROKEN_S'] = pd.Series(0.0, index=df.index, dtype=np.float32)
            return df, states

        # --- 2. 计算四维核心评分组件 ---
        norm_window = get_param_value(p.get('norm_window'), 120)
        min_periods = max(1, norm_window // 5)
        
        # 2.1 成本稳定性分 (Stability Score)
        peak_cost = df['peak_cost_D']
        rolling_cost = peak_cost.rolling(get_param_value(p.get('cost_cv_lookback'), 5))
        with np.errstate(divide='ignore', invalid='ignore'):
            coeff_of_variation = (rolling_cost.std() / rolling_cost.mean()).fillna(1.0)
        cv_rank = coeff_of_variation.rolling(window=norm_window, min_periods=min_periods).rank(pct=True).fillna(0.5)
        cost_stability_score = (1 - cv_rank)

        # 2.2 成本动能分 (Momentum Score) 
        cost_slope_series = [
            df[f'SLOPE_{p}_peak_cost_D'].rolling(window=norm_window, min_periods=min_periods).rank(pct=True).fillna(0.5)
            for p in cost_periods
        ]
        cost_momentum_score = pd.Series(np.mean(np.array([s.values for s in cost_slope_series]), axis=0), index=df.index)

        # 2.3 成本加速分 (Acceleration Score) 
        cost_accel_series = [
            df[f'ACCEL_{p if p > 5 else 5}_peak_cost_D'].rolling(window=norm_window, min_periods=min_periods).rank(pct=True).fillna(0.5)
            for p in cost_periods
        ]
        cost_accel_score = pd.Series(np.mean(np.array([s.values for s in cost_accel_series]), axis=0), index=df.index)

        # 2.4 宏观环境分 (Context Score) 
        context_health_score = atomic.get('SCORE_MA_HEALTH', pd.Series(0.5, index=df.index))

        # --- 3. 融合生成B/A/S三级平台质量分 ---
        states['SCORE_PLATFORM_QUALITY_B'] = cost_stability_score.astype(np.float32)
        states['SCORE_PLATFORM_QUALITY_A'] = (cost_stability_score * cost_momentum_score).astype(np.float32)
        states['SCORE_PLATFORM_QUALITY_S'] = (
            cost_stability_score * cost_momentum_score * cost_accel_score * context_health_score
        ).fillna(0.0).astype(np.float32)
        
        # --- 4. 生成兼容旧版的布尔信号与平台价格 ---
        threshold = get_param_value(p.get('final_score_threshold_for_bool'), 0.7)
        # 使用最高质量的S级分数来定义最可靠的平台
        stable_formed_series = states['SCORE_PLATFORM_QUALITY_S'] > threshold
        states['PLATFORM_STATE_STABLE_FORMED'] = stable_formed_series
        states['STRUCTURE_BOX_ACCUMULATION_A'] = stable_formed_series # 兼容信号
        df['PLATFORM_PRICE_STABLE'] = peak_cost.where(stable_formed_series)

        # --- 5. 平台破位风险评分 ---
        was_on_platform = stable_formed_series.shift(1, fill_value=False)
        is_breaking_down = df['close_D'] < df['PLATFORM_PRICE_STABLE'].ffill().shift(1)
        platform_failure_series = was_on_platform & is_breaking_down
        states['STRUCTURE_PLATFORM_BROKEN'] = platform_failure_series
        
        # 5.1 计算破位强度分
        candle_range = (df['high_D'] - df['low_D']).replace(0, np.nan)
        breakdown_candle_score = ((df['open_D'] - df['close_D']) / candle_range).clip(0, 1).fillna(0.0)
        volume_score = df['volume_D'].rolling(window=norm_window, min_periods=min_periods).rank(pct=True).fillna(0.5)
        breakdown_intensity_score = breakdown_candle_score * volume_score
        
        # 5.2 融合生成最终风险分
        platform_quality_yesterday = states['SCORE_PLATFORM_QUALITY_S'].shift(1).fillna(0.0)
        states['SCORE_RISK_PLATFORM_BROKEN_S'] = (
            platform_quality_yesterday * breakdown_intensity_score
        ).where(platform_failure_series, 0.0).astype(np.float32)
        
        # --- 6. 平台向上突破机会评分 ---
        is_breaking_up = df['close_D'] > df['PLATFORM_PRICE_STABLE'].ffill().shift(1) # 定义向上突破事件
        platform_breakout_series = was_on_platform & is_breaking_up # 必须是昨日在平台上，今日突破
        states['STRUCTURE_PLATFORM_BREAKOUT'] = platform_breakout_series # 生成布尔事件信号
        # 计算突破强度分
        breakout_candle_score = ((df['close_D'] - df['open_D']) / candle_range).clip(0, 1).fillna(0.0)
        breakout_intensity_score = breakout_candle_score * volume_score
        # 融合生成最终机会分 (结合昨日平台质量、今日突破强度、今日宏观环境)
        states['SCORE_OPP_PLATFORM_BREAKOUT_S'] = (
            platform_quality_yesterday * breakout_intensity_score * context_health_score
        ).where(platform_breakout_series, 0.0).astype(np.float32)

        # print("        -> [诊断模块 V3.0 动态力学诊断版] 分析完毕。") # 更新打印信息
        return df, states

    def diagnose_fibonacci_support(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V4.0 动态过程诊断版】斐波那契反攻诊断模块
        - 核心重构: 从“静态触碰”模型升级为“动态过程”诊断模型。
        - 核心逻辑: 机会分 = f(战备质量, 确认强度, 宏观环境)。
          - 战备质量 (Setup): 评估价格接近支撑位的速度(势能)与触碰日K线的拒绝强度。
          - 确认强度 (Confirmation): 消费上游的强力反转信号。
          - 宏观环境 (Context): 消费上游的趋势健康总分。
        - 新增信号 (数值型):
          - SCORE_FIB_REBOUND_S/A/B: S/A/B三级反弹机会分，对应不同重要性的斐波那契水平。
        """
        print("        -> [斐波那契反攻诊断模块 V4.0 动态过程诊断版] 启动...") # 更新版本号和打印信息
        states = {}
        p = get_params_block(self.strategy, 'fibonacci_support_params')
        if not get_param_value(p.get('enabled'), True):
            return {}
        
        # --- 1. 军备检查 (Arsenal Check) ---
        atomic = self.strategy.atomic_states
        triggers = self.strategy.trigger_events
        fib_levels = {'S': 'FIB_0_618_D', 'A': 'FIB_0_500_D', 'B': 'FIB_0_382_D'}
        required_cols = list(fib_levels.values()) + ['low_D', 'high_D', 'close_D', 'open_D', 'SLOPE_5_close_D']
        required_signals = ['TRIGGER_DOMINANT_REVERSAL', 'SCORE_MA_HEALTH']
        
        missing_cols = [col for col in required_cols if col not in df.columns]
        missing_signals = [s for s in required_signals if s not in triggers and s not in atomic]
        
        if missing_cols or missing_signals:
            print(f"          -> [警告] 斐波那契诊断缺少关键数据: 列{missing_cols}, 信号{missing_signals}。模块跳过。")
            return {}

        # --- 2. 获取核心动态信号与参数 ---
        confirmation_score = triggers.get('TRIGGER_DOMINANT_REVERSAL', pd.Series(False, index=df.index)).astype(np.float32)
        context_score = atomic.get('SCORE_MA_HEALTH', pd.Series(0.0, index=df.index, dtype=np.float32))
        proximity_ratio = get_param_value(p.get('proximity_ratio'), 0.01)
        norm_window = get_param_value(p.get('norm_window'), 120)
        min_periods = max(1, norm_window // 5)

        # --- 3. 计算“战备质量分” (Setup Score) ---
        # 3.1 下跌速度分 (越快，势能越大，反弹潜力越高)
        approach_velocity_score = (1 - df['SLOPE_5_close_D'].rolling(window=norm_window, min_periods=min_periods).rank(pct=True)).fillna(0.5)
        
        # 3.2 K线拒绝强度分 (下影线越长，拒绝信号越强)
        candle_range = (df['high_D'] - df['low_D']).replace(0, np.nan)
        lower_wick_ratio = ((df[['open_D', 'close_D']].min(axis=1) - df['low_D']) / candle_range).clip(0, 1).fillna(0.0)
        rejection_quality_score = lower_wick_ratio

        # 3.3 融合为总的战备分
        setup_quality_score = (approach_velocity_score * rejection_quality_score)

        # --- 4. 核心逻辑：融合三段式评分 ---
        # 逻辑: 最终分数在“确认日”产生，但它依赖于“接触日”(前一日)的战备质量。
        def calculate_rebound_score(fib_level_col: str) -> pd.Series:
            fib_level = df[fib_level_col]
            # 定义接触日: 当日最低价触及或略微跌破斐波那契水平
            is_contact_today = (df['low_D'] <= fib_level * (1 + proximity_ratio)) & (df['high_D'] >= fib_level * (1 - proximity_ratio))
            
            # 获取接触日的战备质量分
            setup_score_on_contact_day = setup_quality_score.where(is_contact_today, 0.0)
            
            # 最终分数 = 前一日的战备质量 * 今日的确认强度 * 今日的宏观环境
            final_score = setup_score_on_contact_day.shift(1).fillna(0.0) * confirmation_score * context_score
            return final_score

        # --- 5. 为不同级别的支撑生成S/A/B三级分数 ---
        states['SCORE_FIB_REBOUND_S'] = calculate_rebound_score(fib_levels['S']).astype(np.float32)
        states['SCORE_FIB_REBOUND_A'] = calculate_rebound_score(fib_levels['A']).astype(np.float32)
        states['SCORE_FIB_REBOUND_B'] = calculate_rebound_score(fib_levels['B']).astype(np.float32)

        # --- 6. 兼容旧版信号 (可选，但建议保留平滑过渡) ---
        # 将S/A/B级分数合并，用于驱动旧的布尔信号
        max_rebound_score = np.maximum.reduce([
            states['SCORE_FIB_REBOUND_S'].values,
            states['SCORE_FIB_REBOUND_A'].values,
            states['SCORE_FIB_REBOUND_B'].values
        ])
        max_rebound_series = pd.Series(max_rebound_score, index=df.index)
        
        if (max_rebound_series > 0).any():
            print(f"          -> [情报] 侦测到 {(max_rebound_series > 0).sum()} 次斐波那契反弹机会，最高分: {max_rebound_series.max():.2f}。")

        # 为了兼容，可以保留旧的信号名，但其内容由新的数值化逻辑驱动
        states['SCORE_FIB_SUPPORT_GOLDEN_POCKET_S'] = states['SCORE_FIB_REBOUND_S']
        states['SCORE_FIB_SUPPORT_STANDARD_A'] = np.maximum(states['SCORE_FIB_REBOUND_A'], states['SCORE_FIB_REBOUND_B']).astype(np.float32)

        print("        -> [斐波那契反攻诊断模块 V4.0 动态过程诊断版] 分析完毕。") # 更新打印信息
        return states

    def diagnose_structural_mechanics_scores(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V6.0 共振-反转对称诊断版】结构力学诊断引擎
        - 核心重构: 遵循“共振/反转”对称原则，全面升级为多维交叉验证的数值化评分体系。
        - 核心逻辑:
          - 共振信号 (多周期交叉): 评估成本、筹码、势能等要素在多周期上的一致性。
          - 反转信号 (同周期交叉): 评估“静态战备”与“动态点火”的结合。
        - 新增信号 (数值型, 对称设计):
          - SCORE_MECHANICS_BULLISH_RESONANCE_S/A/B: 上升共振机会分。
          - SCORE_MECHANICS_BEARISH_RESONANCE_S/A/B: 下跌共振风险分。
          - SCORE_MECHANICS_BOTTOM_REVERSAL_S/A/B: 底部反转机会分。
          - SCORE_MECHANICS_TOP_REVERSAL_S/A/B: 顶部反转风险分。
        """
        print("        -> [结构力学诊断引擎 V6.0 共振-反转对称诊断版] 启动...") # 更新版本号和打印信息
        states = {}
        p = get_params_block(self.strategy, 'structural_mechanics_params')
        if not get_param_value(p.get('enabled'), True): return {}

        # --- 1. 军备检查 (Arsenal Check) ---
        periods = get_param_value(p.get('dynamic_periods'), [5, 21, 55])
        required_cols = ['energy_ratio_D', 'BBW_21_2.0_D']
        dynamic_sources = {'cost': 'peak_cost_D', 'conc': 'concentration_90pct_D'}
        
        for period in periods:
            required_cols.extend([
                f'SLOPE_{period}_{dynamic_sources["cost"]}',
                f'ACCEL_{period if period > 5 else 5}_{dynamic_sources["cost"]}',
                f'SLOPE_{period}_{dynamic_sources["conc"]}',
            ])
        
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"          -> [严重警告] 结构力学引擎缺少关键数据列: {missing_cols}，模块已跳过！")
            return states

        # --- 2. 核心力学要素数值化 (归一化处理) ---
        norm_window = get_param_value(p.get('norm_window'), 120)
        min_periods = max(1, norm_window // 5)

        def normalize(series):
            return series.rolling(window=norm_window, min_periods=min_periods).rank(pct=True).fillna(0.5)

        # 动态评分组件
        cost_momentum_scores = {p: normalize(df[f'SLOPE_{p}_{dynamic_sources["cost"]}']) for p in periods}
        cost_accel_scores = {p: normalize(df[f'ACCEL_{p if p > 5 else 5}_{dynamic_sources["cost"]}']) for p in periods}
        supply_lock_scores = {p: normalize(df[f'SLOPE_{p}_{dynamic_sources["conc"]}']) for p in periods} # 斜率越大，锁定度越高
        
        # 静态评分组件
        energy_advantage_score = normalize(df['energy_ratio_D'])
        vol_compression_score = 1 - normalize(df['BBW_21_2.0_D']) # BBW越小，压缩度越高

        # --- 3. 共振信号合成 (多时间周期交叉验证) ---
        avg_cost_momentum = pd.Series(np.mean(np.array([s.values for s in cost_momentum_scores.values()]), axis=0), index=df.index)
        avg_supply_lock = pd.Series(np.mean(np.array([s.values for s in supply_lock_scores.values()]), axis=0), index=df.index)

        # 3.1 上升共振 (Bullish Resonance)
        states['SCORE_MECHANICS_BULLISH_RESONANCE_B'] = avg_cost_momentum.astype(np.float32)
        states['SCORE_MECHANICS_BULLISH_RESONANCE_A'] = (avg_cost_momentum * avg_supply_lock).astype(np.float32)
        states['SCORE_MECHANICS_BULLISH_RESONANCE_S'] = (avg_cost_momentum * avg_supply_lock * energy_advantage_score).astype(np.float32)

        # 3.2 下跌共振 (Bearish Resonance) - 对称逻辑
        avg_cost_decline = 1 - avg_cost_momentum
        avg_supply_disperse = 1 - avg_supply_lock
        energy_disadvantage_score = 1 - energy_advantage_score
        states['SCORE_MECHANICS_BEARISH_RESONANCE_B'] = avg_cost_decline.astype(np.float32)
        states['SCORE_MECHANICS_BEARISH_RESONANCE_A'] = (avg_cost_decline * avg_supply_disperse).astype(np.float32)
        states['SCORE_MECHANICS_BEARISH_RESONANCE_S'] = (avg_cost_decline * avg_supply_disperse * energy_disadvantage_score).astype(np.float32)

        # --- 4. 反转信号合成 (同周期多维度交叉验证) ---
        avg_cost_accel = pd.Series(np.mean(np.array([s.values for s in cost_accel_scores.values()]), axis=0), index=df.index)

        # 4.1 底部反转 (Bottom Reversal) = 战备分 * 点火分
        bottom_reversal_setup_score = vol_compression_score * avg_supply_lock # 战备: 波动压缩 + 悄然吸筹
        bottom_reversal_trigger_score = avg_cost_accel # 点火: 成本重心开始加速抬升
        
        states['SCORE_MECHANICS_BOTTOM_REVERSAL_B'] = bottom_reversal_trigger_score.astype(np.float32)
        states['SCORE_MECHANICS_BOTTOM_REVERSAL_A'] = (vol_compression_score * bottom_reversal_trigger_score).astype(np.float32)
        states['SCORE_MECHANICS_BOTTOM_REVERSAL_S'] = (bottom_reversal_setup_score * bottom_reversal_trigger_score).astype(np.float32)

        # 4.2 顶部反转 (Top Reversal) - 对称逻辑
        top_reversal_setup_score = (1 - vol_compression_score) * avg_supply_disperse # 战备: 波动放大 + 悄然派发
        top_reversal_trigger_score = 1 - avg_cost_accel # 点火: 成本重心开始加速下移
        
        states['SCORE_MECHANICS_TOP_REVERSAL_B'] = top_reversal_trigger_score.astype(np.float32)
        states['SCORE_MECHANICS_TOP_REVERSAL_A'] = ((1 - vol_compression_score) * top_reversal_trigger_score).astype(np.float32)
        states['SCORE_MECHANICS_TOP_REVERSAL_S'] = (top_reversal_setup_score * top_reversal_trigger_score).astype(np.float32)

        print("        -> [结构力学诊断引擎 V6.0 共振-反转对称诊断版] 分析完毕。") # 更新打印信息
        return states

    def diagnose_mtf_trend_synergy_scores(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V5.0 共振-反转对称诊断版】战略协同评分引擎
        - 核心重构: 遵循“共振/反转”对称原则，构建一个完整的跨时间框架(MTF)信号矩阵。
        - 核心逻辑:
          - 共振信号: 评估日线与周线在动能(slope)与势能(accel)上的一致性。
          - 反转信号: 评估“周线趋势(环境)”与“日线拐点(触发)”的组合。
        - 新增信号 (数值型, 对称设计):
          - SCORE_MTF_BULLISH_RESONANCE_S/A/B: MTF上升共振机会分。
          - SCORE_MTF_BEARISH_RESONANCE_S/A/B: MTF下跌共振风险分。
          - SCORE_MTF_BOTTOM_REVERSAL_S/A/B: MTF底部反转机会分。
          - SCORE_MTF_TOP_REVERSAL_S/A/B: MTF顶部反转风险分。
        """
        print("        -> [战略协同引擎 V5.0 共振-反转对称诊断版] 启动...") # 更新版本号和打印信息
        states = {}
        p = get_params_block(self.strategy, 'mtf_trend_synergy_params')
        if not get_param_value(p.get('enabled'), True): return {}
        
        # --- 1. 军备检查 (Arsenal Check) ---
        atomic = self.strategy.atomic_states
        required_daily_signals = ['SCORE_MA_DYN_RESONANCE', 'SCORE_MA_ACCEL_RESONANCE']
        # 动态发现所有可用的周线斜率和加速度列
        weekly_slope_cols = [col for col in df.columns if 'SLOPE' in col and col.endswith('_W')]
        weekly_accel_cols = [col for col in df.columns if 'ACCEL' in col and col.endswith('_W')]
        
        missing_signals = [s for s in required_daily_signals if s not in atomic]
        if missing_signals or not weekly_slope_cols or not weekly_accel_cols:
            print(f"          -> [严重警告] 战略协同引擎缺少关键数据: 日线信号{missing_signals}, 周线斜率列(需至少1个), 周线加速列(需至少1个)。模块已跳过！")
            return {}

        # --- 2. 获取/构建日线与周线的核心动态评分 ---
        norm_window = get_param_value(p.get('norm_window'), 120)
        min_periods = max(1, norm_window // 5)
        default_series = pd.Series(0.5, index=df.index, dtype=np.float32)

        # 2.1 获取日线高级分数
        daily_momentum_score = atomic.get('SCORE_MA_DYN_RESONANCE', default_series)
        daily_accel_score = atomic.get('SCORE_MA_ACCEL_RESONANCE', default_series)

        # 2.2 构建周线高级分数
        def get_weekly_score(cols):
            if not cols: return default_series
            series_list = [df[col].rolling(window=norm_window, min_periods=min_periods).rank(pct=True).fillna(0.5) for col in cols]
            score_values = np.mean(np.array([s.values for s in series_list]), axis=0)
            return pd.Series(score_values, index=df.index, dtype=np.float32)

        weekly_momentum_score = get_weekly_score(weekly_slope_cols)
        weekly_accel_score = get_weekly_score(weekly_accel_cols)

        # --- 3. 共振信号合成 (多时间周期交叉验证) ---
        # 3.1 上升共振 (Bullish Resonance)
        bullish_momentum_resonance = daily_momentum_score * weekly_momentum_score
        states['SCORE_MTF_BULLISH_RESONANCE_B'] = bullish_momentum_resonance.astype(np.float32)
        states['SCORE_MTF_BULLISH_RESONANCE_A'] = (bullish_momentum_resonance * daily_accel_score).astype(np.float32)
        states['SCORE_MTF_BULLISH_RESONANCE_S'] = (bullish_momentum_resonance * daily_accel_score * weekly_accel_score).astype(np.float32)

        # 3.2 下跌共振 (Bearish Resonance) - 对称逻辑
        bearish_momentum_resonance = (1 - daily_momentum_score) * (1 - weekly_momentum_score)
        states['SCORE_MTF_BEARISH_RESONANCE_B'] = bearish_momentum_resonance.astype(np.float32)
        states['SCORE_MTF_BEARISH_RESONANCE_A'] = (bearish_momentum_resonance * (1 - daily_accel_score)).astype(np.float32)
        states['SCORE_MTF_BEARISH_RESONANCE_S'] = (bearish_momentum_resonance * (1 - daily_accel_score) * (1 - weekly_accel_score)).astype(np.float32)

        # --- 4. 反转信号合成 (环境 x 拐点) ---
        # 4.1 底部反转 (Bottom Reversal) = 周线看跌环境 x 日线看涨拐点
        weekly_downtrend_setup = 1 - weekly_momentum_score
        daily_bullish_trigger = daily_accel_score
        bottom_reversal_base = weekly_downtrend_setup * daily_bullish_trigger
        
        states['SCORE_MTF_BOTTOM_REVERSAL_B'] = daily_bullish_trigger.astype(np.float32) # B级: 仅看日线拐点
        states['SCORE_MTF_BOTTOM_REVERSAL_A'] = bottom_reversal_base.astype(np.float32) # A级: 结合周线环境
        states['SCORE_MTF_BOTTOM_REVERSAL_S'] = (bottom_reversal_base * (1 - weekly_accel_score)).astype(np.float32) # S级: 周线下跌也在减速

        # 4.2 顶部反转 (Top Reversal) - 对称逻辑
        weekly_uptrend_setup = weekly_momentum_score
        daily_bearish_trigger = 1 - daily_accel_score
        top_reversal_base = weekly_uptrend_setup * daily_bearish_trigger

        states['SCORE_MTF_TOP_REVERSAL_B'] = daily_bearish_trigger.astype(np.float32) # B级: 仅看日线拐点
        states['SCORE_MTF_TOP_REVERSAL_A'] = top_reversal_base.astype(np.float32) # A级: 结合周线环境
        states['SCORE_MTF_TOP_REVERSAL_S'] = (top_reversal_base * weekly_accel_score).astype(np.float32) # S级: 周线上涨也在减速

        print("        -> [战略协同引擎 V5.0 共振-反转对称诊断版] 分析完毕。") # 更新打印信息
        return states

    def diagnose_structural_risks_and_regimes_scores(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V4.0 风险共振-状态诊断版】结构风险与市场状态诊断引擎
        - 核心重构: 遵循“风险共振”与“状态分类”原则，升级为多维交叉验证的数值化评分体系。
        - 核心逻辑:
          - 风险信号: 融合“静态风险”与“动态风险加剧”的程度，形成风险共振分。
          - 状态信号: 对市场的波动环境进行分类，识别“压缩”与“扩张”状态。
        - 新增信号 (数值型, 对称设计):
          - SCORE_RISK_..._S/A/B: 价格乖离、获利盘兑现、结构破损三大风险分。
          - SCORE_REGIME_..._S/A/B: 波动压缩、波动扩张两大市场状态分。
        """
        print("        -> [结构风险与状态引擎 V4.0 风险共振-状态诊断版] 启动...") # 更新版本号和打印信息
        states = {}
        p = get_params_block(self.strategy, 'structural_risks_params')
        if not get_param_value(p.get('enabled'), True): return {}

        # --- 1. 军备检查 (Arsenal Check) ---
        required_cols = [
            'price_to_peak_ratio_D', 'SLOPE_5_price_to_peak_ratio_D',
            'winner_profit_margin_D', 'SLOPE_5_winner_profit_margin_D',
            'chip_health_score_D', 'SLOPE_5_chip_health_score_D', 'is_chip_fault_formed_D',
            'BBW_21_2.0_D', 'SLOPE_5_BBW_21_2.0_D', 'ACCEL_5_BBW_21_2.0_D',
            'SLOPE_5_concentration_90pct_D', 'SLOPE_21_concentration_90pct_D',
        ]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"          -> [严重警告] 结构风险引擎缺少关键数据列: {missing_cols}，模块已跳过！")
            return states

        # --- 2. 核心要素数值化 (归一化处理) ---
        norm_window = get_param_value(p.get('norm_window'), 120)
        min_periods = max(1, norm_window // 5)

        def normalize(series):
            return series.rolling(window=norm_window, min_periods=min_periods).rank(pct=True).fillna(0.5)
        
        def normalize_inverse(series):
            return 1 - normalize(series)

        # 风险要素
        price_deviation_static = normalize(df['price_to_peak_ratio_D'])
        price_deviation_dynamic = normalize(df['SLOPE_5_price_to_peak_ratio_D'])
        profit_exhaustion_static = normalize(df['winner_profit_margin_D'])
        profit_exhaustion_dynamic = normalize(df['SLOPE_5_winner_profit_margin_D'])
        structural_health_static = normalize(df['chip_health_score_D'])
        structural_health_dynamic = normalize(df['SLOPE_5_chip_health_score_D'])
        is_fault_formed = df['is_chip_fault_formed_D'].astype(float)

        # 状态要素
        vol_static = normalize(df['BBW_21_2.0_D'])
        vol_dynamic = normalize(df['SLOPE_5_BBW_21_2.0_D'])
        vol_accel = normalize(df['ACCEL_5_BBW_21_2.0_D'])

        # --- 3. 风险信号合成 (多维度风险共振) ---
        # 3.1 价格乖离风险
        states['SCORE_RISK_PRICE_DEVIATION_B'] = price_deviation_static.astype(np.float32)
        states['SCORE_RISK_PRICE_DEVIATION_A'] = (price_deviation_static * price_deviation_dynamic).astype(np.float32)
        states['SCORE_RISK_PRICE_DEVIATION_S'] = (states['SCORE_RISK_PRICE_DEVIATION_A'] * vol_static * vol_dynamic).astype(np.float32)

        # 3.2 获利盘兑现风险
        structural_deterioration = 1 - structural_health_dynamic
        states['SCORE_RISK_PROFIT_EXHAUSTION_B'] = profit_exhaustion_static.astype(np.float32)
        states['SCORE_RISK_PROFIT_EXHAUSTION_A'] = (profit_exhaustion_static * profit_exhaustion_dynamic).astype(np.float32)
        states['SCORE_RISK_PROFIT_EXHAUSTION_S'] = (states['SCORE_RISK_PROFIT_EXHAUSTION_A'] * structural_deterioration).astype(np.float32)

        # 3.3 结构破损风险
        structural_unhealth = 1 - structural_health_static
        states['SCORE_RISK_STRUCTURAL_FAULT_B'] = structural_unhealth.astype(np.float32)
        states['SCORE_RISK_STRUCTURAL_FAULT_A'] = (structural_unhealth * structural_deterioration).astype(np.float32)
        states['SCORE_RISK_STRUCTURAL_FAULT_S'] = (states['SCORE_RISK_STRUCTURAL_FAULT_A'] * is_fault_formed).astype(np.float32)
        # --- 筹码结构衰退风险 ---
        is_short_term_diverging = df['SLOPE_5_concentration_90pct_D'] > 0
        is_mid_term_diverging = df['SLOPE_21_concentration_90pct_D'] > 0
        states['CONTEXT_RISK_CHIP_STRUCTURE_DECAY'] = is_short_term_diverging & is_mid_term_diverging

        # --- 4. 状态信号合成 (市场环境分类) ---
        # 4.1 波动压缩状态
        vol_compression_static = 1 - vol_static
        vol_compression_dynamic = 1 - vol_dynamic
        vol_compression_accel = 1 - vol_accel
        states['SCORE_REGIME_VOL_COMPRESSION_B'] = vol_compression_static.astype(np.float32)
        states['SCORE_REGIME_VOL_COMPRESSION_A'] = (vol_compression_static * vol_compression_dynamic).astype(np.float32)
        states['SCORE_REGIME_VOL_COMPRESSION_S'] = (states['SCORE_REGIME_VOL_COMPRESSION_A'] * vol_compression_accel).astype(np.float32)

        # 4.2 波动扩张状态 - 对称逻辑
        vol_expansion_static = vol_static
        vol_expansion_dynamic = vol_dynamic
        vol_expansion_accel = vol_accel
        states['SCORE_REGIME_VOL_EXPANSION_B'] = vol_expansion_static.astype(np.float32)
        states['SCORE_REGIME_VOL_EXPANSION_A'] = (vol_expansion_static * vol_expansion_dynamic).astype(np.float32)
        states['SCORE_REGIME_VOL_EXPANSION_S'] = (states['SCORE_REGIME_VOL_EXPANSION_A'] * vol_expansion_accel).astype(np.float32)

        print("        -> [结构风险与状态引擎 V4.0 风险共振-状态诊断版] 分析完毕。") # 更新打印信息
        return states

    def diagnose_advanced_structural_patterns_scores(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V3.0 模式交叉验证版】高级结构模式诊断引擎
        - 核心重构: 遵循“共振/反转”对称原则，将离散的模式信号，通过与连续的动能/环境信号交叉验证，升级为数值化的置信度评分。
        - 核心逻辑: 模式置信度分 = f(模式识别信号, 趋势动能信号, 市场环境信号)。
        - 新增信号 (数值型, 对称设计):
          - SCORE_PATTERN_BULLISH_RESONANCE_S/A/B: 看涨模式(突破等)的确认分。
          - SCORE_PATTERN_BEARISH_RESONANCE_S/A/B: 看跌模式(跌破等)的确认分。
          - SCORE_PATTERN_BOTTOM_REVERSAL_S/A/B: 底部模式(吸筹等)的确认分。
          - SCORE_PATTERN_TOP_REVERSAL_S/A/B: 顶部模式(派发等)的确认分。
          - SCORE_PATTERN_CONSOLIDATION_S/A/B: 盘整中继模式的确认分。
        """
        print("        -> [高级结构模式引擎 V3.0 模式交叉验证版] 启动...") # 更新版本号和打印信息
        states = {}
        p = get_params_block(self.strategy, 'advanced_patterns_params')
        if not get_param_value(p.get('enabled'), True): return {}
        atomic = self.strategy.atomic_states

        # --- 1. 军备检查 (Arsenal Check) ---
        # 1.1 检查数据层的模式信号
        required_pattern_cols = [
            'is_breakthrough_D', 'is_breakdown_D', 'is_accumulation_D',
            'is_distribution_D', 'is_consolidation_D'
        ]
        missing_cols = [col for col in required_pattern_cols if col not in df.columns]
        if missing_cols:
            print(f"          -> [严重警告] 高级模式引擎缺少关键数据列: {missing_cols}，模块已跳过！")
            return states
        
        # 1.2 检查上游模块的确认信号
        required_atomic_scores = [
            'SCORE_MA_DYN_RESONANCE', 'SCORE_MA_ACCEL_RESONANCE',
            'SCORE_REGIME_VOL_COMPRESSION_S', 'SCORE_REGIME_VOL_EXPANSION_S'
        ]
        missing_scores = [s for s in required_atomic_scores if s not in atomic]
        if missing_scores:
            print(f"          -> [严重警告] 高级模式引擎缺少上游确认分数: {missing_scores}，模块已跳过！")
            return states

        # --- 2. 获取核心信号并数值化 ---
        # 2.1 模式信号 (布尔 -> 浮点)
        is_breakthrough = df.get('is_breakthrough_D', 0).astype(float)
        is_breakdown = df.get('is_breakdown_D', 0).astype(float)
        is_accumulation = df.get('is_accumulation_D', 0).astype(float)
        is_distribution = df.get('is_distribution_D', 0).astype(float)
        is_consolidation = df.get('is_consolidation_D', 0).astype(float)
        
        # 2.2 确认信号
        daily_momentum_score = atomic['SCORE_MA_DYN_RESONANCE']
        daily_accel_score = atomic['SCORE_MA_ACCEL_RESONANCE']
        vol_compression_score = atomic['SCORE_REGIME_VOL_COMPRESSION_S']
        vol_expansion_score = atomic['SCORE_REGIME_VOL_EXPANSION_S']

        # --- 3. 模式信号交叉验证与评分 ---
        # 3.1 上升共振模式 (突破)
        bullish_pattern_base = is_breakthrough # 未来可扩展: max(is_breakthrough, is_uptrend_channel)
        states['SCORE_PATTERN_BULLISH_RESONANCE_B'] = bullish_pattern_base.astype(np.float32)
        states['SCORE_PATTERN_BULLISH_RESONANCE_A'] = (bullish_pattern_base * daily_momentum_score).astype(np.float32)
        states['SCORE_PATTERN_BULLISH_RESONANCE_S'] = (states['SCORE_PATTERN_BULLISH_RESONANCE_A'] * vol_expansion_score).astype(np.float32)

        # 3.2 下跌共振模式 (跌破) - 对称逻辑
        bearish_pattern_base = is_breakdown
        states['SCORE_PATTERN_BEARISH_RESONANCE_B'] = bearish_pattern_base.astype(np.float32)
        states['SCORE_PATTERN_BEARISH_RESONANCE_A'] = (bearish_pattern_base * (1 - daily_momentum_score)).astype(np.float32)
        states['SCORE_PATTERN_BEARISH_RESONANCE_S'] = (states['SCORE_PATTERN_BEARISH_RESONANCE_A'] * vol_expansion_score).astype(np.float32)

        # 3.3 底部反转模式 (吸筹)
        bottom_reversal_base = is_accumulation
        states['SCORE_PATTERN_BOTTOM_REVERSAL_B'] = bottom_reversal_base.astype(np.float32)
        states['SCORE_PATTERN_BOTTOM_REVERSAL_A'] = (bottom_reversal_base * vol_compression_score).astype(np.float32)
        states['SCORE_PATTERN_BOTTOM_REVERSAL_S'] = (states['SCORE_PATTERN_BOTTOM_REVERSAL_A'] * daily_accel_score).astype(np.float32)

        # 3.4 顶部反转模式 (派发) - 对称逻辑
        top_reversal_base = is_distribution
        states['SCORE_PATTERN_TOP_REVERSAL_B'] = top_reversal_base.astype(np.float32)
        states['SCORE_PATTERN_TOP_REVERSAL_A'] = (top_reversal_base * vol_compression_score).astype(np.float32)
        states['SCORE_PATTERN_TOP_REVERSAL_S'] = (states['SCORE_PATTERN_TOP_REVERSAL_A'] * (1 - daily_accel_score)).astype(np.float32)

        # 3.5 盘整中继模式 (独立)
        consolidation_base = is_consolidation
        # 动能中性分: 越接近0.5分越高, 用 1 - 2 * abs(score - 0.5) 计算
        momentum_neutrality = 1 - 2 * abs(daily_momentum_score - 0.5)
        states['SCORE_PATTERN_CONSOLIDATION_B'] = consolidation_base.astype(np.float32)
        states['SCORE_PATTERN_CONSOLIDATION_A'] = (consolidation_base * vol_compression_score).astype(np.float32)
        states['SCORE_PATTERN_CONSOLIDATION_S'] = (states['SCORE_PATTERN_CONSOLIDATION_A'] * momentum_neutrality).astype(np.float32)

        print("        -> [高级结构模式引擎 V3.0 模式交叉验证版] 分析完毕。") # 更新打印信息
        return states

    def diagnose_fused_behavioral_structure_risks(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V1.0】行为与结构融合风险诊断模块
        - 核心职责: 承接原 CognitiveIntelligence 中的复杂风险计算逻辑，融合行为、价格、
                      筹码等多维度信息，生成高质量的原子风险信号。
        - 收益: 净化了认知层的职责，确保复杂的原子信号在正确的情报层生成。
        """
        print("        -> [行为-结构融合风险模块 V1.0] 启动...")
        states = {}
        atomic = self.strategy.atomic_states
        default_series = pd.Series(False, index=df.index)
        # --- 1. 军备检查 ---
        required_cols = [
            'close_D', 'high_D', 'SLOPE_5_volume_D',
            'SLOPE_5_VPA_EFFICIENCY_D', 'SLOPE_5_concentration_90pct_D'
        ]
        required_scores = [
            'SCORE_PROFIT_TAKING_TOP_REVERSAL_S',
        ]
        missing_cols = [c for c in required_cols if c not in df.columns]
        missing_scores = [s for s in required_scores if s not in atomic]
        if missing_cols or missing_scores:
            print(f"          -> [警告] 行为-结构融合风险模块缺少关键数据: 列{missing_cols}, 分数{missing_scores}。模块已跳过。")
            return states
        # --- 2. 计算“动能衰竭”风险 (CONTEXT_RISK_MOMENTUM_EXHAUSTION) ---
        is_at_high_price = df['close_D'] > df['high_D'].rolling(60).max() * 0.85
        is_profit_cushion_shrinking = atomic.get('SCORE_PROFIT_TAKING_TOP_REVERSAL_S', default_series) > 0.8
        is_market_engine_stalling = atomic.get('RISK_BEHAVIOR_MARKET_ENGINE_STALLING_B', default_series)
        states['CONTEXT_RISK_MOMENTUM_EXHAUSTION'] = is_at_high_price & (is_profit_cushion_shrinking | is_market_engine_stalling)
        # --- 3. 计算“动态对倒嫌疑”风险 (COGNITIVE_RISK_DYNAMIC_DECEPTIVE_CHURN) ---
        is_volume_increasing = df['SLOPE_5_volume_D'] > 0
        is_vpa_efficiency_declining = df['SLOPE_5_VPA_EFFICIENCY_D'] < 0
        is_chip_diverging_for_churn = df['SLOPE_5_concentration_90pct_D'] > 0
        states['COGNITIVE_RISK_DYNAMIC_DECEPTIVE_CHURN'] = is_volume_increasing & is_vpa_efficiency_declining & is_chip_diverging_for_churn
        print("        -> [行为-结构融合风险模块 V1.0] 分析完毕。")
        return states











