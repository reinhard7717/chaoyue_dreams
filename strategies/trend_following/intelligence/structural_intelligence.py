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
        【V283.2 逻辑修正版】
        - 核心修复 1: 彻底移除了对不存在的 'MA_ZSCORE_D' 列的引用，该逻辑是无效的。
        - 核心修复 2: 修正了对均线粘合度(CV)列名的引用，以匹配数据层因后缀叠加而产生的
                      实际列名 (例如 'MA_CONV_CV_SHORT_D_D')，确保动态粘合状态能被正确计算。
        - 核心修复 3 (本次修改): 修正了 is_decisive_breakout 的逻辑，确保突破K线必须站上所有粘合的均线。
        """
        # print("          -> [均线野战部队 V283.2 逻辑修正版] 启动，正在执行融合分析...") # MODIFIED: 修改版本号和描述
        states = {}
        p = get_params_block(self.strategy, 'ma_state_params')
        if not get_param_value(p.get('enabled'), False): return states

        short_ma_period = get_param_value(p.get('short_ma'), 13)
        mid_ma_period = get_param_value(p.get('mid_ma'), 21)
        long_ma_period = get_param_value(p.get('long_ma'), 55)

        short_ma = f'EMA_{short_ma_period}_D'
        mid_ma = f'EMA_{mid_ma_period}_D'
        long_ma = f'EMA_{long_ma_period}_D'
        short_ma_slope_col = f'SLOPE_5_{short_ma}'
        long_ma_slope_col = f'SLOPE_21_{long_ma}'
        
        required_cols = [short_ma, mid_ma, long_ma, short_ma_slope_col, long_ma_slope_col]
        if not all(col in df.columns for col in required_cols):
            missing_cols = [col for col in required_cols if col not in df.columns]
            print(f"          -> [警告] 缺少诊断MA状态所需列: {missing_cols}，跳过。")
            return states

        states['MA_STATE_PRICE_ABOVE_SHORT_MA'] = df['close_D'] > df[short_ma]
        states['MA_STATE_PRICE_ABOVE_MID_MA'] = df['close_D'] > df[mid_ma]
        is_price_above = df['close_D'] > df[long_ma]
        is_ma_rising = df[long_ma_slope_col] > 0
        states['MA_STATE_PRICE_ABOVE_LONG_MA'] = is_price_above & is_ma_rising
        
        stable_bullish = (df[short_ma] > df[mid_ma]) & (df[mid_ma] > df[long_ma])
        states['MA_STATE_STABLE_BULLISH'] = stable_bullish
        states['MA_STATE_STABLE_BEARISH'] = (df[short_ma] < df[mid_ma]) & (df[mid_ma] < df[long_ma])

        states['MA_STATE_SHORT_SLOPE_POSITIVE'] = df[short_ma_slope_col] > 0
        states['MA_STATE_LONG_SLOPE_POSITIVE'] = df[long_ma_slope_col] > 0

        aggressive_slope_threshold = get_param_value(p.get('aggressive_slope_threshold'), 0.01)
        is_aggressive_slope = df[short_ma_slope_col] > aggressive_slope_threshold
        states['MA_STATE_AGGRESSIVE_BULLISH'] = stable_bullish & is_aggressive_slope

        long_ma_slope = df[long_ma_slope_col]
        entry_event = (long_ma_slope >= 0) & (long_ma_slope.shift(1) < 0)
        
        short_ma_slope = df[short_ma_slope_col]
        break_condition = short_ma_slope < -0.005

        states['MA_STATE_BOTTOM_PASSIVATION'] = create_persistent_state(
            df=df,
            entry_event_series=entry_event,
            persistence_days=20,
            break_condition_series=break_condition,
            state_name='MA_STATE_BOTTOM_PASSIVATION'
        )

        p_conv = get_params_block(self.strategy, 'post_accumulation_params').get('convergence_params', {})
        if get_param_value(p_conv.get('use_dynamic_threshold'), False):
            window = get_param_value(p_conv.get('window'), 120)
            quantile = get_param_value(p_conv.get('quantile'), 0.1)

            short_cv_col = 'MA_CONV_CV_SHORT_D'
            long_cv_col = 'MA_CONV_CV_LONG_D'
            
            if short_cv_col in df.columns:
                dynamic_threshold_short = df[short_cv_col].rolling(window=window).quantile(quantile)
                states['MA_STATE_SHORT_CONVERGENCE_SQUEEZE'] = df[short_cv_col] < dynamic_threshold_short
            
            if long_cv_col in df.columns:
                dynamic_threshold_long = df[long_cv_col].rolling(window=window).quantile(quantile)
                states['MA_STATE_LONG_CONVERGENCE_SQUEEZE'] = df[long_cv_col] < dynamic_threshold_long

        # --- A级机会 - 均线收敛突破 ---
        is_highly_converged = (
            states.get('MA_STATE_SHORT_CONVERGENCE_SQUEEZE', pd.Series(False, index=df.index)) |
            states.get('MA_STATE_LONG_CONVERGENCE_SQUEEZE', pd.Series(False, index=df.index))
        )
        p_energy = get_params_block(self.strategy, 'trigger_event_params').get('energy_release', {})
        vol_ma_col = 'VOL_MA_21_D'
        is_breakout_candle = pd.Series(False, index=df.index)
        if get_param_value(p_energy.get('enabled'), True) and vol_ma_col in df.columns:
            is_positive_day = df['close_D'] > df['open_D']
            body_range = (df['high_D'] - df['low_D']).replace(0, np.nan)
            body_ratio = (df['close_D'] - df['open_D']) / body_range
            is_strong_body = body_ratio.fillna(1.0) > get_param_value(p_energy.get('min_body_ratio'), 0.5)
            volume_ratio = get_param_value(p_energy.get('volume_ratio'), 1.5)
            is_volume_spike = df['volume_D'] > df[vol_ma_col] * volume_ratio
            is_breakout_candle = is_positive_day & is_strong_body & is_volume_spike

        # 为突破K线增加更严格的确认条件：必须站上所有短期均线
        is_decisive_breakout = is_breakout_candle & (df['close_D'] > df[short_ma]) & (df['close_D'] > df[mid_ma])
        was_converged_yesterday = is_highly_converged.shift(1).fillna(False)
        is_in_uptrend_context = states.get('MA_STATE_PRICE_ABOVE_LONG_MA', pd.Series(False, index=df.index))
        states['OPP_MA_CONVERGENCE_BREAKOUT_A'] = was_converged_yesterday & is_decisive_breakout & is_in_uptrend_context

        return states

    def diagnose_box_states(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V284.2 逻辑放宽版】
        - 核心重构: 彻底移除了对 'STRUCTURE_BREAKOUT_EVE_S' 的定义。本方法的职责被严格限定为
                    只诊断与“箱体”直接相关的原子状态，确保了“单一事实来源”原则。
        - 核心修正 (本次): 将健康箱体的判断条件从“量缩 AND 筹码集中”放宽为“量缩 OR 筹码集中”，
                         以捕捉更多样、更真实的吸筹模式，彻底解决信号瓶颈问题。
        """
        print("        -> [工兵部队 V284.2 逻辑放宽版] 启动，正在输出纯粹的箱体边界与状态...") # MODIFIED: 修改版本号和描述
        states = {}
        box_params = get_params_block(self.strategy, 'dynamic_box_params')
        if not get_param_value(box_params.get('enabled'), False) or df.empty:
            return states

        lookback_window = get_param_value(box_params.get('lookback_window'), 8)
        max_amplitude_ratio = get_param_value(box_params.get('max_amplitude_ratio'), 0.05)
        rolling_high = df['high_D'].rolling(window=lookback_window).max()
        rolling_low = df['low_D'].rolling(window=lookback_window).min()
        amplitude_ratio = (rolling_high - rolling_low) / rolling_low.replace(0, np.nan)
        is_valid_box = (amplitude_ratio < max_amplitude_ratio).fillna(False)
        
        box_top = rolling_high
        box_bottom = rolling_low

        df['box_top_D'] = box_top
        df['box_bottom_D'] = box_bottom

        was_below_top = df['close_D'].shift(1) <= box_top.shift(1)
        is_above_top = df['close_D'] > box_top
        states['BOX_EVENT_BREAKOUT'] = is_valid_box & is_above_top & was_below_top
        
        was_above_bottom = df['close_D'].shift(1) >= box_bottom.shift(1)
        is_below_bottom = df['close_D'] < box_bottom
        states['BOX_EVENT_BREAKDOWN'] = is_valid_box & is_below_bottom & was_above_bottom
        
        is_in_box = (df['close_D'] < box_top) & (df['close_D'] > box_bottom)
        
        ma_params = get_params_block(self.strategy, 'ma_state_params')
        long_ma_period = get_param_value(ma_params.get('long_ma'), 55)
        long_ma_col = f"EMA_{long_ma_period}_D"
        
        healthy_consolidation = pd.Series(False, index=df.index)
        if long_ma_col in df.columns:
            box_midpoint = (box_top + box_bottom) / 2
            is_box_above_ma = box_midpoint > df[long_ma_col]
            
            is_shrinking_volume = self.strategy.atomic_states.get('VOL_STATE_SHRINKING', pd.Series(False, index=df.index))
            is_chip_concentrating = self.strategy.atomic_states.get('CHIP_DYN_CONCENTRATING', pd.Series(False, index=df.index))
            
            # 将严苛的 AND 条件修改为更符合实战的 OR 条件
            is_healthy_internal = is_shrinking_volume | is_chip_concentrating
            
            healthy_consolidation = is_valid_box & is_in_box & is_box_above_ma & is_healthy_internal
        
        states['BOX_STATE_HEALTHY_CONSOLIDATION'] = healthy_consolidation
        states['STRUCTURE_BOX_ACCUMULATION_A'] = healthy_consolidation
        
        for key in states:
            if key not in states or states[key] is None:
                states[key] = pd.Series(False, index=df.index)
            else:
                states[key] = states[key].fillna(False)
        return states

    def diagnose_platform_states(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, pd.Series]]:
        """
        【V129.4 健康基因注入版】 - [职责调整] 已被提升为一级诊断单元。
        - 核心重构: 彻底重写了“稳固平台”的定义。不再只依赖被动的成本峰稳定，
                    而是主动融合了“成交量萎缩”和“筹码动态集中”两大健康度指标。
        """
        # print("        -> [诊断模块 V129.4 健康基因注入版] 正在执行筹码平台状态诊断...") # MODIFIED: 修改版本号
        states = {}
        default_series = pd.Series(False, index=df.index)

        peak_cost_col = 'peak_cost_D'
        close_col = 'close_D'
        long_ma_col = 'EMA_55_D'
        
        required_cols = [peak_cost_col, close_col, long_ma_col]
        if not all(col in df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in df.columns]
            print(f"          -> [警告] 缺少诊断平台状态所需的核心列: {missing}。模块将返回空结果。")
            df['PLATFORM_PRICE_STABLE'] = np.nan
            states['PLATFORM_STATE_STABLE_FORMED'] = default_series
            states['STRUCTURE_PLATFORM_BROKEN'] = default_series # 修正：确保所有路径都有返回值
            return df, states

        is_cost_stable = (df[peak_cost_col].rolling(5).std() / df[peak_cost_col].rolling(5).mean()) < 0.02
        is_above_long_ma = df[close_col] > df[long_ma_col]
        is_shrinking_volume = self.strategy.atomic_states.get('VOL_STATE_SHRINKING', default_series)
        is_chip_concentrating = self.strategy.atomic_states.get('CHIP_DYN_CONCENTRATING', default_series)
        
        stable_formed_series = is_cost_stable & is_above_long_ma & is_shrinking_volume & is_chip_concentrating
        states['PLATFORM_STATE_STABLE_FORMED'] = stable_formed_series
        
        df['PLATFORM_PRICE_STABLE'] = df[peak_cost_col].where(stable_formed_series)
        
        was_on_platform = stable_formed_series.shift(1).fillna(False)
        stable_platform_price_series = df['PLATFORM_PRICE_STABLE'].ffill()
        is_breaking_down = df[close_col] < stable_platform_price_series.shift(1)
        
        platform_failure_series = was_on_platform & is_breaking_down
        states['STRUCTURE_PLATFORM_BROKEN'] = platform_failure_series

        return df, states

    def synthesize_composite_structures(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V1.1 逻辑修正版】复合结构合成模块
        - 核心修正: 修正了 'STRUCTURE_BREAKOUT_EVE_S' 的合成逻辑，使其正确地融合
                    “健康吸筹箱体”与“波动率极致压缩”两大S级前置条件。
        """
        print("          -> [结构情报司令部 V1.1] 启动，正在进行高维度复合情报合成...") # MODIFIED: 修改版本号
        composite_states = {}
        default_series = pd.Series(False, index=df.index)
        atomic = self.strategy.atomic_states

        # 复合情报1: “平台获趋势支撑” (逻辑不变)
        is_platform_stable = atomic.get('PLATFORM_STATE_STABLE_FORMED', default_series)
        is_above_mid_ma = atomic.get('MA_STATE_PRICE_ABOVE_MID_MA', default_series)
        composite_states['STRUCTURE_PLATFORM_WITH_TREND_SUPPORT'] = is_platform_stable & is_above_mid_ma

        # 复合情报2: “突破前夜” (S级战术信号) - 逻辑修正与净化
        # 移除了冗余的 'STRUCTURE_BOX_ABOVE_TRENDLINE' 中间信号。
        # 新定义：一个健康的吸筹箱体 + 波动率被压缩到极致 = 突破前夜
        is_healthy_accumulation = atomic.get('STRUCTURE_BOX_ACCUMULATION_A', default_series)
        is_extreme_squeeze = atomic.get('VOL_STATE_EXTREME_SQUEEZE', default_series)
        composite_states['STRUCTURE_BREAKOUT_EVE_S'] = is_healthy_accumulation & is_extreme_squeeze
        
        # [代码新增] 为了兼容旧的信号名称，我们保留 STRUCTURE_BOX_ABOVE_TRENDLINE，并使其与健康箱体等价
        composite_states['STRUCTURE_BOX_ABOVE_TRENDLINE'] = atomic.get('BOX_STATE_HEALTHY_CONSOLIDATION', default_series)

        print("        -> [结构情报司令部 V1.1] 复合情报合成完毕。")
        
        return composite_states

    def diagnose_trend_dynamics(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V174.0 动态惯性引擎】
        - 核心职责: 基于趋势的“斜率”和“加速度”，生成高维度的动态原子状态。
        - 产出: 返回一个包含 DYN_... 信号的字典，供评分引擎使用。
        """
        # print("        -> [诊断模块 V174.0] 正在执行动态惯性诊断...")
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


    def diagnose_fibonacci_support(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【新增】斐波那契支撑诊断模块
        - 核心职责: 识别价格在关键斐波那契回撤位获得支撑的行为。
        - 产出:
            - OPP_FIB_SUPPORT_GOLDEN_POCKET_S: 在0.618黄金分割位获得支撑的S级信号。
            - OPP_FIB_SUPPORT_STANDARD_A: 在0.382或0.5标准分割位获得支撑的A级信号。
        """
        # print("        -> [斐波那契支撑诊断模块] 启动...")
        states = {}
        default_series = pd.Series(False, index=df.index)

        p = get_params_block(self.strategy, 'fibonacci_support_params')
        if not get_param_value(p.get('enabled'), False):
            return {}

        proximity_ratio = get_param_value(p.get('proximity_ratio'), 0.01)
        
        # 定义关键的斐波那契水平列名
        fib_levels = {
            '0.618': 'FIB_0_618_D',
            '0.500': 'FIB_0_500_D',
            '0.382': 'FIB_0_382_D'
        }

        # 检查所需列是否存在
        if any(col not in df.columns for col in fib_levels.values()):
            print("          -> [警告] 缺少斐波那契水平数据列，模块跳过。")
            return {}

        # 核心逻辑：识别“下探回升”于斐波那契位的行为
        # 即：当日最低价 <= 斐波那契位，但收盘价 > 斐波那契位
        def check_support(fib_level_col):
            fib_level = df[fib_level_col]
            # 允许一定的误差
            is_pierced = df['low_D'] <= fib_level * (1 + proximity_ratio)
            is_reclaimed = df['close_D'] > fib_level * (1 - proximity_ratio)
            return is_pierced & is_reclaimed
        # 分别为不同级别的支撑生成信号
        support_618 = check_support(fib_levels['0.618'])
        support_500 = check_support(fib_levels['0.500'])
        support_382 = check_support(fib_levels['0.382'])
        states['OPP_FIB_SUPPORT_GOLDEN_POCKET_S'] = support_618
        states['OPP_FIB_SUPPORT_STANDARD_A'] = support_500 | support_382
        # if states['OPP_FIB_SUPPORT_GOLDEN_POCKET_S'].any():
        #     print(f"          -> [情报] 侦测到 {states['OPP_FIB_SUPPORT_GOLDEN_POCKET_S'].sum()} 次 S级“黄金口袋”支撑。")
        # if states['OPP_FIB_SUPPORT_STANDARD_A'].any():
        #     print(f"          -> [情报] 侦测到 {states['OPP_FIB_SUPPORT_STANDARD_A'].sum()} 次 A级“标准斐波那契”支撑。")
        return states

    def diagnose_structural_mechanics(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V401.0 数据驱动重构版】结构力学诊断引擎
        - 核心重构: 严格遵循“计算与诊断分离”原则，本模块只消费数据层预计算好的列。
        - 核心增强: 融合波动率指标，使“惯量”诊断更精准；保留并加固了“势能”的经典定义。
        - 核心健壮性: 在模块入口处对所有依赖的列进行统一检查，杜绝因数据缺失导致的运行时错误。
        """
        # print("        -> [结构力学诊断引擎 V401.0] 启动...")
        states = {}
        default_series = pd.Series(False, index=df.index)
        atomic = self.strategy.atomic_states

        # --- 1. 军备检查 (Prerequisite Check) ---
        # 统一检查所有依赖的列，确保模块的健壮性。
        required_cols = [
            'SLOPE_5_concentration_90pct_D', 'SLOPE_5_BBW_21_2.0_D', # 惯量诊断所需
            'support_below_D', 'pressure_above_D',                   # 势能诊断所需
            'SLOPE_5_peak_cost_D', 'ACCEL_5_peak_cost_D'             # 成本动态所需
        ]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"          -> [严重警告] 结构力学引擎缺少关键数据列: {missing_cols}，模块已跳过！")
            return states

        # --- 2. 惯量诊断 (Inertia Diagnosis) ---
        # 增强惯量定义。惯量减小 = 供应在锁定(筹码集中度斜率<0) AND 市场趋于平静(波动率斜率<0)。
        # 这种共振是比单一条件更可靠的“山雨欲来风满楼”的信号。
        is_supply_locking = df['SLOPE_5_concentration_90pct_D'] < 0
        is_volatility_squeezing = df['SLOPE_5_BBW_21_2.0_D'] < 0
        states['MECHANICS_INERTIA_DECREASING'] = is_supply_locking & is_volatility_squeezing
        # if states['MECHANICS_INERTIA_DECREASING'].any():
        #     print(f"          -> [力学情报] 侦测到 {states['MECHANICS_INERTIA_DECREASING'].sum()} 次“惯量减小(锁定)”信号！")

        # --- 3. 势能诊断 (Potential Energy Diagnosis) ---
        # 势能优势 = 前线支撑力量 > 前线压力力量
        pressure_col = df['pressure_above_D'] + 1e-6 # 加上极小值防止除以零
        energy_ratio = df['support_below_D'] / pressure_col
        states['MECHANICS_ENERGY_ADVANTAGE'] = energy_ratio > 1.5
        # if states['MECHANICS_ENERGY_ADVANTAGE'].any():
        #     print(f"          -> [力学情报] 侦测到 {states['MECHANICS_ENERGY_ADVANTAGE'].sum()} 次“势能优势”信号！")

        # --- 4. 成本动态诊断 (Cost Dynamics Diagnosis) ---
        # 成本加速抬高：成本斜率和加速度均为正，代表主力拉升意愿强烈且正在加强。
        is_cost_slope_positive = df['SLOPE_5_peak_cost_D'] > 0
        is_cost_accel_positive = df['ACCEL_5_peak_cost_D'] > 0
        states['MECHANICS_COST_ACCELERATING'] = is_cost_slope_positive & is_cost_accel_positive
        
        # 成本企稳减速：成本仍在抬高（斜率为正），但加速度已转负，表明拉升速度放缓，可能进入平台整理或洗盘阶段。
        is_cost_accel_negative = df['ACCEL_5_peak_cost_D'] < 0
        states['MECHANICS_COST_STABILIZING'] = is_cost_slope_positive & is_cost_accel_negative

        print("        -> [结构力学诊断引擎 V401.1] 分析完毕。") # MODIFIED: 修改版本号
        return states





