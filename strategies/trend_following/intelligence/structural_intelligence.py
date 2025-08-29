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
        【V283.4 风险哨兵加固版】
        - 核心修复: 为胜率仅12.1%的“稳定多头排列”信号，增加了“筹码集中”或“动能健康”
                      的双重确认过滤器。
        - 核心加固 (本次修改): 为“稳定空头排列”增加了“长期均线斜率必须为负”的硬性约束，
                        将其从一个形态信号，升级为一个确认的趋势风险信号。
        - 收益: 彻底解决了该信号在趋势末期或震荡市中频繁发出假信号的致命缺陷。
                新的定义确保了我们只在“形神兼备”的真正健康趋势中确认多头状态，
                并只在趋势确认的空头行情中发出风险预警。
        """
        # print("          -> [均线野战部队 V283.4 风险哨兵加固版] 启动，正在执行融合分析...")
        states = {}
        p = get_params_block(self.strategy, 'ma_state_params')
        if not get_param_value(p.get('enabled'), False): return states
        default_series = pd.Series(False, index=df.index)
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
        is_ma_arrangement_bullish = (df[short_ma] > df[mid_ma]) & (df[mid_ma] > df[long_ma])
        is_chip_concentrating = self.strategy.atomic_states.get('CHIP_DYN_CONCENTRATING', default_series)
        is_trend_healthy_accel = self.strategy.atomic_states.get('DYN_TREND_HEALTHY_ACCELERATING', default_series)
        is_healthy_foundation = is_chip_concentrating | is_trend_healthy_accel
        stable_bullish = is_ma_arrangement_bullish & is_healthy_foundation
        states['MA_STATE_STABLE_BULLISH'] = stable_bullish
        # 步骤1: 定义纯粹的、基于形态的“均线空头排列”
        is_ma_arrangement_bearish = (df[short_ma] < df[mid_ma]) & (df[mid_ma] < df[long_ma])
        # 步骤2: 定义趋势的“风险确认”过滤器 (长期均线斜率必须为负)
        is_long_trend_down = df[long_ma_slope_col] < 0
        # 步骤3: 最终的“稳定空头排列” = 形态排列正确 AND 长期趋势确认向下
        states['MA_STATE_STABLE_BEARISH'] = is_ma_arrangement_bearish & is_long_trend_down
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
        return states

    def diagnose_box_states(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V285.0 职责收缩版】
        - 核心重构: 彻底废除了本模块内关于“健康箱体”的定义。回测证明，基于价格的箱体
                    胜率仅为5.5%，是无效的噪声信号。本模块的职责被严格限定为只输出
                    纯粹的、客观的箱体边界和突破/跌破事件。
        """
        # print("        -> [工兵部队 V285.0 职责收缩版] 启动，正在输出纯粹的箱体边界...")
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
        
        # 彻底移除基于价格箱体的“健康吸筹”判断逻辑。
        #           该信号 (STRUCTURE_BOX_ACCUMULATION_A) 胜率仅5.5%，是高质量策略的“毒药”。
        #           其定义权将移交给基于成本分析的 diagnose_platform_states 模块。
        # healthy_consolidation, BOX_STATE_HEALTHY_CONSOLIDATION, STRUCTURE_BOX_ACCUMULATION_A 已被删除
        
        for key in states:
            if key not in states or states[key] is None:
                states[key] = pd.Series(False, index=df.index)
            else:
                states[key] = states[key].fillna(False)
        return states

    def diagnose_platform_states(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, pd.Series]]:
        """
        【V130.0 权力扩张版】
        - 核心升级: 本模块正式接管“健康吸筹结构”的定义权。
        - 新增职责: 在输出高质量的“稳固平台”信号的同时，也输出 `STRUCTURE_BOX_ACCUMULATION_A`
                    信号。这使得所有依赖旧箱体信号的上游模块，能无缝地、自动地切换到
                    这个胜率更高（14.1% vs 5.5%）的、基于成本的结构判断上来。
        """
        # print("        -> [诊断模块 V130.0 权力扩张版] 正在执行筹码平台状态诊断...")
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
            states['STRUCTURE_PLATFORM_BROKEN'] = default_series
            states['STRUCTURE_BOX_ACCUMULATION_A'] = default_series # 确保有返回值
            return df, states

        is_cost_stable = (df[peak_cost_col].rolling(5).std() / df[peak_cost_col].rolling(5).mean()) < 0.02
        is_above_long_ma = df[close_col] > df[long_ma_col]
        is_shrinking_volume = self.strategy.atomic_states.get('VOL_STATE_SHRINKING', default_series)
        is_chip_concentrating = self.strategy.atomic_states.get('CHIP_DYN_CONCENTRATING', default_series)
        
        stable_formed_series = is_cost_stable & is_above_long_ma & is_shrinking_volume & is_chip_concentrating
        states['PLATFORM_STATE_STABLE_FORMED'] = stable_formed_series
        
        # 权力交接的核心。让基于成本的“稳固平台”成为“健康吸筹结构”的唯一定义。
        #           这使得所有上游模块自动升级，用14.1%胜率的信号替换掉5.5%的噪声。
        states['STRUCTURE_BOX_ACCUMULATION_A'] = stable_formed_series
        
        df['PLATFORM_PRICE_STABLE'] = df[peak_cost_col].where(stable_formed_series)
        
        was_on_platform = stable_formed_series.shift(1).fillna(False)
        stable_platform_price_series = df['PLATFORM_PRICE_STABLE'].ffill()
        is_breaking_down = df[close_col] < stable_platform_price_series.shift(1)
        
        platform_failure_series = was_on_platform & is_breaking_down
        states['STRUCTURE_PLATFORM_BROKEN'] = platform_failure_series

        return df, states

    def synthesize_composite_structures(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V2.1 角色明确版】复合结构合成模块
        - 核心升级: 本模块现在消费的是由 `diagnose_platform_states` 提供的、基于成本的
                    `STRUCTURE_BOX_ACCUMULATION_A` 信号。
        - 角色明确 (本次修改): 明确 `STRUCTURE_BREAKOUT_EVE_S` (突破前夜) 是一个高质量的、
                        但【非方向性】的“战备状态”信号。它的核心价值在于作为高级战术剧本
                        的输入，而不应被直接用于计分。
        - 收益: 使得“突破前夜” (`STRUCTURE_BREAKOUT_EVE_S`) 信号的质量得到了根本性的提升，
                其基础从5.5%胜率的“价格箱体”升级为14.1%胜率的“成本平台”。
        """
        # print("          -> [结构情报司令部 V2.1 角色明确版] 启动，正在进行高维度复合情报合成...")
        composite_states = {}
        default_series = pd.Series(False, index=df.index)
        atomic = self.strategy.atomic_states
        is_platform_stable = atomic.get('PLATFORM_STATE_STABLE_FORMED', default_series)
        is_above_mid_ma = atomic.get('MA_STATE_PRICE_ABOVE_MID_MA', default_series)
        composite_states['STRUCTURE_PLATFORM_WITH_TREND_SUPPORT'] = is_platform_stable & is_above_mid_ma
        # 增加注释，明确信号的“战备”性质，它本身不预测方向。
        # 复合情报2: “突破前夜” (S级战备状态) - 质量已自动升级
        # 新定义：一个健康的“成本平台” + 波动率被压缩到极致 = 高质量的突破前夜 (高势能，方向待确认)
        is_healthy_accumulation = atomic.get('STRUCTURE_BOX_ACCUMULATION_A', default_series)
        is_extreme_squeeze = atomic.get('VOL_STATE_EXTREME_SQUEEZE', default_series)
        composite_states['STRUCTURE_BREAKOUT_EVE_S'] = is_healthy_accumulation & is_extreme_squeeze
        composite_states['STRUCTURE_BOX_ABOVE_TRENDLINE'] = atomic.get('STRUCTURE_BOX_ACCUMULATION_A', default_series)
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
        【V2.0 动态确认版】斐波那契反攻诊断模块
        - 核心升级: 彻底废除了“静态触碰”的无效逻辑。新范式要求必须有“动态力量”的确认。
        - 新逻辑: 斐波那契支撑 = (昨日触及斐波那契位) + (今日出现显性反转阳线) + (处于上升趋势中)。
        - 收益: 将一个胜率仅6%的噪声信号，提纯为一个具备实战价值的高质量“反攻”信号。
        """
        # print("        -> [斐波那契反攻诊断模块 V2.0 动态确认版] 启动...")
        states = {}
        default_series = pd.Series(False, index=df.index)

        p = get_params_block(self.strategy, 'fibonacci_support_params')
        if not get_param_value(p.get('enabled'), False):
            return {}

        proximity_ratio = get_param_value(p.get('proximity_ratio'), 0.01)
        
        fib_levels = {
            '0.618': 'FIB_0_618_D',
            '0.500': 'FIB_0_500_D',
            '0.382': 'FIB_0_382_D'
        }

        if any(col not in df.columns for col in fib_levels.values()):
            print("          -> [警告] 缺少斐波那契水平数据列，模块跳过。")
            return {}

        # 引入“确认”和“上下文”两大过滤器，根治信号泛滥且无效的问题。
        # 过滤器1: “力量确认” - 今日必须出现显性反转阳线，代表主力资金已入场确认支撑。
        is_reversal_confirmed_today = self.strategy.trigger_events.get('TRIGGER_DOMINANT_REVERSAL', default_series)
        
        # 过滤器2: “战略上下文” - 必须处于一个基本的上升趋势中，避免在下跌中继中接飞刀。
        is_in_uptrend_context = self.strategy.atomic_states.get('MA_STATE_PRICE_ABOVE_LONG_MA', default_series)

        # 核心逻辑：识别“昨日下探斐波那契位，今日确认反攻”的行为
        def check_confirmed_rebound(fib_level_col):
            fib_level = df[fib_level_col]
            # 条件1: 昨日最低价触及或跌破斐波那契位 (试探支撑)
            was_pierced_yesterday = (df['low_D'].shift(1) <= fib_level.shift(1) * (1 + proximity_ratio)).fillna(False)
            
            # 最终裁定: (昨日试探) AND (今日主力确认反攻) AND (战略趋势向上)
            return was_pierced_yesterday & is_reversal_confirmed_today & is_in_uptrend_context

        # 分别为不同级别的支撑生成信号
        support_618 = check_confirmed_rebound(fib_levels['0.618'])
        support_500 = check_confirmed_rebound(fib_levels['0.500'])
        support_382 = check_confirmed_rebound(fib_levels['0.382'])
        
        states['OPP_FIB_SUPPORT_GOLDEN_POCKET_S'] = support_618
        states['OPP_FIB_SUPPORT_STANDARD_A'] = support_500 | support_382
        
        if states['OPP_FIB_SUPPORT_GOLDEN_POCKET_S'].any():
            print(f"          -> [情报] 侦测到 {states['OPP_FIB_SUPPORT_GOLDEN_POCKET_S'].sum()} 次 S级“黄金口袋”确认反攻。")
        if states['OPP_FIB_SUPPORT_STANDARD_A'].any():
            print(f"          -> [情报] 侦测到 {states['OPP_FIB_SUPPORT_STANDARD_A'].sum()} 次 A级“标准斐波那契”确认反攻。")
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
        
        # --- 5. 筹码驱动动能诊断 (Chip-driven Momentum Diagnosis) ---
        # 筹码吸筹动能：筹码集中度在增加 (SLOPE_5_concentration_90pct_D < 0) 且 成本峰在抬升 (SLOPE_5_peak_cost_D > 0)
        is_chip_concentrating = df['SLOPE_5_concentration_90pct_D'] < 0
        is_cost_rising = df['SLOPE_5_peak_cost_D'] > 0
        states['MECHANICS_CHIP_ACCUMULATION_MOMENTUM'] = is_chip_concentrating & is_cost_rising
        # if states['MECHANICS_CHIP_ACCUMULATION_MOMENTUM'].any():
        #     print(f"          -> [力学情报] 侦测到 {states['MECHANICS_CHIP_ACCUMULATION_MOMENTUM'].sum()} 次“筹码吸筹动能”信号！")

        # 筹码派发动能：筹码集中度在发散 (SLOPE_5_concentration_90pct_D > 0) 且 成本峰在下降 (SLOPE_5_peak_cost_D < 0)
        is_chip_diverging = df['SLOPE_5_concentration_90pct_D'] > 0
        is_cost_falling = df['SLOPE_5_peak_cost_D'] < 0
        states['MECHANICS_CHIP_DISTRIBUTION_MOMENTUM'] = is_chip_diverging & is_cost_falling
        # if states['MECHANICS_CHIP_DISTRIBUTION_MOMENTUM'].any():
        #     print(f"          -> [力学情报] 侦测到 {states['MECHANICS_CHIP_DISTRIBUTION_MOMENTUM'].sum()} 次“筹码派发动能”信号！")

        # print("        -> [结构力学诊断引擎 V401.1] 分析完毕。")
        return states





