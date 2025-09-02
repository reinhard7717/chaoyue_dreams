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
        # S级风险信号: “死亡交叉确认”
        # 定义: 中期均线(21)刚刚下穿长期均线(55)，这是一个非常经典和重要的趋势转空信号。
        was_mid_above_long = df[mid_ma].shift(1) >= df[long_ma].shift(1)
        is_mid_below_long = df[mid_ma] < df[long_ma]
        states['RISK_MA_DEATH_CROSS_CONFIRMED_S'] = was_mid_above_long & is_mid_below_long

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
        # S级机会信号: “多时间维度趋势共振”
        # 定义: 日线结构形成多头排列的“战术”信号，必须发生在周线趋势已经确认向上的“战略”背景下。
        # 这是过滤掉熊市反弹陷阱，捕捉主升浪中继的强大过滤器。
        is_daily_bullish_structure = atomic.get('MA_STATE_STABLE_BULLISH', default_series)
        weekly_ma_slope_col = 'SLOPE_5_EMA_21_W'
        if weekly_ma_slope_col in df.columns:
            is_weekly_trend_confirmed_up = df[weekly_ma_slope_col] > 0
            composite_states['STRUCTURE_MTF_TREND_ALIGNMENT_S'] = is_daily_bullish_structure & is_weekly_trend_confirmed_up
        else:
            print(f"          -> [警告] 缺少周线斜率列 '{weekly_ma_slope_col}'，无法生成S级多维共振信号。")
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
        short_accel_col = 'ACCEL_13_EMA_13_D'

        required_cols = [long_slope_col, long_accel_col, short_slope_col, short_accel_col]
        if not all(c in df.columns for c in required_cols):
            missing_cols = [c for c in required_cols if c not in df.columns]
            print(f"          -> [错误] 动态惯性诊断缺少必要的斜率/加速度列: {missing_cols}，跳过。")
            return {}

        long_slope = df[long_slope_col]
        long_accel = df[long_accel_col]
        short_slope = df[short_slope_col]
        short_accel = df[short_accel_col]

        # --- 2. 定义基础布尔条件 ---
        is_long_slope_positive = long_slope > 0
        is_long_slope_negative = long_slope < 0
        is_long_accel_positive = long_accel > 0
        is_long_accel_negative = long_accel < 0
        
        is_short_slope_positive = short_slope > 0
        is_short_slope_negative = short_slope < 0
        is_short_accel_positive = short_accel > 0
        
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
        is_price_rising = df['pct_change_D'] > 0
        is_short_slope_falling = short_slope < short_slope.shift(1)
        # 定义：价格在上涨，但短期上涨动能（斜率）在减弱，这是顶背离的微观表现
        dynamics_states['DYN_MOMENTUM_PRICE_SLOPE_DIVERGENCE'] = is_price_rising & is_short_slope_falling

        # 基于加速度的原子信号
        # 【中性状态】上涨趋势稳定期 (主升浪中段): 长期趋势向上，但加速度为负，表明从爆发式增长进入稳定增长
        dynamics_states['DYN_ACCEL_STATE_UPTREND_STABILIZING'] = is_long_slope_positive & is_long_accel_negative
        # 【中性状态】下跌趋势减速期 (潜在底部区): 长期趋势向下，但加速度为正，表明下跌动能正在衰竭
        dynamics_states['DYN_ACCEL_STATE_DOWNTREND_WEAKENING'] = is_long_slope_negative & is_long_accel_positive
        # 【中性状态】短期动能爆发: 短期趋势的加速度为正，表明短期内有资金在快速入场
        dynamics_states['DYN_ACCEL_STATE_SHORT_TERM_BURST'] = is_short_accel_positive

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

    def diagnose_mtf_trend_synergy(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V300.0 战略协同引擎】多时间维度趋势协同诊断
        - 核心职责: 交叉验证战略层面(周线)和战术层面(日线)的趋势动态(斜率与加速度)，
                    生成最高级别的S++级机会与风险信号。
        - 核心逻辑:
          - S++机会(“战略点火”): 周线趋势开始加速的同时，日线趋势也在加速，形成最强共振。
          - S++风险(“战略衰竭”): 周线趋势已在减速，但日线仍在加速冲顶，是典型的顶部背离陷阱。
        """
        print("        -> [战略协同引擎 V300.0] 启动，正在进行多维动态交叉验证...") # [新增代码行]
        states = {}
        default_series = pd.Series(False, index=df.index)

        # --- 1. 定义并检查所需的多维动态数据列 ---
        # 战略层面 (周线)
        weekly_slope_col = 'SLOPE_5_EMA_21_W'
        weekly_accel_col = 'ACCEL_5_EMA_21_W'
        # 战术层面 (日线)
        daily_slope_col = 'SLOPE_13_EMA_13_D'
        daily_accel_col = 'ACCEL_13_EMA_13_D'

        required_cols = [weekly_slope_col, weekly_accel_col, daily_slope_col, daily_accel_col]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"          -> [严重警告] 战略协同引擎缺少关键数据列: {missing_cols}，模块已跳过！")
            return states

        # --- 2. 获取数据并定义基础布尔状态 ---
        weekly_slope = df[weekly_slope_col]
        weekly_accel = df[weekly_accel_col]
        daily_slope = df[daily_slope_col]
        daily_accel = df[daily_accel_col]

        is_weekly_slope_positive = weekly_slope > 0
        is_weekly_slope_negative = weekly_slope < 0
        is_weekly_accel_positive = weekly_accel > 0
        is_weekly_accel_negative = weekly_accel < 0

        is_daily_slope_positive = daily_slope > 0
        is_daily_accel_positive = daily_accel > 0

        # --- 3. 组合生成S++级战略协同信号 ---

        # S++级机会: “战略点火，战术强攻” (Strategic Ignition & Tactical Assault)
        # 解读: 长期趋势开始进入主升段 (周线加速)，短期趋势也正猛烈上攻 (日线加速)。
        # 这是最值得重仓参与的黄金时刻，代表长短周期力量完美共振。
        states['OPP_STRUCTURE_MTF_IGNITION_S_PLUS'] = is_weekly_accel_positive & is_daily_accel_positive

        # A+级机会: “战略顺风，战术助推” (Strategic Tailwind & Tactical Boost)
        # 解读: 长期趋势已确立 (周线斜率>0)，短期趋势开始发力 (日线加速)。
        # 这是主升浪途中的健康加仓或入场信号。
        states['OPP_STRUCTURE_MTF_TAILWIND_A_PLUS'] = is_weekly_slope_positive & is_daily_accel_positive

        # S++级风险: “战略衰竭，战术诱多” (Strategic Exhaustion & Tactical Deception)
        # 解读: 长期趋势的上涨动能已在衰竭 (周线减速)，但短期趋势仍在制造最后的疯狂 (日线加速)。
        # 这是典型的顶部背离，是逃顶的绝佳信号。
        states['RISK_STRUCTURE_MTF_EXHAUSTION_S_PLUS'] = is_weekly_accel_negative & is_daily_accel_positive

        # A+级风险: “战略逆风，战术反弹” (Strategic Headwind & Tactical Rebound)
        # 解读: 长期趋势已是明确的下跌 (周线斜率<0)，任何日线级别的上涨 (日线斜率>0) 都可能是熊市反弹。
        # 这是需要高度警惕的陷阱，应避免参与。
        states['RISK_STRUCTURE_MTF_HEADWIND_A_PLUS'] = is_weekly_slope_negative & is_daily_slope_positive

        print("        -> [战略协同引擎 V300.0] 分析完毕。") # [新增代码行]
        return states

    def diagnose_static_dynamic_fusion(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V400.0 静态-动态融合引擎】
        - 核心职责: 将一个高质量的“静态战备”信号与多个“多维动态”信号进行交叉验证，
                    生成具备因果逻辑的、最高置信度的S++级信号。
        - 核心逻辑:
          - S++机会(“引爆点”): 在“突破前夜”的静态高势能状态下，同时观测到“战略与战术”
                           层面的双重加速，确认总攻开始。
          - S++风险(“陷阱”): 在“突破前夜”的静态高势能状态下，观测到“战略衰竭”与
                         “战术诱多”的致命背离，确认是假突破陷阱。
        """
        print("        -> [静态-动态融合引擎 V400.0] 启动，正在寻找战场引爆点...") # [新增代码行]
        states = {}
        default_series = pd.Series(False, index=df.index)
        atomic = self.strategy.atomic_states

        # --- 1. 定义并检查所需的核心信号 ---
        # 核心静态战备信号 (高势能状态)
        static_setup_signal = 'STRUCTURE_BREAKOUT_EVE_S'
        # 核心多维动态机会信号 (最强动能确认)
        dynamic_opportunity_signal = 'OPP_STRUCTURE_MTF_IGNITION_S_PLUS'
        # 核心多维动态风险信号 (最强背离确认)
        dynamic_risk_signal = 'RISK_STRUCTURE_MTF_EXHAUSTION_S_PLUS'

        required_signals = [static_setup_signal, dynamic_opportunity_signal, dynamic_risk_signal]
        missing_signals = [s for s in required_signals if s not in atomic]
        if missing_signals:
            print(f"          -> [严重警告] 静态-动态融合引擎缺少核心原子信号: {missing_signals}，模块已跳过！")
            return states

        # --- 2. 获取核心信号序列 ---
        is_static_setup_ready = atomic.get(static_setup_signal, default_series)
        is_dynamic_opportunity_confirmed = atomic.get(dynamic_opportunity_signal, default_series)
        is_dynamic_risk_confirmed = atomic.get(dynamic_risk_signal, default_series)

        # --- 3. 组合生成S++级融合信号 ---

        # S++级机会: “静态-动态融合·引爆点” (Static-Dynamic Fusion: Ignition Point)
        # 解读: 战场(极致压缩的稳定平台)已就绪，且总司令部(周线)和前线指挥官(日线)同时下达了“加速总攻”的命令。
        # 这是整个策略体系中确定性最高的进攻信号之一。
        states['OPP_STATIC_DYN_FUSION_IGNITION_S_PLUS'] = is_static_setup_ready & is_dynamic_opportunity_confirmed

        # S++级风险: “静态-动态融合·陷阱” (Static-Dynamic Fusion: Bull Trap)
        # 解读: 战场看似就绪，但前线部队(日线)的“加速冲锋”并未得到总司令部(周线)的支援，
        # 反而总部正在撤退(周线减速)。这是典型的诱多出货陷阱。
        states['RISK_STATIC_DYN_FUSION_TRAP_S_PLUS'] = is_static_setup_ready & is_dynamic_risk_confirmed

        print("        -> [静态-动态融合引擎 V400.0] 分析完毕。") # [新增代码行]
        return states

    def diagnose_structural_risks_and_regimes(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V1.0 新增】结构性风险与市场状态诊断模块
        - 核心职责: 引入BIAS和Hurst指数，从结构稳定性和市场根本属性上进行诊断。
        - 新增信号:
          1. 结构性超涨风险: 基于动态阈值的BIAS指标，识别价格偏离均线过远的风险。
          2. 市场状态识别: 基于Hurst指数，判断当前市场是趋势市还是均值回归市。
          3. 多维共振超涨风险: 结合日线与周线BIAS，识别最高级别的顶部风险。
        """
        print("        -> [结构风险与状态诊断模块 V1.0] 启动...") # [新增代码行]
        states = {}
        p = get_params_block(self.strategy, 'structural_risk_params')
        if not get_param_value(p.get('enabled'), True): return states
        # --- 1. 军备检查 (Prerequisite Check) ---
        required_cols = ['BIAS_21_D', 'BIAS_55_D', 'hurst_120d_D', 'BIAS_20_W']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"          -> [警告] 结构风险诊断缺少关键数据列: {missing_cols}，部分信号可能无法生成。")
        # --- 2. 结构性超涨风险 (Structural Overextension Risk) ---
        # 使用动态分位数作为阈值，比固定值更具适应性
        if 'BIAS_21_D' in df.columns:
            short_term_threshold = df['BIAS_21_D'].rolling(250).quantile(0.95)
            states['RISK_STRUCTURE_OVEREXTENDED_SHORT_TERM_A'] = df['BIAS_21_D'] > short_term_threshold
        if 'BIAS_55_D' in df.columns:
            long_term_threshold = df['BIAS_55_D'].rolling(250).quantile(0.95)
            states['RISK_STRUCTURE_OVEREXTENDED_LONG_TERM_S'] = df['BIAS_55_D'] > long_term_threshold
        # --- 3. 市场状态识别 (Market Regime Identification) ---
        if 'hurst_120d_D' in df.columns:
            hurst_trending_threshold = get_param_value(p.get('hurst_trending_threshold'), 0.55)
            hurst_reverting_threshold = get_param_value(p.get('hurst_reverting_threshold'), 0.45)
            states['STRUCTURE_REGIME_TRENDING'] = df['hurst_120d_D'] > hurst_trending_threshold
            states['STRUCTURE_REGIME_MEAN_REVERTING'] = df['hurst_120d_D'] < hurst_reverting_threshold
        # --- 4. 多时间维度共振超涨风险 (MTF Overextended Resonance Risk) ---
        if all(c in df.columns for c in ['BIAS_21_D', 'BIAS_20_W']):
            is_daily_overextended = states.get('RISK_STRUCTURE_OVEREXTENDED_SHORT_TERM_A', pd.Series(False, index=df.index))
            # 周线BIAS也使用动态阈值
            weekly_threshold = df['BIAS_20_W'].rolling(52).quantile(0.95) # 周线看过去一年(52周)
            is_weekly_overextended = df['BIAS_20_W'] > weekly_threshold
            states['RISK_STRUCTURE_MTF_OVEREXTENDED_RESONANCE_S'] = is_daily_overextended & is_weekly_overextended
        print("        -> [结构风险与状态诊断模块 V1.0] 诊断完毕。") # [新增代码行]
        return states



