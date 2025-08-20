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
        【V283.1 健壮修复版】
        - 核心修复 1: 彻底移除了对不存在的 'MA_ZSCORE_D' 列的引用，该逻辑是无效的。
        - 核心修复 2: 修正了对均线粘合度(CV)列名的引用，以匹配数据层因后缀叠加而产生的
                      实际列名 (例如 'MA_CONV_CV_SHORT_D_D')，确保动态粘合状态能被正确计算。
        """
        # print("          -> [均线野战部队 V283.1 健壮修复版] 启动，正在执行融合分析...")
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
        # 条件A: 短期或长期均线必须处于“粘合压缩”状态
        is_highly_converged = (
            states.get('MA_STATE_SHORT_CONVERGENCE_SQUEEZE', pd.Series(False, index=df.index)) |
            states.get('MA_STATE_LONG_CONVERGENCE_SQUEEZE', pd.Series(False, index=df.index))
        )
        # 条件B: 出现能量释放阳线作为突破确认
        # 注意：这里需要从 self.strategy.trigger_events 获取，但它在 intelligence_layer 的后期才生成。
        # 因此，我们在这里直接实现其核心逻辑，以避免循环依赖。
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

        # 最终信号：昨日处于收敛状态，今日出现突破阳线
        was_converged_yesterday = is_highly_converged.shift(1).fillna(False)
        # 增加趋势过滤器，确保只在上升趋势中寻找突破机会
        is_in_uptrend_context = states.get('MA_STATE_PRICE_ABOVE_LONG_MA', pd.Series(False, index=df.index))
        # 将趋势过滤器加入最终的逻辑判断
        states['OPP_MA_CONVERGENCE_BREAKOUT_A'] = was_converged_yesterday & is_breakout_candle & is_in_uptrend_context
        
        return states

    def diagnose_box_states(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V283.2 健康基因注入版】
        - 核心升级: 为“健康箱体”的定义注入了“成交量萎缩”和“筹码集中”两大核心健康基因，
                    从根本上过滤掉“阴跌平台”和“死股盘整”等伪信号。
        """
        print("        -> [工兵部队 V283.2 健康基因注入版] 启动，正在执行融合分析并输出箱体边界...") # MODIFIED: 修改版本号和描述
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
        # 使用长期均线作为趋势过滤器，更可靠
        long_ma_period = get_param_value(ma_params.get('long_ma'), 55)
        long_ma_col = f"EMA_{long_ma_period}_D"
        
        # 重新定义“健康盘整”，注入两大健康基因
        healthy_consolidation = pd.Series(False, index=df.index)
        if long_ma_col in df.columns:
            box_midpoint = (box_top + box_bottom) / 2
            is_box_above_ma = box_midpoint > df[long_ma_col]
            
            # 健康基因1: 成交量必须萎缩
            is_shrinking_volume = self.strategy.atomic_states.get('VOL_STATE_SHRINKING', pd.Series(False, index=df.index))
            # 健康基因2: 筹码必须在集中
            is_chip_concentrating = self.strategy.atomic_states.get('CHIP_DYN_CONCENTRATING', pd.Series(False, index=df.index))

            # 最终裁定：形态(is_valid_box & is_in_box) + 趋势(is_box_above_ma) + 内部行为(is_shrinking_volume & is_chip_concentrating)
            healthy_consolidation = is_valid_box & is_in_box & is_box_above_ma & is_shrinking_volume & is_chip_concentrating
        
        states['BOX_STATE_HEALTHY_CONSOLIDATION'] = healthy_consolidation

        is_shrinking_volume = self.strategy.atomic_states.get('VOL_STATE_SHRINKING', pd.Series(False, index=df.index))
        is_chip_concentrating = self.strategy.atomic_states.get('CHIP_DYN_CONCENTRATING', pd.Series(False, index=df.index))
        healthy_accumulation_a = healthy_consolidation # healthy_consolidation 已经包含了所有健康条件
        states['STRUCTURE_BOX_ACCUMULATION_A'] = healthy_accumulation_a
        is_extreme_squeeze = self.strategy.atomic_states.get('VOL_STATE_EXTREME_SQUEEZE', pd.Series(False, index=df.index))
        states['STRUCTURE_BREAKOUT_EVE_S'] = healthy_accumulation_a & is_extreme_squeeze
        
        for key in states:
            if key not in states or states[key] is None:
                states[key] = pd.Series(False, index=df.index)
            else:
                states[key] = states[key].fillna(False)
        return states

    def diagnose_platform_states(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, pd.Series]]:
        """
        【V129.3 健康基因注入版 - 筹码平台诊断模块】
        - 核心重构: 彻底重写了“稳固平台”的定义。不再只依赖被动的成本峰稳定，
                    而是主动融合了“成交量萎缩”和“筹码动态集中”两大健康度指标。
        - 收益: 新的平台定义能够精确识别出“健康的吸筹平台”，从根本上过滤掉
                “滞涨盘整”和“阴跌”等伪平台信号，为下游战法提供高质量的基石。
        """
        # print("        -> [诊断模块 V129.3 健康基因注入版] 正在执行筹码平台状态诊断...")
        states = {}
        default_series = pd.Series(False, index=df.index)

        # --- 步骤1: 检查核心数据是否存在 ---
        peak_cost_col = 'peak_cost_D'
        close_col = 'close_D'
        long_ma_col = 'EMA_55_D' # 平台必须位于长期均线上方才有意义
        
        required_cols = [peak_cost_col, close_col, long_ma_col]
        if not all(col in df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in df.columns]
            print(f"          -> [警告] 缺少诊断平台状态所需的核心列: {missing}。模块将返回空结果。")
            # 即使失败，也要确保返回标准化的输出结构，防止下游模块调用失败
            df['PLATFORM_PRICE_STABLE'] = np.nan
            states['PLATFORM_STATE_STABLE_FORMED'] = default_series
            states['PLATFORM_FAILURE'] = default_series
            return df, states

        # --- 步骤2: 定义并计算“稳固平台形成”状态 ---
        # 条件A: 筹码峰成本在短期内高度稳定 (滚动5日的标准差/均值 < 2%)
        is_cost_stable = (df[peak_cost_col].rolling(5).std() / df[peak_cost_col].rolling(5).mean()) < 0.02
        
        # 条件B: 当前价格位于长期趋势均线之上，确保平台处于上升趋势中
        is_above_long_ma = df[close_col] > df[long_ma_col]
        
        # 条件C : 成交量健康 - 必须处于缩量状态
        is_shrinking_volume = self.strategy.atomic_states.get('VOL_STATE_SHRINKING', default_series)
        
        # 条件D : 筹码健康 - 必须处于动态集中过程
        is_chip_concentrating = self.strategy.atomic_states.get('CHIP_DYN_CONCENTRATING', default_series)
        
        # 组合成最终的“稳固平台形成”状态
        stable_formed_series = is_cost_stable & is_above_long_ma & is_shrinking_volume & is_chip_concentrating
        states['PLATFORM_STATE_STABLE_FORMED'] = stable_formed_series
        
        # --- 步骤3: 将有效的平台价格记录下来，供后续模块使用 ---
        # 只有在平台形成当天，才记录下当天的平台价格，否则为NaN
        df['PLATFORM_PRICE_STABLE'] = df[peak_cost_col].where(stable_formed_series)
        
        # --- 步骤4: 定义并计算“平台破位”风险 ---
        # 条件A: 昨日处于稳固平台之上
        was_on_platform = stable_formed_series.shift(1).fillna(False)
        
        # 条件B: 今日收盘价跌破了昨日的平台价格
        # 使用 ffill() 填充平台价格，以处理平台形成后、破位前的那些天
        stable_platform_price_series = df['PLATFORM_PRICE_STABLE'].ffill()
        is_breaking_down = df[close_col] < stable_platform_price_series.shift(1)
        
        # 组合成最终的“平台破位”风险信号
        platform_failure_series = was_on_platform & is_breaking_down
        states['STRUCTURE_PLATFORM_BROKEN'] = platform_failure_series

        # --- 步骤5: 打印诊断日志 ---
        # print(f"          -> '稳固平台形成' 状态诊断完成，共激活 {stable_formed_series.sum()} 天。")
        # print(f"          -> '平台破位' 风险诊断完成，共激活 {platform_failure_series.sum()} 天。")

        return df, states

    def diagnose_market_structure_command(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V272.0 市场结构战区司令部】
        - 核心重构: 这不是一次合并，而是一次战略整合。
        - 新架构:
          1. 本司令部统一指挥下属的三个专业化兵种：均线野战部队、价格工兵部队、筹码特种侦察部队。
          2. 它首先收集所有基础的“原子结构情报”。
          3. 然后，它进行情报融合，生成更高维度的、包含协同作战思想的“复合结构情报”。
        - 收益: 极大地提升了代码的组织性和可读性，并能产出远比单个模块更有价值的协同信号。
        """
        # print("        -> [市场结构战区司令部 V272.0] 启动，正在整合全战场结构情报...")
        
        # --- 1. 依次调动下属的专业化兵种，收集原子情报 ---
        # print("          -> 正在调动：均线野战部队、价格工兵部队、筹码特种侦察部队...")
        ma_states = self.diagnose_ma_states(df)
        box_states = self.diagnose_box_states(df)
        df, platform_states = self.diagnose_platform_states(df) # 平台诊断会修改df，需要接收
        
        # 将所有原子情报汇总
        atomic_structure_states = {**ma_states, **box_states, **platform_states}
        
        # --- 2. 进行情报融合与战术研判，生成复合情报 ---
        # print("          -> 正在进行情报融合，生成高维度复合情报...")
        composite_states = {}
        default_series = pd.Series(False, index=df.index)

        # 复合情报1: “阵地优势” - 一个稳固的平台，必须得到动态趋势线的确认
        is_platform_stable = atomic_structure_states.get('PLATFORM_STATE_STABLE_FORMED', default_series)
        is_above_mid_ma = atomic_structure_states.get('MA_STATE_PRICE_ABOVE_MID_MA', default_series)
        composite_states['STRUCTURE_PLATFORM_WITH_TREND_SUPPORT'] = is_platform_stable & is_above_mid_ma

        # 复合情报2: “健康盘整” - 一个箱体整理，必须发生在关键趋势线上方
        is_in_box = atomic_structure_states.get('BOX_STATE_CONSOLIDATING', default_series)
        is_above_long_ma = atomic_structure_states.get('MA_STATE_PRICE_ABOVE_LONG_MA', default_series)
        composite_states['STRUCTURE_BOX_ABOVE_TRENDLINE'] = is_in_box & is_above_long_ma

        # 复合情报3: “突破前夜” (S级战术信号) - 极致的共振信号
        # 定义：价格被压缩在一个健康的箱体内，而这个箱体本身就建立在一个稳固的筹码平台上，
        #       同时，整个结构都位于主升趋势线之上。这是大战一触即发的终极信号！
        is_healthy_box = composite_states.get('STRUCTURE_BOX_ABOVE_TRENDLINE', default_series)
        is_on_stable_platform = atomic_structure_states.get('PLATFORM_STATE_STABLE_FORMED', default_series)
        composite_states['STRUCTURE_BREAKOUT_EVE_S'] = is_healthy_box & is_on_stable_platform

        # print("        -> [市场结构战区司令部 V272.0] 情报整合完毕。")
        
        # 返回所有原子情报和复合情报的集合，以及可能被修改的df
        return df, {**atomic_structure_states, **composite_states}


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





