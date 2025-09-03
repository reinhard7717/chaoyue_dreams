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

        # 预先计算前一日的数据，避免重复调用 .shift(1)，提高效率
        prev_close = df['close_D'].shift(1)
        prev_box_top = box_top.shift(1)
        prev_box_bottom = box_bottom.shift(1)
        # 使用预计算的变量判断突破
        was_below_top = prev_close <= prev_box_top
        is_above_top = df['close_D'] > box_top
        states['BOX_EVENT_BREAKOUT'] = is_valid_box & is_above_top & was_below_top
        
        # 使用预计算的变量判断跌破
        was_above_bottom = prev_close >= prev_box_bottom
        is_below_bottom = df['close_D'] < box_bottom
        states['BOX_EVENT_BREAKDOWN'] = is_valid_box & is_below_bottom & was_above_bottom
        
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

        # 预先计算滚动均值和标准差，避免重复计算，并明确计算变异系数(CV)
        rolling_cost = df[peak_cost_col].rolling(5)
        rolling_mean_cost = rolling_cost.mean()
        rolling_std_cost = rolling_cost.std()
        
        # 使用 np.errstate 避免在均值为0时出现除零警告
        with np.errstate(divide='ignore', invalid='ignore'):
            coeff_of_variation = rolling_std_cost / rolling_mean_cost
        
        # 将计算中可能产生的 NaN 或 inf 结果安全地处理为 False
        is_cost_stable = (coeff_of_variation < 0.02).fillna(False)
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
        【V500.0 数值化多维引擎】
        - 核心重构: 从布尔信号引擎全面升级为数值化(0-1)评分引擎。
        - 核心扩展: 趋势分析维度从(13, 55)扩展至(5, 13, 21, 55)，构建全景动态视图。
        - 新增信号 (数值型):
          - SCORE_TREND_RESONANCE: 趋势共振分。所有周期同向看涨时得分高。
          - SCORE_TREND_ACCEL_RESONANCE: 加速度共振分。所有周期同向加速时得分高。
          - SCORE_TREND_DIVERGENCE_RISK: 趋势背离风险分。长涨短跌时得分高。
          - SCORE_TREND_INFLECTION_OPP: 趋势拐点机会分。长跌短涨时得分高。
        - 收益: 提供了对趋势“强度”和“质量”的量化评估，为下游评分系统提供更精细的输入。
        """
        print("        -> [诊断模块 V500.0 数值化多维引擎] 启动...") # 更新打印信息
        states = {}
        p = get_params_block(self.strategy, 'multi_dim_trend_params') # 使用新的参数块
        if not get_param_value(p.get('enabled'), True): return {} # 增加模块开关
        # --- 1. 定义多维分析周期并构建所需列名 ---
        ma_periods = get_param_value(p.get('ma_periods'), [5, 13, 21, 55]) # 从配置读取周期
        # 动态构建列名，统一使用5周期的斜率和加速度进行比较
        slope_cols = {p: f'SLOPE_5_EMA_{p}_D' for p in ma_periods}
        accel_cols = {p: f'ACCEL_5_EMA_{p}_D' for p in ma_periods}
        required_cols = list(slope_cols.values()) + list(accel_cols.values())
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"          -> [错误] 数值化多维引擎缺少必要列: {missing_cols}，跳过。")
            return {}
        # --- 2. 核心步骤: 将所有斜率和加速度进行数值化/归一化 ---
        normalized_scores = {}
        norm_window = get_param_value(p.get('norm_window'), 120) # 归一化窗口可配置
        min_periods = max(1, norm_window // 5)
        for period in ma_periods:
            # 归一化斜率 (值越大，上涨动能越强)
            slope_series = df[slope_cols[period]]
            normalized_scores[f'slope_{period}'] = slope_series.rolling(
                window=norm_window, min_periods=min_periods
            ).rank(pct=True).fillna(0.5)
            # 归一化加速度 (值越大，加速越猛)
            accel_series = df[accel_cols[period]]
            normalized_scores[f'accel_{period}'] = accel_series.rolling(
                window=norm_window, min_periods=min_periods
            ).rank(pct=True).fillna(0.5)
        # --- 3. 交叉验证与合成高级数值信号 ---
        # 信号1: 趋势共振分 (所有周期同向看涨的强度)
        slope_series_list = [normalized_scores[f'slope_{p}'] for p in ma_periods]
        states['SCORE_TREND_RESONANCE'] = pd.concat(slope_series_list, axis=1).mean(axis=1)
        # 信号2: 加速度共振分 (所有周期同向加速的强度)
        accel_series_list = [normalized_scores[f'accel_{p}'] for p in ma_periods]
        states['SCORE_TREND_ACCEL_RESONANCE'] = pd.concat(accel_series_list, axis=1).mean(axis=1)
        # 信号3: 趋势背离风险分 (长周期向上，但短周期开始掉头)
        long_term_strength = normalized_scores[f'slope_{ma_periods[-1]}'] # 最长周期
        short_term_weakness = 1 - normalized_scores[f'slope_{ma_periods[0]}'] # 最短周期
        states['SCORE_TREND_DIVERGENCE_RISK'] = (long_term_strength * short_term_weakness).fillna(0.5)
        # 信号4: 趋势拐点机会分 (长周期见底，短周期开始抬头)
        long_term_bottoming = 1 - normalized_scores[f'slope_{ma_periods[-1]}']
        short_term_reversal = normalized_scores[f'slope_{ma_periods[0]}']
        states['SCORE_TREND_INFLECTION_OPP'] = (long_term_bottoming * short_term_reversal).fillna(0.5)
        print("        -> [诊断模块 V500.0 数值化多维引擎] 分析完毕。") # 更新打印信息
        return states

    # 融合静态战备信号与多维动态数值信号
    def diagnose_fusion_breakout_scores(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V1.0 新增】静态-动态融合评分引擎
        - 核心职责: 将一个高质量的“静态战备”信号与多个“多维动态”数值信号进行交叉相乘，
                    生成具备因果逻辑的、最高置信度的S++级数值化信号。
        - 核心逻辑:
          - S++机会(“引爆点”): 最终得分 = (突破前夜状态) * (加速度共振分)。
                           只有在战备就绪(值为1)时，动态分才能通过，否则为0。
          - S++风险(“陷阱”): 最终得分 = (突破前夜状态) * (趋势背离风险分)。
        """
        print("        -> [静态-动态融合评分引擎 V1.0] 启动...")
        states = {}
        default_series = pd.Series(0.0, index=df.index) # 默认值改为0.0以适配数值计算
        atomic = self.strategy.atomic_states
        # --- 1. 定义并检查所需的核心信号 ---
        # 核心静态战备信号 (高势能状态)
        static_setup_signal = 'STRUCTURE_BREAKOUT_EVE_S'
        # 核心多维动态机会信号 (最强动能确认)
        dynamic_opportunity_score = 'SCORE_TREND_ACCEL_RESONANCE'
        # 核心多维动态风险信号 (最强背离确认)
        dynamic_risk_score = 'SCORE_TREND_DIVERGENCE_RISK'
        required_signals = [static_setup_signal, dynamic_opportunity_score, dynamic_risk_score]
        missing_signals = [s for s in required_signals if s not in atomic]
        if missing_signals:
            print(f"          -> [严重警告] 静态-动态融合引擎缺少核心原子信号: {missing_signals}，模块已跳过！")
            return {}
        # --- 2. 获取核心信号序列 ---
        is_static_setup_ready = atomic.get(static_setup_signal, default_series).astype(float)
        dynamic_ignition_score = atomic.get(dynamic_opportunity_score, default_series)
        dynamic_trap_score = atomic.get(dynamic_risk_score, default_series)
        # --- 3. 组合生成S++级融合数值化信号 ---
        # S++级机会: “静态-动态融合·引爆点”
        # 解读: 在“突破前夜”的静态高势能状态下，观测到“全周期加速共振”的动态点火信号。
        states['SCORE_FUSION_IGNITION_POINT_S_PLUS'] = is_static_setup_ready * dynamic_ignition_score
        # S++级风险: “静态-动态融合·陷阱”
        # 解读: 在“突破前夜”的静态高势能状态下，观测到“长短周期趋势背离”的风险信号。
        states['SCORE_FUSION_BULL_TRAP_S_PLUS'] = is_static_setup_ready * dynamic_trap_score
        print("        -> [静态-动态融合评分引擎 V1.0] 分析完毕。")
        return states

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
        # print("        -> [战略协同引擎 V300.0] 启动，正在进行多维动态交叉验证...")
        states = {}
        # default_series = pd.Series(False, index=df.index)

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

        print("        -> [战略协同引擎 V300.0] 分析完毕。")
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
        # print("        -> [静态-动态融合引擎 V400.0] 启动，正在寻找战场引爆点...")
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

        print("        -> [静态-动态融合引擎 V400.0] 分析完毕。")
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
        # print("        -> [结构风险与状态诊断模块 V1.0] 启动...")
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
        print("        -> [结构风险与状态诊断模块 V1.0] 诊断完毕。")
        return states

    def diagnose_advanced_structural_patterns(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V1.0 新增】先进结构模式诊断模块
        - 核心职责: 引入波动率、趋势质量、价格行为和多维静态对齐等新的维度，生成更丰富的结构性原子信号。
        - 新增信号:
          1. ATR 压缩: 利用ATR的绝对水平识别市场能量积蓄状态。
          2. 高品质趋势: 利用价格变异系数(CV)量化趋势的平稳性和健康度。
          3. 日线孕线形态: 捕捉经典的、代表短期整理的K线模式。
          4. 多维静态对齐: 交叉验证日线和周线的静态趋势结构，提供强大的趋势确认。
        """
        print("        -> [先进结构模式诊断模块 V1.0] 启动...")
        states = {}
        default_series = pd.Series(False, index=df.index)
        p = get_params_block(self.strategy, 'advanced_structural_params')
        if not get_param_value(p.get('enabled'), True):
            return states
        # --- 1. 军备检查 (Prerequisite Check) ---
        required_cols = [
            'ATR_14_D', 'price_cv_60d_D', 'SLOPE_21_EMA_55_D',
            'high_D', 'low_D', 'close_D', 'EMA_55_D',
            'close_W', 'EMA_21_W'
        ]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"          -> [警告] 先进结构诊断缺少关键数据列: {missing_cols}，部分信号可能无法生成。")
        # --- 2. ATR 压缩诊断 (ATR Compression) ---
        if 'ATR_14_D' in df.columns:
            window = get_param_value(p.get('atr_compression_window'), 120)
            quantile = get_param_value(p.get('atr_compression_quantile'), 0.10)
            # 计算ATR的滚动分位数阈值
            atr_threshold = df['ATR_14_D'].rolling(window=window).quantile(quantile)
            # 当ATR低于这个动态阈值时，认为波动率被极度压缩
            states['STRUCTURE_ATR_COMPRESSION_A'] = df['ATR_14_D'] < atr_threshold
        # --- 3. 高品质趋势诊断 (High-Quality Trend) ---
        if all(c in df.columns for c in ['price_cv_60d_D', 'SLOPE_21_EMA_55_D']):
            cv_window = get_param_value(p.get('quality_trend_cv_window'), 120)
            cv_quantile = get_param_value(p.get('quality_trend_cv_quantile'), 0.20)
            # 价格变异系数(CV)越小，代表价格走势越平稳，趋势质量越高
            cv_threshold = df['price_cv_60d_D'].rolling(window=cv_window).quantile(cv_quantile)
            is_trend_smooth = df['price_cv_60d_D'] < cv_threshold
            # 必须在长期趋势向上的背景下，平稳才有意义
            is_long_trend_up = df['SLOPE_21_EMA_55_D'] > 0
            states['STRUCTURE_HIGH_QUALITY_TREND_A'] = is_trend_smooth & is_long_trend_up
        # --- 4. 日线孕线形态诊断 (Inside Day Pattern) ---
        if all(c in df.columns for c in ['high_D', 'low_D']):
            # 当天的最高价低于昨天，最低价高于昨天
            is_inside_day = (df['high_D'] < df['high_D'].shift(1)) & (df['low_D'] > df['low_D'].shift(1))
            states['STRUCTURE_INSIDE_DAY_CANDLE_N'] = is_inside_day.fillna(False)
        # --- 5. 多时间框架静态对齐诊断 (MTF Static Alignment) ---
        if all(c in df.columns for c in ['close_D', 'EMA_55_D', 'close_W', 'EMA_21_W']):
            # 战术层面（日线）确认多头结构
            is_daily_structurally_bullish = df['close_D'] > df['EMA_55_D']
            # 战略层面（周线）确认多头结构
            is_weekly_structurally_bullish = df['close_W'] > df['EMA_21_W']
            # 两者必须同时满足，形成最强共振
            states['STRUCTURE_MTF_STATIC_ALIGNMENT_S'] = is_daily_structurally_bullish & is_weekly_structurally_bullish
        # --- 6. 填充可能缺失的信号，确保返回结果的完整性 ---
        all_signal_keys = [
            'STRUCTURE_ATR_COMPRESSION_A', 'STRUCTURE_HIGH_QUALITY_TREND_A',
            'STRUCTURE_INSIDE_DAY_CANDLE_N', 'STRUCTURE_MTF_STATIC_ALIGNMENT_S'
        ]
        for key in all_signal_keys:
            if key not in states:
                states[key] = default_series
        print("        -> [先进结构模式诊断模块 V1.0] 诊断完毕。")
        return states



