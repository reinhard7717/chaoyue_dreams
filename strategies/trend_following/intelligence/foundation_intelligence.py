# 文件: strategies/trend_following/intelligence/foundation_intelligence.py
# 基础情报模块 (波动率, 震荡指标)
import pandas as pd
import numpy as np
from typing import Dict
from strategies.trend_following.utils import get_params_block, get_param_value, create_persistent_state

class FoundationIntelligence:
    def __init__(self, strategy_instance):
        """
        初始化基础情报模块。
        :param strategy_instance: 策略主实例的引用，用于访问df和atomic_states。
        """
        self.strategy = strategy_instance

    def run_foundation_analysis_command(self) -> Dict[str, pd.Series]:
        """
        【V2.0 新增】基础情报分析总指挥
        - 核心职责: 统一调度波动率、震荡指标、资金流以及协同效应的诊断，
                      生成一套完整的基础层原子信号。
        """
        # print("        -> [基础情报分析总指挥 V2.0] 启动...")
        states = {}
        df = self.strategy.df_indicators

        # --- 步骤1: 诊断波动率状态 (BBW & ATR) ---
        vol_states = self.diagnose_volatility_states(df)
        states.update(vol_states)

        # --- 步骤2: 诊断震荡指标状态 (RSI, MACD, BIAS) ---
        osc_states = self.diagnose_oscillator_states(df)
        states.update(osc_states)

        # --- 步骤3: 诊断资金流与绝对波幅状态 (CMF, ATR) ---
        capital_range_states = self.diagnose_capital_and_range_states(df)
        states.update(capital_range_states)
        
        # --- 步骤4: 诊断经典技术指标 (MACD交叉, 量比) ---
        classic_states = self.diagnose_classic_indicators(df)
        states.update(classic_states)
        
        # 在生成所有基础原子后，更新 self.strategy.atomic_states
        # 这样后续步骤才能消费到最新的原子状态
        self.strategy.atomic_states.update(states)

        # --- 步骤4: 诊断协同效应状态 (交叉验证) ---
        synergy_states = self.diagnose_synergy_states(df)
        states.update(synergy_states)
        
        # --- 步骤5: 诊断多时间维度协同效应 (交叉验证) ---
        mtf_synergy_states = self.diagnose_multi_timeframe_synergy(df)
        states.update(mtf_synergy_states)
        
        # --- 步骤6: 诊断静态-多动态协同效应 (最高置信度信号) ---
        static_multi_dyn_states = self.diagnose_static_multi_dynamic_synergy(df)
        states.update(static_multi_dyn_states)

        # 再次更新，确保协同信号也被添加
        self.strategy.atomic_states.update(states)
        
        # print("          -> [基础情报分析总指挥] 波动率/震荡/资金流/协同情报已全部生成。")
        return states

    def diagnose_volatility_states(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V283.2 确认打击改造版】
        - 核心加固: 明确检查所有依赖的预计算列是否存在，提升代码健壮性。
        - 核心改造 (本次修改): 废除将“压缩”本身作为看涨信号的旧逻辑。新增一个高质量的A级机会信号
                        `OPP_SQUEEZE_BREAKOUT_CONFIRMED_A`，它严格遵循“昨日压缩 + 今日确认突破”
                        的新作战条令，将一个低胜率的“准备状态”升级为高胜率的“战术机会”。
        """
        states = {}
        p = get_params_block(self.strategy, 'volatility_state_params')
        if not get_param_value(p.get('enabled'), False): return states
        required_cols = ['BBW_21_2.0_D', 'SLOPE_5_BBW_21_2.0_D', 'VOL_MA_21_D', 'volume_D', 'BBM_21_2.0_D']
        if not all(c in df.columns for c in required_cols):
            missing_cols = [c for c in required_cols if c not in df.columns]
            print(f"          -> [警告] 缺少诊断波动所需列，跳过。缺失: {missing_cols}")
            return states
        bbw_col = 'BBW_21_2.0_D'
        bbw_slope_col = 'SLOPE_5_BBW_21_2.0_D'
        vol_ma_col = 'VOL_MA_21_D'
        # --- 1. 静态分析：定义压缩事件和缩量状态 (作为观察哨，而非直接战斗信号) ---
        squeeze_threshold = df[bbw_col].rolling(60).quantile(get_param_value(p.get('squeeze_percentile'), 0.1))
        squeeze_event = (df[bbw_col] < squeeze_threshold) & (df[bbw_col].shift(1) >= squeeze_threshold)
        states['VOL_EVENT_SQUEEZE'] = squeeze_event
        states['VOL_STATE_SHRINKING'] = df['volume_D'] < df[vol_ma_col] * get_param_value(p.get('shrinking_ratio'), 0.8)
        # 1.1 定义持续低波动状态 (盘整期)
        low_vol_threshold = df[bbw_col].rolling(120).quantile(get_param_value(p.get('low_vol_percentile'), 0.20))
        states['VOL_STATE_LOW_VOLATILITY'] = df[bbw_col] < low_vol_threshold
        # 1.2 定义持续高波动状态 (高风险/高机会期)
        high_vol_threshold = df[bbw_col].rolling(120).quantile(get_param_value(p.get('high_vol_percentile'), 0.80))
        states['VOL_STATE_HIGH_VOLATILITY'] = df[bbw_col] > high_vol_threshold
        
        # --- 2. 状态机：生成基础的“压缩窗口” (作为观察哨) ---
        p_context = p.get('squeeze_context', {})
        volume_break_ratio = get_param_value(p_context.get('volume_break_ratio'), 1.5)
        break_condition = df['volume_D'] > df[vol_ma_col] * volume_break_ratio
        persistence_days = get_param_value(p_context.get('persistence_days'), 10)
        squeeze_window = create_persistent_state(
            df=df, entry_event_series=squeeze_event, persistence_days=persistence_days,
            break_condition_series=break_condition, state_name='VOL_STATE_SQUEEZE_WINDOW'
        )
        states['VOL_STATE_SQUEEZE_WINDOW'] = squeeze_window
        # --- 3. 【融合生成】高质量信号 ---
        is_still_squeezing = df[bbw_slope_col] < 0
        states['VOL_STATE_EXTREME_SQUEEZE'] = squeeze_window & is_still_squeezing
        is_expanding = df[bbw_slope_col] > 0
        high_expansion_threshold = df[bbw_col].rolling(60).quantile(0.8)
        is_in_high_expansion_zone = df[bbw_col] > high_expansion_threshold
        states['VOL_STATE_EXPANDING_SHARPLY'] = is_expanding & is_in_high_expansion_zone
        return states

    def diagnose_oscillator_states(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V234.2 机会提纯版】震荡指标状态诊断中心
        - 核心增强 (本次修改): 为“负向乖离”机会信号(OPP_STATE_NEGATIVE_DEVIATION)增加了一个关键的
                        过滤器：`df['close_D'] > df['open_D']`。
        - 战术收益: 此修改将一个纯粹的“超卖状态”提纯为了一个“超卖反弹机会”。它要求在满足统计学
                    超卖的当天，必须出现买盘力量压倒卖盘力量（收阳），这代表市场开始对超卖
                    状态做出积极反应，从而极大地提升了信号的实战价值和胜率。
        """
        states = {}
        p = get_params_block(self.strategy, 'oscillator_state_params')
        if not get_param_value(p.get('enabled'), False): return states
        
        # --- RSI 相关状态 ---
        rsi_col = 'RSI_13_D'
        rsi_slope_col = 'SLOPE_5_RSI_13_D' # [新增代码行]
        rsi_accel_col = 'ACCEL_5_RSI_13_D' # [新增代码行]
        if rsi_col in df.columns:
            states['OSC_STATE_RSI_OVERBOUGHT'] = df[rsi_col] > get_param_value(p.get('rsi_overbought'), 80)
            states['OSC_STATE_RSI_OVERSOLD'] = df[rsi_col] < get_param_value(p.get('rsi_oversold'), 25)

        # --- RSI 动态分析 ---
        if all(c in df.columns for c in [rsi_col, rsi_slope_col, rsi_accel_col]):
            # 机会信号: RSI在多头区域(>50)且仍在加速上行，是强劲的上涨动能信号
            states['OSC_DYN_RSI_ACCELERATING_BULLISH'] = (df[rsi_col] > 50) & (df[rsi_accel_col] > 0)
            # 风险信号: RSI在超买区(>70)但加速度转负(开始减速)，是典型的动能衰竭/顶背离风险
            states['OSC_DYN_RSI_DECELERATING_BEARISH'] = (df[rsi_col] > 70) & (df[rsi_accel_col] < 0)

        # --- MACD 相关状态 ---
        macd_h_col = 'MACDh_13_34_8_D'
        macd_z_col = 'MACD_HIST_ZSCORE_D'
        if macd_h_col in df.columns:
            states['OSC_STATE_MACD_BULLISH'] = df[macd_h_col] > 0
        if macd_z_col in df.columns:
            is_price_higher = df['close_D'] > df['close_D'].rolling(10).max().shift(1)
            is_macd_z_lower = df[macd_z_col] < df[macd_z_col].rolling(10).max().shift(1)
            states['OSC_STATE_MACD_DIVERGENCE'] = is_price_higher & is_macd_z_lower

        # --- BIAS机会状态的诊断 ---
        p_bias = p.get('bias_dynamic_threshold', {})
        bias_col = 'BIAS_55_D'
        if bias_col in df.columns:
            window = get_param_value(p_bias.get('window'), 120)
            quantile = get_param_value(p_bias.get('quantile'), 0.1)
            dynamic_oversold_threshold = df[bias_col].rolling(window=window).quantile(quantile)

            # 核心条件：乖离率低于动态阈值，进入超卖区
            is_oversold = df[bias_col] < dynamic_oversold_threshold
            # 增强过滤器：当天必须收阳，代表有资金开始尝试承接
            is_rebound_attempt = df['close_D'] > df['open_D']
            # 最终裁定：超卖状态 + 反弹尝试 = 高质量机会
            states['OPP_STATE_NEGATIVE_DEVIATION'] = is_oversold & is_rebound_attempt
        else:
            print(f"          -> [警告] 缺少诊断BIAS所需列 '{bias_col}'，跳过。")

        return states

    def diagnose_capital_and_range_states(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V1.0 新增】资金流与绝对波幅诊断模块
        - 核心职责: 基于CMF和ATR指标，生成资金流向和绝对价格波动范围的原子信号。
        """
        states = {}
        
        # --- 1. CMF 资金流状态 ---
        p_capital = get_params_block(self.strategy, 'capital_state_params')
        if get_param_value(p_capital.get('enabled'), False):
            cmf_col = 'CMF_21_D'
            if cmf_col in df.columns:
                bullish_threshold = get_param_value(p_capital.get('cmf_bullish_threshold'), 0.05)
                # 机会信号: CMF持续为正，表明资金在净流入，处于吸筹状态
                states['CAPITAL_STATE_ACCUMULATION_CONFIRMED'] = df[cmf_col] > bullish_threshold
                # 风险信号: CMF持续为负，表明资金在净流出，处于派发状态
                states['CAPITAL_STATE_DISTRIBUTION_CONFIRMED'] = df[cmf_col] < -bullish_threshold
            else:
                print(f"          -> [警告] 缺少诊断CMF所需列 '{cmf_col}'，跳过。")

        # --- 2. ATR 绝对波幅状态 ---
        atr_col = 'ATR_14_D'
        if atr_col in df.columns:
            # 状态1: ATR 压缩，代表市场进入绝对平静期，横盘或窄幅波动
            low_atr_threshold = df[atr_col].rolling(120).quantile(0.10)
            states['VOL_STATE_ATR_COMPRESSION'] = df[atr_col] < low_atr_threshold
            # 状态2: ATR 扩张，代表市场波动加剧，可能开启新趋势或进入高风险期
            high_atr_threshold = df[atr_col].rolling(120).quantile(0.90)
            states['VOL_STATE_ATR_EXPANSION'] = df[atr_col] > high_atr_threshold
        else:
            print(f"          -> [警告] 缺少诊断ATR所需列 '{atr_col}'，跳过。")
            
        return states

    def diagnose_synergy_states(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V1.0 新增】基础情报协同诊断模块
        - 核心职责: 交叉验证不同维度的基础原子信号，生成更高置信度的复合信号。
        """
        states = {}
        atomic = self.strategy.atomic_states # 从策略实例中获取已生成的原子状态
        default_series = pd.Series(False, index=df.index)

        # --- 军备检查: 确保依赖的原子信号已存在 ---
        required_states = [
            'VOL_STATE_HIGH_VOLATILITY', 'OSC_STATE_RSI_OVERBOUGHT',
            'VOL_STATE_EXTREME_SQUEEZE', 'OSC_STATE_RSI_OVERSOLD'
        ]
        if any(key not in atomic for key in required_states):
            missing = [k for k in required_states if k not in atomic]
            print(f"          -> [警告] 协同诊断缺少原子信号: {missing}，跳过。")
            return states

        # 信号1 (A级风险): “高位放量滞涨/冲顶”
        # 解读: 在市场波动率已经很高的区域，RSI同时进入超买区，这是典型的“过热”或“力竭”信号，
        #      后续回调风险极大。
        is_high_vol = atomic.get('VOL_STATE_HIGH_VOLATILITY', default_series)
        is_overbought = atomic.get('OSC_STATE_RSI_OVERBOUGHT', default_series)
        states['RISK_VOL_BLOWOFF_TOP_A'] = is_high_vol & is_overbought

        # 信号2 (B级机会): “压缩超卖见底”
        # 解读: 股价经历了长期的波动率压缩（多空力量平衡），同时RSI进入超卖区，
        #      表明下跌动能衰竭，空头力量耗尽，是一个潜在的左侧底部机会。
        is_squeezing = atomic.get('VOL_STATE_EXTREME_SQUEEZE', default_series)
        is_oversold = atomic.get('OSC_STATE_RSI_OVERSOLD', default_series)
        states['OPP_VOL_SQUEEZE_BOTTOM_B'] = is_squeezing & is_oversold

        return states

    def diagnose_multi_timeframe_synergy(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V1.0 新增】多时间维度协同诊断模块
        - 核心职责: 交叉验证日线与周线的基础指标动态，生成具备战略纵深的原子信号。
        """
        states = {}
        
        # --- 军备检查: 确保所有必需的日线和周线指标列都存在 ---
        required_cols = [
            'RSI_13_D', 'ACCEL_5_RSI_13_D',
            'RSI_13_W', 'SLOPE_5_RSI_13_W'
        ]
        if not all(c in df.columns for c in required_cols):
            missing = [c for c in required_cols if c not in df.columns]
            print(f"          -> [警告] 多时间维度协同诊断缺少必需列: {missing}，模块已跳过。")
            return states

        # --- 信号1 (A级机会): “周线看涨，日线点火” (MTF Bullish Alignment) ---
        # 战略 (周线): RSI趋势向上，代表大方向是多头。
        # 战术 (日线): RSI加速上行，代表短期买盘力量正在爆发。
        # 解读: 在顺风（周线）的航道上，引擎（日线）刚刚点火，是绝佳的右侧追涨信号。
        is_weekly_bullish = df['SLOPE_5_RSI_13_W'] > 0
        is_daily_igniting = df['ACCEL_5_RSI_13_D'] > 0
        states['OPP_MTF_RSI_BULLISH_ALIGNMENT_A'] = is_weekly_bullish & is_daily_igniting

        # --- 信号2 (S级风险): “周线背离，日线诱多” (MTF Bearish Divergence) ---
        # 战略 (周线): RSI趋势向下，代表长期动能已经衰竭。
        # 战术 (日线): RSI处于超买区，代表短期情绪过热，但随时可能逆转。
        # 解读: 望远镜看到的是悬崖，显微镜看到的却是鲜花。这是典型的诱多陷阱，风险极高。
        is_weekly_diverging = df['SLOPE_5_RSI_13_W'] < 0
        is_daily_overbought = df['RSI_13_D'] > 75 # 使用稍严格的75作为阈值
        states['RISK_MTF_RSI_BEARISH_DIVERGENCE_S'] = is_weekly_diverging & is_daily_overbought

        return states

    def diagnose_static_multi_dynamic_synergy(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V1.0 新增】静态-多动态协同诊断模块 (Combined Arms Doctrine)
        - 核心职责: 将一个静态的“战场环境”信号与多个动态的“进攻/撤退”信号进行交叉验证，
                      生成具备极高置信度的S级原子信号。
        """
        states = {}
        atomic = self.strategy.atomic_states
        default_series = pd.Series(False, index=df.index)

        # --- 军备检查: 确保所有必需的列和原子状态都存在 ---
        required_cols = [
            'ACCEL_5_close_D', 'ACCEL_5_RSI_13_D', 'SLOPE_5_BBW_21_2.0_D'
        ]
        required_states = ['VOL_STATE_EXTREME_SQUEEZE', 'VOL_STATE_HIGH_VOLATILITY']
        
        if not all(c in df.columns for c in required_cols):
            missing = [c for c in required_cols if c not in df.columns]
            print(f"          -> [警告] 静态-多动态协同诊断缺少必需列: {missing}，模块已跳过。")
            return states
        if not all(s in atomic for s in required_states):
            missing = [s for s in required_states if s not in atomic]
            print(f"          -> [警告] 静态-多动态协同诊断缺少原子状态: {missing}，模块已跳过。")
            return states

        # --- 信号1 (S级机会): “极致压缩后的协同突破” (Coiled Spring Breakout) ---
        # 静态环境 (Setup): 波动率处于极致压缩状态，市场如被压紧的弹簧。
        is_setup_ready = atomic.get('VOL_STATE_EXTREME_SQUEEZE', default_series)
        
        # 动态协同 (Trigger): 多个关键指标同时加速，形成“三军协同”的突破信号。
        # 1. 价格本身在加速上涨 (主力部队)
        is_price_accelerating = df['ACCEL_5_close_D'] > 0
        # 2. 内部动能RSI在加速 (士气高涨)
        is_rsi_accelerating = df['ACCEL_5_RSI_13_D'] > 0
        # 3. 波动率开始扩张 (弹簧释放)
        is_volatility_expanding = df['SLOPE_5_BBW_21_2.0_D'] > 0
        
        is_trigger_confirmed = is_price_accelerating & is_rsi_accelerating & is_volatility_expanding
        
        states['OPP_SQUEEZE_MULTI_CONFIRMED_BREAKOUT_S'] = is_setup_ready & is_trigger_confirmed

        # --- 信号2 (S级风险): “高波动区的背离诱多” (Exhaustion Rally Trap) ---
        # 静态环境 (Setup): 市场已处于高波动风险区，交投混乱，趋势随时可能逆转。
        is_setup_risky = atomic.get('VOL_STATE_HIGH_VOLATILITY', default_series)
        
        # 动态背离 (Trigger): 价格与内部动能发生严重背离。
        # 1. 价格仍在加速上涨，制造繁荣假象 (最后的疯狂)。
        is_price_still_accelerating = df['ACCEL_5_close_D'] > 0
        # 2. 但内部动能RSI已开始减速甚至转负，后续无力 (后继部队已撤退)。
        is_rsi_decelerating = df['ACCEL_5_RSI_13_D'] < 0
        
        is_trigger_diverging = is_price_still_accelerating & is_rsi_decelerating
        
        states['RISK_HIGH_VOL_DIVERGENT_RALLY_S'] = is_setup_risky & is_trigger_diverging
        
        return states

    def diagnose_classic_indicators(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V1.0 新增】经典技术指标诊断模块
        - 核心职责: 诊断MACD金叉/死叉事件和量比异动，生成经典的动能和成交量原子信号。
        - 收益: 捕捉被广泛市场参与者认可的交易信号，增强策略的普适性。
        """
        states = {}
        p = get_params_block(self.strategy, 'classic_indicator_params')
        if not get_param_value(p.get('enabled'), True): return states

        # --- 1. MACD 金叉/死叉事件 ---
        macd_line_col = 'MACD_13_34_8_D'
        signal_line_col = 'MACDs_13_34_8_D'
        if all(c in df.columns for c in [macd_line_col, signal_line_col]):
            macd_line = df[macd_line_col]
            signal_line = df[signal_line_col]
            # 机会信号 (B级): MACD金叉，短期动能上穿长期动能
            is_golden_cross = (macd_line > signal_line) & (macd_line.shift(1) <= signal_line.shift(1))
            states['OSC_TRIGGER_MACD_GOLDEN_CROSS_B'] = is_golden_cross

            # 风险信号 (B级): MACD死叉，短期动能下穿长期动能
            is_death_cross = (macd_line < signal_line) & (macd_line.shift(1) >= signal_line.shift(1))
            states['RISK_TRIGGER_MACD_DEATH_CROSS_B'] = is_death_cross
        else:
            print(f"          -> [警告] 缺少诊断MACD交叉所需列: '{macd_line_col}' 或 '{signal_line_col}'，跳过。")

        # --- 2. 量比异动信号 ---
        vol_ratio_col = 'volume_ratio_D'
        if vol_ratio_col in df.columns:
            volume_spike_threshold = get_param_value(p.get('volume_ratio_spike_threshold'), 2.5)
            is_volume_spike = df[vol_ratio_col] > volume_spike_threshold
            # 机会信号 (A级): 放量上涨，量价齐升，是强烈的买盘确认信号
            is_price_up = df['close_D'] > df['open_D']
            states['VOL_PRICE_SPIKE_UP_A'] = is_volume_spike & is_price_up

            # 风险信号 (A级): 放量下跌，恐慌或出货迹象，是强烈的风险预警
            is_price_down = df['close_D'] < df['open_D']
            states['RISK_VOL_PRICE_SPIKE_DOWN_A'] = is_volume_spike & is_price_down
        else:
            print(f"          -> [警告] 缺少诊断量比所需列 '{vol_ratio_col}'，跳过。")

        return states










