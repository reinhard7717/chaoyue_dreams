# 文件: strategies/trend_following/intelligence/playbook_engine.py
# 剧本与触发器引擎
import pandas as pd
from typing import Dict, Tuple, List
import numpy as np

from strategies.trend_following.utils import get_params_block, get_param_value

class PlaybookEngine:
    def __init__(self, strategy_instance):
        """
        初始化剧本与触发器引擎。
        :param strategy_instance: 策略主实例的引用。
        """
        self.strategy = strategy_instance
        self.playbook_blueprints = self._get_playbook_blueprints()
        self.kline_params = get_params_block(self.strategy, 'kline_pattern_params')

    def _get_playbook_blueprints(self) -> List[Dict]:
        """
        【V2.1 双层打击版】剧本蓝图知识库
        - 核心升级: 为突破类剧本换装全新的S级/A级双层触发器，提升战术适应性。
        """
        return [
            {
                'name': 'PLAYBOOK_EXTREME_SQUEEZE_EXPLOSION_S_PLUS',
                'setup': ['VOL_STATE_EXTREME_SQUEEZE'],
                'trigger': ['TRIGGER_EXPLOSIVE_BREAKOUT_S'],
                'comment': 'S+级 - 波动极致压缩后，出现高确定性的暴力突破。'
            },
            {
                'name': 'PLAYBOOK_BREAKOUT_EVE_S',
                'setup': ['STRUCTURE_BREAKOUT_EVE_S'], 
                'trigger': ['TRIGGER_EXPLOSIVE_BREAKOUT_S', 'TRIGGER_GRINDING_ADVANCE_A'],
                'comment': 'S级 - 在结构、波动、筹码三维共振的“突破前夜”发动的总攻。'
            },
            {
                'name': 'PLAYBOOK_NORMAL_SQUEEZE_BREAKOUT_A',
                'setup': ['VOL_STATE_SQUEEZE_WINDOW'],
                'trigger': ['TRIGGER_EXPLOSIVE_BREAKOUT_S', 'TRIGGER_GRINDING_ADVANCE_A'],
                'comment': 'A级 - 波动压缩后，出现暴力突破或温和推进。'
            },
            {
                'name': 'PLAYBOOK_PULLBACK_REBOUND_A',
                'setup': ['STRUCTURE_MAIN_UPTREND_WAVE_S'],
                'trigger': ['TRIGGER_PULLBACK_REBOUND'],
                'comment': 'A级 - 主升浪结构中，出现均线或平台的回踩反弹。'
            },
            {
                'name': 'PLAYBOOK_STABLE_PLATFORM_REBOUND_A_PLUS',
                'setup': ['PLATFORM_STATE_STABLE_FORMED'], # 前置条件：必须已形成稳固平台
                'trigger': ['TRIGGER_PLATFORM_PULLBACK_REBOUND'], # 触发事件：在该平台上发生回踩反弹
                'comment': 'A+级 - 在已形成的稳固筹码平台上发生的回踩反弹，确定性更高。'
            },
            {
                'name': 'PLAYBOOK_GOLDEN_PIT_A_PLUS',
                'setup': ['OPP_CONSTRUCTIVE_WASHOUT_ABSORPTION_A'],
                'trigger': ['TRIGGER_DOMINANT_REVERSAL'],
                'comment': 'A+级 - 主力打压吸筹后，出现显性反转K线确认。'
            },
            {
                'name': 'PLAYBOOK_MEAN_REVERSION_A',
                'setup': ['OPP_STATE_NEGATIVE_DEVIATION', 'OSC_STATE_RSI_OVERSOLD'],
                'trigger': ['OPP_BEHAVIOR_SELLING_EXHAUSTION_A', 'TRIGGER_DOMINANT_REversal'],
                'comment': 'A级 - 统计学超卖 + 卖盘衰竭或反转K线。'
            },
            # “箱体底部反转”剧本
            {
                'name': 'PLAYBOOK_BOX_REVERSAL_B',
                'setup': ['BOX_STATE_HEALTHY_CONSOLIDATION', 'VOL_STATE_SQUEEZE_WINDOW'],
                'trigger': ['TRIGGER_BOX_BOTTOM_REVERSAL'],
                'comment': 'B级 - 在健康箱体或压缩区底部，出现反转信号，执行低吸。'
            }
        ]

    def define_trigger_events(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V234.0 最终净化版 - 战术触发事件定义中心】
        - 核心升级: 严格遵循“V234.0 作战条例”，所有参数均从唯一的 trigger_event_params 配置块中获取，
                    确保了配置的单一来源原则，使整个触发体系清晰、健壮、易于维护。
        - 职责: 识别所有可以作为“开火信号”的瞬时战术事件(Trigger)。
        """
        # print("        -> [触发事件中心 V234.0] 启动，正在定义所有原子化触发事件...")
        triggers = {}
        default_series = pd.Series(False, index=df.index)
        trigger_params = get_params_block(self.strategy, 'trigger_event_params')
        if not get_param_value(trigger_params.get('enabled'), True):
            print("          -> 触发事件引擎被禁用，跳过。")
            return triggers
        vol_ma_col = 'VOL_MA_21_D'
        # --- 1. K线形态触发器 (Candlestick Triggers) ---
        # 1.1 【通用级】反转确认阳线
        p_reversal = trigger_params.get('reversal_confirmation_candle', {})
        if get_param_value(p_reversal.get('enabled'), True):
            is_green = df['close_D'] > df['open_D']
            is_strong_rally = df['pct_change_D'] > get_param_value(p_reversal.get('min_pct_change'), 0.03)
            is_closing_strong = df['close_D'] > (df['high_D'] + df['low_D']) / 2
            triggers['TRIGGER_REVERSAL_CONFIRMATION_CANDLE'] = is_green & is_strong_rally & is_closing_strong

        # 1.2 【精英级】显性反转阳线 (在通用级基础上，要求力量压制前一日)
        p_dominant = trigger_params.get('dominant_reversal_candle', {})
        if get_param_value(p_dominant.get('enabled'), True):
            base_reversal_signal = triggers.get('TRIGGER_REVERSAL_CONFIRMATION_CANDLE', default_series)
            today_body_size = df['close_D'] - df['open_D']
            yesterday_body_size = abs(df['close_D'].shift(1) - df['open_D'].shift(1))
            was_yesterday_red = df['close_D'].shift(1) < df['open_D'].shift(1)
            recovery_ratio = get_param_value(p_dominant.get('recovery_ratio'), 0.5)
            is_power_recovered = today_body_size >= (yesterday_body_size * recovery_ratio)
            triggers['TRIGGER_DOMINANT_REVERSAL'] = base_reversal_signal & (~was_yesterday_red | is_power_recovered)

        # 1.3 【企稳型】突破阳线 (通常用于底部企稳或平台整理后的首次突破)
        p_breakout_candle = trigger_params.get('breakout_candle', {})
        if get_param_value(p_breakout_candle.get('enabled'), True):
            boll_mid_col = 'BBM_21_2.0_D'
            if boll_mid_col in df.columns:
                min_body_ratio = get_param_value(p_breakout_candle.get('min_body_ratio'), 0.4)
                body_range = (df['high_D'] - df['low_D']).replace(0, np.nan)
                is_strong_positive_candle = (
                    (df['close_D'] > df['open_D']) &
                    (((df['close_D'] - df['open_D']) / body_range).fillna(1.0) >= min_body_ratio)
                )
                is_breaking_boll_mid = df['close_D'] > df[boll_mid_col]
                triggers['TRIGGER_BREAKOUT_CANDLE'] = is_strong_positive_candle & is_breaking_boll_mid

        # 1.4 【进攻型】能量释放阳线 (强调实体和成交量的双重确认)
        p_energy = trigger_params.get('energy_release', {})
        if get_param_value(p_energy.get('enabled'), True) and vol_ma_col in df.columns:
            is_positive_day = df['close_D'] > df['open_D']
            body_range = (df['high_D'] - df['low_D']).replace(0, np.nan)
            body_ratio = (df['close_D'] - df['open_D']) / body_range
            is_strong_body = body_ratio.fillna(1.0) > get_param_value(p_energy.get('min_body_ratio'), 0.5)
            volume_ratio = get_param_value(p_energy.get('volume_ratio'), 1.5)
            is_volume_spike = df['volume_D'] > df[vol_ma_col] * volume_ratio
            triggers['TRIGGER_ENERGY_RELEASE'] = is_positive_day & is_strong_body & is_volume_spike

        # --- 2. 结构与趋势触发器 (Structure & Trend Triggers) ---
        # 2.1 【经典】放量突破近期高点
        p_vol_breakout = trigger_params.get('volume_spike_breakout', {})
        if get_param_value(p_vol_breakout.get('enabled'), True) and vol_ma_col in df.columns:
            volume_ratio = get_param_value(p_vol_breakout.get('volume_ratio'), 2.0)
            lookback = get_param_value(p_vol_breakout.get('lookback_period'), 20)
            is_volume_spike = df['volume_D'] > df[vol_ma_col] * volume_ratio
            is_price_breakout = df['close_D'] > df['high_D'].shift(1).rolling(lookback).max()
            triggers['TRIGGER_VOLUME_SPIKE_BREAKOUT'] = is_volume_spike & is_price_breakout

        # 2.2 【均线】回踩支撑反弹
        p_ma_rebound = trigger_params.get('pullback_rebound_trigger_params', {})
        if get_param_value(p_ma_rebound.get('enabled'), True):
            support_ma_period = get_param_value(p_ma_rebound.get('support_ma'), 21)
            support_ma_col = f'EMA_{support_ma_period}_D'
            if support_ma_col in df.columns:
                was_touching_support = df['low_D'].shift(1) <= df[support_ma_col].shift(1)
                is_rebounded_above = df['close_D'] > df[support_ma_col]
                is_positive_day = df['close_D'] > df['open_D']
                triggers['TRIGGER_PULLBACK_REBOUND'] = was_touching_support & is_rebounded_above & is_positive_day

        # 2.3 【筹码】回踩平台反弹 (S级战术动作)
        p_platform_rebound = trigger_params.get('platform_pullback_trigger_params', {})
        if get_param_value(p_platform_rebound.get('enabled'), True):
            platform_price_col = 'PLATFORM_PRICE_STABLE'
            if platform_price_col in df.columns:
                proximity_ratio = get_param_value(p_platform_rebound.get('proximity_ratio'), 0.01)
                is_touching_platform = df['low_D'] <= df[platform_price_col] * (1 + proximity_ratio)
                is_closing_above = df['close_D'] > df[platform_price_col]
                is_positive_day = df['close_D'] > df['open_D']
                is_shrinking_volume = df['volume_D'] < df[vol_ma_col]
                triggers['TRIGGER_PLATFORM_PULLBACK_REBOUND'] = is_touching_platform & is_closing_above & is_positive_day & is_shrinking_volume

        # 2.4 【趋势】趋势延续确认K线
        p_cont = trigger_params.get('trend_continuation_candle', {})
        if get_param_value(p_cont.get('enabled'), True):
            lookback_period = get_param_value(p_cont.get('lookback_period'), 8)
            is_positive_day = df['close_D'] > df['open_D']
            is_new_high = df['close_D'] >= df['high_D'].shift(1).rolling(window=lookback_period).max()
            triggers['TRIGGER_TREND_CONTINUATION_CANDLE'] = is_positive_day & is_new_high
        
        # 双层突破识别系统，以应对不同市场环境，取代单一、僵化的突破定义。
        # --- S级触发器 (攻城锤): 暴力突破 ---
        # 逻辑: 严格的三维标准（创20日高点 + 2倍以上放量 + 光头强实体阳线），捕捉高确定性的暴力突破。
        is_price_breakout_s = df['close_D'] >= df['high_D'].rolling(20).max().shift(1)
        volume_ma_20 = df['volume_D'].rolling(20).mean()
        is_volume_confirmed_s = df['volume_D'] > (volume_ma_20 * 2.0)
        price_range_s = (df['high_D'] - df['low_D']).replace(0, 0.0001)
        close_position_in_range_s = (df['close_D'] - df['low_D']) / price_range_s
        is_strength_confirmed_s = close_position_in_range_s >= 0.9
        triggers['TRIGGER_EXPLOSIVE_BREAKOUT_S'] = is_price_breakout_s & is_volume_confirmed_s & is_strength_confirmed_s

        # --- A级触发器 (手术刀): 温和推进 ---
        # 逻辑: 更灵活的标准，捕捉温和、持续、重心不断上移的“碎步”上涨。
        # 维度1: 位置 - 突破短期动量 (8日高点)
        is_price_breakout_a = df['close_D'] >= df['high_D'].rolling(8).max().shift(1)
        # 维度2: 能量 - 成交量健康温和 (>20日均量)
        is_volume_confirmed_a = df['volume_D'] > volume_ma_20
        # 维度3: 特征 - 重心上移 (当日优势 + 连续攻击)
        is_winner_of_the_day = df['close_D'] > (df['high_D'] + df['low_D']) / 2
        is_consecutive_attack = (df['high_D'] > df['high_D'].shift(1)) & (df['low_D'] > df['low_D'].shift(1))
        is_character_confirmed_a = is_winner_of_the_day & is_consecutive_attack
        triggers['TRIGGER_GRINDING_ADVANCE_A'] = is_price_breakout_a & is_volume_confirmed_a & is_character_confirmed_a

        # --- 3. 复合形态与指标触发器 (Pattern & Indicator Triggers) ---
        # 3.1 N字形态突破 (依赖原子状态)
        p_nshape = self.kline_params.get('n_shape_params', {})
        if get_param_value(p_nshape.get('enabled'), True):
            n_shape_consolidation_state = self.strategy.atomic_states.get('KLINE_STATE_N_SHAPE_CONSOLIDATION', default_series)
            consolidation_high = df['high_D'].where(n_shape_consolidation_state, np.nan).ffill()
            is_breaking_consolidation = df['close_D'] > consolidation_high.shift(1)
            triggers['TRIGGER_N_SHAPE_BREAKOUT'] = (df['close_D'] > df['open_D']) & is_breaking_consolidation

        # 3.2 指标金叉 (MACD)
        p_cross = trigger_params.get('indicator_cross_params', {})
        if get_param_value(p_cross.get('enabled'), True):
            macd_p = p_cross.get('macd_cross', {})
            if get_param_value(macd_p.get('enabled'), True):
                macd_col, signal_col = 'MACD_13_34_8_D', 'MACDs_13_34_8_D'
                if all(c in df.columns for c in [macd_col, signal_col]):
                    is_golden_cross = (df[macd_col] > df[signal_col]) & (df[macd_col].shift(1) <= df[signal_col].shift(1))
                    low_level = get_param_value(macd_p.get('low_level'), -0.5)
                    triggers['TRIGGER_MACD_LOW_CROSS'] = is_golden_cross & (df[macd_col] < low_level)

        # --- 4. 从其他诊断模块接收的事件 (Event Reception) ---
        # 这些事件由其他专业部门生成，本部门只负责接收和汇报
        triggers['TRIGGER_BOX_BREAKOUT'] = self.strategy.atomic_states.get('BOX_EVENT_BREAKOUT', default_series)
        triggers['TRIGGER_EARTH_HEAVEN_BOARD'] = self.strategy.atomic_states.get('BOARD_EVENT_EARTH_HEAVEN', default_series)
        triggers['TRIGGER_TREND_STABILIZING'] = self.strategy.atomic_states.get('MA_STATE_D_STABILIZING', default_series)

        # [核心新增] 定义“箱体底部反转”复合触发器
        p_box_reversal = trigger_params.get('box_bottom_reversal', {})
        if get_param_value(p_box_reversal.get('enabled'), True) and 'box_bottom_D' in df.columns:
            # 条件1: 价格触及或轻微跌破箱体下轨
            proximity_ratio = get_param_value(p_box_reversal.get('proximity_ratio'), 0.015)
            is_near_bottom = df['low_D'] <= df['box_bottom_D'] * (1 + proximity_ratio)
            
            # 条件2: 出现显性的反转K线作为确认信号
            is_reversal_confirmed = triggers.get('TRIGGER_DOMINANT_REVERSAL', default_series)
            
            triggers['TRIGGER_BOX_BOTTOM_REVERSAL'] = is_near_bottom & is_reversal_confirmed

        # --- 5. 最终安全检查 (Final Safety Check) ---
        # 确保所有触发器都已正确初始化，防止因计算失败导致后续流程出错
        for key in list(triggers.keys()):
            if triggers[key] is None:
                triggers[key] = pd.Series(False, index=df.index)
            else:
                triggers[key] = triggers[key].fillna(False)
                
        # print("        -> [触发事件中心 V234.0] 所有触发事件定义完成。")
        return triggers

    def generate_playbook_states(self, trigger_events: Dict[str, pd.Series]) -> Tuple[Dict[str, pd.Series], Dict[str, pd.Series]]:
        """
        【V2.0 大一统重构版】剧本状态生成引擎
        - 核心重构: 采用统一的、数据驱动的模式生成所有剧本状态。
        """
        # print("        -> [剧本状态生成引擎 V2.0 大一统版] 启动...")
        df = self.strategy.df_indicators
        if df.empty:
            print("          -> [警告] 主数据帧为空，无法生成剧本状态。")
            return {}, {}
        playbook_states = {}
        atomic_states = self.strategy.atomic_states
        default_series = pd.Series(False, index=df.index)

        for blueprint in self.playbook_blueprints:
            playbook_name = blueprint['name']
            
            # --- 1. 组合 Setup 条件 (any_of) ---
            setup_conditions = pd.Series(False, index=df.index)
            for state_name in blueprint.get('setup', []):
                setup_conditions |= atomic_states.get(state_name, default_series)
            
            # --- 2. 组合 Trigger 条件 (any_of) ---
            trigger_conditions = pd.Series(False, index=df.index)
            for trigger_name in blueprint.get('trigger', []):
                trigger_conditions |= trigger_events.get(trigger_name, default_series)

            # --- 3. 最终裁定: Setup 和 Trigger 的共振 ---
            # 通常，我们检查触发当天的 Trigger 和前一天的 Setup
            final_signal = setup_conditions.shift(1).fillna(False) & trigger_conditions
            playbook_states[playbook_name] = final_signal

        print(f"        -> [剧本状态生成引擎 V2.0] 分析完毕，共生成 {len(playbook_states)} 个标准剧本状态。")
        # setup_scores 在这个新架构下不再需要由本模块生成，返回空字典
        return {}, playbook_states