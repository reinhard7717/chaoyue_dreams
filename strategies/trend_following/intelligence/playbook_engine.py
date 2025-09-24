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
    def generate_playbook_states(self, trigger_events: Dict[str, pd.Series]) -> Tuple[Dict[str, pd.Series], Dict[str, pd.Series]]:
        """
        【V4.0 逻辑简化版】剧本状态生成引擎
        - 核心重构 (本次修改):
          - [逻辑简化] 移除了复杂的“战场环境(Contextual Setups)”定义。
          - 由于新的剧本蓝图直接依赖于高质量的触发器，这些触发器本身已经内含了对环境的判断，
            因此不再需要在此处进行重复的环境检查。
        - 收益: 引擎职责更纯粹，只负责根据蓝图执行“触发”逻辑，代码更简洁、高效。
        """
        df = self.strategy.df_indicators
        if df.empty:
            return {}, {}
        playbook_states = {}
        default_series = pd.Series(False, index=df.index)
        # --- 步骤 1: 循环执行剧本蓝图 ---
        for blueprint in self.playbook_blueprints:
            playbook_name = blueprint['name']
            # 组合 Trigger 条件 (OR logic)
            trigger_conditions = pd.Series(False, index=df.index)
            for trigger_name in blueprint.get('trigger', []):
                trigger_conditions |= trigger_events.get(trigger_name, default_series)
            playbook_states[playbook_name] = trigger_conditions
        # setup_scores 在这个新架构下不再需要，返回空字典
        return {}, playbook_states

    def _get_playbook_blueprints(self) -> List[Dict]:
        """
        【V6.1 均值回归扩充版】剧本蓝图知识库
        - 核心升级 (本次修改):
          - [王牌剧本] 新增 `PLAYBOOK_TRUE_ACCUMULATION_BREAKOUT_S_PLUS` 剧本，专门捕捉经过多维度交叉验证的“真实吸筹”后的突破机会。
          - [逆向剧本] 新增 `PLAYBOOK_CAPITULATION_REVERSAL_A_PLUS` 剧本，用于执行“恐慌盘投降反转”这一高胜率左侧交易。
          - [剧本优化] 将“筹码共振-价格滞后”剧本的触发器升级为消费更可靠的“真实吸筹”信号。
          - [战术扩充] 新增 `PLAYBOOK_MEAN_REVERSION_GRID_A` 剧本，用于执行震荡市的网格交易。
        """
        
        return [
            # 新增V型反转王牌剧本，并置于最前，体现其最高优先级
            {
                'name': 'PLAYBOOK_V_REVERSAL_ACE_S_PLUS',
                'trigger': ['TRIGGER_V_REVERSAL_ACE_S_PLUS'],
                'comment': 'S++级(王牌) - [V型反转] 在前一日确认的“恐慌抛售”后，由当日强劲的“显性反转K线”确认的最高确定性反转剧本。'
            },
            # --- 基于“真实吸筹”的王牌剧本 ---
            {
                'name': 'PLAYBOOK_TRUE_ACCUMULATION_BREAKOUT_S_PLUS',
                'trigger': ['TRIGGER_TRUE_ACCUMULATION_BREAKOUT_S_PLUS'],
                'comment': 'S++级(王牌) - [真实吸筹突破] 在确认的“真实吸筹”阶段后，由多域点火共振确认的突破，是最高置信度的买点之一。'
            },
            {
                'name': 'PLAYBOOK_CONVICTION_BREAKOUT_S_PLUS',
                'trigger': ['TRIGGER_CONVICTION_BREAKOUT_S_PLUS'],
                'comment': 'S++级(王牌) - [信念突破] 由资金流情报部门直接确认的、核心主力带头、买盘形成碾压的最高质量突破。'
            },
            # 大反转后期·共振初起 (逆向王牌)
            {
                'name': 'PLAYBOOK_POST_REVERSAL_RESONANCE_S_PLUS',
                'trigger': ['TRIGGER_POST_REVERSAL_RESONANCE_S_PLUS'],
                'comment': 'S++级(逆向王牌) - [大反转后期·共振初起] 在股价深度下跌、主力完成吸筹后，由趋势企稳和早期动能确认的、最安全的左侧转右侧买点。'
            },
            # --- 优化“筹码共振-价格滞后”剧本 ---
            {
                'name': 'PLAYBOOK_CHIP_RESONANCE_PRICE_LAG_BREAKOUT_S',
                'trigger': ['TRIGGER_CHIP_RESONANCE_PRICE_LAG_BREAKOUT_S'],
                'comment': 'S级(王牌潜伏) - [筹码共振-价格滞后] 在筹码高度共振、价格被压制的战备状态后，由温和点火信号确认的黄金突破口。'
            },
            # --- S++级 王牌剧本 (Ace Playbooks) ---
            {
                'name': 'PLAYBOOK_SUSTAINED_IGNITION_S_PLUS_PLUS',
                'trigger': ['TRIGGER_SUSTAINED_IGNITION_S_PLUS'],
                'comment': 'S++级(王牌) - [持续点火] “多域点火”信号经过了后续观察期的考验，确认了趋势的可持续性。'
            },
            {
                'name': 'PLAYBOOK_PRIME_CHIP_IGNITION_S_PLUS_PLUS',
                'trigger': ['TRIGGER_PRIME_CHIP_IGNITION_S_PLUS_PLUS'],
                'comment': 'S++级(王牌) - [黄金筹码] 在“黄金筹码结构”形成后，由【经过确认的】“多域点火共振”确认总攻。'
            },
            # --- S+级 核心增强剧本 (Enhanced Core Playbooks) ---
            {
                'name': 'PLAYBOOK_CORE_HOLDER_IGNITION_S_PLUS',
                'trigger': ['TRIGGER_CORE_HOLDER_IGNITION_S_PLUS'],
                'comment': 'S+级 - [核心庄家点火] 在核心持仓者完成隐蔽吸筹后，由“锁仓再集中”信号确认启动，规避洗盘。'
            },
            {
                'name': 'PLAYBOOK_EXTREME_SQUEEZE_EXPLOSION_S_PLUS',
                'trigger': ['TRIGGER_EXTREME_SQUEEZE_EXPLOSION_S_PLUS'],
                'comment': 'S+级 - [极致压缩突破] 在波动率极致压缩后，由高强度的压缩突破信号确认。'
            },
            # --- S级 核心剧本 (Core Playbooks) ---
            {
                'name': 'PLAYBOOK_IGNITION_RESONANCE_S',
                'trigger': ['TRIGGER_UPTREND_IGNITION_RESONANCE_S'],
                'comment': 'S级 - [主升浪点火] 在确认的主升浪结构中，由“多域点火共振”确认总攻，规避回踩。'
            },
            {
                'name': 'PLAYBOOK_BOTTOM_REVERSAL_S',
                'trigger': ['TRIGGER_BOTTOM_REVERSAL_RESONANCE_S'],
                'comment': 'S级 - [底部反转] 由“多域底部反转共振”确认战略拐点。'
            },
            {
                'name': 'PLAYBOOK_BREAKOUT_EVE_S',
                'trigger': ['TRIGGER_BREAKOUT_EVE_S'],
                'comment': 'S级 - [突破前夜] 在平台和波动率双重压缩的突破前夜，由最高级别的点火共振信号确认。'
            },
            # --- A+级 优势战术剧本 (Advantaged Tactical Playbooks) ---
            # --- 基于“恐慌盘投降”的逆向剧本 ---
            {
                'name': 'PLAYBOOK_CAPITULATION_REVERSAL_A_PLUS',
                'trigger': ['TRIGGER_CAPITULATION_REVERSAL_A_PLUS'],
                'comment': 'A+级(逆向) - [恐慌盘投降反转] 捕捉市场极度悲观，套牢盘集中割肉后的高胜率V型反转机会。'
            },
            {
                'name': 'PLAYBOOK_WASHOUT_ABSORPTION_REVERSAL_A_PLUS',
                'trigger': ['TRIGGER_WASHOUT_ABSORPTION_REVERSAL_A_PLUS'],
                'comment': 'A+级 - [洗盘吸筹反转] 在确认的“洗盘吸筹”行为后，由“显性反转K线”确认V型反转。'
            },
            # --- A级 战术剧本 (Tactical Playbooks) ---
            # 新增均值回归网格剧本
            {
                'name': 'PLAYBOOK_MEAN_REVERSION_GRID_A',
                'trigger': ['TRIGGER_MEAN_REVERSION_GRID_BUY_A'],
                'comment': 'A级 - [均值回归网格] 在震荡市中，于价格触及统计下轨时买入。'
            },
            {
                'name': 'PLAYBOOK_HEALTHY_PULLBACK_A',
                'trigger': ['TRIGGER_HEALTHY_PULLBACK_CONFIRMED_S'],
                'comment': 'A级 - [健康回踩] 捕捉经过确认的健康回踩买点。'
            },
            {
                'name': 'PLAYBOOK_CLASSIC_PATTERN_BREAKOUT_A',
                'trigger': ['TRIGGER_CLASSIC_PATTERN_BREAKOUT_S'],
                'comment': 'A级 - [形态突破] 执行对经典看涨形态的突破。'
            },
            {
                'name': 'PLAYBOOK_SHAKEOUT_REVERSAL_A',
                'trigger': ['TRIGGER_SHAKEOUT_REVERSAL_A'],
                'comment': 'A级 - [洗盘反转] 捕捉“压缩区洗盘后反转”的V型机会。'
            },
            {
                'name': 'PLAYBOOK_CONSOLIDATION_BREAKOUT_A',
                'trigger': ['TRIGGER_CONSOLIDATION_BREAKOUT_A'],
                'comment': 'A级 - [盘整突破] 执行对盘整中继形态的突破。'
            },
            {
                'name': 'PLAYBOOK_NORMAL_SQUEEZE_BREAKOUT_A',
                'trigger': ['TRIGGER_NORMAL_SQUEEZE_BREAKOUT_A'],
                'comment': 'A级 - [常规压缩突破] 在常规波动率压缩后，由任意压缩突破信号确认。'
            },
        ]

    def _deploy_v_reversal_probe(self, df: pd.DataFrame, atomic: Dict, probe_date: str, setup_score: pd.Series, trigger_score: pd.Series, setup_threshold: float, trigger_threshold: float):
        """
        【V1.0 新增】V型反转法医探针
        - 核心职责: 深度解剖 V型反转王牌剧本的触发逻辑，清晰展示“昨日战备”与“今日点火”两个核心环节的得分与判定过程。
        """
        print("\n" + "="*35 + f" [V型反转法医探针 V1.0] 正在解剖 {probe_date} 的触发逻辑 " + "="*35)
        try:
            probe_ts_naive = pd.to_datetime(probe_date)
            if df.index.tz is not None:
                probe_ts = probe_ts_naive.tz_localize(df.index.tz)
            else:
                probe_ts = probe_ts_naive
            
            yesterday_ts = probe_ts - pd.Timedelta(days=1)
            while yesterday_ts not in df.index and yesterday_ts > df.index.min():
                yesterday_ts -= pd.Timedelta(days=1)

            if probe_ts not in df.index or yesterday_ts not in df.index:
                print(f"  [错误] 探针日期 {probe_date} 或其前一个交易日不在数据范围内。解剖终止。")
                return

            # --- 第一部分: 解剖昨日战备分 ---
            print(f"\n--- [第一部分: 解剖 {yesterday_ts.date()} 的战备分 (Setup)] ---")
            was_setup_yesterday_score = setup_score.shift(1).get(probe_ts, 0.0)
            print(f"  - 昨日战备分 (SCORE_SETUP_PANIC_SELLING_S): {was_setup_yesterday_score:.4f}")
            print(f"  - 战备阈值: {setup_threshold:.4f}")
            was_setup_ok = was_setup_yesterday_score > setup_threshold
            print(f"  - [判定] 昨日战备是否就绪? {'✅ 是' if was_setup_ok else '❌ 否'}")
            
            print("    -> 战备分由以下三者相乘得到:")
            price_drop = self._normalize_score(df['pct_change_D'].clip(upper=0), 60, False).get(yesterday_ts, 0.0)
            volume_spike = self._normalize_score(df['volume_D'] / df['VOL_MA_21_D'], 60, True).get(yesterday_ts, 0.0)
            chip_breakdown = self._fuse_multi_level_scores(df, 'CHIP_BEARISH_RESONANCE').get(yesterday_ts, 0.0)
            print(f"      - 价格下跌分: {price_drop:.4f}")
            print(f"      - 成交放量分: {volume_spike:.4f}")
            print(f"      - 筹码崩溃分: {chip_breakdown:.4f}")
            
            # --- 第二部分: 解剖今日点火分 ---
            print(f"\n--- [第二部分: 解剖 {probe_ts.date()} 的点火分 (Trigger)] ---")
            is_triggered_today_score = trigger_score.get(probe_ts, 0.0)
            print(f"  - 今日点火分 (TRIGGER_DOMINANT_REVERSAL): {is_triggered_today_score:.4f}")
            print(f"  - 点火阈值: {trigger_threshold:.4f}")
            is_trigger_ok = is_triggered_today_score > trigger_threshold
            print(f"  - [判定] 今日是否成功点火? {'✅ 是' if is_trigger_ok else '❌ 否'}")

            print("    -> 点火分由以下三者相乘得到:")
            p_dominant = get_params_block(self.strategy, 'trigger_event_params', {}).get('dominant_reversal_candle', {})
            today_body_size = (df.at[probe_ts, 'close_D'] - df.at[probe_ts, 'open_D'])
            yesterday_body_size = (df.at[yesterday_ts, 'open_D'] - df.at[yesterday_ts, 'close_D'])
            recovery = (today_body_size / yesterday_body_size if yesterday_body_size > 0 else 0)
            recovery_score = np.clip(recovery, 0, 2) / 2.0
            
            lookback = get_param_value(p_dominant.get('position_lookback_days'), 60)
            rolling_high = df['high_D'].rolling(lookback).max().at[probe_ts]
            rolling_low = df['low_D'].rolling(lookback).min().at[probe_ts]
            price_range = (rolling_high - rolling_low) if (rolling_high - rolling_low) > 1e-9 else 1e-9
            position_score = 1.0 - np.clip(((df.at[probe_ts, 'close_D'] - rolling_low) / price_range), 0, 1)
            
            candle_quality = atomic.get('COGNITIVE_SCORE_EARLY_MOMENTUM_IGNITION_A', pd.Series(0.0, index=df.index)).get(probe_ts, 0.0)
            print(f"      - K线收复分: {recovery_score:.4f}")
            print(f"      - 底部位置分: {position_score:.4f}")
            print(f"      - K线质量分: {candle_quality:.4f}")

            # --- 第三部分: 最终结论 ---
            print("\n--- [第三部分: 最终结论] ---")
            final_trigger_result = was_setup_ok and is_trigger_ok
            print(f"  - 触发器最终逻辑: (昨日战备就绪 AND 今日成功点火)")
            print(f"  - 计算结果: ({was_setup_ok} AND {is_trigger_ok}) = {final_trigger_result}")
            print(f"  - [结论] V型反转王牌剧本在 {probe_date} {'✅ 已触发' if final_trigger_result else '❌ 未触发'}")

        except Exception as e:
            print(f"  [探针错误] 在执行V型反转法医探针时发生异常: {e}")
        finally:
            print("="*95 + "\n")

    def define_trigger_events(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V8.1 · 法医探针集成版】战术触发事件定义中心
        - 核心升级 (本次修改):
          - [新增探针] 集成了“V型反转法医探针”，可在调试模式下深度解剖该王牌剧本的触发逻辑。
        """
        print("      -> [战术触发事件定义中心 V8.1 · 法医探针集成版] 启动...") # [代码修改] 更新版本号和说明
        triggers = {}
        atomic = self.strategy.atomic_states
        default_score = pd.Series(0.0, index=df.index, dtype=np.float32)
        default_series = pd.Series(False, index=df.index)
        p_triggers = get_params_block(self.strategy, 'trigger_event_params', {})
        thresholds = {
            'ignition_s': get_param_value(p_triggers.get('ignition_s_threshold'), 0.5),
            'bottom_reversal_s': get_param_value(p_triggers.get('bottom_reversal_s_threshold'), 0.4),
            'healthy_pullback_s': get_param_value(p_triggers.get('healthy_pullback_s_threshold'), 0.3),
            'classic_pattern_s': get_param_value(p_triggers.get('classic_pattern_s_threshold'), 0.4),
            'shakeout_reversal_a': get_param_value(p_triggers.get('shakeout_reversal_a_threshold'), 0.3),
            'consolidation_breakout_a': get_param_value(p_triggers.get('consolidation_breakout_a_threshold'), 0.3),
            'chip_ignition_s': get_param_value(p_triggers.get('chip_ignition_s_threshold'), 0.7),
            'invalidation_risk': get_param_value(p_triggers.get('invalidation_risk_threshold'), 0.5),
            'washout_reversal_a_plus': get_param_value(p_triggers.get('washout_reversal_a_plus_threshold'), 0.6),
            'extreme_squeeze_s_plus': get_param_value(p_triggers.get('extreme_squeeze_s_plus_threshold'), 0.7),
            'breakout_eve_s': get_param_value(p_triggers.get('breakout_eve_s_threshold'), 0.6),
            'normal_squeeze_a': get_param_value(p_triggers.get('normal_squeeze_a_threshold'), 0.5),
            'prime_chip_ignition_s_plus_plus': get_param_value(p_triggers.get('prime_chip_ignition_s_plus_plus_threshold'), 0.7),
            'chip_resonance_price_lag_s': get_param_value(p_triggers.get('chip_resonance_price_lag_s_threshold'), 0.5),
            'conviction_breakout_s_plus': get_param_value(p_triggers.get('conviction_breakout_s_plus_threshold'), 0.7),
            'true_accumulation_breakout_s_plus': get_param_value(p_triggers.get('true_accumulation_breakout_s_plus_threshold'), 0.6),
            'capitulation_reversal_a_plus': get_param_value(p_triggers.get('capitulation_reversal_a_plus_threshold'), 0.7),
            'post_reversal_resonance_s_plus': get_param_value(p_triggers.get('post_reversal_resonance_s_plus_threshold'), 0.65),
            'mean_reversion_grid_buy_a': get_param_value(p_triggers.get('mean_reversion_grid_buy_a_threshold'), 0.8),
            'panic_selling_setup_threshold': get_param_value(p_triggers.get('panic_selling_setup_threshold'), 0.4),
            'dominant_reversal_trigger_threshold': get_param_value(p_triggers.get('dominant_reversal_trigger_threshold'), 0.2),
        }
        p_dominant = p_triggers.get('dominant_reversal_candle', {})
        today_body_size = (df['close_D'] - df['open_D']).clip(lower=0)
        yesterday_body_size = (df['open_D'].shift(1) - df['close_D'].shift(1)).clip(lower=0)
        recovery_score = (today_body_size / yesterday_body_size.replace(0, np.nan)).clip(0, 2).fillna(0) / 2.0
        lookback = get_param_value(p_dominant.get('position_lookback_days'), 60)
        rolling_high = df['high_D'].rolling(lookback).max()
        rolling_low = df['low_D'].rolling(lookback).min()
        price_range = (rolling_high - rolling_low).replace(0, 1e-9)
        price_position = ((df['close_D'] - rolling_low) / price_range).clip(0, 1).fillna(0.5)
        position_score = 1.0 - price_position
        candle_quality_score = atomic.get('COGNITIVE_SCORE_EARLY_MOMENTUM_IGNITION_A', default_score)
        dominant_reversal_score = (recovery_score * position_score * candle_quality_score).astype(np.float32)
        triggers['TRIGGER_DOMINANT_REVERSAL'] = dominant_reversal_score
        
        panic_selling_setup_score = atomic.get('SCORE_SETUP_PANIC_SELLING_S', default_score)
        was_setup_yesterday = panic_selling_setup_score.shift(1).fillna(0.0) > thresholds['panic_selling_setup_threshold']
        is_triggered_today = triggers.get('TRIGGER_DOMINANT_REVERSAL', default_score) > thresholds['dominant_reversal_trigger_threshold']
        triggers['TRIGGER_V_REVERSAL_ACE_S_PLUS'] = was_setup_yesterday & is_triggered_today
        
        # [代码新增] 调用新的法医探针
        debug_params = get_params_block(self.strategy, 'debug_params')
        probe_date_str = get_param_value(debug_params.get('probe_date'))
        if probe_date_str and get_param_value(debug_params.get('enable_v_reversal_probe'), False):
            self._deploy_v_reversal_probe(
                df, atomic, probe_date_str, 
                setup_score=panic_selling_setup_score, 
                trigger_score=dominant_reversal_score,
                setup_threshold=thresholds['panic_selling_setup_threshold'],
                trigger_threshold=thresholds['dominant_reversal_trigger_threshold']
            )
            
        # ... 后续触发器定义逻辑保持不变 ...
        triggers['TRIGGER_TRUE_ACCUMULATION_BREAKOUT_S_PLUS'] = (atomic.get('SCORE_CHIP_TRUE_ACCUMULATION', default_score).shift(1).fillna(0.0) > thresholds['true_accumulation_breakout_s_plus']) & (atomic.get('COGNITIVE_SCORE_IGNITION_RESONANCE_S', default_score) > thresholds['ignition_s'])
        triggers['TRIGGER_CAPITULATION_REVERSAL_A_PLUS'] = (atomic.get('SCORE_CHIP_PLAYBOOK_CAPITULATION_REVERSAL', default_score) > thresholds['capitulation_reversal_a_plus']) & (triggers['TRIGGER_DOMINANT_REVERSAL'] > 0.5)
        triggers['TRIGGER_CONVICTION_BREAKOUT_S_PLUS'] = atomic.get('SCORE_FF_PLAYBOOK_CONVICTION_BREAKOUT', default_score) > thresholds['conviction_breakout_s_plus']
        triggers['TRIGGER_POST_REVERSAL_RESONANCE_S_PLUS'] = atomic.get('COGNITIVE_SCORE_OPP_POST_REVERSAL_RESONANCE_A_PLUS', default_score) > thresholds['post_reversal_resonance_s_plus']
        triggers['TRIGGER_MEAN_REVERSION_GRID_BUY_A'] = atomic.get('SCORE_PLAYBOOK_MEAN_REVERSION_GRID_BUY_A', default_score) > thresholds['mean_reversion_grid_buy_a']
        was_true_accumulation_yesterday_for_lag = atomic.get('SCORE_CHIP_TRUE_ACCUMULATION', default_score).shift(1).fillna(0.0) > 0.5
        price_momentum_suppressed_score = (1 - self.strategy.df_indicators['SLOPE_5_close_D'].rolling(120).rank(pct=True)).fillna(0.5)
        was_price_suppressed_yesterday = price_momentum_suppressed_score.shift(1).fillna(0.0) > 0.8
        was_setup_yesterday_for_lag = was_true_accumulation_yesterday_for_lag & was_price_suppressed_yesterday
        is_gentle_lift_today = atomic.get('COGNITIVE_SCORE_EARLY_MOMENTUM_IGNITION_A', default_score) > 0.3
        triggers['TRIGGER_CHIP_RESONANCE_PRICE_LAG_BREAKOUT_S'] = was_setup_yesterday_for_lag & is_gentle_lift_today
        triggers['TRIGGER_BOTTOM_REVERSAL_RESONANCE_S'] = atomic.get('COGNITIVE_SCORE_BOTTOM_REVERSAL_RESONANCE_S', default_score) > thresholds['bottom_reversal_s']
        triggers['TRIGGER_HEALTHY_PULLBACK_CONFIRMED_S'] = (atomic.get('COGNITIVE_SCORE_PULLBACK_HEALTHY_S', default_score).shift(1).fillna(0.0) > thresholds['healthy_pullback_s']) & (triggers['TRIGGER_DOMINANT_REVERSAL'] > 0.5)
        triggers['TRIGGER_CLASSIC_PATTERN_BREAKOUT_S'] = atomic.get('COGNITIVE_SCORE_CLASSIC_PATTERN_OPP_S', default_score) > thresholds['classic_pattern_s']
        triggers['TRIGGER_SHAKEOUT_REVERSAL_A'] = atomic.get('COGNITIVE_SCORE_OPP_SQUEEZE_SHAKEOUT_REVERSAL_A', default_score) > thresholds['shakeout_reversal_a']
        triggers['TRIGGER_CONSOLIDATION_BREAKOUT_A'] = atomic.get('COGNITIVE_SCORE_CONSOLIDATION_BREAKOUT_OPP_A', default_score) > thresholds['consolidation_breakout_a']
        triggers['TRIGGER_WASHOUT_ABSORPTION_REVERSAL_A_PLUS'] = atomic.get('COGNITIVE_SCORE_OPP_BOTTOM_REVERSAL', default_score) > thresholds['washout_reversal_a_plus']
        triggers['TRIGGER_CORE_HOLDER_IGNITION_S_PLUS'] = atomic.get('TACTIC_LOCK_CHIP_RECONCENTRATION_S_PLUS', default_series)
        is_in_main_uptrend = atomic.get('STRUCTURE_MAIN_UPTREND_WAVE_S', default_series)
        ignition_score = atomic.get('COGNITIVE_SCORE_IGNITION_RESONANCE_S', default_score)
        triggers['TRIGGER_UPTREND_IGNITION_RESONANCE_S'] = is_in_main_uptrend & (ignition_score > thresholds['ignition_s'])
        vol_compression_score = atomic.get('COGNITIVE_SCORE_VOL_COMPRESSION_FUSED', default_score)
        platform_quality_score = atomic.get('SCORE_PLATFORM_QUALITY_S', default_score)
        squeeze_breakout_score = atomic.get('COGNITIVE_SCORE_VOL_BREAKOUT_S', default_score)
        vol_breakout_a_score = atomic.get('COGNITIVE_SCORE_VOL_BREAKOUT_A', default_score)
        setup_extreme_squeeze = vol_compression_score > 0.9
        triggers['TRIGGER_EXTREME_SQUEEZE_EXPLOSION_S_PLUS'] = setup_extreme_squeeze.shift(1).fillna(False) & (squeeze_breakout_score > thresholds['extreme_squeeze_s_plus'])
        setup_breakout_eve = (platform_quality_score * vol_compression_score) > 0.7
        triggers['TRIGGER_BREAKOUT_EVE_S'] = setup_breakout_eve.shift(1).fillna(False) & (ignition_score > thresholds['breakout_eve_s'])
        setup_normal_squeeze = vol_compression_score > 0.5
        any_breakout_trigger = np.maximum(squeeze_breakout_score, vol_breakout_a_score) > thresholds['normal_squeeze_a']
        triggers['TRIGGER_NORMAL_SQUEEZE_BREAKOUT_A'] = setup_normal_squeeze.shift(1).fillna(False) & any_breakout_trigger & ~triggers['TRIGGER_EXTREME_SQUEEZE_EXPLOSION_S_PLUS']
        initial_ignition = atomic.get('COGNITIVE_SCORE_IGNITION_RESONANCE_S', default_score) > thresholds['ignition_s']
        invalidation_risk_score = np.maximum.reduce([
            atomic.get('COGNITIVE_SCORE_TOP_REVERSAL_RESONANCE_S', default_score).values,
            atomic.get('COGNITIVE_SCORE_BREAKDOWN_RESONANCE_S', default_score).values,
            atomic.get('COGNITIVE_SCORE_TREND_FATIGUE_RISK', default_score).values,
        ])
        invalidation_risk_series = pd.Series(invalidation_risk_score, index=df.index)
        risk_remained_low = invalidation_risk_series.rolling(window=3, min_periods=1).max() < thresholds['invalidation_risk']
        had_recent_ignition = initial_ignition.rolling(window=3, min_periods=1).max().astype(bool)
        is_sustained_day = had_recent_ignition & ~initial_ignition
        triggers['TRIGGER_SUSTAINED_IGNITION_S_PLUS'] = is_sustained_day & risk_remained_low
        setup_prime_chip = atomic.get('CHIP_SCORE_PRIME_OPPORTUNITY_S', default_score) > 0.7
        triggers['TRIGGER_PRIME_CHIP_IGNITION_S_PLUS_PLUS'] = setup_prime_chip & triggers['TRIGGER_SUSTAINED_IGNITION_S_PLUS']
        for key in list(triggers.keys()):
            if key in triggers and isinstance(triggers[key], pd.Series):
                 triggers[key] = triggers[key].fillna(False)
        return triggers

    def _normalize_score(self, series: pd.Series, window: int, ascending: bool = True, default: float = 0.5) -> pd.Series:
        """
        【V1.1 统一签名版】将一个 Series 归一化到 0-1 区间。
        - 核心升级 (本次修改):
          - [统一签名] 新增了 `default` 参数，使其与项目中其他 `_normalize_score` 方法的签名保持一致。
          - [健壮性] 优化了空 Series 的处理逻辑，确保在任何情况下都能返回一个带有正确索引和默认值的 Series。
        - 收益: 解决了因函数签名不一致导致的 TypeError，提升了代码的健壮性和可维护性。
        """
        # 检查输入是否有效
        if series is None or series.empty:
            # 如果输入为空，返回一个填充了指定默认值的Series
            # 优先使用 series 的索引，如果 series 为 None，则回退到主 df 的索引
            index = series.index if series is not None else self.strategy.df_indicators.index
            return pd.Series(default, index=index)
        
        # 使用滚动窗口计算百分位排名，min_periods保证在数据初期也能尽快产出分数
        min_periods = window // 4
        rank_pct = series.rolling(window, min_periods=min_periods).rank(pct=True)
        
        # 根据排序方向调整分数
        if ascending:
            score = rank_pct
        else:
            score = 1.0 - rank_pct
            
        # 用指定的默认值填充因窗口期不足而产生的NaN
        return score.fillna(default)

    def _fuse_multi_level_scores(self, df: pd.DataFrame, base_name: str, weights: Dict[str, float] = None) -> pd.Series:
        """
        【V1.0 新增】辅助函数：融合S+/S/A/B等多层置信度分数的辅助函数。
        - 从其他情报模块迁移而来，用于支持探针的内部计算。
        """
        if weights is None:
            weights = {'S_PLUS': 1.5, 'S': 1.0, 'A': 0.6, 'B': 0.3}
        total_score = pd.Series(0.0, index=df.index)
        total_weight = 0.0
        for level in ['S_PLUS', 'S', 'A', 'B']:
            if level not in weights: continue
            weight = weights[level]
            score_name = f"SCORE_{base_name}_{level}"
            if score_name in self.strategy.atomic_states:
                score_series = self.strategy.atomic_states[score_name]
                if len(score_series) > 0:
                    total_score += score_series.reindex(df.index).fillna(0.0) * weight
                    total_weight += weight
        if total_weight == 0:
            single_score_name = f"SCORE_{base_name}"
            if single_score_name in self.strategy.atomic_states:
                return self.strategy.atomic_states[single_score_name].reindex(df.index).fillna(0.5)
            return pd.Series(0.5, index=df.index)
        return (total_score / total_weight).clip(0, 1)













