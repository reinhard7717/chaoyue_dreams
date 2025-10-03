# 文件: strategies/trend_following/intelligence/playbook_engine.py
# 剧本与触发器引擎
import pandas as pd
from typing import Dict, Tuple, List
import numpy as np
from strategies.trend_following.utils import get_params_block, get_param_value, get_unified_score, normalize_score

class PlaybookEngine:
    def __init__(self, strategy_instance):
        """
        初始化剧本与触发器引擎。
        :param strategy_instance: 策略主实例的引用。
        """
        self.strategy = strategy_instance
        self.playbook_blueprints = self._get_playbook_blueprints()
        self.kline_params = get_params_block(self.strategy, 'kline_pattern_params')

    def _get_atomic_score(self, df: pd.DataFrame, name: str, default=0.0) -> pd.Series:
        """安全地从原子状态库中获取分数。"""
        return self.strategy.atomic_states.get(name, pd.Series(default, index=df.index))

    def generate_playbook_states(self, trigger_events: Dict[str, pd.Series]) -> Tuple[Dict[str, pd.Series], Dict[str, pd.Series]]:
        """
        【V5.0 · 信号适配版】剧本状态生成引擎
        - 核心重构 (本次修改):
          - [信号适配] 更新了剧本蓝图，确保所有剧本都消费最新的、有效的触发器。
        - 收益: 剧本引擎与信号体系完全同步，能够正确执行所有战术。
        """
        df = self.strategy.df_indicators
        if df.empty:
            return {}, {}
        playbook_states = {}
        default_series = pd.Series(False, index=df.index)
        
        for blueprint in self.playbook_blueprints:
            playbook_name = blueprint['name']
            trigger_conditions = pd.Series(False, index=df.index)
            for trigger_name in blueprint.get('trigger', []):
                trigger_conditions |= trigger_events.get(trigger_name, default_series)
            playbook_states[playbook_name] = trigger_conditions
            
        return {}, playbook_states

    def _get_playbook_blueprints(self) -> List[Dict]:
        """
        【V7.1 · 信号净化版】剧本蓝图知识库
        - 核心修复: 全面净化了所有剧本的名称和其引用的触发器名称，移除了所有等级后缀。
        """
        return [
            # 全面净化剧本名称
            {'name': 'PLAYBOOK_V_REVERSAL_ACE', 'trigger': ['TRIGGER_V_REVERSAL_ACE']},
            {'name': 'PLAYBOOK_PRIME_STRUCTURE_BREAKOUT', 'trigger': ['TACTIC_PRIME_STRUCTURE_BREAKOUT']},
            {'name': 'PLAYBOOK_PERFECT_STORM_BOTTOM', 'trigger': ['COGNITIVE_OPP_PERFECT_STORM_BOTTOM']},
            {'name': 'PLAYBOOK_CRUISE_PIT_REVERSAL', 'trigger': ['TACTIC_CRUISE_PIT_REVERSAL']},
            {'name': 'PLAYBOOK_CRUISE_PULLBACK_REVERSAL', 'trigger': ['TACTIC_CRUISE_PULLBACK_REVERSAL']},
            {'name': 'PLAYBOOK_EXTREME_SQUEEZE_EXPLOSION', 'trigger': ['PLAYBOOK_EXTREME_SQUEEZE_EXPLOSION']},
            {'name': 'PLAYBOOK_BREAKOUT_EVE', 'trigger': ['PLAYBOOK_BREAKOUT_EVE']},
            {'name': 'PLAYBOOK_CHIP_PRICE_LAG', 'trigger': ['PLAYBOOK_CHIP_PRICE_LAG']},
            {'name': 'PLAYBOOK_CAPITULATION_REVERSAL', 'trigger': ['TRIGGER_CAPITULATION_REVERSAL']},
            {'name': 'PLAYBOOK_ASCENT_PIT_REVERSAL', 'trigger': ['TACTIC_ASCENT_PIT_REVERSAL']},
            {'name': 'PLAYBOOK_ASCENT_PULLBACK_REVERSAL', 'trigger': ['TACTIC_ASCENT_PULLBACK_REVERSAL']},
            {'name': 'PLAYBOOK_NORMAL_SQUEEZE_BREAKOUT', 'trigger': ['PLAYBOOK_NORMAL_SQUEEZE_BREAKOUT']},
            {'name': 'PLAYBOOK_MEAN_REVERSION_GRID', 'trigger': ['TRIGGER_MEAN_REVERSION_GRID_BUY']},
            {'name': 'PLAYBOOK_CYCLICAL_BOTTOM_FISHING', 'trigger': ['TRIGGER_CYCLICAL_BOTTOM_FISHING']},
        ]

    def define_trigger_events(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V9.1 · 信号净化版】战术触发事件定义中心
        - 核心修复: 全面净化了所有触发器的定义，使其消费和生产的都是净化后的信号名。
        """
        # print("      -> [战术触发事件定义中心 V9.1 · 信号净化版] 启动...") # 更新版本号
        triggers = {}
        atomic = self.strategy.atomic_states
        default_score = pd.Series(0.0, index=df.index, dtype=np.float32)
        p_triggers = get_params_block(self.strategy, 'trigger_event_params', {})
        
        # --- 1. 定义核心触发器：显性反转K线 ---
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
        
        # 消费净化后的信号名
        candle_quality_score = atomic.get('COGNITIVE_SCORE_EARLY_MOMENTUM_IGNITION', default_score)
        dominant_reversal_score = (recovery_score * position_score * candle_quality_score).astype(np.float32)
        triggers['TRIGGER_DOMINANT_REVERSAL'] = dominant_reversal_score > get_param_value(p_dominant.get('trigger_threshold'), 0.4)

        # --- 2. 定义剧本触发器 ---
        # V型反转王牌剧本
        # 消费净化后的信号名
        setup_score = self._get_atomic_score(df, 'SCORE_SETUP_PANIC_SELLING')
        was_setup_yesterday = setup_score.shift(1).fillna(0.0) > get_param_value(p_triggers.get('panic_selling_setup_threshold'), 0.4)
        # 生产净化后的信号名
        triggers['TRIGGER_V_REVERSAL_ACE'] = was_setup_yesterday & triggers['TRIGGER_DOMINANT_REVERSAL']

        # 均值回归剧本
        # 消费净化后的信号名
        mean_reversion_score = self._get_atomic_score(df, 'SCORE_PLAYBOOK_MEAN_REVERSION_GRID_BUY')
        # 生产净化后的信号名
        triggers['TRIGGER_MEAN_REVERSION_GRID_BUY'] = mean_reversion_score > get_param_value(p_triggers.get('mean_reversion_grid_buy_a_threshold'), 0.8)

        # 周期底捞剧本 (新增)
        p_cyclical = p_triggers.get('cyclical_bottom_fishing', {})
        is_in_trough = self._get_atomic_score(df, 'DOMINANT_CYCLE_PHASE').fillna(0) < get_param_value(p_cyclical.get('phase_threshold'), -0.8)
        is_cyclical_regime = self._get_atomic_score(df, 'SCORE_CYCLICAL_REGIME') > get_param_value(p_cyclical.get('cyclical_regime_threshold'), 0.3)
        # 生产净化后的信号名
        triggers['TRIGGER_CYCLICAL_BOTTOM_FISHING'] = is_in_trough & is_cyclical_regime & triggers['TRIGGER_DOMINANT_REVERSAL']
        
        # 定义新的“恐慌投降反转”剧本触发器
        p_capitulation = p_triggers.get('capitulation_reversal', {'trigger_threshold': 0.4})
        capitulation_reversal_score = self._get_atomic_score(df, 'SCORE_PLAYBOOK_CAPITULATION_REVERSAL')
        triggers['TRIGGER_CAPITULATION_REVERSAL'] = capitulation_reversal_score > get_param_value(p_capitulation.get('trigger_threshold'), 0.4)

        # --- 3. 填充其他剧本的布尔状态 (这些剧本的逻辑已在TacticEngine中计算) ---
        # 全面净化信号名
        playbook_signals = [
            'PLAYBOOK_EXTREME_SQUEEZE_EXPLOSION',
            'PLAYBOOK_BREAKOUT_EVE',
            'PLAYBOOK_NORMAL_SQUEEZE_BREAKOUT',
            'PLAYBOOK_CHIP_PRICE_LAG',
            'TACTIC_PRIME_STRUCTURE_BREAKOUT',
            'TACTIC_CRUISE_PIT_REVERSAL',
            'TACTIC_CRUISE_PULLBACK_REVERSAL',
            'TACTIC_ASCENT_PIT_REVERSAL',
            'TACTIC_ASCENT_PULLBACK_REVERSAL',
        ]
        for signal_name in playbook_signals:
            triggers[signal_name] = self._get_atomic_score(df, signal_name, default=False).astype(bool)

        # --- 4. 填充旧的、兼容性的触发器 (废弃) ---
        # 废弃所有旧的、带后缀的触发器，以保持系统纯净
        # ignition_score = self._get_atomic_score(df, 'COGNITIVE_SCORE_IGNITION_RESONANCE')
        # is_in_main_uptrend = get_unified_score(self.strategy.atomic_states, df.index, 'STRUCTURE_BULLISH_RESONANCE') > 0.5
        # triggers['TRIGGER_UPTREND_IGNITION_RESONANCE'] = is_in_main_uptrend & (ignition_score > get_param_value(p_triggers.get('ignition_s_threshold'), 0.5))

        # 确保所有触发器都是布尔类型
        for key in list(triggers.keys()):
            if key in triggers and isinstance(triggers[key], pd.Series):
                 triggers[key] = triggers[key].fillna(False).astype(bool)
                 
        return triggers

    def _deploy_v_reversal_probe(self, df: pd.DataFrame, atomic: Dict, probe_date: str, setup_score: pd.Series, trigger_score: pd.Series, setup_threshold: float, trigger_threshold: float):
        """
        【V1.0 新增】V型反转法医探针
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

            print(f"\n--- [第一部分: 解剖 {yesterday_ts.date()} 的战备分 (Setup)] ---")
            was_setup_yesterday_score = setup_score.shift(1).get(probe_ts, 0.0)
            print(f"  - 昨日战备分 (SCORE_SETUP_PANIC_SELLING_S): {was_setup_yesterday_score:.4f}")
            print(f"  - 战备阈值: {setup_threshold:.4f}")
            was_setup_ok = was_setup_yesterday_score > setup_threshold
            print(f"  - [判定] 昨日战备是否就绪? {'✅ 是' if was_setup_ok else '❌ 否'}")
            
            print("    -> 战备分由以下三者相乘得到:")
            price_drop = normalize_score(df['pct_change_D'].clip(upper=0), df.index, 60, False).get(yesterday_ts, 0.0)
            volume_spike = normalize_score(df['volume_D'] / df['VOL_MA_21_D'], df.index, 60, True).get(yesterday_ts, 0.0)
            chip_breakdown = get_unified_score(self.strategy.atomic_states, df.index, 'CHIP_BEARISH_RESONANCE').get(yesterday_ts, 0.0)
            print(f"      - 价格下跌分: {price_drop:.4f}")
            print(f"      - 成交放量分: {volume_spike:.4f}")
            print(f"      - 筹码崩溃分: {chip_breakdown:.4f}")
            
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

            print("\n--- [第三部分: 最终结论] ---")
            final_trigger_result = was_setup_ok and is_trigger_ok
            print(f"  - 触发器最终逻辑: (昨日战备就绪 AND 今日成功点火)")
            print(f"  - 计算结果: ({was_setup_ok} AND {is_trigger_ok}) = {final_trigger_result}")
            print(f"  - [结论] V型反转王牌剧本在 {probe_date} {'✅ 已触发' if final_trigger_result else '❌ 未触发'}")

        except Exception as e:
            print(f"  [探针错误] 在执行V型反转法医探针时发生异常: {e}")
        finally:
            print("="*95 + "\n")

