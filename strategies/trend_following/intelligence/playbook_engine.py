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
        【V7.0 · 终极信号适配版】剧本蓝图知识库
        - 核心重构 (本次修改):
          - [全面适配] 全面更新了所有剧本的触发器引用，使其与 `define_trigger_events` V9.0 版本完全同步。
          - [新增剧本] 新增了基于周期信号的 `PLAYBOOK_CYCLICAL_BOTTOM_FISHING_A` 等新战术。
        - 收益: 剧本库与信号工厂完全对齐，确保所有战术意图都能被正确执行。
        """
        return [
            # S++级 王牌剧本 (Ace Playbooks)
            {
                'name': 'PLAYBOOK_V_REVERSAL_ACE_S_PLUS',
                'trigger': ['TRIGGER_V_REVERSAL_ACE_S_PLUS'],
                'comment': 'S++级(王牌) - [V型反转] 在前一日确认的“恐慌抛售”后，由当日强劲的“显性反转K线”确认的最高确定性反转剧本。'
            },
            {
                'name': 'PLAYBOOK_PRIME_STRUCTURE_BREAKOUT_S_PLUS_PLUS',
                'trigger': ['TACTIC_PRIME_STRUCTURE_BREAKOUT_S_PLUS_PLUS'],
                'comment': 'S++级(王牌) - [黄金结构突破] 在“黄金筹码+波动压缩+能量优势”的完美战备后，由点火共振确认的突破。'
            },
            {
                'name': 'PLAYBOOK_PERFECT_STORM_BOTTOM_S_PLUS',
                'trigger': ['COGNITIVE_OPP_PERFECT_STORM_BOTTOM_S_PLUS'],
                'comment': 'S++级(王牌) - [完美风暴·底] 行为层、结构层、基础层信号在底部同时共振。'
            },
            # S+级 核心增强剧本
            {
                'name': 'PLAYBOOK_CRUISE_PIT_REVERSAL_S_PLUS',
                'trigger': ['TACTIC_CRUISE_PIT_REVERSAL_S_TRIPLE_PLUS'],
                'comment': 'S+级 - [巡航深坑反转] 在主升浪巡航阶段，出现“打压式”回踩后，由反转信号确认的黄金坑买点。'
            },
            {
                'name': 'PLAYBOOK_CRUISE_PULLBACK_REVERSAL_S_PLUS',
                'trigger': ['TACTIC_CRUISE_PULLBACK_REVERSAL_S_PLUS'],
                'comment': 'S+级 - [巡航回踩反转] 在主升浪巡航阶段，出现“健康”回踩后，由反转信号确认的买点。'
            },
            {
                'name': 'PLAYBOOK_EXTREME_SQUEEZE_EXPLOSION_S_PLUS',
                'trigger': ['PLAYBOOK_EXTREME_SQUEEZE_EXPLOSION_S_PLUS'],
                'comment': 'S+级 - [极致压缩突破] 在波动率极致压缩后，由高强度的压缩突破信号确认。'
            },
            # S级 核心剧本
            {
                'name': 'PLAYBOOK_BREAKOUT_EVE_S',
                'trigger': ['PLAYBOOK_BREAKOUT_EVE_S'],
                'comment': 'S级 - [突破前夜] 在平台和波动率双重压缩的突破前夜，由最高级别的点火共振信号确认。'
            },
            {
                'name': 'PLAYBOOK_CHIP_PRICE_LAG_S',
                'trigger': ['PLAYBOOK_CHIP_PRICE_LAG_S'],
                'comment': 'S级 - [筹码价格滞后] 在筹码高度共振、价格被压制的战备状态后，由温和点火信号确认的黄金突破口。'
            },
            # A+级 优势战术剧本
            {
                'name': 'PLAYBOOK_ASCENT_PIT_REVERSAL_A_PLUS',
                'trigger': ['TACTIC_ASCENT_PIT_REVERSAL_A_PLUS'],
                'comment': 'A+级 - [初升浪深坑反转] 在初升浪阶段，出现“打压式”回踩后，由反转信号确认的买点。'
            },
            # A级 战术剧本
            {
                'name': 'PLAYBOOK_ASCENT_PULLBACK_REVERSAL_A',
                'trigger': ['TACTIC_ASCENT_PULLBACK_REVERSAL_A'],
                'comment': 'A级 - [初升浪回踩反转] 在初升浪阶段，出现“健康”回踩后，由反转信号确认的买点。'
            },
            {
                'name': 'PLAYBOOK_NORMAL_SQUEEZE_BREAKOUT_A',
                'trigger': ['PLAYBOOK_NORMAL_SQUEEZE_BREAKOUT_A'],
                'comment': 'A级 - [常规压缩突破] 在常规波动率压缩后，由任意压缩突破信号确认。'
            },
            {
                'name': 'PLAYBOOK_MEAN_REVERSION_GRID_A',
                'trigger': ['TRIGGER_MEAN_REVERSION_GRID_BUY_A'],
                'comment': 'A级 - [均值回归网格] 在震荡市中，于价格触及统计下轨时买入。'
            },
            {
                'name': 'PLAYBOOK_CYCLICAL_BOTTOM_FISHING_A',
                'trigger': ['TRIGGER_CYCLICAL_BOTTOM_FISHING_A'],
                'comment': 'A级(新增) - [周期底捞] 在FFT周期信号指示波谷，且出现反转K线时买入。'
            },
        ]

    def define_trigger_events(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V9.0 · 终极信号适配版】战术触发事件定义中心
        - 核心重构 (本次修改):
          - [全面适配] 全面审查并更新了所有触发器的定义，使其消费最新的终极原子信号。
          - [新增触发器] 新增了基于周期信号的 `TRIGGER_CYCLICAL_BOTTOM_FISHING_A` 等触发器。
        - 收益: 触发器工厂与信号体系完全同步，确保剧本引擎的决策基于最可靠的情报。
        """
        # print("      -> [战术触发事件定义中心 V9.0 · 终极信号适配版] 启动...") # 更新版本号
        triggers = {}
        atomic = self.strategy.atomic_states
        default_score = pd.Series(0.0, index=df.index, dtype=np.float32)
        default_series = pd.Series(False, index=df.index)
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
        
        candle_quality_score = atomic.get('COGNITIVE_SCORE_EARLY_MOMENTUM_IGNITION_A', default_score)
        dominant_reversal_score = (recovery_score * position_score * candle_quality_score).astype(np.float32)
        triggers['TRIGGER_DOMINANT_REVERSAL'] = dominant_reversal_score > get_param_value(p_dominant.get('trigger_threshold'), 0.4)

        # --- 2. 定义剧本触发器 ---
        # V型反转王牌剧本
        setup_score = self._get_atomic_score(df, 'SCORE_SETUP_PANIC_SELLING_S')
        was_setup_yesterday = setup_score.shift(1).fillna(0.0) > get_param_value(p_triggers.get('panic_selling_setup_threshold'), 0.4)
        triggers['TRIGGER_V_REVERSAL_ACE_S_PLUS'] = was_setup_yesterday & triggers['TRIGGER_DOMINANT_REVERSAL']

        # 均值回归剧本
        mean_reversion_score = self._get_atomic_score(df, 'SCORE_PLAYBOOK_MEAN_REVERSION_GRID_BUY_A')
        triggers['TRIGGER_MEAN_REVERSION_GRID_BUY_A'] = mean_reversion_score > get_param_value(p_triggers.get('mean_reversion_grid_buy_a_threshold'), 0.8)

        # 周期底捞剧本 (新增)
        p_cyclical = p_triggers.get('cyclical_bottom_fishing', {})
        is_in_trough = self._get_atomic_score(df, 'DOMINANT_CYCLE_PHASE').fillna(0) < get_param_value(p_cyclical.get('phase_threshold'), -0.8)
        is_cyclical_regime = self._get_atomic_score(df, 'SCORE_CYCLICAL_REGIME') > get_param_value(p_cyclical.get('cyclical_regime_threshold'), 0.3)
        triggers['TRIGGER_CYCLICAL_BOTTOM_FISHING_A'] = is_in_trough & is_cyclical_regime & triggers['TRIGGER_DOMINANT_REVERSAL']

        # --- 3. 填充其他剧本的布尔状态 (这些剧本的逻辑已在TacticEngine中计算) ---
        # 这些信号已经是布尔型，直接从 atomic_states 中获取
        playbook_signals = [
            'PLAYBOOK_EXTREME_SQUEEZE_EXPLOSION_S_PLUS',
            'PLAYBOOK_BREAKOUT_EVE_S',
            'PLAYBOOK_NORMAL_SQUEEZE_BREAKOUT_A',
            'PLAYBOOK_CHIP_PRICE_LAG_S',
            'TACTIC_PRIME_STRUCTURE_BREAKOUT_S_PLUS_PLUS',
            'TACTIC_CRUISE_PIT_REVERSAL_S_TRIPLE_PLUS',
            'TACTIC_CRUISE_PULLBACK_REVERSAL_S_PLUS',
            'TACTIC_ASCENT_PIT_REVERSAL_A_PLUS',
            'TACTIC_ASCENT_PULLBACK_REVERSAL_A',
        ]
        for signal_name in playbook_signals:
            triggers[signal_name] = self._get_atomic_score(df, signal_name, default=False).astype(bool)

        # --- 4. 填充旧的、兼容性的触发器 (如果需要) ---
        # 这里的逻辑可以根据需要保留或逐步废弃
        # 例如，旧的 'TRIGGER_UPTREND_IGNITION_RESONANCE_S'
        ignition_score = self._get_atomic_score(df, 'COGNITIVE_SCORE_IGNITION_RESONANCE_S')
        is_in_main_uptrend = get_unified_score(self.strategy.atomic_states, df.index, 'STRUCTURE_BULLISH_RESONANCE') > 0.5
        triggers['TRIGGER_UPTREND_IGNITION_RESONANCE_S'] = is_in_main_uptrend & (ignition_score > get_param_value(p_triggers.get('ignition_s_threshold'), 0.5))

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

