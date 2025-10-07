# 文件: strategies/trend_following/judgment_layer.py
# 统合判断层
import pandas as pd
import numpy as np
from typing import Tuple
from .utils import get_params_block, get_param_value

class JudgmentLayer:
    def __init__(self, strategy_instance):
        self.strategy = strategy_instance

    def make_final_decisions(self, score_details_df: pd.DataFrame, risk_details_df: pd.DataFrame):
        """
        【V533.1 · 奥德修斯之眼协议版】
        - 核心升级: 在所有关键计算节点部署“观察哨”(print)，实时追踪分数和状态的流转。
        - 收益: 提供一份详细的、逐帧的行动报告，以彻底揭示任何潜在的分数篡改行为。
        """
        print("    --- [最高作战指挥部 V533.1 · 奥德修斯之眼协议版] 启动...")
        df = self.strategy.df_indicators
        # [代码新增] 增加一个调试开关，只对探针日期打印详细信息
        debug_params = get_params_block(self.strategy, 'debug_params', {})
        probe_dates_str = debug_params.get('probe_dates', [])
        probe_dates = [pd.to_datetime(d).date() for d in probe_dates_str]
        def debug_print(date, message):
            if date.date() in probe_dates:
                print(f"      -> [观察哨 @ {date.date()}] {message}")
        # 步骤 1: 计算基础最终得分
        debug_print(df.index[-1], f"进入审判庭。初始 entry_score: {df['entry_score'].iloc[-1]:.0f}")
        chimera_conflict_score = self.strategy.atomic_states.get('COGNITIVE_SCORE_CHIMERA_CONFLICT', pd.Series(0.0, index=df.index))
        dominant_signal_type = self._get_dominant_offense_type(score_details_df)
        is_reversal_day = (dominant_signal_type == 'positional')
        dynamic_chimera_score = chimera_conflict_score.where(~is_reversal_day, chimera_conflict_score * 0.5)
        confidence_damper = 1.0 - dynamic_chimera_score
        df['final_score'] = (df['entry_score'] * confidence_damper)
        for idx, row in df.iterrows():
            if idx.date() in probe_dates:
                debug_print(idx, f"奇美拉衰减计算: entry_score({row['entry_score']:.0f}) * damper({confidence_damper.get(idx, 1.0):.2f}) -> pre_final_score: {row['final_score']:.0f}")
        # 步骤 2: 进行权威的风险等级裁决
        df['alert_level'], df['alert_reason'], fused_risks_df = self._adjudicate_risk_level()
        for idx, row in df.iterrows():
            if idx.date() in probe_dates and row['alert_level'] > 0:
                debug_print(idx, f"风险裁决完成。ALERT_LEVEL: {row['alert_level']}, REASON: {row['alert_reason']}")
        df['dynamic_action'] = self._get_dynamic_combat_action()
        df['risk_score'] = self.strategy.atomic_states.get('COGNITIVE_FUSED_RISK_SCORE', pd.Series(0.0, index=df.index)).fillna(0.0)
        # 步骤 3: 根据得分和风险，生成唯一的最终信号类型
        p_judge_common = get_params_block(self.strategy, 'four_layer_scoring_params').get('judgment_params', {})
        final_score_threshold = get_param_value(p_judge_common.get('final_score_threshold'), 400)
        df['signal_type'] = '无信号'
        is_score_sufficient = df['final_score'] > final_score_threshold
        is_veto_by_alert = df['alert_level'] >= 3
        potential_buy_condition = is_score_sufficient & ~is_veto_by_alert
        df.loc[potential_buy_condition, 'signal_type'] = '买入信号'
        # 步骤 4: 根据最终信号类型，反向修正分数和信号
        alert_veto_condition = is_score_sufficient & is_veto_by_alert
        df.loc[alert_veto_condition, 'signal_type'] = '风险否决'
        for idx, row in df[alert_veto_condition].iterrows():
            if idx.date() in probe_dates:
                debug_print(idx, f"风险否决触发！final_score 从 {row['final_score']:.0f} 被强制清零。")
        df.loc[alert_veto_condition, 'final_score'] = 0
        exit_triggers_df = self.strategy.exit_triggers
        strategic_exit_mask = exit_triggers_df.get('EXIT_STRATEGY_INVALIDATED', pd.Series(False, index=df.index))
        tactical_exit_mask = exit_triggers_df.get('EXIT_TREND_BROKEN', pd.Series(False, index=df.index)) & ~strategic_exit_mask
        df.loc[strategic_exit_mask & ~potential_buy_condition, 'signal_type'] = '战略失效离场'
        df.loc[tactical_exit_mask & ~potential_buy_condition, 'signal_type'] = '趋势破位离场'
        for idx, row in df.iterrows():
            if idx.date() in probe_dates:
                debug_print(idx, f"最终信号裁决完成。Signal_Type: '{row['signal_type']}', Final_Score: {row['final_score']:.0f}")
        # 步骤 5: 生成人类可读的摘要报告
        debug_print(df.index[-1], "进入报告生成模块 (_get_human_readable_summary)...")
        df['signal_details_cn'] = self._get_human_readable_summary(score_details_df, risk_details_df, df['signal_type'])
        # 步骤 6: 终结信号
        self._finalize_signals()
        debug_print(df.index[-1], "审判流程结束。")

    def _get_human_readable_summary(self, score_details_df: pd.DataFrame, risk_details_df: pd.DataFrame, signal_type_series: pd.Series) -> pd.Series:
        """
        【V3.9.1 · 奥德修斯之眼协议版】
        - 核心升级: 增加关键节点打印，追踪报告生成逻辑。
        """
        # [代码新增] 增加调试打印
        debug_params = get_params_block(self.strategy, 'debug_params', {})
        probe_dates_str = debug_params.get('probe_dates', [])
        probe_dates = [pd.to_datetime(d).date() for d in probe_dates_str]
        def debug_print(date, message):
            if date.date() in probe_dates:
                print(f"      -> [观察哨 @ {date.date()}] (报告生成) {message}")
        score_map = get_params_block(self.strategy, 'score_type_map', {})
        def process_details_df(details_df, is_risk_df=False):
            if details_df is None or details_df.empty: return pd.Series(dtype=object)
            active_cols = details_df.columns[(details_df.fillna(0) != 0).any()]
            if active_cols.empty: return pd.Series(dtype=object)
            long_df = details_df[active_cols].melt(ignore_index=False, var_name='signal', value_name='score').reset_index()
            long_df = long_df[long_df['score'].fillna(0) != 0].copy()
            if long_df.empty: return pd.Series(dtype=object)
            date_col_name = long_df.columns[0]
            cn_name_map = {k: v.get('cn_name', k) for k, v in score_map.items() if isinstance(v, dict)}
            long_df['cn_name'] = long_df['signal'].map(cn_name_map).fillna(long_df['signal'])
            long_df['contribution'] = long_df['score'].fillna(0).astype(int)
            long_df = long_df[long_df['contribution'] != 0]
            long_df['summary_dict'] = long_df.apply(lambda row: {'name': row['cn_name'], 'score': row['contribution']}, axis=1)
            return long_df.groupby(date_col_name)['summary_dict'].apply(list)
        all_summaries = process_details_df(score_details_df, is_risk_df=False)
        summary_df = pd.DataFrame({'details': all_summaries}).reindex(self.strategy.df_indicators.index)
        def generate_final_summary(row):
            final_signal_type = signal_type_series.get(row.name)
            # [代码新增] 增加调试打印
            debug_print(row.name, f"正在为最终信号 '{final_signal_type}' 生成报告详情...")
            if final_signal_type == '买入信号':
                details_list = row['details'] if isinstance(row['details'], list) else []
                offense_list = [d for d in details_list if d.get('score', 0) > 0]
                risk_list = [d for d in details_list if d.get('score', 0) < 0]
                # [代码新增] 增加调试打印
                debug_print(row.name, f"裁决为'买入信号'，保留 {len(offense_list)} 个进攻项和 {len(risk_list)} 个风险项。")
                return {'offense': offense_list, 'risk': risk_list}
            else:
                # [代码新增] 增加调试打印
                debug_print(row.name, f"裁决非'买入信号'，清空所有进攻项和风险项。")
                return {'offense': [], 'risk': []}
        return summary_df.apply(generate_final_summary, axis=1)

    def _get_dynamic_combat_action(self) -> pd.Series:
        """
        【V319.0 · 终极信号适配版】动态力学战术矩阵
        - 核心重构: 全面消费由 DynamicMechanicsEngine 生成的终极信号。
        """
        df = self.strategy.df_indicators
        atomic = self.strategy.atomic_states
        default_score = pd.Series(0.0, index=df.index, dtype=np.float32)
        
        # 注意：这里的信号名可能需要根据你的 intelligence layer 的最终输出进行调整
        offensive_resonance_score = atomic.get('SCORE_DYN_BULLISH_RESONANCE', default_score)
        risk_expansion_score = atomic.get('SCORE_DYN_BEARISH_RESONANCE', default_score)
        
        is_force_attack = offensive_resonance_score > 0.6
        is_avoid = risk_expansion_score > 0.6
        is_caution = (offensive_resonance_score > 0.4) & (risk_expansion_score > 0.4)
        
        actions = pd.Series('HOLD', index=df.index)
        actions.loc[is_caution] = 'PROCEED_WITH_CAUTION'
        actions.loc[is_force_attack] = 'FORCE_ATTACK'
        actions.loc[is_avoid] = 'AVOID'
        return actions

    def _finalize_signals(self):
        """
        【V522.0 · 统一号令版】
        - 核心革命: 重建指挥链。signal_entry 成为所有入场信号的唯一官方旗帜。
        - 核心逻辑: 无论是“买入信号”还是“先知入场”，都会将 signal_entry 设置为 True。
        - 收益: 为下游的 simulation_layer 提供了单一、明确的建仓指令。
        """
        df = self.strategy.df_indicators
        df['signal_entry'] = False
        df['exit_signal_code'] = 0
        
        # 统一号令：只为“买入信号”升起'signal_entry'旗帜。
        final_buy_condition = (df['signal_type'] == '买入信号')
        
        df.loc[final_buy_condition, 'signal_entry'] = True
        df.loc[final_buy_condition, 'exit_signal_code'] = 0

    def _adjudicate_risk_level(self) -> Tuple[pd.Series, pd.Series, pd.DataFrame]:
        """
        【V2.8 · 数据纯净法案版】风险裁决者 (Risk Adjudicator)
        - 核心修复: 不再将字符串类型的 alert_reason 存入 atomic_states，确保其只包含数值状态。
        - 核心逻辑: alert_reason 作为最终描述性结果，直接返回给 make_final_decisions 方法处理。
        - 收益: 根除了因数据类型污染导致的下游模块（如探针）崩溃问题。
        """
        df = self.strategy.df_indicators
        atomic = self.strategy.atomic_states
        risk_categories = {
            'ARCHANGEL_RISK': ['SCORE_ARCHANGEL_TOP_REVERSAL'],
            'TOP_REVERSAL': ['SCORE_BEHAVIOR_TOP_REVERSAL', 'SCORE_CHIP_TOP_REVERSAL', 'SCORE_FF_TOP_REVERSAL', 'SCORE_STRUCTURE_TOP_REVERSAL', 'SCORE_DYN_TOP_REVERSAL', 'SCORE_FOUNDATION_TOP_REVERSAL'],
            'BEARISH_RESONANCE': ['SCORE_BEHAVIOR_BEARISH_RESONANCE', 'SCORE_CHIP_BEARISH_RESONANCE', 'SCORE_FF_BEARISH_RESONANCE', 'SCORE_STRUCTURE_BEARISH_RESONANCE', 'SCORE_DYN_BEARISH_RESONANCE', 'SCORE_FOUNDATION_BEARISH_RESONANCE'],
            'MICRO_RISK': ['COGNITIVE_SCORE_RISK_POWER_SHIFT_TO_RETAIL', 'COGNITIVE_SCORE_RISK_MAIN_FORCE_CONVICTION_WEAKENING'],
            'EUPHORIA_RISK': ['COGNITIVE_SCORE_RISK_EUPHORIA_ACCELERATION']
        }
        fused_risks = {}
        for category, signals in risk_categories.items():
            signal_scores = [atomic.get(s, pd.Series(0.0, index=df.index)).reindex(df.index).fillna(0.0) for s in signals]
            fused_risks[category] = np.maximum.reduce(signal_scores) if signal_scores else pd.Series(0.0, index=df.index)
        fused_risks_df = pd.DataFrame(fused_risks, index=df.index)
        p_judge = get_params_block(self.strategy, 'judgment_day_params', {})
        predictive_exhaustion_risk = atomic.get('PREDICTIVE_RISK_CLIMACTIC_RUN_EXHAUSTION', pd.Series(0, index=df.index))
        prophet_threshold = get_param_value(p_judge.get('prophet_alert_threshold'), 0.7)
        archangel_threshold = get_param_value(p_judge.get('archangel_alert_threshold'), 0.7)
        top_reversal_threshold = get_param_value(p_judge.get('top_reversal_alert_threshold'), 0.8)
        resonance_threshold = get_param_value(p_judge.get('bearish_resonance_alert_threshold'), 0.7)
        euphoria_threshold = get_param_value(p_judge.get('euphoria_alert_threshold'), 0.75)
        micro_risk_threshold = get_param_value(p_judge.get('micro_risk_alert_threshold'), 0.6)
        is_uptrend_context = df.get('close_D', 0) > df.get('EMA_5_D', 0)
        conditions = [
            (predictive_exhaustion_risk > prophet_threshold) & is_uptrend_context,
            fused_risks_df['ARCHANGEL_RISK'] > archangel_threshold,
            fused_risks_df['TOP_REVERSAL'] > top_reversal_threshold,
            (fused_risks_df['BEARISH_RESONANCE'] > resonance_threshold) | (fused_risks_df['EUPHORIA_RISK'] > euphoria_threshold),
            fused_risks_df['MICRO_RISK'] > micro_risk_threshold,
        ]
        choices_level = [3, 3, 3, 2, 1]
        choices_reason = [
            '红色警报: 先知-高潮衰竭',
            '红色警报: 天使长-明确顶部形态',
            '红色警报: 顶部反转风险',
            '橙色警报: 共振或亢奋风险',
            '黄色警报: 微观结构风险'
        ]
        alert_level = pd.Series(np.select(conditions, choices_level, default=0), index=df.index)
        alert_reason = pd.Series(np.select(conditions, choices_reason, default=''), index=df.index)
        # [代码修改] 只将数值型的 alert_level 存入 atomic_states
        self.strategy.atomic_states['ALERT_LEVEL'] = alert_level.astype(np.int8)
        # [代码删除] 不再将字符串类型的 alert_reason 存入 atomic_states
        # self.strategy.atomic_states['ALERT_REASON'] = alert_reason
        return alert_level, alert_reason, fused_risks_df

    def _get_dominant_offense_type(self, score_details_df: pd.DataFrame) -> pd.Series:
        """
        【V1.0 · 新增】识别每日最强的进攻信号及其类型 ('positional' 或 'dynamic')。
        """
        if score_details_df is None or score_details_df.empty:
            return pd.Series('unknown', index=self.strategy.df_indicators.index)

        # 获取信号字典
        score_map = get_params_block(self.strategy, 'score_type_map', {})
        
        # 找出得分最高的信号列名
        dominant_signal_names = score_details_df.idxmax(axis=1)
        
        # 创建一个从信号名到类型的映射
        signal_to_type_map = {
            name: meta.get('type', 'unknown') 
            for name, meta in score_map.items() 
            if isinstance(meta, dict)
        }
        
        # 将最强信号名映射到其类型
        dominant_types = dominant_signal_names.map(signal_to_type_map).fillna('unknown')
        
        return dominant_types.reindex(self.strategy.df_indicators.index).fillna('unknown')
















