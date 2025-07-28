# 文件: strategies/trend_following/reporting_layer.py
# 报告层 (V312.0 - 全息记录版)
import pandas as pd
import numpy as np
from typing import Dict, List, Any
from utils.data_sanitizer import sanitize_for_json
from .utils import get_params_block, get_param_value

class ReportingLayer:
    def __init__(self, strategy_instance):
        self.strategy = strategy_instance
        self.scoring_params = get_params_block(self.strategy, 'four_layer_scoring_params')
        self.signal_metadata = self.scoring_params.get('metadata', {})
        playbook_blueprints = self.strategy.offensive_layer.playbook_blueprints
        playbook_cn_map = {p['name']: p.get('cn_name', p['name']) for p in playbook_blueprints}
        self.signal_metadata.update(playbook_cn_map)

    def prepare_db_records(self, stock_code: str, result_df: pd.DataFrame, score_details_df: pd.DataFrame, risk_details_df: pd.DataFrame, params: dict, result_timeframe: str) -> List[Dict[str, Any]]:
        """
        【V312.0 全息记录版】
        - 核心升级: 不再手动挑选字段，而是将信号日的整行数据(row)转换为字典，
                    实现所有计算结果的“全息”记录。这确保了所有在主流程中
                    计算的列（如 is_long_term_bullish, stable_platform_price 等）
                    都能被完整地存入数据库。
        """
        print(f"      -> [战报司令部 V312.0 全息记录版] 启动，正在执行归档...")
        db_records = []
        # 只选择有明确信号（非中性）的日子进行记录
        signal_days_df = result_df[result_df['signal_type'] != '中性'].copy()

        for trade_time, row in signal_days_df.iterrows():
            # --- 核心改造 ---
            # 1. 将当天的所有数据（一个 Series）转换为一个基础字典
            record_data = row.to_dict()

            # 2. 更新或添加非 DataFrame 中的元数据
            record_data.update({
                'stock_code': stock_code,
                'trade_time': trade_time,
                'timeframe': result_timeframe,
                'strategy_name': get_param_value(self.strategy.strategy_info.get('name'), 'TrendFollow'),
                'entry_score': row.get('final_score', 0.0), # 使用 final_score 作为最终的 entry_score
                'entry_signal': row.get('signal_entry', False),
                'is_risk_warning': (row['signal_type'] != '买入信号') and (row['signal_type'] != '卖出信号') and (row.get('alert_level', 0) > 0)
            })

            # 3. 创建并填充记录
            record = self._create_signal_record(**record_data)
            record = self._fill_signal_details(record, row, score_details_df, risk_details_df)
            
            db_records.append(record)
            
        print(f"      -> [战报司令部 V312.0] 归档完成，共生成 {len(db_records)} 条全息记录。")
        return db_records

    def _create_signal_record(self, **kwargs) -> Dict[str, Any]:
        """
        创建一个基础记录字典，并用传入的所有参数进行更新。
        """
        trade_time_input = kwargs.get('trade_time')
        if trade_time_input is None: raise ValueError("创建信号记录时必须提供 'trade_time'")
        ts = pd.to_datetime(trade_time_input)
        standard_trade_time = ts.tz_localize('Asia/Shanghai').tz_convert('UTC').to_pydatetime() if ts.tzinfo is None else ts.tz_convert('UTC').to_pydatetime()
        
        # 定义一个包含所有数据库列的模板，确保所有字段都存在
        record = {
            "stock_code": None, "trade_time": standard_trade_time, "timeframe": "N/A",
            "strategy_name": "UNKNOWN", "close_price": None, "entry_signal": False,
            "exit_signal_code": 0, "entry_score": 0.0, "triggered_playbooks": "",
            "context_snapshot": {}, "is_breakout_trigger": False, "is_continuation_entry": False,
            "is_pullback_entry": False, "rejection_code": 0, "washout_score": 0,
            "pullback_target_price": None, "is_long_term_bullish": False, "is_mid_term_bullish": False,
            "is_pullback_setup": False, "exit_severity_level": 0, "exit_signal_reason": "",
            "stable_platform_price": None, "is_risk_warning": False, "risk_score": 0.0
            # ...可以添加更多数据库字段的默认值...
        }
        
        # 用传入的 kwargs 更新默认值。kwargs 中有的字段会被覆盖，没有的则保留默认值。
        # 为了安全，只更新 record 中已定义的键
        valid_kwargs = {k: v for k, v in kwargs.items() if k in record}
        record.update(valid_kwargs)

        # 对特定字段进行类型净化
        for key in ['close_price', 'pullback_target_price', 'stable_platform_price']:
            if key in record:
                record[key] = sanitize_for_json(record.get(key))
        for key in ['entry_score', 'risk_score', 'washout_score']:
             if key in record:
                record[key] = float(sanitize_for_json(record.get(key, 0.0)))

        return record

    def _fill_signal_details(self, record: Dict, signal_row: pd.Series, score_details_df: pd.DataFrame, risk_details_df: pd.DataFrame) -> Dict:
        """
        填充 triggered_playbooks 字段，描述信号的构成原因。
        """
        # 注意：这里的 trade_time 需要用原始的，带时区的 pandas Timestamp
        trade_time = signal_row.name 
        details_list = []

        # 优先显示离场原因
        exit_reason = record.get('exit_signal_reason') or signal_row.get('alert_reason')
        if exit_reason:
            details_list.append(exit_reason)

        # 如果是买入信号，显示得分构成
        if record.get('signal_type') == '买入信号':
            if not score_details_df.empty and trade_time in score_details_df.index:
                score_details_today = score_details_df.loc[trade_time]
                activated_rules_en = score_details_today[score_details_today > 0].index.tolist()
                details_list.extend([self.signal_metadata.get(rule, rule) for rule in activated_rules_en])
        # 如果是卖出信号，显示风险构成
        elif record.get('signal_type') == '卖出信号':
            if not risk_details_df.empty and trade_time in risk_details_df.index:
                risk_details_today = risk_details_df.loc[trade_time]
                activated_risks_en = risk_details_today[risk_details_today > 0].index.tolist()
                risk_details_cn = [self.signal_metadata.get(risk, risk) for risk in activated_risks_en]
                if risk_details_cn:
                    details_list.append(f"风险构成: {', '.join(risk_details_cn)}")
        
        record['triggered_playbooks'] = ", ".join(filter(None, details_list))
        return record

