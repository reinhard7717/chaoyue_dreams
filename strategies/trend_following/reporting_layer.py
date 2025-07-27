# 文件: strategies/trend_following/reporting_layer.py
# 报告层
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
        print(f"      -> [战报司令部 V306.0] 启动，正在执行“双轨记录制”归档...")
        db_records = []
        signal_days_df = result_df[result_df['signal_type'] != '中性'].copy()
        for trade_time, row in signal_days_df.iterrows():
            is_pure_warning = (row['signal_type'] != '买入信号') and (row['signal_type'] != '卖出信号') and (row.get('alert_level', 0) > 0)
            record = self._create_signal_record(
                stock_code=stock_code, trade_time=trade_time, timeframe=result_timeframe,
                strategy_name=get_param_value(self.strategy.strategy_info.get('name'), 'TrendFollow'),
                signal_type=row['signal_type'], entry_score=row.get('final_score', 0.0),
                risk_score=row.get('risk_score', 0.0), close_price=row.get('close_D'),
                entry_signal=row.get('signal_entry', False), is_risk_warning=is_pure_warning,
                exit_signal_code=row.get('exit_signal_code', 0), exit_severity_level=row.get('alert_level', 0)
            )
            record = self._fill_signal_details(record, row, score_details_df, risk_details_df)
            db_records.append(record)
        print(f"      -> [战报司令部 V306.0] “双轨记录制”归档完成，共生成 {len(db_records)} 条记录。")
        return db_records

    def _create_signal_record(self, **kwargs) -> Dict[str, Any]:
        trade_time_input = kwargs.get('trade_time')
        if trade_time_input is None: raise ValueError("创建信号记录时必须提供 'trade_time'")
        ts = pd.to_datetime(trade_time_input)
        standard_trade_time = ts.tz_localize('Asia/Shanghai').tz_convert('UTC').to_pydatetime() if ts.tzinfo is None else ts.tz_convert('UTC').to_pydatetime()
        record = {
            "stock_code": None, "trade_time": standard_trade_time, "timeframe": "N/A",
            "strategy_name": "UNKNOWN", "signal_type": "中性", "entry_score": 0.0,
            "risk_score": 0.0, "triggered_playbooks": "", "close_price": None,
            "entry_signal": False, "is_risk_warning": False, "exit_signal_code": 0,
            "exit_severity_level": 0
        }
        record.update(kwargs)
        record['close_price'] = sanitize_for_json(record.get('close_price'))
        record['entry_score'] = float(record['entry_score'])
        if 'risk_score' in record: record['risk_score'] = float(record['risk_score'])
        return record

    def _fill_signal_details(self, record: Dict, signal_row: pd.Series, score_details_df: pd.DataFrame, risk_details_df: pd.DataFrame) -> Dict:
        trade_time = record['trade_time']
        details_list = []
        if record['signal_type'] == '买入信号':
            if not score_details_df.empty and trade_time in score_details_df.index:
                score_details_today = score_details_df.loc[trade_time]
                activated_rules_en = score_details_today[score_details_today > 0].index.tolist()
                details_list = [self.signal_metadata.get(rule, rule) for rule in activated_rules_en]
        elif record['signal_type'] == '卖出信号':
            reason = signal_row.get('alert_reason')
            if reason: details_list.append(reason)
            if not risk_details_df.empty and trade_time in risk_details_df.index:
                risk_details_today = risk_details_df.loc[trade_time]
                activated_risks_en = risk_details_today[risk_details_today > 0].index.tolist()
                risk_details_cn = [self.signal_metadata.get(risk, risk) for risk in activated_risks_en]
                if risk_details_cn: details_list.append(f"风险构成: {', '.join(risk_details_cn)}")
        record['triggered_playbooks'] = ", ".join(details_list)
        return record
