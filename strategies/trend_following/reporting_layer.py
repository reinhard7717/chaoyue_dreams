# 文件: strategies/trend_following/reporting_layer.py
# 报告层 (V314.0 - 健壮合并版)
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

        self.COLUMN_MAP = {
            'close_D': 'close_price',
            'signal_entry': 'entry_signal',
        }

    def prepare_db_records(self, stock_code: str, result_df: pd.DataFrame, score_details_df: pd.DataFrame, risk_details_df: pd.DataFrame, params: dict, result_timeframe: str) -> List[Dict[str, Any]]:
        """
        【V314.1 risk_score 修正版】
        - 核心修复: 在构建 record_data 时，明确地从 row 中提取 risk_score，
                    确保其能被正确传递给 _create_signal_record 并存入数据库。
        """
        print(f"      -> [战报司令部 V314.1 risk_score 修正版] 启动，正在执行归档...")
        db_records = []
        signal_days_df = result_df[result_df['signal_type'] != '中性'].copy()

        for trade_time, row in signal_days_df.iterrows():
            record_data = row.to_dict()

            for df_col, db_col in self.COLUMN_MAP.items():
                if df_col in record_data:
                    record_data[db_col] = record_data.pop(df_col)

            record_data.update({
                'stock_code': stock_code,
                'trade_time': trade_time,
                'timeframe': result_timeframe,
                'strategy_name': get_param_value(self.strategy.strategy_info.get('name'), 'TrendFollow'),
                'entry_score': row.get('final_score', 0.0),
                'risk_score': row.get('risk_score', 0.0),
                'holding_health_score': row.get('holding_health_score', 0.0),
                'is_risk_warning': (row['signal_type'] != '买入信号') and (row['signal_type'] != '卖出信号') and (row.get('alert_level', 0) > 0)
            })

            record = self._create_signal_record(**record_data)
            record = self._fill_signal_details(record, row, score_details_df, risk_details_df)
            
            db_records.append(record)
            
        print(f"      -> [战报司令部 V314.1] 归档完成，共生成 {len(db_records)} 条全息记录。")
        return db_records

    def _create_signal_record(self, **kwargs) -> Dict[str, Any]:
        """
        【V314.2 终极净化版】
        创建一个基础记录字典，并对所有数值和布尔字段进行严格的NaN净化。
        """
        trade_time_input = kwargs.get('trade_time')
        if trade_time_input is None: raise ValueError("创建信号记录时必须提供 'trade_time'")
        ts = pd.to_datetime(trade_time_input)
        standard_trade_time = ts.tz_localize('Asia/Shanghai').tz_convert('UTC').to_pydatetime() if ts.tzinfo is None else ts.tz_convert('UTC').to_pydatetime()
        
        db_template = {
            "stock_code": None, "trade_time": standard_trade_time, "timeframe": "N/A",
            "strategy_name": "UNKNOWN", "close_price": None, "entry_signal": False,
            "exit_signal_code": 0, "entry_score": 0.0, "triggered_playbooks": "",
            "context_snapshot": {}, "is_breakout_trigger": False, "is_continuation_entry": False,
            "is_pullback_entry": False, "rejection_code": 0, "washout_score": 0,
            "pullback_target_price": None, "is_long_term_bullish": False, "is_mid_term_bullish": False,
            "is_pullback_setup": False, "exit_severity_level": 0, "exit_signal_reason": "",
            "stable_platform_price": None, "is_risk_warning": False, "risk_score": 0.0,
            "signal_type": "中性",
            "holding_health_score": 0.0,
            "veto_votes": 0
        }
        
        final_record = db_template.copy()
        final_record.update(kwargs)

        record = {key: final_record.get(key) for key in db_template}

        # --- 数值字段净化 ---
        numeric_fields_with_defaults = {
            'close_price': None, 'pullback_target_price': None, 'stable_platform_price': None,
            'entry_score': 0.0, 'risk_score': 0.0, 'holding_health_score': 0.0,
            'washout_score': 0, 'exit_signal_code': 0, 'exit_severity_level': 0,
            'rejection_code': 0, 'veto_votes': 0
        }
        for field, default_value in numeric_fields_with_defaults.items():
            value = record.get(field)
            if pd.isna(value):
                record[field] = default_value
            else:
                if default_value is None: record[field] = float(value) if value is not None else None
                elif isinstance(default_value, float): record[field] = float(value)
                elif isinstance(default_value, int): record[field] = int(value)

        boolean_fields = [
            'entry_signal', 'is_breakout_trigger', 'is_continuation_entry',
            'is_pullback_entry', 'is_long_term_bullish', 'is_mid_term_bullish',
            'is_pullback_setup', 'is_risk_warning'
        ]
        for field in boolean_fields:
            value = record.get(field)
            # 如果值是 NaN, None, 或者其他非布尔值，都将其安全地转换为 False
            if pd.isna(value) or not isinstance(value, bool):
                record[field] = False

        return record

    def _fill_signal_details(self, record: Dict, signal_row: pd.Series, score_details_df: pd.DataFrame, risk_details_df: pd.DataFrame) -> Dict:
        """
        填充 triggered_playbooks 字段，描述信号的构成原因。
        """
        trade_time = signal_row.name 
        details_list = []

        exit_reason = record.get('exit_signal_reason') or signal_row.get('alert_reason')
        if exit_reason and not pd.isna(exit_reason):
            details_list.append(str(exit_reason))

        signal_type = record.get('signal_type')

        if signal_type == '买入信号':
            if not score_details_df.empty and trade_time in score_details_df.index:
                score_details_today = score_details_df.loc[trade_time]
                activated_rules_en = score_details_today[score_details_today > 0].index.tolist()
                details_list.extend([self.signal_metadata.get(rule, rule) for rule in activated_rules_en])
        elif signal_type == '卖出信号':
            if not risk_details_df.empty and trade_time in risk_details_df.index:
                risk_details_today = risk_details_df.loc[trade_time]
                activated_risks_en = risk_details_today[risk_details_today > 0].index.tolist()
                risk_details_cn = [self.signal_metadata.get(risk, risk) for risk in activated_risks_en]
                if risk_details_cn:
                    details_list.append(f"风险构成: {', '.join(risk_details_cn)}")

        record['triggered_playbooks'] = ", ".join(filter(None, details_list))
        return record
