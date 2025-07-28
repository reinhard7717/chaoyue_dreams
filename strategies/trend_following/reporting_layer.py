# 文件: strategies/trend_following/reporting_layer.py
# 报告层 (V313.0 - 名称映射版)
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

        # ▼▼▼【核心改造 V313.0】: 定义列名映射 ▼▼▼
        # 解决了 DataFrame 中的列名 (如 'close_D') 与数据库字段名 (如 'close_price') 不一致的问题。
        self.COLUMN_MAP = {
            'close_D': 'close_price',
            'signal_entry': 'entry_signal',
            # 在这里可以添加更多映射，例如:
            # 'volume_D': 'volume',
            # 'pct_change_D': 'pct_change'
        }
        # ▲▲▲【核心改造 V313.0】结束 ▲▲▲

    def prepare_db_records(self, stock_code: str, result_df: pd.DataFrame, score_details_df: pd.DataFrame, risk_details_df: pd.DataFrame, params: dict, result_timeframe: str) -> List[Dict[str, Any]]:
        """
        【V313.0 名称映射版】
        - 核心升级: 引入列名映射(COLUMN_MAP)，确保DataFrame中的数据能准确填充到数据库对应字段。
        """
        print(f"      -> [战报司令部 V313.0 名称映射版] 启动，正在执行归档...")
        db_records = []
        signal_days_df = result_df[result_df['signal_type'] != '中性'].copy()

        for trade_time, row in signal_days_df.iterrows():
            # 1. 将当天的所有数据（一个 Series）转换为一个基础字典
            record_data = row.to_dict()

            # 2. 【新增】应用列名映射
            for df_col, db_col in self.COLUMN_MAP.items():
                if df_col in record_data:
                    record_data[db_col] = record_data[df_col]

            # 3. 更新或添加非 DataFrame 中的元数据
            record_data.update({
                'stock_code': stock_code,
                'trade_time': trade_time,
                'timeframe': result_timeframe,
                'strategy_name': get_param_value(self.strategy.strategy_info.get('name'), 'TrendFollow'),
                'entry_score': row.get('final_score', 0.0),
                'is_risk_warning': (row['signal_type'] != '买入信号') and (row['signal_type'] != '卖出信号') and (row.get('alert_level', 0) > 0)
            })

            # 4. 创建并填充记录
            record = self._create_signal_record(**record_data)
            record = self._fill_signal_details(record, row, score_details_df, risk_details_df)
            
            db_records.append(record)
            
        print(f"      -> [战报司令部 V313.0] 归档完成，共生成 {len(db_records)} 条全息记录。")
        return db_records

    def _create_signal_record(self, **kwargs) -> Dict[str, Any]:
        """
        创建一个基础记录字典，并用传入的所有参数进行更新。
        """
        trade_time_input = kwargs.get('trade_time')
        if trade_time_input is None: raise ValueError("创建信号记录时必须提供 'trade_time'")
        ts = pd.to_datetime(trade_time_input)
        standard_trade_time = ts.tz_localize('Asia/Shanghai').tz_convert('UTC').to_pydatetime() if ts.tzinfo is None else ts.tz_convert('UTC').to_pydatetime()
        
        record = {
            "stock_code": None, "trade_time": standard_trade_time, "timeframe": "N/A",
            "strategy_name": "UNKNOWN", "close_price": None, "entry_signal": False,
            "exit_signal_code": 0, "entry_score": 0.0, "triggered_playbooks": "",
            "context_snapshot": {}, "is_breakout_trigger": False, "is_continuation_entry": False,
            "is_pullback_entry": False, "rejection_code": 0, "washout_score": 0,
            "pullback_target_price": None, "is_long_term_bullish": False, "is_mid_term_bullish": False,
            "is_pullback_setup": False, "exit_severity_level": 0, "exit_signal_reason": "",
            "stable_platform_price": None, "is_risk_warning": False, "risk_score": 0.0
        }
        
        valid_kwargs = {k: v for k, v in kwargs.items() if k in record}
        record.update(valid_kwargs)

        # 类型净化
        for key in ['close_price', 'pullback_target_price', 'stable_platform_price']:
            if key in record:
                record[key] = sanitize_for_json(record.get(key))
        for key in ['entry_score', 'risk_score', 'washout_score']:
             if key in record:
                record[key] = float(sanitize_for_json(record.get(key, 0.0)))

        # 确保关键非空字段有值
        if record.get('close_price') is None:
            # 如果映射后仍然为空，可以考虑抛出异常或记录日志，这里我们先保持数据库报错
            pass

        return record

    def _fill_signal_details(self, record: Dict, signal_row: pd.Series, score_details_df: pd.DataFrame, risk_details_df: pd.DataFrame) -> Dict:
        """
        填充 triggered_playbooks 字段，描述信号的构成原因。
        """
        trade_time = signal_row.name 
        details_list = []

        exit_reason = record.get('exit_signal_reason') or signal_row.get('alert_reason')
        if exit_reason:
            details_list.append(str(exit_reason))

        if record.get('signal_type') == '买入信号':
            if not score_details_df.empty and trade_time in score_details_df.index:
                score_details_today = score_details_df.loc[trade_time]
                activated_rules_en = score_details_today[score_details_today > 0].index.tolist()
                details_list.extend([self.signal_metadata.get(rule, rule) for rule in activated_rules_en])
        elif record.get('signal_type') == '卖出信号':
            if not risk_details_df.empty and trade_time in risk_details_df.index:
                risk_details_today = risk_details_df.loc[trade_time]
                activated_risks_en = risk_details_today[risk_details_today > 0].index.tolist()
                risk_details_cn = [self.signal_metadata.get(risk, risk) for risk in activated_risks_en]
                if risk_details_cn:
                    details_list.append(f"风险构成: {', '.join(risk_details_cn)}")
        
        record['triggered_playbooks'] = ", ".join(filter(None, details_list))
        return record
