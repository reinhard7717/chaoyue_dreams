# 文件: strategies/trend_following/reporting_layer.py
# 报告层 (V400.0 - ORM重构版)
import pandas as pd
from asgiref.sync import sync_to_async
from typing import Dict, List, Any, Tuple
from stock_models.stock_analytics import TradingSignal, Playbook, SignalPlaybookDetail, StrategyDailyScore, StrategyScoreComponent


from .utils import get_params_block, get_param_value

class ReportingLayer:
    def __init__(self, strategy_instance):
        """
        【V401.0 异步安全版】
        - 核心重构: 采用延迟加载模式。__init__ 不再执行任何数据库操作。
                    缓存将在第一次需要时，通过一个专门的异步方法加载。
        """
        self.strategy = strategy_instance
        self.playbooks_cache = None
        print(f"    -> [报告层] 初始化完成 (V401.0 异步安全版)，战法定义将延迟加载。")

    async def _ensure_playbooks_cached(self):
        """
        【V401.0 新增】确保战法定义已被缓存的异步方法。
        - 核心逻辑: 如果缓存尚未加载，则以异步安全的方式从数据库加载。
        """
        # 如果缓存已加载，则直接返回，避免重复查询
        if self.playbooks_cache is not None:
            return
        print("    -> [报告层] 首次需要，正在异步加载战法定义缓存...")
        try:
            sync_db_call = lambda: {p.name: p for p in Playbook.objects.all()}
            # 异步执行它
            self.playbooks_cache = await sync_to_async(sync_db_call, thread_sensitive=True)()
            print(f"    -> [报告层] 已成功缓存 {len(self.playbooks_cache)} 个战法定义。")
        except Exception as e:
            # 如果加载失败，初始化为空字典以保证后续代码不会出错
            self.playbooks_cache = {}
            print(f"    -> [报告层] 警告：异步加载战法定义缓存失败。错误: {e}")

    async def prepare_db_records(self, stock_code: str, result_df: pd.DataFrame, score_details_df: pd.DataFrame, risk_details_df: pd.DataFrame, params: dict, result_timeframe: str) -> Tuple[List, List, List, List]:
        """
        【V506.1 诊断探针版】
        - 核心升级: 增加了详细的调试打印，用于追踪 score_components_to_create 列表的填充过程。
        """
        print(f"      -> [战报司令部 V506.1 - 探针模式] 启动，正在构建信号与全量分数对象...")
        await self._ensure_playbooks_cached()
        
        signals_to_create, signal_details_to_create = [], []
        daily_scores_to_create, score_components_to_create = [], []

        tf_params = get_params_block(self.strategy, 'trend_follow')
        strategy_info = tf_params.get('strategy_info', {})
        save_all_days = get_param_value(strategy_info.get('save_all_days'), False)
        strategy_name = get_param_value(strategy_info.get('name'), 'TrendFollow')
        
        scoring_params = tf_params.get('four_layer_scoring_params', {})
        score_type_map = scoring_params.get('score_type_map', {})

        # --- Part 1: 生成 TradingSignal (逻辑不变) ---
        signal_days_df = result_df[result_df['signal_type'].isin(['买入信号', '卖出信号', '风险预警'])].copy()
        print(f"        -> [事件驱动] 发现 {len(signal_days_df)} 个交易信号日。")
        
        for trade_time, row in signal_days_df.iterrows():
            signal_type_map_enum = {
                '买入信号': TradingSignal.SignalType.BUY,
                '卖出信号': TradingSignal.SignalType.SELL,
                '风险预警': TradingSignal.SignalType.WARN,
            }
            signal_obj = TradingSignal(
                stock_id=stock_code, trade_time=trade_time, timeframe=result_timeframe,
                strategy_name=strategy_name, signal_type=signal_type_map_enum.get(row['signal_type']),
                entry_score=row.get('entry_score', 0.0), risk_score=row.get('risk_score', 0.0),
                veto_votes=int(row.get('veto_votes', 0)), close_price=row.get('close_D', 0.0),
                health_change_summary=row.get('health_change_summary', {})
            )
            signals_to_create.append(signal_obj)

            # 检查索引是否存在，避免KeyError
            if trade_time in score_details_df.index and trade_time in risk_details_df.index:
                combined_details = pd.concat([
                    score_details_df.loc[trade_time][score_details_df.loc[trade_time] > 0],
                    risk_details_df.loc[trade_time][risk_details_df.loc[trade_time] > 0]
                ])
                for name, score in combined_details.items():
                    playbook_obj = self.playbooks_cache.get(name)
                    if playbook_obj:
                        signal_details_to_create.append(SignalPlaybookDetail(
                            signal=signal_obj, playbook=playbook_obj, contributed_score=score
                        ))

        # --- Part 2: 生成 StrategyDailyScore (增加诊断探针) ---
        if save_all_days:
            print(f"        -> [全量预计算] 已启用，将为 {len(result_df)} 天生成每日分数记录。")
            
            # 探针开关，只在第一次迭代时打印详细信息
            is_first_iteration = True
            
            for trade_time, row in result_df.iterrows():
                daily_score_obj = StrategyDailyScore(
                    stock_id=stock_code, trade_date=trade_time.date(), strategy_name=strategy_name,
                    offensive_score=int(row.get('entry_score', 0)), risk_score=int(row.get('risk_score', 0)),
                    final_score=row.get('final_score', 0.0), signal_type=row.get('signal_type', '无信号'),
                    score_details_json={}
                )
                
                all_details_for_json = {}
                
                # 检查索引是否存在，避免KeyError
                if trade_time not in score_details_df.index or trade_time not in risk_details_df.index:
                    if is_first_iteration:
                        print(f"        -> [探针警告] 日期 {trade_time.date()} 在 score_details_df 或 risk_details_df 中没有索引，跳过该日成分生成。")
                    continue # 跳过这一天

                combined_details = pd.concat([
                    score_details_df.loc[trade_time][score_details_df.loc[trade_time] > 0],
                    risk_details_df.loc[trade_time][risk_details_df.loc[trade_time] > 0]
                ])

                # 【探针核心】
                if is_first_iteration:
                    print(f"        -------------------- [诊断探针 @ {trade_time.date()}] --------------------")
                    print(f"        [探针] 当日总进攻分: {row.get('entry_score', 0)}, 总风险分: {row.get('risk_score', 0)}")
                    print(f"        [探针] 过滤掉0分后，合并的成分详情 (combined_details):")
                    if combined_details.empty:
                        print("        [探针] -> combined_details 为空！这是没有生成Component的原因。")
                    else:
                        print(combined_details)
                    print(f"        -----------------------------------------------------------------")

                # 只有当 combined_details 不为空时，才会进入这个循环
                for signal_name, score_value in combined_details.items():
                    clean_signal_name = signal_name.replace('trg_', '')
                    signal_meta = score_type_map.get(clean_signal_name, {})
                    cn_name = signal_meta.get('cn_name', clean_signal_name)
                    score_type = signal_meta.get('type', 'unknown')

                    score_components_to_create.append(StrategyScoreComponent(
                        daily_score=daily_score_obj, signal_name=clean_signal_name,
                        signal_cn_name=cn_name, score_type=score_type, score_value=int(score_value)
                    ))
                    
                    if score_type not in all_details_for_json:
                        all_details_for_json[score_type] = []
                    all_details_for_json[score_type].append({'name': cn_name, 'score': int(score_value)})
                
                daily_score_obj.score_details_json = all_details_for_json
                daily_scores_to_create.append(daily_score_obj)
                
                # 关闭探针，避免后续打印
                is_first_iteration = False

        print(f"      -> [战报司令部 V506.1] 构建完成。")
        print(f"         - 交易信号: {len(signals_to_create)} 条, 信号详情: {len(signal_details_to_create)} 条")
        print(f"         - 每日分数: {len(daily_scores_to_create)} 条, 分数成分: {len(score_components_to_create)} 条")
        
        return (signals_to_create, signal_details_to_create, daily_scores_to_create, score_components_to_create)






