# 文件: strategies/trend_following/reporting_layer.py
# 报告层 (V400.0 - ORM重构版)
import pandas as pd
from asgiref.sync import sync_to_async
from typing import Dict, List, Any, Tuple
from stock_models.stock_analytics import TradingSignal, Playbook, SignalPlaybookDetail


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

    async def prepare_db_records(self, stock_code: str, result_df: pd.DataFrame, score_details_df: pd.DataFrame, risk_details_df: pd.DataFrame, params: dict, result_timeframe: str) -> Tuple[List, List]:
        """
        【V505.1 适配修复版】
        - 核心修复: 修正了配置读取路径，从正确的 `strategy_params.trend_follow.strategy_info` 读取配置。
        - 核心升级: 根据配置文件的 "save_all_days" 开关，决定是只保存信号日，还是保存所有日期的记录。
        """
        print(f"      -> [战报司令部 V505.1] 启动，正在构建信号对象...")
        await self._ensure_playbooks_cached()
        signals_to_create = []
        details_to_create = []

        # 1. 从正确的路径读取配置
        # 首先获取 trend_follow 整个大块
        tf_params = get_params_block(self.strategy, 'trend_follow')
        # 然后从这个大块里获取 strategy_info
        strategy_info = tf_params.get('strategy_info', {})
        
        save_all_days = get_param_value(strategy_info.get('save_all_days'), False)
        strategy_name = get_param_value(strategy_info.get('name'), 'TrendFollow')

        # 2. 根据配置决定是否筛选数据
        if save_all_days:
            days_to_process_df = result_df.copy()
            print("        -> [模式] 已启用“每日记录”模式，将保存所有日期的分析结果。")
        else:
            days_to_process_df = result_df[result_df['signal_type'] != '无信号'].copy()
            print("        -> [模式] 已启用“事件驱动”模式，将只保存在有信号的日子。")

        if days_to_process_df.empty:
            print("        -> [信息] 没有需要记录的日期，任务完成。")
            return ([], [])

        for trade_time, row in days_to_process_df.iterrows():
            # 3. 创建主信号对象 (TradingSignal)
            signal_type_map = {
                '买入信号': TradingSignal.SignalType.BUY,
                '卖出信号': TradingSignal.SignalType.SELL,
                '风险预警': TradingSignal.SignalType.WARN,
                '无信号': TradingSignal.SignalType.HOLD,
            }
            
            signal_obj = TradingSignal(
                stock_id=stock_code,
                trade_time=trade_time,
                timeframe=result_timeframe,
                strategy_name=strategy_name, # 使用从配置中读取的名称
                signal_type=signal_type_map.get(row['signal_type'], TradingSignal.SignalType.HOLD),
                entry_score=row.get('entry_score', 0.0),
                risk_score=row.get('risk_score', 0.0),
                veto_votes=int(row.get('veto_votes', 0)),
                close_price=row.get('close_D', 0.0),
                health_change_summary=row.get('health_change_summary', {})
            )
            signals_to_create.append(signal_obj)

            # 4. 创建信号构成详情对象 (SignalPlaybookDetail)
            # (这部分逻辑保持不变)
            if not score_details_df.empty and trade_time in score_details_df.index:
                activated_offense = score_details_df.loc[trade_time]
                for name, score in activated_offense[activated_offense > 0].items():
                    playbook_obj = self.playbooks_cache.get(name)
                    if playbook_obj:
                        details_to_create.append(SignalPlaybookDetail(
                            signal=signal_obj,
                            playbook=playbook_obj,
                            contributed_score=score
                        ))
            
            all_risk_details = pd.Series(dtype=float)
            if not risk_details_df.empty and trade_time in risk_details_df.index:
                all_risk_details = all_risk_details.add(risk_details_df.get(trade_time, pd.Series(dtype=float)), fill_value=0)
            
            for name, score in all_risk_details[all_risk_details > 0].items():
                playbook_obj = self.playbooks_cache.get(name)
                if playbook_obj:
                     details_to_create.append(SignalPlaybookDetail(
                        signal=signal_obj,
                        playbook=playbook_obj,
                        contributed_score=score
                    ))

        print(f"      -> [战报司令部 V505.1] 构建完成，共生成 {len(signals_to_create)} 条主信号和 {len(details_to_create)} 条详情记录。")
        
        return (signals_to_create, details_to_create)








