# 文件: strategies/trend_following/intelligence/fund_flow_intelligence.py
# 资金流情报模块
import pandas as pd
import numpy as np
from typing import Dict
from strategies.trend_following.utils import get_params_block, get_param_value

class FundFlowIntelligence:
    def __init__(self, strategy_instance):
        """
        初始化资金流情报模块。
        :param strategy_instance: 策略主实例的引用。
        """
        self.strategy = strategy_instance

    def _calculate_normalized_score(self, series: pd.Series, window: int, ascending: bool = True) -> pd.Series:
        """
        【V12.0 性能优化版】计算一个系列在滚动窗口内的归一化得分 (0-1)。
        - 核心: 得分是基于该值在窗口期内的百分位排名。
        - 优化: 移除了if/else分支，通过直接传递参数使代码更简洁高效。
        :param series: 输入数据系列 (pd.Series)。
        :param window: 滚动窗口大小 (int)。
        :param ascending: True表示值越大得分越高，False反之 (bool)。
        :return: 归一化得分系列 (pd.Series, 范围0-1)。
        """
        return series.rolling(
            window=window, 
            min_periods=int(window * 0.2)
        ).rank(
            pct=True, 
            ascending=ascending
        ).fillna(0.5)

    def _diagnose_quantitative_fund_flow_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        【V10.0 性能优化版】资金流情报定量评分中心
        - 核心: 在完全不改变业务逻辑的前提下，对V9.0版本进行深度性能优化。
        - 策略:
          1.  预计算与复用: 提前计算所有信号都会用到的通用中间变量（如K线范围、上涨/下跌日判断），避免重复计算。
          2.  高效列检查: 使用集合(set)进行列存在性检查，将时间复杂度从O(N)降至O(1)。
          3.  减少内存开销: 通过链式表达式，减少不必要的中间Series对象创建。
          4.  代码结构化: 将逻辑分为“预检查”、“共识指标计算”、“通用变量计算”和“信号评分计算”四个阶段，提升可读性。
        """
        # print("            -> [资金流评分中心 V10.0 性能优化版] 启动...")

        # --- 阶段一: 预检查与准备 ---
        # 创建一个包含所有可用列的集合，用于后续高效的O(1)复杂度检查。
        available_cols = set(df.columns)

        # --- 阶段二: 创建三方数据源的“共识”复合指标 ---
        print("               - 正在创建三方数据源的'共识'复合指标...")
        # 共识指标1: 平均主力净流入 (Tushare的主力 + 东财的主力)
        main_force_cols = {'net_mf_amount_fund_flow_tushare_D', 'net_amount_fund_flow_dc_D'}
        if main_force_cols.issubset(available_cols):
            df['consensus_main_force_inflow'] = df[list(main_force_cols)].mean(axis=1)

        # 共识指标2: 平均全市场净流入 (Tushare主力 + 东财主力 + 同花顺全市场)
        overall_cols = {'net_mf_amount_fund_flow_tushare_D', 'net_amount_fund_flow_dc_D', 'net_amount_fund_flow_ths_D'}
        if overall_cols.issubset(available_cols):
            df['consensus_overall_inflow'] = df[list(overall_cols)].mean(axis=1)
            # 基于共识指标计算其5日斜率和加速度
            consensus_overall_inflow_series = df['consensus_overall_inflow']
            df['SLOPE_5_consensus_overall_inflow'] = consensus_overall_inflow_series.diff(5) / 5
            df['ACCEL_5_consensus_overall_inflow'] = df['SLOPE_5_consensus_overall_inflow'].diff(1)

        # 共识指标3: 平均主力买入占比 (东财超大单 + 同花顺大单)
        main_buy_rate_cols = {'buy_elg_amount_rate_fund_flow_dc_D', 'buy_lg_amount_rate_fund_flow_ths_D'}
        if main_buy_rate_cols.issubset(available_cols):
            df['consensus_main_buy_rate'] = df[list(main_buy_rate_cols)].mean(axis=1)
            df['SLOPE_5_consensus_main_buy_rate'] = df['consensus_main_buy_rate'].diff(5) / 5

        # 共识指标4: 平均散户买入占比 (同花顺中单 + 同花顺小单)
        retail_buy_rate_cols = {'buy_md_amount_rate_fund_flow_ths_D', 'buy_sm_amount_rate_fund_flow_ths_D'}
        if retail_buy_rate_cols.issubset(available_cols):
            df['consensus_retail_buy_rate'] = df[list(retail_buy_rate_cols)].sum(axis=1)
            df['SLOPE_5_consensus_retail_buy_rate'] = df['consensus_retail_buy_rate'].diff(5) / 5
        
        # 更新可用列集合，加入刚刚创建的共识指标列
        available_cols.update(df.columns)

        # --- 阶段三: 预计算通用中间变量，避免在各信号中重复计算 ---
        # 预计算K线相关指标
        if {'high_D', 'low_D', 'close_D'}.issubset(available_cols):
            # 计算K线实体范围，避免在信号4和11中重复计算
            k_line_range = df['high_D'] - df['low_D']
            # 计算上影线比例和下影线比例，并直接存为临时列
            df['_upper_shadow_ratio'] = np.divide(df['high_D'] - df['close_D'], k_line_range, out=np.zeros_like(df['high_D'], dtype=float), where=k_line_range!=0)
            df['_lower_shadow_ratio'] = np.divide(df['close_D'] - df['low_D'], k_line_range, out=np.zeros_like(df['close_D'], dtype=float), where=k_line_range!=0)
            available_cols.update(['_upper_shadow_ratio', '_lower_shadow_ratio'])
        pct_change = df.get('pct_change_D')
        # --- 阶段四: 基于“共识”指标和预计算变量，重构所有信号的计算逻辑 ---
        # print("               - 正在基于'共识'指标计算所有信号得分...")
        # --- 1. 【机会】资金底背离反转得分 ---
        # 逻辑: 股价创阶段性新低，但“共识”资金流出动能减弱或转为流入。
        if {'close_D', 'SLOPE_5_consensus_overall_inflow'}.issubset(available_cols):
            price_new_low_score = 1 - self._calculate_normalized_score(df['close_D'], 20, ascending=True)
            consensus_flow_improvement_score = self._calculate_normalized_score(df['SLOPE_5_consensus_overall_inflow'], 60)
            df['FF_SCORE_OPP_BULLISH_DIVERGENCE'] = price_new_low_score * consensus_flow_improvement_score

        # --- 2. 【风险】主力派发、散户接盘得分 ---
        # 逻辑: “共识”主力资金买入意愿下降，而“共识”散户资金买入意愿上升。
        if {'SLOPE_5_consensus_main_buy_rate', 'SLOPE_5_consensus_retail_buy_rate'}.issubset(available_cols):
            main_force_selling_score = self._calculate_normalized_score(df['SLOPE_5_consensus_main_buy_rate'], 60, ascending=False)
            retail_fomo_score = self._calculate_normalized_score(df['SLOPE_5_consensus_retail_buy_rate'], 60, ascending=True)
            df['FF_SCORE_RISK_DISTRIBUTION_STRUCTURE'] = main_force_selling_score * retail_fomo_score

        # --- 3. 【机会/触发器】资金点火得分 ---
        # 逻辑: “共识”资金从沉寂/流出状态，突然转为爆发式加速流入。
        if {'consensus_overall_inflow', 'ACCEL_5_consensus_overall_inflow'}.issubset(available_cols):
            consensus_inflow = df['consensus_overall_inflow']
            ignition_potential_score = self._calculate_normalized_score(consensus_inflow.shift(1).fillna(0), 60, ascending=False)
            current_inflow_strength = self._calculate_normalized_score(consensus_inflow, 60)
            acceleration_score = self._calculate_normalized_score(df['ACCEL_5_consensus_overall_inflow'], 60)
            df['FF_SCORE_TRIGGER_IGNITION'] = ignition_potential_score * current_inflow_strength * acceleration_score

        # --- 4. 【行为】主力洗盘吸筹得分 ---
        # 逻辑: 在周线趋势向好的背景下，下跌日伴随“共识”资金流出，但收出长下影线且有中单资金在吸筹。
        required_cols = {'pct_change_D', 'consensus_overall_inflow', '_lower_shadow_ratio', 'buy_md_amount_rate_fund_flow_ths_D', 'CMF_21_W'}
        if required_cols.issubset(available_cols):
            selloff_score = self._calculate_normalized_score(pct_change, 60, ascending=False)
            outflow_score = self._calculate_normalized_score(df['consensus_overall_inflow'], 60, ascending=False)
            price_recovery_score = self._calculate_normalized_score(df['_lower_shadow_ratio'], 60)
            hidden_accumulation_score = self._calculate_normalized_score(df['buy_md_amount_rate_fund_flow_ths_D'], 60)
            weekly_confirmation_factor = (1 + self._calculate_normalized_score(df['CMF_21_W'], 120)).fillna(1) # CMF越高，因子越大
            df['FF_SCORE_BEHAVIOR_WASH_ACCUMULATION'] = selloff_score * outflow_score * price_recovery_score * hidden_accumulation_score * weekly_confirmation_factor

        # --- 5. 【行为】主力隐蔽吸筹得分 ---
        # 逻辑: “共识”主力资金大额流入，但市场参与度（成交笔数）相对较低。
        if {'consensus_main_force_inflow', 'trade_count_fund_flow_tushare_D'}.issubset(available_cols):
            main_inflow = df['consensus_main_force_inflow']
            inflow_strength_score = self._calculate_normalized_score(main_inflow, 60)
            stealth_degree_score = self._calculate_normalized_score(df['trade_count_fund_flow_tushare_D'], 120, ascending=False)
            df['FF_SCORE_BEHAVIOR_STEALTH_ACCUMULATION'] = inflow_strength_score * stealth_degree_score

        # --- 6. 【机会】主力恐慌盘涌出得分 ---
        # 逻辑: 股价显著下跌，成交量放出天量，同时主力资金（Tushare口径）卖出额达到历史极值。
        required_cols = {'pct_change_D', 'volume_D', 'sell_lg_amount_fund_flow_tushare_D', 'sell_elg_amount_fund_flow_tushare_D'}
        if required_cols.issubset(available_cols):
            main_force_sell_amount = df['sell_lg_amount_fund_flow_tushare_D'] + df['sell_elg_amount_fund_flow_tushare_D']
            df['FF_SCORE_OPP_CAPITULATION_REVERSAL'] = (
                self._calculate_normalized_score(pct_change, 120, ascending=False) *
                self._calculate_normalized_score(df['volume_D'], 120) *
                self._calculate_normalized_score(main_force_sell_amount, 120)
            )

        # --- 7. 【风险】主力资金高位对倒得分 ---
        # 逻辑: 股价滞涨，但主力买卖总额（Tushare口径）异常放大。
        required_cols = {'pct_change_D', 'buy_lg_amount_fund_flow_tushare_D', 'buy_elg_amount_fund_flow_tushare_D', 'sell_lg_amount_fund_flow_tushare_D', 'sell_elg_amount_fund_flow_tushare_D'}
        if required_cols.issubset(available_cols):
            churn_volume = (df['buy_lg_amount_fund_flow_tushare_D'] + df['buy_elg_amount_fund_flow_tushare_D'] +
                            df['sell_lg_amount_fund_flow_tushare_D'] + df['sell_elg_amount_fund_flow_tushare_D'])
            stagnation_score = self._calculate_normalized_score(df['pct_change_D'].abs(), 120, ascending=False)
            churn_intensity_score = self._calculate_normalized_score(churn_volume, 120)
            df['FF_SCORE_RISK_MAIN_FORCE_CHURNING'] = stagnation_score * churn_intensity_score

        # --- 8. 【机会】全市场资金共振买入得分 ---
        # 逻辑: “共识”资金为净流入，且所有层级资金（超大、大、中、小单）的买入意愿均处于高位。
        required_cols = {'consensus_overall_inflow', 'buy_elg_amount_rate_fund_flow_dc_D', 'buy_lg_amount_rate_fund_flow_ths_D', 'buy_md_amount_rate_fund_flow_ths_D', 'buy_sm_amount_rate_fund_flow_ths_D'}
        if required_cols.issubset(available_cols):
            df['FF_SCORE_OPP_CONSENSUS_CONVERGENCE'] = (
                self._calculate_normalized_score(df['consensus_overall_inflow'], 60) *
                self._calculate_normalized_score(df['buy_elg_amount_rate_fund_flow_dc_D'], 60) *
                self._calculate_normalized_score(df['buy_lg_amount_rate_fund_flow_ths_D'], 60) *
                self._calculate_normalized_score(df['buy_md_amount_rate_fund_flow_ths_D'], 60) *
                self._calculate_normalized_score(df['buy_sm_amount_rate_fund_flow_ths_D'], 60)
            )

        # --- 9. 【行为】主力资金韧性吸筹得分 ---
        # 逻辑: “共识”主力资金在一段时期内持续、坚定地净流入，尤其是在股价下跌时依然买入。
        if {'consensus_main_force_inflow', 'pct_change_D'}.issubset(available_cols):
            window = 10
            main_inflow = df['consensus_main_force_inflow']
            consistency_score = (main_inflow > 0).rolling(window=window).sum() / window
            magnitude_score = self._calculate_normalized_score(main_inflow.rolling(window=window).sum(), 120)
            resilience_weight = self._calculate_normalized_score(pct_change, 60, ascending=False)
            weighted_resilient_inflow = main_inflow.clip(lower=0) * resilience_weight
            resilient_inflow_sum = weighted_resilient_inflow.rolling(window=window).sum()
            total_positive_inflow_sum = main_inflow.clip(lower=0).rolling(window=window).sum()
            # 计算韧性比率：加权后的韧性流入 / 总流入
            resilience_ratio = np.divide(resilient_inflow_sum, total_positive_inflow_sum, out=np.zeros_like(resilient_inflow_sum, dtype=float), where=total_positive_inflow_sum!=0)
            # 最终得分 = 持续性 * 量级 * (1 + 韧性比率)，韧性比率作为加成项
            df['FF_SCORE_BEHAVIOR_RESILIENT_ACCUMULATION'] = consistency_score * magnitude_score * (1 + resilience_ratio)

        # --- 10. 【机会】主力资金攻击效率得分 ---
        # 逻辑: 少量“共识”主力资金（标准化后）撬动了巨大的股价涨幅，表明卖盘枯竭。
        if {'pct_change_D', 'consensus_main_force_inflow', 'circ_mv_D'}.issubset(available_cols):
            main_inflow = df['consensus_main_force_inflow']
            is_valid_day = (df['pct_change_D'] > 0) & (main_inflow > 0) & (df['circ_mv_D'] > 0)
            normalized_inflow = np.divide(main_inflow, df['circ_mv_D'], out=np.zeros_like(main_inflow, dtype=float), where=df['circ_mv_D']!=0)
            driving_efficiency = np.divide(df['pct_change_D'], normalized_inflow, out=np.zeros_like(df['pct_change_D'], dtype=float), where=normalized_inflow!=0)
            df['FF_SCORE_OPP_DRIVING_EFFICIENCY'] = self._calculate_normalized_score(driving_efficiency.where(is_valid_day, 0), 120)

        # --- 11. 【风险】资金衰竭性派发得分 ---
        # 逻辑: 在极高的“共识”资金流入和成交量下，股价冲高回落（上影线长），是经典顶部信号。
        if {'consensus_overall_inflow', 'volume_D', '_upper_shadow_ratio'}.issubset(available_cols):
            df['FF_SCORE_RISK_EXHAUSTION_DISTRIBUTION'] = (
                self._calculate_normalized_score(df['consensus_overall_inflow'], 120) *
                self._calculate_normalized_score(df['volume_D'], 120) *
                self._calculate_normalized_score(df['_upper_shadow_ratio'], 120)
            )

        # --- 12. 【风险】资金结构断裂得分 ---
        # 逻辑: 上涨过程中，“共识”主力卖出意愿增强，而“共识”散户买入意愿狂热。
        if {'pct_change_D', 'consensus_main_buy_rate', 'consensus_retail_buy_rate'}.issubset(available_cols):
            rally_strength_score = self._calculate_normalized_score(pct_change, 60)
            main_force_selling_score = self._calculate_normalized_score(df['consensus_main_buy_rate'], 120, ascending=False)
            retail_buying_score = self._calculate_normalized_score(df['consensus_retail_buy_rate'], 120)
            df['FF_SCORE_RISK_CAPITAL_FRACTURE'] = rally_strength_score * main_force_selling_score * retail_buying_score

        # --- 13. 【行为】主力卖压真空得分 ---
        # 逻辑: 上涨过程中，主力资金（Tushare口径）卖出金额占比异常之低，表明强烈惜售。
        required_cols = {'pct_change_D', 'sell_lg_amount_fund_flow_tushare_D', 'sell_elg_amount_fund_flow_tushare_D', 'amount_D'}
        if required_cols.issubset(available_cols):
            main_force_sell_amount = df['sell_lg_amount_fund_flow_tushare_D'] + df['sell_elg_amount_fund_flow_tushare_D']
            main_force_sell_rate = np.divide(main_force_sell_amount, df['amount_D'], out=np.full_like(main_force_sell_amount, np.nan, dtype=float), where=df['amount_D']!=0)
            selling_vacuum_score = self._calculate_normalized_score(pd.Series(main_force_sell_rate, index=df.index).fillna(0), 120, ascending=False)
            rally_strength_score = self._calculate_normalized_score(pct_change, 60)
            df['FF_SCORE_BEHAVIOR_SELLING_VACUUM'] = rally_strength_score * selling_vacuum_score

        # --- 14. 【行为】资金博弈烈度得分 ---
        # 逻辑: 短期“共识”资金净流入的波动性（标准差）极大，衡量市场多空分歧激烈。
        if {'consensus_overall_inflow'}.issubset(available_cols):
            battle_intensity_raw = df['consensus_overall_inflow'].rolling(window=10).std()
            df['FF_SCORE_BEHAVIOR_BATTLE_INTENSITY'] = self._calculate_normalized_score(battle_intensity_raw.fillna(0), 120)

        # --- 15. 【风险】主力内部分歧得分 ---
        # 逻辑: 上涨过程中，战术主力（大单）与战略主力（超大单）行为显著背离。
        required_cols = {'pct_change_D', 'buy_lg_amount_rate_fund_flow_ths_D', 'buy_elg_amount_rate_fund_flow_dc_D'}
        if required_cols.issubset(available_cols):
            lg_buying_frenzy_score = self._calculate_normalized_score(df['buy_lg_amount_rate_fund_flow_ths_D'], 120)
            elg_absence_score = self._calculate_normalized_score(df['buy_elg_amount_rate_fund_flow_dc_D'], 120, ascending=False)
            rally_strength_score = self._calculate_normalized_score(pct_change, 60)
            df['FF_SCORE_RISK_INTERNAL_DIVERGENCE'] = rally_strength_score * lg_buying_frenzy_score * elg_absence_score

        # --- 16. 【机会】主力协同攻击得分 ---
        # 逻辑: 上涨过程中，战术主力（大单）与战略主力（超大单）达成高度共识、协同买入。
        if required_cols.issubset(available_cols): # 复用上一信号的列检查
            lg_buying_score = self._calculate_normalized_score(df['buy_lg_amount_rate_fund_flow_ths_D'], 120)
            elg_buying_score = self._calculate_normalized_score(df['buy_elg_amount_rate_fund_flow_dc_D'], 120)
            rally_strength_score = self._calculate_normalized_score(pct_change, 60)
            df['FF_SCORE_OPP_COORDINATED_ATTACK'] = rally_strength_score * lg_buying_score * elg_buying_score

        # --- 17. 【风险】无量空涨 / 资金派发式拉升得分 ---
        # 逻辑: 股价显著上涨，但“共识”资金却在净流出。
        if {'pct_change_D', 'consensus_overall_inflow'}.issubset(available_cols):
            price_rally_score = self._calculate_normalized_score(pct_change, 120)
            flow_contradiction_score = self._calculate_normalized_score(df['consensus_overall_inflow'], 120, ascending=False)
            df['FF_SCORE_RISK_WEIGHTLESS_RALLY'] = price_rally_score * flow_contradiction_score

        # --- 18. 【机会】吸筹式下跌 / 恐慌盘吸收得分 ---
        # 逻辑: 股价显著下跌，但“共识”资金却在净流入。
        if {'pct_change_D', 'consensus_overall_inflow'}.issubset(available_cols):
            price_breakdown_score = self._calculate_normalized_score(pct_change, 120, ascending=False)
            flow_absorption_score = self._calculate_normalized_score(df['consensus_overall_inflow'], 120)
            df['FF_SCORE_OPP_ABSORPTION_BREAKDOWN'] = price_breakdown_score * flow_absorption_score
        
        # --- 19. 【行为】资金趋势强化得分 ---
        # 注意: 此信号的逻辑已移至 _diagnose_multi_timeframe_fund_flow_scores 中，
        #       以利用多周期聚合的共识指标，此处无需计算。

        # 调用多时间维度分析模块，它将使用我们刚刚创建的共识指标
        df = self._diagnose_multi_timeframe_fund_flow_scores(df)
        
        # 在函数末尾删除所有临时的中间列，减少内存占用
        temp_cols_to_drop = [col for col in df.columns if col.startswith('consensus_') or col.startswith('_')]
        if temp_cols_to_drop:
            df.drop(columns=temp_cols_to_drop, inplace=True)

        print("            -> [资金流评分中心 V10.0 性能优化版] 计算完毕。")
        return df

    def diagnose_fund_flow_states(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V11.0 性能优化版】资金流原子信号诊断模块
        - 核心升级:
          1.  【架构重构】: 引入配置驱动的信号生成模式，提升可维护性。
          2.  【逻辑精简】: 全面转向基于数值化评分的动态信号体系。
        - 性能优化:
          1.  【内存优化】: 预先创建并复用一个全为False的Series，避免在循环中重复创建临时对象。
          2.  【效率提升】: 增加对全NaN列的检查，跳过不必要的滚动分位数计算。
        """
        # print("        -> [资金流情报模块 V11.0 性能优化版] 启动...") # 更新版本号和描述
        states = {}
        p = get_params_block(self.strategy, 'fund_flow_params')
        if not get_param_value(p.get('enabled'), False):
            return states
        
        # --- 步骤一: 调用评分中心，生成所有数值化得分 ---
        # 注意：df在这里被就地修改，并由评分中心函数返回其引用
        df = self._diagnose_quantitative_fund_flow_scores(df)
        available_cols = set(df.columns)

        # --- 步骤二: 生成需要特殊逻辑的复合信号评分 ---
        # 将原有的布尔复合信号升级为数值化评分，以保持体系一致性
        if {'FF_SCORE_BEHAVIOR_WASH_ACCUMULATION', 'CMF_21_W', 'SLOPE_5_CMF_21_W'}.issubset(available_cols):
            wash_accumulation_score = df.get('FF_SCORE_BEHAVIOR_WASH_ACCUMULATION', 0)
            # 周线资金流入强度得分
            weekly_capital_score = self._calculate_normalized_score(df['CMF_21_W'], 120)
            # 周线资金动能改善得分
            weekly_momentum_score = self._calculate_normalized_score(df['SLOPE_5_CMF_21_W'], 120)
            # 将基础洗盘分、周线资金分、周线动能分相乘，得到更高置信度的复合分
            df['FF_SCORE_OPP_MTF_WASH_CONFIRMED'] = wash_accumulation_score * weekly_capital_score * weekly_momentum_score
            print("           - 已生成复合评分: FF_SCORE_OPP_MTF_WASH_CONFIRMED")
            available_cols.add('FF_SCORE_OPP_MTF_WASH_CONFIRMED')

        # --- 步骤三: 定义信号配置，实现配置化、自动化生成 ---
        # 格式: '最终信号名': ('评分列名', 触发分位数, '窗口期')
        SIGNAL_CONFIG = {
            # --- 机会信号 (Opportunity) ---
            'OPP_FUND_FLOW_COORDINATED_ATTACK_S': ('FF_SCORE_OPP_COORDINATED_ATTACK', 0.95, 250),
            'OPP_FUND_FLOW_BULLISH_DIVERGENCE_B': ('FF_SCORE_OPP_BULLISH_DIVERGENCE', 0.85, 120),
            'OPP_FUND_FLOW_ABSORPTION_BREAKDOWN_B': ('FF_SCORE_OPP_ABSORPTION_BREAKDOWN', 0.90, 120),
            'OPP_FUND_FLOW_CAPITULATION_REVERSAL_B': ('FF_SCORE_OPP_CAPITULATION_REVERSAL', 0.95, 250),
            'OPP_FUND_FLOW_CONSENSUS_CONVERGENCE_A': ('FF_SCORE_OPP_CONSENSUS_CONVERGENCE', 0.90, 250),
            'OPP_FUND_FLOW_RESILIENT_ACCUMULATION_A': ('FF_SCORE_BEHAVIOR_RESILIENT_ACCUMULATION', 0.90, 250),
            'OPP_FUND_FLOW_DRIVING_EFFICIENCY_A': ('FF_SCORE_OPP_DRIVING_EFFICIENCY', 0.90, 250),
            'OPP_FF_MTF_TREND_CONFIRMATION_S': ('FF_SCORE_MTF_TREND_CONFIRMATION', 0.98, 250),
            'OPP_FF_MTF_STRATEGIC_ACCUMULATION_S': ('FF_SCORE_MTF_STRATEGIC_ACCUMULATION', 0.98, 250),
            'OPP_FF_MTF_WASH_CONFIRMED_S': ('FF_SCORE_OPP_MTF_WASH_CONFIRMED', 0.98, 250),
            # --- 风险信号 (Risk) ---
            'RISK_FUND_FLOW_WEIGHTLESS_RALLY_A': ('FF_SCORE_RISK_WEIGHTLESS_RALLY', 0.95, 120),
            'RISK_FUND_FLOW_DISTRIBUTION_STRUCTURE_A': ('FF_SCORE_RISK_DISTRIBUTION_STRUCTURE', 0.90, 120),
            'RISK_FUND_FLOW_MAIN_FORCE_CHURNING_A': ('FF_SCORE_RISK_MAIN_FORCE_CHURNING', 0.90, 250),
            'RISK_FUND_FLOW_EXHAUSTION_DISTRIBUTION_A': ('FF_SCORE_RISK_EXHAUSTION_DISTRIBUTION', 0.95, 250),
            'RISK_FUND_FLOW_CAPITAL_FRACTURE_A': ('FF_SCORE_RISK_CAPITAL_FRACTURE', 0.95, 250),
            'RISK_FUND_FLOW_INTERNAL_DIVERGENCE_A': ('FF_SCORE_RISK_INTERNAL_DIVERGENCE', 0.95, 250),
            # --- 行为信号 (Behavior) ---
            'BEHAVIOR_FUND_FLOW_TREND_REINFORCEMENT_A': ('FF_SCORE_BEHAVIOR_TREND_REINFORCEMENT', 0.95, 250),
            'BEHAVIOR_FUND_FLOW_WASH_ACCUMULATION_A': ('FF_SCORE_BEHAVIOR_WASH_ACCUMULATION', 0.95, 120),
            'BEHAVIOR_FUND_FLOW_STEALTH_ACCUMULATION_B': ('FF_SCORE_BEHAVIOR_STEALTH_ACCUMULATION', 0.85, 120),
            'BEHAVIOR_FUND_FLOW_SELLING_VACUU_A': ('FF_SCORE_BEHAVIOR_SELLING_VACUUM', 0.90, 250),
            'BEHAVIOR_FUND_FLOW_BATTLE_INTENSE_B': ('FF_SCORE_BEHAVIOR_BATTLE_INTENSITY', 0.90, 120),
            # --- 触发器信号 (Trigger) ---
            'TRIGGER_FUND_FLOW_IGNITION': ('FF_SCORE_TRIGGER_IGNITION', 0.95, 120),
            'TRIGGER_FF_MTF_ACCEL_INFLECTION_A': ('FF_SCORE_MTF_TRIGGER_ACCEL_INFLECTION', 0.95, 120),
        }
        all_false_series = pd.Series(False, index=df.index)
        # print("        -> 正在根据配置，自动化生成最终布尔信号...")
        for signal_name, (score_col, quantile, window) in SIGNAL_CONFIG.items():
            if score_col in available_cols and not df[score_col].isnull().all():
                score = df[score_col]
                threshold = score.rolling(window=window, min_periods=int(window*0.2)).quantile(quantile)
                states[signal_name] = score > threshold
            else:
                states[signal_name] = all_false_series
        print("        -> [资金流情报模块 V13.0 最终数值化版] 诊断完毕。")
        return states

    def _diagnose_multi_timeframe_fund_flow_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        【V10.0 性能优化版】多时间维度资金流情报分析中心
        - 核心: 专注于处理多周期聚合资金流数据，生成基于多维交叉验证的高置信度信号。
        - 优化:
          1. 高效列检查: 使用集合(set)进行列存在性检查，将重复检查的时间复杂度从O(N)降至O(1)。
          2. 功能补全: 根据V9.0的设计，将“资金趋势强化得分”的计算逻辑移入此方法，使其名副其实。
          3. 结构清晰: 将逻辑划分为“预检查”、“多维信号计算”、“趋势强化计算”和“信号加固”四个部分。
        """
        print("            -> [资金流升维引擎 V10.0] 正在生成多维交叉验证信号...")

        # --- 阶段一: 预检查与准备 ---
        available_cols = set(df.columns) # 创建一个包含所有可用列的集合，用于后续高效的O(1)复杂度检查。

        # --- 阶段二: 计算多维交叉验证信号得分 ---
        # --- 信号1: 【机会】多维趋势共振确认 (Multi-Dimensional Trend Confirmation) ---
        # 逻辑: 短(5日)、中(21日)、长(55日)三个周期的资金流入趋势(5日斜率)同时向上，形成强大的趋势共振。
        required_cols = { # 使用集合定义依赖列
            'SLOPE_5_net_d5_net_amount_fund_flow_ths_D', 'SLOPE_5_net_d21_net_amount_fund_flow_ths_D', 'SLOPE_5_net_d55_net_amount_fund_flow_ths_D',
            'SLOPE_5_net_d5_net_amount_fund_flow_dc_D', 'SLOPE_5_net_d21_net_amount_fund_flow_dc_D', 'SLOPE_5_net_d55_net_amount_fund_flow_dc_D',
            'SLOPE_5_net_d5_net_mf_amount_fund_flow_tushare_D', 'SLOPE_5_net_d21_net_mf_amount_fund_flow_tushare_D', 'SLOPE_5_net_d55_net_mf_amount_fund_flow_tushare_D'
        }
        if required_cols.issubset(available_cols): # 使用更高效的 issubset 方法进行检查
            # a. 分别计算同花顺、东方财富、Tushare三个数据源的共振得分
            score_ths = (self._calculate_normalized_score(df['SLOPE_5_net_d5_net_amount_fund_flow_ths_D'], 120) *
                         self._calculate_normalized_score(df['SLOPE_5_net_d21_net_amount_fund_flow_ths_D'], 120) *
                         self._calculate_normalized_score(df['SLOPE_5_net_d55_net_amount_fund_flow_ths_D'], 120))
            score_dc = (self._calculate_normalized_score(df['SLOPE_5_net_d5_net_amount_fund_flow_dc_D'], 120) *
                        self._calculate_normalized_score(df['SLOPE_5_net_d21_net_amount_fund_flow_dc_D'], 120) *
                        self._calculate_normalized_score(df['SLOPE_5_net_d55_net_amount_fund_flow_dc_D'], 120))
            score_tushare = (self._calculate_normalized_score(df['SLOPE_5_net_d5_net_mf_amount_fund_flow_tushare_D'], 120) *
                             self._calculate_normalized_score(df['SLOPE_5_net_d21_net_mf_amount_fund_flow_tushare_D'], 120) *
                             self._calculate_normalized_score(df['SLOPE_5_net_d55_net_mf_amount_fund_flow_tushare_D'], 120))
            # b. 对三个数据源的得分进行平均，只有三方共振时得分才高，增加信号鲁棒性
            df['FF_SCORE_MTF_TREND_CONFIRMATION'] = (score_ths + score_dc + score_tushare) / 3

        # --- 信号2: 【机会】战略性吸筹确认 (Strategic Accumulation Confirmation) ---
        # 逻辑: 满足“静态存量(长期资金为正) + 动态增量(中短期趋势向上)”的双重确认。
        required_cols = { # 使用集合定义依赖列
            'net_d55_net_amount_fund_flow_ths_D', 'SLOPE_5_net_d5_net_amount_fund_flow_ths_D', 'SLOPE_5_net_d21_net_amount_fund_flow_ths_D',
            'net_d55_net_amount_fund_flow_dc_D', 'SLOPE_5_net_d5_net_amount_fund_flow_dc_D', 'SLOPE_5_net_d21_net_amount_fund_flow_dc_D'
        }
        if required_cols.issubset(available_cols): # 使用更高效的 issubset 方法进行检查
            static_score_ths = self._calculate_normalized_score(df['net_d55_net_amount_fund_flow_ths_D'], 120)
            static_score_dc = self._calculate_normalized_score(df['net_d55_net_amount_fund_flow_dc_D'], 120)
            static_condition_score = (static_score_ths + static_score_dc) / 2 # 长期资金流入越多，静态得分越高
            dynamic_score_ths = (self._calculate_normalized_score(df['SLOPE_5_net_d5_net_amount_fund_flow_ths_D'], 120) *
                                 self._calculate_normalized_score(df['SLOPE_5_net_d21_net_amount_fund_flow_ths_D'], 120))
            dynamic_score_dc = (self._calculate_normalized_score(df['SLOPE_5_net_d5_net_amount_fund_flow_dc_D'], 120) *
                                self._calculate_normalized_score(df['SLOPE_5_net_d21_net_amount_fund_flow_dc_D'], 120))
            final_dynamic_score = (dynamic_score_ths + dynamic_score_dc) / 2
            df['FF_SCORE_MTF_STRATEGIC_ACCUMULATION'] = static_condition_score * final_dynamic_score

        # --- 信号3: 【触发器】趋势加速拐点 (Trend Acceleration Inflection) ---
        # 逻辑: 捕捉中期趋势(21日)的“加速度”出现显著拐点的时刻，通常是主升浪起点。
        required_cols = { # 使用集合定义依赖列
            'ACCEL_5_net_d21_net_amount_fund_flow_ths_D', 'ACCEL_5_net_d21_net_amount_fund_flow_dc_D',
            'SLOPE_5_net_d5_net_amount_fund_flow_ths_D'
        }
        if required_cols.issubset(available_cols): # 使用更高效的 issubset 方法进行检查
            base_condition_score = self._calculate_normalized_score(df['SLOPE_5_net_d5_net_amount_fund_flow_ths_D'], 120)
            accel_score = (self._calculate_normalized_score(df['ACCEL_5_net_d21_net_amount_fund_flow_ths_D'], 120) +
                           self._calculate_normalized_score(df['ACCEL_5_net_d21_net_amount_fund_flow_dc_D'], 120)) / 2
            df['FF_SCORE_MTF_TRIGGER_ACCEL_INFLECTION'] = base_condition_score * accel_score

        # --- 阶段三: 【新增】计算资金趋势强化得分 ---
        # 补全V9.0版本中规划移至此处的“资金趋势强化”信号逻辑
        # 逻辑: 捕捉中期资金流入趋势（存量）已形成，且短期内该趋势还在急剧加速（增量）的“戴维斯双击”时刻。
        required_cols = {'net_d5_net_amount_fund_flow_ths_D', 'net_d5_net_amount_fund_flow_dc_D'}
        if required_cols.issubset(available_cols):
            # a. 使用同花顺和东方财富的5日净流入均值，作为更可靠的中期趋势“共识”指标
            consensus_net_d5_inflow = df[['net_d5_net_amount_fund_flow_ths_D', 'net_d5_net_amount_fund_flow_dc_D']].mean(axis=1)
            trend_strength_score = self._calculate_normalized_score(consensus_net_d5_inflow, 120)
            reinforcement_metric = consensus_net_d5_inflow.diff(1)
            reinforcement_score = self._calculate_normalized_score(reinforcement_metric.fillna(0), 120)
            df['FF_SCORE_BEHAVIOR_TREND_REINFORCEMENT'] = trend_strength_score * reinforcement_score

        # --- 阶段四: 信号加固模块 ---
        # 逻辑: 将战术信号得分与本方法生成的战略环境得分相乘，生成置信度更高的复合信号。
        
        # --- 加固信号1: 【机会】战略背景下的资金点火 ---
        required_cols = {'FF_SCORE_TRIGGER_IGNITION', 'FF_SCORE_MTF_STRATEGIC_ACCUMULATION'}
        if required_cols.issubset(set(df.columns)): # 使用集合检查，并实时获取最新列
            df['FF_SCORE_REINFORCED_STRATEGIC_IGNITION'] = df['FF_SCORE_TRIGGER_IGNITION'] * df['FF_SCORE_MTF_STRATEGIC_ACCUMULATION']

        # --- 加固信号2: 【机会】趋势共振中的主力协同攻击 ---
        required_cols = {'FF_SCORE_OPP_COORDINATED_ATTACK', 'FF_SCORE_MTF_TREND_CONFIRMATION'}
        if required_cols.issubset(set(df.columns)): # 使用集合检查，并实时获取最新列
            df['FF_SCORE_REINFORCED_RESONANT_ATTACK'] = df['FF_SCORE_OPP_COORDINATED_ATTACK'] * df['FF_SCORE_MTF_TREND_CONFIRMATION']

        # --- 加固信号3: 【风险】趋势衰竭时的资金结构断裂 ---
        required_cols = {'FF_SCORE_RISK_CAPITAL_FRACTURE', 'SLOPE_5_net_d21_net_amount_fund_flow_ths_D'}
        if required_cols.issubset(available_cols): # 使用集合检查
            # a. 量化“中期趋势衰竭度” (21日资金流斜率越小，衰竭度得分越高)
            trend_exhaustion_score = self._calculate_normalized_score(df['SLOPE_5_net_d21_net_amount_fund_flow_ths_D'], 120, ascending=False)
            # b. 战术风险得分 * 趋势环境风险得分 = 复合风险得分
            df['FF_SCORE_REINFORCED_EXHAUSTION_FRACTURE'] = df['FF_SCORE_RISK_CAPITAL_FRACTURE'] * trend_exhaustion_score
            
        return df
















