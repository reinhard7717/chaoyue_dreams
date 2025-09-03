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
        计算一个系列在滚动窗口内的归一化得分 (0-1)。
        得分是基于该值在窗口期内的百分位排名。
        :param series: 输入数据系列。
        :param window: 滚动窗口大小。
        :param ascending: True表示值越大得分越高，False反之。
        :return: 归一化得分系列 (0-1)。
        """
        if ascending:
            rank = series.rolling(window=window, min_periods=int(window*0.2)).rank(pct=True)
        else:
            # 对于逆向指标，值越小，排名越高
            rank = series.rolling(window=window, min_periods=int(window*0.2)).rank(pct=True, ascending=False)
        return rank.fillna(0.5) # 用0.5填充NaN，表示中性

    def _diagnose_quantitative_fund_flow_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        【V5.0 新增】资金流情报定量评分中心
        - 核心: 集中计算所有基于资金流的数值化得分 (0-1范围)。
        """
        print("            -> [资金流评分中心 V5.0] 正在生成新增战术信号得分...")

        # --- 1. 【机会】资金底背离反转得分 (Bullish Divergence Score) ---
        # 逻辑: 股价创阶段性新低，但资金流出动能减弱或转为流入。
        if 'close_D' in df.columns and 'SLOPE_5_net_amount_fund_flow_ths_D' in df.columns:
            # a. 量化价格新低程度 (价格越接近近期低点，得分越高)
            #    我们用 1 - 价格的百分位排名 来表示。价格排名越低(接近0)，得分越高(接近1)。
            price_new_low_score = 1 - self._calculate_normalized_score(df['close_D'], window=20, ascending=True)
            # b. 量化资金趋势改善程度 (资金流入斜率越大，得分越高)
            flow_improvement_score = self._calculate_normalized_score(df['SLOPE_5_net_amount_fund_flow_ths_D'], window=60, ascending=True)
            # c. 合成信号: 只有在价格创新低的同时资金趋势显著改善，得分才高
            df['FF_SCORE_OPP_BULLISH_DIVERGENCE'] = price_new_low_score * flow_improvement_score
            # print("               - 已生成'资金底背离'机会得分: FF_SCORE_OPP_BULLISH_DIVERGENCE")

        # --- 2. 【风险】主力派发、散户接盘得分 (Distribution Structure Score) ---
        # 逻辑: 主力资金买入意愿下降，而散户资金买入意愿上升。
        required_cols_dist = [
            'SLOPE_5_buy_lg_amount_rate_fund_flow_ths_D', 
            'buy_sm_amount_rate_fund_flow_ths_D', 
            'buy_md_amount_rate_fund_flow_ths_D'
        ]
        if all(c in df.columns for c in required_cols_dist):
            # a. 量化主力派发强度 (大单买入占比的斜率越负，得分越高)
            main_force_selling_score = self._calculate_normalized_score(df['SLOPE_5_buy_lg_amount_rate_fund_flow_ths_D'], window=60, ascending=False)
            # b. 量化散户接盘强度 (小单+中单买入占比的斜率越正，得分越高)
            retail_buy_rate = df['buy_sm_amount_rate_fund_flow_ths_D'] + df['buy_md_amount_rate_fund_flow_ths_D']
            #   动态计算散户买入占比的5日斜率
            retail_buy_slope = retail_buy_rate.diff(5) / 5 
            retail_fomo_score = self._calculate_normalized_score(retail_buy_slope, window=60, ascending=True)
            # c. 合成信号: 主力派发和散户接盘同时发生时，风险得分最高
            df['FF_SCORE_RISK_DISTRIBUTION_STRUCTURE'] = main_force_selling_score * retail_fomo_score
            # print("               - 已生成'主力派发散户接盘'风险得分: FF_SCORE_RISK_DISTRIBUTION_STRUCTURE")

        # --- 3. 【机会/触发器】资金点火得分 (Ignition Score) ---
        # 逻辑: 资金从沉寂/流出状态，突然转为爆发式加速流入。
        if 'net_amount_fund_flow_ths_D' in df.columns and 'ACCEL_5_net_amount_fund_flow_ths_D' in df.columns:
            # a. 量化“拐点”强度 (前一日流出/沉寂，当日强力流入)
            is_prev_neg_or_zero = (df['net_amount_fund_flow_ths_D'].shift(1) <= 0).astype(int)
            current_inflow_strength = self._calculate_normalized_score(df['net_amount_fund_flow_ths_D'], window=60, ascending=True)
            inflection_score = is_prev_neg_or_zero * current_inflow_strength
            # b. 量化“加速度”强度 (资金流入加速度越大，得分越高)
            acceleration_score = self._calculate_normalized_score(df['ACCEL_5_net_amount_fund_flow_ths_D'], window=60, ascending=True)
            # c. 合成信号: 强拐点与强加速同时发生，形成点火信号
            df['FF_SCORE_TRIGGER_IGNITION'] = inflection_score * acceleration_score
            # print("               - 已生成'资金点火'触发器得分: FF_SCORE_TRIGGER_IGNITION")
        # --- 4. 【行为】主力洗盘吸筹得分 (Wash Trading & Accumulation Score) ---
        # 逻辑: 识别“大单假意砸盘，拆单低位接回”的洗盘行为。
        required_cols_wash = [
            'pct_change_D', 'net_amount_fund_flow_ths_D', 'low_D', 'close_D', 'high_D',
            'buy_md_amount_rate_fund_flow_ths_D'
        ]
        if all(c in df.columns for c in required_cols_wash):
            # a. 量化“砸盘”背景 (当天需为下跌+资金净流出)
            is_selloff_day = ((df['pct_change_D'] < -0.01) & (df['net_amount_fund_flow_ths_D'] < 0)).astype(int)
            # b. 量化“价格顽强收回”程度 (下影线长度占比越高，得分越高)
            #    下影线长度 = close - low, K线实体总长度 = high - low
            #    使用 np.divide 来避免除以零的错误
            k_line_range = df['high_D'] - df['low_D']
            lower_shadow_ratio = np.divide(
                df['close_D'] - df['low_D'], 
                k_line_range, 
                out=np.zeros_like(df['close_D'], dtype=float), 
                where=k_line_range!=0
            )
            price_recovery_score = self._calculate_normalized_score(pd.Series(lower_shadow_ratio, index=df.index), window=60, ascending=True)
            # c. 量化“中单秘密吸筹”强度 (中单买入占比越高，得分越高)
            hidden_accumulation_score = self._calculate_normalized_score(df['buy_md_amount_rate_fund_flow_ths_D'], window=60, ascending=True)
            # d. 合成信号: 必须在砸盘日，同时出现价格强力收回和中单异常买入，得分才高
            df['FF_SCORE_BEHAVIOR_WASH_ACCUMULATION'] = is_selloff_day * price_recovery_score * hidden_accumulation_score
            # print("               - 已生成'主力洗盘吸筹'行为得分: FF_SCORE_BEHAVIOR_WASH_ACCUMULATION")
        # --- 5. 【行为】主力隐蔽吸筹得分 (Stealth Accumulation Score) ---
        # 逻辑: 识别“大额资金流入，但市场参与度（成交笔数）相对较低”的现象。
        required_cols_stealth = ['net_mf_amount_fund_flow_tushare_D', 'trade_count_fund_flow_tushare_D']
        if all(c in df.columns for c in required_cols_stealth):
            # a. 必须是主力资金净流入
            is_main_force_inflow = (df['net_mf_amount_fund_flow_tushare_D'] > 0).astype(int)
            # b. 量化“资金流入强度” (流入金额越大，得分越高)
            inflow_strength_score = self._calculate_normalized_score(df['net_mf_amount_fund_flow_tushare_D'], window=60, ascending=True)
            # c. 量化“行为隐蔽度” (成交笔数越少，得分越高)
            #    ascending=False 表示值越小，排名越高，得分越高
            stealth_degree_score = self._calculate_normalized_score(df['trade_count_fund_flow_tushare_D'], window=120, ascending=False)
            # d. 合成信号: 必须是主力净流入日，且流入强度高、行为隐蔽度高，得分才高
            df['FF_SCORE_BEHAVIOR_STEALTH_ACCUMULATION'] = is_main_force_inflow * inflow_strength_score * stealth_degree_score
            # print("               - 已生成'主力隐蔽吸筹'行为得分: FF_SCORE_BEHAVIOR_STEALTH_ACCUMULATION")
        # --- 6. 【机会】主力恐慌盘涌出得分 (Capitulation Reversal Score) ---
        # 逻辑: 识别“价跌量增”背景下，主力资金（大单+超大单）卖出金额达到历史极值的恐慌日。
        required_cols_capitulation = [
            'pct_change_D', 'volume_D', 
            'sell_lg_amount_fund_flow_tushare_D', 'sell_elg_amount_fund_flow_tushare_D'
        ]
        if all(c in df.columns for c in required_cols_capitulation):
            # a. 必须是下跌日，最好是显著下跌
            is_selloff_day = (df['pct_change_D'] < -0.03).astype(int)
            # b. 量化“价格恐慌度” (跌幅越深，得分越高)
            price_panic_score = self._calculate_normalized_score(df['pct_change_D'], window=120, ascending=False)
            # c. 量化“成交恐慌度” (成交量越大，得分越高)
            volume_panic_score = self._calculate_normalized_score(df['volume_D'], window=120, ascending=True)
            # d. 量化“主力投降度” (主力卖出金额越大，得分越高)
            main_force_sell_amount = df['sell_lg_amount_fund_flow_tushare_D'] + df['sell_elg_amount_fund_flow_tushare_D']
            main_force_capitulation_score = self._calculate_normalized_score(main_force_sell_amount, window=120, ascending=True)
            # e. 合成信号: 必须是恐慌下跌日，且价格、成交量、主力卖盘三者共振，得分才高
            df['FF_SCORE_OPP_CAPITULATION_REVERSAL'] = is_selloff_day * price_panic_score * volume_panic_score * main_force_capitulation_score
            # print("               - 已生成'主力恐慌盘涌出'机会得分: FF_SCORE_OPP_CAPITULATION_REVERSAL")
        # --- 7. 【风险】主力资金高位对倒得分 (Main Force Churning Score) ---
        # 逻辑: 识别股价滞涨，但主力买卖总额（而非净额）异常放大的高风险状态。
        required_cols_churning = [
            'pct_change_D', 'buy_lg_amount_fund_flow_tushare_D', 'buy_elg_amount_fund_flow_tushare_D',
            'sell_lg_amount_fund_flow_tushare_D', 'sell_elg_amount_fund_flow_tushare_D'
        ]
        if all(c in df.columns for c in required_cols_churning):
            # a. 量化“价格滞涨度” (日涨跌幅的绝对值越小，得分越高)
            #    ascending=False 表示值越小，得分越高
            stagnation_score = self._calculate_normalized_score(df['pct_change_D'].abs(), window=120, ascending=False)
            # b. 计算主力总交易额（买+卖），衡量“对倒”的激烈程度
            main_force_buy_total = df['buy_lg_amount_fund_flow_tushare_D'] + df['buy_elg_amount_fund_flow_tushare_D']
            main_force_sell_total = df['sell_lg_amount_fund_flow_tushare_D'] + df['sell_elg_amount_fund_flow_tushare_D']
            churn_volume = main_force_buy_total + main_force_sell_total
            # c. 量化“对倒激烈度” (主力总交易额越大，得分越高)
            churn_intensity_score = self._calculate_normalized_score(churn_volume, window=120, ascending=True)
            # d. 合成信号: 价格越滞涨、对倒越激烈，风险得分越高
            df['FF_SCORE_RISK_MAIN_FORCE_CHURNING'] = stagnation_score * churn_intensity_score
            # print("               - 已生成'主力资金高位对倒'风险得分: FF_SCORE_RISK_MAIN_FORCE_CHURNING")
        # --- 8. 【机会】全市场资金共振买入得分 (Consensus Convergence Score) ---
        # 逻辑: 识别所有层级资金（超大、大、中、小单）的买入意愿均处于高位的罕见状态。
        required_cols_consensus = [
            'net_amount_fund_flow_ths_D', 'buy_elg_amount_rate_fund_flow_dc_D', 
            'buy_lg_amount_rate_fund_flow_ths_D', 'buy_md_amount_rate_fund_flow_ths_D', 
            'buy_sm_amount_rate_fund_flow_ths_D'
        ]
        if all(c in df.columns for c in required_cols_consensus):
            # a. 基础条件：当天必须是整体资金净流入
            is_overall_inflow = (df['net_amount_fund_flow_ths_D'] > 0).astype(int)
            # b. 量化各层级资金的买入强度 (买入占比越高，得分越高)
            score_elg = self._calculate_normalized_score(df['buy_elg_amount_rate_fund_flow_dc_D'], window=60, ascending=True)
            score_lg = self._calculate_normalized_score(df['buy_lg_amount_rate_fund_flow_ths_D'], window=60, ascending=True)
            score_md = self._calculate_normalized_score(df['buy_md_amount_rate_fund_flow_ths_D'], window=60, ascending=True)
            score_sm = self._calculate_normalized_score(df['buy_sm_amount_rate_fund_flow_ths_D'], window=60, ascending=True)
            # c. 合成信号: 必须是净流入日，且所有层级的买入强度得分都高，最终得分才高。
            #    使用乘法可以确保“木桶效应”，任何一方的缺席都会显著拉低总分。
            df['FF_SCORE_OPP_CONSENSUS_CONVERGENCE'] = is_overall_inflow * score_elg * score_lg * score_md * score_sm
            # print("               - 已生成'全市场资金共振买入'机会得分: FF_SCORE_OPP_CONSENSUS_CONVERGENCE")
        # --- 9. 【行为】主力资金韧性吸筹得分 (Resilient Accumulation Score) ---
        # 逻辑: 识别主力资金在一段时期内（如10天）持续、坚定，尤其是在股价下跌时依然买入的战略性吸筹行为。
        required_cols_resilience = ['net_mf_amount_fund_flow_tushare_D', 'pct_change_D']
        if all(c in df.columns for c in required_cols_resilience):
            window = 10 # 定义观察窗口期为10天
            # a. 量化“持续性”：过去10天中，主力净流入天数的占比
            inflow_days = (df['net_mf_amount_fund_flow_tushare_D'] > 0).rolling(window=window).sum()
            consistency_score = inflow_days / window
            # b. 量化“强度”：过去10天累计的主力净流入金额的强度分
            total_inflow = df['net_mf_amount_fund_flow_tushare_D'].rolling(window=window).sum()
            magnitude_score = self._calculate_normalized_score(total_inflow, window=120, ascending=True)
            # c. 量化“韧性”：过去10天中，在股价下跌或平盘时买入的金额占总买入金额的比例（作为加分项）
            #    clip(lower=0)确保我们只考虑净流入的部分
            resilient_inflow = df['net_mf_amount_fund_flow_tushare_D'].where(df['pct_change_D'] <= 0, 0).clip(lower=0)
            total_positive_inflow = df['net_mf_amount_fund_flow_tushare_D'].clip(lower=0)
            resilient_inflow_sum = resilient_inflow.rolling(window=window).sum()
            total_positive_inflow_sum = total_positive_inflow.rolling(window=window).sum()
            # 使用np.divide避免除以零
            resilience_ratio = np.divide(
                resilient_inflow_sum, 
                total_positive_inflow_sum,
                out=np.zeros_like(resilient_inflow_sum, dtype=float),
                where=total_positive_inflow_sum!=0
            )
            # d. 合成信号: 持续性 * 强度 * (1 + 韧性比例)。韧性作为核心加分项。
            #    最终得分体现了主力在一段时间内的战略决心和执行力。
            df['FF_SCORE_BEHAVIOR_RESILIENT_ACCUMULATION'] = consistency_score * magnitude_score * (1 + resilience_ratio)
            # print("               - 已生成'主力资金韧性吸筹'行为得分: FF_SCORE_BEHAVIOR_RESILIENT_ACCUMULATION")
        # --- 10. 【机会】主力资金攻击效率得分 (Driving Efficiency Score) ---
        # 逻辑: 识别“四两拨千斤”的时刻，即少量的主力资金（标准化后）撬动了巨大的股价涨幅，表明卖盘枯竭。
        required_cols_efficiency = ['pct_change_D', 'net_mf_amount_fund_flow_tushare_D', 'circ_mv_D']
        if all(c in df.columns for c in required_cols_efficiency):
            # a. 定义有效日：必须是上涨日，且主力资金净流入，且流通市值有效
            is_valid_day = (df['pct_change_D'] > 0) & (df['net_mf_amount_fund_flow_tushare_D'] > 0) & (df['circ_mv_D'] > 0)
            # b. 计算标准化的主力买入压力 (主力净流入 / 流通市值)
            #    这消除了盘子大小的影响，使得不同股票之间可比
            normalized_inflow = np.divide(
                df['net_mf_amount_fund_flow_tushare_D'],
                df['circ_mv_D'],
                out=np.zeros_like(df['net_mf_amount_fund_flow_tushare_D'], dtype=float),
                where=df['circ_mv_D']!=0
            )
            # c. 计算核心指标：资金驱动效率 (涨幅 / 标准化买入压力)
            #    该值越高，说明撬动股价所需的资金越少，卖压越轻
            driving_efficiency = np.divide(
                df['pct_change_D'],
                normalized_inflow,
                out=np.zeros_like(df['pct_change_D'], dtype=float),
                where=normalized_inflow!=0
            )
            # d. 只在有效日计算效率，其他日子为0，然后进行归一化评分
            driving_efficiency_on_valid_days = driving_efficiency.where(is_valid_day, 0)
            efficiency_score = self._calculate_normalized_score(driving_efficiency_on_valid_days, window=120, ascending=True)
            df['FF_SCORE_OPP_DRIVING_EFFICIENCY'] = efficiency_score
            # print("               - 已生成'主力资金攻击效率'机会得分: FF_SCORE_OPP_DRIVING_EFFICIENCY")
        # --- 11. 【风险】资金衰竭性派发得分 (Exhaustion Distribution Score) ---
        # 逻辑: 识别在极高的资金流入和成交量下，股价上涨乏力或冲高回落的“多头陷阱”，是经典的顶部信号。
        required_cols_exhaustion = [
            'net_amount_fund_flow_ths_D', 'volume_D', 'high_D', 'low_D', 'close_D'
        ]
        if all(c in df.columns for c in required_cols_exhaustion):
            # a. 定义信号的有效背景：当天必须是资金净流入
            is_inflow_day = (df['net_amount_fund_flow_ths_D'] > 0).astype(int)
            # b. 量化“资金狂热度” (资金流入越高，得分越高)
            inflow_frenzy_score = self._calculate_normalized_score(df['net_amount_fund_flow_ths_D'], window=120, ascending=True)
            # c. 量化“成交狂热度” (成交量越高，得分越高)
            volume_frenzy_score = self._calculate_normalized_score(df['volume_D'], window=120, ascending=True)
            # d. 量化“冲高回落压力” (上影线占K线总长度的比例越高，得分越高)
            k_line_range = df['high_D'] - df['low_D']
            upper_shadow_ratio = np.divide(
                df['high_D'] - df['close_D'],
                k_line_range,
                out=np.zeros_like(df['high_D'], dtype=float),
                where=k_line_range!=0
            )
            reversal_pressure_score = self._calculate_normalized_score(pd.Series(upper_shadow_ratio, index=df.index), window=120, ascending=True)
            # e. 合成信号: 必须是净流入日，且资金狂热、成交狂热、回落压力三者共振，风险得分才高。
            df['FF_SCORE_RISK_EXHAUSTION_DISTRIBUTION'] = is_inflow_day * inflow_frenzy_score * volume_frenzy_score * reversal_pressure_score
            # print("               - 已生成'资金衰竭性派发'风险得分: FF_SCORE_RISK_EXHAUSTION_DISTRIBUTION")
        # --- 12. 【风险】资金结构断裂得分 (Capital Fracture Score) ---
        # 逻辑: 识别上涨过程中，主力资金（超大+大单）与散户资金（中+小单）行为的极端背离，是高位派发的精准信号。
        required_cols_fracture = [
            'pct_change_D', 'buy_elg_amount_rate_fund_flow_dc_D', 'buy_lg_amount_rate_fund_flow_ths_D',
            'buy_md_amount_rate_fund_flow_ths_D', 'buy_sm_amount_rate_fund_flow_ths_D'
        ]
        if all(c in df.columns for c in required_cols_fracture):
            # a. 定义信号的有效背景：当天必须是上涨日，因为这是最典型的派发场景
            is_rally_day = (df['pct_change_D'] > 0.01).astype(int)
            # b. 量化“主力资金”的卖出意愿 (买入占比越低，得分越高)
            #    我们简单地将超大单和大单的买入占比平均，代表主力行为
            main_force_buy_rate = (df['buy_elg_amount_rate_fund_flow_dc_D'] + df['buy_lg_amount_rate_fund_flow_ths_D']) / 2
            main_force_selling_score = self._calculate_normalized_score(main_force_buy_rate, window=120, ascending=False)
            # c. 量化“散户资金”的买入狂热度 (买入占比越高，得分越高)
            #    我们将中单和小单的买入占比相加，代表散户行为
            retail_buy_rate = df['buy_md_amount_rate_fund_flow_ths_D'] + df['buy_sm_amount_rate_fund_flow_ths_D']
            retail_buying_score = self._calculate_normalized_score(retail_buy_rate, window=120, ascending=True)
            # d. 合成信号: 必须是上涨日，且主力卖出意愿和散户买入意愿同时达到极端，风险得分才高。
            df['FF_SCORE_RISK_CAPITAL_FRACTURE'] = is_rally_day * main_force_selling_score * retail_buying_score
            # print("               - 已生成'资金结构断裂'风险得分: FF_SCORE_RISK_CAPITAL_FRACTURE")
        # --- 13. 【行为】主力卖压真空得分 (Selling Vacuum Score) ---
        # 逻辑: 识别上涨过程中，主力资金（超大+大单）卖出金额占比异常之低的状态，表明其强烈惜售，看好后市。
        required_cols_vacuum = [
            'pct_change_D', 'sell_lg_amount_fund_flow_tushare_D', 
            'sell_elg_amount_fund_flow_tushare_D', 'amount_D'
        ]
        if all(c in df.columns for c in required_cols_vacuum):
            # a. 定义信号的有效背景：当天必须是上涨日
            is_rally_day = (df['pct_change_D'] > 0).astype(int)
            # b. 计算主力总卖出额，并进行标准化（占总成交额的比例）
            main_force_sell_amount = df['sell_lg_amount_fund_flow_tushare_D'] + df['sell_elg_amount_fund_flow_tushare_D']
            main_force_sell_rate = np.divide(
                main_force_sell_amount,
                df['amount_D'],
                out=np.full_like(main_force_sell_amount, np.nan, dtype=float), # 使用nan填充，以便后续处理
                where=df['amount_D']!=0
            )
            # c. 量化“卖压真空度” (主力卖出占比越低，得分越高)
            #    使用 ascending=False，值越小，得分越高
            selling_vacuum_score = self._calculate_normalized_score(pd.Series(main_force_sell_rate, index=df.index).fillna(0), window=120, ascending=False)
            # d. 合成信号: 必须是上涨日，且主力卖压真空度高，得分才高。
            df['FF_SCORE_BEHAVIOR_SELLING_VACUUM'] = is_rally_day * selling_vacuum_score
            # print("               - 已生成'主力卖压真空'行为得分: FF_SCORE_BEHAVIOR_SELLING_VACUUM")
        # --- 14. 【行为】资金博弈烈度得分 (Battle Intensity Score) ---
        # 逻辑: 计算短期资金净流入的波动性（标准差），衡量市场多空分歧和博弈的激烈程度。
        if 'net_amount_fund_flow_ths_D' in df.columns:
            # a. 计算10日资金净流入的滚动标准差，作为博弈烈度的原始度量
            battle_intensity_raw = df['net_amount_fund_flow_ths_D'].rolling(window=10).std()
            # b. 将原始烈度值进行归一化评分，得分越高，博弈越激烈
            battle_intensity_score = self._calculate_normalized_score(battle_intensity_raw.fillna(0), window=120, ascending=True)
            df['FF_SCORE_BEHAVIOR_BATTLE_INTENSITY'] = battle_intensity_score
            print("               - 已生成'资金博弈烈度'行为得分: FF_SCORE_BEHAVIOR_BATTLE_INTENSITY")
        # --- 15. 【风险】主力内部分歧得分 (Internal Divergence Score) ---
        # 逻辑: 识别上涨过程中，战术主力（大单）与战略主力（超大单）行为的显著背离，是“游资击鼓传花”式拉升的危险信号。
        required_cols_internal_div = [
            'pct_change_D', 'buy_lg_amount_rate_fund_flow_ths_D', 'buy_elg_amount_rate_fund_flow_dc_D'
        ]
        if all(c in df.columns for c in required_cols_internal_div):
            # a. 定义信号的有效背景：当天必须是上涨日
            is_rally_day = (df['pct_change_D'] > 0).astype(int)
            # b. 量化“战术主力（大单）”的买入狂热度 (买入占比越高，得分越高)
            lg_buying_frenzy_score = self._calculate_normalized_score(df['buy_lg_amount_rate_fund_flow_ths_D'], window=120, ascending=True)
            # c. 量化“战略主力（超大单）”的缺席/卖出意愿 (买入占比越低，得分越高)
            #    使用 ascending=False，值越小，得分越高
            elg_absence_score = self._calculate_normalized_score(df['buy_elg_amount_rate_fund_flow_dc_D'], window=120, ascending=False)
            # d. 合成信号: 必须是上涨日，且大单狂热买入与超大单缺席同时发生，风险得分才高。
            df['FF_SCORE_RISK_INTERNAL_DIVERGENCE'] = is_rally_day * lg_buying_frenzy_score * elg_absence_score
            # print("               - 已生成'主力内部分歧'风险得分: FF_SCORE_RISK_INTERNAL_DIVERGENCE")
        # --- 16. 【机会】主力协同攻击得分 (Coordinated Attack Score) ---
        # 逻辑: 识别上涨过程中，战术主力（大单）与战略主力（超大单）达成高度共识、协同买入的黄金时刻。
        if all(c in df.columns for c in required_cols_internal_div): # 复用上面的列检查
            # a. 定义信号的有效背景：当天必须是上涨日
            is_rally_day = (df['pct_change_D'] > 0).astype(int)
            # b. 量化“战术主力（大单）”的买入强度 (买入占比越高，得分越高)
            lg_buying_score = self._calculate_normalized_score(df['buy_lg_amount_rate_fund_flow_ths_D'], window=120, ascending=True)
            # c. 量化“战略主力（超大单）”的买入强度 (买入占比越高，得分越高)
            elg_buying_score = self._calculate_normalized_score(df['buy_elg_amount_rate_fund_flow_dc_D'], window=120, ascending=True)
            # d. 合成信号: 必须是上涨日，且大单和超大单同时猛烈买入，机会得分才高。
            df['FF_SCORE_OPP_COORDINATED_ATTACK'] = is_rally_day * lg_buying_score * elg_buying_score
            # print("               - 已生成'主力协同攻击'机会得分: FF_SCORE_OPP_COORDINATED_ATTACK")
        # --- 17. 【风险】无量空涨 / 资金派发式拉升得分 (Weightless Rally Score) ---
        required_cols_contradiction = ['pct_change_D', 'net_amount_fund_flow_ths_D']
        if all(c in df.columns for c in required_cols_contradiction):
            # a. 定义信号的有效背景：当天必须是上涨，且资金净流出
            is_contradiction_day = ((df['pct_change_D'] > 0.01) & (df['net_amount_fund_flow_ths_D'] < 0)).astype(int)
            # b. 量化“价格上涨强度” (涨幅越大，得分越高)
            price_rally_score = self._calculate_normalized_score(df['pct_change_D'], window=120, ascending=True)
            # c. 量化“资金反向流出强度” (净流出额越大，即数值越小，得分越高)
            flow_contradiction_score = self._calculate_normalized_score(df['net_amount_fund_flow_ths_D'], window=120, ascending=False)
            # d. 合成信号: 必须是矛盾日，且价格上涨越强、资金流出越猛，风险得分越高
            df['FF_SCORE_RISK_WEIGHTLESS_RALLY'] = is_contradiction_day * price_rally_score * flow_contradiction_score
            # print("               - 已生成'无量空涨'风险得分: FF_SCORE_RISK_WEIGHTLESS_RALLY")
        # --- 18. 【机会】吸筹式下跌 / 恐慌盘吸收得分 (Absorption Breakdown Score) ---
        if all(c in df.columns for c in required_cols_contradiction): # 复用上面的列检查
            # a. 定义信号的有效背景：当天必须是下跌，且资金净流入
            is_contradiction_day = ((df['pct_change_D'] < -0.01) & (df['net_amount_fund_flow_ths_D'] > 0)).astype(int)
            # b. 量化“价格下跌强度” (跌幅越大，即数值越小，得分越高)
            price_breakdown_score = self._calculate_normalized_score(df['pct_change_D'], window=120, ascending=False)
            # c. 量化“资金反向流入强度” (净流入额越大，得分越高)
            flow_absorption_score = self._calculate_normalized_score(df['net_amount_fund_flow_ths_D'], window=120, ascending=True)
            # d. 合成信号: 必须是矛盾日，且价格下跌越深、资金流入越强，机会得分越高
            df['FF_SCORE_OPP_ABSORPTION_BREAKDOWN'] = is_contradiction_day * price_breakdown_score * flow_absorption_score
            # print("               - 已生成'吸筹式下跌'机会得分: FF_SCORE_OPP_ABSORPTION_BREAKDOWN")
        # --- 19. 【行为】资金趋势强化得分 (Trend Reinforcement Score) ---
        # 逻辑: 捕捉中期资金流入趋势（存量）已经形成，且短期内该趋势还在急剧加速（增量）的“戴维斯双击”时刻。
        if 'net_d5_amount_fund_flow_ths_D' in df.columns:
            # a. 定义信号的有效背景：5日累积净流入必须为正，代表中期趋势已形成
            is_positive_trend = (df['net_d5_amount_fund_flow_ths_D'] > 0).astype(int)
            # b. 量化“中期趋势强度”（存量强度）: 5日累积净流入额越高，得分越高
            trend_strength_score = self._calculate_normalized_score(df['net_d5_amount_fund_flow_ths_D'], window=120, ascending=True)
            # c. 量化“短期趋势加速”（增量强度）: 5日累积净流入额的当日增量越大，得分越高
            #    diff(1) 计算的是 net_d5_today - net_d5_yesterday，这等价于 net_flow_today - net_flow_5_days_ago
            #    它衡量了当前流入相对于5天前流入的加速度，是趋势强化的完美指标。
            reinforcement_metric = df['net_d5_amount_fund_flow_ths_D'].diff(1)
            reinforcement_score = self._calculate_normalized_score(reinforcement_metric.fillna(0), window=120, ascending=True)
            # d. 合成信号: 必须是中期趋势为正，且存量强度与增量强度共振，得分才高。
            df['FF_SCORE_BEHAVIOR_TREND_REINFORCEMENT'] = is_positive_trend * trend_strength_score * reinforcement_score
            # print("               - 已生成'资金趋势强化'行为得分: FF_SCORE_BEHAVIOR_TREND_REINFORCEMENT")
        return df

    def diagnose_fund_flow_states(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V6.0 最终重构版】资金流原子信号诊断模块
        - 核心升级:
          1.  【架构重构】: 引入配置驱动的信号生成模式，取代冗长的if-else，提升可维护性。
          2.  【信号收官】: 新增“主力卖压真空”信号，完成对资金行为的全维度刻画。
          3.  【逻辑精简】: 移除旧版布尔信号，全面转向基于数值化评分的动态信号体系。
        """
        print("        -> [资金流情报模块 V6.0 最终重构版] 启动...") # [修改] 更新版本号和描述
        states = {}
        p = get_params_block(self.strategy, 'fund_flow_params')
        if not get_param_value(p.get('enabled'), False):
            return states
        
        # --- 步骤一: 调用评分中心，生成所有数值化得分 ---
        df = self._diagnose_quantitative_fund_flow_scores(df)

        # --- 步骤二: 定义信号配置，实现配置化、自动化生成 ---
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
            'BEHAVIOR_FUND_FLOW_SELLING_VACUUM_A': ('FF_SCORE_BEHAVIOR_SELLING_VACUUM', 0.90, 250),
            'BEHAVIOR_FUND_FLOW_BATTLE_INTENSE_B': ('FF_SCORE_BEHAVIOR_BATTLE_INTENSITY', 0.90, 120),
            # --- 触发器信号 (Trigger) ---
            'TRIGGER_FUND_FLOW_IGNITION': ('FF_SCORE_TRIGGER_IGNITION', 0.95, 120),
        }

        # print("        -> 正在根据配置，自动化生成最终布尔信号...")
        for signal_name, (score_col, quantile, window) in SIGNAL_CONFIG.items():
            if score_col in df.columns:
                score = df[score_col]
                # 计算动态阈值：在指定的滚动窗口内，当前得分需要超过历史分位数
                threshold = score.rolling(window=window, min_periods=int(window*0.2)).quantile(quantile)
                states[signal_name] = score > threshold
                print(f"           - 已生成信号: {signal_name}")
            else:
                # 如果评分列不存在，则创建一个全为False的Series，以保证信号字典的完整性
                states[signal_name] = pd.Series(False, index=df.index)
        
        # --- 步骤三: 生成需要特殊逻辑的复合信号 (可选，此处保留几个经典的) ---
        # 复合信号可以基于基础信号或评分进行更复杂的组合
        if 'CMF_21_W' in df.columns and 'SLOPE_5_CMF_21_W' in df.columns:
            is_wash_accumulation = states.get('BEHAVIOR_FUND_FLOW_WASH_ACCUMULATION_A', pd.Series(False, index=df.index))
            is_weekly_capital_bullish = df['CMF_21_W'] > 0.05
            is_weekly_momentum_improving = df['SLOPE_5_CMF_21_W'] > 0
            states['OPP_FF_MTF_WASH_CONFIRMED_S'] = is_wash_accumulation & is_weekly_capital_bullish & is_weekly_momentum_improving
            print("           - 已生成复合信号: OPP_FF_MTF_WASH_CONFIRMED_S")

        print("        -> [资金流情报模块 V6.0 最终重构版] 诊断完毕。")
        return states


















