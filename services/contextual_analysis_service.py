# 新增文件: services/contextual_analysis_service.py

import asyncio
import datetime
from datetime import date
import logging
import traceback
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from dao_manager.tushare_daos.fund_flow_dao import FundFlowDao
from dao_manager.tushare_daos.index_basic_dao import IndexBasicDAO
from dao_manager.tushare_daos.indicator_dao import IndicatorDAO
from dao_manager.tushare_daos.industry_dao import IndustryDao
from dao_manager.tushare_daos.stock_time_trade_dao import StockTimeTradeDAO
from utils.cache_manager import CacheManager
from stock_models.industry import ConceptMaster, SwIndustry, ThsIndex, DcIndex, KplConceptInfo

logger = logging.getLogger("services")

class ContextualAnalysisService:
    """
    上下文分析服务
    - 核心职责: 专注于分析超越个股K线之外的宏观与中观信息，为策略提供“大局观”和“战场情报”。
    """
    def __init__(self, cache_manager_instance: CacheManager):
        self.cache_manager = cache_manager_instance
        # 按需初始化DAO
        self.indicator_dao = IndicatorDAO(cache_manager_instance)
        self.industry_dao = IndustryDao(cache_manager_instance)
        self.fund_flow_dao = FundFlowDao(cache_manager_instance)
        self.stock_trade_dao = StockTimeTradeDAO(cache_manager_instance)
        self.momentum_lookback = 60
        self.fund_flow_lookback = 5

    async def analyze_industry_rotation(self, end_date: datetime.date, lookback_days: int = 21, market_code: str = '000300.SH') -> Dict:
        """
        【V3.2 深度分析版】分析所有来源的板块轮动，并将结果存入数据库。
        - 核心升级: 1. 在生命周期判断中，融入了内部广度和龙头效应，使判断更精准。
                    2. 将新维度的数据一并存入 IndustryLifecycle 表，深化数据应用。
        - 优化: 使用向量化操作(np.select)替代 apply，大幅提升生命周期阶段的分配效率。
        """
        print(f"\n--- [行业生命周期预计算 V3.2] 开始分析截至 {end_date} 的所有来源板块轮动情况 ---")
        sources_to_process = ['sw', 'ths', 'dc'] 
        all_save_results = {}
        for source in sources_to_process:
            print(f"\n--- 正在处理来源: {source.upper()} ---")
            all_concepts = await self.industry_dao.get_all_concepts_by_source(source)
            if not all_concepts:
                logger.warning(f"来源 '{source}' 未找到任何板块/概念，跳过。")
                continue
            trade_dates = [end_date - datetime.timedelta(days=i) for i in range(lookback_days)][::-1]
            tasks = [self.calculate_industry_strength_rank(td, market_code, source) for td in trade_dates]
            daily_rank_results = await asyncio.gather(*tasks)
            all_ranks = []
            for i, df_rank in enumerate(daily_rank_results):
                if not df_rank.empty and 'strength_rank' in df_rank.columns:
                    df_rank['trade_date'] = trade_dates[i]
                    all_ranks.append(df_rank)
            if not all_ranks:
                logger.warning(f"来源 '{source}' 未能获取任何历史排名数据，轮动分析中止。")
                continue
            rotation_df = pd.concat(all_ranks, ignore_index=True)
            # 在分组前对关键列进行排序，可以提高后续 groupby 操作的性能。
            rotation_df.sort_values(['concept_code', 'trade_date'], inplace=True)
            def calculate_lifecycle_metrics(group):
                """为每个板块分组计算其最新的排名、斜率、加速度以及广度和龙头分数"""
                # group = group.sort_values('trade_date') # 已在外部排序，此行不再需要，提升效率
                if len(group) < 5: 
                    latest_row = group.iloc[-1]
                    return pd.Series({
                        'latest_rank': latest_row['strength_rank'],
                        'rank_slope': 0.0,
                        'rank_accel': 0.0,
                        'latest_breadth': latest_row.get('breadth_score', 0.0),
                        'latest_leader': latest_row.get('leader_score', 0.0)
                    })
                ranks = group['strength_rank'].values
                slope = np.polyfit(np.arange(min(5, len(ranks))), ranks[-5:], 1)[0] if len(ranks) >= 2 else 0.0
                accel = 0.0
                if len(group) >= 10:
                    slope_1 = np.polyfit(np.arange(5), ranks[-10:-5], 1)[0]
                    slope_2 = np.polyfit(np.arange(5), ranks[-5:], 1)[0]
                    accel = slope_2 - slope_1
                latest_row = group.iloc[-1]
                return pd.Series({
                    'latest_rank': ranks[-1],
                    'rank_slope': slope,
                    'rank_accel': accel,
                    'latest_breadth': latest_row.get('breadth_score', 0.0),
                    'latest_leader': latest_row.get('leader_score', 0.0)
                })
            # 修改-优化: 使用 sort=False 配合预排序，进一步提升性能
            lifecycle_metrics = rotation_df.groupby('concept_code', sort=False).apply(calculate_lifecycle_metrics)
            # 使用numpy.select进行向量化判断，取代原有的 apply(axis=1) 行级操作，大幅提升计算效率。
            conditions = [
                # 条件1: 预热期 (PREHEAT)
                (lifecycle_metrics['latest_rank'] < 0.4) & (lifecycle_metrics['rank_slope'] > 0.008) & (lifecycle_metrics['rank_accel'] > 0.001) & (lifecycle_metrics['latest_leader'] > 0.2),
                # 条件2: 主升段 (MARKUP)
                (lifecycle_metrics['latest_rank'] > 0.6) & (lifecycle_metrics['rank_slope'] > 0.01) & (lifecycle_metrics['latest_breadth'] > 0.5),
                # 条件3: 滞涨期 (STAGNATION)
                (lifecycle_metrics['latest_rank'] > 0.8) & (lifecycle_metrics['rank_slope'] < 0),
                # 条件4: 下跌期 (DOWNTREND)
                (lifecycle_metrics['latest_rank'] < 0.4) & (lifecycle_metrics['rank_slope'] < -0.005)
            ]
            choices = ['PREHEAT', 'MARKUP', 'STAGNATION', 'DOWNTREND']
            lifecycle_metrics['lifecycle_stage'] = np.select(conditions, choices, default='TRANSITION')
            latest_day_data = lifecycle_metrics.copy()
            latest_day_data['trade_date'] = end_date
            # 重命名列以匹配 IndustryLifecycle 模型字段
            latest_day_data.rename(columns={
                'latest_rank': 'strength_rank',
                'latest_breadth': 'breadth_score',
                'latest_leader': 'leader_score'
            }, inplace=True)
            records_to_save = latest_day_data.reset_index().to_dict('records')
            save_result = await self.industry_dao.save_industry_lifecycle(records_to_save)
            all_save_results[source] = save_result
            print(f"--- 来源 {source.upper()} 处理完成，已保存 {len(records_to_save)} 条状态。 ---")
        return all_save_results

    async def find_industry_leaders(self, concept_code: str, trade_date: datetime.date, top_n: int = 5) -> pd.DataFrame:
        """
        寻找指定板块在指定日期的龙头股。
        - 核心逻辑: 结合KPL榜单数据，对板块内的涨停股进行综合评分，找出龙头和梯队。
        - 评分维度: 涨停时间、封单额、连板状态、换手率等。
        - 优化: 对连板状态(status)的评分逻辑进行向量化，替代原有的 apply 写法，提升效率和代码可读性。
        """
        print(f"\n--- [龙头挖掘] 开始为板块 {concept_code} 在 {trade_date} 寻找龙头股 ---")
        # 1. 获取板块成分股
        members = await self.industry_dao.get_concept_members_on_date(concept_code, trade_date)
        if not members:
            print(f"    - [龙头挖掘] 未找到板块 {concept_code} 在 {trade_date} 的成分股。")
            return pd.DataFrame()
        member_codes = [m.stock.stock_code for m in members]
        # 2. 获取成分股中的涨停股信息
        limit_up_df = await self.industry_dao.get_limit_list_for_stocks(member_codes, trade_date, tag='涨停')
        if limit_up_df.empty:
            print(f"    - [龙头挖掘] 板块 {concept_code} 内当日无涨停股。")
            return pd.DataFrame()
        print(f"    - [龙头挖掘] 在板块内发现 {len(limit_up_df)} 只涨停股，开始评分...")
        df = limit_up_df.copy()
        # 3. 计算龙头分数
        # 评分项1: 涨停时间 (越早越好)
        df['lu_time_val'] = pd.to_datetime(df['lu_time'], format='%H:%M:%S', errors='coerce').astype('int64')
        df['score_time'] = 1 - df['lu_time_val'].rank(pct=True)
        # 评分项2: 封单额 (越大越好)
        df['limit_order_ratio'] = df['limit_order'] / df['free_float'].replace(0, np.nan)
        df['score_limit_order'] = df['limit_order_ratio'].rank(pct=True)
        # 评分项3: 连板状态 (越高越好)
        # 使用向量化操作替代 apply，提升评分效率
        status_series = df['status'].astype(str) # 确保为字符串类型以使用 .str 访问器
        scores = pd.Series(0.0, index=df.index) # 初始化分数为0.0，确保浮点数类型
        scores.loc[status_series.str.contains('首板', na=False)] = 0.2
        scores.loc[status_series.str.contains('2连板', na=False)] = 0.5
        scores.loc[status_series.str.contains('3连板', na=False)] = 0.7
        scores.loc[status_series.str.contains('4连板', na=False)] = 0.8
        scores.loc[status_series.str.contains('5连板|6连板', na=False)] = 0.9
        # 使用正则表达式提取连板数字，并对7连板及以上的情况赋值
        num_str = status_series.str.extract(r'(\d+)连板', expand=False)
        high_boards = pd.to_numeric(num_str, errors='coerce') >= 7
        scores.loc[high_boards.fillna(False)] = 1.0
        df['score_status'] = scores
        # 评分项4: 换手率 (适中为佳，这里简化为越高分越高，代表活跃)
        df['score_turnover'] = df['turnover_rate'].rank(pct=True)
        # 4. 计算总分
        weights = {'time': 0.30, 'limit_order': 0.30, 'status': 0.35, 'turnover': 0.05}
        df['leader_score'] = (
            df['score_time'].fillna(0) * weights['time'] +
            df['score_limit_order'].fillna(0) * weights['limit_order'] +
            df['score_status'].fillna(0) * weights['status'] +
            df['score_turnover'].fillna(0) * weights['turnover']
        )
        # 5. 排序并返回结果
        result_cols = ['stock_id', 'name', 'leader_score', 'status', 'lu_time', 'limit_order', 'turnover_rate']
        if 'stock_id' in df.columns:
            df.rename(columns={'stock_id': 'stock_code'}, inplace=True)
            result_cols[0] = 'stock_code'
        final_df = df.sort_values('leader_score', ascending=False).head(top_n)
        print(f"--- [龙头挖掘] 完成，龙头梯队如下: ---")
        print(final_df[result_cols])
        return final_df[result_cols]

    async def calculate_industry_strength_rank(self, trade_date: datetime.date, market_code: str = '000905.SH', source: str = 'ths') -> pd.DataFrame:
        """
        【V3.2 终极修复版】计算指定来源、指定交易日所有板块的强度分及排名。
        修复: 彻底确保 concept_code 始终作为普通列返回，永远不作为索引，以杜绝下游任务的 KeyError。
        """
        start_date = trade_date - datetime.timedelta(days=self.momentum_lookback + 30)
        market_daily_df = await self.indicator_dao.get_market_index_daily_data(market_code, start_date, trade_date)
        if market_daily_df.empty:
            logger.warning(f"无法获取大盘基准 {market_code} 数据，相对强度分析将跳过。")
        all_concepts = await self.industry_dao.get_all_concepts_by_source(source)
        if not all_concepts:
            logger.warning(f"来源 '{source}' 未找到任何板块，计算中止。")
            return pd.DataFrame()
        tasks = [self._process_single_industry_strength(concept, trade_date, market_daily_df) for concept in all_concepts]
        results = await asyncio.gather(*tasks)
        strength_data = [res for res in results if res is not None]
        if not strength_data:
            return pd.DataFrame()
        df = pd.DataFrame(strength_data)
        if 'strength_score' not in df.columns:
            return pd.DataFrame()
        df['strength_rank'] = df['strength_score'].rank(pct=True, ascending=True)
        # 确保返回的 DataFrame 中 'concept_code' 是一个列，而不是索引。
        return df.sort_values('strength_rank', ascending=False)

    async def prepare_fused_industry_signals(self, stock_code: str, start_date: date, end_date: date, params: dict) -> pd.DataFrame:
        """
        【V1.1 向量化重构版】行业背景融合引擎
        - 核心职责: 作为业务逻辑层，负责获取股票的多维行业归属及其原始生命周期数据，
                    然后根据配置进行加权融合，最终生成统一的、数值化的行业背景信号。
        - 优化: 完全重构了计算逻辑，使用pandas向量化操作和矩阵乘法替代原有的循环和apply，
                大幅提升了计算效率，并减少了内存开销。
        """
        # print(f"    - [行业背景融合引擎 V1.1] 启动，为 {stock_code} 生成数值化融合行业背景...")
        # 1. 从配置中获取来源权重
        source_weights = params.get('source_weights', {})
        # 2. 从DAO获取原始数据
        raw_lifecycle_df = await self.industry_dao.get_raw_lifecycle_data_for_stock(stock_code, start_date, end_date)
        if raw_lifecycle_df.empty:
            print(f"    - [行业背景融合引擎] 未获取到原始数据，无法进行融合。")
            return pd.DataFrame()
        df = raw_lifecycle_df
        df['trade_date'] = pd.to_datetime(df['trade_date'], utc=True)
        # 3. 数据透视
        pivot_df = df.pivot_table(
            index='trade_date',
            columns='concept_code',
            values=['strength_rank', 'rank_slope', 'rank_accel', 'breadth_score', 'leader_score']
        )
        final_df = pd.DataFrame(index=pivot_df.index)
        # 建立 concept_code 到其来源权重的映射，为后续向量化计算做准备
        concept_source_map = df[['concept_code', 'source']].drop_duplicates().set_index('concept_code')['source']
        concept_weight_map = concept_source_map.map(source_weights).fillna(0.1)
        # 4. 对数值型指标进行加权平均 (向量化优化)
        numeric_metrics = ['strength_rank', 'rank_slope', 'rank_accel', 'breadth_score', 'leader_score']
        for metric in numeric_metrics:
            metric_df = pivot_df.get(metric)
            if metric_df is None or metric_df.empty:
                continue
            # 创建与 metric_df 列对齐的权重Series
            weights = metric_df.columns.to_series().map(concept_weight_map)
            # 向量化计算加权值。直接用DataFrame乘以Series(权重)，pandas会自动按列广播
            weighted_values = metric_df.mul(weights, axis=1)
            # 向量化计算每日的加权总和
            weighted_sum = weighted_values.sum(axis=1)
            # 向量化计算每日的有效总权重。首先得到一个布尔矩阵(非空为True)，然后乘以权重，再求和
            total_weight = metric_df.notna().mul(weights, axis=1).sum(axis=1)
            final_df[f'industry_{metric}_D'] = (weighted_sum / total_weight.replace(0, np.nan)).fillna(0)
        # 5. 对分类型指标(lifecycle_stage)进行处理，生成数值化分数 (向量化优化)
        stage_df = df.pivot_table(index='trade_date', columns='concept_code', values='lifecycle_stage', aggfunc='first')
        # 创建与 stage_df 列对齐的权重Series
        aligned_weights = stage_df.columns.to_series().map(concept_weight_map)
        # 使用矩阵乘法 (.dot product) 高效计算每日活跃概念的总权重
        # stage_df.notna() 是一个布尔矩阵，.dot(aligned_weights) 等价于对每行中为True的列，查找其权重并求和
        daily_total_weight = stage_df.notna().dot(aligned_weights)
        # 为每个阶段计算加权置信度分数 (向量化)
        stages = ['PREHEAT', 'MARKUP', 'STAGNATION', 'DOWNTREND']
        for stage in stages:
            # 使用矩阵乘法高效计算每个阶段的加权和
            # (stage_df == stage) 是一个布尔矩阵，.dot(aligned_weights) 计算每日属于该阶段的板块的权重之和
            stage_weight_sum = (stage_df == stage).dot(aligned_weights)
            stage_score = (stage_weight_sum / daily_total_weight.replace(0, np.nan)).fillna(0)
            final_df[f'industry_{stage.lower()}_score_D'] = stage_score
        # print(f"    - [行业背景融合引擎 V1.1] 完成。已为 {stock_code} 生成 {len(final_df)} 天的数值化融合行业背景。")
        return final_df

    async def prepare_hot_money_signals(self, stock_code: str, start_date: datetime.date, end_date: datetime.date, params: dict) -> pd.DataFrame:
        """根据游资明细数据，生成一系列与日线数据对齐的原子信号。"""
        hm_df = await self.fund_flow_dao.get_hm_detail_data(start_date, end_date, stock_codes=[stock_code])
        if hm_df.empty: return pd.DataFrame()
        hm_df['trade_date'] = pd.to_datetime(hm_df['trade_date'], utc=True)
        daily_summary = {}
        any_buy_dates = hm_df[hm_df['net_amount'] > 0]['trade_date'].unique()
        daily_summary['HM_ACTIVE_ANY_D'] = pd.Series(True, index=any_buy_dates)
        top_tier_list = params.get('top_tier_list', [])
        top_tier_df = hm_df[hm_df['hm_name'].isin(top_tier_list)]
        top_tier_buy_dates = top_tier_df[top_tier_df['net_amount'] > 0]['trade_date'].unique()
        daily_summary['HM_ACTIVE_TOP_TIER_D'] = pd.Series(True, index=top_tier_buy_dates)
        coordination_threshold = params.get('coordination_threshold', 3)
        buyers_count_daily = hm_df[hm_df['net_amount'] > 0].groupby('trade_date')['hm_name'].nunique()
        coordinated_dates = buyers_count_daily[buyers_count_daily >= coordination_threshold].index
        daily_summary['HM_COORDINATED_ATTACK_D'] = pd.Series(True, index=coordinated_dates)
        return pd.DataFrame(daily_summary)

    async def prepare_market_sentiment_signals(self, stock_code: str, start_date: datetime.date, end_date: datetime.date, params: dict) -> pd.DataFrame:
        """市场情绪信号引擎"""
        tasks = {
            "limit_d": self.industry_dao.get_limit_list_d_for_range(start_date, end_date),
            "limit_step": self.industry_dao.get_limit_step_for_range(start_date, end_date),
            "limit_cpt": self.industry_dao.get_limit_cpt_list_for_range(start_date, end_date)
        }
        results = await asyncio.gather(*tasks.values())
        limit_d_df, limit_step_df, limit_cpt_df = results
        date_range_index = pd.date_range(start=start_date, end=end_date, freq='D', tz='UTC')
        signals_df = pd.DataFrame(index=date_range_index)
        stock_limit_df = limit_d_df[limit_d_df['stock_code'] == stock_code].set_index('trade_date')
        if not stock_limit_df.empty:
            signals_df['IS_LIMIT_UP_D'] = (stock_limit_df['limit'] == 'U')
            signals_df['CONSECUTIVE_LIMIT_UPS_D'] = stock_limit_df['limit_times']
            signals_df['IS_BAD_LIMIT_UP_D'] = (stock_limit_df['open_times'] > params.get('bad_limit_open_threshold', 3))
        if not limit_step_df.empty:
            limit_step_df = limit_step_df.set_index('trade_date')
            leader_dates = limit_step_df.loc[limit_step_df.groupby('trade_date')['nums'].idxmax()]
            is_leader_series = leader_dates[leader_dates['stock_code'] == stock_code].index
            signals_df.loc[is_leader_series, 'IS_MARKET_LEADER_D'] = True
        if not limit_d_df.empty:
            daily_stats = limit_d_df.groupby('trade_date')['limit'].value_counts().unstack(fill_value=0)
            daily_stats.rename(columns={'U': 'limit_up_count', 'D': 'limit_down_count'}, inplace=True)
            sentiment_score = np.log1p(daily_stats.get('limit_up_count', 0)) - np.log1p(daily_stats.get('limit_down_count', 0))
            signals_df['market_sentiment_score_D'] = sentiment_score
        stock_industries = await self.industry_dao.get_stock_ths_indices(stock_code)
        if stock_industries and not limit_cpt_df.empty:
            industry_codes = [ind.ts_code for ind in stock_industries]
            cpt_df_filtered = limit_cpt_df[limit_cpt_df['industry_code'].isin(industry_codes)]
            if not cpt_df_filtered.empty:
                hottest_industry_daily = cpt_df_filtered.loc[cpt_df_filtered.groupby('trade_date')['rank'].idxmin()]
                hottest_industry_daily = hottest_industry_daily.set_index('trade_date')
                signals_df['industry_hotness_rank_D'] = hottest_industry_daily['rank']
        return signals_df

    async def prepare_smart_money_signals(self, stock_code: str, start_date: date, end_date: date, params: dict) -> pd.DataFrame:
        """
        聪明钱信号引擎
        - 核心职责: 融合游资(HmDetail)和龙虎榜(TopList, TopInst)数据，生成协同与背离信号。
        """
        # print("    - [聪明钱引擎] 开始准备游资与机构协同信号...")
        # 1. 并发获取所有需要的原始数据
        tasks = {
            "hm_detail": self.fund_flow_dao.get_hm_detail_data(start_date, end_date, stock_codes=[stock_code]),
            "top_list": self.fund_flow_dao.get_top_list_data(start_date, end_date, stock_codes=[stock_code]),
            "top_inst": self.fund_flow_dao.get_top_inst_data(start_date, end_date, stock_codes=[stock_code])
        }
        results = await asyncio.gather(*tasks.values())
        hm_df, top_list_df, top_inst_df = results
        # 创建一个以日期为索引的空DataFrame用于存储所有信号
        date_range_index = pd.date_range(start=start_date, end=end_date, freq='D', tz='UTC')
        signals_df = pd.DataFrame(index=date_range_index)
        # 2. 处理游资信号 (Hot Money)
        if not hm_df.empty:
            hm_df['trade_date'] = pd.to_datetime(hm_df['trade_date'], utc=True)
            # 按天聚合游资行为
            hm_daily_summary = hm_df.groupby('trade_date').agg(
                hm_net_amount=('net_amount', 'sum'),
                hm_buyer_count=('hm_name', lambda x: x[hm_df.loc[x.index, 'net_amount'] > 0].nunique())
            )
            signals_df = signals_df.merge(hm_daily_summary, left_index=True, right_index=True, how='left')
            # 信号1: 游资净买入
            signals_df['SMART_MONEY_HM_NET_BUY_D'] = signals_df['hm_net_amount'] > 0
            # 信号2: 游资协同攻击
            coordination_threshold = params.get('hm_coordination_threshold', 3)
            signals_df['SMART_MONEY_HM_COORDINATED_ATTACK_D'] = signals_df['hm_buyer_count'] >= coordination_threshold
        # 3. 处理机构信号 (Institution)
        if not top_inst_df.empty:
            top_inst_df['trade_date'] = pd.to_datetime(top_inst_df['trade_date'], utc=True)
            # 按天聚合机构净买入额
            inst_daily_summary = top_inst_df.groupby('trade_date').agg(inst_net_buy=('net_buy', 'sum'))
            signals_df = signals_df.merge(inst_daily_summary, left_index=True, right_index=True, how='left')
            # 信号3: 机构净买入
            signals_df['SMART_MONEY_INST_NET_BUY_D'] = signals_df['inst_net_buy'] > 0
        # 4. 处理协同与背离信号
        # 信号4: 游资与机构协同买入 (最强看涨信号之一)
        signals_df['SMART_MONEY_SYNERGY_BUY_D'] = signals_df.get('SMART_MONEY_HM_NET_BUY_D', False) & signals_df.get('SMART_MONEY_INST_NET_BUY_D', False)
        # 信号5: 游资买、机构卖 (短期情绪高涨，但中期价值不被认可，潜在风险)
        signals_df['SMART_MONEY_DIVERGENCE_HM_BUY_INST_SELL_D'] = signals_df.get('SMART_MONEY_HM_NET_BUY_D', False) & ~signals_df.get('SMART_MONEY_INST_NET_BUY_D', False)
        # 清理辅助列，只保留最终的布尔信号
        final_cols = [col for col in signals_df.columns if col.startswith('SMART_MONEY_')]
        final_signals_df = signals_df[final_cols].fillna(False).astype(bool)
        # print(f"    - [聪明钱引擎] 信号生成完毕。")
        return final_signals_df

    async def analyze_kpl_theme_hotness(self, stock_code: str, start_date: date, end_date: date, params: dict) -> pd.DataFrame:
        """
        【V1.2 向量化优化版】KPL题材热度分析引擎
        - 修复: 解决了因 'trade_date' 列数据类型不匹配导致的 pd.merge 失败问题。
        - 优化: 改为使用 set_index 和 join 进行合并，更健壮、更高效。
        - 优化: 将 groupby.apply 重构为向量化计算，先计算单项得分再分组求和，提升效率。
        """
        # print(f"    - [KPL热度引擎] 开始为 {stock_code} 分析题材热度...")
        # 1. 获取股票在指定日期范围内所属的所有KPL题材
        stock_themes_df = await self.industry_dao.get_kpl_themes_for_stock(stock_code, start_date, end_date)
        if stock_themes_df.empty:
            print(f"    - [KPL热度引擎] {stock_code} 在指定日期内未归属任何KPL题材。")
            return pd.DataFrame()
        # 2. 获取这些题材在对应日期的热度指标 (涨停数, 排名上升数)
        all_theme_codes = stock_themes_df['concept_code'].unique().tolist()
        themes_hotness_df = await self.industry_dao.get_kpl_themes_hotness(all_theme_codes, start_date, end_date)
        if themes_hotness_df.empty:
            print(f"    - [KPL热度引擎] 未能获取到相关题材的热度数据。")
            return pd.DataFrame()
        # 3. 将 'trade_date' 和 'concept_code' 设置为索引
        try:
            stock_themes_df.set_index(['trade_date', 'concept_code'], inplace=True)
            themes_hotness_df.set_index(['trade_date', 'concept_code'], inplace=True)
        except KeyError as e:
            print(f"    - [KPL热度引擎-严重错误] set_index失败，列不存在: {e}")
            return pd.DataFrame()
        # 4. 使用 join (基于索引合并)
        merged_df = stock_themes_df.join(themes_hotness_df, how='left')
        # 6. 计算每日的综合热度分 (向量化优化)
        # 先计算每个题材的热度分，生成一个新列
        zt_weight = params.get('zt_num_weight', 0.7)
        up_weight = params.get('up_num_weight', 0.3)
        merged_df['daily_theme_hotness'] = (merged_df['z_t_num'].fillna(0) * zt_weight + 
                                            merged_df['up_num'].fillna(0) * up_weight)
        # 然后按日期分组对新列求和，这比 groupby.apply 更高效
        daily_hotness = merged_df.groupby(level='trade_date')['daily_theme_hotness'].sum()
        if daily_hotness.empty:
            return pd.DataFrame()
        # 7. 归一化和格式化输出
        max_score = params.get('max_score_clip', 10.0)
        normalized_score = (daily_hotness / max_score).clip(0, 1)
        result_df = pd.DataFrame(normalized_score, columns=['THEME_HOTNESS_SCORE_D'])
        # print(f"    - [KPL热度引擎] 完成分析，已生成题材热度分。")
        return result_df

    async def _process_single_industry_strength(self, concept: ConceptMaster, trade_date: datetime.date, market_daily_df: pd.DataFrame) -> Optional[Dict]:
        """
        【V3.2 深度分析版】处理单个板块/概念的强度计算。
        - 核心升级: 增加了内部广度(breadth_score)和龙头效应(leader_score)两个维度，更贴合A股市场。
        """
        start_date = trade_date - datetime.timedelta(days=self.momentum_lookback + 30)
        try:
            concept_daily_df = await self.industry_dao.get_concept_daily_for_range(concept.code, start_date, trade_date)
            if concept_daily_df.empty:
                return None
            momentum_score = self._calculate_momentum_score(concept_daily_df, trade_date)
            volume_score = await self._calculate_volume_profile_score(concept_daily_df)
            rs_score = await self._calculate_relative_strength_score(concept_daily_df, market_daily_df)
            # --- 计算内部结构分数 ---
            breadth_score = await self._calculate_internal_breadth_score(concept, trade_date)
            leader_score = await self._calculate_leader_effect_score(concept, trade_date)
            # --- 调整权重，加入新维度 ---
            total_score = (
                25 * momentum_score +   # 外部动量权重
                15 * volume_score +    # 成交活跃度权重
                30 * rs_score +        # 相对大盘强度权重
                15 * breadth_score +   # 内部上涨广度权重 (新增)
                15 * leader_score      # 龙头效应权重 (新增)
            )
            # --- 在返回结果中增加新维度的分数，供下游使用 ---
            return {
                'concept_code': concept.code,
                'concept_name': concept.name,
                'strength_score': total_score,
                'breadth_score': breadth_score,
                'leader_score': leader_score
            }
        except Exception as e:
            logger.error(f"处理板块 {concept.name} 时发生错误: {e}", exc_info=True)
            return None

    def _calculate_momentum_score(self, df: pd.DataFrame, trade_date: datetime.date) -> float:
        """计算动量分"""
        if df.empty or trade_date not in df.index: return 0.0
        df['ema20'] = df['close'].ewm(span=20, adjust=False).mean()
        df['ema60'] = df['close'].ewm(span=60, adjust=False).mean()
        today = df.loc[trade_date]
        price_above_ema20 = 1 if today['close'] > today['ema20'] else 0
        price_above_ema60 = 1 if today['close'] > today['ema60'] else 0
        ema_bullish = 1 if today['ema20'] > today['ema60'] else 0
        pct_change_5d = (today['close'] / df['close'].shift(5).loc[trade_date]) - 1 if len(df) > 5 else 0
        return (price_above_ema20 * 2 + price_above_ema60 * 1 + ema_bullish * 2 + (pct_change_5d * 10))

    def _calculate_fund_flow_score(self, df: pd.DataFrame, trade_date: datetime.date) -> float:
        """计算资金流分"""
        if df.empty or trade_date not in df.index: return 0.0
        recent_df = df.loc[:trade_date].tail(self.fund_flow_lookback)
        if recent_df.empty: return 0.0
        net_inflow_sum = recent_df['net_amount'].sum()
        inflow_days_ratio = (recent_df['net_amount'] > 0).sum() / len(recent_df)
        return (net_inflow_sum * 0.1 + inflow_days_ratio * 5)

    async def _calculate_volume_profile_score(self, industry_daily_df: pd.DataFrame) -> float:
        """计算行业成交活跃度得分。"""
        if industry_daily_df.empty or 'turnover_rate' not in industry_daily_df.columns or len(industry_daily_df) < 5:
            return 0.0
        df = industry_daily_df.copy()
        turnover_rank_60d = df['turnover_rate'].rolling(60, min_periods=20).rank(pct=True).iloc[-1]
        df['turnover_ma5'] = df['turnover_rate'].rolling(5, min_periods=1).mean()
        df['turnover_ma20'] = df['turnover_rate'].rolling(20, min_periods=1).mean()
        was_below = df['turnover_ma5'].shift(1) < df['turnover_ma20'].shift(1)
        is_above = df['turnover_ma5'] > df['turnover_ma20']
        is_recent_cross = (was_below & is_above).rolling(3, min_periods=1).sum().iloc[-1] > 0
        score = 0.0
        if pd.notna(turnover_rank_60d) and turnover_rank_60d > 0.9: score += 0.6
        if is_recent_cross: score += 0.4
        return score

    async def _calculate_relative_strength_score(self, industry_daily_df: pd.DataFrame, market_daily_df: pd.DataFrame) -> float:
        """计算行业相对大盘的强度得分。"""
        if industry_daily_df.empty or market_daily_df.empty: return 0.0
        df = pd.merge(industry_daily_df[['close']], market_daily_df, left_index=True, right_index=True, how='inner')
        if df.empty: return 0.0
        df['rs'] = df['close'] / df['market_close']
        df['rs_ma20'] = df['rs'].rolling(20).mean()
        latest = df.iloc[-1]
        return 1.0 if latest['rs'] > latest['rs_ma20'] else 0.0

    def _calculate_sentiment_hotness_score(self, sentiment_data: Optional[Dict]) -> float:
        """根据最强板块统计数据，计算板块的情绪热度分。"""
        if not sentiment_data: return 0.0
        score = 0.0
        rank = sentiment_data.get('rank', 100)
        if rank <= 5: score += 0.5
        elif rank <= 20: score += 0.3
        elif rank <= 50: score += 0.1
        cons_nums = sentiment_data.get('cons_nums', 0)
        if cons_nums >= 3: score += 0.3
        elif cons_nums >= 1: score += 0.15
        up_nums = sentiment_data.get('up_nums', 0)
        if up_nums >= 5: score += 0.2
        elif up_nums >= 2: score += 0.1
        return min(score, 1.0)

    async def _calculate_internal_breadth_score(self, concept: ConceptMaster, trade_date: datetime.date) -> float:
        """
        计算行业内部上涨广度得分。
        - 核心逻辑: 统计板块内上涨家数占比，量化板块的普涨程度。这是判断行业上涨健康度的关键。
        - 返回值: 0-1之间的分数，越高代表上涨广度越好。
        """
        # print(f"      - [广度分析] 开始计算板块 '{concept.name}' 在 {trade_date} 的内部广度...")
        # 1. 获取指定日期的板块成分股
        members = await self.industry_dao.get_concept_members_on_date(concept.code, trade_date)
        if not members:
            print(f"      - [广度分析] 未找到板块 '{concept.name}' 在 {trade_date} 的成分股。")
            return 0.0
        member_codes = [m.stock.stock_code for m in members]
        # 2. 批量获取成分股当日行情
        # 假设 stock_trade_dao 中有 get_stocks_daily_for_date 方法
        daily_data_df = await self.stock_trade_dao.get_stocks_daily_for_date(member_codes, trade_date)
        if daily_data_df.empty:
            print(f"      - [广度分析] 未能获取板块 '{concept.name}' 成分股的日线数据。")
            return 0.0
        # 3. 计算上涨家数占比
        up_count = (daily_data_df['pct_change'] > 0).sum()
        total_count = len(daily_data_df)
        breadth_ratio = up_count / total_count if total_count > 0 else 0.0
        # 4. 将广度占比（0-1）直接作为分数返回，更具解释性
        score = breadth_ratio
        print(f"      - [广度分析] 板块 '{concept.name}' 上涨家数/总数: {up_count}/{total_count}, 广度得分: {score:.2f}")
        return score

    async def _calculate_leader_effect_score(self, concept: ConceptMaster, trade_date: datetime.date) -> float:
        """
        计算龙头效应得分。
        - 核心逻辑: 检测板块内是否有涨停股，特别是连板股，作为龙头效应的标志。这是A股市场极强的信号。
        - 返回值: 0-1之间的分数，越高代表龙头效应越强。
        """
        print(f"      - [龙头分析] 开始检测板块 '{concept.name}' 在 {trade_date} 的龙头效应...")
        # 1. 获取指定日期的板块成分股
        members = await self.industry_dao.get_concept_members_on_date(concept.code, trade_date)
        if not members:
            return 0.0
        member_codes = [m.stock.stock_code for m in members]
        # 2. 从 KplLimitList 中查询成分股的涨停信息
        limit_up_df = await self.industry_dao.get_limit_list_for_stocks(member_codes, trade_date, tag='涨停')
        if limit_up_df.empty:
            print(f"      - [龙头分析] 板块 '{concept.name}' 内无涨停股。")
            return 0.0
        # 3. 根据涨停数量和质量进行评分
        score = 0.0
        limit_up_count = len(limit_up_df)
        # 检查是否存在连板股 ('2连板', '3连板'...)
        consecutive_boards_df = limit_up_df[limit_up_df['status'].str.contains('连板', na=False)]
        if not consecutive_boards_df.empty:
            score += 0.6  # 存在连板股，是强烈的龙头信号
        elif limit_up_count > 0:
            score += 0.3  # 存在首板股，是潜在的启动信号
        # 根据涨停家数形成梯队效应加分
        if limit_up_count >= 3:
            score += 0.4  # 形成涨停梯队，板块效应强
        elif limit_up_count >= 2:
            score += 0.2
        final_score = min(score, 1.0)
        print(f"      - [龙头分析] 板块 '{concept.name}' 发现 {limit_up_count} 家涨停股, 龙头效应得分: {final_score:.2f}")
        return final_score












