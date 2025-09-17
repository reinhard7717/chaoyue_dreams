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

logger = logging.getLogger("services")

class ContextualAnalysisService:
    """
    【新增】上下文分析服务
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

    async def analyze_industry_rotation(self, end_date: datetime.date, lookback_days: int = 21, market_code: str = '000300.SH') -> pd.DataFrame:
        """
        【V2.0 生命周期诊断版】分析行业轮动，并诊断每个行业的生命周期阶段。
        """
        print(f"\n--- [行业生命周期诊断 V2.0] 开始分析截至 {end_date} 的过去 {lookback_days} 天行业轮动情况 ---")
        trade_dates = [end_date - datetime.timedelta(days=i) for i in range(lookback_days)][::-1]
        tasks = [self.calculate_industry_strength_rank(td, market_code) for td in trade_dates]
        daily_rank_results = await asyncio.gather(*tasks)
        all_ranks = []
        for i, df_rank in enumerate(daily_rank_results):
            if not df_rank.empty and 'strength_rank' in df_rank.columns:
                df_rank['trade_date'] = trade_dates[i]
                all_ranks.append(df_rank.reset_index())
        if not all_ranks:
            logger.warning("未能获取任何历史排名数据，行业轮动分析中止。")
            return pd.DataFrame()
        rotation_df = pd.concat(all_ranks, ignore_index=True)
        
        def calculate_lifecycle_metrics(group):
            group = group.sort_values('trade_date')
            if len(group) < 5:
                return pd.Series({'latest_rank': group['strength_rank'].iloc[-1], 'rank_slope': 0.0, 'rank_accel': 0.0})
            ranks = group['strength_rank'].values
            slope = np.polyfit(np.arange(min(5, len(ranks))), ranks[-5:], 1)[0] if len(ranks) >= 2 else 0.0
            accel = 0.0
            if len(group) >= 10:
                slope_1 = np.polyfit(np.arange(5), ranks[-10:-5], 1)[0]
                slope_2 = np.polyfit(np.arange(5), ranks[-5:], 1)[0]
                accel = slope_2 - slope_1
            return pd.Series({'latest_rank': ranks[-1], 'rank_slope': slope, 'rank_accel': accel})
            
        lifecycle_metrics = rotation_df.groupby('industry_code').apply(calculate_lifecycle_metrics)
        
        def assign_lifecycle_stage(row):
            if row['latest_rank'] < 0.3 and row['rank_slope'] > 0.005 and row['rank_accel'] > 0: return 'PREHEAT'
            if row['latest_rank'] > 0.5 and row['rank_slope'] > 0.01: return 'MARKUP'
            if row['latest_rank'] > 0.8 and row['rank_slope'] < 0: return 'STAGNATION'
            if row['latest_rank'] < 0.4 and row['rank_slope'] < -0.005: return 'DOWNTREND'
            return 'TRANSITION'
            
        lifecycle_metrics['lifecycle_stage'] = lifecycle_metrics.apply(assign_lifecycle_stage, axis=1)
        industry_names = rotation_df[['industry_code', 'industry_name']].drop_duplicates().set_index('industry_code')
        final_report = pd.merge(lifecycle_metrics, industry_names, left_index=True, right_index=True)
        print("--- [行业生命周期诊断 V2.0] 分析完成 ---")
        return final_report.sort_values('rank_slope', ascending=False)

    async def calculate_industry_strength_rank(self, trade_date: datetime.date, market_code: str = '000905.SH') -> pd.DataFrame:
        """
        【V2.1 结构分析版】计算指定交易日所有行业的强度分及排名。
        """
        start_date = trade_date - datetime.timedelta(days=self.momentum_lookback + 30)
        market_daily_df = await self.indicator_dao.get_market_index_daily_data(market_code, start_date, trade_date)
        if market_daily_df.empty:
            logger.warning(f"无法获取大盘基准 {market_code} 数据，相对强度分析将跳过。")
        
        all_industries = await self.industry_dao.get_ths_index_list() # 假设使用同花顺行业
        if not all_industries:
            logger.warning("未找到任何行业，计算中止。")
            return pd.DataFrame()

        tasks = [self._process_single_industry_strength(industry, trade_date, market_daily_df) for industry in all_industries]
        results = await asyncio.gather(*tasks)
        
        strength_data = [res for res in results if res is not None]
        if not strength_data:
            return pd.DataFrame()

        df = pd.DataFrame(strength_data)
        df['strength_rank'] = df['strength_score'].rank(pct=True, ascending=True)
        return df.sort_values('strength_rank', ascending=False).set_index('industry_code')

    async def _process_single_industry_strength(self, industry, trade_date: datetime.date, market_daily_df: pd.DataFrame) -> Optional[Dict]:
        """
        【V3.0 双引擎版】处理单个行业的强度计算，便于并行化。
        """
        start_date = trade_date - datetime.timedelta(days=self.momentum_lookback + 30)
        try:
            tasks = {
                "daily": self.industry_dao.get_ths_index_daily_for_range(industry.ts_code, start_date, trade_date),
                "flow": self.fund_flow_dao.get_industry_fund_flow(industry.ts_code, start_date, trade_date),
                "sentiment": self.industry_dao.get_limit_cpt_list_for_industry_and_date(industry.ts_code, trade_date)
            }
            results = await asyncio.gather(*tasks.values())
            industry_daily_df, industry_fund_flow_df, industry_sentiment_data = results

            if industry_daily_df.empty:
                return None

            momentum_score = self._calculate_momentum_score(industry_daily_df, trade_date)
            fund_flow_score = self._calculate_fund_flow_score(industry_fund_flow_df, trade_date)
            volume_score = await self._calculate_volume_profile_score(industry_daily_df)
            rs_score = await self._calculate_relative_strength_score(industry_daily_df, market_daily_df)
            sentiment_hotness_score = self._calculate_sentiment_hotness_score(industry_sentiment_data)

            total_score = (
                15 * momentum_score + 15 * fund_flow_score + 10 * volume_score + 
                15 * rs_score + 45 * sentiment_hotness_score
            )
            return {'industry_code': industry.ts_code, 'industry_name': industry.name, 'strength_score': total_score}
        except Exception as e:
            logger.error(f"处理行业 {industry.name} 时发生错误: {e}", exc_info=True)
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

    async def prepare_smart_money_signals(self, stock_code: str, start_date: date, end_date: date, params: dict) -> pd.DataFrame: # 新增方法
        """
        【新增】聪明钱信号引擎
        - 核心职责: 融合游资(HmDetail)和龙虎榜(TopList, TopInst)数据，生成协同与背离信号。
        """
        print("    - [聪明钱引擎] 开始准备游资与机构协同信号...")
        
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

        print(f"    - [聪明钱引擎] 信号生成完毕。")
        return final_signals_df















