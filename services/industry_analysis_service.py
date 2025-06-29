import os
import sys
import json
import django
from datetime import date, timedelta
import pandas as pd
import numpy as np

from stock_models.industry import ThsIndex, ThsIndexMember
from stock_models.time_trade import StockDailyData
# 配置Django环境
# 假设你的项目根目录在当前文件的上两级
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'chaoyue_dreams.settings') # 修改 'your_project_name'
django.setup()

# 注意：我假设你有一个 StockDaily 模型来存储个股的日线数据，这是量化系统标配。
# 如果没有，你需要根据你的实际模型调整，比如从 FundFlowDailyTHS 获取 pct_change。
# 同时，StockDaily 中需要有 is_limit_up 和 consecutive_limit_ups 字段用于涨停分析。
class IndustryAnalysisService:
    """
    行业与板块分析服务
    封装了人气龙头、板块协同性、涨停梯队等高级分析模型。
    """
    def __init__(self, config_path='config/industry_analysis.json'):
        """
        初始化服务，加载配置文件。
        :param config_path: 配置文件的路径。
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        self.sentiment_params = self.config.get('sentiment_leader_params', {})
        self.cohesion_params = self.config.get('cohesion_analysis_params', {})
        self.echelon_params = self.config.get('limit_up_echelon_params', {})
        self.debug = self.config.get('debug_params', {}).get('verbose_logging', False)
        print("IndustryAnalysisService 初始化成功，配置已加载。")

    def _fetch_data_for_analysis(self, trade_date: date):
        """
        为分析准备所有需要的基础数据。
        """
        print(f"[{trade_date}] 开始获取基础数据...")
        # 1. 获取所有需要分析的同花顺行业/概念板块
        all_indices = ThsIndex.objects.filter(type__in=['行业', '概念']).values('ts_code', 'name')
        if not all_indices:
            print("警告：数据库中未找到任何同花顺板块信息。")
            return None, None
        indices_df = pd.DataFrame(list(all_indices))
        print(f"共找到 {len(indices_df)} 个行业/概念板块。")
        # 2. 获取所有板块的成分股
        members = ThsIndexMember.objects.select_related('ths_index', 'stock').filter(
            ths_index__ts_code__in=indices_df['ts_code'].tolist()
        ).values('ths_index__ts_code', 'stock__stock_code')
        if not members:
            print("警告：数据库中未找到任何板块成分股信息。")
            return None, None
        members_df = pd.DataFrame(list(members)).rename(columns={
            'ths_index__ts_code': 'ths_code',
            'stock__stock_code': 'stock_code'
        })
        print(f"共找到 {len(members_df)} 条成分股关系。")
        # 3. 获取所有成分股在当天的日线数据
        # 假设 StockDaily 包含: stock_code, trade_time, pct_change, turnover_rate, is_limit_up, consecutive_limit_ups
        stock_codes = members_df['stock_code'].unique().tolist()
        daily_data = StockDailyData.objects.filter(
            stock__stock_code__in=stock_codes,
            trade_time=trade_date
        ).values(
            'stock__stock_code', 'pct_change', 'turnover_rate',
            'is_limit_up', 'consecutive_limit_ups' # 涨停分析所需字段
        )
        if not daily_data:
            print(f"警告：在 {trade_date} 未找到任何成分股的日线数据。")
            return None, None
        daily_df = pd.DataFrame(list(daily_data)).rename(columns={'stock__stock_code': 'stock_code'})
        # 合并数据，形成一个包含[板块代码, 股票代码, 日线数据]的大表
        analysis_df = pd.merge(members_df, daily_df, on='stock_code', how='inner')
        print(f"数据获取与合并完成，共 {len(analysis_df)} 条记录用于分析。")
        return indices_df, analysis_df

    def _calculate_sentiment_leaders(self, trade_date: date, members_df: pd.DataFrame):
        """
        计算人气龙头。这是一个复杂操作，因为它需要历史数据。
        为简化演示，我们这里用当日的涨幅和换手率作为人气的代理指标。
        一个更完整的实现需要查询 lookback_period 的数据。
        """
        if not self.sentiment_params.get('enabled'):
            return pd.DataFrame()
        print(f"[{trade_date}] 开始计算人气龙头...")
        # 简化版：使用当日数据计算人气分
        # 完整版应查询 lookback_period 天数的数据，计算累计涨幅和平均换手率
        lookback_days = self.sentiment_params.get('lookback_period', 10)
        start_date = trade_date - timedelta(days=lookback_days * 1.5) # 乘以1.5以覆盖非交易日
        hist_data = StockDailyData.objects.filter(
            stock__stock_code__in=members_df['stock_code'].unique(),
            trade_time__range=[start_date, trade_date]
        ).values('stock__stock_code', 'trade_time', 'pct_change', 'turnover_rate')
        if not hist_data:
            print("警告：无法获取用于计算人气龙头的历史数据。")
            return pd.DataFrame()
        hist_df = pd.DataFrame(list(hist_data)).rename(columns={'stock__stock_code': 'stock_code'})
        # 计算累计涨幅和平均换手率
        def calculate_metrics(group):
            group = group.sort_values('trade_time').tail(lookback_days)
            # 修复：确保 pct_change 是浮点数
            group['pct_change'] = pd.to_numeric(group['pct_change'], errors='coerce')
            # 修复：处理涨幅为0的情况，避免prod结果为0
            cumulative_pct_change = (1 + group['pct_change'] / 100).prod() - 1
            avg_turnover_rate = pd.to_numeric(group['turnover_rate'], errors='coerce').mean()
            return pd.Series({
                'cumulative_pct_change': cumulative_pct_change,
                'avg_turnover_rate': avg_turnover_rate
            })
        
        # 修改：按 stock_code 分组计算指标
        popularity_metrics = hist_df.groupby('stock_code').apply(calculate_metrics).reset_index()
        
        # 修改：将计算出的指标与当日的成员数据合并
        analysis_df = pd.merge(members_df, popularity_metrics, on='stock_code', how='inner')
        
        # 计算人气分
        analysis_df['rank_pct_change'] = analysis_df.groupby('ths_code')['cumulative_pct_change'].rank(pct=True)
        analysis_df['rank_turnover'] = analysis_df.groupby('ths_code')['avg_turnover_rate'].rank(pct=True)
        analysis_df['sentiment_score'] = analysis_df['rank_pct_change'] * 0.6 + analysis_df['rank_turnover'] * 0.4
        
        # 识别龙头
        analysis_df = analysis_df.sort_values('sentiment_score', ascending=False)
        leaders = analysis_df.groupby('ths_code').head(self.sentiment_params.get('num_leaders_to_track', 3))
        
        # 检查龙头是否确认
        # 需要合并当日涨幅数据
        today_df = hist_df[hist_df['trade_time'] == trade_date][['stock_code', 'pct_change']]
        leaders = pd.merge(leaders, today_df, on='stock_code', how='left')
        
        leader_threshold = self.sentiment_params.get('leader_pct_change_threshold', 5.0)
        leaders['is_confirmed'] = leaders['pct_change'] > leader_threshold
        
        # 统计结果
        result = leaders.groupby('ths_code').agg(
            confirmed_leaders_count=('is_confirmed', 'sum'),
            leaders_list=('stock_code', lambda x: list(x))
        ).reset_index()
        
        min_confirming = self.sentiment_params.get('min_leaders_confirming', 1)
        result['has_leader_effect'] = result['confirmed_leaders_count'] >= min_confirming
        
        if self.debug:
            print("人气龙头分析结果(部分):")
            print(result[result['has_leader_effect']].head())
        return result

    def _calculate_cohesion(self, analysis_df: pd.DataFrame):
        """
        计算板块协同性。
        """
        if not self.cohesion_params.get('enabled'):
            return pd.DataFrame()
        print("开始计算板块协同性...")
        # 修复：确保 pct_change 是数值类型
        analysis_df['pct_change'] = pd.to_numeric(analysis_df['pct_change'], errors='coerce')
        
        # 按板块分组聚合
        cohesion_stats = analysis_df.groupby('ths_code').agg(
            total_count=('stock_code', 'size'),
            rising_count=('pct_change', lambda x: (x > 0).sum()),
            strong_rising_count=('pct_change', lambda x: (x > self.cohesion_params.get('strong_rising_threshold', 0.05) * 100).sum())
        ).reset_index()
        # 计算上涨比例
        cohesion_stats['rising_pct'] = cohesion_stats['rising_count'] / cohesion_stats['total_count']
        # 判断是否满足协同性条件
        min_rising_pct = self.cohesion_params.get('min_rising_pct_threshold', 0.6)
        min_strong_count = self.cohesion_params.get('min_strong_rising_count', 3)
        cohesion_stats['has_cohesion'] = (cohesion_stats['rising_pct'] >= min_rising_pct) & \
                                         (cohesion_stats['strong_rising_count'] >= min_strong_count)
        if self.debug:
            print("板块协同性分析结果(部分):")
            print(cohesion_stats[cohesion_stats['has_cohesion']].head())
        return cohesion_stats

    def _calculate_limit_up_echelon(self, analysis_df: pd.DataFrame):
        """
        计算涨停梯队效应。
        """
        if not self.echelon_params.get('enabled'):
            return pd.DataFrame()
        print("开始计算涨停梯队...")
        # 筛选出涨停的股票
        limit_up_df = analysis_df[analysis_df['is_limit_up'] == True].copy()
        if limit_up_df.empty:
            print("今日无涨停股票，跳过涨停梯队分析。")
            return pd.DataFrame(columns=['ths_code', 'limit_up_count', 'highest_board', 'has_echelon'])
        # 按板块分组聚合
        echelon_stats = limit_up_df.groupby('ths_code').agg(
            limit_up_count=('stock_code', 'size'),
            highest_board=('consecutive_limit_ups', 'max')
        ).reset_index()
        # 判断是否满足梯队条件
        min_limit_up = self.echelon_params.get('min_limit_up_count', 3)
        require_leader = self.echelon_params.get('require_consecutive_leader', True)
        
        conditions = (echelon_stats['limit_up_count'] >= min_limit_up)
        if require_leader:
            conditions &= (echelon_stats['highest_board'] >= 2)
        
        echelon_stats['has_echelon'] = conditions
        if self.debug:
            print("涨停梯队分析结果(部分):")
            print(echelon_stats[echelon_stats['has_echelon']].head())
        return echelon_stats
 
    def analyze(self, trade_date: date):
        """
        执行所有行业分析，并返回最终的评分结果。
        :param trade_date: 需要分析的交易日期。
        :return: 一个DataFrame，包含每个板块的分析结果和最终得分。
        """
        print(f"\n{'='*20} 开始对 {trade_date} 进行全面行业分析 {'='*20}")
        indices_df, analysis_df = self._fetch_data_for_analysis(trade_date)
        if analysis_df is None or analysis_df.empty:
            print(f"在 {trade_date} 无数据可供分析，操作终止。")
            return pd.DataFrame()
        
        # 执行各项分析
        # 修改：传入 members_df 到 _calculate_sentiment_leaders
        members_df = analysis_df[['ths_code', 'stock_code']].drop_duplicates()
        sentiment_results = self._calculate_sentiment_leaders(trade_date, members_df)
        cohesion_results = self._calculate_cohesion(analysis_df)
        echelon_results = self._calculate_limit_up_echelon(analysis_df)
        
        # 合并所有分析结果
        final_results = indices_df.copy()
        if not sentiment_results.empty:
            final_results = pd.merge(final_results, sentiment_results, left_on='ts_code', right_on='ths_code', how='left')
        if not cohesion_results.empty:
            final_results = pd.merge(final_results, cohesion_results, left_on='ts_code', right_on='ths_code', how='left')
        if not echelon_results.empty:
            final_results = pd.merge(final_results, echelon_results, left_on='ts_code', right_on='ths_code', how='left')
        
        # 清理合并产生的重复列和NaN值
        final_results = final_results.loc[:, ~final_results.columns.duplicated()]
        # 修改：使用 .fillna(False) for boolean columns and 0 for numeric columns
        bool_cols = ['has_leader_effect', 'has_cohesion', 'has_echelon']
        for col in bool_cols:
            if col in final_results.columns:
                final_results[col] = final_results[col].fillna(False)
        
        numeric_cols = ['confirmed_leaders_count', 'rising_pct', 'strong_rising_count', 'limit_up_count', 'highest_board']
        for col in numeric_cols:
            if col in final_results.columns:
                final_results[col] = final_results[col].fillna(0)
        
        # 计算最终得分
        final_results['score'] = 0
        if self.sentiment_params.get('enabled'):
            bonus = self.sentiment_params.get('confirmation_bonus', 0)
            final_results['score'] += final_results['has_leader_effect'] * bonus
        if self.cohesion_params.get('enabled'):
            bonus = self.cohesion_params.get('cohesion_bonus', 0)
            final_results['score'] += final_results['has_cohesion'] * bonus
        if self.echelon_params.get('enabled'):
            bonus = self.echelon_params.get('echelon_bonus', 0)
            final_results['score'] += final_results['has_echelon'] * bonus
        
        # 增加可读性
        final_results = final_results.rename(columns={'ts_code': 'ths_code', 'name': 'ths_name'})
        
        # 筛选并排序
        hot_industries = final_results[final_results['score'] > 0].sort_values('score', ascending=False)
        
        print(f"\n{'='*20} {trade_date} 行业分析完成 {'='*20}")
        if self.debug:
            print("热门板块最终得分榜:")
            print(hot_industries[['ths_code', 'ths_name', 'score', 'has_leader_effect', 'has_cohesion', 'has_echelon']].to_string())
            
        return hot_industries

