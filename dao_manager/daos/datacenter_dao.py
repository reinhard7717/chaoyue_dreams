# dao/datacenter_dao.py
import logging
from datetime import datetime
from typing import List, Dict, Optional, Union, Any
from django.db import transaction
from django.db.models import Q

from models.datacenter import *
from api_manager.mappings.datacenter_mappings import *
from models.datacenter.capital_flow import ConceptCapitalFlow, IndustryCapitalFlow, StockCapitalFlow
from models.datacenter.financial import CircMarketValueRank, MonthlyRankChange, MonthlyStrongStock, PBRatioRank, PERatioRank, ROERank, WeeklyRankChange, WeeklyStrongStock
from models.datacenter.institution import FundHeavyPosition, InstitutionHoldingSummary, QFIIHeavyPosition, SocialSecurityHeavyPosition
from models.datacenter.lhb import BrokerOnList, InstitutionTradeDetail, InstitutionTradeTrack, LhbDaily, StockOnList
from models.datacenter.market_data import ContinuousFall, ContinuousRise, ContinuousVolumeDecrease, ContinuousVolumeIncrease, VolumeDecrease, VolumeIncrease
from models.datacenter.north_south import NorthFundTrend, NorthSouthFundOverview, NorthStockHolding, SouthFundTrend
from models.datacenter.statistics import NewHighStock, NewLowStock, StageHighLow

logger = logging.getLogger('dao')

class DataCenterDAO:
    """数据中心DAO"""
    
    @staticmethod
    def _batch_process(model_class, data_list, mapping, unique_fields, **extra_fields):
        """
        批量处理数据（检查-创建-更新-略过）
        
        Args:
            model_class: 模型类
            data_list: 要处理的数据列表
            mapping: 字段映射
            unique_fields: 用于确定唯一记录的字段列表
            extra_fields: 额外需要添加的字段
            
        Returns:
            dict: 包含创建、更新和略过的记录数
        """
        if not data_list:
            return {'created': 0, 'updated': 0, 'skipped': 0}
        
        # 统计计数
        created_count = 0
        updated_count = 0
        skipped_count = 0
        
        # 获取模型字段列表
        model_fields = get_model_fields(model_class)
        
        # 批量处理，分组进行以减小事务范围
        batch_size = 100
        for i in range(0, len(data_list), batch_size):
            batch = data_list[i:i+batch_size]
            
            with transaction.atomic():
                for data in batch:
                    # 构建筛选条件
                    filter_kwargs = {}
                    for field in unique_fields:
                        if field in data:
                            filter_kwargs[field] = data.get(field)
                    
                    # 检查是否已存在
                    existing = model_class.objects.filter(**filter_kwargs).first()
                    
                    # 准备要保存的数据
                    save_data = {}
                    for api_field, model_field in mapping.items():
                        if api_field in data and model_field in model_fields:
                            save_data[model_field] = data.get(api_field)
                    
                    # 添加额外字段
                    save_data.update(extra_fields)
                    
                    if existing:
                        # 检查数据是否有变化
                        has_changes = False
                        for field, value in save_data.items():
                            if getattr(existing, field) != value:
                                has_changes = True
                                break
                        
                        if has_changes:
                            # 更新已存在的记录
                            for field, value in save_data.items():
                                setattr(existing, field, value)
                            existing.save()
                            updated_count += 1
                        else:
                            # 数据相同，略过
                            skipped_count += 1
                    else:
                        # 创建新记录
                        model_class.objects.create(**save_data)
                        created_count += 1
        
        return {
            'created': created_count,
            'updated': updated_count,
            'skipped': skipped_count
        }
    
    # 龙虎榜相关DAO方法
    
    @staticmethod
    def save_daily_lhb(data: Dict) -> Dict:
        """
        保存今日龙虎榜概览
        
        Args:
            data: 龙虎榜数据
            
        Returns:
            dict: 操作结果统计
        """
        if not data:
            return {'created': 0, 'updated': 0, 'skipped': 0}
        
        # 提取日期，通常为当天
        date_str = data.get('t')
        if not date_str:
            date_str = datetime.now().strftime('%Y-%m-%d')
        
        # 保存数据
        return DataCenterDAO._batch_process(
            model_class=LhbDaily,
            data_list=[data],
            mapping=LHB_DAILY_MAPPING,
            unique_fields=['t']
        )
    
    @staticmethod
    def save_stock_on_list(data_list: List[Dict], days: int) -> Dict:
        """
        保存近n日上榜个股
        
        Args:
            data_list: 个股上榜统计数据列表
            days: 统计天数，可选 5、10、30、60
            
        Returns:
            dict: 操作结果统计
        """
        return DataCenterDAO._batch_process(
            model_class=StockOnList,
            data_list=data_list,
            mapping=STOCK_ON_LIST_MAPPING,
            unique_fields=['dm', 'days'],
            days=days,
            update_time=datetime.now()
        )
    
    @staticmethod
    def save_broker_on_list(data_list: List[Dict], days: int) -> Dict:
        """
        保存近n日营业部上榜统计
        
        Args:
            data_list: 营业部上榜统计数据列表
            days: 统计天数，可选 5、10、30、60
            
        Returns:
            dict: 操作结果统计
        """
        return DataCenterDAO._batch_process(
            model_class=BrokerOnList,
            data_list=data_list,
            mapping=BROKER_ON_LIST_MAPPING,
            unique_fields=['yybmc', 'days'],
            days=days,
            update_time=datetime.now()
        )
    
    @staticmethod
    def save_institution_trade_track(data_list: List[Dict], days: int) -> Dict:
        """
        保存近n日个股机构交易追踪
        
        Args:
            data_list: 机构席位追踪数据列表
            days: 统计天数，可选 5、10、30、60
            
        Returns:
            dict: 操作结果统计
        """
        return DataCenterDAO._batch_process(
            model_class=InstitutionTradeTrack,
            data_list=data_list,
            mapping=INSTITUTION_TRADE_TRACK_MAPPING,
            unique_fields=['dm', 'days'],
            days=days,
            update_time=datetime.now()
        )
    
    @staticmethod
    def save_institution_trade_detail(data_list: List[Dict]) -> Dict:
        """
        保存机构席位成交明细
        
        Args:
            data_list: 机构席位成交明细数据列表
            
        Returns:
            dict: 操作结果统计
        """
        return DataCenterDAO._batch_process(
            model_class=InstitutionTradeDetail,
            data_list=data_list,
            mapping=INSTITUTION_TRADE_DETAIL_MAPPING,
            unique_fields=['dm', 't', 'type']
        )
    
    # 个股统计相关DAO方法
    
    @staticmethod
    def save_stage_high_low(data_list: List[Dict]) -> Dict:
        """
        保存阶段最高最低数据
        
        Args:
            data_list: 阶段最高最低数据列表
            
        Returns:
            dict: 操作结果统计
        """
        return DataCenterDAO._batch_process(
            model_class=StageHighLow,
            data_list=data_list,
            mapping=STAGE_HIGH_LOW_MAPPING,
            unique_fields=['dm', 't']
        )
    
    @staticmethod
    def save_new_high_stocks(data_list: List[Dict]) -> Dict:
        """
        保存盘中创新高个股数据
        
        Args:
            data_list: 盘中创新高个股数据列表
            
        Returns:
            dict: 操作结果统计
        """
        return DataCenterDAO._batch_process(
            model_class=NewHighStock,
            data_list=data_list,
            mapping=NEW_HIGH_STOCK_MAPPING,
            unique_fields=['dm', 't']
        )
    
    @staticmethod
    def save_new_low_stocks(data_list: List[Dict]) -> Dict:
        """
        保存盘中创新低个股数据
        
        Args:
            data_list: 盘中创新低个股数据列表
            
        Returns:
            dict: 操作结果统计
        """
        return DataCenterDAO._batch_process(
            model_class=NewLowStock,
            data_list=data_list,
            mapping=NEW_LOW_STOCK_MAPPING,
            unique_fields=['dm', 't']
        )
    
    # 盘中数据和连续交易相关DAO方法
    
    @staticmethod
    def save_volume_increase(data_list: List[Dict]) -> Dict:
        """
        保存成交骤增个股数据
        
        Args:
            data_list: 成交骤增个股数据列表
            
        Returns:
            dict: 操作结果统计
        """
        return DataCenterDAO._batch_process(
            model_class=VolumeIncrease,
            data_list=data_list,
            mapping=VOLUME_INCREASE_MAPPING,
            unique_fields=['dm', 't']
        )
    
    @staticmethod
    def save_volume_decrease(data_list: List[Dict]) -> Dict:
        """
        保存成交骤减个股数据
        
        Args:
            data_list: 成交骤减个股数据列表
            
        Returns:
            dict: 操作结果统计
        """
        return DataCenterDAO._batch_process(
            model_class=VolumeDecrease,
            data_list=data_list,
            mapping=VOLUME_DECREASE_MAPPING,
            unique_fields=['dm', 't']
        )
    
    @staticmethod
    def save_continuous_volume_increase(data_list: List[Dict]) -> Dict:
        """
        保存连续放量个股数据
        
        Args:
            data_list: 连续放量个股数据列表
            
        Returns:
            dict: 操作结果统计
        """
        return DataCenterDAO._batch_process(
            model_class=ContinuousVolumeIncrease,
            data_list=data_list,
            mapping=CONTINUOUS_VOLUME_INCREASE_MAPPING,
            unique_fields=['dm', 't']
        )
    
    @staticmethod
    def save_continuous_volume_decrease(data_list: List[Dict]) -> Dict:
        """
        保存连续缩量个股数据
        
        Args:
            data_list: 连续缩量个股数据列表
            
        Returns:
            dict: 操作结果统计
        """
        return DataCenterDAO._batch_process(
            model_class=ContinuousVolumeDecrease,
            data_list=data_list,
            mapping=CONTINUOUS_VOLUME_DECREASE_MAPPING,
            unique_fields=['dm', 't']
        )
    
    @staticmethod
    def save_continuous_rise(data_list: List[Dict]) -> Dict:
        """
        保存连续上涨个股数据
        
        Args:
            data_list: 连续上涨个股数据列表
            
        Returns:
            dict: 操作结果统计
        """
        return DataCenterDAO._batch_process(
            model_class=ContinuousRise,
            data_list=data_list,
            mapping=CONTINUOUS_RISE_MAPPING,
            unique_fields=['dm', 't']
        )
    
    @staticmethod
    def save_continuous_fall(data_list: List[Dict]) -> Dict:
        """
        保存连续下跌个股数据
        
        Args:
            data_list: 连续下跌个股数据列表
            
        Returns:
            dict: 操作结果统计
        """
        return DataCenterDAO._batch_process(
            model_class=ContinuousFall,
            data_list=data_list,
            mapping=CONTINUOUS_FALL_MAPPING,
            unique_fields=['dm', 't']
        )
    
    # 财务指标相关DAO方法
    
    @staticmethod
    def save_weekly_rank_change(data_list: List[Dict]) -> Dict:
        """
        保存周涨跌排名数据
        
        Args:
            data_list: 周涨跌排名数据列表
            
        Returns:
            dict: 操作结果统计
        """
        return DataCenterDAO._batch_process(
            model_class=WeeklyRankChange,
            data_list=data_list,
            mapping=WEEKLY_RANK_CHANGE_MAPPING,
            unique_fields=['dm', 't']
        )
    
    @staticmethod
    def save_monthly_rank_change(data_list: List[Dict]) -> Dict:
        """
        保存月涨跌排名数据
        
        Args:
            data_list: 月涨跌排名数据列表
            
        Returns:
            dict: 操作结果统计
        """
        return DataCenterDAO._batch_process(
            model_class=MonthlyRankChange,
            data_list=data_list,
            mapping=MONTHLY_RANK_CHANGE_MAPPING,
            unique_fields=['dm', 't']
        )
    
    @staticmethod
    def save_weekly_strong_stocks(data_list: List[Dict]) -> Dict:
        """
        保存本周强势股数据
        
        Args:
            data_list: 本周强势股数据列表
            
        Returns:
            dict: 操作结果统计
        """
        return DataCenterDAO._batch_process(
            model_class=WeeklyStrongStock,
            data_list=data_list,
            mapping=WEEKLY_STRONG_STOCK_MAPPING,
            unique_fields=['dm', 't']
        )
    
    @staticmethod
    def save_monthly_strong_stocks(data_list: List[Dict]) -> Dict:
        """
        保存本月强势股数据
        
        Args:
            data_list: 本月强势股数据列表
            
        Returns:
            dict: 操作结果统计
        """
        return DataCenterDAO._batch_process(
            model_class=MonthlyStrongStock,
            data_list=data_list,
            mapping=MONTHLY_STRONG_STOCK_MAPPING,
            unique_fields=['dm', 't']
        )
    
    @staticmethod
    def save_circ_market_value_rank(data_list: List[Dict]) -> Dict:
        """
        保存流通市值排行数据
        
        Args:
            data_list: 流通市值排行数据列表
            
        Returns:
            dict: 操作结果统计
        """
        return DataCenterDAO._batch_process(
            model_class=CircMarketValueRank,
            data_list=data_list,
            mapping=CIRC_MARKET_VALUE_RANK_MAPPING,
            unique_fields=['dm', 't']
        )
    
    @staticmethod
    def save_pe_ratio_rank(data_list: List[Dict]) -> Dict:
        """
        保存市盈率排行数据
        
        Args:
            data_list: 市盈率排行数据列表
            
        Returns:
            dict: 操作结果统计
        """
        return DataCenterDAO._batch_process(
            model_class=PERatioRank,
            data_list=data_list,
            mapping=PE_RATIO_RANK_MAPPING,
            unique_fields=['dm', 't']
        )
    
    @staticmethod
    def save_pb_ratio_rank(data_list: List[Dict]) -> Dict:
        """
        保存市净率排行数据
        
        Args:
            data_list: 市净率排行数据列表
            
        Returns:
            dict: 操作结果统计
        """
        return DataCenterDAO._batch_process(
            model_class=PBRatioRank,
            data_list=data_list,
            mapping=PB_RATIO_RANK_MAPPING,
            unique_fields=['dm', 't']
        )
    
    @staticmethod
    def save_roe_rank(data_list: List[Dict]) -> Dict:
        """
        保存ROE排行数据
        
        Args:
            data_list: ROE排行数据列表
            
        Returns:
            dict: 操作结果统计
        """
        return DataCenterDAO._batch_process(
            model_class=ROERank,
            data_list=data_list,
            mapping=ROE_RANK_MAPPING,
            unique_fields=['dm', 'hym']
        )
    
    # 机构持股相关DAO方法
    
    @staticmethod
    def save_institution_holding_summary(data_list: List[Dict], year: int, quarter: int) -> Dict:
        """
        保存机构持股汇总数据
        
        Args:
            data_list: 机构持股汇总数据列表
            year: 报告年份
            quarter: 报告季度，1:一季报，2：中报，3：三季报，4：年报
            
        Returns:
            dict: 操作结果统计
        """
        return DataCenterDAO._batch_process(
            model_class=InstitutionHoldingSummary,
            data_list=data_list,
            mapping=INSTITUTION_HOLDING_SUMMARY_MAPPING,
            unique_fields=['dm', 'year', 'quarter'],
            year=year,
            quarter=quarter
        )
    
    @staticmethod
    def save_fund_heavy_positions(data_list: List[Dict], year: int, quarter: int) -> Dict:
        """
        保存基金重仓数据
        
        Args:
            data_list: 基金重仓数据列表
            year: 报告年份
            quarter: 报告季度，1:一季报，2：中报，3：三季报，4：年报
            
        Returns:
            dict: 操作结果统计
        """
        return DataCenterDAO._batch_process(
            model_class=FundHeavyPosition,
            data_list=data_list,
            mapping=FUND_HEAVY_POSITION_MAPPING,
            unique_fields=['dm', 'year', 'quarter'],
            year=year,
            quarter=quarter
        )
    
    @staticmethod
    def save_social_security_heavy_positions(data_list: List[Dict], year: int, quarter: int) -> Dict:
        """
        保存社保重仓数据
        
        Args:
            data_list: 社保重仓数据列表
            year: 报告年份
            quarter: 报告季度，1:一季报，2：中报，3：三季报，4：年报
            
        Returns:
            dict: 操作结果统计
        """
        return DataCenterDAO._batch_process(
            model_class=SocialSecurityHeavyPosition,
            data_list=data_list,
            mapping=SOCIAL_SECURITY_HEAVY_POSITION_MAPPING,
            unique_fields=['dm', 'year', 'quarter'],
            year=year,
            quarter=quarter
        )
    
    @staticmethod
    def save_qfii_heavy_positions(data_list: List[Dict], year: int, quarter: int) -> Dict:
        """
        保存QFII重仓股数据
        
        Args:
            data_list: QFII重仓股数据列表
            year: 报告年份
            quarter: 报告季度，1:一季报，2：中报，3：三季报，4：年报
            
        Returns:
            dict: 操作结果统计
        """
        return DataCenterDAO._batch_process(
            model_class=QFIIHeavyPosition,
            data_list=data_list,
            mapping=QFII_HEAVY_POSITION_MAPPING,
            unique_fields=['dm', 'year', 'quarter'],
            year=year,
            quarter=quarter
        )
    
    # 资金流向相关DAO方法
    
    @staticmethod
    def save_industry_capital_flow(data_list: List[Dict]) -> Dict:
        """
        保存行业资金流向数据
        
        Args:
            data_list: 行业资金流向数据列表
            
        Returns:
            dict: 操作结果统计
        """
        return DataCenterDAO._batch_process(
            model_class=IndustryCapitalFlow,
            data_list=data_list,
            mapping=INDUSTRY_CAPITAL_FLOW_MAPPING,
            unique_fields=['hymc', 't']
        )
    
    @staticmethod
    def save_concept_capital_flow(data_list: List[Dict]) -> Dict:
        """
        保存概念板块资金流向数据
        
        Args:
            data_list: 概念板块资金流向数据列表
            
        Returns:
            dict: 操作结果统计
        """
        return DataCenterDAO._batch_process(
            model_class=ConceptCapitalFlow,
            data_list=data_list,
            mapping=CONCEPT_CAPITAL_FLOW_MAPPING,
            unique_fields=['gnmc', 't']
        )
    
    @staticmethod
    def save_stock_capital_flow(data_list: List[Dict]) -> Dict:
        """
        保存个股资金流向数据
        
        Args:
            data_list: 个股资金流向数据列表
            
        Returns:
            dict: 操作结果统计
        """
        return DataCenterDAO._batch_process(
            model_class=StockCapitalFlow,
            data_list=data_list,
            mapping=STOCK_CAPITAL_FLOW_MAPPING,
            unique_fields=['dm', 't']
        )
    
    # 南北向资金相关DAO方法
    
    @staticmethod
    def save_north_south_fund_overview(data_list: List[Dict]) -> Dict:
        """
        保存南北向资金流向概览数据
        
        Args:
            data_list: 南北向资金流向概览数据列表
            
        Returns:
            dict: 操作结果统计
        """
        return DataCenterDAO._batch_process(
            model_class=NorthSouthFundOverview,
            data_list=data_list,
            mapping=NORTH_SOUTH_FUND_OVERVIEW_MAPPING,
            unique_fields=['t']
        )
    
    @staticmethod
    def save_north_fund_trend(data_list: List[Dict]) -> Dict:
        """
        保存北向资金历史走势数据
        
        Args:
            data_list: 北向资金历史走势数据列表
            
        Returns:
            dict: 操作结果统计
        """
        return DataCenterDAO._batch_process(
            model_class=NorthFundTrend,
            data_list=data_list,
            mapping=NORTH_FUND_TREND_MAPPING,
            unique_fields=['t']
        )
    
    @staticmethod
    def save_south_fund_trend(data_list: List[Dict]) -> Dict:
        """
        保存南向资金历史走势数据
        
        Args:
            data_list: 南向资金历史走势数据列表
            
        Returns:
            dict: 操作结果统计
        """
        return DataCenterDAO._batch_process(
            model_class=SouthFundTrend,
            data_list=data_list,
            mapping=SOUTH_FUND_TREND_MAPPING,
            unique_fields=['t']
        )
    
    @staticmethod
    def save_north_stock_holding(data_list: List[Dict], period: str) -> Dict:
        """
        保存北向持股明细数据
        
        Args:
            data_list: 北向持股明细数据列表
            period: 统计周期，可选 LD、3D、5D、10D、LM、LQ、LY
            
        Returns:
            dict: 操作结果统计
        """
        return DataCenterDAO._batch_process(
            model_class=NorthStockHolding,
            data_list=data_list,
            mapping=NORTH_STOCK_HOLDING_MAPPING,
            unique_fields=['dm', 't', 'period'],
            period=period
        )
