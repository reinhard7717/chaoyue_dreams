# api_manager/apis/datacenter_api.py
import aiohttp
import asyncio
import logging
import json
from datetime import datetime
from typing import Dict, List, Optional, Union, Any

from api_manager.base_api import BaseAPI

logger = logging.getLogger('api')

class DataCenterAPI(BaseAPI):
    """沪深数据中心API"""
    # 龙虎榜相关API
    async def get_daily_lhb(self) -> Optional[Dict]:
        """
        获取今日龙虎榜概览
        Returns:
            Optional[Dict]: 今日龙虎榜数据，如请求失败则返回None
        """
        try:
            logger.info("开始请求龙虎榜数据API")
            response = await self.get('data/all/ld', expected_type='dict')
            if not response:
                logger.warning("龙虎榜API返回空响应")
                return None
            return response
        except Exception as e:
            logger.error(f"请求龙虎榜数据API出错: {str(e)}")
            return None
    async def get_stock_on_list(self, days: int) -> Optional[List[Dict]]:
        """
        获取近n日上榜个股
        Args:
            days: 统计天数，可选 5、10、30、60
        Returns:
            Optional[List[Dict]]: 个股上榜统计数据，如请求失败则返回None
        """
        try:
            # 转换为整数并验证
            days = int(days)
            if days not in [5, 10, 30, 60]:
                logger.error(f"无效的统计天数: {days}，必须是 5、10、30、60 之一")
                return None
            return await self.get(f'data/all/gg/{days}', expected_type='list')
        except Exception as e:
            logger.error(f"获取近{days}日上榜个股数据出错: {str(e)}")
            return None
    async def get_broker_on_list(self, days: int) -> Optional[List[Dict]]:
        """
        获取近n日营业部上榜统计
        Args:
            days: 统计天数，可选 5、10、30、60
        Returns:
            Optional[List[Dict]]: 营业部上榜统计数据，如请求失败则返回None
        """
        try:
            # 转换为整数并验证
            days = int(days)
            if days not in [5, 10, 30, 60]:
                logger.error(f"无效的统计天数: {days}，必须是 5、10、30、60 之一")
                return None
            return await self.get(f'data/all/yyb/{days}', expected_type='list')
        except Exception as e:
            logger.error(f"获取近{days}日营业部上榜统计数据出错: {str(e)}")
            return None
  
    async def get_institution_trade_track(self, days: int) -> Optional[List[Dict]]:
        """
        获取近n日个股机构交易追踪
        Args:
            days: 统计天数，可选 5、10、30、60
        Returns:
            Optional[List[Dict]]: 机构席位追踪数据，如请求失败则返回None
        """
        try:
            # 转换为整数并验证
            days = int(days)
            if days not in [5, 10, 30, 60]:
                logger.error(f"无效的统计天数: {days}，必须是 5、10、30、60 之一")
                return None
            return await self.get(f'data/all/jgzz/{days}', expected_type='list')
        except Exception as e:
            logger.error(f"获取近{days}日个股机构交易追踪数据出错: {str(e)}")
            return None
    async def get_institution_trade_detail(self, days: int) -> List[Dict]:
        """
        获取机构席位成交明细
        Args:
            days: 统计天数，可选 5、10、30、60
        Returns:
            List[Dict]: 机构席位成交明细数据列表
        """
        try:
            # 转换为整数并验证
            days = int(days)
            if days not in [5, 10, 30, 60]:
                logger.error(f"无效的统计天数: {days}，必须是 5、10、30、60 之一")
                return []
                
            return await self.get(f'data/all/jgcj/{days}', expected_type='list')
        except Exception as e:
            logger.error(f"请求机构席位成交明细数据出错: {str(e)}")
            return []
    # 个股统计相关API
    async def get_stage_high_low(self) -> Optional[List[Dict]]:
        """
        获取阶段最高最低
        Returns:
            Optional[List[Dict]]: 阶段最高最低数据，如请求失败则返回None
        """
        return await self.get('data/all/jdgd', expected_type='list')
    async def get_new_high_stocks(self) -> Optional[List[Dict]]:
        """
        获取盘中创新高个股
        Returns:
            Optional[List[Dict]]: 盘中创新高个股数据，如请求失败则返回None
        """
        return await self.get('data/all/cxg', expected_type='list')
    async def get_new_low_stocks(self) -> Optional[List[Dict]]:
        """
        获取盘中创新低个股
        Returns:
            Optional[List[Dict]]: 盘中创新低个股数据，如请求失败则返回None
        """
        return await self.get('data/all/cxd', expected_type='list')
    # 盘中数据和连续交易相关API
    async def get_volume_increase(self) -> Optional[List[Dict]]:
        """
        获取成交骤增个股
        Returns:
            Optional[List[Dict]]: 成交骤增个股数据，如请求失败则返回None
        """
        return await self.get('data/all/cjzz', expected_type='list')
    async def get_volume_decrease(self) -> Optional[List[Dict]]:
        """
        获取成交骤减个股
        Returns:
            Optional[List[Dict]]: 成交骤减个股数据，如请求失败则返回None
        """
        return await self.get('data/all/cjzj', expected_type='list')
    async def get_continuous_volume_increase(self) -> Optional[List[Dict]]:
        """
        获取连续放量个股
        Returns:
            Optional[List[Dict]]: 连续放量个股数据，如请求失败则返回None
        """
        return await self.get('data/all/lxfl', expected_type='list')
    async def get_continuous_volume_decrease(self) -> Optional[List[Dict]]:
        """
        获取连续缩量个股
        Returns:
            Optional[List[Dict]]: 连续缩量个股数据，如请求失败则返回None
        """
        return await self.get('data/all/lxsl', expected_type='list')
    async def get_continuous_rise(self) -> Optional[List[Dict]]:
        """
        获取连续上涨个股
        Returns:
            Optional[List[Dict]]: 连续上涨个股数据，如请求失败则返回None
        """
        return await self.get('data/all/lxsz', expected_type='list')
    async def get_continuous_fall(self) -> Optional[List[Dict]]:
        """
        获取连续下跌个股
        Returns:
            Optional[List[Dict]]: 连续下跌个股数据，如请求失败则返回None
        """
        return await self.get('data/all/lxxd', expected_type='list')
    # 财务指标相关API
    async def get_weekly_rank_change(self) -> Optional[List[Dict]]:
        """
        获取周涨跌排名
        Returns:
            Optional[List[Dict]]: 周涨跌排名数据，如请求失败则返回None
        """
        try:
            data = await self.get('data/all/zzdpm', expected_type='list')
            # 检查是否是404错误或其他错误字符串
            if isinstance(data, str) and ("404" in data or "无资源" in data):
                logger.warning(f"获取周涨跌排名失败: {data}")
                return []
                
            return data
        except Exception as e:
            logger.error(f"获取周涨跌排名出错: {str(e)}")
            return None
    async def get_monthly_rank_change(self) -> Optional[List[Dict]]:
        """
        获取月涨跌排名
        Returns:
            Optional[List[Dict]]: 月涨跌排名数据，如请求失败则返回None
        """
        try:
            data = await self.get('data/all/yzdpm', expected_type='list')
            # 检查是否是404错误或其他错误字符串
            if isinstance(data, str) and ("404" in data or "无资源" in data):
                logger.warning(f"获取月涨跌排名失败: {data}")
                return []
                
            return data
        except Exception as e:
            logger.error(f"获取月涨跌排名出错: {str(e)}")
            return None
    async def get_weekly_strong_stocks(self) -> Optional[List[Dict]]:
        """
        获取本周强势股
        Returns:
            Optional[List[Dict]]: 本周强势股数据，如请求失败则返回None
        """
        try:
            data = await self.get('data/all/bzqsg', expected_type='list')
            # 检查是否是404错误或其他错误字符串
            if isinstance(data, str) and ("404" in data or "无资源" in data):
                logger.warning(f"获取本周强势股失败: {data}")
                return []
                
            return data
        except Exception as e:
            logger.error(f"获取本周强势股出错: {str(e)}")
            return None
    async def get_monthly_strong_stocks(self) -> Optional[List[Dict]]:
        """
        获取本月强势股
        Returns:
            Optional[List[Dict]]: 本月强势股数据，如请求失败则返回None
        """
        try:
            data = await self.get('data/all/byqsg', expected_type='list')
            # 检查是否是404错误或其他错误字符串
            if isinstance(data, str) and ("404" in data or "无资源" in data):
                logger.warning(f"获取本月强势股失败: {data}")
                return []
                
            return data
        except Exception as e:
            logger.error(f"获取本月强势股出错: {str(e)}")
            return None
    async def get_circ_market_value_rank(self) -> Optional[List[Dict]]:
        """
        获取流通市值排行
        Returns:
            Optional[List[Dict]]: 流通市值排行数据，如请求失败则返回None
        """
        return await self.get('data/all/ltsz', expected_type='list')
    async def get_pe_ratio_rank(self) -> Optional[List[Dict]]:
        """
        获取市盈率排行
        Returns:
            Optional[List[Dict]]: 市盈率排行数据，如请求失败则返回None
        """
        return await self.get('data/all/syl', expected_type='list')
    async def get_pb_ratio_rank(self) -> Optional[List[Dict]]:
        """
        获取市净率排行
        Returns:
            Optional[List[Dict]]: 市净率排行数据，如请求失败则返回None
        """
        return await self.get('data/all/sjl', expected_type='list')
    async def get_roe_rank(self) -> Optional[List[Dict]]:
        """
        获取ROE排行
        Returns:
            Optional[List[Dict]]: ROE排行数据，如请求失败则返回None
        """
        return await self.get('data/all/roe', expected_type='list')
    # 财务报表相关API
    async def get_financial_profit(self, year: int, quarter: int) -> Optional[List[Dict]]:
        """
        获取盈利能力数据
        Args:
            year: 报告年份
            quarter: 报告季度，1:一季报，2：中报，3：三季报，4：年报
        Returns:
            Optional[List[Dict]]: 盈利能力数据，如请求失败则返回None
        """
        if quarter not in [1, 2, 3, 4]:
            logger.error(f"无效的季度: {quarter}，必须是 1、2、3、4 之一")
            return None
        return await self.get(f'data/all/finyl/{year}_{quarter}', expected_type='list')
    async def get_financial_operation(self, year: int, quarter: int) -> Optional[List[Dict]]:
        """
        获取运营能力数据
        Args:
            year: 报告年份
            quarter: 报告季度，1:一季报，2：中报，3：三季报，4：年报
        Returns:
            Optional[List[Dict]]: 运营能力数据，如请求失败则返回None
        """
        if quarter not in [1, 2, 3, 4]:
            logger.error(f"无效的季度: {quarter}，必须是 1、2、3、4 之一")
            return None
        return await self.get(f'data/all/finyynl/{year}_{quarter}', expected_type='list')
    async def get_financial_growth(self, year: int, quarter: int) -> Optional[List[Dict]]:
        """
        获取成长能力数据
        Args:
            year: 报告年份
            quarter: 报告季度，1:一季报，2：中报，3：三季报，4：年报
        Returns:
            Optional[List[Dict]]: 成长能力数据，如请求失败则返回None
        """
        if quarter not in [1, 2, 3, 4]:
            logger.error(f"无效的季度: {quarter}，必须是 1、2、3、4 之一")
            return None
        return await self.get(f'data/all/fincznl/{year}_{quarter}', expected_type='list')
    async def get_financial_debt(self, year: int, quarter: int) -> Optional[List[Dict]]:
        """
        获取偿债能力数据
        Args:
            year: 报告年份
            quarter: 报告季度，1:一季报，2：中报，3：三季报，4：年报
        Returns:
            Optional[List[Dict]]: 偿债能力数据，如请求失败则返回None
        """
        if quarter not in [1, 2, 3, 4]:
            logger.error(f"无效的季度: {quarter}，必须是 1、2、3、4 之一")
            return None
        return await self.get(f'data/all/finchzhainl/{year}_{quarter}', expected_type='list')
    async def get_financial_cash_flow(self, year: int, quarter: int) -> Optional[List[Dict]]:
        """
        获取现金流量数据
        Args:
            year: 报告年份
            quarter: 报告季度，1:一季报，2：中报，3：三季报，4：年报
        Returns:
            Optional[List[Dict]]: 现金流量数据，如请求失败则返回None
        """
        if quarter not in [1, 2, 3, 4]:
            logger.error(f"无效的季度: {quarter}，必须是 1、2、3、4 之一")
            return None
        return await self.get(f'data/all/finxjll/{year}_{quarter}', expected_type='list')
    async def get_financial_report(self, year: int, quarter: int) -> Optional[List[Dict]]:
        """
        获取业绩报表数据
        Args:
            year: 报告年份
            quarter: 报告季度，1:一季报，2：中报，3：三季报，4：年报
        Returns:
            Optional[List[Dict]]: 业绩报表数据，如请求失败则返回None
        """
        if quarter not in [1, 2, 3, 4]:
            logger.error(f"无效的季度: {quarter}，必须是 1、2、3、4 之一")
            return None
        return await self.get(f'data/all/finyjbb/{year}_{quarter}', expected_type='list')
    async def get_financial_forecast(self, year: int, quarter: int) -> Optional[List[Dict]]:
        """
        获取业绩预告数据
        Args:
            year: 报告年份
            quarter: 报告季度，1:一季报，2：中报，3：三季报，4：年报
        Returns:
            Optional[List[Dict]]: 业绩预告数据，如请求失败则返回None
        """
        if quarter not in [1, 2, 3, 4]:
            logger.error(f"无效的季度: {quarter}，必须是 1、2、3、4 之一")
            return None
        return await self.get(f'data/all/finyjyg/{year}_{quarter}', expected_type='list')
    async def get_financial_express(self, year: int, quarter: int) -> Optional[List[Dict]]:
        """
        获取业绩快报数据
        Args:
            year: 报告年份
            quarter: 报告季度，1:一季报，2：中报，3：三季报，4：年报
        Returns:
            Optional[List[Dict]]: 业绩快报数据，如请求失败则返回None
        """
        if quarter not in [1, 2, 3, 4]:
            logger.error(f"无效的季度: {quarter}，必须是 1、2、3、4 之一")
            return None
        return await self.get(f'data/all/finyjkb/{year}_{quarter}', expected_type='list')
    async def get_financial_profit_detail(self) -> Optional[List[Dict]]:
        """
        获取利润细分数据
        Returns:
            Optional[List[Dict]]: 利润细分数据，如请求失败则返回None
        """
        return await self.get('data/all/finlrxf', expected_type='list')
    # 机构持股相关API
    async def get_institution_holding_summary(self, year: int, quarter: int) -> Optional[List[Dict]]:
        """
        获取机构持股汇总数据
        Args:
            year: 报告年份
            quarter: 报告季度，1:一季报，2：中报，3：三季报，4：年报
        Returns:
            Optional[List[Dict]]: 机构持股汇总数据，如请求失败则返回None
        """
        if quarter not in [1, 2, 3, 4]:
            logger.error(f"无效的季度: {quarter}，必须是 1、2、3、4 之一")
            return None
        return await self.get(f'data/all/orgcghz/{year}_{quarter}', expected_type='list')
    async def get_fund_heavy_positions(self, year: int, quarter: int) -> Optional[List[Dict]]:
        """
        获取基金重仓数据
        Args:
            year: 报告年份
            quarter: 报告季度，1:一季报，2：中报，3：三季报，4：年报
        Returns:
            Optional[List[Dict]]: 基金重仓数据，如请求失败则返回None
        """
        if quarter not in [1, 2, 3, 4]:
            logger.error(f"无效的季度: {quarter}，必须是 1、2、3、4 之一")
            return None
        return await self.get(f'data/all/orgjjzc/{year}_{quarter}', expected_type='list')
    async def get_social_security_heavy_positions(self, year: int, quarter: int) -> Optional[List[Dict]]:
        """
        获取社保重仓数据
        Args:
            year: 报告年份
            quarter: 报告季度，1:一季报，2：中报，3：三季报，4：年报
        Returns:
            Optional[List[Dict]]: 社保重仓数据，如请求失败则返回None
        """
        if quarter not in [1, 2, 3, 4]:
            logger.error(f"无效的季度: {quarter}，必须是 1、2、3、4 之一")
            return None
        return await self.get(f'data/all/orgsbzc/{year}_{quarter}', expected_type='list')
    async def get_qfii_heavy_positions(self, year: int, quarter: int) -> Optional[List[Dict]]:
        """
        获取QFII重仓股数据
        Args:
            year: 报告年份
            quarter: 报告季度，1:一季报，2：中报，3：三季报，4：年报
        Returns:
            Optional[List[Dict]]: QFII重仓股数据，如请求失败则返回None
        """
        if quarter not in [1, 2, 3, 4]:
            logger.error(f"无效的季度: {quarter}，必须是 1、2、3、4 之一")
            return None
        return await self.get(f'data/all/orgqfiizc/{year}_{quarter}', expected_type='list')
    # 资金流向相关API
    async def get_industry_capital_flow(self) -> Optional[List[Dict]]:
        """
        获取证监会行业资金路线图数据
        Returns:
            Optional[List[Dict]]: 证监会行业资金路线图数据，如请求失败则返回None
        """
        return await self.get('data/all/zjlx/zjhhyzjlx', expected_type='list')
    async def get_concept_capital_flow(self) -> Optional[List[Dict]]:
        """
        获取概念板块资金路线图数据
        Returns:
            Optional[List[Dict]]: 概念板块资金路线图数据，如请求失败则返回None
        """
        return await self.get('data/all/zjlx/gnbklx', expected_type='list')
    async def get_stock_period_statistics_overview(self) -> Optional[List[Dict]]:
        """
        获取个股阶段统计总览数据
        Returns:
            Optional[List[Dict]]: 个股阶段统计总览数据，如请求失败则返回None
        """
        return await self.get('data/all/zjlx/ggjdtjzl', expected_type='list')
    async def get_stock_period_statistics_3(self) -> Optional[List[Dict]]:
        """
        获取个股阶段统计数据
        Returns:
            Optional[List[Dict]]: 个股阶段统计数据，如请求失败则返回None
        """
        return await self.get('data/all/zjlx/ggjdtj_3', expected_type='list')
    async def get_stock_period_statistics_5(self) -> Optional[List[Dict]]:
        """
        获取个股阶段统计数据
        Returns:
            Optional[List[Dict]]: 个股阶段统计数据，如请求失败则返回None
        """
        return await self.get('data/all/zjlx/ggjdtj_5', expected_type='list')
    async def get_stock_period_statistics_10(self) -> Optional[List[Dict]]:
        """
        获取个股阶段统计数据
        Returns:
            Optional[List[Dict]]: 个股阶段统计数据，如请求失败则返回None
        """
        return await self.get('data/all/zjlx/ggjdtj_10', expected_type='list')
    async def get_stock_period_statistics_20(self) -> Optional[List[Dict]]:
        """
        获取个股阶段统计数据
        Returns:
            Optional[List[Dict]]: 个股阶段统计数据，如请求失败则返回None
        """
        return await self.get('data/all/zjlx/ggjdtj_20', expected_type='list')
    async def get_net_inflow_amount_rank(self) -> Optional[List[Dict]]:
        """
        获取净流入额排名数据
        Returns:
            Optional[List[Dict]]: 净流入额排名数据，如请求失败则返回None
        """
        return await self.get('data/all/zjlx/jlrepm', expected_type='list')
    async def get_net_inflow_rate_rank(self) -> Optional[List[Dict]]:
        """
        获取净流入率排名数据
        Returns:
            Optional[List[Dict]]: 净流入率排名数据，如请求失败则返回None
        """
        return await self.get('data/all/zjlx/jlrlpm', expected_type='list')
    async def get_main_net_inflow_amount_rank(self) -> Optional[List[Dict]]:
        """
        获取主力净流入额排名数据
        Returns:
            Optional[List[Dict]]: 主力净流入额排名数据，如请求失败则返回None
        """
        return await self.get('data/all/zjlx/zljlrepm', expected_type='list')
    async def get_main_net_inflow_rate_rank(self) -> Optional[List[Dict]]:
        """
        获取主力净流入率排名数据
        Returns:
            Optional[List[Dict]]: 主力净流入率排名数据，如请求失败则返回None
        """
        return await self.get('data/all/zjlx/zljlrlpm', expected_type='list')
    async def get_retail_net_inflow_amount_rank(self) -> Optional[List[Dict]]:
        """
        获取散户净流入额排名数据
        Returns:
            Optional[List[Dict]]: 散户净流入额排名数据，如请求失败则返回None
        """
        return await self.get('data/all/zjlx/shjlrepm', expected_type='list')
    async def get_retail_net_inflow_rate_rank(self) -> Optional[List[Dict]]:
        """
        获取散户净流入率排名数据
        Returns:
            Optional[List[Dict]]: 散户净流入率排名数据，如请求失败则返回None
        """
        return await self.get('data/all/zjlx/shjlrlpm', expected_type='list')
    # 南北向资金相关API
    async def get_north_south_fund_overview(self) -> Optional[List[Dict]]:
        """
        获取南北向资金流向概览数据
        Returns:
            Optional[List[Dict]]: 南北向资金流向概览数据，如请求失败则返回None
        """
        return await self.get('data/all/nxbx/zjgl', expected_type='list')
    async def get_north_fund_history_trend(self, period: str) -> Optional[List[Dict]]:
        """
        获取北向资金历史走势数据
        Args:
            period: 时间段，可选 1m、6m、1y、all
        Returns:
            Optional[List[Dict]]: 北向资金历史走势数据，如请求失败则返回None
        """
        if period not in ['1m', '6m', '1y', 'all']:
            logger.error(f"无效的时间段: {period}，必须是 1m、6m、1y、all 之一")
            return None
        return await self.get(f'data/all/nxbx/bxzjlszs/{period}', expected_type='list')
    async def get_south_fund_history_trend(self, period: str) -> Optional[List[Dict]]:
        """
        获取南向资金历史走势数据
        Args:
            period: 时间段，可选 1m、6m、1y、all
        Returns:
            Optional[List[Dict]]: 南向资金历史走势数据，如请求失败则返回None
        """
        if period not in ['1m', '6m', '1y', 'all']:
            logger.error(f"无效的时间段: {period}，必须是 1m、6m、1y、all 之一")
            return None
        return await self.get(f'data/all/nxbx/nxzjlszs/{period}', expected_type='list')
    async def get_north_fund_history_overview(self) -> Optional[Dict]:
        """
        获取北向资金历史总览数据
        Returns:
            Optional[Dict]: 北向资金历史总览数据，如请求失败则返回None
        """
        return await self.get('data/all/nxbx/bxzjlsgl', expected_type='dict')
    async def get_south_fund_history_overview(self) -> Optional[Dict]:
        """
        获取南向资金历史总览数据
        Returns:
            Optional[Dict]: 南向资金历史总览数据，如请求失败则返回None
        """
        return await self.get('data/all/nxbx/nxzjlsgl', expected_type='dict')
    async def get_north_stock_period_rank(self, period: str) -> Optional[List[Dict]]:
        """
        获取北向个股周期排名数据
        Args:
            period: 周期，可选 1D、3D、5D、10D、LM、LQ、LY
        Returns:
            Optional[List[Dict]]: 北向个股周期排名数据，如请求失败则返回None
        """
        valid_periods = ['1D', '3D', '5D', '10D', 'LM', 'LQ', 'LY']
        if period not in valid_periods:
            logger.error(f"无效的周期: {period}，必须是 {', '.join(valid_periods)} 之一")
            return None
        return await self.get(f'data/all/nxbx/bxggpm/{period}', expected_type='list')
