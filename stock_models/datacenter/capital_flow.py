# models/datacenter/capital_flow.py
from django.db import models
from django.utils.translation import gettext_lazy as _
from stock_models.index import IndexInfo
from stock_models.stock_basic import MarketCategory, StockInfo
from utils.models import BaseModel
from decimal import Decimal

# 证监会行业资金流向
class IndustryCapitalFlow(BaseModel):
    update_time = models.DateTimeField(verbose_name="更新时间")
    industry = models.ForeignKey(MarketCategory, on_delete=models.CASCADE, related_name="industry_capital_flow", verbose_name=_("行业"))
    average_price = models.DecimalField(max_digits=12, decimal_places=2, verbose_name="均价")
    change_percent = models.DecimalField(max_digits=8, decimal_places=2, verbose_name="涨跌幅")
    inflow_capital = models.DecimalField(max_digits=20, decimal_places=4, verbose_name="流入资金")
    outflow_capital = models.DecimalField(max_digits=20, decimal_places=4, verbose_name="流出资金")
    net_inflow = models.DecimalField(max_digits=20, decimal_places=4, verbose_name="净流入")
    net_inflow_rate = models.DecimalField(max_digits=8, decimal_places=2, verbose_name="净流入率")
    leading_stock_name = models.CharField(max_length=100, verbose_name="领涨股名称")
    leading_stock_code = models.CharField(max_length=20, verbose_name="领涨股代码")
    leading_stock_net_inflow_rate = models.DecimalField(max_digits=8, decimal_places=2, verbose_name="领涨股净流入率")

    class Meta:
        db_table = "industry_capital_flow"
        verbose_name = "证监会行业资金流向"
        verbose_name_plural = verbose_name
        indexes = [
            models.Index(fields=['update_time'], name='icf_update_time_idx'),
            models.Index(fields=['industry'], name='icf_industry_code_idx'),
            models.Index(fields=['net_inflow'], name='icf_net_inflow_idx'),
            models.Index(fields=['change_percent'], name='icf_change_percent_idx'),
        ]

    def __code__(self):
        return self.industry.code

# 概念板块资金流向
class ConceptCapitalFlow(BaseModel):
    update_time = models.DateTimeField(verbose_name="更新时间")
    concept = models.ForeignKey(MarketCategory, on_delete=models.CASCADE, related_name="concept_capital_flow", verbose_name=_("概念板块"))
    average_price = models.DecimalField(max_digits=12, decimal_places=2, verbose_name="均价")
    change_percent = models.DecimalField(max_digits=8, decimal_places=2, verbose_name="涨跌幅")
    inflow_capital = models.DecimalField(max_digits=20, decimal_places=4, verbose_name="流入资金")
    outflow_capital = models.DecimalField(max_digits=20, decimal_places=4, verbose_name="流出资金")
    net_inflow = models.DecimalField(max_digits=20, decimal_places=4, verbose_name="净流入")
    net_inflow_rate = models.DecimalField(max_digits=8, decimal_places=2, verbose_name="净流入率")
    leading_stock_name = models.CharField(max_length=100, verbose_name="领涨股名称")
    leading_stock_code = models.CharField(max_length=20, verbose_name="领涨股代码")
    leading_stock_net_inflow_rate = models.DecimalField(max_digits=8, decimal_places=2, verbose_name="领涨股净流入率")

    class Meta:
        db_table = "concept_capital_flow"
        verbose_name = "概念板块资金流向"
        verbose_name_plural = verbose_name
        indexes = [
            models.Index(fields=['update_time'], name='ccf_update_time_idx'),
            models.Index(fields=['concept'], name='ccf_concept_code_idx'),
            models.Index(fields=['net_inflow'], name='ccf_net_inflow_idx'),
            models.Index(fields=['change_percent'], name='ccf_change_percent_idx'),
        ]

    def __code__(self):
        return self.concept.code

# 净流入额排名
class NetInflowRanking(BaseModel):
    update_time = models.DateTimeField(verbose_name="更新时间")
    stock = models.ForeignKey(StockInfo, on_delete=models.CASCADE, related_name="net_inflow_ranking", verbose_name=_("股票"))
    latest_price = models.DecimalField(max_digits=12, decimal_places=2, verbose_name="最新价")
    change_percent = models.DecimalField(max_digits=8, decimal_places=2, verbose_name="涨跌幅")
    turnover_rate = models.DecimalField(max_digits=8, decimal_places=2, verbose_name="换手率")
    transaction_amount = models.DecimalField(max_digits=20, decimal_places=4, verbose_name="成交额")
    outflow_capital = models.DecimalField(max_digits=20, decimal_places=4, verbose_name="流出资金")
    inflow_capital = models.DecimalField(max_digits=20, decimal_places=4, verbose_name="流入资金")
    net_inflow = models.DecimalField(max_digits=20, decimal_places=4, verbose_name="净流入")
    net_inflow_rate = models.DecimalField(max_digits=8, decimal_places=2, verbose_name="净流入率")

    class Meta:
        db_table = "stock_net_inflow_ranking"
        verbose_name = "净流入额排名"
        verbose_name_plural = verbose_name
        indexes = [
            models.Index(fields=['update_time'], name='nir_update_time_idx'),
            models.Index(fields=['stock'], name='nir_stock_idx'),
            models.Index(fields=['net_inflow'], name='nir_net_inflow_idx'),
            models.Index(fields=['change_percent'], name='nir_change_percent_idx'),
        ]

    def __code__(self):
        return self.stock.stock_code

# 净流入率排名
class NetInflowRateRanking(BaseModel):
    update_time = models.DateTimeField(verbose_name="更新时间")
    stock = models.ForeignKey(StockInfo, on_delete=models.CASCADE, related_name="net_inflow_rate_ranking", verbose_name=_("股票"))
    latest_price = models.DecimalField(max_digits=12, decimal_places=2, verbose_name="最新价")
    change_percent = models.DecimalField(max_digits=8, decimal_places=2, verbose_name="涨跌幅")
    turnover_rate = models.DecimalField(max_digits=8, decimal_places=2, verbose_name="换手率")
    transaction_amount = models.DecimalField(max_digits=20, decimal_places=4, verbose_name="成交额")
    outflow_capital = models.DecimalField(max_digits=20, decimal_places=4, verbose_name="流出资金")
    inflow_capital = models.DecimalField(max_digits=20, decimal_places=4, verbose_name="流入资金")
    net_inflow = models.DecimalField(max_digits=20, decimal_places=4, verbose_name="净流入")
    net_inflow_rate = models.DecimalField(max_digits=8, decimal_places=2, verbose_name="净流入率")

    class Meta:
        db_table = "stock_net_inflow_rate_ranking"
        verbose_name = "净流入率排名"
        verbose_name_plural = verbose_name
        indexes = [
            models.Index(fields=['update_time'], name='nirr_update_time_idx'),
            models.Index(fields=['stock'], name='nirr_stock_idx'),
            models.Index(fields=['net_inflow_rate'], name='nirr_net_inflow_rate_idx'),
            models.Index(fields=['change_percent'], name='nirr_change_percent_idx'),
        ]

    def __code__(self):
        return self.stock.stock_code

# 主力净流入额排名
class MainForceNetInflowRanking(BaseModel):
    update_time = models.DateTimeField(verbose_name="更新时间")
    stock = models.ForeignKey(StockInfo, on_delete=models.CASCADE, related_name="main_force_net_inflow_ranking", verbose_name=_("股票"))
    latest_price = models.DecimalField(max_digits=12, decimal_places=2, verbose_name="最新价")
    change_percent = models.DecimalField(max_digits=8, decimal_places=2, verbose_name="涨跌幅")
    turnover_rate = models.DecimalField(max_digits=8, decimal_places=2, verbose_name="换手率")
    transaction_amount = models.DecimalField(max_digits=20, decimal_places=4, verbose_name="成交额")
    main_force_outflow = models.DecimalField(max_digits=20, decimal_places=4, verbose_name="主力流出资金")
    main_force_inflow = models.DecimalField(max_digits=20, decimal_places=4, verbose_name="主力流入资金")
    main_force_net_inflow = models.DecimalField(max_digits=20, decimal_places=4, verbose_name="主力净流入")
    main_force_net_inflow_rate = models.DecimalField(max_digits=8, decimal_places=2, verbose_name="主力净流入率")

    class Meta:
        db_table = "stock_main_force_net_inflow_ranking"
        verbose_name = "主力净流入额排名"
        verbose_name_plural = verbose_name
        indexes = [
            models.Index(fields=['update_time'], name='mfnir_update_time_idx'),
            models.Index(fields=['stock'], name='mfnir_stock_idx'),
            models.Index(fields=['main_force_net_inflow'], name='mfnir_main_force_inflow_idx'),
            models.Index(fields=['change_percent'], name='mfnir_change_percent_idx'),
        ]

    def __code__(self):
        return self.stock.stock_code

# 主力净流入率排名
class MainForceNetInflowRateRanking(BaseModel):
    update_time = models.DateTimeField(verbose_name="更新时间")
    stock = models.ForeignKey(StockInfo, on_delete=models.CASCADE, related_name="main_force_net_inflow_rate_ranking", verbose_name=_("股票"))
    latest_price = models.DecimalField(max_digits=12, decimal_places=2, verbose_name="最新价")
    change_percent = models.DecimalField(max_digits=8, decimal_places=2, verbose_name="涨跌幅")
    turnover_rate = models.DecimalField(max_digits=8, decimal_places=2, verbose_name="换手率")
    transaction_amount = models.DecimalField(max_digits=20, decimal_places=4, verbose_name="成交额")
    main_force_outflow = models.DecimalField(max_digits=20, decimal_places=4, verbose_name="主力流出资金")
    main_force_inflow = models.DecimalField(max_digits=20, decimal_places=4, verbose_name="主力流入资金")
    main_force_net_inflow = models.DecimalField(max_digits=20, decimal_places=4, verbose_name="主力净流入")
    main_force_net_inflow_rate = models.DecimalField(max_digits=8, decimal_places=2, verbose_name="主力净流入率")

    class Meta:
        db_table = "stock_main_force_net_inflow_rate_ranking"
        verbose_name = "主力净流入率排名"
        verbose_name_plural = verbose_name
        indexes = [
            models.Index(fields=['update_time'], name='mfnirr_update_time_idx'),
            models.Index(fields=['stock'], name='mfnirr_stock_idx'),
            models.Index(fields=['main_force_net_inflow_rate'], name='mfnirr_inflow_rate_idx'),
            models.Index(fields=['change_percent'], name='mfnirr_change_percent_idx'),
        ]

    def __code__(self):
        return self.stock.stock_code

# 散户净流入额排名
class RetailNetInflowRanking(BaseModel):
    update_time = models.DateTimeField(verbose_name="更新时间")
    stock = models.ForeignKey(StockInfo, on_delete=models.CASCADE, related_name="retail_net_inflow_ranking", verbose_name=_("股票"))
    latest_price = models.DecimalField(max_digits=12, decimal_places=2, verbose_name="最新价")
    change_percent = models.DecimalField(max_digits=8, decimal_places=2, verbose_name="涨跌幅")
    turnover_rate = models.DecimalField(max_digits=8, decimal_places=2, verbose_name="换手率")
    transaction_amount = models.DecimalField(max_digits=20, decimal_places=4, verbose_name="成交额")
    retail_outflow = models.DecimalField(max_digits=20, decimal_places=4, verbose_name="散户流出资金")
    retail_inflow = models.DecimalField(max_digits=20, decimal_places=4, verbose_name="散户流入资金")
    retail_net_inflow = models.DecimalField(max_digits=20, decimal_places=4, verbose_name="散户净流入")
    retail_net_inflow_rate = models.DecimalField(max_digits=8, decimal_places=2, verbose_name="散户净流入率")

    class Meta:
        db_table = "stock_retail_net_inflow_ranking"
        verbose_name = "散户净流入额排名"
        verbose_name_plural = verbose_name
        indexes = [
            models.Index(fields=['update_time'], name='rnir_update_time_idx'),
            models.Index(fields=['stock'], name='rnir_stock_idx'),
            models.Index(fields=['retail_net_inflow'], name='rnir_retail_net_inflow_idx'),
            models.Index(fields=['change_percent'], name='rnir_change_percent_idx'),
        ]

    def __code__(self):
        return self.stock.stock_code

# 散户净流入率排名
class RetailNetInflowRateRanking(BaseModel):
    update_time = models.DateTimeField(verbose_name="更新时间")
    stock = models.ForeignKey(StockInfo, on_delete=models.CASCADE, related_name="retail_net_inflow_rate_ranking", verbose_name=_("股票"))
    latest_price = models.DecimalField(max_digits=12, decimal_places=2, verbose_name="最新价")
    change_percent = models.DecimalField(max_digits=8, decimal_places=2, verbose_name="涨跌幅")
    turnover_rate = models.DecimalField(max_digits=8, decimal_places=2, verbose_name="换手率")
    transaction_amount = models.DecimalField(max_digits=20, decimal_places=4, verbose_name="成交额")
    retail_outflow = models.DecimalField(max_digits=20, decimal_places=4, verbose_name="散户流出资金")
    retail_inflow = models.DecimalField(max_digits=20, decimal_places=4, verbose_name="散户流入资金")
    retail_net_inflow = models.DecimalField(max_digits=20, decimal_places=4, verbose_name="散户净流入")
    retail_net_inflow_rate = models.DecimalField(max_digits=8, decimal_places=2, verbose_name="散户净流入率")

    class Meta:
        db_table = "stock_retail_net_inflow_rate_ranking"
        verbose_name = "散户净流入率排名"
        verbose_name_plural = verbose_name
        indexes = [
            models.Index(fields=['update_time'], name='rnirr_update_time_idx'),
            models.Index(fields=['stock'], name='rnirr_stock_idx'),
            models.Index(fields=['retail_net_inflow_rate'], name='rnirr_retail_inflow_rate_idx'),
            models.Index(fields=['change_percent'], name='rnirr_change_percent_idx'),
        ]

    def __code__(self):
        return self.stock.stock_code

# 证监会行业资金路线图
class IndustryCapitalFlowRoute(BaseModel):
    update_time = models.DateTimeField(verbose_name="更新时间")
    industry = models.ForeignKey(MarketCategory, on_delete=models.CASCADE, related_name="industry_capital_flow_route", verbose_name=_("行业"))
    change_3days = models.DecimalField(max_digits=8, decimal_places=2, verbose_name="近三日涨跌幅")
    net_inflow_3days = models.DecimalField(max_digits=20, decimal_places=4, verbose_name="近三日净流入")
    inflow_rate_3days = models.DecimalField(max_digits=8, decimal_places=2, verbose_name="近三日净流入率")
    change_5days = models.DecimalField(max_digits=8, decimal_places=2, verbose_name="近五日涨跌幅")
    net_inflow_5days = models.DecimalField(max_digits=20, decimal_places=4, verbose_name="近五日净流入")
    inflow_rate_5days = models.DecimalField(max_digits=8, decimal_places=2, verbose_name="近五日净流入率")
    change_10days = models.DecimalField(max_digits=8, decimal_places=2, verbose_name="近十日涨跌幅")
    net_inflow_10days = models.DecimalField(max_digits=20, decimal_places=4, verbose_name="近十日净流入")
    inflow_rate_10days = models.DecimalField(max_digits=8, decimal_places=2, verbose_name="近十日净流入率")

    class Meta:
        db_table = "industry_capital_flow_route"
        verbose_name = "证监会行业资金路线图"
        verbose_name_plural = verbose_name
        indexes = [
            models.Index(fields=['update_time'], name='icfr_update_time_idx'),
            models.Index(fields=['industry'], name='icfr_industry_code_idx'),
            models.Index(fields=['net_inflow_3days'], name='icfr_net_inflow_3days_idx'),
            models.Index(fields=['net_inflow_5days'], name='icfr_net_inflow_5days_idx'),
            models.Index(fields=['net_inflow_10days'], name='icfr_net_inflow_10days_idx'),
        ]

    def __code__(self):
        return self.industry.code

# 概念板块资金路线图
class ConceptCapitalFlowRoute(BaseModel):
    update_time = models.DateTimeField(verbose_name="更新时间")
    concept = models.ForeignKey(MarketCategory, on_delete=models.CASCADE, related_name="concept_capital_flow_route", verbose_name=_("概念板块"))
    change_3days = models.DecimalField(max_digits=8, decimal_places=2, verbose_name="近三日涨跌幅")
    net_inflow_3days = models.DecimalField(max_digits=20, decimal_places=4, verbose_name="近三日净流入")
    inflow_rate_3days = models.DecimalField(max_digits=8, decimal_places=2, verbose_name="近三日净流入率")
    change_5days = models.DecimalField(max_digits=8, decimal_places=2, verbose_name="近五日涨跌幅")
    net_inflow_5days = models.DecimalField(max_digits=20, decimal_places=4, verbose_name="近五日净流入")
    inflow_rate_5days = models.DecimalField(max_digits=8, decimal_places=2, verbose_name="近五日净流入率")
    change_10days = models.DecimalField(max_digits=8, decimal_places=2, verbose_name="近十日涨跌幅")
    net_inflow_10days = models.DecimalField(max_digits=20, decimal_places=4, verbose_name="近十日净流入")
    inflow_rate_10days = models.DecimalField(max_digits=8, decimal_places=2, verbose_name="近十日净流入率")

    class Meta:
        db_table = "concept_capital_flow_route"
        verbose_name = "概念板块资金路线图"
        verbose_name_plural = verbose_name
        indexes = [
            models.Index(fields=['update_time'], name='ccfr_update_time_idx'),
            models.Index(fields=['concept'], name='ccfr_concept_code_idx'),
            models.Index(fields=['net_inflow_3days'], name='ccfr_net_inflow_3days_idx'),
            models.Index(fields=['net_inflow_5days'], name='ccfr_net_inflow_5days_idx'),
            models.Index(fields=['net_inflow_10days'], name='ccfr_net_inflow_10days_idx'),
        ]

    def __code__(self):
        return self.concept.code

# 个股阶段统计总览
class StockPeriodStatisticsOverview(BaseModel):
    update_time = models.DateTimeField(verbose_name="更新时间")
    stock = models.ForeignKey(StockInfo, on_delete=models.CASCADE, related_name="stock_period_statistics_overview", verbose_name=_("股票"))
    latest_price = models.DecimalField(max_digits=12, decimal_places=2, verbose_name="最新价")
    change_percent = models.DecimalField(max_digits=8, decimal_places=2, verbose_name="涨跌幅")
    turnover_rate = models.DecimalField(max_digits=8, decimal_places=2, verbose_name="换手率")
    net_inflow_rate_3days = models.DecimalField(max_digits=8, decimal_places=2, verbose_name="3日净流入率")
    net_inflow_rate_5days = models.DecimalField(max_digits=8, decimal_places=2, verbose_name="5日净流入率")
    net_inflow_rate_10days = models.DecimalField(max_digits=8, decimal_places=2, verbose_name="10日净流入率")
    net_inflow_rate_20days = models.DecimalField(max_digits=8, decimal_places=2, verbose_name="20日净流入率")
    net_inflow_rate_60days = models.DecimalField(max_digits=8, decimal_places=2, verbose_name="60日净流入率")

    class Meta:
        db_table = "stock_period_statistics_overview"
        verbose_name = "个股阶段统计总览"
        verbose_name_plural = verbose_name
        indexes = [
            models.Index(fields=['update_time'], name='spso_update_time_idx'),
            models.Index(fields=['stock'], name='spso_stock_idx'),
            models.Index(fields=['net_inflow_rate_3days'], name='spso_inflow_rate_3days_idx'),
            models.Index(fields=['net_inflow_rate_10days'], name='spso_inflow_rate_10days_idx'),
        ]

    def __code__(self):
        return self.stock.stock_code

# 个股阶段统计
class StockPeriodStatistics(BaseModel):
    update_time = models.DateTimeField(verbose_name="更新时间")
    stock = models.ForeignKey(StockInfo, on_delete=models.CASCADE, related_name="stock_period_statistics", verbose_name=_("股票"))
    period_end_price = models.DecimalField(max_digits=12, decimal_places=2, verbose_name="阶段结束价")
    period_change_percent = models.DecimalField(max_digits=8, decimal_places=2, verbose_name="阶段涨跌幅")
    period_turnover_rate = models.DecimalField(max_digits=8, decimal_places=2, verbose_name="阶段换手率")
    period_net_inflow = models.DecimalField(max_digits=20, decimal_places=4, null=True, blank=True, verbose_name="阶段净流入")
    period_net_inflow_rate = models.DecimalField(max_digits=8, decimal_places=2, verbose_name="阶段净流入率")
    period_main_force_net_inflow = models.DecimalField(max_digits=20, decimal_places=4, verbose_name="阶段主力净流入")
    period_main_force_net_inflow_rate = models.DecimalField(max_digits=8, decimal_places=2, verbose_name="阶段主力净流入率")
    period_days = models.IntegerField(verbose_name="阶段天数")

    class Meta:
        db_table = "stock_period_statistics"
        verbose_name = "个股阶段统计"
        verbose_name_plural = verbose_name
        indexes = [
            models.Index(fields=['update_time'], name='sps_update_time_idx'),
            models.Index(fields=['stock'], name='sps_stock_idx'),
            models.Index(fields=['period_days'], name='sps_period_days_idx'),
            models.Index(fields=['period_net_inflow_rate'], name='sps_period_net_inflow_rate_idx'),
        ]

    def __code__(self):
        return self.stock.stock_code

# 主力连续净流入/流出
class MainForceContinuousFlow(BaseModel):
    update_time = models.DateTimeField(verbose_name="更新时间")
    stock = models.ForeignKey(StockInfo, on_delete=models.CASCADE, related_name="main_force_continuous_flow", verbose_name=_("股票"))
    flow_days = models.IntegerField(verbose_name="流入/流出天数")
    latest_price = models.DecimalField(max_digits=12, decimal_places=2, verbose_name="最新价")
    period_change_percent = models.DecimalField(max_digits=8, decimal_places=2, verbose_name="阶段涨跌幅")
    period_turnover_rate = models.DecimalField(max_digits=8, decimal_places=2, verbose_name="阶段换手率")
    period_net_flow = models.DecimalField(max_digits=20, decimal_places=4, null=True, blank=True, verbose_name="阶段净流入/流出")
    period_flow_rate = models.DecimalField(max_digits=8, decimal_places=2, verbose_name="阶段流入/流出率")
    main_force_net_flow = models.DecimalField(max_digits=20, decimal_places=4, verbose_name="主力净流入/流出")

    class Meta:
        db_table = "stock_main_force_continuous_flow"
        verbose_name = "主力连续净流入/流出"
        verbose_name_plural = verbose_name
        indexes = [
            models.Index(fields=['update_time'], name='mfcf_update_time_idx'),
            models.Index(fields=['stock'], name='mfcf_stock_idx'),
            models.Index(fields=['flow_days'], name='mfcf_flow_days_idx'),
            models.Index(fields=['main_force_net_flow'], name='mfcf_main_force_net_flow_idx'),
        ]

    def __code__(self):
        return self.stock.stock_code

# 新资金流向概览
class NewCapitalFlowOverview(BaseModel):
    index = models.ForeignKey(IndexInfo, on_delete=models.CASCADE, related_name="new_capital_flow_overview", verbose_name=_("股票指数"))
    update_time = models.DateTimeField(verbose_name="更新时间")
    flow_type = models.CharField(max_length=20, verbose_name="类型")
    plate_name = models.CharField(max_length=50, verbose_name="板块")
    flow_direction = models.CharField(max_length=10, verbose_name="资金方向")
    net_buy_amount = models.DecimalField(max_digits=20, decimal_places=2, verbose_name="成交净买额(万)")
    net_inflow_amount = models.DecimalField(max_digits=20, decimal_places=2, verbose_name="资金净流入(万)")
    remaining_amount = models.DecimalField(max_digits=20, decimal_places=2, verbose_name="当日资金余额(万)")
    up_stocks = models.IntegerField(verbose_name="上涨股票数")
    flat_stocks = models.IntegerField(verbose_name="持平股票数")
    down_stocks = models.IntegerField(verbose_name="下跌股票数")
    index_change_percent = models.DecimalField(max_digits=8, decimal_places=2, verbose_name="相关指数涨跌幅")
    trade_status = models.IntegerField(verbose_name="交易状态")

    class Meta:
        db_table = "stock_new_capital_flow_overview"
        verbose_name = "新资金流向概览"
        verbose_name_plural = verbose_name
        indexes = [
            models.Index(fields=['update_time'], name='ncfo_update_time_idx'),
            models.Index(fields=['flow_type'], name='ncfo_flow_type_idx'),
            models.Index(fields=['flow_direction'], name='ncfo_flow_direction_idx'),
            models.Index(fields=['net_inflow_amount'], name='ncfo_net_inflow_amount_idx'),
        ]

    def __code__(self):
        return self.index.code


