# models/datacenter/capital_flow.py
from django.db import models
from utils.models import BaseModel
from datetime import datetime


class IndustryCapitalFlow(BaseModel):
    """行业资金流向"""
    trade_date = models.DateField(verbose_name="日期")  # 原 t
    industry_name = models.CharField(max_length=50, verbose_name="行业名称")  # 原 hymc
    net_inflow = models.DecimalField(max_digits=20, decimal_places=2, verbose_name="资金净流入(万元)")  # 原 zjjlr
    main_force_net_inflow = models.DecimalField(max_digits=20, decimal_places=2, verbose_name="主力净流入(万元)")  # 原 zljlr
    retail_net_inflow = models.DecimalField(max_digits=20, decimal_places=2, verbose_name="散户净流入(万元)")  # 原 shjlr
    net_inflow_rate = models.DecimalField(max_digits=10, decimal_places=2, verbose_name="净流入率(%)")  # 原 jlrl
    main_force_net_inflow_rate = models.DecimalField(max_digits=10, decimal_places=2, verbose_name="主力净流入率(%)")  # 原 zljlrl
    retail_net_inflow_rate = models.DecimalField(max_digits=10, decimal_places=2, verbose_name="散户净流入率(%)")  # 原 shjlrl
    average_net_inflow = models.DecimalField(max_digits=20, decimal_places=2, verbose_name="净流入均额(万元)")  # 原 jlrpj
    change_rate = models.DecimalField(max_digits=10, decimal_places=2, verbose_name="行业涨跌幅(%)")  # 原 zdf
    
    class Meta:
        verbose_name = "行业资金流向"
        verbose_name_plural = verbose_name
        indexes = [
            models.Index(fields=['trade_date']),
            models.Index(fields=['industry_name']),
        ]


class ConceptCapitalFlow(BaseModel):
    """概念板块资金流向"""
    trade_date = models.DateField(verbose_name="日期")  # 原 t
    concept_name = models.CharField(max_length=50, verbose_name="概念名称")  # 原 gnmc
    net_inflow = models.DecimalField(max_digits=20, decimal_places=2, verbose_name="资金净流入(万元)")  # 原 zjjlr
    main_force_net_inflow = models.DecimalField(max_digits=20, decimal_places=2, verbose_name="主力净流入(万元)")  # 原 zljlr
    retail_net_inflow = models.DecimalField(max_digits=20, decimal_places=2, verbose_name="散户净流入(万元)")  # 原 shjlr
    net_inflow_rate = models.DecimalField(max_digits=10, decimal_places=2, verbose_name="净流入率(%)")  # 原 jlrl
    main_force_net_inflow_rate = models.DecimalField(max_digits=10, decimal_places=2, verbose_name="主力净流入率(%)")  # 原 zljlrl
    retail_net_inflow_rate = models.DecimalField(max_digits=10, decimal_places=2, verbose_name="散户净流入率(%)")  # 原 shjlrl
    average_net_inflow = models.DecimalField(max_digits=20, decimal_places=2, verbose_name="净流入均额(万元)")  # 原 jlrpj
    change_rate = models.DecimalField(max_digits=10, decimal_places=2, verbose_name="概念涨跌幅(%)")  # 原 zdf
    
    class Meta:
        verbose_name = "概念板块资金流向"
        verbose_name_plural = verbose_name
        indexes = [
            models.Index(fields=['trade_date']),
            models.Index(fields=['concept_name']),
        ]


class StockCapitalFlow(BaseModel):
    """个股资金流向"""
    trade_date = models.DateField(verbose_name="日期")  # 原 t
    stock_code = models.CharField(max_length=10, verbose_name="股票代码")  # 原 dm
    stock_name = models.CharField(max_length=50, verbose_name="股票名称")  # 原 mc
    net_inflow = models.DecimalField(max_digits=20, decimal_places=2, verbose_name="资金净流入(万元)")  # 原 zjjlr
    main_force_net_inflow = models.DecimalField(max_digits=20, decimal_places=2, verbose_name="主力净流入(万元)")  # 原 zljlr
    retail_net_inflow = models.DecimalField(max_digits=20, decimal_places=2, verbose_name="散户净流入(万元)")  # 原 shjlr
    net_inflow_rate = models.DecimalField(max_digits=10, decimal_places=2, verbose_name="净流入率(%)")  # 原 jlrl
    main_force_net_inflow_rate = models.DecimalField(max_digits=10, decimal_places=2, verbose_name="主力净流入率(%)")  # 原 zljlrl
    retail_net_inflow_rate = models.DecimalField(max_digits=10, decimal_places=2, verbose_name="散户净流入率(%)")  # 原 shjlrl
    change_rate = models.DecimalField(max_digits=10, decimal_places=2, verbose_name="涨跌幅(%)")  # 原 zdf
    trading_amount = models.DecimalField(max_digits=20, decimal_places=2, verbose_name="成交额(万元)")  # 原 cje
    total_market_value = models.DecimalField(max_digits=20, decimal_places=2, verbose_name="总市值(万元)")  # 原 zsz
    turnover_rate = models.DecimalField(max_digits=10, decimal_places=2, verbose_name="换手率(%)")  # 原 hs
    
    class Meta:
        verbose_name = "个股资金流向"
        verbose_name_plural = verbose_name
        indexes = [
            models.Index(fields=['trade_date']),
            models.Index(fields=['stock_code']),
        ]
