# split_stock_daily_data.py
from django.core.management.base import BaseCommand
from stock_models.time_trade import (
    StockDailyData_SZ, StockDailyData_SH,
    StockDailyData_CY, StockDailyData_KC, StockDailyData_BJ
)
from django.db import transaction

class Command(BaseCommand):
    help = '将StockDailyData表数据按规则拆分到新表'
    def handle(self, *args, **options):
        # 批量处理，防止内存溢出
        batch_size = 30000
        total = StockDailyData.objects.count()
        print(f"总共需要迁移{total}条数据")
        offset = 0
        while True:
            batch = list(StockDailyData.objects.all()[offset:offset+batch_size])
            if not batch:
                break
            print(f"正在处理第{offset+1}到{offset+len(batch)}条数据")
            objs_sz, objs_cy, objs_sh, objs_kc, objs_bj = [], [], [], [], []
            for obj in batch:
                code = obj.stock_id  # stock_id等价于stock_code
                # 创业板
                if code.startswith('3') and code.endswith('.SZ'):
                    objs_cy.append(StockDailyData_CY(
                        stock=obj.stock, trade_time=obj.trade_time, open=obj.open, high=obj.high, low=obj.low,
                        close=obj.close, pre_close=obj.pre_close, change=obj.change, pct_change=obj.pct_change,
                        vol=obj.vol, amount=obj.amount, adj_factor=obj.adj_factor,
                        open_qfq=obj.open_qfq, high_qfq=obj.high_qfq, low_qfq=obj.low_qfq, close_qfq=obj.close_qfq,
                        pre_close_qfq=obj.pre_close_qfq, open_hfq=obj.open_hfq, high_hfq=obj.high_hfq,
                        low_hfq=obj.low_hfq, close_hfq=obj.close_hfq, pre_close_hfq=obj.pre_close_hfq
                    ))
                # 深市主板
                elif code.endswith('.SZ'):
                    objs_sz.append(StockDailyData_SZ(
                        stock=obj.stock, trade_time=obj.trade_time, open=obj.open, high=obj.high, low=obj.low,
                        close=obj.close, pre_close=obj.pre_close, change=obj.change, pct_change=obj.pct_change,
                        vol=obj.vol, amount=obj.amount, adj_factor=obj.adj_factor,
                        open_qfq=obj.open_qfq, high_qfq=obj.high_qfq, low_qfq=obj.low_qfq, close_qfq=obj.close_qfq,
                        pre_close_qfq=obj.pre_close_qfq, open_hfq=obj.open_hfq, high_hfq=obj.high_hfq,
                        low_hfq=obj.low_hfq, close_hfq=obj.close_hfq, pre_close_hfq=obj.pre_close_hfq
                    ))
                # 科创板
                elif code.startswith('68') and code.endswith('.SH'):
                    objs_kc.append(StockDailyData_KC(
                        stock=obj.stock, trade_time=obj.trade_time, open=obj.open, high=obj.high, low=obj.low,
                        close=obj.close, pre_close=obj.pre_close, change=obj.change, pct_change=obj.pct_change,
                        vol=obj.vol, amount=obj.amount, adj_factor=obj.adj_factor,
                        open_qfq=obj.open_qfq, high_qfq=obj.high_qfq, low_qfq=obj.low_qfq, close_qfq=obj.close_qfq,
                        pre_close_qfq=obj.pre_close_qfq, open_hfq=obj.open_hfq, high_hfq=obj.high_hfq,
                        low_hfq=obj.low_hfq, close_hfq=obj.close_hfq, pre_close_hfq=obj.pre_close_hfq
                    ))
                # 沪市主板
                elif code.endswith('.SH'):
                    objs_sh.append(StockDailyData_SH(
                        stock=obj.stock, trade_time=obj.trade_time, open=obj.open, high=obj.high, low=obj.low,
                        close=obj.close, pre_close=obj.pre_close, change=obj.change, pct_change=obj.pct_change,
                        vol=obj.vol, amount=obj.amount, adj_factor=obj.adj_factor,
                        open_qfq=obj.open_qfq, high_qfq=obj.high_qfq, low_qfq=obj.low_qfq, close_qfq=obj.close_qfq,
                        pre_close_qfq=obj.pre_close_qfq, open_hfq=obj.open_hfq, high_hfq=obj.high_hfq,
                        low_hfq=obj.low_hfq, close_hfq=obj.close_hfq, pre_close_hfq=obj.pre_close_hfq
                    ))
                # 北交所
                elif code.endswith('.BJ'):
                    objs_bj.append(StockDailyData_BJ(
                        stock=obj.stock, trade_time=obj.trade_time, open=obj.open, high=obj.high, low=obj.low,
                        close=obj.close, pre_close=obj.pre_close, change=obj.change, pct_change=obj.pct_change,
                        vol=obj.vol, amount=obj.amount, adj_factor=obj.adj_factor,
                        open_qfq=obj.open_qfq, high_qfq=obj.high_qfq, low_qfq=obj.low_qfq, close_qfq=obj.close_qfq,
                        pre_close_qfq=obj.pre_close_qfq, open_hfq=obj.open_hfq, high_hfq=obj.high_hfq,
                        low_hfq=obj.low_hfq, close_hfq=obj.close_hfq, pre_close_hfq=obj.pre_close_hfq
                    ))
                else:
                    print(f"未识别的股票代码: {code}")
            # 批量插入
            with transaction.atomic():
                if objs_cy:
                    StockDailyData_CY.objects.bulk_create(objs_cy, ignore_conflicts=True)
                if objs_sz:
                    StockDailyData_SZ.objects.bulk_create(objs_sz, ignore_conflicts=True)
                if objs_kc:
                    StockDailyData_KC.objects.bulk_create(objs_kc, ignore_conflicts=True)
                if objs_sh:
                    StockDailyData_SH.objects.bulk_create(objs_sh, ignore_conflicts=True)
                if objs_bj:
                    StockDailyData_BJ.objects.bulk_create(objs_bj, ignore_conflicts=True)
            offset += batch_size
        print("数据迁移完成！")
