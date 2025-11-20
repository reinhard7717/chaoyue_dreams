from django.core.management.base import BaseCommand
from stock_models.time_trade import (
    StockMinuteData,
    StockMinuteData_5_SZ, StockMinuteData_5_SH, StockMinuteData_5_BJ, StockMinuteData_5_CY, StockMinuteData_5_KC,
    StockMinuteData_15_SZ, StockMinuteData_15_SH, StockMinuteData_15_BJ, StockMinuteData_15_CY, StockMinuteData_15_KC,
    StockMinuteData_30_SZ, StockMinuteData_30_SH, StockMinuteData_30_BJ, StockMinuteData_30_CY, StockMinuteData_30_KC,
    StockMinuteData_60_SZ, StockMinuteData_60_SH, StockMinuteData_60_BJ, StockMinuteData_60_CY, StockMinuteData_60_KC,
)
from django.db import transaction

# 迁移规则映射
MODEL_MAP = {
    '5': {
        'SZ': StockMinuteData_5_SZ,
        'SH': StockMinuteData_5_SH,
        'BJ': StockMinuteData_5_BJ,
        'CY': StockMinuteData_5_CY,
        'KC': StockMinuteData_5_KC,
    },
    '15': {
        'SZ': StockMinuteData_15_SZ,
        'SH': StockMinuteData_15_SH,
        'BJ': StockMinuteData_15_BJ,
        'CY': StockMinuteData_15_CY,
        'KC': StockMinuteData_15_KC,
    },
    '30': {
        'SZ': StockMinuteData_30_SZ,
        'SH': StockMinuteData_30_SH,
        'BJ': StockMinuteData_30_BJ,
        'CY': StockMinuteData_30_CY,
        'KC': StockMinuteData_30_KC,
    },
    '60': {
        'SZ': StockMinuteData_60_SZ,
        'SH': StockMinuteData_60_SH,
        'BJ': StockMinuteData_60_BJ,
        'CY': StockMinuteData_60_CY,
        'KC': StockMinuteData_60_KC,
    },
}

def get_target_model(time_level, stock_code):
    # 根据time_level和stock_code返回目标模型
    if stock_code.endswith('.SZ'):
        if stock_code.startswith('3'):
            return MODEL_MAP[time_level]['CY']
        else:
            return MODEL_MAP[time_level]['SZ']
    elif stock_code.endswith('.SH'):
        if stock_code.startswith('68'):
            return MODEL_MAP[time_level]['KC']
        else:
            return MODEL_MAP[time_level]['SH']
    elif stock_code.endswith('.BJ'):
        return MODEL_MAP[time_level]['BJ']
    else:
        return None

class Command(BaseCommand):
    help = '拆分StockMinuteData数据到新表'
    def handle(self, *args, **options):
        # 只处理5/15/30/60分钟
        time_levels = ['5', '15', '30', '60']
        batch_size = 30000  # 批量处理，防止内存溢出
        # 获取所有有分钟数据的股票代码
        stock_codes = StockMinuteData.objects.values_list('stock__stock_code', flat=True).distinct()
        print(f"共发现{len(stock_codes)}只股票需要迁移")
        for stock_code in stock_codes:  # 外层循环：遍历每个股票代码
            for time_level in time_levels:  # 内层循环：遍历每个时间级别
                queryset = StockMinuteData.objects.filter(stock__stock_code=stock_code, time_level=time_level).order_by('trade_time')  # 按trade_time正序查询
                total = queryset.count()
                if total > 0:
                    print(f"开始迁移 股票代码={stock_code}，time_level={time_level} 的数据，共{total}条")
                    i = 0
                    while i < total:
                        batch = list(queryset[i:i+batch_size])  # 获取批次数据
                        to_create = {}  # 用于收集当前time_level的key和对象
                        for obj in batch:
                            model = get_target_model(time_level, stock_code)  # 获取目标模型
                            if model is None:
                                print(f"未识别stock_code: {stock_code}，跳过")
                                continue
                            # 构造新表对象
                            new_obj = model(
                                stock=obj.stock,
                                trade_time=obj.trade_time,
                                open=obj.open,
                                high=obj.high,
                                low=obj.low,
                                close=obj.close,
                                vol=obj.vol,
                                amount=obj.amount,
                            )
                            # 确定key，例如'SZ'等
                            if stock_code.endswith('.SZ'):
                                if stock_code.startswith('3'):
                                    key = 'CY'
                                else:
                                    key = 'SZ'
                            elif stock_code.endswith('.SH'):
                                if stock_code.startswith('68'):
                                    key = 'KC'
                                else:
                                    key = 'SH'
                            elif stock_code.endswith('.BJ'):
                                key = 'BJ'
                            if key not in to_create:
                                to_create[key] = []
                            to_create[key].append(new_obj)  # 分类收集
                        # 批量写入
                        for key, objs in to_create.items():
                            if objs:
                                model_class = MODEL_MAP[time_level][key]
                                try:
                                    with transaction.atomic():
                                        model_class.objects.bulk_create(objs, ignore_conflicts=True)
                                    print(f"已迁移{len(objs)}条到{model_class._meta.db_table} for stock_code={stock_code}, time_level={time_level}")
                                except Exception as e:
                                    print(f"迁移{model_class._meta.db_table}时出错: {e}")
                        i += batch_size
                    print(f"股票代码={stock_code}，time_level={time_level} 迁移完成")
        print("全部迁移完成")
