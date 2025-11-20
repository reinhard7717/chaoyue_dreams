# stock_models/management/commands/split_stock_daily_basic_data.py
# 导入Django管理命令的基础类
from django.core.management.base import BaseCommand
# 导入需要操作的模型，包括原始总表和各个分表
from stock_models.time_trade import (
    StockDailyBasic, StockDailyBasic_SZ, StockDailyBasic_SH,
    StockDailyBasic_CY, StockDailyBasic_KC, StockDailyBasic_BJ
)
# 导入数据库事务处理，确保数据迁移的原子性
from django.db import transaction

class Command(BaseCommand):
    """
    一个Django管理命令，用于将 stock_time_trade_day_basic 表中的数据
    根据股票代码规则，迁移到按板块划分的各个分表中。
    """
    # 定义在命令行中执行 `manage.py help <command_name>` 时显示的帮助信息
    help = '将StockDailyBasic表数据按规则拆分到新表'
    def handle(self, *args, **options):
        """
        命令执行的入口方法。
        """
        # 设置批量处理的大小，以防止一次性加载过多数据导致内存溢出
        batch_size = 30000
        # 获取源表中的总数据量，并打印提示信息
        total = StockDailyBasic.objects.count()
        print(f"总共需要迁移 {total} 条 StockDailyBasic 数据")
        # 初始化偏移量，用于分批次查询
        offset = 0
        # 循环处理所有数据，直到处理完所有批次
        while True:
            # 分批次从数据库中获取数据
            # 这里直接查询，没有使用 .select_related('stock')，因为我们直接用 obj.stock_id，效率更高
            batch = list(StockDailyBasic.objects.all()[offset:offset+batch_size])
            # 如果当前批次没有数据，说明已经处理完毕，退出循环
            if not batch:
                break
            # 打印当前处理批次的进度信息
            print(f"正在处理第 {offset+1} 到 {offset+len(batch)} 条数据")
            # 为不同板块的表创建空列表，用于暂存待批量插入的对象
            objs_sz, objs_cy, objs_sh, objs_kc, objs_bj = [], [], [], [], []
            # 遍历当前批次获取到的每一条数据
            for obj in batch:
                # 获取股票代码。obj.stock_id 是外键字段的原始值(即stock_code)，比 obj.stock.stock_code 更高效
                code = obj.stock_id
                # 根据股票代码的规则判断其所属板块，并将数据复制到对应板块的模型实例中
                # 创业板 (代码以'3'开头，以'.SZ'结尾)
                if code.startswith('3') and code.endswith('.SZ'):
                    objs_cy.append(self._create_new_instance(StockDailyBasic_CY, obj))
                # 深市主板 (代码以'.SZ'结尾，但不属于创业板)
                elif code.endswith('.SZ'):
                    objs_sz.append(self._create_new_instance(StockDailyBasic_SZ, obj))
                # 科创板 (代码以'68'开头，以'.SH'结尾)
                elif code.startswith('68') and code.endswith('.SH'):
                    objs_kc.append(self._create_new_instance(StockDailyBasic_KC, obj))
                # 沪市主板 (代码以'.SH'结尾，但不属于科创板)
                elif code.endswith('.SH'):
                    objs_sh.append(self._create_new_instance(StockDailyBasic_SH, obj))
                # 北交所 (代码以'.BJ'结尾)
                elif code.endswith('.BJ'):
                    objs_bj.append(self._create_new_instance(StockDailyBasic_BJ, obj))
                # 处理未识别的股票代码
                else:
                    print(f"警告：未识别的股票代码: {code}")
            # 使用数据库事务，确保同批次数据插入的原子性，要么全部成功，要么全部失败
            with transaction.atomic():
                # 批量插入数据到各个分表
                # ignore_conflicts=True 可以在存在唯一键冲突时忽略错误，保证脚本可重复执行而不会报错
                if objs_cy:
                    StockDailyBasic_CY.objects.bulk_create(objs_cy, ignore_conflicts=True)
                if objs_sz:
                    StockDailyBasic_SZ.objects.bulk_create(objs_sz, ignore_conflicts=True)
                if objs_kc:
                    StockDailyBasic_KC.objects.bulk_create(objs_kc, ignore_conflicts=True)
                if objs_sh:
                    StockDailyBasic_SH.objects.bulk_create(objs_sh, ignore_conflicts=True)
                if objs_bj:
                    StockDailyBasic_BJ.objects.bulk_create(objs_bj, ignore_conflicts=True)
            # 更新偏移量，准备处理下一批数据
            offset += batch_size
        # 所有数据处理完成后，打印最终的成功信息
        print("StockDailyBasic 数据迁移完成！")
    def _create_new_instance(self, model_class, source_obj):
        """
        一个辅助方法，用于从源对象创建一个新的模型实例并复制所有字段。
        Args:
            model_class: 目标模型的类 (例如 StockDailyBasic_CY)
            source_obj: 源数据对象 (StockDailyBasic 的实例)
        Returns:
            一个新的 model_class 实例
        """
        # 通过将源对象的字段一一对应，创建目标模型的新实例
        return model_class(
            stock=source_obj.stock,
            trade_time=source_obj.trade_time,
            close=source_obj.close,
            turnover_rate=source_obj.turnover_rate,
            turnover_rate_f=source_obj.turnover_rate_f,
            volume_ratio=source_obj.volume_ratio,
            pe=source_obj.pe,
            pe_ttm=source_obj.pe_ttm,
            pb=source_obj.pb,
            ps=source_obj.ps,
            ps_ttm=source_obj.ps_ttm,
            dv_ratio=source_obj.dv_ratio,
            dv_ttm=source_obj.dv_ttm,
            total_share=source_obj.total_share,
            float_share=source_obj.float_share,
            free_share=source_obj.free_share,
            total_mv=source_obj.total_mv,
            circ_mv=source_obj.circ_mv,
            limit_status=source_obj.limit_status
        )
