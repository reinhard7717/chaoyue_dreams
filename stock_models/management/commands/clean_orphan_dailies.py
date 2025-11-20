# stock_models/management/commands/clean_orphan_dailies.py

from django.core.management.base import BaseCommand
from django.db.models import F
from stock_models.industry import ThsIndex, ThsIndexDaily, DcIndex, DcIndexDaily

class Command(BaseCommand):
    help = 'Deletes orphan daily records from ThsIndexDaily and DcIndexDaily that lack a corresponding parent index.'
    def handle(self, *args, **options):
        self.stdout.write(self.style.SUCCESS("开始清理孤儿日线数据..."))
        # 清理 ThsIndexDaily
        self.stdout.write("正在检查 ThsIndexDaily...")
        # 获取所有父表中的ts_code
        parent_codes_ths = set(ThsIndex.objects.values_list('ts_code', flat=True))
        # 使用 ths_index__ts_code 进行跨关系查询
        # 找出子表中存在但父表中不存在的ts_code
        orphan_codes_ths = ThsIndexDaily.objects.exclude(ths_index__ts_code__in=parent_codes_ths).values_list('ths_index__ts_code', flat=True).distinct()
        orphan_list_ths = list(orphan_codes_ths)
        if orphan_list_ths:
            self.stdout.write(self.style.WARNING(f"在 ThsIndexDaily 中发现 {len(orphan_list_ths)} 个孤儿ts_code: {orphan_list_ths}"))
            # 使用 ths_index__ts_code 进行过滤删除
            # 删除这些孤儿记录
            deleted_count, _ = ThsIndexDaily.objects.filter(ths_index__ts_code__in=orphan_list_ths).delete()
            self.stdout.write(self.style.SUCCESS(f"成功从 ThsIndexDaily 中删除 {deleted_count} 条孤儿记录。"))
        else:
            self.stdout.write(self.style.SUCCESS("ThsIndexDaily 表数据一致，无需清理。"))
        # 清理 DcIndexDaily
        self.stdout.write("\n正在检查 DcIndexDaily...")
        parent_codes_dc = set(DcIndex.objects.values_list('ts_code', flat=True))
        # 使用 dc_index__ts_code 进行跨关系查询
        orphan_codes_dc = DcIndexDaily.objects.exclude(dc_index__ts_code__in=parent_codes_dc).values_list('dc_index__ts_code', flat=True).distinct()
        orphan_list_dc = list(orphan_codes_dc)
        if orphan_list_dc:
            self.stdout.write(self.style.WARNING(f"在 DcIndexDaily 中发现 {len(orphan_list_dc)} 个孤儿ts_code: {orphan_list_dc}"))
            # 使用 dc_index__ts_code 进行过滤删除
            deleted_count, _ = DcIndexDaily.objects.filter(dc_index__ts_code__in=orphan_list_dc).delete()
            self.stdout.write(self.style.SUCCESS(f"成功从 DcIndexDaily 中删除 {deleted_count} 条孤儿记录。"))
        else:
            self.stdout.write(self.style.SUCCESS("DcIndexDaily 表数据一致，无需清理。"))
        self.stdout.write(self.style.SUCCESS("\n所有孤儿日线数据清理完毕。"))
