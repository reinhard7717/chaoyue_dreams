# 文件: stock_data/management/commands/migrate_to_concept_master.py

import asyncio
from django.core.management.base import BaseCommand
from django.db import transaction
from asgiref.sync import sync_to_async
from stock_models.industry import ConceptMaster, SwIndustry, ThsIndex, DcIndex, KplConceptInfo

class Command(BaseCommand):
    help = '【一次性数据迁移】将所有来源的行业/概念数据迁移到统一的 ConceptMaster 模型中。'
    def handle(self, *args, **options):
        self.stdout.write(self.style.SUCCESS("====== 开始迁移数据到 ConceptMaster ======"))
        try:
            # --- 修改行开始: 使用同步的事务管理器 ---
            # 将事务管理放在同步的 handle 方法中
            with transaction.atomic():
                # 在同步事务块内部，运行我们的异步主逻辑
                asyncio.run(self.async_main())
            self.stdout.write(self.style.SUCCESS("\n====== 数据迁移成功！所有操作已提交。 ======"))
        except Exception as e:
            self.stderr.write(self.style.ERROR(f"迁移过程中发生严重错误: {e}"))
            self.stdout.write(self.style.WARNING("由于错误发生，数据库事务已自动回滚，未做任何更改。"))
    async def async_main(self):
        """
        异步主逻辑，现在它只负责业务操作，不再关心事务。
        """
        # 1. 清空 ConceptMaster 表，防止重复执行导致数据错误
        deleted_count, _ = await ConceptMaster.objects.all().adelete()
        self.stdout.write(f"  - 已清空 ConceptMaster 表，删除 {deleted_count} 条旧记录。")
        # 2. 并行执行所有来源的迁移任务
        tasks = [
            self.migrate_sw_industry(),
            self.migrate_ths_index(),
            self.migrate_dc_index(),
            self.migrate_kpl_concept(),
        ]
        results = await asyncio.gather(*tasks)
        total_migrated = sum(results)
        # 成功信息移到 handle 方法中，确保事务提交后才显示
        # self.stdout.write(self.style.SUCCESS(f"\n====== 数据迁移成功！共迁移 {total_migrated} 条记录。 ======"))

    async def migrate_sw_industry(self):
        self.stdout.write("  -> 正在迁移 [申万行业] 数据...")
        # 使用 aall() 替代 sync_to_async(list) 以获得更好的异步性能
        sw_industries = [ind async for ind in SwIndustry.objects.all()]
        concepts_to_create = [
            ConceptMaster(
                code=ind.index_code,
                name=ind.industry_name,
                source='sw',
                type=f"L{ind.level}"
            )
            for ind in sw_industries
        ]
        if concepts_to_create:
            await ConceptMaster.objects.abulk_create(concepts_to_create, ignore_conflicts=True)
        self.stdout.write(f"     ...完成，处理 {len(concepts_to_create)} 条申万行业记录。")
        return len(concepts_to_create)
    async def migrate_ths_index(self):
        self.stdout.write("  -> 正在迁移 [同花顺板块] 数据...")
        ths_indices = [ind async for ind in ThsIndex.objects.all()]
        concepts_to_create = [
            ConceptMaster(
                code=ind.ts_code,
                name=ind.name,
                source='ths',
                type=ind.type
            )
            for ind in ths_indices
        ]
        if concepts_to_create:
            await ConceptMaster.objects.abulk_create(concepts_to_create, ignore_conflicts=True)
        self.stdout.write(f"     ...完成，处理 {len(concepts_to_create)} 条同花顺板块记录。")
        return len(concepts_to_create)
    async def migrate_dc_index(self):
        self.stdout.write("  -> 正在迁移 [东方财富板块] 数据...")
        dc_indices = [ind async for ind in DcIndex.objects.all()]
        concepts_to_create = [
            ConceptMaster(
                code=ind.ts_code,
                name=ind.name,
                source='dc',
                type=ind.type
            )
            for ind in dc_indices
        ]
        if concepts_to_create:
            await ConceptMaster.objects.abulk_create(concepts_to_create, ignore_conflicts=True)
        self.stdout.write(f"     ...完成，处理 {len(concepts_to_create)} 条东方财富板块记录。")
        return len(concepts_to_create)
    async def migrate_kpl_concept(self):
        self.stdout.write("  -> 正在迁移 [开盘啦题材] 数据...")
        kpl_concepts = [cpt async for cpt in KplConceptInfo.objects.all()]
        concepts_to_create = [
            ConceptMaster(
                code=cpt.ts_code,
                name=cpt.name,
                source='kpl',
                type='题材'
            )
            for cpt in kpl_concepts
        ]
        if concepts_to_create:
            await ConceptMaster.objects.abulk_create(concepts_to_create, ignore_conflicts=True)
        self.stdout.write(f"     ...完成，处理 {len(concepts_to_create)} 条开盘啦题材记录。")
        return len(concepts_to_create)

