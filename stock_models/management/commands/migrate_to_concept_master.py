# 文件: stock_data/management/commands/migrate_to_concept_master.py

import asyncio
from django.core.management.base import BaseCommand
from django.db import transaction
from asgiref.sync import sync_to_async
from stock_models.industry import ConceptMaster, SwIndustry, ThsIndex, DcIndex, KplConceptInfo

class Command(BaseCommand):
    help = '【一次性数据迁移】将所有来源的行业/概念数据迁移到统一的 ConceptMaster 模型中。'

    # 修改行: 移除 async 关键字，handle 方法本身是同步的
    def handle(self, *args, **options):
        self.stdout.write(self.style.SUCCESS("====== 开始迁移数据到 ConceptMaster ======"))
        
        # 修改行: 使用 asyncio.run() 来执行我们的异步主逻辑
        try:
            asyncio.run(self.async_main())
        except Exception as e:
            self.stderr.write(self.style.ERROR(f"迁移过程中发生严重错误: {e}"))
            self.stdout.write(self.style.WARNING("如果错误发生在数据库操作中，可能需要手动检查数据一致性。"))

    # 新增方法: 将原来的 handle 逻辑封装到一个异步方法中
    async def async_main(self):
        try:
            # 使用事务确保数据一致性
            async with transaction.atomic():
                # 1. 清空 ConceptMaster 表，防止重复执行导致数据错误
                deleted_count, _ = await ConceptMaster.objects.all().adelete() # 使用异步删除
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
                self.stdout.write(self.style.SUCCESS(f"\n====== 数据迁移成功！共迁移 {total_migrated} 条记录。 ======"))

        except Exception as e:
            # 重新抛出异常，让外层的同步 handle 捕获并打印
            raise e

    async def migrate_sw_industry(self):
        self.stdout.write("  -> 正在迁移 [申万行业] 数据...")
        sw_industries = await sync_to_async(list)(SwIndustry.objects.all())
        concepts_to_create = [
            ConceptMaster(
                code=ind.index_code,
                name=ind.industry_name,
                source='sw',
                type=f"L{ind.level}" # 存储层级信息
            )
            for ind in sw_industries
        ]
        if concepts_to_create:
            await ConceptMaster.objects.abulk_create(concepts_to_create, ignore_conflicts=True)
        self.stdout.write(f"     ...完成，处理 {len(concepts_to_create)} 条申万行业记录。")
        return len(concepts_to_create)

    async def migrate_ths_index(self):
        self.stdout.write("  -> 正在迁移 [同花顺板块] 数据...")
        ths_indices = await sync_to_async(list)(ThsIndex.objects.all())
        concepts_to_create = [
            ConceptMaster(
                code=ind.ts_code,
                name=ind.name,
                source='ths',
                type=ind.type # 同花顺自带类型
            )
            for ind in ths_indices
        ]
        if concepts_to_create:
            await ConceptMaster.objects.abulk_create(concepts_to_create, ignore_conflicts=True)
        self.stdout.write(f"     ...完成，处理 {len(concepts_to_create)} 条同花顺板块记录。")
        return len(concepts_to_create)

    async def migrate_dc_index(self):
        self.stdout.write("  -> 正在迁移 [东方财富板块] 数据...")
        dc_indices = await sync_to_async(list)(DcIndex.objects.all())
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
        kpl_concepts = await sync_to_async(list)(KplConceptInfo.objects.all())
        concepts_to_create = [
            ConceptMaster(
                code=cpt.ts_code,
                name=cpt.name,
                source='kpl',
                type='题材' # 开盘啦都是题材
            )
            for cpt in kpl_concepts
        ]
        if concepts_to_create:
            await ConceptMaster.objects.abulk_create(concepts_to_create, ignore_conflicts=True)
        self.stdout.write(f"     ...完成，处理 {len(concepts_to_create)} 条开盘啦题材记录。")
        return len(concepts_to_create)
