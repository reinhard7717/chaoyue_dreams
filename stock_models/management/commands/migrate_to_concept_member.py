# 文件: stock_data/management/commands/migrate_to_concept_member.py

import asyncio
from datetime import datetime, date  # 修改行: 直接导入 date 类型
from django.core.management.base import BaseCommand
from django.db import transaction
from stock_models.industry import (
    ConceptMaster, ConceptMember,
    SwIndustryMember, ThsIndexMember, DcIndexMember, KplConceptConstituent
)
from stock_models.stock_basic import StockInfo

# 定义批量创建的大小，防止一次性加载过多数据到内存
BULK_CREATE_BATCH_SIZE = 5000

class Command(BaseCommand):
    """
    【一次性数据迁移】将所有来源的成分股数据迁移到统一的 ConceptMember 模型中。
    
    运行方式:
    python manage.py migrate_to_concept_member
    
    注意事项:
    1. 运行此脚本前，请务必先成功运行 `migrate_to_concept_master`。
    2. 此脚本会先清空 `ConceptMember` 表，请在确认无误后执行。
    3. 整个过程包裹在数据库事务中，若中途失败，所有更改将自动回滚。
    """
    help = '【一次性数据迁移】将所有来源的成分股数据迁移到统一的 ConceptMember 模型中。'

    def handle(self, *args, **options):
        self.stdout.write(self.style.SUCCESS("====== 开始迁移成分股数据到 ConceptMember ======"))
        
        try:
            # 使用同步的事务管理器包裹整个异步流程，确保数据一致性
            with transaction.atomic():
                asyncio.run(self.async_main())
            
            self.stdout.write(self.style.SUCCESS("\n====== 成分股数据迁移成功！所有操作已提交。 ======"))

        except Exception as e:
            # 捕获任何异常，打印错误信息并提示用户事务已回滚
            self.stderr.write(self.style.ERROR(f"迁移过程中发生严重错误: {e}"))
            self.stdout.write(self.style.WARNING("由于错误发生，数据库事务已自动回滚，未做任何更改。"))

    async def async_main(self):
        """
        异步主逻辑，负责编排整个迁移过程。
        """
        # 1. 清空 ConceptMember 表，确保迁移的幂等性
        deleted_count, _ = await ConceptMember.objects.all().adelete()
        self.stdout.write(f"  - 已清空 ConceptMember 表，删除 {deleted_count} 条旧记录。")

        # 2. 预加载 ConceptMaster 的映射关系，避免 N+1 查询
        self.stdout.write("  - 正在预加载 ConceptMaster 的映射关系...")
        concept_map = {c.code: c.id async for c in ConceptMaster.objects.all()}
        self.stdout.write(f"     ...完成，加载了 {len(concept_map)} 个概念。")

        # 3. 并行执行所有来源的迁移任务
        self.stdout.write("  - 开始并行处理所有数据源...")
        tasks = [
            self.migrate_sw_members(concept_map),
            self.migrate_ths_members(concept_map),
            self.migrate_dc_members(concept_map),
            self.migrate_kpl_members(concept_map),
        ]
        results = await asyncio.gather(*tasks)

        total_migrated = sum(results)
        self.stdout.write(f"\n  - 所有来源处理完毕，总计迁移 {total_migrated} 条成分股记录。")

    async def _parse_date(self, date_val):
        """健壮的日期解析函数，处理字符串、date对象和None值"""
        if not date_val:
            return None
        # 修改行: 直接使用导入的 date 类型进行比较
        if isinstance(date_val, date):
            return date_val
        if isinstance(date_val, str):
            try:
                # 仅处理 YYYYMMDD 格式的字符串
                if len(date_val) == 8:
                     return datetime.strptime(date_val, '%Y%m%d').date()
            except (ValueError, TypeError):
                # 如果字符串格式不正确或不是字符串，则解析失败
                return None
        # 对于其他非字符串、非date类型，返回None
        return None

    async def migrate_sw_members(self, concept_map):
        """迁移申万行业成分"""
        self.stdout.write("  -> 正在迁移 [申万行业] 成分股...")
        count = 0
        members_to_create = []
        # 使用 aiterator() 进行异步流式处理，并用 select_related 优化外键查询
        async for member in SwIndustryMember.objects.select_related('stock').aiterator():
            if not member.stock or not member.stock.stock_code:
                continue
            stock_pk = member.stock.stock_code

            # SwIndustryMember 模型中没有 l3_code 属性，外键字段名为 l3_industry。
            # 使用 `_id` 后缀 (`l3_industry_id`) 是获取外键值的最高效方式。
            l3_code_val = member.l3_industry_id
            
            # 将 L1, L2, L3 的代码一起处理
            for level_code in [member.l1_code, member.l2_code, l3_code_val]:
                concept_id = concept_map.get(level_code)
                if concept_id:
                    members_to_create.append(ConceptMember(
                        concept_id=concept_id,
                        stock_id=stock_pk,
                        source='sw',
                        in_date=await self._parse_date(member.in_date),
                        out_date=await self._parse_date(member.out_date)
                    ))
            
            # 分批次批量创建，防止内存溢出
            if len(members_to_create) >= BULK_CREATE_BATCH_SIZE:
                await ConceptMember.objects.abulk_create(members_to_create, ignore_conflicts=True)
                count += len(members_to_create)
                members_to_create.clear()
        
        # 处理最后一批不足 BULK_CREATE_BATCH_SIZE 的数据
        if members_to_create:
            await ConceptMember.objects.abulk_create(members_to_create, ignore_conflicts=True)
            count += len(members_to_create)

        self.stdout.write(f"     ...完成，处理 {count} 条申万行业成分记录。")
        return count

    async def migrate_ths_members(self, concept_map):
        """迁移同花顺板块成分"""
        self.stdout.write("  -> 正在迁移 [同花顺板块] 成分股...")
        count = 0
        members_to_create = []
        async for member in ThsIndexMember.objects.select_related('ths_index', 'stock').aiterator():
            if not (member.ths_index and member.stock and member.stock.stock_code):
                continue
            concept_id = concept_map.get(member.ths_index.ts_code)
            stock_pk = member.stock.stock_code
            if concept_id and stock_pk:
                members_to_create.append(ConceptMember(
                    concept_id=concept_id,
                    stock_id=stock_pk,
                    source='ths',
                    in_date=member.in_date,
                    out_date=member.out_date
                ))
            
            if len(members_to_create) >= BULK_CREATE_BATCH_SIZE:
                await ConceptMember.objects.abulk_create(members_to_create, ignore_conflicts=True)
                count += len(members_to_create)
                members_to_create.clear()

        if members_to_create:
            await ConceptMember.objects.abulk_create(members_to_create, ignore_conflicts=True)
            count += len(members_to_create)

        self.stdout.write(f"     ...完成，处理 {count} 条同花顺板块成分记录。")
        return count

    async def migrate_dc_members(self, concept_map):
        """迁移东方财富板块成分"""
        self.stdout.write("  -> 正在迁移 [东方财富板块] 成分股...")
        count = 0
        members_to_create = []
        async for member in DcIndexMember.objects.select_related('dc_index', 'stock').aiterator():
            if not (member.dc_index and member.stock and member.stock.stock_code):
                continue
            concept_id = concept_map.get(member.dc_index.ts_code)
            stock_pk = member.stock.stock_code
            if concept_id and stock_pk:
                # 东方财富是每日快照，in_date是当天，out_date为None
                members_to_create.append(ConceptMember(
                    concept_id=concept_id,
                    stock_id=stock_pk,
                    source='dc',
                    in_date=member.trade_time,
                    out_date=None
                ))

            if len(members_to_create) >= BULK_CREATE_BATCH_SIZE:
                await ConceptMember.objects.abulk_create(members_to_create, ignore_conflicts=True)
                count += len(members_to_create)
                members_to_create.clear()

        if members_to_create:
            await ConceptMember.objects.abulk_create(members_to_create, ignore_conflicts=True)
            count += len(members_to_create)

        self.stdout.write(f"     ...完成，处理 {count} 条东方财富板块成分记录。")
        return count

    async def migrate_kpl_members(self, concept_map):
        """迁移开盘啦题材成分"""
        self.stdout.write("  -> 正在迁移 [开盘啦题材] 成分股...")
        count = 0
        members_to_create = []
        async for member in KplConceptConstituent.objects.select_related('concept_info', 'stock').aiterator():
            if not (member.concept_info and member.stock and member.stock.stock_code):
                continue
            concept_id = concept_map.get(member.concept_info.ts_code)
            stock_pk = member.stock.stock_code
            if concept_id and stock_pk:
                # 开盘啦也是每日快照，in_date是当天，out_date为None
                members_to_create.append(ConceptMember(
                    concept_id=concept_id,
                    stock_id=stock_pk,
                    source='kpl',
                    in_date=member.trade_time,
                    out_date=None
                ))

            if len(members_to_create) >= BULK_CREATE_BATCH_SIZE:
                await ConceptMember.objects.abulk_create(members_to_create, ignore_conflicts=True)
                count += len(members_to_create)
                members_to_create.clear()

        if members_to_create:
            await ConceptMember.objects.abulk_create(members_to_create, ignore_conflicts=True)
            count += len(members_to_create)

        self.stdout.write(f"     ...完成，处理 {count} 条开盘啦题材成分记录。")
        return count
