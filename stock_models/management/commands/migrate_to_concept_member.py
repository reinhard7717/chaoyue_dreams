# 文件: stock_data/management/commands/migrate_to_concept_member.py

import asyncio
from datetime import datetime, date
from django.core.management.base import BaseCommand
from django.db import transaction
from stock_models.industry import (
    ConceptMaster, ConceptMember,
    SwIndustryMember, ThsIndexMember, DcIndexMember, KplConceptConstituent
)
# from stock_models.stock_basic import StockInfo # 不再直接需要

# 定义批量创建的大小
BULK_CREATE_BATCH_SIZE = 5000

class Command(BaseCommand):
    """
    【V2.0 终极版 - 一次性数据迁移】将所有来源的成分股数据迁移到统一的 ConceptMember 模型中。
    - 核心修正: 对同花顺(ths)来源的迁移逻辑进行了根本性重构，以匹配最新的“当前快照”建模思想。
    运行方式:
    python manage.py migrate_to_concept_member
    """
    help = '【V2.0 终极版】将所有来源的成分股数据迁移到统一的 ConceptMember 模型中。'
    def handle(self, *args, **options):
        self.stdout.write(self.style.SUCCESS("====== 开始迁移成分股数据到 ConceptMember (V2.0 终极版) ======"))
        try:
            with transaction.atomic():
                asyncio.run(self.async_main())
            self.stdout.write(self.style.SUCCESS("\n====== 成分股数据迁移成功！所有操作已提交。 ======"))
        except Exception as e:
            self.stderr.write(self.style.ERROR(f"迁移过程中发生严重错误: {e}"))
            self.stdout.write(self.style.WARNING("由于错误发生，数据库事务已自动回滚，未做任何更改。"))
    async def async_main(self):
        """
        异步主逻辑，负责编排整个迁移过程。
        """
        deleted_count, _ = await ConceptMember.objects.all().adelete()
        self.stdout.write(f"  - 已清空 ConceptMember 表，删除 {deleted_count} 条旧记录。")
        # 预加载 ConceptMaster 和 StockInfo 的映射关系
        # 预加载 ConceptMaster 的 code -> id 映射
        self.stdout.write("  - 正在预加载 ConceptMaster 的映射关系...")
        concept_map = {c.code: c.id async for c in ConceptMaster.objects.all()}
        self.stdout.write(f"     ...完成，加载了 {len(concept_map)} 个概念。")
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
        """健壮的日期解析函数"""
        if isinstance(date_val, date):
            return date_val
        if isinstance(date_val, str) and len(date_val) == 8:
            try:
                return datetime.strptime(date_val, '%Y%m%d').date()
            except (ValueError, TypeError):
                return None
        return None
    async def migrate_sw_members(self, concept_map):
        """迁移申万行业成分"""
        self.stdout.write("  -> 正在迁移 [申万行业] 成分股...")
        count = 0
        members_to_create = []
        # 使用 avalues() 直接获取需要的字段，比 select_related 更高效
        async for member_data in SwIndustryMember.objects.values(
            'l1_code', 'l2_code', 'l3_industry_id', 'stock_id', 'in_date', 'out_date'
        ).aiterator():
            stock_pk = member_data['stock_id']
            if not stock_pk:
                continue
            # 将 L1, L2, L3 的代码一起处理
            for level_code in [member_data['l1_code'], member_data['l2_code'], member_data['l3_industry_id']]:
                concept_id = concept_map.get(level_code)
                if concept_id:
                    in_date_obj = await self._parse_date(member_data['in_date'])
                    # 只有在 in_date 有效时才创建
                    if in_date_obj:
                        members_to_create.append(ConceptMember(
                            concept_id=concept_id,
                            stock_id=stock_pk,
                            source='sw',
                            in_date=in_date_obj,
                            out_date=await self._parse_date(member_data['out_date'])
                        ))
            if len(members_to_create) >= BULK_CREATE_BATCH_SIZE:
                await ConceptMember.objects.abulk_create(members_to_create, ignore_conflicts=True)
                count += len(members_to_create)
                members_to_create.clear()
        if members_to_create:
            await ConceptMember.objects.abulk_create(members_to_create, ignore_conflicts=True)
            count += len(members_to_create)
        self.stdout.write(f"     ...完成，处理 {count} 条申万行业成分记录。")
        return count
    # 彻底重构 migrate_ths_members
    async def migrate_ths_members(self, concept_map):
        """
        【V2.0 终极版】迁移同花顺板块成分
        - 核心逻辑: 将 in_date 设置为代理日期 1990-01-01，以支持历史回溯。
        """
        self.stdout.write("  -> 正在迁移 [同花顺板块] 成分股 (V2.0 终极版)...")
        count = 0
        members_to_create = []
        proxy_in_date = date(1990, 1, 1) # 定义代理“纳入日期”
        # 使用 avalues() 直接获取外键ID，这是最高效的方式
        async for member_data in ThsIndexMember.objects.values('ths_index_id', 'stock_id').aiterator():
            concept_code = member_data['ths_index_id'] # ths_index_id 实际上就是 ts_code
            stock_pk = member_data['stock_id'] # stock_id 实际上就是 stock_code
            concept_id = concept_map.get(concept_code)
            if concept_id and stock_pk:
                members_to_create.append(ConceptMember(
                    concept_id=concept_id,
                    stock_id=stock_pk,
                    source='ths',
                    in_date=proxy_in_date, # 使用代理日期
                    out_date=None # out_date 永远为 None
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
        # 使用 avalues() 提高效率
        async for member_data in DcIndexMember.objects.values('dc_index_id', 'stock_id', 'trade_time').aiterator():
            concept_code = member_data['dc_index_id']
            stock_pk = member_data['stock_id']
            concept_id = concept_map.get(concept_code)
            if concept_id and stock_pk and member_data['trade_time']:
                members_to_create.append(ConceptMember(
                    concept_id=concept_id,
                    stock_id=stock_pk,
                    source='dc',
                    in_date=member_data['trade_time'],
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
        # 使用 avalues() 提高效率
        async for member_data in KplConceptConstituent.objects.values('concept_info_id', 'stock_id', 'trade_time').aiterator():
            concept_code = member_data['concept_info_id']
            stock_pk = member_data['stock_id']
            concept_id = concept_map.get(concept_code)
            if concept_id and stock_pk and member_data['trade_time']:
                members_to_create.append(ConceptMember(
                    concept_id=concept_id,
                    stock_id=stock_pk,
                    source='kpl',
                    in_date=member_data['trade_time'],
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
