# tasks\tushare\industry_tasks.py

import datetime
import logging
import asyncio
from asgiref.sync import async_to_sync
from utils.task_helpers import with_cache_manager
# 假设 StockBasicInfoDao 存在且可用
from dao_manager.tushare_daos.index_basic_dao import IndexBasicDAO
from dao_manager.tushare_daos.industry_dao import IndustryDao

# 假设 celery 实例存在且可用
from chaoyue_dreams.celery import app as celery_app
from utils.cache_manager import CacheManager

logger = logging.getLogger("tasks")

# 获取本周一和本周五的日期
def get_this_monday_and_friday():
    """获取本周一和本周五的日期"""
    today = datetime.date.today()
    this_monday = today - datetime.timedelta(days=today.weekday())
    this_friday = this_monday + datetime.timedelta(days=4)
    return this_monday, this_friday

# 获取上周一和上周五的日期
def get_last_monday_and_friday():
    """获取上周一和上周五的日期"""
    today = datetime.date.today()
    this_monday = today - datetime.timedelta(days=today.weekday())
    last_monday = this_monday - datetime.timedelta(days=7)
    last_friday = last_monday + datetime.timedelta(days=4)
    return last_monday, last_friday

# 每日任务：全面获取所有行业、概念及市场情绪数据
@celery_app.task(name='tasks.tushare.industry_tasks.save_all_daily_industry_concept_data_task', queue='SaveHistoryData_TimeTrade')
@with_cache_manager
def save_all_daily_industry_concept_data_task(cache_manager=None):
    """
    【V2.0 综合版】
    Celery调度器任务：每日盘后执行，保存所有渠道的行业、概念、题材及市场情绪数据。
    - 核心升级: 将原有的同花顺任务扩展为覆盖所有数据源的综合任务。
    - 健壮性增强: 为每个数据源的获取过程添加了独立的异常捕获，防止单点失败导致整个任务中断。
    - 覆盖范围:
      - 申万 (SW): 行业分类、成分、日线行情
      - 同花顺 (THS): 板块指数、成分、日线行情
      - 开盘啦 (KPL): 题材库、题材成分、榜单数据
      - 市场情绪: Tushare版涨跌停、同花顺版涨跌停、连板天梯、最强板块
      - 东方财富 (DC): 板块日线行情 (注: 成分股需单独任务循环获取，此处暂不包含)
    """
    task_name = 'save_all_daily_industry_concept_data_task'
    logger.info(f"开始执行【全面获取所有行业、概念及市场情绪数据】任务: {task_name}")
    industry_dao = IndustryDao(cache_manager)
    today = datetime.date.today()
    async def main():
        # --- 1. 申万数据 ---
        try:
            print("\n--- [任务块 1/5] 开始处理【申万】相关数据 ---")
            logger.info("--- [任务块 1/5] 开始处理【申万】相关数据 ---")
            # 1.1 申万行业分类 (不常变，但每日更新可确保最新)
            sw_list_res = await industry_dao.save_swan_industry_list()
            print(f"  - 保存申万行业分类完成，结果: {sw_list_res}")
            logger.info(f"  - 保存申万行业分类完成，结果: {sw_list_res}")
            # 1.2 申万行业成分 (不常变，但每日更新可确保最新)
            sw_member_res = await industry_dao.save_sw_industry_member()
            print(f"  - 保存申万行业成分完成，结果: {sw_member_res}")
            logger.info(f"  - 保存申万行业成分完成，结果: {sw_member_res}")
            # 1.3 申万行业当日行情
            sw_daily_res = await industry_dao.save_sw_industry_daily(trade_time=today)
            print(f"  - 保存申万行业当日({today})行情完成，结果: {sw_daily_res}")
            logger.info(f"  - 保存申万行业当日({today})行情完成，结果: {sw_daily_res}")
        except Exception as e:
            logger.error(f"[{task_name}] 处理【申万】数据时发生错误: {e}", exc_info=True)
        # --- 2. 同花顺数据 ---
        try:
            print("\n--- [任务块 2/5] 开始处理【同花顺】相关数据 ---")
            logger.info("--- [任务块 2/5] 开始处理【同花顺】相关数据 ---")
            # 2.1 同花顺板块指数列表
            ths_list_res = await industry_dao.save_ths_index_list()
            print(f"  - 保存同花顺板块指数列表完成，结果: {ths_list_res}")
            logger.info(f"  - 保存同花顺板块指数列表完成，结果: {ths_list_res}")
            # 2.2 同花顺板块成分
            ths_member_res = await industry_dao.save_ths_index_member()
            print(f"  - 保存同花顺板块成分完成，结果: {ths_member_res}")
            logger.info(f"  - 保存同花顺板块成分完成，结果: {ths_member_res}")
            # 2.3 同花顺板块当日行情
            ths_daily_res = await industry_dao.save_ths_index_daily_by_trade_date(trade_date=today)
            print(f"  - 保存同花顺板块当日({today})行情完成，结果: {ths_daily_res}")
            logger.info(f"  - 保存同花顺板块当日({today})行情完成，结果: {ths_daily_res}")
        except Exception as e:
            logger.error(f"[{task_name}] 处理【同花顺】数据时发生错误: {e}", exc_info=True)
        # --- 3. 开盘啦数据 ---
        try:
            print("\n--- [任务块 3/5] 开始处理【开盘啦】相关数据 ---")
            logger.info("--- [任务块 3/5] 开始处理【开盘啦】相关数据 ---")
            # 3.1 开盘啦题材库
            kpl_concept_res = await industry_dao.save_kpl_concept_list_by_date(trade_date=today)
            print(f"  - 保存开盘啦当日({today})题材库完成，结果: {kpl_concept_res}")
            logger.info(f"  - 保存开盘啦当日({today})题材库完成，结果: {kpl_concept_res}")
            # 3.2 开盘啦题材成分
            kpl_member_res = await industry_dao.save_kpl_concept_members_by_date(trade_date=today)
            print(f"  - 保存开盘啦当日({today})题材成分完成，结果: {kpl_member_res}")
            logger.info(f"  - 保存开盘啦当日({today})题材成分完成，结果: {kpl_member_res}")
            # 3.3 开盘啦榜单数据
            kpl_list_res = await industry_dao.save_kpl_list_by_date(trade_date=today)
            print(f"  - 保存开盘啦当日({today})榜单数据完成，结果: {kpl_list_res}")
            logger.info(f"  - 保存开盘啦当日({today})榜单数据完成，结果: {kpl_list_res}")
        except Exception as e:
            logger.error(f"[{task_name}] 处理【开盘啦】数据时发生错误: {e}", exc_info=True)
        # --- 4. 东方财富数据 ---
        try:
            print("\n--- [任务块 4/5] 开始处理【东方财富】相关数据 ---")
            logger.info("--- [任务块 4/5] 开始处理【东方财富】相关数据 ---")
            # 4.1 更新东方财富板块列表 (主表)
            dc_list_res = await industry_dao.save_dc_index_list_by_date(trade_date=today)
            print(f"  - 保存东方财富板块列表当日({today})完成，结果: {dc_list_res}")
            logger.info(f"  - 保存东方财富板块列表当日({today})完成，结果: {dc_list_res}")
            # 4.2 获取东方财富板块当日行情
            dc_daily_res = await industry_dao.save_dc_index_daily_by_trade_time(trade_time=today)
            print(f"  - 保存东方财富板块当日({today})行情完成，结果: {dc_daily_res}")
            logger.info(f"  - 保存东方财富板块当日({today})行情完成，结果: {dc_daily_res}")
            # 4.3 获取东方财富板块当日成分
            dc_member_res = await industry_dao.save_dc_index_members_by_date(trade_date=today)
            print(f"  - 保存东方财富板块当日({today})成分完成，结果: {dc_member_res}")
            logger.info(f"  - 保存东方财富板块当日({today})成分完成，结果: {dc_member_res}")
        except Exception as e:
            logger.error(f"[{task_name}] 处理【东方财富】数据时发生错误: {e}", exc_info=True)
        # --- 5. 市场情绪数据 (涨跌停等) ---
        try:
            print("\n--- [任务块 5/5] 开始处理【市场情绪】相关数据 ---")
            logger.info("--- [任务块 5/5] 开始处理【市场情绪】相关数据 ---")
            # 5.1 同花顺涨跌停榜单
            limit_ths_res = await industry_dao.save_limit_list_ths_by_date(trade_date=today)
            print(f"  - 保存同花顺涨跌停榜单当日({today})数据完成，结果: {limit_ths_res}")
            logger.info(f"  - 保存同花顺涨跌停榜单当日({today})数据完成，结果: {limit_ths_res}")
            # 5.2 Tushare版A股涨跌停列表
            limit_d_res = await industry_dao.save_limit_list_d_by_date(trade_date=today)
            print(f"  - 保存Tushare涨跌停列表当日({today})数据完成，结果: {limit_d_res}")
            logger.info(f"  - 保存Tushare涨跌停列表当日({today})数据完成，结果: {limit_d_res}")
            # 5.3 连板天梯
            limit_step_res = await industry_dao.save_limit_step_by_date(trade_date=today)
            print(f"  - 保存连板天梯当日({today})数据完成，结果: {limit_step_res}")
            logger.info(f"  - 保存连板天梯当日({today})数据完成，结果: {limit_step_res}")
            # 5.4 最强板块统计
            limit_cpt_res = await industry_dao.save_limit_cpt_list_by_date(trade_date=today)
            print(f"  - 保存最强板块统计当日({today})数据完成，结果: {limit_cpt_res}")
            logger.info(f"  - 保存最强板块统计当日({today})数据完成，结果: {limit_cpt_res}")
        except Exception as e:
            logger.error(f"[{task_name}] 处理【市场情绪】数据时发生错误: {e}", exc_info=True)
        # --- 6. 完成 ---
        final_message = f"【全面获取所有行业、概念及市场情绪数据】任务: {task_name} 全部数据块执行完毕。"
        logger.info(final_message)
        return final_message

# 历史数据回补任务：全面获取所有渠道的历史数据
@celery_app.task(name='tasks.tushare.industry_tasks.save_all_historical_data_task', queue='SaveHistoryData_TimeTrade')
@with_cache_manager
def save_all_historical_data_task(cache_manager=None, days_to_fetch: int = 30):
    """
    【V2.0 综合并行版】
    Celery调度器任务：一次性获取并保存所有渠道过去N天的历史数据。
    - 核心升级:
      - 将原同花顺历史任务扩展为覆盖所有渠道的综合历史回补任务。
      - 采用按天并行处理的模式，极大提升数据回补效率。
    - 健壮性增强: 为每个渠道的单日数据获取添加了独立的异常捕获。
    - :param days_to_fetch: 需要回补的交易日天数，默认为30天。
    """
    task_name = 'save_all_historical_data_task'
    logger.info(f"开始执行【全面获取所有渠道的历史数据】任务: {task_name}，回补天数: {days_to_fetch}")
    industry_dao = IndustryDao(cache_manager)
    index_dao = IndexBasicDAO(cache_manager)
    async def main():
        # --- 步骤1: 获取需要回补的交易日列表 ---
        trade_dates = await index_dao.get_last_n_trade_cal_open(n=days_to_fetch)
        if not trade_dates:
            logger.warning("未能获取到任何交易日，历史数据回补任务终止。")
            return
        print(f"准备回补以下 {len(trade_dates)} 个交易日的数据: { [d.strftime('%Y-%m-%d') for d in trade_dates] }")
        logger.info(f"准备回补 {len(trade_dates)} 个交易日的数据。")
        # --- 步骤2: 静态数据更新 (这些数据不按天变化，只需执行一次) ---
        print("\n--- [静态数据更新] 开始处理不按天变化的元数据 ---")
        # try:
        #     # 申万行业分类
        #     await industry_dao.save_swan_industry_list()
        #     print("  - 申万行业分类元数据更新完成。")
        #     # 申万行业成分
        #     await industry_dao.save_sw_industry_member()
        #     print("  - 申万行业成分元数据更新完成。")
        #     # 同花顺板块指数列表
        #     await industry_dao.save_ths_index_list()
        #     print("  - 同花顺板块指数元数据更新完成。")
        #     # 同花顺板块成分
        #     await industry_dao.save_ths_index_member()
        #     print("  - 同花顺板块成分元数据更新完成。")
        # except Exception as e:
        #     logger.error(f"[{task_name}] 更新静态元数据时发生错误: {e}", exc_info=True)
        # --- 步骤3: 按天并行处理所有渠道的日度数据 ---
        for i, trade_date in enumerate(trade_dates):
            print(f"\n--- [进度 {i+1}/{len(trade_dates)}] 开始处理交易日: {trade_date.strftime('%Y-%m-%d')} 的所有数据 ---")
            # 为当前交易日创建一组并发任务
            tasks = []
            # 申万
            tasks.append(run_safely(industry_dao.save_sw_industry_daily, "申万行业行情", trade_date=trade_date))
            # 同花顺
            tasks.append(run_safely(industry_dao.save_ths_index_daily_by_trade_date, "同花顺板块行情", trade_date=trade_date))
            # 开盘啦
            tasks.append(run_safely(industry_dao.save_kpl_concept_list_by_date, "开盘啦题材库", trade_date=trade_date))
            tasks.append(run_safely(industry_dao.save_kpl_concept_members_by_date, "开盘啦题材成分", trade_date=trade_date))
            tasks.append(run_safely(industry_dao.save_kpl_list_by_date, "开盘啦榜单", trade_date=trade_date))
            # 东方财富
            tasks.append(run_safely(industry_dao.save_dc_index_list_by_date, "东方财富板块列表", trade_date=trade_date))
            tasks.append(run_safely(industry_dao.save_dc_index_daily_by_trade_time, "东方财富板块行情", trade_time=trade_date))
            tasks.append(run_safely(industry_dao.save_dc_index_members_by_date, "东方财富板块成分", trade_date=trade_date))
            # 市场情绪
            tasks.append(run_safely(industry_dao.save_limit_list_ths_by_date, "同花顺涨跌停榜单", trade_date=trade_date))
            tasks.append(run_safely(industry_dao.save_limit_list_d_by_date, "Tushare涨跌停列表", trade_date=trade_date))
            tasks.append(run_safely(industry_dao.save_limit_step_by_date, "连板天梯", trade_date=trade_date))
            tasks.append(run_safely(industry_dao.save_limit_cpt_list_by_date, "最强板块统计", trade_date=trade_date))
            # 并发执行当天的所有任务
            await asyncio.gather(*tasks)
    async def run_safely(coro, name, **kwargs):
        """一个安全的协程运行器，用于捕获单个任务的异常"""
        trade_date_str = kwargs.get('trade_date', kwargs.get('trade_time', 'N/A'))
        if isinstance(trade_date_str, datetime.date):
            trade_date_str = trade_date_str.strftime('%Y-%m-%d')
        try:
            print(f"    -> 开始获取 [{name}] 数据...")
            await coro(**kwargs)
            print(f"    -- 完成 [{name}] 数据获取。")
        except Exception as e:
            logger.error(f"在处理日期 {trade_date_str} 的 [{name}] 数据时发生错误: {e}", exc_info=True)
            print(f"    !! 失败 [{name}] 数据获取失败: {e}")
    async_to_sync(main)()
    final_message = f"【全面获取所有渠道的历史数据】任务: {task_name} 全部交易日处理完毕。"
    logger.info(final_message)
    return final_message










