# your_app/management/commands/dispatch_history_tasks.py
import asyncio
import logging
from django.core.management.base import BaseCommand
from celery import group

logger = logging.getLogger(__name__)

class Command(BaseCommand):
    help = '分发任务以并发保存所有股票的历史分时/K线数据'

    def handle(self, *args, **options):
        """命令入口点"""
        data_type = options['data_type']
        stock_codes = options['stock_codes']
        # stock_codes = stock_codes_str.split(',') if stock_codes_str else None

        self.stdout.write(self.style.SUCCESS(f'开始获取数据，类型: {data_type}'))
        
        try:
            # self.stdout.write(f'handle - stock_codes: {stock_codes}, stock_codes_type: {type(stock_codes)}')
            asyncio.run(self.fetch_data(data_type, stock_codes))
            self.stdout.write(self.style.SUCCESS('数据获取完成'))
        except Exception as e:
            self.stdout.write(self.style.ERROR(f'数据获取失败: {str(e)}'))
            logger.exception(f'数据获取失败: {str(e)}')

    async def dispatch_tasks(self, data_type, stock_codes=None):
        """根据数据类型分发任务"""
        if data_type == 'history_time_trade':
            await self.dispatch_history_time_trade()
        elif data_type == 'latest_time_trade':
            await self.dispatch_latest_time_trade()
        elif data_type == 'latest_kdj':
            await self.dispatch_latest_kdj()
        elif data_type == 'history_kdj':
            await self.dispatch_history_kdj()
        elif data_type == 'latest_macd':
            await self.dispatch_latest_macd()
        elif data_type == 'history_macd':
            await self.dispatch_history_macd()


    def dispatch_history_time_trade(self, *args, **options):
        self.stdout.write("开始分发历史数据保存任务...")
        logger.info("Management Command 启动: dispatch_history_time_trade")
        from dao_manager.daos.stock_basic_dao import StockBasicDAO
        from tasks.stock_indicator_tasks import process_single_stock_history_trade
        stock_basic_dao = None # 初始化
        try:
            stock_basic_dao = StockBasicDAO()
            # 获取股票列表 (假设 get_stock_list 是 async)
            try:
                # 在同步命令中运行异步 DAO 方法
                stocks = asyncio.run(stock_basic_dao.get_stock_list())
                logger.info(f"获取到 {len(stocks)} 支股票列表")
                self.stdout.write(f"获取到 {len(stocks)} 支股票列表")
            except Exception as e:
                 logger.error(f"获取股票列表时出错: {e}", exc_info=True)
                 self.stderr.write(self.style.ERROR(f"获取股票列表失败: {e}"))
                 return # 获取列表失败，则无法继续
            if not stocks:
                logger.warning("未获取到任何股票信息，任务分发结束")
                self.stdout.write(self.style.WARNING("未获取到任何股票信息，无需分发任务"))
                return
            # 创建子任务签名列表
            tasks_signatures = []
            for stock in stocks:
                # 为每支股票创建一个子任务签名
                tasks_signatures.append(process_single_stock_history_trade.s(stock.stock_code))

            if not tasks_signatures:
                 logger.warning("没有生成任何子任务签名，任务分发结束")
                 self.stdout.write(self.style.WARNING("没有有效的股票生成子任务，无需分发"))
                 return
            # 创建任务组
            task_group = group(tasks_signatures)
            logger.info(f"已创建包含 {len(tasks_signatures)} 个历史数据保存子任务的任务组")
            # 异步执行任务组
            # 注意：这里只是将任务组发送到消息队列，由 Celery worker 异步执行
            group_result = task_group.apply_async()

            logger.info(f"任务组已提交执行，Group ID: {group_result.id}")
            self.stdout.write(self.style.SUCCESS(
                f"成功分发 {len(tasks_signatures)} 个历史数据保存子任务。Group ID: {group_result.id}"
            ))
        except Exception as e:
            logger.error(f"分发任务期间发生错误: {e}", exc_info=True)
            self.stderr.write(self.style.ERROR(f"分发历史数据保存任务失败: {e}"))
        finally:
            # 关闭 DAO
            if stock_basic_dao:
                try:
                     # 假设 close 是异步的
                    if asyncio.iscoroutinefunction(getattr(stock_basic_dao, 'close', None)):
                         asyncio.run(stock_basic_dao.close())
                    elif callable(getattr(stock_basic_dao, 'close', None)):
                         stock_basic_dao.close()
                    logger.debug("Management Command DAO closed.")
                except Exception as close_err:
                    logger.error(f"关闭 Management Command DAO 时出错: {close_err}", exc_info=True)
            logger.info("Management Command 执行流程结束: dispatch_history_tasks")
            self.stdout.write("任务分发流程结束。")

    def dispatch_latest_time_trade(self, *args, **options):
        self.stdout.write("开始分发最新数据处理任务...")
        logger.info("Management Command 启动: dispatch_latest_time_trade")
        from dao_manager.daos.stock_basic_dao import StockBasicDAO
        from tasks.stock_indicator_tasks import process_single_stock_latest_trade
        stock_basic_dao = None # 初始化
        try:
            stock_basic_dao = StockBasicDAO()
            # 获取股票列表 (假设 get_stock_list 是 async)
            try:
                # 在同步命令中运行异步 DAO 方法
                stocks = asyncio.run(stock_basic_dao.get_stock_list())
                logger.info(f"获取到 {len(stocks)} 支股票列表")
                self.stdout.write(f"获取到 {len(stocks)} 支股票列表")
            except Exception as e:
                 logger.error(f"获取股票列表时出错: {e}", exc_info=True)
                 self.stderr.write(self.style.ERROR(f"获取股票列表失败: {e}"))
                 return # 获取列表失败，则无法继续
            if not stocks:
                logger.warning("未获取到任何股票信息，任务分发结束")
                self.stdout.write(self.style.WARNING("未获取到任何股票信息，无需分发任务"))
                return
            # 创建子任务签名列表
            tasks_signatures = []
            for stock in stocks:
                # 为每支股票创建一个处理实时数据的子任务签名
                tasks_signatures.append(process_single_stock_latest_trade.s(stock.stock_code))
            if not tasks_signatures:
                 logger.warning("没有生成任何子任务签名，任务分发结束")
                 self.stdout.write(self.style.WARNING("没有有效的股票生成子任务，无需分发"))
                 return
            # 创建任务组
            task_group = group(tasks_signatures)
            logger.info(f"已创建包含 {len(tasks_signatures)} 个实时数据处理子任务的任务组")
            # 异步执行任务组
            group_result = task_group.apply_async()
            logger.info(f"任务组已提交执行，Group ID: {group_result.id}")
            self.stdout.write(self.style.SUCCESS(
                f"成功分发 {len(tasks_signatures)} 个实时数据处理子任务。Group ID: {group_result.id}"
            ))
        except Exception as e:
            logger.error(f"分发任务期间发生错误: {e}", exc_info=True)
            self.stderr.write(self.style.ERROR(f"分发实时数据处理任务失败: {e}"))
        finally:
            # 关闭 DAO
            if stock_basic_dao:
                try:
                     # 假设 close 是异步的
                    if asyncio.iscoroutinefunction(getattr(stock_basic_dao, 'close', None)):
                         asyncio.run(stock_basic_dao.close())
                    elif callable(getattr(stock_basic_dao, 'close', None)):
                         stock_basic_dao.close()
                    logger.debug("Management Command DAO closed.")
                except Exception as close_err:
                    logger.error(f"关闭 Management Command DAO 时出错: {close_err}", exc_info=True)
            logger.info("Management Command 执行流程结束: dispatch_realtime_tasks")
            self.stdout.write("任务分发流程结束。")

    def dispatch_latest_kdj(self, *args, **options):
        self.stdout.write("开始分发最新KDJ计算任务...")
        logger.info("Management Command 启动: dispatch_latest_kdj")
        from dao_manager.daos.stock_basic_dao import StockBasicDAO
        from tasks.stock_indicator_tasks import process_single_stock_latest_kdj
        stock_basic_dao = None # 初始化 DAO 实例变量
        try:
            # 实例化用于获取股票列表的 DAO
            stock_basic_dao = StockBasicDAO()
            # 获取股票列表
            try:
                # 假设 get_stock_list 是异步方法
                stocks = asyncio.run(stock_basic_dao.get_stock_list())
                stock_count = len(stocks)
                logger.info(f"获取到 {stock_count} 支股票列表")
                self.stdout.write(f"获取到 {stock_count} 支股票列表")
            except Exception as e:
                 logger.error(f"获取股票列表时出错: {e}", exc_info=True)
                 self.stderr.write(self.style.ERROR(f"获取股票列表失败: {e}"))
                 return # 获取列表失败，无法继续
            if not stocks:
                logger.warning("未获取到任何股票信息，任务分发结束")
                self.stdout.write(self.style.WARNING("未获取到任何股票信息，无需分发任务"))
                return
            # 为每支股票创建子任务签名
            tasks_signatures = [process_single_stock_latest_kdj.s(stock.stock_code) for stock in stocks]
            if not tasks_signatures:
                 logger.warning("没有生成任何子任务签名，任务分发结束")
                 self.stdout.write(self.style.WARNING("没有有效的股票生成子任务，无需分发"))
                 return
            # 创建任务组
            task_group = group(tasks_signatures)
            task_count = len(tasks_signatures)
            logger.info(f"已创建包含 {task_count} 个最新KDJ计算子任务的任务组")
            # 异步提交任务组到 Celery 队列
            group_result = task_group.apply_async()
            logger.info(f"任务组已提交执行，Group ID: {group_result.id}")
            self.stdout.write(self.style.SUCCESS(
                f"成功分发 {task_count} 个最新KDJ计算子任务。Group ID: {group_result.id}"
            ))
        except Exception as e:
            logger.error(f"分发KDJ任务期间发生错误: {e}", exc_info=True)
            self.stderr.write(self.style.ERROR(f"分发最新KDJ计算任务失败: {e}"))
        finally:
            # 关闭用于获取股票列表的 DAO
            if stock_basic_dao:
                try:
                    # 检查 close 方法是否是异步的
                    if asyncio.iscoroutinefunction(getattr(stock_basic_dao, 'close', None)):
                         asyncio.run(stock_basic_dao.close())
                         logger.debug("Management Command StockBasicDAO (async) closed.")
                    # 否则，假设它是同步的
                    elif callable(getattr(stock_basic_dao, 'close', None)):
                         stock_basic_dao.close()
                         logger.debug("Management Command StockBasicDAO (sync) closed.")
                except Exception as close_err:
                    logger.error(f"关闭 Management Command StockBasicDAO 时出错: {close_err}", exc_info=True)

            logger.info("Management Command 执行流程结束: dispatch_kdj_tasks")
            self.stdout.write("任务分发流程结束。")

    def dispatch_history_kdj(self, *args, **options):
        self.stdout.write("开始分发历史KDJ计算任务...")
        logger.info("Management Command 启动: dispatch_history_kdj")
        from dao_manager.daos.stock_basic_dao import StockBasicDAO
        from tasks.stock_indicator_tasks import process_single_stock_history_kdj
        stock_basic_dao = None # 初始化 DAO 实例变量
        try:
            # 实例化用于获取股票列表的 DAO
            stock_basic_dao = StockBasicDAO()
            # 获取股票列表
            try:
                # 假设 get_stock_list 是异步方法
                stocks = asyncio.run(stock_basic_dao.get_stock_list())
                stock_count = len(stocks)
                logger.info(f"获取到 {stock_count} 支股票列表")
                self.stdout.write(f"获取到 {stock_count} 支股票列表")
            except Exception as e:
                 logger.error(f"获取股票列表时出错: {e}", exc_info=True)
                 self.stderr.write(self.style.ERROR(f"获取股票列表失败: {e}"))
                 return # 获取列表失败，无法继续
            if not stocks:
                logger.warning("未获取到任何股票信息，任务分发结束")
                self.stdout.write(self.style.WARNING("未获取到任何股票信息，无需分发任务"))
                return
            # 为每支股票创建子任务签名
            tasks_signatures = [process_single_stock_history_kdj.s(stock.stock_code) for stock in stocks]
            if not tasks_signatures:
                 logger.warning("没有生成任何子任务签名，任务分发结束")
                 self.stdout.write(self.style.WARNING("没有有效的股票生成子任务，无需分发"))
                 return
            # 创建任务组
            task_group = group(tasks_signatures)
            task_count = len(tasks_signatures)
            logger.info(f"已创建包含 {task_count} 个历史KDJ计算子任务的任务组")
            # 异步提交任务组到 Celery 队列
            group_result = task_group.apply_async()
            logger.info(f"任务组已提交执行，Group ID: {group_result.id}")
            self.stdout.write(self.style.SUCCESS(
                f"成功分发 {task_count} 个历史KDJ计算子任务。Group ID: {group_result.id}"
            ))
        except Exception as e:
            logger.error(f"分发KDJ任务期间发生错误: {e}", exc_info=True)
            self.stderr.write(self.style.ERROR(f"分发历史KDJ计算任务失败: {e}"))
        finally:
            # 关闭用于获取股票列表的 DAO
            if stock_basic_dao:
                try:
                    # 检查 close 方法是否是异步的
                    if asyncio.iscoroutinefunction(getattr(stock_basic_dao, 'close', None)):
                         asyncio.run(stock_basic_dao.close())
                         logger.debug("Management Command StockBasicDAO (async) closed.")
                    # 否则，假设它是同步的
                    elif callable(getattr(stock_basic_dao, 'close', None)):
                         stock_basic_dao.close()
                         logger.debug("Management Command StockBasicDAO (sync) closed.")
                except Exception as close_err:
                    logger.error(f"关闭 Management Command StockBasicDAO 时出错: {close_err}", exc_info=True)

            logger.info("Management Command 执行流程结束: dispatch_history_kdj_tasks")
            self.stdout.write("任务分发流程结束。")

    def dispatch_latest_macd(self, *args, **options):
        self.stdout.write("开始分发最新MACD计算任务...")
        logger.info("Management Command 启动: dispatch_latest_macd")
        from dao_manager.daos.stock_basic_dao import StockBasicDAO
        from tasks.stock_indicator_tasks import process_single_stock_latest_macd
        stock_basic_dao = None # 初始化 DAO 实例变量
        try:
            # 实例化用于获取股票列表的 DAO
            stock_basic_dao = StockBasicDAO()
            # 获取股票列表
            try:
                # 假设 get_stock_list 是异步方法
                stocks = asyncio.run(stock_basic_dao.get_stock_list())
                stock_count = len(stocks)
                logger.info(f"获取到 {stock_count} 支股票列表")
                self.stdout.write(f"获取到 {stock_count} 支股票列表")
            except Exception as e:
                 logger.error(f"获取股票列表时出错: {e}", exc_info=True)
                 self.stderr.write(self.style.ERROR(f"获取股票列表失败: {e}"))
                 return # 获取列表失败，无法继续
            if not stocks:
                logger.warning("未获取到任何股票信息，任务分发结束")
                self.stdout.write(self.style.WARNING("未获取到任何股票信息，无需分发任务"))
                return
            # 为每支股票创建子任务签名
            tasks_signatures = [process_single_stock_latest_macd.s(stock.stock_code) for stock in stocks]
            if not tasks_signatures:
                 logger.warning("没有生成任何子任务签名，任务分发结束")
                 self.stdout.write(self.style.WARNING("没有有效的股票生成子任务，无需分发"))
                 return
            # 创建任务组
            task_group = group(tasks_signatures)
            task_count = len(tasks_signatures)
            logger.info(f"已创建包含 {task_count} 个最新MACD计算子任务的任务组")
            # 异步提交任务组到 Celery 队列
            group_result = task_group.apply_async()
            logger.info(f"任务组已提交执行，Group ID: {group_result.id}")
            self.stdout.write(self.style.SUCCESS(
                f"成功分发 {task_count} 个最新MACD计算子任务。Group ID: {group_result.id}"
            ))
        except Exception as e:
            logger.error(f"分发MACD任务期间发生错误: {e}", exc_info=True)
            self.stderr.write(self.style.ERROR(f"分发最新MACD计算任务失败: {e}"))
        finally:
            # 关闭用于获取股票列表的 DAO
            if stock_basic_dao:
                try:
                    # 检查 close 方法是否是异步的
                    if asyncio.iscoroutinefunction(getattr(stock_basic_dao, 'close', None)):
                         asyncio.run(stock_basic_dao.close())
                         logger.debug("Management Command StockBasicDAO (async) closed.")
                    # 否则，假设它是同步的
                    elif callable(getattr(stock_basic_dao, 'close', None)):
                         stock_basic_dao.close()
                         logger.debug("Management Command StockBasicDAO (sync) closed.")
                except Exception as close_err:
                    logger.error(f"关闭 Management Command StockBasicDAO 时出错: {close_err}", exc_info=True)

            logger.info("Management Command 执行流程结束: dispatch_latest_macd_tasks")
            self.stdout.write("任务分发流程结束。")

    def dispatch_history_macd(self, *args, **options):
        self.stdout.write("开始分发历史MACD计算任务...")
        logger.info("Management Command 启动: dispatch_history_macd")
        from dao_manager.daos.stock_basic_dao import StockBasicDAO
        from tasks.stock_indicator_tasks import process_single_stock_history_macd
        stock_basic_dao = None # 初始化 DAO 实例变量
        try:
            # 实例化用于获取股票列表的 DAO
            stock_basic_dao = StockBasicDAO()
            # 获取股票列表
            try:
                # 假设 get_stock_list 是异步方法
                stocks = asyncio.run(stock_basic_dao.get_stock_list())
                stock_count = len(stocks)
                logger.info(f"获取到 {stock_count} 支股票列表")
                self.stdout.write(f"获取到 {stock_count} 支股票列表")
            except Exception as e:
                 logger.error(f"获取股票列表时出错: {e}", exc_info=True)
                 self.stderr.write(self.style.ERROR(f"获取股票列表失败: {e}"))
                 return # 获取列表失败，无法继续
            if not stocks:
                logger.warning("未获取到任何股票信息，任务分发结束")
                self.stdout.write(self.style.WARNING("未获取到任何股票信息，无需分发任务"))
                return
            # 为每支股票创建子任务签名
            tasks_signatures = [process_single_stock_history_macd.s(stock.stock_code) for stock in stocks]
            if not tasks_signatures:
                 logger.warning("没有生成任何子任务签名，任务分发结束")
                 self.stdout.write(self.style.WARNING("没有有效的股票生成子任务，无需分发"))
                 return
            # 创建任务组
            task_group = group(tasks_signatures)
            task_count = len(tasks_signatures)
            logger.info(f"已创建包含 {task_count} 个历史MACD计算子任务的任务组")
            # 异步提交任务组到 Celery 队列
            group_result = task_group.apply_async()
            logger.info(f"任务组已提交执行，Group ID: {group_result.id}")
            self.stdout.write(self.style.SUCCESS(
                f"成功分发 {task_count} 个历史MACD计算子任务。Group ID: {group_result.id}"
            ))
        except Exception as e:
            logger.error(f"分发MACD任务期间发生错误: {e}", exc_info=True)
            self.stderr.write(self.style.ERROR(f"分发历史MACD计算任务失败: {e}"))
        finally:
            # 关闭用于获取股票列表的 DAO
            if stock_basic_dao:
                try:
                    # 检查 close 方法是否是异步的
                    if asyncio.iscoroutinefunction(getattr(stock_basic_dao, 'close', None)):
                         asyncio.run(stock_basic_dao.close())
                         logger.debug("Management Command StockBasicDAO (async) closed.")
                    # 否则，假设它是同步的
                    elif callable(getattr(stock_basic_dao, 'close', None)):
                         stock_basic_dao.close()
                         logger.debug("Management Command StockBasicDAO (sync) closed.")
                except Exception as close_err:
                    logger.error(f"关闭 Management Command StockBasicDAO 时出错: {close_err}", exc_info=True)

            logger.info("Management Command 执行流程结束: dispatch_history_macd_tasks")
            self.stdout.write("任务分发流程结束。")









