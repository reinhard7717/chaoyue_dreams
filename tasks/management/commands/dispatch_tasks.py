# your_app/management/commands/dispatch_history_tasks.py
import asyncio
import logging
from django.core.management.base import BaseCommand, CommandError
from celery import group

logger = logging.getLogger(__name__)

class Command(BaseCommand):
    # 更新 help 信息以反映其通用性
    help = '根据指定类型分发处理股票数据的 Celery 任务 (历史/最新 K线/KDJ/MACD)'

    def add_arguments(self, parser):
        """定义命令行参数"""
        # 1. data_type 参数 (必需)
        parser.add_argument(
            'data_type',
            type=str,
            choices=[ # 使用 choices 明确允许的类型
                'history_time_trade', 'latest_time_trade','latest_time_trade_trading_hours',
                'latest_kdj', 'history_kdj',
                'latest_macd', 'history_macd'
            ],
            help='要分发的任务类型 (例如: history_time_trade, latest_kdj)'
        )
        # 2. stock_codes 参数 (可选, 允许多个)
        parser.add_argument(
            '--stock_codes', # 使用 '--' 使其成为可选命名参数
            nargs='*',       # '*' 表示接受零个或多个值，存储在列表中
            type=str,
            help='(可选) 指定要处理的一个或多个股票代码 (空格分隔)。如果省略，则由具体分发逻辑决定 (当前默认处理所有股票)。'
        )

    def handle(self, *args, **options):
        """命令执行入口点"""
        data_type = options['data_type']
        # stock_codes 会是一个列表 (如果提供了参数) 或 None (如果未提供)
        stock_codes = options['stock_codes']

        self.stdout.write(self.style.SUCCESS(f'准备分发任务，类型: {data_type}'))
        if stock_codes:
            self.stdout.write(f"指定处理股票: {', '.join(stock_codes)}")
        else:
            # 根据当前 dispatch_xxx 的实现，这里实际上会处理所有股票
            self.stdout.write("未指定特定股票 (当前将由分发逻辑处理所有股票)")

        try:
            # 直接调用同步的分发路由方法
            self.dispatch_tasks(data_type, stock_codes)
            self.stdout.write(self.style.SUCCESS(f'类型为 "{data_type}" 的任务已分发'))
        except ValueError as ve: # 捕获无效 data_type 的错误
             self.stderr.write(self.style.ERROR(f"错误: {ve}"))
             logger.warning(f"无效的数据类型请求: {data_type}")
        except Exception as e:
            # 记录异常并报告错误
            logger.exception(f'分发类型为 "{data_type}" 的任务时发生意外错误')
            # 使用 CommandError 可以更好地集成 Django 的错误报告机制
            raise CommandError(f'任务分发失败: {str(e)}')
            # 或者保持原有风格:
            # self.stderr.write(self.style.ERROR(f'任务分发失败: {str(e)}'))


    # --- dispatch_tasks 修改为同步方法 ---
    def dispatch_tasks(self, data_type, stock_codes=None):
        """
        根据 data_type 调用相应的具体分发方法。
        注意：目前 stock_codes 参数会被传递，但下游的 dispatch_xxx 方法尚未修改以使用它。
        """
        logger.info(f"路由任务分发请求: data_type='{data_type}', stock_codes={stock_codes}")

        # !!! 重要提醒 !!!
        # 下面的调用会将 stock_codes 传递给 dispatch_xxx 方法。
        # 但由于这些方法目前没有使用 stock_codes 参数来过滤股票，
        # 它们仍然会处理所有股票。你需要后续修改 dispatch_xxx 方法
        # 来实现基于 stock_codes 的过滤。

        if data_type == 'history_time_trade':
            # 调用原始方法，传递参数 (即使它现在可能不用)
            self.dispatch_history_time_trade(stock_codes=stock_codes) # 显式传递
        elif data_type == 'latest_time_trade':
            self.dispatch_latest_time_trade(stock_codes=stock_codes)
        elif data_type == 'latest_time_trade_trading_hours':
            self.dispatch_latest_time_trade_trading_hours(stock_codes=stock_codes)
        elif data_type == 'latest_kdj':
            self.dispatch_latest_kdj(stock_codes=stock_codes)
        elif data_type == 'history_kdj':
            self.dispatch_history_kdj(stock_codes=stock_codes)
        elif data_type == 'latest_macd':
            self.dispatch_latest_macd(stock_codes=stock_codes)
        elif data_type == 'history_macd':
            self.dispatch_history_macd(stock_codes=stock_codes)
        else:
            # 如果 choices 正常工作，这里理论上不会到达，但作为保险
            logger.error(f"接收到未知的 data_type: {data_type}")
            raise ValueError(f"不支持的数据类型: {data_type}")

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

    def dispatch_latest_time_trade_trading_hours(self, *args, **options):
        self.stdout.write("开始分发交易时段最新数据处理任务...")
        logger.info("Management Command 启动: dispatch_latest_time_trade_trading_hours")
        from dao_manager.daos.stock_basic_dao import StockBasicDAO
        from tasks.stock_indicator_tasks import process_single_stock_latest_trade_trading_hours
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
                tasks_signatures.append(process_single_stock_latest_trade_trading_hours.s(stock.stock_code))
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









