from django.core.management.base import BaseCommand
import redis

from stock_models.stock_basic import StockInfo, StockTimeTrade

class Command(BaseCommand):
    help = '批量生成历史数据的高级别合成数据'

    def add_arguments(self, parser):
        parser.add_argument('--stock_code', type=str, help='指定股票代码，例如 "600000"')
        parser.add_argument('--limit', type=int, default=10, help='处理股票数量上限')

    def handle(self, *args, **options):
        stock_code = options['stock_code']
        limit = options['limit']
        
        # # Redis 连接（可选，用于缓存已处理股票）
        # redis_client = redis.Redis(host='localhost', port=6379, db=0)  # 假设 Redis 已配置
        # processed_key = 'processed_stocks'  # Redis key 用于存储已处理股票列表
        
        if stock_code:
            stocks = StockInfo.objects.filter(stock_code=stock_code)
        else:
            stocks = StockInfo.objects.all()[:limit]  # 限制处理数量，避免超时
        
        for stock in stocks:
            # if redis_client.sismember(processed_key, stock.stock_code):
            #     self.stdout.write(f"股票 {stock.stock_code} 已处理，跳过。")
            #     continue  # 已处理过
            
            StockTimeTrade.generate_higher_level_data(stock)
            # redis_client.sadd(processed_key, stock.stock_code)  # 标记为已处理
            self.stdout.write(f"完成股票 {stock.stock_code} 的历史数据融合。")
        
        self.stdout.write(self.style.SUCCESS('历史数据融合完成。'))
