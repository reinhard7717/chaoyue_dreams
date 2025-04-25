from rest_framework import serializers
import umsgpack

from stock_models.stock_basic import StockInfo
from users.models import FavoriteStock
from utils import cache_constants as cc
from utils.cache_manager import CacheManager
from utils.cash_key import StockCashKey


class IndexInfoSerializer(serializers.ModelSerializer):
    class Meta:
        from stock_models.index import IndexInfo
        model = IndexInfo
        fields = '__all__'

class StockInfoSerializer(serializers.ModelSerializer):
    class Meta:
        from stock_models.stock_basic import StockInfo
        model = StockInfo
        fields = ['stock_code', 'stock_name'] # 只序列化代码和名称

class FavoriteStockSerializer(serializers.ModelSerializer):
    stock = StockInfoSerializer(read_only=True) # 输出时显示 stock 详情
    stock_code = serializers.CharField(write_only=True, required=True, label="股票代码")

    class Meta:
        model = FavoriteStock
        # fields 列表现在只包含模型字段和只读字段，以及上面定义的 write_only 字段
        fields = ['id', 'stock', 'added_at', 'stock_code']
        read_only_fields = ['id', 'added_at', 'stock']

    def create(self, validated_data):
        stock_code_data = validated_data.pop('stock_code')
        user = self.context['request'].user
        try:
            stock_instance = StockInfo.objects.get(stock_code=stock_code_data)
        except StockInfo.DoesNotExist:
            raise serializers.ValidationError({"stock_code": [f"股票代码 {stock_code_data} 不存在。"]})
        except StockInfo.MultipleObjectsReturned:
             raise serializers.ValidationError({"stock_code": [f"找到多个股票代码为 {stock_code_data} 的记录，数据异常。"]})

        if FavoriteStock.objects.filter(user=user, stock=stock_instance).exists():
             raise serializers.ValidationError({"detail": "该股票已在自选列表中。"})

        # validated_data 现在可能为空，或者包含 FavoriteStock 的其他可写字段 (如果你的模型有的话)
        favorite = FavoriteStock.objects.create(stock=stock_instance, **validated_data)
        return favorite

class StockRealtimeDataSerializer(serializers.ModelSerializer):
    class Meta:
        from stock_models.stock_realtime import StockRealtimeData
        model = StockRealtimeData
        fields = '__all__'

class StockLevel5DataSerializer(serializers.ModelSerializer):
    class Meta:
        from stock_models.stock_realtime import StockLevel5Data
        model = StockLevel5Data
        fields = '__all__'

class StockTradeDetailSerializer(serializers.ModelSerializer):
    class Meta:
        from stock_models.stock_realtime import StockTradeDetail
        model = StockTradeDetail
        fields = '__all__'

class StockTimeDealSerializer(serializers.ModelSerializer):
    class Meta:
        from stock_models.stock_realtime import StockTimeDeal
        model = StockTimeDeal
        fields = '__all__'


class StockPricePercentSerializer(serializers.ModelSerializer):
    class Meta:
        from stock_models.stock_realtime import StockPricePercent
        model = StockPricePercent
        fields = '__all__'

class StockBigDealSerializer(serializers.ModelSerializer):
    class Meta:
        from stock_models.stock_realtime import StockBigDeal
        model = StockBigDeal
        fields = '__all__'

class StockAbnormalMovementSerializer(serializers.ModelSerializer):
    class Meta:
        from stock_models.stock_realtime import StockAbnormalMovement
        model = StockAbnormalMovement
        fields = '__all__'
        
        
        
