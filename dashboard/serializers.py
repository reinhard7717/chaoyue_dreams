from rest_framework import serializers
import umsgpack
from django.utils import timezone
from stock_models.stock_basic import StockInfo
from users.models import FavoriteStock
from utils import cache_constants as cc
from stock_models.stock_analytics import Transaction
from stock_models.stock_analytics import PositionTracker


# stock_models\financial.py
class IncomeSerializer(serializers.ModelSerializer):
    class Meta:
        from stock_models.financial import Income
        model = Income
        fields = '__all__'

class BalanceSheetSerializer(serializers.ModelSerializer):
    class Meta:
        from stock_models.financial import BalanceSheet
        model = BalanceSheet
        fields = '__all__'

class CashFlowSerializer(serializers.ModelSerializer):
    class Meta:
        from stock_models.financial import CashFlow
        model = CashFlow
        fields = '__all__'

class ForecastSerializer(serializers.ModelSerializer):
    class Meta:
        from stock_models.financial import Forecast
        model = Forecast
        fields = '__all__'

class ExpressSerializer(serializers.ModelSerializer):
    class Meta:
        from stock_models.financial import Express
        model = Express
        fields = '__all__'

class DividendSerializer(serializers.ModelSerializer):
    class Meta:
        from stock_models.financial import Dividend
        model = Dividend
        fields = '__all__'

class FinaIndicatorSerializer(serializers.ModelSerializer):
    class Meta:
        from stock_models.financial import FinaIndicator
        model = FinaIndicator
        fields = '__all__'

class FinaAuditSerializer(serializers.ModelSerializer):
    class Meta:
        from stock_models.financial import FinaAudit
        model = FinaAudit
        fields = '__all__'

class FinaMainBZSerializer(serializers.ModelSerializer):
    class Meta:
        from stock_models.financial import FinaMainBZ
        model = FinaMainBZ
        fields = '__all__'

class DisclosureDateSerializer(serializers.ModelSerializer):
    class Meta:
        from stock_models.financial import DisclosureDate
        model = DisclosureDate
        fields = '__all__'


class FundFlowDailyTHSSerializer(serializers.ModelSerializer):
    class Meta:
        from stock_models.fund_flow import FundFlowDailyTHS
        model = FundFlowDailyTHS
        fields = '__all__'

class FundFlowDailyDCSerializer(serializers.ModelSerializer):
    class Meta:
        from stock_models.fund_flow import FundFlowDailyDC
        model = FundFlowDailyDC
        fields = '__all__'

class FundFlowCntTHSSerializer(serializers.ModelSerializer):
    class Meta:
        from stock_models.fund_flow import FundFlowCntTHS
        model = FundFlowCntTHS
        fields = '__all__'

class FundFlowCntDCSerializer(serializers.ModelSerializer):
    class Meta:
        from stock_models.fund_flow import FundFlowCntDC
        model = FundFlowCntDC
        fields = '__all__'

class FundFlowIndustryTHSSerializer(serializers.ModelSerializer):
    class Meta:
        from stock_models.fund_flow import FundFlowIndustryTHS
        model = FundFlowIndustryTHS
        fields = '__all__'

class FundFlowMarketDcSerializer(serializers.ModelSerializer):
    class Meta:
        from stock_models.fund_flow import FundFlowMarketDc
        model = FundFlowMarketDc
        fields = '__all__'

# stock_models\index.py
class IndexInfoSerializer(serializers.ModelSerializer):
    class Meta:
        from stock_models.index import IndexInfo
        model = IndexInfo
        fields = '__all__'

class IndexWeightSerializer(serializers.ModelSerializer):
    class Meta:
        from stock_models.index import IndexWeight
        model = IndexWeight
        fields = '__all__'

class IndexDailyBasicSerializer(serializers.ModelSerializer):
    class Meta:
        from stock_models.index import IndexDailyBasic
        model = IndexDailyBasic
        fields = '__all__'

# stock_models\industry.py
class SwIndustrySerializer(serializers.ModelSerializer):
    class Meta:
        from stock_models.industry import SwIndustry
        model = SwIndustry
        fields = '__all__'

class SwIndustryMemberSerializer(serializers.ModelSerializer):
    class Meta:
        from stock_models.industry import SwIndustryMember
        model = SwIndustryMember
        fields = '__all__'

class SwIndustryDailySerializer(serializers.ModelSerializer):
    class Meta:
        from stock_models.industry import SwIndustryDaily
        model = SwIndustryDaily
        fields = '__all__'

class CiIndexMemberSerializer(serializers.ModelSerializer):
    class Meta:
        from stock_models.industry import CiIndexMember
        model = CiIndexMember
        fields = '__all__'

class CiDailySerializer(serializers.ModelSerializer):
    class Meta:
        from stock_models.industry import CiDaily
        model = CiDaily
        fields = '__all__'

class KplConceptInfoSerializer(serializers.ModelSerializer):
    class Meta:
        from stock_models.industry import KplConceptInfo
        model = KplConceptInfo
        fields = '__all__'

class KplConceptConstituentSerializer(serializers.ModelSerializer):
    class Meta:
        from stock_models.industry import KplConceptConstituent
        model = KplConceptConstituent
        fields = '__all__'

class ThsIndexSerializer(serializers.ModelSerializer):
    class Meta:
        from stock_models.industry import ThsIndex
        model = ThsIndex
        fields = '__all__'

class ThsIndexMemberSerializer(serializers.ModelSerializer):
    class Meta:
        from stock_models.industry import ThsIndexMember
        model = ThsIndexMember
        fields = '__all__'

class ThsIndexDailySerializer(serializers.ModelSerializer):
    class Meta:
        from stock_models.industry import ThsIndexDaily
        model = ThsIndexDaily
        fields = '__all__'

class DcIndexSerializer(serializers.ModelSerializer):
    class Meta:
        from stock_models.industry import DcIndex
        model = DcIndex
        fields = '__all__'

class DcIndexDailySerializer(serializers.ModelSerializer):
    class Meta:
        from stock_models.industry import DcIndexDaily
        model = DcIndexDaily
        fields = '__all__'

class DcIndexMemberSerializer(serializers.ModelSerializer):
    class Meta:
        from stock_models.industry import DcIndexMember
        model = DcIndexMember
        fields = '__all__'

# stock_models\market.py
class MarketDailyInfoSerializer(serializers.ModelSerializer):
    class Meta:
        from stock_models.market import MarketDailyInfo
        model = MarketDailyInfo
        fields = '__all__'

class HmListSerializer(serializers.ModelSerializer):
    class Meta:
        from stock_models.market import HmList
        model = HmList
        fields = '__all__'

class HmDetailSerializer(serializers.ModelSerializer):
    class Meta:
        from stock_models.market import HmDetail
        model = HmDetail
        fields = '__all__'

# stock_models\stock_basic.py
class StockInfoSerializer(serializers.ModelSerializer):
    class Meta:
        from stock_models.stock_basic import StockInfo
        model = StockInfo
        fields = '__all__'

class StockCompanySerializer(serializers.ModelSerializer):
    class Meta:
        from stock_models.stock_basic import StockCompany
        model = StockCompany
        fields = '__all__'

class HSConstSerializer(serializers.ModelSerializer):
    class Meta:
        from stock_models.stock_basic import HSConst
        model = HSConst
        fields = '__all__'

# stock_models\time_trade.py
class StockDailyBasicSerializer(serializers.ModelSerializer):
    class Meta:
        from stock_models.time_trade import StockDailyBasic
        model = StockDailyBasic
        fields = '__all__'

class StockMinuteDataSerializer(serializers.ModelSerializer):
    class Meta:
        from stock_models.time_trade import StockMinuteData
        model = StockMinuteData
        fields = '__all__'

class StockWeeklyDataSerializer(serializers.ModelSerializer):
    class Meta:
        from stock_models.time_trade import StockWeeklyData
        model = StockWeeklyData
        fields = '__all__'

class StockMonthlyDataSerializer(serializers.ModelSerializer):
    class Meta:
        from stock_models.time_trade import StockMonthlyData
        model = StockMonthlyData
        fields = '__all__'

class StockCyqPerfSerializer(serializers.ModelSerializer):
    class Meta:
        from stock_models.time_trade import StockCyqPerf
        model = StockCyqPerf
        fields = '__all__'

class IndexDailySerializer(serializers.ModelSerializer):
    class Meta:
        from stock_models.time_trade import IndexDaily
        model = IndexDaily
        fields = '__all__'

class IndexWeeklySerializer(serializers.ModelSerializer):
    class Meta:
        from stock_models.time_trade import IndexWeekly
        model = IndexWeekly
        fields = '__all__'

class IndexMonthlySerializer(serializers.ModelSerializer):
    class Meta:
        from stock_models.time_trade import IndexMonthly
        model = IndexMonthly
        fields = '__all__'



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

class TransactionSerializer(serializers.ModelSerializer):
    """
    交易流水序列化器
    """
    # 让 tracker 字段在创建时只接受主键ID，在响应时显示详细信息
    tracker = serializers.PrimaryKeyRelatedField(queryset=PositionTracker.objects.all())
    class Meta:
        model = Transaction
        fields = [
            'id',
            'tracker',
            'transaction_type',
            'quantity',
            'price',
            'transaction_date',
            'created_at'
        ]
        read_only_fields = ('created_at',)
    def validate_transaction_date(self, value):
        # 确保交易日期不能在未来
        if value > timezone.now():
            raise serializers.ValidationError("交易日期不能在未来。")
        return value
    def validate_quantity(self, value):
        if value <= 0:
            raise serializers.ValidationError("交易数量必须是正数。")
        return value
        
