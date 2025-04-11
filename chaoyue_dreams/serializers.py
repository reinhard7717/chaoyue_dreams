from rest_framework import serializers
import umsgpack
from utils import cache_constants as cc
from utils.cache_manager import CacheManager
from utils.cash_key import StockCashKey



class IndexInfoSerializer(serializers.ModelSerializer):
    class Meta:
        from stock_models.index import IndexInfo
        model = IndexInfo
        fields = '__all__'

class IndexRealTimeDataSerializer(serializers.ModelSerializer):
    class Meta:
        from stock_models.index import IndexRealTimeData
        model = IndexRealTimeData
        fields = '__all__'

    def cache_model_instance_umsgpack_drf(model_instance):
        """
        将模型实例序列化为umsgpack格式
        """
        cache_manager = CacheManager()
        cache_key_stock = StockCashKey()
        serializer = model_instance.serializer(model_instance)
        serializer_data = umsgpack.packb(serializer.data, use_bin_type=True)
        cache_key = cache_manager.generate_key(
            cache_type=cc.TYPE_STATIC,
            entity_type=cc.ENTITY_INDEX,
            entity_id=cc.ID_ALL
        )
        cache_manager.set_cache(cache_key, serializer_data)

class MarketOverviewSerializer(serializers.ModelSerializer):
    class Meta:
        from stock_models.index import MarketOverview
        model = MarketOverview
        fields = '__all__'

class IndexTimeSeriesDataSerializer(serializers.ModelSerializer):
    class Meta:
        from stock_models.index import IndexTimeSeriesData
        model = IndexTimeSeriesData
        fields = '__all__'


class StockInfoSerializer(serializers.ModelSerializer):
    class Meta:
        from stock_models.stock_basic import StockInfo
        model = StockInfo
        fields = '__all__'

class NewStockCalendarSerializer(serializers.ModelSerializer):
    class Meta:
        from stock_models.stock_basic import NewStockCalendar
        model = NewStockCalendar
        fields = '__all__'

class STStockListSerializer(serializers.ModelSerializer):
    class Meta:
        from stock_models.stock_basic import STStockList
        model = STStockList
        fields = '__all__'

class CompanyInfoSerializer(serializers.ModelSerializer):
    class Meta:
        from stock_models.stock_basic import CompanyInfo
        model = CompanyInfo
        fields = '__all__'

class StockBelongsIndexSerializer(serializers.ModelSerializer):
    class Meta:
        from stock_models.stock_basic import StockBelongsIndex
        model = StockBelongsIndex
        fields = '__all__'

class QuarterlyProfitSerializer(serializers.ModelSerializer):
    class Meta:
        from stock_models.stock_basic import QuarterlyProfit
        model = QuarterlyProfit
        fields = '__all__'

class MarketCategorySerializer(serializers.ModelSerializer):
    class Meta:
        from stock_models.stock_basic import MarketCategory
        model = MarketCategory
        fields = '__all__'

class StockTimeTradeSerializer(serializers.ModelSerializer):
    class Meta:
        from stock_models.stock_realtime import StockTimeTrade
        model = StockTimeTrade
        fields = '__all__'

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


class IndustryCapitalFlowSerializer(serializers.ModelSerializer):
    class Meta:
        from stock_models.datacenter.capital_flow import IndustryCapitalFlow
        model = IndustryCapitalFlow
        fields = '__all__'

class ConceptCapitalFlowSerializer(serializers.ModelSerializer):
    class Meta:
        from stock_models.datacenter.capital_flow import ConceptCapitalFlow
        model = ConceptCapitalFlow
        fields = '__all__'

class NetInflowRankingSerializer(serializers.ModelSerializer):
    class Meta:
        from stock_models.datacenter.capital_flow import NetInflowRanking
        model = NetInflowRanking
        fields = '__all__'

class NetInflowRateRankingSerializer(serializers.ModelSerializer):
    class Meta:
        from stock_models.datacenter.capital_flow import NetInflowRateRanking
        model = NetInflowRateRanking
        fields = '__all__'

class MainForceNetInflowRankingSerializer(serializers.ModelSerializer):
    class Meta:
        from stock_models.datacenter.capital_flow import MainForceNetInflowRanking
        model = MainForceNetInflowRanking
        fields = '__all__'

class MainForceNetInflowRateRankingSerializer(serializers.ModelSerializer):
    class Meta:
        from stock_models.datacenter.capital_flow import MainForceNetInflowRateRanking
        model = MainForceNetInflowRateRanking
        fields = '__all__'

class RetailNetInflowRankingSerializer(serializers.ModelSerializer):
    class Meta:
        from stock_models.datacenter.capital_flow import RetailNetInflowRanking
        model = RetailNetInflowRanking
        fields = '__all__'

class RetailNetInflowRateRankingSerializer(serializers.ModelSerializer):
    class Meta:
        from stock_models.datacenter.capital_flow import RetailNetInflowRateRanking
        model = RetailNetInflowRateRanking
        fields = '__all__'

class IndustryCapitalFlowRouteSerializer(serializers.ModelSerializer):
    class Meta:
        from stock_models.datacenter.capital_flow import IndustryCapitalFlowRoute
        model = IndustryCapitalFlowRoute
        fields = '__all__'

class ConceptCapitalFlowRouteSerializer(serializers.ModelSerializer):
    class Meta:
        from stock_models.datacenter.capital_flow import ConceptCapitalFlowRoute
        model = ConceptCapitalFlowRoute
        fields = '__all__'

class StockPeriodStatisticsOverviewSerializer(serializers.ModelSerializer):
    class Meta:
        from stock_models.datacenter.capital_flow import StockPeriodStatisticsOverview
        model = StockPeriodStatisticsOverview
        fields = '__all__'

class StockPeriodStatisticsSerializer(serializers.ModelSerializer):
    class Meta:
        from stock_models.datacenter.capital_flow import StockPeriodStatistics
        model = StockPeriodStatistics
        fields = '__all__'

class MainForceContinuousFlowSerializer(serializers.ModelSerializer):
    class Meta:
        from stock_models.datacenter.capital_flow import MainForceContinuousFlow
        model = MainForceContinuousFlow
        fields = '__all__'

class NewCapitalFlowOverviewSerializer(serializers.ModelSerializer):
    class Meta:
        from stock_models.datacenter.capital_flow import NewCapitalFlowOverview
        model = NewCapitalFlowOverview
        fields = '__all__'

# ============  financial  ============
class WeeklyRankChangeSerializer(serializers.ModelSerializer):
    class Meta:
        from stock_models.datacenter.financial import WeeklyRankChange
        model = WeeklyRankChange
        fields = '__all__'

class MonthlyRankChangeSerializer(serializers.ModelSerializer):
    class Meta:
        from stock_models.datacenter.financial import MonthlyRankChange
        model = MonthlyRankChange
        fields = '__all__'

class WeeklyStrongStockSerializer(serializers.ModelSerializer):
    class Meta:
        from stock_models.datacenter.financial import WeeklyStrongStock
        model = WeeklyStrongStock
        fields = '__all__'
        
class MonthlyStrongStockSerializer(serializers.ModelSerializer):
    class Meta:
        from stock_models.datacenter.financial import MonthlyStrongStock
        model = MonthlyStrongStock
        fields = '__all__'
        
class CircMarketValueRankSerializer(serializers.ModelSerializer):
    class Meta:
        from stock_models.datacenter.financial import CircMarketValueRank
        model = CircMarketValueRank
        fields = '__all__'
        
class PERatioRankSerializer(serializers.ModelSerializer):
    class Meta:
        from stock_models.datacenter.financial import PERatioRank
        model = PERatioRank
        fields = '__all__'
        
class PBRatioRankSerializer(serializers.ModelSerializer):
    class Meta:
        from stock_models.datacenter.financial import PBRatioRank
        model = PBRatioRank
        fields = '__all__'
        
class ROERankSerializer(serializers.ModelSerializer):
    class Meta:
        from stock_models.datacenter.financial import ROERank
        model = ROERank
        fields = '__all__'
        

# ============  institution  ============
class InstitutionHoldingSummarySerializer(serializers.ModelSerializer):
    class Meta:
        from stock_models.datacenter.institution import InstitutionHoldingSummary
        model = InstitutionHoldingSummary
        fields = '__all__'
        
class FundHeavyPositionSerializer(serializers.ModelSerializer):
    class Meta:
        from stock_models.datacenter.institution import FundHeavyPosition
        model = FundHeavyPosition
        fields = '__all__'
        
class SocialSecurityHeavyPositionSerializer(serializers.ModelSerializer):
    class Meta:
        from stock_models.datacenter.institution import SocialSecurityHeavyPosition
        model = SocialSecurityHeavyPosition
        fields = '__all__'
        
class QFIIHeavyPositionSerializer(serializers.ModelSerializer):
    class Meta:
        from stock_models.datacenter.institution import QFIIHeavyPosition
        model = QFIIHeavyPosition
        fields = '__all__'
        
# ============  lhb  ============
class LhbDetailSerializer(serializers.ModelSerializer):
    class Meta:
        from stock_models.datacenter.lhb import LhbDetail
        model = LhbDetail
        fields = '__all__'
        
class LhbDailySerializer(serializers.ModelSerializer):
    class Meta:
        from stock_models.datacenter.lhb import LhbDaily
        model = LhbDaily
        fields = '__all__'
        
class StockOnListSerializer(serializers.ModelSerializer):
    class Meta:
        from stock_models.datacenter.lhb import StockOnList
        model = StockOnList
        fields = '__all__'
        
class BrokerOnListSerializer(serializers.ModelSerializer):
    class Meta:
        from stock_models.datacenter.lhb import BrokerOnList
        model = BrokerOnList
        fields = '__all__'
        
class InstitutionTradeTrackSerializer(serializers.ModelSerializer):
    class Meta:
        from stock_models.datacenter.lhb import InstitutionTradeTrack
        model = InstitutionTradeTrack
        fields = '__all__'
        
class InstitutionTradeDetailSerializer(serializers.ModelSerializer):
    class Meta:
        from stock_models.datacenter.lhb import InstitutionTradeDetail
        model = InstitutionTradeDetail
        fields = '__all__'
        
# ============  market_data  ============
class VolumeIncreaseSerializer(serializers.ModelSerializer):
    class Meta:
        from stock_models.datacenter.market_data import VolumeIncrease
        model = VolumeIncrease
        fields = '__all__'
        
class VolumeDecreaseSerializer(serializers.ModelSerializer):
    class Meta:
        from stock_models.datacenter.market_data import VolumeDecrease
        model = VolumeDecrease
        fields = '__all__'
        
class ContinuousVolumeIncreaseSerializer(serializers.ModelSerializer):
    class Meta:
        from stock_models.datacenter.market_data import ContinuousVolumeIncrease
        model = ContinuousVolumeIncrease
        fields = '__all__'
        
class ContinuousVolumeDecreaseSerializer(serializers.ModelSerializer):
    class Meta:
        from stock_models.datacenter.market_data import ContinuousVolumeDecrease
        model = ContinuousVolumeDecrease
        fields = '__all__'
        
class ContinuousRiseSerializer(serializers.ModelSerializer):
    class Meta:
        from stock_models.datacenter.market_data import ContinuousRise
        model = ContinuousRise
        fields = '__all__'
        
class ContinuousFallSerializer(serializers.ModelSerializer):
    class Meta:
        from stock_models.datacenter.market_data import ContinuousFall
        model = ContinuousFall
        fields = '__all__'
        
        
# ============  statistics  ============
class StageHighLowSerializer(serializers.ModelSerializer):
    class Meta:
        from stock_models.datacenter.statistics import StageHighLow
        model = StageHighLow
        fields = '__all__'

class NewHighStockSerializer(serializers.ModelSerializer):
    class Meta:
        from stock_models.datacenter.statistics import NewHighStock
        model = NewHighStock
        fields = '__all__'
        
class NewLowStockSerializer(serializers.ModelSerializer):
    class Meta:
        from stock_models.datacenter.statistics import NewLowStock
        model = NewLowStock
        fields = '__all__'
        
        
# ============  ADL  ============
class IndexAdlSerializer(serializers.ModelSerializer):
    class Meta:
        from stock_models.indicator.adl import IndexAdl
        model = IndexAdl
        fields = '__all__'
        
class StockAdlSerializer(serializers.ModelSerializer):
    class Meta:
        from stock_models.indicator.adl import StockAdl
        model = StockAdl
        fields = '__all__'
        

# ============  ATR  ============
class IndexAtrSerializer(serializers.ModelSerializer):
    class Meta:
        from stock_models.indicator.atr import IndexAtr
        model = IndexAtr
        fields = '__all__'

class StockAtrSerializer(serializers.ModelSerializer):
    class Meta:
        from stock_models.indicator.atr import StockAtr
        model = StockAtr
        fields = '__all__'

# ============  BOLL  ============
class IndexBollDataSerializer(serializers.ModelSerializer):
    class Meta:
        from stock_models.indicator.boll import IndexBollData
        model = IndexBollData
        fields = '__all__'
        
class StockBollIndicatorSerializer(serializers.ModelSerializer):
    class Meta:
        from stock_models.indicator.boll import StockBollIndicator
        model = StockBollIndicator
        fields = '__all__'
        
# ============  CCI  ============
class IndexCciFIBSerializer(serializers.ModelSerializer):
    class Meta:
        from stock_models.indicator.cci import IndexCciFIB
        model = IndexCciFIB
        fields = '__all__'
        
class StockCciFIBSerializer(serializers.ModelSerializer):
    class Meta:
        from stock_models.indicator.cci import StockCciFIB
        model = StockCciFIB
        fields = '__all__'
        
# ============  CMF  ============
class IndexCmfFIBSerializer(serializers.ModelSerializer):
    class Meta:
        from stock_models.indicator.cmf import IndexCmfFIB
        model = IndexCmfFIB
        fields = '__all__'
        
class StockCmfFIBSerializer(serializers.ModelSerializer):
    class Meta:
        from stock_models.indicator.cmf import StockCmfFIB
        model = StockCmfFIB
        fields = '__all__'

# ============  DMI  ============
class IndexDmiFIBSerializer(serializers.ModelSerializer):
    class Meta:
        from stock_models.indicator.dmi import IndexDmiFIB
        model = IndexDmiFIB
        fields = '__all__'

class StockDmiFIBSerializer(serializers.ModelSerializer):
    class Meta:
        from stock_models.indicator.dmi import StockDmiFIB
        model = StockDmiFIB
        fields = '__all__'


# ============  Ichimoku  ============
class IndexIchimokuSerializer(serializers.ModelSerializer):
    class Meta:
        from stock_models.indicator.ichimoku import IndexIchimoku
        model = IndexIchimoku
        fields = '__all__'

class StockIchimokuSerializer(serializers.ModelSerializer):
    class Meta:
        from stock_models.indicator.ichimoku import StockIchimoku
        model = StockIchimoku
        fields = '__all__'


# ============  KC  ============
class IndexKcFIBSerializer(serializers.ModelSerializer):
    class Meta:
        from stock_models.indicator.kc import IndexKcFIB
        model = IndexKcFIB
        fields = '__all__'

class StockKcFIBSerializer(serializers.ModelSerializer):
    class Meta:
        from stock_models.indicator.kc import StockKcFIB
        model = StockKcFIB
        fields = '__all__'


# ============  KDJ  ============
class IndexKDJDataSerializer(serializers.ModelSerializer):
    class Meta:
        from stock_models.indicator.kdj import IndexKDJData
        model = IndexKDJData
        fields = '__all__'
        
class StockKDJIndicatorSerializer(serializers.ModelSerializer):
    class Meta:
        from stock_models.indicator.kdj import StockKDJIndicator
        model = StockKDJIndicator
        fields = '__all__'
        
class IndexKDJFIBSerializer(serializers.ModelSerializer):
    class Meta:
        from stock_models.indicator.kdj import IndexKDJFIB
        model = IndexKDJFIB
        fields = '__all__'
        
class StockKDJFIBSerializer(serializers.ModelSerializer):
    class Meta:
        from stock_models.indicator.kdj import StockKDJFIB
        model = StockKDJFIB
        fields = '__all__'
        
        
# ============  MA  ============
class IndexMADataSerializer(serializers.ModelSerializer):
    class Meta:
        from stock_models.indicator.ma import IndexMAData
        model = IndexMAData
        fields = '__all__'
        
class StockMAIndicatorSerializer(serializers.ModelSerializer):
    class Meta:
        from stock_models.indicator.ma import StockMAIndicator
        model = StockMAIndicator
        fields = '__all__'
        
class IndexEmaFIBSerializer(serializers.ModelSerializer):
    class Meta:
        from stock_models.indicator.ma import IndexEmaFIB
        model = IndexEmaFIB
        fields = '__all__'
        
class StockEmaFIBSerializer(serializers.ModelSerializer):
    class Meta:
        from stock_models.indicator.ma import StockEmaFIB
        model = StockEmaFIB
        fields = '__all__'
        
class IndexAmountMaFIBSerializer(serializers.ModelSerializer):
    class Meta:
        from stock_models.indicator.ma import IndexAmountMaFIB
        model = IndexAmountMaFIB
        fields = '__all__'
        
class StockAmountMaFIBSerializer(serializers.ModelSerializer):
    class Meta:
        from stock_models.indicator.ma import StockAmountMaFIB
        model = StockAmountMaFIB
        fields = '__all__'


# ============  MACD  ============
class IndexMACDDataSerializer(serializers.ModelSerializer):
    class Meta:
        from stock_models.indicator.macd import IndexMACDData
        model = IndexMACDData
        fields = '__all__'
        
class StockMACDIndicatorSerializer(serializers.ModelSerializer):
    class Meta:
        from stock_models.indicator.macd import StockMACDIndicator
        model = StockMACDIndicator
        fields = '__all__'
        
class IndexMACDFIBSerializer(serializers.ModelSerializer):
    class Meta:
        from stock_models.indicator.macd import IndexMACDFIB
        model = IndexMACDFIB
        fields = '__all__'
        
class StockMACDFIBSerializer(serializers.ModelSerializer):
    class Meta:
        from stock_models.indicator.macd import StockMACDFIB
        model = StockMACDFIB
        fields = '__all__'
        
        
        
