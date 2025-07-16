# 文件: utils/display_maps.py
# 版本: V165.0 - 动态元数据加载版
# 描述: 这是一个集中的配置文件，用于将策略和剧本的内部英文ID映射到前端显示的中文名称。
#       V165.0 核心升级: 新增了从策略JSON文件中动态加载评分项中文名的能力。

import logging
from utils.config_loader import load_strategy_config

logger = logging.getLogger(__name__)

# ▼▼▼【代码新增 V165.0】: 动态元数据加载器 ▼▼▼
def _load_dynamic_scoring_map() -> dict:
    """
    【V165.0 新增】动态加载器。
    - 职责: 读取策略配置文件，并从 "four_layer_scoring_params.metadata" 中
            提取所有评分项的中文名称，生成一个动态的映射字典。
    - 收益: 实现了配置与显示的解耦。未来在JSON中新增或修改评分项，前端将自动
            显示正确的中文名，无需修改任何Python代码。
    """
    dynamic_map = {}
    try:
        # 加载策略配置文件，这是我军唯一的“真理来源”
        config = load_strategy_config('config/trend_follow_strategy.json')
        
        # 精准定位到元数据模块
        metadata = config.get('strategy_params', {})\
                         .get('trend_follow', {})\
                         .get('four_layer_scoring_params', {})\
                         .get('metadata', {})
        
        if not metadata:
            logger.warning("在 trend_follow_strategy.json 中未找到 'metadata' 模块，无法加载动态评分项名称。")
            return {}
            
        # 遍历元数据，构建 "英文ID: 中文名" 的映射
        for key, value in metadata.items():
            if isinstance(value, dict) and 'cn_name' in value:
                dynamic_map[key] = value['cn_name']
                
        logger.info(f"成功从JSON配置中动态加载了 {len(dynamic_map)} 个评分项的中文名称。")
        return dynamic_map

    except Exception as e:
        logger.error(f"加载动态评分项中文名时发生错误: {e}", exc_info=True)
        # 在发生错误时返回一个空字典，确保程序不会崩溃
        return {}

# 核心策略名称映射 (保持不变)
STRATEGY_NAME_MAP = {
    'multi_timeframe_trend_strategy': '三级引擎协同策略',
    'trend_follow_strategy': '日线趋势跟踪策略',
    'weekly_trend_follow_strategy': '周线战略趋势策略',
    'multi_timeframe_collaboration': '日线战术协同策略',
    'RESONANCE_FRACTAL_ROCKET': '5分钟分形火箭',
    'INTRADAY_TAKE_PROFIT': '盘中止盈预警',
}

# 剧本(Playbook)与旧版得分项名称映射 (保持不变, 用于兼容)
PLAYBOOK_NAME_MAP = {
    # ==================================================================
    # ===               日线战术剧本 (Tactical Playbooks)             ===
    # ==================================================================
    'ABYSS_GAZE_S': '【S级】深渊凝视',
    'CAPITULATION_PIT_REVERSAL': '投降坑反转',
    'CAPITAL_DIVERGENCE_REVERSAL': '【A-级】资本逆行者',
    'BEAR_TRAP_RALLY': '【C+级】熊市反弹',
    'WASHOUT_REVERSAL_A': '【A级】巨阴洗盘反转',
    'BOTTOM_STABILIZATION_B': '【B级】底部企稳',
    'TREND_EMERGENCE_B_PLUS': '【B+级】右侧萌芽',
    'DEEP_ACCUMULATION_BREAKOUT': '潜龙出海',
    'ENERGY_COMPRESSION_BREAKOUT': '能量压缩突破',
    'PLATFORM_SUPPORT_PULLBACK': '平台支撑回踩',
    'HEALTHY_BOX_BREAKOUT': '【A-级】健康箱体突破',
    'HEALTHY_MARKUP_A': '【A级】健康主升浪',
    'N_SHAPE_CONTINUATION_A': '【A级】N字板接力',
    'GAP_SUPPORT_PULLBACK_B_PLUS': '【B+级】缺口支撑回踩',
    'EARTH_HEAVEN_BOARD': '【S+】地天板',    
    'CHIP_PURITY_MULTIPLIER': '筹码至纯',
    'VOLATILITY_SILENCE_MULTIPLIER': '波动至静',
    'BONUS_HIGH_CONTROL_MARKUP': '高控盘拉升',
    'INTRADAY_ENTRY_CONFIRMATION': '分钟级共振引擎',
    'ENTRY_INTRADAY_CONFIRMATION':'分钟级买入确认',

    'CHIP_DYNAMICS_MULTIPLIER': '【筹码】火力放大',
    # ... 此处省略其他旧的映射，以保持简洁 ...
    'EXIT_LEVEL_1': '【一级预警】趋势减速',
    'EXIT_LEVEL_2': '【二级警报】短期转弱',
    'EXIT_LEVEL_3': '【三级警报】跌破日线支撑',
    'EXIT_INTRADAY_UPTHRUST_REJECTION': '冲高回落',
    'EXIT_INTRADAY_PULLBACK_FAILED': '急跌反弹失败',

    'TRIGGER_DOMINANT_REVERSAL': '显性反转阳线',
    'TRIGGER_BREAKOUT_CANDLE': '突破阳线(企稳型)',
    'TRIGGER_PLATFORM_PULLBACK_REBOUND': '筹码平台回踩反弹',
    'FINAL_MULTIPLIER': '火力放大器',
    'TRIGGER_ENERGY_RELEASE': '能量释放(突破型)',
    'VOL_BREAKOUT_FROM_SQUEEZE': '突破盘整区',
    '': '',
}

# ▼▼▼【代码修改 V165.0】: 合并静态与动态词典，生成最终的显示映射 ▼▼▼
# 1. 调用新的加载器，获取所有在JSON中定义的动态名称
dynamic_scoring_map = _load_dynamic_scoring_map()

# 2. 合并所有映射，生成最终的、最全的显示词典
#    合并顺序确保了特异性最高的名称（如策略名）能覆盖通用名称
DISPLAY_MAP = {
    **PLAYBOOK_NAME_MAP,      # 首先加载旧的、兼容性的剧本名
    **dynamic_scoring_map,    # 然后加载所有来自JSON的动态评分项名称 (如果重名会覆盖旧的)
    **STRATEGY_NAME_MAP       # 最后加载策略名，其优先级最高
}
# ▲▲▲【代码修改 V165.0】▲▲▲
