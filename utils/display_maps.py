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
                         .get('score_type_map', {})
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

# 核心策略名称映射
STRATEGY_NAME_MAP = {
    'multi_timeframe_trend_strategy': '三级引擎协同策略',
    'trend_follow_strategy': '日线趋势跟踪策略',
    'weekly_trend_follow_strategy': '周线战略趋势策略',
    'multi_timeframe_collaboration': '日线战术协同策略',
    'RESONANCE_FRACTAL_ROCKET': '5分钟分形火箭',
    'INTRADAY_TAKE_PROFIT': '盘中止盈预警',
}
