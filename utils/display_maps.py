# 文件: utils/display_maps.py
# 版本: V2.0 - 全面重构，对齐V22+版策略
# 描述: 这是一个集中的配置文件，用于将策略和剧本的内部英文ID映射到前端显示的中文名称。

# 核心策略名称映射
STRATEGY_NAME_MAP = {
    'multi_timeframe_trend_strategy': '三级引擎协同策略',
    'trend_follow_strategy': '日线趋势跟踪策略',
    'weekly_trend_follow_strategy': '周线战略趋势策略',
    'multi_timeframe_collaboration': '日线战术协同策略',
    'RESONANCE_FRACTAL_ROCKET': '5分钟分形火箭',
    # ...可以根据需要添加更多策略名称...
}

# 剧本(Playbook)与得分项(Scoring Item)名称映射
PLAYBOOK_NAME_MAP = {
    # ▼▼▼【代码修改】: 新增V22+版日线战术剧本，并按类型重新组织 ▼▼▼

    # === 日线战术剧本 (主要机会信号) ===
    # --- 主力行为/高确定性 ---
    'CHIP_CONCENTRATION_BREAKTHROUGH': '【日线】筹码集中突破',
    'COST_AREA_REINFORCEMENT': '【日线】成本区增强',
    'CHIP_PRESSURE_RELEASE': '【日线】筹码压力释放',
    'CHIP_COST_BREAKTHROUGH': '【日线】筹码成本区突破',
    'ENERGY_COMPRESSION_BREAKOUT': '【日线】潜龙在渊',
    'OLD_DUCK_HEAD': '【日线】老鸭头',
    'EARTH_HEAVEN_BOARD': '【日线】地天板',
    # --- 左侧反转/先手博弈 ---
    'WINNER_RATE_REVERSAL': '【日线】投降坑反转',
    'KLINE_MORNING_STAR': '【日线】早晨之星',
    'BOTTOM_DIVERGENCE': '【日线】复合底背离',
    'CAPITAL_FLOW_DIVERGENCE': '【日线】资金暗流',
    # --- 趋势持续/右侧跟踪 ---
    'MA_ACCELERATION': '【日线】均线加速上涨',
    'PULLBACK_FIBONACCI': '【日线】斐波那契回撤',
    'CONSOLIDATION_BREAKOUT': '【日线】盘整区突破',
    'BULLISH_FLAG': '【日线】上升旗形',
    'PULLBACK_NORMAL': '【日线】常规回踩',
    'PULLBACK_STEADY_CLIMB': '【日线】稳步回踩',
    'FIRST_BREAKOUT': '【日线】底部首板',
    'BBAND_SQUEEZE_BREAKOUT': '【日线】布林收口突破',
    'V_SHAPE_REVERSAL': '【日线】V型反转',
    'CHIP_HURDLE_CLEAR': '【日线】筹码关口扫清',
    'KLINE_THREE_SOLDIERS': '【日线】红三兵',
    # --- 基础信号/确认指标 ---
    'MACD_ZERO_CROSS': '【日线】MACD零轴金叉',
    'MACD_LOW_CROSS': '【日线】MACD低位金叉',
    'DMI_CROSS': '【日线】DMI金叉',
    'PULLBACK_SETUP': '【日线】回踩准备',

    # === 协同/冲突规则 (加减分项) ===
    'BONUS_VWAP_SUPPORT': '【加分】VWAP支撑',
    'BONUS_CMF_CONFIRM': '【加分】资金流入确认',
    'BONUS_FUND_FLOW_CONFIRM': '【加分】主力资金确认',
    'BONUS_TURNOVER_BOARD': '【加分】换手板次日',
    'BONUS_PULLBACK_KLINE_DECENT': '【加分】回踩K线确认(弱)',
    'BONUS_PULLBACK_KLINE_PERFECT': '【加分】回踩K线确认(强)',
    'BONUS_BB_MOMENTUM_COMBO': '【加分】布林动量组合',
    'BONUS_STEADY_CLIMB_MACD_ZERO': '【加分】稳步回踩+MACD零轴',
    'BONUS_MULTIPLIER': '【加分】动量放大',
    'PENALTY_CONFLICT': '【减分】信号冲突',
    'INDUSTRY_MULTIPLIER_ADJ': '【行业】强度乘数调整',
    'INDUSTRY_TOP_TIER_BONUS': '【行业】龙头板块加分',

    # ▲▲▲【代码修改结束】▲▲▲

    # === 周线战略得分项 (由周线剧本转换而来) ===
    'BASE_MA20_TURN_UP': '【周策】MA20拐头向上',
    'BASE_EARLY_UPTREND': '【周策】早期上升趋势',
    'BASE_CLASSIC_BREAKOUT': '【周策】经典高点突破',
    'BASE_MA_UPTREND': '【周策】均线多头排列',
    'BASE_BOX_BREAKOUT': '【周策】专业箱体突破',
    'BASE_SIGNAL_BREAKOUT_TRIGGER': '【周策-王牌】关键位突破',

    # === 周线战略剧本 (原始ID, 兼容保留) ===
    'playbook_ma20_turn_up_event_W': '【周线】MA20拐头向上',
    'playbook_early_uptrend_W': '【周线】早期上升趋势',
    'playbook_classic_breakout_W': '【周线】经典高点突破',
    'playbook_ma_uptrend_W': '【周线】均线多头排列',
    'playbook_box_consolidation_breakout_W': '【周线】专业箱体突破',
    'playbook_ace_signal_breakout_trigger_W': '【周线】王牌突破信号',

    # === 分钟级执行剧本 (Execution Playbooks) ===
    'RESONANCE_FRACTAL_ROCKET': '【分钟】分形火箭',

    # === 复合信号与旧版ID (兼容性保留) ===
    'BREAKOUT_TRIGGER_SCORE': '【王牌】周线突破观察',
    'EXPERT_PLAYBOOK_VOL_PRICE_DIVERGENCE': '【专家】量价背离抄底',
    'EXPERT_PLAYBOOK_LOW_VOL_BREAKOUT': '【专家】缩量蓄势突破',

    # === 止盈信号  ===
    'EXIT_CODE_101': '15分钟MACD死叉',
    'EXIT_CODE_102': '15分钟RSI顶背离',
    'EXIT_CODE_103': '15分钟KDJ在高位死叉',
    'INTRADAY_TAKE_PROFIT': '盘中止盈预警'
}

# 合并所有映射，方便过滤器调用
# 策略名称的优先级高于剧本名称，以防重名
DISPLAY_MAP = {**PLAYBOOK_NAME_MAP, **STRATEGY_NAME_MAP}
