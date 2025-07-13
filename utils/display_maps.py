# 文件: utils/display_maps.py
# 版本: V3.1 - 对齐周线策略大写命名规范
# 描述: 这是一个集中的配置文件，用于将策略和剧本的内部英文ID映射到前端显示的中文名称。

# 核心策略名称映射
STRATEGY_NAME_MAP = {
    'multi_timeframe_trend_strategy': '三级引擎协同策略',
    'trend_follow_strategy': '日线趋势跟踪策略',
    'weekly_trend_follow_strategy': '周线战略趋势策略',
    'multi_timeframe_collaboration': '日线战术协同策略',
    'RESONANCE_FRACTAL_ROCKET': '5分钟分形火箭',
    'INTRADAY_TAKE_PROFIT': '盘中止盈预警',
}

# 剧本(Playbook)与得分项(Scoring Item)名称映射
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

    # ==================================================================
    # ===            周线战略指令 (Strategic Directives)             ===
    # ==================================================================
    # 描述: 这些是由总指挥官(MultiTimeframeTrendStrategy)将周线原始信号翻译成的、
    #      日线策略能直接理解的作战指令。它们作为基础分项(BASE_)或前提条件出现。

    # --- A. 许可型指令 (用于解锁左侧交易) ---
    'CONTEXT_STRATEGIC_BOTTOMING_W': '【周策-许可】战略性筑底', # 来自 playbook_COPPOCK_STABILIZING_W

    # --- B. 增强型指令 (作为右侧交易的强力加分项) ---
    'EVENT_STRATEGIC_ACCELERATING_W': '【周策-增强】战略性加速', # 来自 playbook_COPPOCK_ACCELERATING_W
    
    # --- C. 基础背景/突破指令 (作为基础分) ---
    'BASE_SIGNAL_BREAKOUT_TRIGGER': '【周策-王牌】关键位突破', # 来自 ace_signal_breakout_trigger_playbook
    'MA20_RISING_STATE_W': '【周策】MA21上升状态', # 来自 ma20_rising_state_playbook
    'MA20_TURN_UP_EVENT_W': '【周策】MA21拐头向上', # 来自 ma20_turn_up_event_playbook
    'EARLY_UPTREND_W': '【周策】早期上升趋势', # 来自 early_uptrend_playbook
    'CLASSIC_BREAKOUT_W': '【周策】经典高点突破', # 来自 classic_breakout_playbook
    'MA_UPTREND_W': '【周策】均线多头排列', # 来自 ma_uptrend_playbook
    'BOX_CONSOLIDATION_BREAKOUT_W': '【周策】专业箱体突破', # 来自 box_consolidation_breakout_playbook
    'BASE_OVERSOLD_REBOUND_BIAS': '【周策】BIAS超跌反弹', # 来自 oversold_rebound_bias_playbook
    'TRIX_GOLDEN_CROSS_W': '【周策】TRIX金叉', # 来自 trix_golden_cross_playbook
    'BASE_STRATEGIC_ACCEL': '【周策-王牌】估波曲线加速', # 来自 EVENT_STRATEGIC_ACCELERATING_W 的加分项
    'ACE_SIGNAL_BREAKOUT_TRIGGER': 'ACE信号突破',

    # ==================================================================
    # ===              协同/冲突规则 (Bonus & Penalty)               ===
    # ==================================================================
    # 描述: 这些不是独立的交易剧本，而是用于对核心剧本进行加分或减分的辅助规则。
    #      在V2.4版后，它们已被正确过滤，不应出现在 triggered_playbooks 列表中。
    #      此处保留映射，以备调试或未来扩展之用。
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

    # ==================================================================
    # ===                  止盈信号 (Take-Profit)                    ===
    # ==================================================================
    'EXIT_LEVEL_1': '【一级预警】趋势减速',
    'EXIT_LEVEL_2': '【二级警报】短期转弱',
    'EXIT_LEVEL_3': '【三级警报】跌破日线支撑',
    'EXIT_UPTHRUST_REJECTION': '冲高回落',
    
    # ==================================================================
    # ===                  兼容性/旧版ID (Legacy)                    ===
    # ==================================================================
    'playbook_coppock_stabilizing_W': '【周线-原始】估波曲线企稳',
    'COPPOCK_ACCELERATING_W': '【周线-原始】估波曲线加速',
}

# 合并所有映射，方便过滤器调用
# 策略名称的优先级高于剧本名称，以防重名
DISPLAY_MAP = {**PLAYBOOK_NAME_MAP, **STRATEGY_NAME_MAP}
