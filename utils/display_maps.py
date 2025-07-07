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
    
    # --- A. 左侧交易剧本 (Left-Side Plays) ---
    # 描述: 捕捉趋势转折点，通常需要周线企稳信号(CONTEXT_STRATEGIC_BOTTOMING_W)作为前置许可。
    'BOTTOM_DIVERGENCE': '【日线-左侧】复合底背离',
    'CAPITAL_FLOW_DIVERGENCE': '【日线-左侧】资金暗流',
    'WINNER_RATE_REVERSAL': '【日线-左侧】投降坑反转',
    'V_SHAPE_REVERSAL': '【日线-左侧】V型反转',
    'WASHOUT_REVERSAL': '【日线-左侧】巨阴洗盘反转',
    'KLINE_MORNING_STAR': '【日线-左侧】早晨之星',
    'BIAS_REVERSAL': '【日线-左侧】BIAS超跌反弹',

    # --- B. 右侧交易剧本 (Right-Side Plays) ---
    # 描述: 追随已形成的趋势，通常需要周线趋势健康(context_mid_term_bullish)作为基础环境。
    # B.1 趋势启动/突破类
    'CHIP_CONCENTRATION_BREAKTHROUGH': '【日线】筹码集中突破',
    'COST_AREA_REINFORCEMENT': '【日线-王牌】成本区增强',
    'ENERGY_COMPRESSION_BREAKOUT': '【日线-王牌】潜龙在渊',
    'FIRST_BREAKOUT': '【日线-右侧】底部首板',
    'BBAND_SQUEEZE_BREAKOUT': '【日线-右侧】布林收口突破',
    'CONSOLIDATION_BREAKOUT': '【日线-右侧】盘整区突破',
    'CHIP_COST_BREAKTHROUGH': '【日线-右侧】筹码成本区突破',
    # B.2 趋势持续/加速类
    'OLD_DUCK_HEAD': '【日线-王牌】老鸭头',
    'CHIP_PRESSURE_RELEASE': '【日线-王牌】筹码压力释放',
    'MA_ACCELERATION': '【日线-右侧】均线加速上涨',
    'N_SHAPE_RELAY': '【日线-右侧】N字板接力',
    'BULLISH_FLAG': '【日线-右侧】上升旗形',
    'CHIP_HURDLE_CLEAR': '【日线-右侧】筹码关口扫清',
    'KLINE_THREE_SOLDIERS': '【日线-右侧】红三兵',
    'MOMENTUM_BREAKOUT': '【日线-右侧】动量突破',
    'DOJI_CONTINUATION': '【日线-右侧】十字星中继',
    'RELATIVE_STRENGTH_MAVERICK': '【日线-右侧】逆市强人',
    # B.3 回踩买入类
    'PULLBACK_FIBONACCI': '【日线-右侧】斐波那契回撤',
    'PULLBACK_NORMAL': '【日线-右侧】常规回踩',
    'PULLBACK_STEADY_CLIMB': '【日线-右侧】稳步回踩',
    'PULLBACK_SETUP': '【日线-右侧】回踩准备', # 注意：这是一个准备信号，不是直接买点
    # B.4 指标确认类
    'MACD_ZERO_CROSS': '【日线-指标】MACD零轴金叉',
    'MACD_LOW_CROSS': '【日线-指标】MACD低位金叉',
    'MACD_HIGH_CROSS': '【日线-指标】MACD高位金叉',
    'DMI_CROSS': '【日线-指标】DMI金叉',

    # --- C. 独立/特殊剧本 (Context-Independent Plays) ---
    # 描述: 基于极端市场情绪或特殊结构，无需常规周线前提。
    'EARTH_HEAVEN_BOARD': '【日线-终极】地天板',

    # ==================================================================
    # ===            周线战略指令 (Strategic Directives)             ===
    # ==================================================================
    # 描述: 这些是由总指挥官(MultiTimeframeTrendStrategy)将周线原始信号翻译成的、
    #      日线策略能直接理解的作战指令。它们作为基础分项(BASE_)或前提条件出现。

    # --- A. 许可型指令 (用于解锁左侧交易) ---
    'CONTEXT_STRATEGIC_BOTTOMING_W': '【周策-许可】战略性筑底', # 来自 playbook_COPPOCK_STABILIZING_W

    # --- B. 增强型指令 (作为右侧交易的强力加分项) ---
    'EVENT_STRATEGIC_ACCELERATING_W': '【周策-增强】战略性加速', # 来自 playbook_COPPOCK_ACCELERATING_W
    
    # ▼▼▼【代码修改】: 更新周线剧本ID以匹配代码生成的大写全名，并补全缺失项 ▼▼▼
    # --- C. 基础背景/突破指令 (作为基础分) ---
    'BASE_SIGNAL_BREAKOUT_TRIGGER': '【周策-王牌】关键位突破', # 来自 ace_signal_breakout_trigger_playbook
    'BASE_MA20_RISING_STATE': '【周策】MA21上升状态', # 来自 ma20_rising_state_playbook
    'BASE_MA20_TURN_UP_EVENT': '【周策】MA21拐头向上', # 来自 ma20_turn_up_event_playbook
    'BASE_EARLY_UPTREND': '【周策】早期上升趋势', # 来自 early_uptrend_playbook
    'BASE_CLASSIC_BREAKOUT': '【周策】经典高点突破', # 来自 classic_breakout_playbook
    'BASE_MA_UPTREND': '【周策】均线多头排列', # 来自 ma_uptrend_playbook
    'BASE_BOX_CONSOLIDATION_BREAKOUT': '【周策】专业箱体突破', # 来自 box_consolidation_breakout_playbook
    'BASE_OVERSOLD_REBOUND_BIAS': '【周策】BIAS超跌反弹', # 来自 oversold_rebound_bias_playbook
    'BASE_TRIX_GOLDEN_CROSS': '【周策】TRIX金叉', # 来自 trix_golden_cross_playbook
    'BASE_STRATEGIC_ACCEL': '【周策-王牌】估波曲线加速', # 来自 EVENT_STRATEGIC_ACCELERATING_W 的加分项
    # ▲▲▲【代码修改结束】▲▲▲

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
    
    # ==================================================================
    # ===                  兼容性/旧版ID (Legacy)                    ===
    # ==================================================================
    'playbook_coppock_stabilizing_W': '【周线-原始】估波曲线企稳',
    'playbook_coppock_accelerating_W': '【周线-原始】估波曲线加速',
}

# 合并所有映射，方便过滤器调用
# 策略名称的优先级高于剧本名称，以防重名
DISPLAY_MAP = {**PLAYBOOK_NAME_MAP, **STRATEGY_NAME_MAP}
