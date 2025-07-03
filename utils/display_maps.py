# 文件: utils/display_maps.py
# 版本: V1.0 - 中文显示映射
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

# 剧本(Playbook)名称映射
PLAYBOOK_NAME_MAP = {
    # === 日线战术剧本 (Tactical Playbooks) ===
    'BASE_MA_UPTREND': '【基础】均线多头',
    'BASE_EARLY_UPTREND': '【基础】趋势早期',
    'BASE_BREAKOUT_CONFIRM': '【基础】突破确认',
    'BASE_PULLBACK_BUY': '【基础】强势回调',
    'BONUS_CMF_CONFIRM': '【加分】资金流入确认',
    'BONUS_MULTIPLIER': '【加分】动量放大',
    'EXPERT_PLAYBOOK_VOL_PRICE_DIVERGENCE': '【专家】量价背离抄底',
    'EXPERT_PLAYBOOK_LOW_VOL_BREAKOUT': '【专家】缩量蓄势突破',
    'BASE_SIGNAL_BREAKOUT_TRIGGER': '关键位突破/放量突破',

    # === 周线战略剧本 (Strategic Playbooks) ===
    'playbook_ma20_rising_state_W': '【周线】MA20上升状态',
    'playbook_ma20_turn_up_event_W': '【周线】MA20拐头向上',
    'playbook_early_uptrend_W': '【周线】早期上升趋势',
    'playbook_classic_breakout_W': '【周线】经典高点突破',
    'playbook_ma_uptrend_W': '【周线】均线多头排列',
    'playbook_box_consolidation_breakout_W': '【周线】专业箱体突破',
    'playbook_oversold_rebound_bias_W': '【周线】BIAS超跌反弹',
    'playbook_trix_golden_cross_W': '【周线】TRIX金叉',
    'playbook_coppock_bottom_reversal_W': '【周线】Coppock底部反转',
    'playbook_ace_signal_breakout_trigger_W': '【周线】王牌突破信号',
    
    # === 分钟级执行剧本 (Execution Playbooks) ===
    'RESONANCE_FRACTAL_ROCKET': '【分钟】分形火箭',

    # === 复合信号 (Composite Signals) ===
    'BREAKOUT_TRIGGER_SCORE': '【王牌】周线突破观察',
    
    # === 止盈信号  ===
    'EXIT_CODE_101': '15分钟MACD死叉',
    'EXIT_CODE_102': '15分钟RSI顶背离',
    'EXIT_CODE_103': '15分钟KDJ在高位（80以上）死叉',
    
    # ...可以根据需要添加更多剧本名称...
}

# 合并所有映射，方便过滤器调用
# 策略名称的优先级高于剧本名称，以防重名
DISPLAY_MAP = {**PLAYBOOK_NAME_MAP, **STRATEGY_NAME_MAP}

