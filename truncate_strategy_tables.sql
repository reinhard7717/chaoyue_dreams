-- 步骤1：暂时禁用外键约束检查。这是本次操作的关键。
SET FOREIGN_KEY_CHECKS = 0;

-- 步骤2：现在可以安全地清空这些表，顺序不再重要。
-- 因为外键检查被禁用了，数据库不会再报错。
TRUNCATE TABLE strategy_score_component;
TRUNCATE TABLE strategy_signal_playbook_detail;
TRUNCATE TABLE strategy_daily_position_snapshot;
TRUNCATE TABLE strategy_daily_score;
TRUNCATE TABLE strategy_trading_signal;
TRUNCATE TABLE strategy_position_tracker;
TRUNCATE TABLE strategy_daily_state;
TRUNCATE TABLE strategy_atomic_signal_performance;
-- 步骤3：【至关重要】重新启用外键约束检查，恢复数据库的正常保护机制。
SET FOREIGN_KEY_CHECKS = 1;
