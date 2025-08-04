# stock_models/migrations/0072_temp_reset_positiontracker.py

from django.db import migrations

class Migration(migrations.Migration):

    dependencies = [
        # 依赖于之前那个出错的迁移文件
        ('stock_models', '0071_alter_positiontracker_options_and_more'),
    ]

    operations = [
        # --- 代码修改开始 ---
        # [修改原因] 升级修复脚本，以处理外键约束导致的无法删表问题。
        
        # 步骤1: 暂时关闭外键约束检查，允许我们进行危险操作
        migrations.RunSQL(
            "SET FOREIGN_KEY_CHECKS = 0;",
            reverse_sql="SET FOREIGN_KEY_CHECKS = 1;",
        ),
        
        # 步骤2: 删除引用 PositionTracker 的“子表”
        migrations.RunSQL(
            "DROP TABLE IF EXISTS strategy_daily_position_snapshot;",
            reverse_sql=migrations.RunSQL.noop,
        ),
        
        # 步骤3: 删除 PositionTracker 表本身（“父表”）
        migrations.RunSQL(
            "DROP TABLE IF EXISTS strategy_position_tracker;",
            reverse_sql=migrations.RunSQL.noop,
        ),
        
        # 步骤4: （以防万一）删除 Django 默认命名的表
        migrations.RunSQL(
            "DROP TABLE IF EXISTS stock_models_positiontracker;",
            reverse_sql=migrations.RunSQL.noop,
        ),
        
        # 步骤5: 重新开启外键约束检查，恢复数据库的正常保护机制
        migrations.RunSQL(
            "SET FOREIGN_KEY_CHECKS = 1;",
            reverse_sql="SET FOREIGN_KEY_CHECKS = 0;",
        ),
        # --- 代码修改结束 ---
    ]
