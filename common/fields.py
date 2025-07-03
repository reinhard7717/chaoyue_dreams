# common/fields.py

from django.db import models

class TimestampField(models.DateTimeField):
    """
    自定义一个DateTimeField，使其在MySQL中映射为TIMESTAMP类型。
    这对于需要数据库层面自动处理时区的场景至关重要。
    """
    def db_type(self, connection):
        # 检查数据库连接的后端是否是MySQL
        if connection.settings_dict['ENGINE'] == 'django.db.backends.mysql':
            # 返回'timestamp'，Django迁移时会生成对应的SQL
            # (6) 表示精度到微秒，与Django的DateTimeField默认行为一致
            return 'timestamp(6)' 
        # 对于其他数据库，保持默认行为
        return super().db_type(connection)

