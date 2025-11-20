# /var/www/chaoyue_dreams/dao_manager/apps.py  (这是您新建的文件)

import os
import logging
from pathlib import Path
from django.apps import AppConfig
from django.conf import settings

logger = logging.getLogger(__name__)

# 请将下面的 'StockAnalysisConfig' 和 'dao_manager' 替换成您自己的app名称
# 例如，如果您的app叫 'core'，则类名为 'CoreConfig'，name属性为 'core'
class DaoManagerConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'dao_manager' # 【重要】这里必须是您的app的准确名称
    def ready(self):
        """
        【核心代码】Django应用启动时执行的初始化逻辑。
        - 策略:
          1. 在应用启动时，为当前运行用户（通常是 www-data）创建 Tushare 所需的 token 文件。
          2. 这是为了解决 Tushare 旧接口（如 realtime_quote）强制从文件系统读取 token 的问题。
          3. 此方法可以根治 'EmptyDataError: No columns to parse from file' 错误。
        """
        super().ready()
        token = getattr(settings, 'API_LICENCES_TUSHARE', None)
        if not token:
            logger.critical("TUSHARE_TOKEN 未在 Django settings 中配置，Tushare 初始化被跳过。")
            return
        try:
            home_dir = Path.home()
            tushare_dir = home_dir / ".tushare"
            token_file = tushare_dir / "token"
            tushare_dir.mkdir(parents=True, exist_ok=True)
            if not token_file.exists() or token_file.read_text() != token:
                token_file.write_text(token)
                current_user = os.environ.get('USER', '未知用户') 
                logger.info(f"Tushare token 文件已为用户 '{current_user}' 在路径 '{token_file}' 成功创建/更新。")
        except Exception as e:
            logger.error(f"在 Django AppConfig.ready 中创建 Tushare token 文件时发生严重错误: {e}", exc_info=True)
