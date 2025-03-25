from django.db import models


class BaseModel(models.Model):
    """
    基础模型类
    
    所有模型的基类，包含通用字段如ID、创建时间和更新时间
    """
    id = models.BigAutoField(primary_key=True, verbose_name="ID")
    created_at = models.DateTimeField(auto_now_add=True, verbose_name="创建时间")
    updated_at = models.DateTimeField(auto_now=True, verbose_name="更新时间")
    
    class Meta:
        abstract = True  # 设置为抽象类，这样Django不会为这个模型创建数据表
        
    def __str__(self):
        """
        默认的字符串表示方法
        如果子类没有覆盖此方法，将显示模型名称和ID
        """
        return f"{self.__class__.__name__}-{self.id}"
