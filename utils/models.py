from django.db import models
from django.core.serializers.json import DjangoJSONEncoder

class BaseModel(models.Model):
    """
    基础模型类
    
    所有模型的基类，包含通用字段如ID、创建时间和更新时间
    """
    # id = models.BigAutoField(primary_key=True, verbose_name="ID", default=None)
    # created_at = models.DateTimeField(auto_now_add=True, verbose_name="创建时间")
    # updated_at = models.DateTimeField(auto_now=True, verbose_name="更新时间")
    
    class Meta:
        abstract = True  # 设置为抽象类，这样Django不会为这个模型创建数据表
        
    def __str__(self):
        """
        默认的字符串表示方法
        如果子类没有覆盖此方法，将显示模型名称和ID
        """
        return f"{self.__class__.__name__}-{self.id}"
    
    def to_dict(self):
        """将模型转换为字典"""
        result = {}
        for field in self._meta.fields:
            value = getattr(self, field.name)
            if isinstance(value, models.Model):
                # 如果是关联模型，只获取其主键
                result[field.name] = value.pk
            else:
                result[field.name] = value
        return result
    
class ModelJSONEncoder(DjangoJSONEncoder):
    """自定义JSON编码器，用于序列化模型对象"""
    def default(self, obj):
        if isinstance(obj, models.Model):
            return obj.to_dict()
        return super().default(obj)
