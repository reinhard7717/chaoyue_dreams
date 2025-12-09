from django.db import models
from django.core.serializers.json import DjangoJSONEncoder

class BaseModel(models.Model):
    class Meta:
        abstract = True  # 设置为抽象类，这样Django不会为这个模型创建数据表
    def __str__(self):
        """
        默认的字符串表示方法
        如果子类没有覆盖此方法，将显示模型名称和ID
        """
        return f"{self.__class__.__name__}-{self.id}"
    def to_dict(self):
        """
        将模型转换为字典
        Returns:
            dict: 包含模型所有字段值的字典
        """
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
    """
    自定义JSON编码器，用于序列化模型对象
    继承自Django的JSON编码器，增加了对模型对象的序列化支持
    """
    def default(self, obj):
        if isinstance(obj, models.Model):
            return obj.to_dict()
        return super().default(obj)
