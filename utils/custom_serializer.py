from django_redis.serializers.json import JSONSerializer
import json

class CustomJSONSerializer(JSONSerializer):
    def dumps(self, value):
        return json.dumps(value, ensure_ascii=False).encode('utf-8')
