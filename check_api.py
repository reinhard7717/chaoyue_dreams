import asyncio
import sys

# 解决Python 3.12上asyncio.coroutines问题
if sys.version_info >= (3, 12):
    if not hasattr(asyncio.coroutines, '_DEBUG'):
        asyncio.coroutines._DEBUG = asyncio.coroutines._is_debug_mode()

from api_manager.base_api import BaseAPI

async def check_api_return():
    api = BaseAPI()
    data = await api.get('/data/base/gplist')
    print('数据长度:', len(data))
    
    if data and len(data) > 0:
        print('第一条数据字段:', list(data[0].keys()))
        print('第一条数据内容:', data[0])
    
    return data

if __name__ == "__main__":
    asyncio.run(check_api_return()) 