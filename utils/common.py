# utils/common.py

def chunk_list(input_list, chunk_size):
    """
    将列表分割成指定大小的块
    """
    return [input_list[i:i + chunk_size] for i in range(0, len(input_list), chunk_size)]
