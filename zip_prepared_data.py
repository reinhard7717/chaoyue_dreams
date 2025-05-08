import os
import re
import py7zr
import argparse
import numpy as np
from datetime import datetime

def is_stock_code_dir(name):
    return re.match(r'^\d{6}\.(SZ|SH)$', name, re.IGNORECASE) is not None

def compress_npy_to_npz(prepared_data_dir):
    for file in os.listdir(prepared_data_dir):
        if file.endswith('.npy'):
            npy_path = os.path.join(prepared_data_dir, file)
            npz_path = npy_path.replace('.npy', '.npz')
            if os.path.exists(npz_path):
                continue  # 已有压缩文件则跳过
            arr = np.load(npy_path)
            # 降低精度为float32
            if arr.dtype == np.float64:
                arr = arr.astype(np.float32)
            np.savez_compressed(npz_path, arr)
            print(f"压缩 {npy_path} -> {npz_path}")
            # 可选：删除原始npy文件
            # os.remove(npy_path)

def seven_zip_all_prepared_data(base_model_dir, since_time_str):
    since_time = datetime.strptime(since_time_str, "%Y-%m-%d %H:%M:%S")
    time_suffix = since_time.strftime("%Y%m%d_%H%M%S")
    seven_zip_path = os.path.join(base_model_dir, f"all_prepared_data_{time_suffix}.7z")
    print(f"只打包最后修改时间大于 {since_time_str} 的文件。")
    filters = [{'id': py7zr.FILTER_LZMA2, 'preset': 9}]
    with py7zr.SevenZipFile(seven_zip_path, 'w', filters=filters) as archive:
        for stock_code in os.listdir(base_model_dir):
            stock_dir = os.path.join(base_model_dir, stock_code)
            if os.path.isdir(stock_dir) and is_stock_code_dir(stock_code):
                prepared_data_dir = os.path.join(stock_dir, "prepared_data")
                if os.path.isdir(prepared_data_dir):
                    # 先批量压缩npy为npz
                    compress_npy_to_npz(prepared_data_dir)
                    for file_name in os.listdir(prepared_data_dir):
                        file_path = os.path.join(prepared_data_dir, file_name)
                        if os.path.isfile(file_path):
                            # 只打包 .npz 和 .save/.gz 文件
                            if not (file_name.endswith('.npz') or file_name.endswith('.save') or file_name.endswith('.gz')):
                                continue
                            file_mtime = datetime.fromtimestamp(os.path.getmtime(file_path))
                            if file_mtime > since_time:
                                arcname = os.path.join(stock_code, file_name)
                                archive.write(file_path, arcname)
                                print(f"已添加: {arcname} (修改时间: {file_mtime})")
    print(f"所有数据已打包到: {seven_zip_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="批量压缩并打包指定时间之后的prepared_data文件到7z")
    parser.add_argument("base_model_dir", help="模型根目录")
    parser.add_argument("since_time", help="只打包修改时间大于此时间的文件，格式: YYYY-MM-DD HH:MM:SS")
    args = parser.parse_args()
    seven_zip_all_prepared_data(args.base_model_dir, args.since_time)
