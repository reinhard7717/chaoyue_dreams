import os
import re
import py7zr
import argparse
import numpy as np
import joblib # 虽然这里不加载，但为了完整性，如果以后需要处理scaler文件内容，可能需要
from datetime import datetime

# 定义需要合并到 all_prepared_data.npz 的文件及其对应的键名
# 左边是实际的文件名，右边是 npz 中保存的键名 (与 load_prepared_data 对应)
NPZ_DATA_MAP = {
    'lstm_features_scaled_train.npy': 'features_scaled_train',
    'lstm_targets_scaled_train.npy': 'targets_scaled_train',
    'lstm_features_scaled_val.npy': 'features_scaled_val',
    'lstm_targets_scaled_val.npy': 'targets_scaled_val',
    'lstm_features_scaled_test.npy': 'features_scaled_test',
    'lstm_targets_scaled_test.npy': 'targets_scaled_test',
}

# 定义可能的 Scaler 文件名 (包含你实际使用的前缀)
SCALER_FILENAMES = ['trend_following_lstm_feature_scaler.save', 'trend_following_lstm_feature_scaler.save.gz', 'scaler.save', 'scaler.save.gz']
TARGET_SCALER_FILENAMES = ['trend_following_lstm_target_scaler.save', 'trend_following_lstm_target_scaler.save.gz', 'target_scaler.save', 'target_scaler.save.gz']

def is_stock_code_dir(name):
    """检查目录名是否符合股票代码格式"""
    return re.match(r'^\d{6}\.(SZ|SH)$', name, re.IGNORECASE) is not None

def find_existing_file(directory, possible_filenames):
    """在目录中查找存在的第一个文件"""
    for filename in possible_filenames:
        filepath = os.path.join(directory, filename)
        if os.path.exists(filepath):
            return filepath
    return None

def create_combined_npz(prepared_data_dir):
    """
    加载单个 .npy 文件 (根据 NPZ_DATA_MAP 定义)，合并并保存为 all_prepared_data.npz
    返回创建的 npz 文件路径，如果失败则返回 None
    """
    all_data_npz_path = os.path.join(prepared_data_dir, "all_prepared_data.npz")
    data_to_save = {}
    npy_files_exist = True
    total_original_npy_size = 0

    print(f"正在处理目录: {prepared_data_dir}")

    # 检查并加载所有必需的 .npy 文件
    for npy_filename, npz_key in NPZ_DATA_MAP.items():
        npy_path = os.path.join(prepared_data_dir, npy_filename)
        if not os.path.exists(npy_path):
            print(f"  警告: 缺少文件 {npy_filename}，跳过此目录的合并。")
            npy_files_exist = False
            break
        try:
            arr = np.load(npy_path)
            if arr.dtype == np.float64:
                arr = arr.astype(np.float32)
            data_to_save[npz_key] = arr # 使用 npz_key 作为保存的键名
            total_original_npy_size += os.path.getsize(npy_path)
        except Exception as e:
            print(f"  错误: 加载文件 {npy_filename} 时出错: {e}")
            npy_files_exist = False
            break

    if not npy_files_exist or not data_to_save:
        # 如果有文件缺失或加载失败，删除可能已创建的不完整 npz 文件
        if os.path.exists(all_data_npz_path):
             os.remove(all_data_npz_path)
             print(f"  已删除不完整的合并文件: {os.path.basename(all_data_npz_path)}")
        return None, 0, 0 # 返回 None 和 0 大小

    # 保存合并后的 npz 文件
    try:
        np.savez_compressed(all_data_npz_path, **data_to_save)
        combined_npz_size = os.path.getsize(all_data_npz_path)
        print(f"  成功创建合并文件: {os.path.basename(all_data_npz_path)}")
        print(f"  合并前 .npy 总体积: {total_original_npy_size/1024/1024:.2f} MB， 合并后 .npz 体积: {combined_npz_size/1024/1024:.2f} MB")
        if total_original_npy_size > 0:
             print(f"  压缩率: {combined_npz_size/total_original_npy_size:.2%}")
        return all_data_npz_path, total_original_npy_size, combined_npz_size
    except Exception as e:
        print(f"  错误: 保存合并文件 {os.path.basename(all_data_npz_path)} 时出错: {e}")
        # 保存失败也删除可能已创建的不完整文件
        if os.path.exists(all_data_npz_path):
             os.remove(all_data_npz_path)
             print(f"  已删除保存失败的合并文件: {os.path.basename(all_data_npz_path)}")
        return None, 0, 0

def seven_zip_all_prepared_data(base_model_dir, since_time_str):
    """
    遍历股票目录，创建合并的 npz 文件，并打包 npz 和 scaler 文件到 7z
    """
    try:
        since_time = datetime.strptime(since_time_str, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        print(f"错误: 无效的时间格式 '{since_time_str}'。请使用 YYYY-MM-DD HH:MM:SS 格式。")
        return

    time_suffix = since_time.strftime("%Y%m%d_%H%M%S")
    seven_zip_path = os.path.join(base_model_dir, f"all_prepared_data_{time_suffix}.7z")
    print(f"只打包最后修改时间大于 {since_time_str} 的文件。")

    filters = [{'id': py7zr.FILTER_LZMA2, 'preset': 9}]
    total_original_npy_size_all = 0
    total_combined_npz_size_all = 0
    files_added_count = 0
    processed_stock_count = 0
    added_stock_count = 0

    with py7zr.SevenZipFile(seven_zip_path, 'w', filters=filters) as archive:
        for stock_code in os.listdir(base_model_dir):
            stock_dir = os.path.join(base_model_dir, stock_code)
            if os.path.isdir(stock_dir) and is_stock_code_dir(stock_code):
                prepared_data_dir = os.path.join(stock_dir, "prepared_data")
                if os.path.isdir(prepared_data_dir):
                    processed_stock_count += 1
                    print(f"\n--- 正在处理股票: {stock_code} ---")

                    # 1. 创建合并的 all_prepared_data.npz 文件
                    combined_npz_path, original_npy_size, combined_npz_size = create_combined_npz(prepared_data_dir)

                    if combined_npz_path:
                        total_original_npy_size_all += original_npy_size
                        total_combined_npz_size_all += combined_npz_size

                        # 2. 查找 Scaler 文件
                        scaler_path = find_existing_file(prepared_data_dir, SCALER_FILENAMES)
                        target_scaler_path = find_existing_file(prepared_data_dir, TARGET_SCALER_FILENAMES)

                        # 3. 确定需要打包的文件列表
                        files_to_archive = []
                        if combined_npz_path:
                            files_to_archive.append(combined_npz_path)
                        if scaler_path:
                            files_to_archive.append(scaler_path)
                        if target_scaler_path:
                            files_to_archive.append(target_scaler_path)

                        # 4. 检查修改时间并添加到压缩包
                        stock_added_to_archive = False
                        print(f"  检查 {stock_code} 的文件是否需要打包...")
                        for file_path in files_to_archive:
                            if os.path.exists(file_path):
                                file_mtime = datetime.fromtimestamp(os.path.getmtime(file_path))
                                if file_mtime > since_time:
                                    # 构建在压缩包内的相对路径
                                    # 注意：这里将 prepared_data 也包含在路径中，与 load_prepared_data 的逻辑一致
                                    arcname = os.path.join(stock_code, os.path.basename(prepared_data_dir), os.path.basename(file_path))
                                    try:
                                        archive.write(file_path, arcname)
                                        print(f"    已添加: {arcname} (修改时间: {file_mtime})")
                                        files_added_count += 1
                                        stock_added_to_archive = True
                                    except Exception as e:
                                        print(f"    错误: 添加文件 {file_path} 到压缩包时出错: {e}")
                                else:
                                     print(f"    跳过文件: {os.path.basename(file_path)} (修改时间: {file_mtime} <= {since_time_str})")
                            else:
                                print(f"    警告: 文件 {file_path} 不存在，跳过。")
                        if stock_added_to_archive:
                            added_stock_count += 1
                    else:
                        print(f"--- 股票 {stock_code} 跳过打包，因为合并 NPZ 文件失败 ---")


    print(f"\n==== 打包完成 ====")
    print(f"总共处理了 {processed_stock_count} 个股票目录。")
    print(f"总共打包了 {added_stock_count} 个股票的数据 (至少有一个文件符合时间条件)。")
    print(f"总共添加了 {files_added_count} 个文件到压缩包。")
    print(f"所有符合条件的数据已打包到: {seven_zip_path}")
    print(f"\n==== 总体积统计 (针对成功合并 NPZ 的目录) ====")
    print(f"合并前 .npy 总体积: {total_original_npy_size_all/1024/1024:.2f} MB")
    print(f"合并后 .npz 总体积: {total_combined_npz_size_all/1024/1024:.2f} MB")
    if total_original_npy_size_all > 0:
        print(f"总体压缩率 (仅 .npy -> .npz): {total_combined_npz_size_all/total_original_npy_size_all:.2%}")
    else:
        print("无 .npy 文件被成功合并。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="批量合并股票准备数据到 all_prepared_data.npz，并打包 npz 和 scaler 文件到 7z，只打包指定时间之后修改的文件。")
    parser.add_argument("base_model_dir", help="模型根目录")
    parser.add_argument("since_time", help="只打包修改时间大于此时间的文件，格式: YYYY-MM-DD HH:MM:SS")
    args = parser.parse_args()
    seven_zip_all_prepared_data(args.base_model_dir, args.since_time)
