import os
import sys
import argparse
import datetime  # 虽然不再用于日期检查，但保留以防万一或未来扩展
import subprocess
import json
import pickle

# --- 配置设置 ---
# 策略数据根目录
STRATEGY_DATA_DIR = '/data/chaoyue_dreams/models'
# 需要检查的子文件夹名称
PREPARED_DATA_SUBDIR = 'prepared_data'
# 输出的压缩包名称
ARCHIVE_NAME = '/data/chaoyue_dreams/prepared_data_archive.7z'
# 7z 命令基础 (a: 添加到压缩包, -mx=9: 最高压缩率)
SEVEN_ZIP_COMMAND_BASE = ['7z', 'a', '-mx=9']
# 必需的特定文件名集合
REQUIRED_FILENAMES = {
    'all_prepared_data_transformer.npz',
    'trend_following_transformer_feature_scaler.save',
    'trend_following_transformer_selected_features.json',
    'trend_following_transformer_target_scaler.save'
}

def main():
    """
    主函数，解析参数，遍历目录范围，打包文件。
    """
    # --- 参数解析 ---
    parser = argparse.ArgumentParser(description='根据子文件夹名称范围打包股票策略准备数据。')
    parser.add_argument('first_item_name', type=str, help='范围的起始子文件夹名称 (包含)')
    parser.add_argument('last_item_name', type=str, help='范围的结束子文件夹名称 (包含)')
    args = parser.parse_args()
    first_item_name = args.first_item_name
    last_item_name = args.last_item_name
    print(f"INFO: Processing data for item names from '{first_item_name}' to '{last_item_name}' (inclusive).")
    # 存储需要打包的 .npz、.save 和 .json 文件的相对路径（相对于 STRATEGY_DATA_DIR）
    files_to_archive_relative = []
    # 存储脚本执行时的当前工作目录
    original_cwd = os.getcwd()
    # 构建输出压缩包的完整路径，确保它在脚本执行的当前目录下创建
    archive_full_path = os.path.join(original_cwd, ARCHIVE_NAME)
    # --- 遍历目录 ---
    print(f"INFO: Scanning directory: {STRATEGY_DATA_DIR}")
    if not os.path.isdir(STRATEGY_DATA_DIR):
        print(f"ERROR: Strategy data directory not found: {STRATEGY_DATA_DIR}")
        sys.exit(1)
    # 获取所有子项名称并按字典序排序，以便处理范围
    all_item_names = sorted(os.listdir(STRATEGY_DATA_DIR))
    # 查找起始和结束 item_name 在排序列表中的位置
    try:
        start_index = all_item_names.index(first_item_name)
    except ValueError:
        print(f"ERROR: First item name '{first_item_name}' not found in {STRATEGY_DATA_DIR}.")
        sys.exit(1)
    try:
        end_index = all_item_names.index(last_item_name)
    except ValueError:
        print(f"ERROR: Last item name '{last_item_name}' not found in {STRATEGY_DATA_DIR}.")
        sys.exit(1)
    # 检查范围是否有效
    if start_index > end_index:
        print(f"ERROR: Start item name '{first_item_name}' comes after end item name '{last_item_name}' in sorted order.")
        sys.exit(1)
    # 遍历指定范围内的子项
    items_to_process = all_item_names[start_index : end_index + 1]
    print(f"INFO: Found {len(items_to_process)} items within the specified range.")
    for item_name in items_to_process:
        stock_code_path = os.path.join(STRATEGY_DATA_DIR, item_name)
        # 检查是否是目录
        if os.path.isdir(stock_code_path):
            prepared_data_path = os.path.join(stock_code_path, PREPARED_DATA_SUBDIR)
            # 检查是否存在 prepared_data 子目录
            if os.path.isdir(prepared_data_path):
                print(f"INFO: Checking {prepared_data_path}")
                # --- 检查必需的特定文件是否存在 ---
                files_in_prepared_data = set(os.listdir(prepared_data_path))
                if not REQUIRED_FILENAMES.issubset(files_in_prepared_data):
                    missing_files = REQUIRED_FILENAMES - files_in_prepared_data
                    print(f"INFO: Skipping {prepared_data_path}: Missing required specific files: {missing_files}")
                    continue
                print(f"INFO: All required specific files found in {prepared_data_path}.")
                # 遍历 prepared_data 目录，收集所有 .npz、.save 和 .json 文件的相对路径
                for root, _, files in os.walk(prepared_data_path):
                    for file in files:
                        if file.endswith('.npz') or file.endswith('.save') or file.endswith('.json'):  # 同时收集 .npz、.save 和 .json 文件
                            full_path = os.path.join(root, file)
                            # 计算相对于 STRATEGY_DATA_DIR 的路径，保留目录结构
                            relative_path = os.path.relpath(full_path, STRATEGY_DATA_DIR)
                            print(f"INFO: Adding file to archive list: {relative_path}")  #打印，方便调试
                            files_to_archive_relative.append(relative_path)
    # --- 执行打包 ---
    if not files_to_archive_relative:
        print(f"INFO: No .npz, .save or .json files found within the range '{first_item_name}' to '{last_item_name}' with all required specific files. No archive will be created.")
        sys.exit(0)
    # 构建 7z 命令
    # 命令格式: 7z a -mx=9 <archive_full_path> <file1_relative> <file2_relative> ...
    seven_zip_command = SEVEN_ZIP_COMMAND_BASE + [archive_full_path] + files_to_archive_relative
    # print(f"INFO: Executing command from {STRATEGY_DATA_DIR}: {' '.join(seven_zip_command)}")
    try:
        # 切换到 STRATEGY_DATA_DIR 目录，以便 7z 使用相对路径
        os.chdir(STRATEGY_DATA_DIR)
        print(f"INFO: Changed current directory to {os.getcwd()}")
        # 执行 7z 命令
        result = subprocess.run(seven_zip_command, capture_output=True, text=True, check=True)
        print("INFO: 7z command executed successfully.")
        print("--- 7z STDOUT ---")
        print(result.stdout)
        print("--- 7z STDERR ---")
        print(result.stderr)
        print(f"INFO: Archive created: {archive_full_path}")
    except FileNotFoundError:
        print("ERROR: 7z command not found. Please ensure 7z is installed and in your system's PATH.")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"ERROR: 7z command failed with exit code {e.returncode}")
        print("--- 7z STDOUT ---")
        print(e.stdout)
        print("--- 7z STDERR ---")
        print(e.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: An unexpected error occurred during archiving: {e}")
        sys.exit(1)
    finally:
        # 切换回原始工作目录
        os.chdir(original_cwd)
        print(f"INFO: Changed back to original directory {os.getcwd()}")


if __name__ == "__main__":
    main()
