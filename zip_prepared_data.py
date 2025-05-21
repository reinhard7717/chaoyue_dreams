import os
import sys
import argparse
import datetime # 虽然不再用于日期检查，但保留以防万一或未来扩展
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

# --- 函数定义 ---

# 移除 is_file_within_date_range 函数，因为不再需要日期检查

def main():
    """
    主函数，解析参数，遍历目录范围，打包文件。
    """
    # --- 参数解析 ---
    parser = argparse.ArgumentParser(description='根据子文件夹名称范围打包股票策略准备数据。')
    # 修改：参数变为起始和结束的 item_name (子文件夹名称)
    parser.add_argument('first_item_name', type=str, help='范围的起始子文件夹名称 (包含)') # 修改行
    parser.add_argument('last_item_name', type=str, help='范围的结束子文件夹名称 (包含)') # 修改行

    args = parser.parse_args()

    first_item_name = args.first_item_name # 修改行
    last_item_name = args.last_item_name # 修改行
    print(f"INFO: Processing data for item names from '{first_item_name}' to '{last_item_name}' (inclusive).") # 修改行

    # 存储需要打包的 prepared_data 文件夹的相对路径 (相对于 STRATEGY_DATA_DIR)
    dirs_to_archive_relative = []

    # 存储脚本执行时的当前工作目录
    original_cwd = os.getcwd()
    # 构建输出压缩包的完整路径，确保它在脚本执行的当前目录下创建
    archive_full_path = os.path.join(original_cwd, ARCHIVE_NAME)

    # --- 遍历目录 ---
    print(f"INFO: Scanning directory: {STRATEGY_DATA_DIR}") # 提示信息
    if not os.path.isdir(STRATEGY_DATA_DIR):
        print(f"ERROR: Strategy data directory not found: {STRATEGY_DATA_DIR}") # 错误信息
        sys.exit(1)

    # 获取所有子项名称并按字典序排序，以便处理范围
    all_item_names = sorted(os.listdir(STRATEGY_DATA_DIR)) # 修改行

    # 查找起始和结束 item_name 在排序列表中的位置
    try:
        start_index = all_item_names.index(first_item_name) # 修改行
    except ValueError:
        print(f"ERROR: First item name '{first_item_name}' not found in {STRATEGY_DATA_DIR}.") # 修改行
        sys.exit(1)

    try:
        end_index = all_item_names.index(last_item_name) # 修改行
    except ValueError:
        print(f"ERROR: Last item name '{last_item_name}' not found in {STRATEGY_DATA_DIR}.") # 修改行
        sys.exit(1)

    # 检查范围是否有效
    if start_index > end_index: # 修改行
        print(f"ERROR: Start item name '{first_item_name}' comes after end item name '{last_item_name}' in sorted order.") # 修改行
        sys.exit(1)

    # 遍历指定范围内的子项
    items_to_process = all_item_names[start_index : end_index + 1] # 修改行
    print(f"INFO: Found {len(items_to_process)} items within the specified range.") # 修改行

    for item_name in items_to_process: # 修改行
        stock_code_path = os.path.join(STRATEGY_DATA_DIR, item_name)

        # 检查是否是目录
        if os.path.isdir(stock_code_path):
            prepared_data_path = os.path.join(stock_code_path, PREPARED_DATA_SUBDIR)

            # 检查是否存在 prepared_data 子目录
            if os.path.isdir(prepared_data_path):
                print(f"INFO: Checking {prepared_data_path}") # 提示信息

                # --- 检查必需的特定文件是否存在 ---
                files_in_prepared_data = set(os.listdir(prepared_data_path))

                # 检查是否所有必需的特定文件名都存在
                if not REQUIRED_FILENAMES.issubset(files_in_prepared_data):
                    missing_files = REQUIRED_FILENAMES - files_in_prepared_data
                    print(f"INFO: Skipping {prepared_data_path}: Missing required specific files: {missing_files}") # 提示信息
                    continue # 跳过当前 prepared_data 目录

                print(f"INFO: All required specific files found in {prepared_data_path}.") # 修改行 (移除日期检查相关信息)

                # --- 添加到打包列表 (不再检查日期，只要在范围内且文件齐全就打包) ---
                relative_prepared_data_path = os.path.join(item_name, PREPARED_DATA_SUBDIR) # 提示信息
                print(f"INFO: Adding relative path {relative_prepared_data_path} to archive list.") # 提示信息
                dirs_to_archive_relative.append(relative_prepared_data_path) # 提示信息

    # --- 执行打包 ---
    if not dirs_to_archive_relative:
        # 修改提示信息
        print(f"INFO: No prepared_data directories found within the range '{first_item_name}' to '{last_item_name}' with all required specific files. No archive will be created.") # 修改行
        sys.exit(0)

    # 构建 7z 命令
    # 命令格式: 7z a -mx=9 <archive_full_path> <dir1_relative> <dir2_relative> ...
    # 注意：这里将 archive_full_path 作为第一个参数，后面是相对路径列表
    seven_zip_command = SEVEN_ZIP_COMMAND_BASE + [archive_full_path] + dirs_to_archive_relative
    print(f"INFO: Executing command from {STRATEGY_DATA_DIR}: {' '.join(seven_zip_command)}") # 提示信息

    # 使用 try...finally 确保无论是否发生错误都能切换回原始目录
    try:
        # 切换到 STRATEGY_DATA_DIR 目录，以便 7z 使用相对路径
        os.chdir(STRATEGY_DATA_DIR)
        print(f"INFO: Changed current directory to {os.getcwd()}") # 提示信息

        # 执行 7z 命令
        result = subprocess.run(seven_zip_command, capture_output=True, text=True, check=True)
        print("INFO: 7z command executed successfully.") # 提示信息
        print("--- 7z STDOUT ---") # 提示信息
        print(result.stdout) # 输出 7z 的标准输出
        print("--- 7z STDERR ---") # 提示信息
        print(result.stderr) # 输出 7z 的标准错误
        print(f"INFO: Archive created: {archive_full_path}") # 提示信息

    except FileNotFoundError:
        print("ERROR: 7z command not found. Please ensure 7z is installed and in your system's PATH.") # 错误信息
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"ERROR: 7z command failed with exit code {e.returncode}") # 错误信息
        print("--- 7z STDOUT ---") # 提示信息
        print(e.stdout) # 输出 7z 的标准输出
        print("--- 7z STDERR ---") # 提示信息
        print(e.stderr) # 输出 7z 的标准错误
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: An unexpected error occurred during archiving: {e}") # 错误信息
        sys.exit(1)
    finally:
        # 切换回原始工作目录
        os.chdir(original_cwd)
        print(f"INFO: Changed back to original directory {os.getcwd()}") # 提示信息


# --- 脚本入口 ---
if __name__ == "__main__":
    main()
