import os
import sys
import argparse
import datetime
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

def is_file_within_date_range(filepath, start_date, end_date):
    """
    检查文件的最后修改日期是否在指定的日期范围内。

    Args:
        filepath (str): 文件的完整路径。
        start_date (datetime.date): 起始日期 (包含)。
        end_date (datetime.date): 截止日期 (包含)。

    Returns:
        bool: 如果文件修改日期在范围内则返回 True，否则返回 False。
    """
    try:
        # 获取文件的最后修改时间戳
        mod_timestamp = os.stat(filepath).st_mtime
        # 将时间戳转换为datetime对象
        mod_datetime = datetime.datetime.fromtimestamp(mod_timestamp)
        # 提取日期部分
        mod_date = mod_datetime.date()

        # 检查日期是否在范围内
        is_within = start_date <= mod_date <= end_date
        # print(f"DEBUG: Checking file {filepath}, modified date: {mod_date}, within range [{start_date} to {end_date}]? {is_within}") # 调试信息
        return is_within
    except FileNotFoundError:
        # 如果文件不存在，则肯定不在日期范围内
        print(f"WARNING: Required file not found during date check: {filepath}") # 警告信息
        return False
    except Exception as e:
        print(f"ERROR: Could not get modification date for {filepath}: {e}") # 错误信息
        return False

def main():
    """
    主函数，解析参数，遍历目录，打包文件。
    """
    # --- 参数解析 ---
    parser = argparse.ArgumentParser(description='根据文件修改日期打包股票策略准备数据。')
    parser.add_argument('start_date', type=str, help='起始日期 (YYYY-MM-DD)')
    parser.add_argument('end_date', type=str, help='截止日期 (YYYY-MM-DD)')

    args = parser.parse_args()

    try:
        # 将输入的日期字符串转换为 datetime.date 对象
        start_date_obj = datetime.datetime.strptime(args.start_date, '%Y-%m-%d').date()
        end_date_obj = datetime.datetime.strptime(args.end_date, '%Y-%m-%d').date()
        print(f"INFO: Processing data modified between {start_date_obj} and {end_date_obj}") # 提示信息
    except ValueError:
        print("ERROR: Invalid date format. Please use YYYY-MM-DD.") # 错误信息
        sys.exit(1)

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

    # 遍历 STRATEGY_DATA_DIR 下的所有子项 (期望是 stock_code 文件夹)
    for item_name in os.listdir(STRATEGY_DATA_DIR):
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

                print(f"INFO: All required specific files found in {prepared_data_path}. Checking dates.") # 提示信息

                # --- 检查必需文件的修改日期 ---
                found_file_in_range = False
                # 只检查必需的特定文件的日期
                for required_file_name in REQUIRED_FILENAMES:
                    file_path = os.path.join(prepared_data_path, required_file_name)
                    # 确保是文件 (虽然前面已经检查过文件名存在，但再次确认是文件类型)
                    if os.path.isfile(file_path):
                        # 检查文件修改日期是否在范围内
                        if is_file_within_date_range(file_path, start_date_obj, end_date_obj):
                            print(f"INFO: Required file {file_path} is within the date range.") # 提示信息
                            found_file_in_range = True
                            # 找到一个符合条件的必需文件即可，将整个目录添加到打包列表
                            break # 跳出必需文件遍历循环

                # 如果在该 prepared_data 目录中找到了符合条件的必需文件，则将整个目录的相对路径添加到打包列表
                if found_file_in_range:
                    relative_prepared_data_path = os.path.join(item_name, PREPARED_DATA_SUBDIR)
                    print(f"INFO: Adding relative path {relative_prepared_data_path} to archive list.") # 提示信息
                    dirs_to_archive_relative.append(relative_prepared_data_path)
            # else:
                # print(f"DEBUG: {prepared_data_path} does not exist or is not a directory.") # 调试信息

    # --- 执行打包 ---
    if not dirs_to_archive_relative:
        print("INFO: No prepared_data directories found with all required specific files and at least one of these files modified within the specified date range. No archive will be created.") # 提示信息
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
        # 修改点：使用 subprocess.run 替代 os.system，更安全且能捕获输出
        # 修改点：capture_output=True, text=True 用于捕获标准输出和标准错误
        # 修改点：check=True 会在命令返回非零退出码时抛出 CalledProcessError
        # 修改点：cwd 参数不再需要，因为我们已经手动 chdir
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
