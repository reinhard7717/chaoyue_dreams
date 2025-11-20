import os
import sys
import argparse
import subprocess
import shutil  # 用于删除目录

# --- 配置设置 ---
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
STRATEGY_DATA_DIR = os.path.join(PROJECT_DIR, 'models')
TRANSFORMER_RESULT_SUBDIR = 'trained_model'
PREPARED_DATA_SUBDIR = 'prepared_data'
TRANSFORMER_RESULT_NAME = 'transformer_model_result.7z'
ARCHIVE_NAME = os.path.join(PROJECT_DIR, TRANSFORMER_RESULT_NAME)
SEVEN_ZIP_COMMAND_BASE = ['7z', 'a', '-mx=9']

def main():
    parser = argparse.ArgumentParser(description='根据子文件夹名称范围打包股票策略训练完成文件。')
    parser.add_argument('first_item_name', type=str, help='范围的起始子文件夹名称 (包含)')
    parser.add_argument('last_item_name', type=str, help='范围的结束子文件夹名称 (包含)')
    args = parser.parse_args()
    first_item_name = args.first_item_name
    last_item_name = args.last_item_name
    print(f"INFO: Processing trained_model for item names from '{first_item_name}' to '{last_item_name}' (inclusive).")
    files_to_archive_relative = []
    original_cwd = os.getcwd()
    archive_full_path = ARCHIVE_NAME
    print(f"INFO: Scanning directory: {STRATEGY_DATA_DIR}")
    if not os.path.isdir(STRATEGY_DATA_DIR):
        print(f"ERROR: Strategy data directory not found: {STRATEGY_DATA_DIR}")
        sys.exit(1)
    all_item_names = sorted(os.listdir(STRATEGY_DATA_DIR))
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
    if start_index > end_index:
        print(f"ERROR: Start item name '{first_item_name}' comes after end item name '{last_item_name}' in sorted order.")
        sys.exit(1)
    items_to_process = all_item_names[start_index : end_index + 1]
    print(f"INFO: Found {len(items_to_process)} items within the specified range.")
    # 第一步：先批量删除所有需要删除的 prepared_data 目录
    for item_name in items_to_process:
        stock_code_path = os.path.join(STRATEGY_DATA_DIR, item_name)
        if os.path.isdir(stock_code_path):
            trained_model_path = os.path.join(stock_code_path, TRANSFORMER_RESULT_SUBDIR)
            if os.path.isdir(trained_model_path):
                prepared_data_path = os.path.join(stock_code_path, PREPARED_DATA_SUBDIR)
                if os.path.isdir(prepared_data_path):
                    try:
                        shutil.rmtree(prepared_data_path)
                        print(f"INFO: Deleted prepared_data directory: {prepared_data_path}")
                    except Exception as e:
                        print(f"ERROR: Failed to delete prepared_data directory {prepared_data_path}: {e}")
    # 第二步：收集所有 trained_model 目录下的文件
    for item_name in items_to_process:
        stock_code_path = os.path.join(STRATEGY_DATA_DIR, item_name)
        if os.path.isdir(stock_code_path):
            trained_model_path = os.path.join(stock_code_path, TRANSFORMER_RESULT_SUBDIR)
            if os.path.isdir(trained_model_path):
                print(f"INFO: Checking {trained_model_path}")
                for root, _, files in os.walk(trained_model_path):
                    for file in files:
                        full_path = os.path.join(root, file)
                        relative_path = os.path.relpath(full_path, STRATEGY_DATA_DIR)
                        print(f"INFO: Adding file to archive list: {relative_path}")
                        files_to_archive_relative.append(relative_path)
            else:
                print(f"INFO: Skipping {trained_model_path}: Directory does not exist.")
    if not files_to_archive_relative:
        print(f"INFO: No files found in trained_model directories within the range '{first_item_name}' to '{last_item_name}'. No archive will be created.")
        sys.exit(0)
    seven_zip_command = SEVEN_ZIP_COMMAND_BASE + [archive_full_path] + files_to_archive_relative
    print(f"INFO: Executing command from {STRATEGY_DATA_DIR}: {' '.join(seven_zip_command)}")
    try:
        os.chdir(STRATEGY_DATA_DIR)
        print(f"INFO: Changed current directory to {os.getcwd()}")
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
        os.chdir(original_cwd)
        print(f"INFO: Changed back to original directory {os.getcwd()}")

if __name__ == "__main__":
    main()
