import os
import re
import py7zr

def is_stock_code_dir(name):
    # 只匹配A股常见股票代码格式
    return re.match(r'^\d{6}\.(SZ|SH)$', name, re.IGNORECASE) is not None

def seven_zip_all_prepared_data(base_model_dir):
    seven_zip_path = os.path.join(base_model_dir, "all_prepared_data.7z")
    with py7zr.SevenZipFile(seven_zip_path, 'w') as archive:
        for stock_code in os.listdir(base_model_dir):
            stock_dir = os.path.join(base_model_dir, stock_code)
            if os.path.isdir(stock_dir) and is_stock_code_dir(stock_code):
                prepared_data_dir = os.path.join(stock_dir, "prepared_data")
                if os.path.isdir(prepared_data_dir):
                    for file_name in os.listdir(prepared_data_dir):
                        file_path = os.path.join(prepared_data_dir, file_name)
                        if os.path.isfile(file_path):
                            arcname = os.path.join(stock_code, file_name)
                            archive.write(file_path, arcname)
                            print(f"已添加: {arcname}")
    print(f"所有数据已打包到: {seven_zip_path}")

if __name__ == "__main__":
    base_model_dir = "/var/www/chaoyue_dreams/models"
    seven_zip_all_prepared_data(base_model_dir)
