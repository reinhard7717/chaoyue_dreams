import os
import shutil
import django

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "chaoyue_dreams.settings")  # 请替换为你的Django项目settings路径
django.setup()

from django.conf import settings

root_dir = getattr(settings, "STRATEGY_DATA_DIR", None)
if not root_dir:
    raise ValueError("请在settings中配置STRATEGY_DATA_DIR路径")

# 第一次遍历，移动文件
for stock_code in os.listdir(root_dir):
    stock_path = os.path.join(root_dir, stock_code)
    if not os.path.isdir(stock_path):
        continue  # 不是文件夹跳过
    trained_model_dir = os.path.join(stock_path, "trained_model")
    if not os.path.isdir(trained_model_dir):
        continue  # 没有trained_model文件夹跳过
    weights_folder = os.path.join(trained_model_dir, "trend_following_transformer_weights.pth")
    if not os.path.isdir(weights_folder):
        # print(f"缺少文件夹: {weights_folder}，跳过")
        continue
    target_filename = f"best_transformer_model_{stock_code}.pth"
    src_file = os.path.join(weights_folder, target_filename)
    dst_file = os.path.join(trained_model_dir, target_filename)
    if os.path.exists(src_file):
        shutil.move(src_file, dst_file)  # 移动文件到trained_model目录
        print(f"已移动文件: {src_file} -> {dst_file}")
    else:
        print(f"文件不存在，跳过: {src_file}")

# 合并第二次和第三次遍历
new_trained_model_dir = os.path.join(root_dir, "trained_model")
if not os.path.isdir(new_trained_model_dir):
    os.makedirs(new_trained_model_dir)  # 目录不存在则创建，修改处
    print(f"目录不存在，已创建: {new_trained_model_dir}")

for stock_code in os.listdir(root_dir):
    stock_path = os.path.join(root_dir, stock_code)
    if not os.path.isdir(stock_path):
        continue
    trained_model_dir = os.path.join(stock_path, "trained_model")
    if not os.path.isdir(trained_model_dir):
        continue
    weights_folder = os.path.join(trained_model_dir, "trend_following_transformer_weights.pth")
    # 取消非空判断，直接删除文件夹及其内容
    if os.path.isdir(weights_folder):
        try:
            shutil.rmtree(weights_folder)  # 递归删除文件夹及所有内容，修改处
            print(f"已删除文件夹及其内容: {weights_folder}")
        except Exception as e:
            print(f"删除文件夹失败: {weights_folder}，原因: {e}")

# 将trained_model_dir中所有子目录的.pth文件移动到new_trained_model_dir，存在同名文件则覆盖
    for item in os.listdir(trained_model_dir):
        item_path = os.path.join(trained_model_dir, item)
        if os.path.isfile(item_path) and item_path.endswith(".pth"):
            # 直接是.pth文件，移动
            dst_file = os.path.join(new_trained_model_dir, item)
            shutil.move(item_path, dst_file)
            print(f"已移动文件: {item_path} -> {dst_file}（覆盖同名文件）")
        elif os.path.isdir(item_path):
            # 是目录，遍历目录内的.pth文件
            for file in os.listdir(item_path):
                file_path = os.path.join(item_path, file)
                if os.path.isfile(file_path) and file.endswith(".pth"):
                    dst_file = os.path.join(new_trained_model_dir, file)
                    shutil.move(file_path, dst_file)
                    print(f"已移动文件: {file_path} -> {dst_file}（覆盖同名文件）")




