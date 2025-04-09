import os


def generate_tree(path, prefix=''):
    items = os.listdir(path)
    for index, item in enumerate(items):
        if item.startswith('.') or item == '__pycache__':
            continue
        full_path = os.path.join(path, item)
        is_last = index == len(items) - 1
        if os.path.isdir(full_path):
            print(prefix + ('└── ' if is_last else '├── ') + item + '/')
            new_prefix = prefix + ('    ' if is_last else '│   ')
            generate_tree(full_path, new_prefix)
        else:
            print(prefix + ('└── ' if is_last else '├── ') + item)


if __name__ == "__main__":
    project_path = '.'  # 当前目录，可替换为你的项目路径
    generate_tree(project_path)
    