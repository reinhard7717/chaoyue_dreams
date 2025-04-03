import mysql.connector
from mysql.connector import Error

def reset_database():
    try:
        # 连接到MySQL服务器
        connection = mysql.connector.connect(
            host='39.101.65.133',
            user='stocker',
            password='Asdf+1234'
        )
        
        if connection.is_connected():
            cursor = connection.cursor()
            
            # 删除数据库（如果存在）
            cursor.execute("DROP DATABASE IF EXISTS beyond_dreams")
            print("数据库 beyond_dreams 已删除")
            
            # 创建新数据库
            cursor.execute("CREATE DATABASE beyond_dreams CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
            print("数据库 beyond_dreams 已创建")
            
            cursor.close()
            connection.close()
            print("数据库连接已关闭")
            
    except Error as e:
        print(f"发生错误: {e}")

if __name__ == "__main__":
    reset_database() 