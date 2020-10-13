import sqlite3

# 链接人员信息数据库创建 name,file_name
connect = sqlite3.connect('awesome.db')
cursor = connect.cursor()
# create table
sql = "CREATE TABLE IF NOT EXISTS people_info(id INTEGER PRIMARY KEY,name TEXT,file_name TEXT)"
cursor.execute(sql)