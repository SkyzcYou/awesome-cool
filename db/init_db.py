import sqlite3,os

# 链接人员信息数据库创建 name,file_name
path2 = os.path.abspath('..')
connect = sqlite3.connect(path2+'db/awesome.db')
cursor = connect.cursor()
# create table
sql = "CREATE TABLE IF NOT EXISTS people_info(name TEXT NOT NULL,file_name TEXT NOT NULL )"
cursor.execute(sql)
connect.commit()
name = "youdddddddd"
file_name = "dddddddddddddddddd"
inset = "INSERT INTO people_info values('{}','{}')".format(name,file_name)
cursor.execute(inset)
connect.commit()

cursor.execute("select * from people_info")
print(cursor.fetchall())

# 关闭游标
cursor.close()
# 断开数据库连接
cursor.close()