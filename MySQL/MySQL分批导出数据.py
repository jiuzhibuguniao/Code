from remoteCtrl import remoteCtrl
import os
import time
import MySQLdb

# myHandler=remoteCtrl()
# ret,ret_info=myHandler.transfer('39.108.100.28','w*K19910909','/root/datazhen/','E:\\datazhen','pull')
# print(ret,ret_info)

mydb=MySQLdb.connect(host='39.108.100.28',user='Rooobins',password='19910909kai',database='crawlerData')
cursor=mydb.cursor()
cursor.execute("show tables;")
ls_table=cursor.fetchall()

# remoteCtrl.command('39.108.100.28','w*K19910909','ls')

for i in range(30,len(ls_table)):
    if (i+1)%30==0:
        time.sleep(60*60*24)
    else:
        cursor.execute("select * from %s into outfile '/root/datazhen/%s.txt'" % (ls_table[i][0], ls_table[i][0]))