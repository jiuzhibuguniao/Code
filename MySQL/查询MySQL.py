import MySQLdb

mydb=MySQLdb.connect(host='39.108.100.28',user='Rooobins',password='19910909kai',database='crawlerData')
cursor=mydb.cursor()
cursor.execute("show tables;")
ls_table=cursor.fetchall()
for row in ls_table:
    cursor.execute('select count(*) from %s;'%(row[0]))
    No=cursor.fetchall()
    # cursor.execute("select * from {} limit 1;".format(row[0]))
    # info=cursor.fetchall()
    print("[TABLE]",row[0],'==>>',No[0][0])
    # print(info)
print(len(ls_table),"table ==>>",'is crawlered.')