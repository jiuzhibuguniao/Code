import paramiko
import os
import MySQLdb
import time
from remoteCtrl import remoteCtrl


"""
MySQL传输数据至本地
Auther:@Rooobins
Date:2019-07-15 22:54
"""


#连接MySQL
def mydbList(host,user,password,database):
    mydb=MySQLdb.connect(host,user,password,database)
    cursor=mydb.cursor()
    cursor.execute('show tables;')
    data=cursor.fetchall()
    cursor.close()
    mydb.close()
    return data

#Python操作MySQL
def operaMySQL(host,user,password,database,cmd):
    mydb=MySQLdb.connect(host,user,password,database)
    cursor=mydb.cursor()
    cursor.execute(cmd)
    cursor.close()
    mydb.close()

#Python操作Linux
def operatLinux(ip,port,user,passwd,timeout,cmd):
    client=paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(hostname=ip,username=user,port=port,password=passwd,timeout=timeout,allow_agent=False,look_for_keys=False)
    client.exec_command(cmd)
    return

def main():
    filename=time.strftime("%b_%d_%a_%H_%M",time.localtime())
    operatLinux('39.108.100.28', 22, 'root', 'w*K19910909', 60, 'mkdir /root/%s' % (filename))  # 创建文件夹
    operatLinux('39.108.100.28', 22, 'root', 'w*K19910909', 60, 'chmod 777 %s' % (filename))  # 更改文件权限
    # mydb_list=mydbList('39.108.100.28','Rooobins','19910909kai','crawlerData')
    with open("C:/Users/Rooobins/Desktop/listname.txt",'r') as f:
        filelist=f.readlines()
        filelist=[file.strip("\n") for file in filelist]
        f.close()
    i=0
    while i < len(filelist):
        if (i+1)%60==0 or i==len(filelist)-1:
            #拉取Linux文件至本地
            myHandler = remoteCtrl()
            ret, ret_info = myHandler.transfer('39.108.100.28', 'w*K19910909', '/root/%s/'%(filename), 'E:\\datazhen','pull')

            #删除文件夹
            operatLinux('39.108.100.28', 22, 'root', 'w*K19910909', 60, 'rm -rf /root/%s' % (filename))

            #创建文件夹
            filename = time.strftime("%b_%d_%a_%H_%M",time.localtime())
            operatLinux('39.108.100.28', 22, 'root', 'w*K19910909', 60, 'mkdir /root/%s' % (filename))  # 创建文件夹
            operatLinux('39.108.100.28', 22, 'root', 'w*K19910909', 60, 'chmod 777 %s' % (filename))  # 更改文件权限
        else:
            operaMySQL('39.108.100.28','Rooobins','19910909kai','crawlerData',"select * from %s into outfile '/root/%s/%s.txt';"%(filelist[i],filename,filelist[i]))    #mysql导出数据至指定文件
        i+=1

    return

if __name__=="__main__":
    main()
