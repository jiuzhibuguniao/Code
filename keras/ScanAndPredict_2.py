#扫描TB_DSJ_SJ,匹配对应的炉号值并预测
#Date:2019-10-15

from keras.models import load_model
import joblib
import datetime
import numpy as np
import cx_Oracle
import pandas as pd
import time

def ScanTable(table):
    MARK=1
    conn = cx_Oracle.connect("YGSJPT/YGSJPT@10.4.133.222/YGSJPT")    #此处需修改
    cursor=conn.cursor()
    sql='select * from %s'%(table)
    cursor.execute(sql)
    data=cursor.fetchall()
    cursor.close()
    conn.close()
    _data=[]
    for row in data:
        if row[2]==0:
            _data.append(row)
    tuple_data=_data.copy()
    if len(_data)==0:
        MARK=0
        pass
    else:
        _data = pd.DataFrame(_data)
        _data = _data.drop([1, 2,3,4, 5], axis=1)
        _data[8] = _data[8].astype('float')
        _data = _data.values
    return _data,MARK,tuple_data

def LoadModel(filename):
    # model=load_model(filename+"my_model.h5")
    MinMaxScaler_X=joblib.load(filename+"MinMaxScaler_X.m")
    # MinMaxScaler_y=joblib.load(filename+"MinMaxScaler_y.m")
    LabelEncoder_baokuang=joblib.load(filename+"LabelEncoder_baokuang.m")
    LabelEncoder_gangzhong=joblib.load(filename+"LabelEncoder_gangzhong.m")
    OneHotEncoder_baokuang=joblib.load(filename+"OneHotEncoder_baokuang.m")
    OneHotEncoder_gangzhong=joblib.load(filename+"OneHotEncoder_gangzhong.m")
    return MinMaxScaler_X,OneHotEncoder_baokuang,OneHotEncoder_gangzhong,LabelEncoder_baokuang,LabelEncoder_gangzhong

def DataHandle(data,filename):
    # filename='C:/Users/Rooobins/Desktop/VD/'
    MinMaxScaler_X, OneHotEncoder_baokuang, OneHotEncoder_gangzhong,LabelEncoder_baokuang,LabelEncoder_gangzhong\
        =LoadModel(filename)
    data_baokuang=data[:,1]
    data_gangzhong=data[:,0]
    data_baokuang_Label=LabelEncoder_baokuang.transform(data_baokuang)
    data_baokuang_Encoder=OneHotEncoder_baokuang.transform(data_baokuang_Label.reshape(data_baokuang_Label.shape[0],1))
    data_gangzhong_Label=LabelEncoder_gangzhong.transform(data_gangzhong)
    data_gangzhong_Encoder=OneHotEncoder_gangzhong.transform(data_gangzhong_Label.reshape(data_gangzhong_Label.shape[0],1))
    data = np.delete(data, 0, axis=1)
    data = np.delete(data, 0, axis=1)
    data = np.c_[data, data_baokuang_Encoder]
    data = np.c_[data, data_gangzhong_Encoder]
    data = MinMaxScaler_X.transform(data)
    return data

def predict(filename,data):
    YC_START=datetime.datetime.now()
    model=load_model(filename+"my_model.h5")
    MinMaxScaler_y=joblib.load(filename+"MinMaxScaler_y.m")
    predict_y=model.predict(data)
    predict_y=MinMaxScaler_y.inverse_transform(predict_y)
    predict_y=int(np.round(predict_y[0][0]))
    YC_END=datetime.datetime.now()
    return predict_y,YC_START,YC_END

def ProgramError(IN_START,TIME_YC=-1):
    conn = cx_Oracle.connect("YGSJPT/YGSJPT@10.4.133.222/YGSJPT")    #此处需修改
    cursor=conn.cursor()
    param={"TIME_YC":TIME_YC,"IN_START":IN_START}
    sql="update TB_DSJ_SJ set TIME_YC=:TIME_YC where IN_START=:IN_START"
    cursor.execute(sql,param)
    conn.commit()
    cursor.close()
    conn.close()

def DataError(IN_START,TIME_YC=-9):
    conn = cx_Oracle.connect("YGSJPT/YGSJPT@10.4.133.222/YGSJPT")    #此处需修改
    cursor=conn.cursor()
    param={"TIME_YC":TIME_YC,"IN_START":IN_START}
    sql="update TB_DSJ_SJ set TIME_YC=:TIME_YC where IN_START=:IN_START"
    cursor.execute(sql,param)
    conn.commit()
    cursor.close()
    conn.close()

def AllTrue(IN_START,TIME_YC,YC_START,YC_END,FLAG=1):
    conn = cx_Oracle.connect("YGSJPT/YGSJPT@10.4.133.222/YGSJPT")    #此处需修改
    cursor=conn.cursor()
    param={"TIME_YC":TIME_YC,"FLAG":FLAG,"YC_START":YC_START,"YC_END":YC_END,"IN_START":IN_START}
    sql_update = "update TB_DSJ_SJ set TIME_YC=:TIME_YC,FLAG=:FLAG,YC_START=:YC_START,YC_END=:YC_END where IN_START=:IN_START"
    cursor.execute(sql_update,param)
    conn.commit()
    cursor.close()
    conn.close()

def main():
    while True:
        data_TB_DSJ_SJ,MARK,tuple_TB_DSJ_SJ=ScanTable(table="TB_DSJ_SJ")
        if MARK==1:
            for row in data_TB_DSJ_SJ:
                i=0
                row=row.reshape(1,row.shape[0])
                IN_START=tuple_TB_DSJ_SJ[i][3]
                data_Orial=np.delete(row,0,axis=1)
                filename = 'C:/Users/Rooobins/Desktop/VD/VD/'
                #filename = 'C:/Users/Administrator/Desktop/VD/'
                try:
                    _data = DataHandle(data_Orial, filename)
                except ValueError:  # 数据缺失
                    DataError(IN_START)
                    continue
                except:  # 其他错误：程序错误
                    ProgramError(IN_START)
                    continue
                else:
                    predict_y, YC_START, YC_END = predict(filename, _data)
                    AllTrue(IN_START, predict_y, YC_START, YC_END)
                i+=1
        time.sleep(5)

if __name__=="__main__":
    main()