#导入保存的模型并预测
#Date:2019-09-16

from  keras.models import load_model
import joblib
import numpy as np
import cx_Oracle

def InputData(args):
    data=[]
    for key in args:
        _data=input("请输入:%s"%key)
        data.append(eval(_data))
    data=np.array(data,dtype='object')
    data=data.reshape(1,data.shape[0])
    return data

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
    # filename='C:/Users/Administrator/Desktop/VD/'
    MinMaxScaler_X, OneHotEncoder_baokuang, OneHotEncoder_gangzhong,LabelEncoder_baokuang,LabelEncoder_gangzhong\
        =LoadModel(filename)
    data_baokuang=data[:,0]
    data_gangzhong=data[:,4]
    data_baokuang_Label=LabelEncoder_baokuang.transform(data_baokuang)
    data_baokuang_Encoder=OneHotEncoder_baokuang.transform(data_baokuang_Label.reshape(data_baokuang_Label.shape[0],1))
    data_gangzhong_Label=LabelEncoder_gangzhong.transform(data_gangzhong)
    data_gangzhong_Encoder=OneHotEncoder_gangzhong.transform(data_gangzhong_Label.reshape(data_gangzhong_Label.shape[0],1))
    data = np.delete(data, 0, axis=1)
    data = np.delete(data, 3, axis=1)
    data = np.c_[data, data_baokuang_Encoder]
    data = np.c_[data, data_gangzhong_Encoder]
    data = MinMaxScaler_X.transform(data)
    return data

def ProgramError(LH,TIME_YC=-1):
    conn = cx_Oracle.connect("YGSJPT/YGSJPT@10.4.133.222/YGSJPT")    #此处需修改
    cursor=conn.cursor()
    sql="insert into TB_DSJ_SJ(LH,TIME_YC) values('%s',%d)"%(LH,TIME_YC)
    cursor.execute(sql)
    conn.commit()
    cursor.close()
    conn.close()

def DataError(LH,TIME_YC=-9):
    conn = cx_Oracle.connect("YGSJPT/YGSJPT@10.4.133.222/YGSJPT")    #此处需修改
    cursor=conn.cursor()
    sql="insert into TB_DSJ_SJ(LH,TIME_YC) values('%s',%d)"%(LH,TIME_YC)
    cursor.execute(sql)
    conn.commit()
    cursor.close()
    conn.close()

def AllTrue(LH,TIME_YC,FLAG=1):
    conn = cx_Oracle.connect("YGSJPT/YGSJPT@10.4.133.222/YGSJPT")    #此处需修改
    cursor=conn.cursor()
    sql="insert into TB_DSJ_SJ(LH,TIME_YC,FLAG) values('%s',%d,%d)"%(LH,TIME_YC,FLAG)
    cursor.execute(sql)
    conn.commit()
    cursor.close()
    conn.close()


def predict(filename,data):
    model=load_model(filename+"my_model.h5")
    MinMaxScaler_y=joblib.load(filename+"MinMaxScaler_y.m")
    predict_y=model.predict(data)
    predict_y=MinMaxScaler_y.inverse_transform(predict_y)
    predict_y=np.round(predict_y[0][0])
    return predict_y

def main():
    while True:
        properties = ["炉号","包况", "包龄", "精炼冶时", "总送电时间", "钢种", "进工位温度", "破空温度", "出工位温度", "氩气流量"]
        data=InputData(properties)
        LH=data[0][0]
        data=np.delete(data,0,axis=1)
        filename='C:/Users/Administrator/Desktop/VD/'
        try:
            _data=DataHandle(data,filename)
        except ValueError:       #数据缺失
            DataError(LH)
            continue
        except:                  #其他错误：程序错误
            ProgramError(LH)
            continue
        else:
            predict_y = predict(filename, _data)
            AllTrue(LH,predict_y)
            # print('预测温度：', "-->", predict_y)


if __name__=="__main__":
    main()