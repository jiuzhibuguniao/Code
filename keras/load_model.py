#导入保存的模型并预测
#Date:2019-09-16

from  keras.models import load_model
import joblib
import numpy as np

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
    OneHotEncoder_baokuang=joblib.load(filename+"OneHotEncoder_baokuang.m")
    OneHotEncoder_gangzhong=joblib.load(filename+"OneHotEncoder_gangzhong.m")
    return MinMaxScaler_X,OneHotEncoder_baokuang,OneHotEncoder_gangzhong

def DataHandle(data,filename):
    # filename='C:/Users/Rooobins/Desktop/VD/'
    MinMaxScaler_X, OneHotEncoder_baokuang, OneHotEncoder_gangzhong=LoadModel(filename)
    data_baokuang=data[:,0]
    data_gangzhong=data[:,4]
    data_baokuang_Encoder=OneHotEncoder_baokuang.transform(data_baokuang.reshape(data_baokuang.shape[0],1))
    data_gangzhong_Encoder=OneHotEncoder_gangzhong.transform(data_gangzhong.reshape(data_gangzhong.shape[0],1))
    data = np.delete(data, 0, axis=1)
    data = np.delete(data, 3, axis=1)
    data = np.c_[data, data_baokuang_Encoder]
    data = np.c_[data, data_gangzhong_Encoder]
    data = MinMaxScaler_X.transform(data)
    return data


def predict(filename,data):
    model=load_model(filename+"my_model.h5")
    MinMaxScaler_y=joblib.load(filename+"MinMaxScaler_y.m")
    predict_y=model.predict(data)
    predict_y=MinMaxScaler_y.inverse_transform(predict_y)
    return predict_y

def main():
    while True:
        properties = ["包况", "包龄", "精炼冶时", "总送电时间", "钢种", "进工位温度", "破空温度", "出工位温度", "氩气流量"]
        data=InputData(properties)
        filename='C:/Users/Rooobins/Desktop/VD/'
        try:
            _data=DataHandle(data,filename)
            predict_y = predict(filename, _data)
            # print(predict_y)
            print('预测温度：',"-->",np.round(predict_y[0][0]))
        except:
            print("包况或者钢种第一次遇到！")
            continue


if __name__=="__main__":
    main()