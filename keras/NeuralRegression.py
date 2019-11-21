#全连通神经网络模拟回归分析VD炉
#Date:2019-9-16

import matplotlib.pyplot as plt
from math import sqrt
import pandas as pd
import numpy as np
from numpy import concatenate
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers.core import Dense,Dropout,Activation
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import OneHotEncoder
import joblib
import os
import cx_Oracle
import time

# Oracle中筛选出数据
def SelectOracle(sql):
    conn = cx_Oracle.connect("YGSJPT/YGSJPT@10.4.133.222/YGSJPT")    #此处需修改
    # sql = "select * from VD"
    cursor = conn.cursor()
    cursor.execute(sql)
    OracleData = cursor.fetchall()
    columns = ["炉号", "包况", "包龄", "精炼冶时", "总送电时间", "钢种", "进工位温度",
               "破空温度", "出工位温度", "氩气流量", "抽真空时间"]
    OrialData = pd.DataFrame(OracleData, columns=columns)
    cursor.close()
    conn.close()
    return OrialData


#数据处理
def DataProcessing(OrialData):
    OrialData = OrialData.dropna()  # 删除带有空格的行和列
    OrialData['抽真空时间'] = OrialData['抽真空时间'].astype('float')
    OrialData = OrialData[(OrialData['抽真空时间'] > 10) & (OrialData['抽真空时间'] < 65)]  # 抽取 10 < 抽真空时间 < 60
    OrialData = OrialData[OrialData['进工位温度']-OrialData['破空温度']>=0]  # 抽取 温差(进工位温度-破空温度) > 0
    # OrialData = OrialData.drop('温差', axis=1)  # 删除温差此列
    OrialData['包龄'] = OrialData['包龄'].astype('float')
    OrialData = OrialData[OrialData['包龄'] != 441]  # 包龄删除441
    OrialData = OrialData[(OrialData['精炼冶时'] > 50) & (OrialData['精炼冶时'] < 200)]  # 精炼冶时 范围集中在 50~200之间
    OrialData = OrialData[(OrialData['总送电时间'] > 20) & (OrialData['总送电时间'] < 60)]  # 总送电时间 范围在  20~60之间
    OrialData = OrialData[(OrialData['氩气流量'] > 200) & (OrialData['氩气流量'] < 700)]  # 氩气流量 范围在  200~700
    OrialData = OrialData.drop('炉号', axis=1)  # 删除炉号
    # Steel = ['524071', '562024', '573011', '573125', '532187', '582271', '241530']  # 删除这几个钢种
    # for row in Steel:
    #     OrialData = OrialData[OrialData['钢种'] != row]
    # 包况 包龄 精炼冶时 总送电时间 钢种 进工位温度 破空温度 出工位温度 氩气流量       抽真空时间
    Data = OrialData.values
    return Data

#文本数据处理
def OneHotEncoderData(Data):
    X = Data[:, :-1]
    y = Data[:, -1]
    enc = OneHotEncoder(sparse=False)
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, shuffle=False)
    train_y = train_y.reshape(train_y.shape[0], 1)
    test_y = test_y.reshape(test_y.shape[0], 1)

    train_X_baokuang = train_X[:, 0]
    train_X_gangzhong = train_X[:, 4]
    test_X_baokuang = test_X[:, 0]
    test_X_gangzhong = test_X[:, 4]

    train_X_baokuang_fit = enc.fit(train_X_baokuang.reshape(train_X_baokuang.shape[0], 1))
    joblib.dump(train_X_baokuang_fit,"C:/Users/Rooobins/Desktop/VD/OneHotEncoder_baokuang.m")
    train_X_baokuang_fit_transform = train_X_baokuang_fit.transform(
        train_X_baokuang.reshape(train_X_baokuang.shape[0], 1))  # 训练集-->包况One-Hot模型
    train_X_gangzhong_fit = enc.fit(train_X_gangzhong.reshape(train_X_gangzhong.shape[0], 1))
    joblib.dump(train_X_gangzhong_fit,"C:/Users/Rooobins/Desktop/VD/OneHotEncoder_gangzhong.m")
    train_X_gangzhong_fit_transform = train_X_gangzhong_fit.transform(
        train_X_gangzhong.reshape(train_X_gangzhong.shape[0], 1))  # 训练集-->钢种One-Hot模型

    train_X_baokuang_fit = enc.fit(train_X_baokuang.reshape(train_X_baokuang.shape[0], 1))
    test_X_baokuang_transform = train_X_baokuang_fit.transform(
        test_X_baokuang.reshape(test_X_baokuang.shape[0], 1))  # 测试集-->包况One-Hot模型
    train_X_gangzhong_fit = enc.fit(train_X_gangzhong.reshape(train_X_gangzhong.shape[0], 1))
    test_X_gangzhong_transform = train_X_gangzhong_fit.transform(
        test_X_gangzhong.reshape(test_X_gangzhong.shape[0], 1))  # 测试集-->钢种One-Hot模型

    train_X = np.delete(train_X, 0, axis=1)  # X训练集删除第0列
    train_X = np.delete(train_X, 3, axis=1)  # X训练集删除第3列
    train_X = np.c_[train_X, train_X_baokuang_fit_transform]  # X训练集添加包况One-Hot模型
    train_X = np.c_[train_X, train_X_gangzhong_fit_transform]  # X训练集添加钢种One-Hot模型

    test_X = np.delete(test_X, 0, axis=1)  # X测试集删除第0列
    test_X = np.delete(test_X, 3, axis=1)  # X测试集删除第3列
    test_X = np.c_[test_X, test_X_baokuang_transform]  # X测试集添加包况One-Hot模型
    test_X = np.c_[test_X, test_X_gangzhong_transform]  # X测试集添加钢种One-Hot模型
    return train_X,test_X,train_y,test_y

#标准化
def MinMaxScalerData(train_X,test_X,train_y,test_y):
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_X_scaler_fit = scaler.fit(train_X)
    joblib.dump(train_X_scaler_fit,"C:/Users/Rooobins/Desktop/VD/MinMaxScaler_X.m")
    train_X_scaler_fit_transform = train_X_scaler_fit.transform(train_X)  # X训练集-->标准化   train_X_scaler_fit_transform
    train_X_scaler_fit = scaler.fit(train_X)
    test_X_scaler_transform = train_X_scaler_fit.transform(test_X)  # X测试集-->标准化   test_X_scaler_transform

    train_y_scaler_fit = scaler.fit(train_y)
    joblib.dump(train_y_scaler_fit,"C:/Users/Rooobins/Desktop/VD/MinMaxScaler_y.m")
    train_y_scaler_fit_transform = train_y_scaler_fit.transform(train_y)  # y训练集-->标准化   train_y_scaler_fit_transform
    train_y_scaler_fit = scaler.fit(train_y)
    test_y_scaler_transform = train_y_scaler_fit.transform(test_y)  # y测试集-->标准化   test_y_scaler_transform
    return train_X_scaler_fit_transform,test_X_scaler_transform,train_y_scaler_fit_transform,test_y_scaler_transform

#建立模型
def BuildModel(train_X_scaler_fit_transform):
    # 全连通神经网络
    model = Sequential()
    input = train_X_scaler_fit_transform.shape[1]
    # 隐藏层1000
    model.add(Dense(1000, input_dim=input))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    # 隐藏层1000
    model.add(Dense(1000))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    # 无激活函数
    model.add(Dense(1))
    # 编译
    model.compile(loss='mean_squared_error', optimizer=Adam())
    return model


def main():
    sql="select a.炉号,a.包况,a.包龄,a.精炼冶时,总送电时间,钢种,b.arrive_gw_temp 进工位温度," \
        "b.vacuum_break_temp 破空温度,b.ladle_depart_temp 出工位温度,b.fine_vacuum_ar_flow 氩气流量," \
        "b.vacuum_pump_duration 抽真空时间 from v_lf_sj a ,TB_VD_PRODATA b " \
        "where a.炉号 = b.heatno order by a.炉号"   #此处需修改
    OrialData=SelectOracle(sql)
    Data=DataProcessing(OrialData)
    train_X,test_X,train_y,test_y=OneHotEncoderData(Data)
    train_X_scaler_fit_transform, test_X_scaler_transform, train_y_scaler_fit_transform, test_y_scaler_transform\
        =MinMaxScalerData(train_X,test_X,train_y,test_y)
    model=BuildModel(train_X_scaler_fit_transform)

    # 早停法
    early_stopping = EarlyStopping(monitor='val_loss', patience=50, verbose=2)

    model.fit(train_X_scaler_fit_transform, train_y_scaler_fit_transform, epochs=300, batch_size=10,
                        validation_data=(test_X_scaler_transform, test_y_scaler_transform), verbose=2,
                        shuffle=False, callbacks=[early_stopping])
    model.save("C:/Users/Rooobins/Desktop/VD/my_model.h5")
    del model
    time.sleep(7200)

if __name__=="__main__":
    while True:
        main()