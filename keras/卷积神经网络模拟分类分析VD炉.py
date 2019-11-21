                                         #  卷积神经网络模拟分类分析VD炉  #

from keras.layers import Dense,Conv2D,Activation,Dropout,MaxPooling2D,BatchNormalization,Flatten
from sklearn.preprocessing import MinMaxScaler,OneHotEncoder
import numpy as np
import pandas as pd
np.random.seed(10)
from keras.models import Sequential
from sklearn.model_selection import train_test_split

OrialData=pd.read_csv("../data/vd.csv",encoding='gb18030')

OrialData=OrialData.dropna()                                                            #删除带有空格的行和列

OrialData=OrialData[(OrialData['抽真空时间']>10) & (OrialData['抽真空时间']<65)]          #抽取 10 < 抽真空时间 < 60

OrialData=OrialData[OrialData['温差']>=0]                                                #抽取 温差 > 0
OrialData=OrialData.drop('温差',axis=1)                                                  #删除温差此列
OrialData=OrialData[OrialData['包龄']!=441]                                              #包龄删除441
OrialData=OrialData[(OrialData['精炼冶时']>50) & (OrialData['精炼冶时']<200)]             #精炼冶时 范围集中在 50~200之间
OrialData = OrialData[(OrialData['总送电时间'] > 20) & (OrialData['总送电时间'] < 60)]    #总送电时间 范围在  20~60之间
OrialData=OrialData[(OrialData['氩气流量'] > 200) & (OrialData['氩气流量'] < 700)]        #氩气流量 范围在  200~700
OrialData=OrialData.drop('炉号',axis=1)                                                  #删除炉号
Steel=['524071', '562024', '573011', '573125', '532187', '582271', '241530']             #删除这几个钢种
for row in Steel:
    OrialData=OrialData[OrialData['钢种']!=row]
                                        #  数据属性  #
# 包况
# 包龄
# 精炼冶时
# 总送电时间
# 钢种
# 进工位温度
# 破空温度
# 出工位温度
# 氩气流量

# 抽真空时间

Data=OrialData.values                                                                                                   #抽取数值

                                                        #  数据处理  #
X=Data[:,:-1]
y=Data[:,-1]

enc=OneHotEncoder(sparse=False)
train_X,test_X,train_y,test_y=train_test_split(X,y,test_size=0.2,shuffle=False)

train_X_baokuang=train_X[:,0]
train_X_gangzhong=train_X[:,4]
test_X_baokuang=test_X[:,0]
test_X_gangzhong=test_X[:,4]

train_X_baokuang_fit=enc.fit(train_X_baokuang.reshape(train_X_baokuang.shape[0],1))
train_X_baokuang_fit_transform=train_X_baokuang_fit.transform(train_X_baokuang.reshape(train_X_baokuang.shape[0],1))             #训练集-->包况One-Hot模型
train_X_gangzhong_fit=enc.fit(train_X_gangzhong.reshape(train_X_gangzhong.shape[0],1))
train_X_gangzhong_fit_transform=train_X_gangzhong_fit.transform(train_X_gangzhong.reshape(train_X_gangzhong.shape[0],1))         #训练集-->钢种One-Hot模型

train_X_baokuang_fit=enc.fit(train_X_baokuang.reshape(train_X_baokuang.shape[0],1))
test_X_baokuang_transform=train_X_baokuang_fit.transform(test_X_baokuang.reshape(test_X_baokuang.shape[0],1))                    #测试集-->包况One-Hot模型
train_X_gangzhong_fit=enc.fit(train_X_gangzhong.reshape(train_X_gangzhong.shape[0],1))
test_X_gangzhong_transform=train_X_gangzhong_fit.transform(test_X_gangzhong.reshape(test_X_gangzhong.shape[0],1))                #测试集-->钢种One-Hot模型

y_=np.linspace(1,100,100)
train_y_fit=enc.fit(y_.reshape(y_.shape[0],1))
train_y_fit_tranform=train_y_fit.transform(train_y.reshape(train_y.shape[0],1))                                                  #训练集-->y值One-Hot模型   train_y_fit_transform
train_y_fit=enc.fit(y_.reshape(y_.shape[0],1))
test_y_transform=train_y_fit.transform(test_y.reshape(test_y.shape[0],1))                                                        #测试集-->y值One-Hot模型   test_y_transform

train_X=np.delete(train_X,0,axis=1)                                       #X训练集删除第0列
train_X=np.delete(train_X,3,axis=1)                                       #X训练集删除第3列
train_X=np.c_[train_X,train_X_baokuang_fit_transform]                     #X训练集添加包况One-Hot模型
train_X=np.c_[train_X,train_X_gangzhong_fit_transform]                    #X训练集添加钢种One-Hot模型

test_X=np.delete(test_X,0,axis=1)                                         #X测试集删除第0列
test_X=np.delete(test_X,3,axis=1)                                         #X测试集删除第3列
test_X=np.c_[test_X,test_X_baokuang_transform]                            #X测试集添加包况One-Hot模型
test_X=np.c_[test_X,test_X_gangzhong_transform]                           #X测试集添加钢种One-Hot模型

train_X=np.c_[train_X,np.zeros([train_X.shape[0],1])]
test_X=np.c_[test_X,np.zeros([test_X.shape[0],1])]

scaler=MinMaxScaler(feature_range=(0,1))
train_X_scaler_fit=scaler.fit(train_X)
train_X_scaler_fit_transform=train_X_scaler_fit.transform(train_X)        #X训练集归一化-->train_X_scaler_fit_transform
train_X_scaler_fit=scaler.fit(train_X)
test_X_scaler_transform=train_X_scaler_fit.transform(test_X)              #X测试集归一化-->test_X_scaler_transform

train_X_reshape=train_X.reshape(train_X_scaler_fit_transform.shape[0],10,14,1)                   #X训练集-->train_X_reshape
test_X_reshape=test_X.reshape(test_X_scaler_transform.shape[0],10,14,1)                          #X测试集-->test_X_reshape

                                                #  创建模型  #
#第一次卷积
model=Sequential()
model.add(Conv2D(filters=96,input_shape=(10,14,1),kernel_size=(3,3),strides=(2,2),padding='same'))
model.add(Activation('relu'))
#第一次池化
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2),padding='same'))
model.add(BatchNormalization())

#第二次卷积
model.add(Conv2D(filters=256,kernel_size=(2,2),strides=(1,1),padding='same'))
model.add(Activation('relu'))
#第二次池化
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2),padding='same'))
model.add(BatchNormalization())

#第三次卷积
model.add(Conv2D(filters=384,kernel_size=(2,2),strides=(1,1),padding='same'))
model.add(Activation('relu'))
model.add(BatchNormalization())

#第四次卷积
model.add(Conv2D(filters=384,kernel_size=(3,3),strides=(1,1),padding='same'))
model.add(Activation('relu'))
model.add(BatchNormalization())

#第五次卷积
model.add(Conv2D(filters=256,kernel_size=(3,3),strides=(1,1),padding='same'))
model.add(Activation('relu'))
#第五次池化
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2),padding='same'))
model.add(BatchNormalization())

#全连接层6
model.add(Flatten())
model.add(Dense(4096,input_shape=(224*224*3,)))
model.add(Activation('relu'))
model.add(Dropout(0.4))
model.add(BatchNormalization())

#全连接层7
model.add(Dense(4096))
model.add(Activation('relu'))
model.add(Dropout(0.4))
model.add(BatchNormalization())

#全连接层8
model.add(Dense(1000))
model.add(Activation('relu'))
model.add(Dropout(0.4))
model.add(BatchNormalization())


#输出层及训练
model.add(Dense(100))
model.add(Activation('softmax'))

#编译
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

#训练
# model.fit(train_X_reshape,train_y_fit_tranform,batch_size=10,epochs=10,verbose=2,shuffle=False,validation_data=(test_X_reshape,test_y_transform))
#
#
# #预测
# predict_y=model.predict(test_X_reshape)
# 
# predict_y=train_y_fit.inverse_transform(predict_y)       #预测值
# _y=train_y_fit.inverse_transform(test_y_transform)       #真实值
#
# predict_y=np.round(np.array(predict_y))
# acc_0=np.mean(abs(_y-predict_y)<=0)
# acc_1=np.mean(abs(_y-predict_y)<=1)
#
# print(acc_0,acc_1)