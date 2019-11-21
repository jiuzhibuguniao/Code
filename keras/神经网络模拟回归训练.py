#_*_coding:utf-8_*_

# from keras.models import Sequential,load_model
# from keras.layers import Dense,Activation
# from keras.optimizers import SGD
# from keras.layers.advanced_activations import LeakyReLU
# from sklearn import preprocessing
#
# from sklearn.preprocessing import OneHotEncoder
# from sklearn.model_selection import train_test_split
# import pandas as pd
# import numpy as np
# from keras.wrappers.scikit_learn import KerasRegressor
# from sklearn.model_selection import cross_val_score,KFold
# from sklearn.preprocessing import StandardScaler
# from sklearn.pipeline import Pipeline
# from sklearn.metrics import accuracy_score




# OrialData=pd.read_csv("./data/vd.csv",encoding='gb18030')
## OrialData=pd.read_csv("./data/vd.csv",encoding='gb18030',delim_whitespace=True,header=None)

################################################   考虑:炉号 包号 包况  钢种   ###########################################

# DropInfo=['测温时间','抽真空起时间','破空时间','温时比']
#
# for Info in DropInfo:
#     OrialData=OrialData.drop(Info,axis=1)
#
# #删除带有空格的行和列
# OrialData=OrialData.dropna()
#
# #OneHotEncoder计算
# #炉号---第0列
# #包号---第1列
# #包况---第2列
# #钢种---第9列
# Data=OrialData.values
# for i in range(Data.shape[0]):
#     Data[i][-1]=eval(Data[i][-1])
# OrialDataL_1=np.array([str(x) for x in OrialData.values[:,1]])
#
# #包号浮点数转化为str
# Data[:,1]=OrialDataL_1
#
# enc=OneHotEncoder(sparse=False)
# DataL_0=enc.fit_transform(Data[:,0].reshape(Data[:,0].shape[0],1))
# DataL_1=enc.fit_transform(Data[:,1].reshape(Data[:,1].shape[0],1))
# DataL_2=enc.fit_transform(Data[:,2].reshape(Data[:,2].shape[0],1))
# DataL_9=enc.fit_transform(Data[:,9].reshape(Data[:,9].shape[0],1))
#
# for i in range(Data.shape[0]):
#     Data[i,0]=DataL_0[i]
#     Data[i,1]=DataL_1[i]
#     Data[i,2]=DataL_2[i]
#     Data[i,9]=DataL_9[i]
#
# #划分训练集和测试集
# Data_y=Data[:,7]
# Data_X=np.delete(Data,7,axis=1)
# Data_X=np.delete(Data_X,0,axis=1)
#
# train_X, test_X, train_y, test_y = train_test_split(Data_X, Data_y, train_size=0.7, random_state=0)
#
# #建立模型
# model=Sequential()
# model.add(Dense(9,input_dim=9,init='normal',activation='relu'))
# model.add(Dense(8,init='normal',activation='relu'))
# model.add(Dense(1,init='normal',activation='linear'))
#
# model.compile(loss='mse',optimizer='adam')
#
# model.fit(train_X,train_y,nb_epoch=100,batch_size=10)

########################################################### 运行错误 ####################################################



################################################   不考虑:炉号 包号 包况  钢种   #########################################

# DropInfo=['测温时间','抽真空起时间','破空时间','温时比','炉号','包号','包况','钢种']
# for Info in DropInfo:
#     OrialData=OrialData.drop(Info,axis=1)
#
# #删除带有空格的行和列
# OrialData=OrialData.dropna()
#
# #抽取 10 < 抽真空时间 < 60
# OrialData=OrialData[(OrialData['抽真空时间']>10) & (OrialData['抽真空时间']<65)]
#
# #抽取 温差 > -100
# OrialData=OrialData[OrialData['抽真空时间']>-100]
#
# #DataFrame提取为矩阵
# Data=OrialData.values
# #
# #
# # # for i in range(Data.shape[0]):
# # #     Data[i][-1]=eval(Data[i][-1])
# # # #筛选数据
# # # for i in range(Data.shape[0]):
# # #     if Data[i][2]>60 or Data[i][2]<10:
# # #         Data[i][2]=0
# #
# #
# #划分训练集和测试集
# Data_y=Data[:,2]
# Data_X=np.delete(Data,2,axis=1)
# train_X, test_X, train_y, test_y = train_test_split(Data_X, Data_y, train_size=0.7, random_state=0)

# #建立模型
# model=Sequential()
# model.add(Dense(12,input_dim=6,init='normal',activation='relu'))
# model.add(Dense(8,init='normal',activation='relu'))
# # model.add(Dense(1,init='normal',activation='linear'))
# model.add(Dense(1,init='normal'))
#
#
# model.compile(loss='mse',optimizer='adam',metrics=['accuracy'])
#
# model.fit(train_X,train_y,nb_epoch=100,batch_size=5)
#
# loss,_accuracy=model.evaluate(test_X,test_y,verbose=0)
# print(loss,_accuracy)
#
# predict_X=model.predict(test_X)
# _sum=0
# static=[]
# for i in range(test_y.shape[0]):
#     static.append(abs((test_y[i]-predict_X[i])))
#     if abs((test_y[i]-predict_X[i]))<0.1:
#         _sum=_sum+1
#
# accuracy=_sum/(test_y.shape[0])
# print("accuracy : %s"%accuracy)

####################################################### 运行成功 #######################################################

################################################# KerasRegressor #######################################################

# #删除无用特征
# DropInfo=['测温时间','抽真空起时间','破空时间','温时比','炉号','包号','包况','钢种']
# for Info in DropInfo:
#     OrialData=OrialData.drop(Info,axis=1)
#
# #删除带有空格的行和列
# OrialData=OrialData.dropna()
#
# #抽取 10 < 抽真空时间 < 60
# OrialData=OrialData[(OrialData['抽真空时间']>10) & (OrialData['抽真空时间']<65)]
#
# #抽取 温差 > -100
# OrialData=OrialData[OrialData['抽真空时间']>-100]
#
# Data=OrialData.values
#
# #划分数据和标签
# Data_y=Data[:,2]
# Data_X=np.delete(Data,2,axis=1)
# train_X, test_X, train_y, test_y = train_test_split(Data_X, Data_y, train_size=0.7, random_state=0)
#
# #建立模型
# def baseline_model():
#     model=Sequential()
#     model.add(Dense(100,input_dim=6,init='normal',activation='relu'))
#     model.add(Dense(50,init='normal',activation='relu'))
#     # model.add(Dense(1,init='normal',activation='linear'))
#     model.add(Dense(1,init='normal'))
#     model.compile(loss='mse',optimizer='adam',metrics=['accuracy'])
#     return model
################################# 共享代码 #################################

################################# 使用Pipline ##############################
# seed=7
# np.random.seed(seed)
# estimators=[]
# estimators.append(('standardize',StandardScaler()))
# estimators.append(('mlp',KerasRegressor(build_fn=baseline_model,nb_epoch=100,batch_size=5,verbose=0)))
# pipeline=Pipeline(estimators)
# kfold=KFold(n_splits=10,random_state=seed)
# results=cross_val_score(pipeline,train_X,train_y,cv=kfold)
#
# estimators[1][1].fit(train_X,train_y)
#
# #预测
# predict=estimators[1][1].predict(test_X)
# prediction=[float(np.round(x)) for x in predict]
# accuracy=np.mean(prediction==test_y)
# print(accuracy)


############################ 不使用 Pipline ################################

# seed=7
# np.random.seed(seed)
# estimator=KerasRegressor(build_fn=baseline_model,nb_epoch=100,batch_size=5,verbose=0)
# # estimator.model=load_model('model_')
# kfold=KFold(n_splits=10,random_state=seed)
# results=cross_val_score(estimator,train_X,train_y,cv=kfold)
# print("Result: %.2f (%.2f) MSE"%(results.mean(),results.std()))
#
# estimator.fit(train_X,train_y)
# prediction=estimator.predict(test_X)
#
# predict_y=[float(np.round(x)) for x in prediction]
# accuracy=np.mean(predict_y==test_y)
# print(accuracy)


########################## keras 多元线性回归 ##############################









################################################## 基于Keras的神经网络回归模型 ###########################################
                                               #   不考虑:炉号 包号 包况  钢种   #
# import matplotlib.pyplot as plt
# from math import sqrt
# import pandas as pd
# import numpy as np
# from numpy import concatenate
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.metrics import mean_squared_error
# from keras.models import Sequential
# from keras.layers.core import Dense,Dropout,Activation
# from keras.optimizers import Adam
# from sklearn.model_selection import train_test_split
# from keras.callbacks import EarlyStopping
#
# #读取数据
# OrialData=pd.read_csv("./data/vd.csv",encoding='gb18030')
#
# #删除无用特征
# DropInfo=['测温时间','抽真空起时间','破空时间','温时比','炉号','包号','包况','钢种']
# for Info in DropInfo:
#     OrialData=OrialData.drop(Info,axis=1)
#
# #删除带有空格的行和列
# OrialData=OrialData.dropna()
#
# #抽取 10 < 抽真空时间 < 60
# OrialData=OrialData[(OrialData['抽真空时间']>10) & (OrialData['抽真空时间']<65)]
#
# #抽取 温差 > -100
# OrialData=OrialData[OrialData['抽真空时间']>-100]
#
# _Data=OrialData.values
# _Data_y=_Data[:,2]
# _Data_y=_Data_y.reshape(len(_Data_y),1)
# _Data_X=np.delete(_Data,2,axis=1)
# Data=np.c_[_Data_X,_Data_y]
#
# scaler=MinMaxScaler(feature_range=(0,1))
# scaled=scaler.fit_transform(Data)
#
# Data_y=scaled[:,-1]
# Data_X=np.delete(scaled,-1,axis=1)
#
# train_X,test_X,train_y,test_y=train_test_split(Data_X,Data_y,test_size=0.25)
#
# #全连通神经网络
# model=Sequential()
# input=Data_X.shape[1]
# #隐藏层128
# model.add(Dense(128,input_dim=input))
# model.add(Activation('relu'))
# model.add(Dropout(0.2))
#
# #隐藏层128
# model.add(Dense(128))
# model.add(Activation('relu'))
# model.add(Dropout(0.2))
#
# #无激活函数
# model.add(Dense(1))
# #编译
# model.compile(loss='mean_squared_error',optimizer=Adam())
# #早停法
# early_stopping=EarlyStopping(monitor='val_loss',patience=50,verbose=2)
#
# history=model.fit(train_X,train_y,epochs=300,batch_size=20,
#                   validation_data=(test_X,test_y),verbose=2,
#                   shuffle=False,callbacks=[early_stopping])
#
# #loss曲线
# plt.plot(history.history['loss'],label='train')
# plt.plot(history.history['val_loss'],label='test')
# plt.legend()
# plt.show()
#
# #预测
# predict_y=model.predict(test_X)
# #预测y 逆标准化
# inv_Ypredict0=concatenate((test_X,predict_y),axis=1)
# inv_Ypredict1=scaler.inverse_transform(inv_Ypredict0)
# inv_Ypredict=inv_Ypredict1[:,-1]
#
# #原始y 逆标准化
# test_y=test_y.reshape(len(test_y),1)
# inv_y0=concatenate((test_X,test_y),axis=1)
# inv_y1=scaler.inverse_transform(inv_y0)
# inv_y=inv_y1[:,-1]
#
# #计算RMSE
# rmse=sqrt(mean_squared_error(inv_y,inv_Ypredict))
# print("Test RMSE : %.3f"%rmse)
# plt.plot(inv_y,label="inv_y")
# plt.plot(inv_Ypredict,label='inv_Ypredict')
# plt.legend()
# plt.show()
#
# mark=np.linspace(1,len(test_X),len(test_X))
# plt.scatter(mark,inv_y,label='inv_y')
# plt.scatter(mark,inv_Ypredict,label="inv_Ypredict")
# plt.legend()
# plt.show()
#
#
#                        #准确率
# #预测值
# predict=[float(np.round(x)) for x in inv_Ypredict]
# static=[]
# for i in range(len(predict)):
#     static.append(abs(predict[i]-inv_y[i]))
#
# static=np.array(static)
#
# count=[]
# mark=np.linspace(1,20,20)
# for row in mark:
#     count.append(np.mean(static<=row))
#
# print(count)

############################################ 1分钟之内的准确率 37%~41%  #################################################



# ################################################## 基于Keras的神经网络回归模型 ###########################################
#                                                #   考虑:炉号 包号 包况  钢种   #
# import matplotlib.pyplot as plt
# from math import sqrt
# import pandas as pd
# import numpy as np
# from numpy import concatenate
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.metrics import mean_squared_error
# from keras.models import Sequential
# from keras.layers.core import Dense,Dropout,Activation
# from keras.optimizers import Adam
# from sklearn.model_selection import train_test_split
# from keras.callbacks import EarlyStopping
# from sklearn.preprocessing import OneHotEncoder
#
#
# OrialData=pd.read_csv("./data/vd.csv",encoding='gb18030')
#
# DropInfo=['炉号','测温时间','抽真空起时间','破空时间','温时比']
#
# for Info in DropInfo:
#     OrialData=OrialData.drop(Info,axis=1)
#
# #删除带有空格的行和列
# OrialData=OrialData.dropna()
# #['包号', '包况', '包龄', '氩气流量', '抽真空时间', '进工位温度', '破空温度', '出工位温度', '钢种', '温差']
#
# #抽取 10 < 抽真空时间 < 60
# OrialData=OrialData[(OrialData['抽真空时间']>10) & (OrialData['抽真空时间']<65)]
#
# #抽取 温差 > -100
# OrialData=OrialData[OrialData['温差']>-100]
#
#
# #OneHotEncoder计算
# #炉号---第0列
# #包号---第1列
# #包况---第2列
# #钢种---第9列
# Data=OrialData.values
#
# Data_y=Data[:,4]
#
# Data_baohao=Data[:,0]
# Data_baokuang=Data[:,1]
# Data_gangzhong=Data[:,8]
#
# #删除y值
# Data_X=np.delete(Data,4,axis=1)
# #删除包号
# Data_X=np.delete(Data_X,0,axis=1)
# #删除包况
# Data_X=np.delete(Data_X,0,axis=1)
# #删除钢种
# Data_X=np.delete(Data_X,-2,axis=1)
#
#
# enc=OneHotEncoder(sparse=False)
# Data_baohao=enc.fit_transform(Data_baohao.reshape(Data_baohao.shape[0],1))
# Data_baokuang=enc.fit_transform(Data_baokuang.reshape(Data_baokuang.shape[0],1))
# Data_gangzhong=enc.fit_transform(Data_gangzhong.reshape(Data_gangzhong.shape[0],1))
#
# # Data_X=np.c_[Data_X,Data_baohao]
# Data_X=np.c_[Data_X,Data_baokuang]
# Data_X=np.c_[Data_X,Data_gangzhong]
# Data=np.c_[Data_X,Data_y]
#
# scaler=MinMaxScaler(feature_range=(0,1))
# scaled=scaler.fit_transform(Data)
#
# Data_y=scaled[:,-1]
# Data_X=np.delete(scaled,-1,axis=1)
#
# train_X,test_X,train_y,test_y=train_test_split(Data_X,Data_y,test_size=0.25)
#
# #全连通神经网络
# model=Sequential()
# input=Data_X.shape[1]
# #隐藏层1000
# model.add(Dense(1000,input_dim=input))
# model.add(Activation('relu'))
# # model.add(Dropout(0.2))
#
# #隐藏层1000
# model.add(Dense(1000))
# model.add(Activation('relu'))
# # model.add(Dropout(0.2))
#
# #无激活函数
# model.add(Dense(1))
# #编译
# model.compile(loss='mean_squared_error',optimizer=Adam())
# #早停法
# early_stopping=EarlyStopping(monitor='val_loss',patience=50,verbose=2)
#
# history=model.fit(train_X,train_y,epochs=300,batch_size=30,
#                   validation_data=(test_X,test_y),verbose=2,
#                   shuffle=False,callbacks=[early_stopping])
#
# #loss曲线
# plt.plot(history.history['loss'],label='train')
# plt.plot(history.history['val_loss'],label='test')
# plt.legend()
# plt.show()
#
# #预测
# predict_y=model.predict(test_X)
#
# #预测y 逆标准化
# inv_Ypredict0=concatenate((test_X,predict_y),axis=1)
# inv_Ypredict1=scaler.inverse_transform(inv_Ypredict0)
# inv_Ypredict=inv_Ypredict1[:,-1]
#
# #原始y 逆标准化
# test_y=test_y.reshape(len(test_y),1)
# inv_y0=concatenate((test_X,test_y),axis=1)
# inv_y1=scaler.inverse_transform(inv_y0)
# inv_y=inv_y1[:,-1]
#
# #计算RMSE
# rmse=sqrt(mean_squared_error(inv_y,inv_Ypredict))
# print("Test RMSE : %.3f"%rmse)
# plt.plot(inv_y,label="inv_y")
# plt.plot(inv_Ypredict,label='inv_Ypredict')
# plt.legend()
# plt.show()
#
# mark=np.linspace(1,len(test_X),len(test_X))
# plt.scatter(mark,inv_y,label='inv_y')
# plt.scatter(mark,inv_Ypredict,label="inv_Ypredict")
# plt.legend()
# plt.show()
#
#
#                        #准确率
# #预测值
# predict=[float(np.round(x)) for x in inv_Ypredict]
# static=[]
# for i in range(len(predict)):
#     static.append(abs(predict[i]-inv_y[i]))
#
# static=np.array(static)
#
# count=[]
# mark=np.linspace(0,20,21)
# for row in mark:
#     count.append(np.mean(static<=row))
#
# print(count)
#
###############################################  完毕  #################################################################

################################################## 基于Keras的神经网络回归模型 ###########################################
                                             #考虑:炉号 包况 钢种 精炼冶时 总送电时间#
                                             #  对比集中不同的数据处理对准确率的影响 #
# import matplotlib.pyplot as plt
# from math import sqrt
# import pandas as pd
# import numpy as np
# from numpy import concatenate
# from sklearn.preprocessing import MinMaxScaler,StandardScaler
# from sklearn.metrics import mean_squared_error
# from keras.models import Sequential
# from keras.layers.core import Dense,Dropout,Activation
# from keras.optimizers import Adam
# from sklearn.model_selection import train_test_split
# from keras.callbacks import EarlyStopping
# from sklearn.preprocessing import OneHotEncoder
#
# OrialData=pd.read_csv("../data/vd.csv",encoding='gb18030')
#
# # DropInfo=['炉号','测温时间','抽真空起时间','破空时间','温时比']
# #
# # for Info in DropInfo:
# #     OrialData=OrialData.drop(Info,axis=1)
#
# #删除带有空格的行和列
# OrialData=OrialData.dropna()
#
# #抽取 10 < 抽真空时间 < 60
# OrialData=OrialData[(OrialData['抽真空时间']>10) & (OrialData['抽真空时间']<65)]
#
# #抽取 温差 > 0
# OrialData=OrialData[OrialData['温差']>=0]
# OrialData=OrialData[OrialData['包龄']!=441]
# OrialData=OrialData[(OrialData['精炼冶时']>50) & (OrialData['精炼冶时']<200)]             #精炼冶时 范围集中在 50~200之间
# OrialData = OrialData[(OrialData['总送电时间'] > 20) & (OrialData['总送电时间'] < 60)]    #总送电时间 范围在  20~60之间
# OrialData=OrialData[(OrialData['氩气流量'] > 200) & (OrialData['氩气流量'] < 700)]        #氩气流量 范围在  200~700
# OrialData=OrialData.drop('温差',axis=1)
#
# #OneHotEncoder计算
# Data=OrialData.values
#
# Data_y=Data[:,-1]
#
# # Data_luhao=Data[:,0]
# Data_baokuang=Data[:,1]
# Data_gangzhong=Data[:,5]
#
# #删除y值
# Data_X=np.delete(Data,-1,axis=1)
# #删除炉号
# Data_X=np.delete(Data_X,0,axis=1)
# #删除包况
# Data_X=np.delete(Data_X,0,axis=1)
# #删除钢种
# Data_X=np.delete(Data_X,-5,axis=1)
#
#
# enc=OneHotEncoder(sparse=False)
# # Data_luhao=enc.fit_transform(Data_luhao.reshape(Data_luhao.shape[0],1))
# Data_baokuang=enc.fit_transform(Data_baokuang.reshape(Data_baokuang.shape[0],1))
# Data_gangzhong=enc.fit_transform(Data_gangzhong.reshape(Data_gangzhong.shape[0],1))
#
# # Data_X=np.c_[Data_X,Data_baohao]
# Data_X=np.c_[Data_X,Data_baokuang]
# Data_X=np.c_[Data_X,Data_gangzhong]
# # Data_X=np.c_[Data_X,Data_luhao]
# Data=np.c_[Data_X,Data_y]
#
#                                            # 归一化 #
# scaler=MinMaxScaler(feature_range=(0,1))
# scaled=scaler.fit_transform(Data)
#
# #                                                # 标准化 #
# # Data_scaler=StandardScaler()
# # Data_scale=Data_scaler.fit_transform(Data)
#
# Data_y=scaled[:,-1]
# Data_X=np.delete(scaled,-1,axis=1)
#
# train_X,test_X,train_y,test_y=train_test_split(Data_X,Data_y,test_size=0.25)
#
# #全连通神经网络
# model=Sequential()
# input=Data_X.shape[1]
# #隐藏层500
# model.add(Dense(1000,input_dim=input))
# model.add(Activation('relu'))
# model.add(Dropout(0.2))
#
# #隐藏层500
# model.add(Dense(1000))
# model.add(Activation('relu'))
# model.add(Dropout(0.2))
#
# #无激活函数
# model.add(Dense(1))
# #编译
# model.compile(loss='mean_squared_error',optimizer=Adam())
# #早停法
# early_stopping=EarlyStopping(monitor='val_loss',patience=50,verbose=2)
#
# history=model.fit(train_X,train_y,epochs=300,batch_size=10,
#                   validation_data=(test_X,test_y),verbose=2,
#                   shuffle=False,callbacks=[early_stopping])
# #loss曲线
# plt.plot(history.history['loss'],label='train')
# plt.plot(history.history['val_loss'],label='test')
# plt.legend()
# plt.show()
#
# #预测
# predict_y=model.predict(test_X)
#
# #预测y 逆标准化
# inv_Ypredict0=concatenate((test_X,predict_y),axis=1)
# inv_Ypredict1=scaler.inverse_transform(inv_Ypredict0)
# inv_Ypredict=inv_Ypredict1[:,-1]
#
# #原始y 逆标准化
# test_y=test_y.reshape(len(test_y),1)
# inv_y0=concatenate((test_X,test_y),axis=1)
# inv_y1=scaler.inverse_transform(inv_y0)
# inv_y=inv_y1[:,-1]
#
# #计算RMSE
# rmse=sqrt(mean_squared_error(inv_y,inv_Ypredict))
# print("Test RMSE : %.3f"%rmse)
# plt.plot(inv_y,label="inv_y")
# plt.plot(inv_Ypredict,label='inv_Ypredict')
# plt.legend()
# plt.show()
#
# mark=np.linspace(1,len(test_X),len(test_X))
# plt.scatter(mark,inv_y,label='inv_y')
# plt.scatter(mark,inv_Ypredict,label="inv_Ypredict")
# plt.legend()
# plt.show()
#
#
#                        #准确率
# #预测值
# predict=[float(np.round(x)) for x in inv_Ypredict]
# static=[]
# for i in range(len(predict)):
#     static.append(abs(predict[i]-inv_y[i]))
#
# static=np.array(static)
#
# count=[]
# mark=np.linspace(0,20,21)
# for row in mark:
#     count.append(np.mean(static<=row))
#
# print(count)

################################################  完毕 准确率：50%左右 ##################################################


################################################  根据波斯顿房价改动  ###################################################
# import numpy as np
# import pandas as pd
# from sklearn.preprocessing import OneHotEncoder
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.wrappers.scikit_learn import KerasRegressor
# from sklearn.model_selection import cross_val_score,KFold
# from sklearn.preprocessing import StandardScaler
# from sklearn.pipeline import Pipeline
#
# OrialData=pd.read_csv("./data/vd.csv",encoding='gb18030')
# OrialData=OrialData.dropna()
#
# #抽取 10 < 抽真空时间 < 60
# OrialData=OrialData[(OrialData['抽真空时间']>10) & (OrialData['抽真空时间']<65)]
#
# #抽取 温差 > 0
# OrialData=OrialData[OrialData['温差']>=0]
#
# OrialData=OrialData.drop('温差',axis=1)
#
# #OneHotEncoder计算
# Data=OrialData.values
#
# Data_y=Data[:,-1]
#
# Data_luhao=Data[:,0]
# Data_baokuang=Data[:,1]
# Data_gangzhong=Data[:,5]
#
# #删除y值
# Data_X=np.delete(Data,-1,axis=1)
# #删除炉号
# Data_X=np.delete(Data_X,0,axis=1)
# #删除包况
# Data_X=np.delete(Data_X,0,axis=1)
# #删除钢种
# Data_X=np.delete(Data_X,-5,axis=1)
#
# enc=OneHotEncoder(sparse=False)
# # Data_luhao=enc.fit_transform(Data_luhao.reshape(Data_luhao.shape[0],1))
# Data_baokuang=enc.fit_transform(Data_baokuang.reshape(Data_baokuang.shape[0],1))
# Data_gangzhong=enc.fit_transform(Data_gangzhong.reshape(Data_gangzhong.shape[0],1))
#
# # Data_X=np.c_[Data_X,Data_baohao]
# Data_X=np.c_[Data_X,Data_baokuang]
# Data_X=np.c_[Data_X,Data_gangzhong]
# # Data_X=np.c_[Data_X,Data_luhao]
# Data=np.c_[Data_X,Data_y]
#
# Data_y=Data[:,-1]
# Data_X=np.delete(Data,-1,axis=1)
# input=Data_X.shape[1]
#                                          #  基类模型  #
# #定义基类模型
# def baseline_model():
#     model=Sequential()
#     model.add(Dense(800,input_dim=input,kernel_initializer='normal',activation='relu'))
#     model.add(Dense(1,kernel_initializer='normal'))
#     model.compile(loss='mean_squared_error',optimizer='adam')
#     return model
#
# #固定随机种子的重现性
# seed=7
# np.random.seed(seed)
# #使用标准化数据集评估模型
# estimator=KerasRegressor(build_fn=baseline_model,epochs=100,batch_size=5,verbose=0)
# #评估此基类模型，使用10倍交叉验证评估模型
# kfold=KFold(n_splits=10,random_state=seed)
# results=cross_val_score(estimator,Data_X,Data_y,cv=kfold)
# print("Results : %.2f(%.2f) MSE"%(results.mean(),results.std()))
#
#
#
# #使用 Pipeline 首先标准化数据集，然后创建和评估基线神经网络模型
# np.random.seed(seed)
# estimators = []
# estimators.append(('standaedize',StandardScaler()))
# estimators.append(('mlp',KerasRegressor(build_fn=baseline_model,epochs=200,batch_size=5,verbose=0)))
# pipeline = Pipeline(estimators)
# # 评估所创建的神经网络模型
# kfold = KFold(n_splits=10,random_state=seed)
# results = cross_val_score(pipeline,Data_X,Data_y,cv=kfold)
# print("Standardized:%.2f(%.2f)MSE"%(results.mean(),results.std()))
#
#
#
#                                          #  增加神经网络层数  #
# # 针对神经网络模型进行优化
# # 提高神经网络性能的一种方法是添加更多层，这可能允许模型提取并重新组合数据中嵌入的高阶特征
# def larger_model():
#     # 创建模型
#     model = Sequential()
#     model.add(Dense(500,input_dim=input,kernel_initializer='normal',activation='relu'))
#     model.add(Dense(500,kernel_initializer='normal',activation='relu'))
#     model.add(Dense(1,kernel_initializer='normal'))
#     # 编译模型
#     model.compile(loss='mean_squared_error',optimizer='adam')
#     return model
#
# # 使用scikit-learn Pipeline 首先标准化数据集，然后创建和评估基线神经网络模型
# np.random.seed(seed)
# estimators = []
# estimators.append(('standaedize',StandardScaler()))
# estimators.append(('mlp',KerasRegressor(build_fn=larger_model,epochs=200,batch_size=5,verbose=0)))
# pipeline = Pipeline(estimators)
# # 评估所创建的神经网络模型
# kfold = KFold(n_splits=10,random_state=seed)
# results = cross_val_score(pipeline,Data_X,Data_y,cv=kfold)
# print("Larger:%.2f(%.2f)MSE"%(results.mean(),results.std()))
#
#
#                                        #  增加神经网络神经元（宽度） #
# def wider_model():
#     # 创建模型
#     model = Sequential()
#     model.add(Dense(1000, input_dim=input, kernel_initializer='normal', activation='relu'))
#     model.add(Dense(1, kernel_initializer='normal'))
#     # 编译模型
#     model.compile(loss='mean_squared_error', optimizer='adam')
#     # predict model
#     model.fit(Data_X,Data_y, epochs=200, batch_size=5)
#     predict = model.predict(Data_X)
#     # print(predict)
#     return model
#
#
# # 使用scikit-learn Pipeline 首先标准化数据集，然后创建和评估基线神经网络模型
# np.random.seed(seed)
# estimators = []
# estimators.append(('standaedize', StandardScaler()))
# estimators.append(('mlp', KerasRegressor(build_fn=wider_model, epochs=200, batch_size=5, verbose=0)))
# pipeline = Pipeline(estimators)
# kfold = KFold(n_splits=10, random_state=seed)
# results = cross_val_score(pipeline, Data_X, Data_y, cv=kfold)
# print("Wider:%.2f(%.2f)MSE" % (results.mean(), results.std()))

############################################  运行时间太长终止  #########################################################

############################################ 神经网络模拟 y=sin(x) ######################################################
#
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import OneHotEncoder
# from sklearn.model_selection import KFold,cross_val_score,LeaveOneOut,train_test_split
# from keras.layers import Dense
# from keras.models import Sequential
# from keras.callbacks import EarlyStopping
#
# #建立数据集
# X=np.linspace(-2*np.pi,2*np.pi,200)
# o=0.1*np.random.rand(1,200)
# y_=np.sin(X)
# y=y_+o
#
# X=X.reshape(X.shape[0],1)
# y=y.reshape(y.shape[1],1)
#
# train_X,test_X,train_y,test_y=train_test_split(X,y,test_size=0.2)
#
# input=train_X.shape[1]
#
# model=Sequential()
# model.add(Dense(1000,input_dim=input,kernel_initializer='normal',activation='relu'))
# model.add(Dense(1000,kernel_initializer='normal',activation='relu'))
# model.add(Dense(1))
#
# early_stopping=EarlyStopping(monitor='val_loss',patience=50,verbose=2)
# model.compile(loss='mean_squared_error',optimizer='adam')
#
# history=model.fit(train_X,train_y,epochs=500,batch_size=5,validation_data=(test_X,test_y),verbose=2,shuffle=False,callbacks=[early_stopping])
#
# plt.plot(history.history['loss'],label='loss')
# plt.plot(history.history['val_loss'],label='val_loss')
# plt.legend()
# plt.show()
#
# predict_y=model.predict(test_X)
# y_cha=predict_y-test_y
# print(np.mean(y_cha<0.1))

############################################ 即使 y=sin(x) 模型 数据准确度也只有 60%左右 #################################

############################################## keras神经网络 LSTM算法 ###################################################

# import numpy
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.layers import LSTM
# from keras.utils import np_utils
# from keras.preprocessing.sequence import pad_sequences
# from theano.tensor.shared_randomstreams import RandomStreams
# # fix random seed for reproducibility
# numpy.random.seed(7)
# # define the raw dataset
# alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
# # create mapping of characters to integers (0-25) and the reverse
# char_to_int = dict((c, i) for i, c in enumerate(alphabet))
# int_to_char = dict((i, c) for i, c in enumerate(alphabet))
# # prepare the dataset of input to output pairs encoded as integers
# num_inputs = 1000
# max_len = 5
# dataX = []
# dataY = []
# for i in range(num_inputs):
#     start = numpy.random.randint(len(alphabet)-2)
#     end = numpy.random.randint(start, min(start+max_len,len(alphabet)-1))
#     sequence_in = alphabet[start:end+1]
#     sequence_out = alphabet[end + 1]
#     dataX.append([char_to_int[char] for char in sequence_in])
#     dataY.append(char_to_int[sequence_out])
#     print(sequence_in, '->', sequence_out)
# # convert list of lists to array and pad sequences if needed
# X = pad_sequences(dataX, maxlen=max_len, dtype='float32')
# # reshape X to be [samples, time steps, features]
# X = numpy.reshape(X, (X.shape[0], max_len, 1))
# # normalize
# X = X / float(len(alphabet))
# # one hot encode the output variable
# y = np_utils.to_categorical(dataY)
# # create and fit the model
# batch_size = 1
# model = Sequential()
# model.add(LSTM(32, input_shape=(X.shape[1], 1)))
# model.add(Dense(y.shape[1], activation='softmax'))
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# model.fit(X, y, epochs=500, batch_size=batch_size, verbose=2)
# # summarize performance of the model
# scores = model.evaluate(X, y, verbose=0)
# print("Model Accuracy: %.2f%%" % (scores[1]*100))
# # demonstrate some model predictions
# for i in range(20):
#     pattern_index = numpy.random.randint(len(dataX))
#     pattern = dataX[pattern_index]
#     x = pad_sequences([pattern], maxlen=max_len, dtype='float32')
#     x = numpy.reshape(x, (1, max_len, 1))
#     x = x / float(len(alphabet))
#     prediction = model.predict(x, verbose=0)
#     index = numpy.argmax(prediction)
#     result = int_to_char[index]
#     seq_in = [int_to_char[value] for value in pattern]
#     print(seq_in, "->", result)

################################################### 参考价值较大 ########################################################



################################################### 卷积神经网络模拟回归 #################################################

# import numpy as np
# import pandas as pd
# np.random.seed(7)
# from keras.models import Sequential
# from keras.layers import Dense,Dropout,Activation,Flatten
# from keras.layers import Convolution2D,MaxPooling2D
# from keras.utils import np_utils
# from sklearn.preprocessing import OneHotEncoder,MinMaxScaler
# from sklearn.model_selection import train_test_split
# from keras import backend as K
#
# #全局变量
# nb_filters=16
# pool_size=(3,3)
# kernel_size=(2,2)
#
# OrialData=pd.read_csv("../data/vd.csv",encoding='gb18030')
# OrialData=OrialData.dropna()
#
# #抽取 10 < 抽真空时间 < 60
# OrialData=OrialData[(OrialData['抽真空时间']>10) & (OrialData['抽真空时间']<65)]
#
# #抽取 温差 > 0
# OrialData=OrialData[OrialData['温差']>=0]
# OrialData=OrialData[OrialData['包龄']!=441]
# OrialData=OrialData[(OrialData['精炼冶时']>50) & (OrialData['精炼冶时']<200)]             #精炼冶时 范围集中在 50~200之间
# OrialData = OrialData[(OrialData['总送电时间'] > 20) & (OrialData['总送电时间'] < 60)]    #总送电时间 范围在  20~60之间
# OrialData=OrialData[(OrialData['氩气流量'] > 200) & (OrialData['氩气流量'] < 700)]        #氩气流量 范围在  200~700
# OrialData=OrialData.drop('温差',axis=1)
#
# #OneHotEncoder计算
# Data=OrialData.values
#
# Data_y=Data[:,-1]
#
# Data_luhao=Data[:,0]
# Data_baokuang=Data[:,1]
# Data_gangzhong=Data[:,5]
#
# #删除y值
# Data_X=np.delete(Data,-1,axis=1)
# #删除炉号
# Data_X=np.delete(Data_X,0,axis=1)
# #删除包况
# Data_X=np.delete(Data_X,0,axis=1)
# #删除钢种
# Data_X=np.delete(Data_X,-5,axis=1)
#
# enc=OneHotEncoder(sparse=False)
# Data_baokuang=enc.fit_transform(Data_baokuang.reshape(Data_baokuang.shape[0],1))
# Data_gangzhong=enc.fit_transform(Data_gangzhong.reshape(Data_gangzhong.shape[0],1))
#
# # Data_baokuang=np_utils.to_categorical(Data_baokuang.reshape(Data_baokuang.shape[0],1),dtype='float')
# # Data_gangzhong=np_utils.to_categorical(Data_gangzhong.reshape(Data_gangzhong.shape[0],1))
#
# Data_X=np.c_[Data_X,Data_baokuang]
# Data_X=np.c_[Data_X,Data_gangzhong]
# Data=np.c_[Data_X,Data_y]
# mark_X=[]
# for i in range(7):
#     mark_X.append(max(Data[:,i]))
# #                                                # 归一化 #
# # scaler=MinMaxScaler(feature_range=(0,1))
# # scaled=scaler.fit_transform(Data)
# for i in range(7):
#     Data[:,i]=Data[:,i]/max(Data[:,i])
#
# Data_y=Data[:,-1]
# mark_y=max(Data_y)
# Data_y=Data_y/mark_y
# Data_X=np.delete(Data,-1,axis=1)
#
# #补充
# X_buchong=np.zeros([2887,4])
# Data_X=np.c_[Data_X,X_buchong]
#
# train_X,test_X,train_y,test_y=train_test_split(Data_X,Data_y,test_size=0.25)
#
# train_X=train_X.reshape(train_X.shape[0],15,10,1)
# test_X=test_X.reshape(test_X.shape[0],15,10,1)
# input=(15,10,1)
#
#
#
#                                             # 构建模型 #
# model=Sequential()
# model.add(Convolution2D(nb_filters,(kernel_size[0],kernel_size[1]),padding="same",input_shape=input))   # 卷积层1
# model.add(Activation('relu'))
# model.add(Convolution2D(nb_filters,(kernel_size[0],kernel_size[1])))   #卷积层2
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=pool_size))
# model.add(Dropout(0.25))
# model.add(Flatten())                                                    #拉成一维数据
# model.add(Dense(500,activation='relu'))                                 #全连接层1
# model.add(Dropout(0.2))
# # model.add(Dense(500,activation='relu'))                                 #全连接层2
# # model.add(Dropout(0.2))
# model.add(Dense(1))                                                     #全连接层2
# model.compile(loss='mean_squared_error',optimizer='adam',metrics=['accuracy'])
# model.fit(train_X,train_y,epochs=20,batch_size=5,validation_data=(test_X,test_y),shuffle=False,verbose=1)
#
#                                           # 评估模型 #
# # predict_y=model.predict(test_X)                            # 0~1之间
# #
# # test_X=test_X.reshape(722,150)
# # #删除补充
# # c=[-1,-2,-3,-4]
# # for i in c:
# #     test_X=np.delete(test_X,i,axis=1)
# #
# # test_X_y=np.c_[test_X,predict_y]
# # inv_test_X_y=scaler.inverse_transform(test_X_y)
# # predict=inv_test_X_y[:,-1]                                #预测值
# #
# # X_y=np.c_[test_X,test_y]
# # inv_X_y=scaler.inverse_transform(X_y)
# # test_y_=inv_X_y[:,-1]
# #                                                           #准确率
# # predict=np.round(predict)
# # cha=predict-test_y_
# #
# # mark=np.linspace(0,20,21)
# # for i in mark:
# #     print(np.mean(abs(cha)<=i))
#
#
#
#
# predict_y=model.predict(test_X)
#
# predict=predict_y*mark_y
#
# test_y=test_y*mark_y
#
#
#
#
# predict=np.round(predict)
# cha=predict-test_y
#
# mark=np.linspace(0,20,21)
# for i in mark:
#     print(np.mean(abs(cha)<=i))

##################################################  准确率提升不明显  ###################################################


##################################################     PCA(主成分分析）       ###########################################

# import numpy as np
# import pandas as pd
# from sklearn.decomposition import PCA
#
# OrialData=pd.read_csv('../data/vd.csv',encoding='gb18030')
# properties=['炉号','包况','钢种']
# for key in properties:
#     OrialData=OrialData.drop(key,axis=1)
#
# OrialData=OrialData.dropna()
#
# Data=OrialData.values
#
# # pca=PCA(n_components='mle')
# # newData=pca.fit_transform(Data)
#
# pca=PCA(n_components='mle')
# newData=pca.fit_transform(Data)
# print(pca.n_components_,pca.explained_variance_,pca.explained_variance_ratio_)

########################################################################################################################


#################################################   PCA实例   ##########################################################

# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from sklearn.datasets.samples_generator import make_blobs
# # X为样本特征，Y为样本簇类别， 共1000个样本，每个样本3个特征，共4个簇
# X, y = make_blobs(n_samples=10000, n_features=3, centers=[[3,3, 3], [0,0,0], [1,1,1], [2,2,2]], cluster_std=[0.2, 0.1, 0.2, 0.2],
#                   random_state =9)
# fig = plt.figure()
# ax = Axes3D(fig, rect=[0, 0, 1, 1], elev=30, azim=20)
# plt.scatter(X[:, 0], X[:, 1], X[:, 2],marker='o')
# plt.show()

########################################################################################################################

###############################################  神经网络实时连接数据库  #################################################

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
import cx_Oracle

def LoadData(sql):
    conn = cx_Oracle.connect("YGSJPT/YGSJPT@10.4.133.222/YGSJPT")    #此处需修改
    cursor=conn.cursor()
    cursor.execute(sql)
    data=cursor.fetchall()
    cursor.close()
    conn.close()
    columns = ["炉号", "钢种", "包况", "包龄", "氩气流量", "精炼冶时", "总送电时间",
               "进工位温度",  "抽真空时间"]
    data=pd.DataFrame(data,columns=columns)
    return data

def Datahandle(data):
    OrialData=data.dropna()
    #抽取 10 < 抽真空时间 < 60
    OrialData['抽真空时间']=OrialData['抽真空时间'].astype('float')
    OrialData['包龄']=OrialData['包龄'].astype('float')
    OrialData=OrialData[(OrialData['抽真空时间']>10) & (OrialData['抽真空时间']<65)]
    OrialData=OrialData[OrialData['包龄']!=441]
    OrialData=OrialData[(OrialData['精炼冶时']>50) & (OrialData['精炼冶时']<200)]             #精炼冶时 范围集中在 50~200之间
    OrialData = OrialData[(OrialData['总送电时间'] > 20) & (OrialData['总送电时间'] < 60)]    #总送电时间 范围在  20~60之间
    OrialData=OrialData[(OrialData['氩气流量'] > 200) & (OrialData['氩气流量'] < 700)]        #氩气流量 范围在  200~700
    OrialData=OrialData.drop('炉号',axis=1)
    Data=OrialData.values
    return Data

def DataProcessingAndModelAndPredict(Data):
    Data_y=Data[:,-1]
    Data_baokuang=Data[:,1]
    Data_gangzhong=Data[:,0]
    #删除y值
    Data_X=np.delete(Data,-1,axis=1)
    #删除包况
    Data_X=np.delete(Data_X,1,axis=1)
    #删除钢种
    Data_X=np.delete(Data_X,0,axis=1)
    enc=OneHotEncoder(sparse=False)
    Data_baokuang=enc.fit_transform(Data_baokuang.reshape(Data_baokuang.shape[0],1))
    Data_gangzhong=enc.fit_transform(Data_gangzhong.reshape(Data_gangzhong.shape[0],1))
    Data_X=np.c_[Data_X,Data_baokuang]
    Data_X=np.c_[Data_X,Data_gangzhong]
    Data=np.c_[Data_X,Data_y]
                                               # 归一化 #
    scaler=MinMaxScaler(feature_range=(0,1))
    scaled=scaler.fit_transform(Data)

    Data_y=scaled[:,-1]
    Data_X=np.delete(scaled,-1,axis=1)

    train_X,test_X,train_y,test_y=train_test_split(Data_X,Data_y,test_size=0.25)

    #全连通神经网络
    model=Sequential()
    input=train_X.shape[1]
    #隐藏层500
    model.add(Dense(1000,input_dim=input))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    #隐藏层500
    model.add(Dense(1000))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    #无激活函数
    model.add(Dense(1))
    #编译
    model.compile(loss='mean_squared_error',optimizer=Adam())
    #早停法
    early_stopping=EarlyStopping(monitor='val_loss',patience=50,verbose=2)

    history=model.fit(train_X,train_y,epochs=300,batch_size=10,
                      validation_split=0.2,verbose=2,
                      shuffle=False,callbacks=[early_stopping])
    #loss曲线
    plt.plot(history.history['loss'],label='train')
    plt.plot(history.history['val_loss'],label='test')
    plt.legend()
    plt.show()

    #预测
    predict_y=model.predict(test_X)

    #预测y 逆标准化
    inv_Ypredict0=concatenate((test_X,predict_y),axis=1)
    inv_Ypredict1=scaler.inverse_transform(inv_Ypredict0)
    inv_Ypredict=inv_Ypredict1[:,-1]

    #原始y 逆标准化
    test_y=test_y.reshape(len(test_y),1)
    inv_y0=concatenate((test_X,test_y),axis=1)
    inv_y1=scaler.inverse_transform(inv_y0)
    inv_y=inv_y1[:,-1]

    #计算RMSE
    rmse=sqrt(mean_squared_error(inv_y,inv_Ypredict))
    print("Test RMSE : %.3f"%rmse)
    plt.plot(inv_y,label="inv_y")
    plt.plot(inv_Ypredict,label='inv_Ypredict')
    plt.legend()
    plt.show()

    mark=np.linspace(1,len(test_X),len(test_X))
    plt.scatter(mark,inv_y,label='inv_y')
    plt.scatter(mark,inv_Ypredict,label="inv_Ypredict")
    plt.legend()
    plt.show()
                           #准确率
    #预测值
    predict=[float(np.round(x)) for x in inv_Ypredict]
    static=[]
    for i in range(len(predict)):
        static.append(abs(predict[i]-inv_y[i]))

    static=np.array(static)

    count=[]
    mark=np.linspace(0,20,21)
    for row in mark:
        count.append(np.mean(static<=row))

    print(count)

def main():
    sql="select a.炉号,钢种,a.包况,a.包龄,b.fine_vacuum_ar_flow 氩气流量,a.精炼冶时,总送电时间,b.arrive_gw_temp 进工位温度,b.vacuum_pump_duration 抽真空时间 " \
          "from v_lf_sj a ,TB_VD_PRODATA b " \
          "where a.炉号 = b.heatno order by a.炉号"
    data=LoadData(sql)
    Data=Datahandle(data)
    DataProcessingAndModelAndPredict(Data)

if __name__=="__main__":
    main()