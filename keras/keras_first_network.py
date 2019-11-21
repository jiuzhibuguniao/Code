#_*_ coding:utf-8_*_

from keras.models import Sequential
from keras.layers import Dense
import numpy
seed=7
numpy.random.seed(seed)


#加载数据
dataset=numpy.loadtxt("../data/pima-indians-diabetes.csv",delimiter=",")

#划分数据和标签
X=dataset[:,0:8]
Y=dataset[:,8]

#定义模型
model=Sequential()
# model.add(Dense(12,input_dim=8,init='uniform',activation='relu'))
model.add(Dense(12,input_shape=(8,),init='uniform',activation='relu'))
model.add(Dense(8,init='uniform',activation='relu'))
# model.add(Dense(1,init='uniform',activation='sigmoid'))
model.add(Dense(1,init='uniform',activation='normal'))

#编译模型
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

#训练模型
model.fit(X,Y,nb_epoch=150,batch_size=10)

#评估模型
scores=model.evaluate(X,Y)
print("%s : %.2f%%"%(model.metrics_names[1],scores[1]*100))

#模型预测
predictions=model.predict(X)