                            #  keras实现花朵卷积神经网络  #


import keras
from keras.models import Sequential
from keras.layers import Dense,Activation,Dropout,Flatten,Conv2D,MaxPooling2D
from keras.layers.normalization import BatchNormalization
import numpy as np
np.random.seed(1000)

import tflearn.datasets.oxflower17 as oxflower17
x,y=oxflower17.load_data(one_hot=True)


                                 #创建模型
#第一次卷积
model=Sequential()
model.add(Conv2D(filters=96,input_shape=(224,224,3),kernel_size=(11,11),strides=(4,4),padding='valid'))
model.add(Activation('relu'))
#第一次池化
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2),padding='valid'))
model.add(BatchNormalization())

#第二次卷积
model.add(Conv2D(filters=256,kernel_size=(11,11),strides=(1,1),padding='valid'))
model.add(Activation('relu'))
#第二次池化
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2),padding='valid'))
model.add(BatchNormalization())

#第三次卷积
model.add(Conv2D(filters=384,kernel_size=(3,3),strides=(1,1),padding='valid'))
model.add(Activation('relu'))
model.add(BatchNormalization())

#第四次卷积
model.add(Conv2D(filters=384,kernel_size=(3,3),strides=(1,1),padding='valid'))
model.add(Activation('relu'))
model.add(BatchNormalization())

#第五次卷积
model.add(Conv2D(filters=256,kernel_size=(3,3),strides=(1,1),padding='valid'))
model.add(Activation('relu'))
#第五次池化
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2),padding='valid'))
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
model.add(Dense(17))
model.add(Activation('softmax'))
model.summary()

#编译
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
#训练
model.fit(x,y,batch_size=64,epochs=1,verbose=1,validation_split=0.2,shuffle=True)