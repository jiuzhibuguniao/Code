#-*-coding:utf-8-*-
import pandas as pd
from folium.plugins import HeatMap
from coordTransform_utils import gcj02_to_wgs84
import folium


"""
Date:2019-11-19
Author:Rooobins
读取TXT文件并生成热力图
"""

#读取文件
def readTXT(path,filename):
    with open(path+filename+'.txt','r',encoding='utf-8') as f:
        OrialData=f.readlines()
        f.close()
    return OrialData

#数据处理并返回坐标值
def dataHandle(OrialData):
    OrialDataA=[row.strip('\n').split("\t") for row in OrialData]
    OrialDataB=pd.DataFrame(OrialDataA)
    OrialDataB[6]=OrialDataB[6].astype('float32')
    OrialDataB[7]=OrialDataB[7].astype('float32')
    OrialDataC=OrialDataB.drop_duplicates(1)
    OrialDataD=OrialDataC[[6,7]]
    BikePosition=OrialDataD.values
    BikePositionValuesList=[gcj02_to_wgs84(row[0],row[1]) for row in BikePosition]
    for i in range(len(BikePositionValuesList)):
        BikePositionValuesList[i].reverse()
    return BikePositionValuesList

#热力图
def DrawHeatMap(BikePositionValuesList,path,filename):
    m = folium.Map([38., 112.], tiles='stamentoner', zoom_start=13)
    HeatMap(BikePositionValuesList,radius=10, gradient={.4: 'blue', .65: 'lime', 1: 'yellow'}).add_to(m)
    m.save(path+filename+'.html')
    return

def main():
    path='C:/Users/Rooobins/Desktop/'
    filename='Fri_Apr_2019_04_12_00_47_51'
    OrialData=readTXT(path,filename)
    data=dataHandle(OrialData)
    DrawHeatMap(data,path,filename)
    return

if __name__=="__main__":
    main()