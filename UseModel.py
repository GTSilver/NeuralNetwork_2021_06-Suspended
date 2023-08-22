print("Импортирование библиотек...")
import os
import numpy

from keras.models import load_model
from numpy import float64

print("Предустановка путей...")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
pathGlobal = "D:/Рабочий стол/Models/"
pathToModel = pathGlobal + "/Model 5.5 4.1 4.1 4.6-R15 (2-10 2-2)" + "/"
pathToEpoch = pathToModel + "/Epoch/"
pathToAdvanceInput = pathToModel + "/data_advanced.txt"
pathToInput = pathGlobal + "/model_input.txt"
barCount = 10
barSize = 5


print("Загрузка модели...")
pathList = os.listdir(pathToEpoch)
model = load_model(pathToEpoch + pathList[0])


print("Получение входных данных...")
fileInput = (open(pathToInput, "r")).readline()
d1Array = fileInput.split('\t')
d1List = []
lenBase = int(round(len(d1Array) / barSize, 0))
inputArray = []

for i in range(0, lenBase):
    d1ListLow = []

    for j in range(0, barSize):
        d1ListLow.append(float64(d1Array[i * barSize + j].replace(',', '.')))
    inputArray.append(numpy.array(d1ListLow))


print("Установка границ...")
fileAdvance = (open(pathToAdvanceInput, "r")).readlines()
dataMaxTemp = numpy.array(fileAdvance[0].split(' '))
dataMax = numpy.hstack(dataMaxTemp)
for i in range(0, barCount - 1):
    dataMax = numpy.vstack((dataMax, dataMaxTemp))
dataMax = float64(dataMax)

dataMinTemp = numpy.array(fileAdvance[1].split(' '))
dataMin = numpy.hstack(dataMinTemp)
for i in range(0, barCount - 1):
    dataMin = numpy.vstack((dataMin, dataMinTemp))
dataMin = float64(dataMin)


print("Вычисление...")
inputArray = numpy.array((inputArray - dataMin) / (dataMax - dataMin))
inputList = [inputArray]
inputList = float64(inputList)
predict = model.predict(inputList, batch_size = len(inputList))[0]

itemClose = str(round(predict[0], 5))
itemMax = str(round(predict[1], 5))
itemMin = str(round(predict[2], 5))

print(str.format("Cls item = \t{0}\nMax item = \t{1}\nMin item = \t{2}\n", itemClose, itemMax, itemMin))
