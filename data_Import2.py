import numpy
from numpy import float64

import lib_Import as libI

print("\nЗагрузка данных для обучения...")

inputFile = open("E:/grathInput 5-1-4-13140.txt", "r")
outputFile = open("E:/grathOutput 5-1-4-13140.txt", "r")

inputStrings = inputFile.readlines()
outputStrings = outputFile.readlines()

inputStrings.remove(inputStrings[len(inputStrings) - 1])
outputStrings.remove(outputStrings[len(outputStrings) - 1])

inputList = []
outputList = []

inputFile.close()
outputFile.close()

inputSize = 0
outputSize = 0

parametersCount = 222
for stringLine in inputStrings:
    d1Array = stringLine.split('\t')

    d1List = []
    barCount = int(round(len(d1Array) / parametersCount, 0))

    for i in range(barCount):
        d1ListLow = []
        for j in range(parametersCount):
            index = parametersCount * i + j
            d1ListLow.append(float64(d1Array[index].replace(',', '.')))

        d1List.append(numpy.array(d1ListLow))

    inputList.append(numpy.array(d1List))

for string in outputStrings:
    temporaryArray = string.split('\t')
    temporaryList = []
    for i in range(0, len(temporaryArray) - 1):
        temporaryList.append(float64(temporaryArray[i].replace(',', '.')))

    outputList.append(temporaryList)
    outputSize = len(temporaryList)

inputArray = libI.np.array(inputList)
outputArray = libI.np.array(outputList)

# print("\t...пример переменных (входные): {x:.4f} / {y:.4f} / {z:.0f}".format(x = inputArray[0][0], y = inputArray[1][3], z = len(inputArray)))
# print("\t...пример переменных (выходные): {x:.4f} / {y:.4f} / {z:.0f}".format(x = outputArray[0][0], y = outputArray[1][3], z = len(outputArray)))
