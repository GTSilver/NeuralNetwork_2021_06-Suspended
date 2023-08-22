from numpy import float64

import lib_Import as libI

print("\nЗагрузка данных для обучения...")

inputFile = open("D:/Рабочий стол/Нейронные сети/Графики для сети/grathInput 3-1-4-13140.txt", "r")
outputFile = open("D:/Рабочий стол/Нейронные сети/Графики для сети/grathOutput 3-1-4-13140.txt", "r")

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
for string in inputStrings:
    temporaryArray = string.split('\t')
    temporaryList = []
    for i in range(0, len(temporaryArray) - 1):
        temporaryList.append(float64(temporaryArray[i].replace(',', '.')))

    inputList.append(temporaryList)
    inputSize = len(temporaryList)

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
