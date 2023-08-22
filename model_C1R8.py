import os.path
import random
from time import sleep

import numpy
import sklearn.metrics
from keras.models import load_model
from matplotlib import pyplot as plt

import data_Import

validationPercent = 0.2
learningRate = 0.0001
batchSize = 32
epochCount = 2048
randomState = 16
patienceCount = 256

import lib_Import as libI
import data_Import as dataI
print("\nСоздание модели...")

dataI.inputArray
dataMax = dataI.inputArray.max(axis=0)
dataMin = dataI.inputArray.min(axis=0)
dataI.inputArray = (dataI.inputArray - dataMin) / (dataMax - dataMin)

(traneX, testX, trainY, testY) = libI.train_test_split(dataI.inputArray,
                                                       dataI.outputArray,
                                                       test_size = validationPercent,
                                                       random_state = randomState)

# print("\t...пример переменных (trainX): {x:.4f} / {y:.4f} / {z:.0f}".format(x = traneX[0][0], y = traneX[1][0], z = len(traneX)))
# print("\t...пример переменных (testX): {x:.4f} / {y:.4f} / {z:.0f}".format(x = testX[0][0], y = testX[1][0], z = len(testX)))
# print("\t...пример переменных (trainY): {x:.4f} / {y:.4f} / {z:.0f}".format(x = trainY[0][3], y = trainY[1][3], z = len(trainY)))
# print("\t...пример переменных (testY): {x:.4f} / {y:.4f} / {z:.0f}\n".format(x = testY[0][3], y = testY[1][3], z = len(testY)))
# print(str(dataMax) + "\n" + str(dataMin))

model = libI.Sequential()
model.add(libI.Conv1D(256, 3, input_shape=(data_Import.inputSize, 1)))
model.add(libI.MaxPool1D(pool_size=4))
# model.add(libI.Dropout(0.15))

# model.add(libI.Conv1D(256, 32, padding="same"))
# model.add(libI.MaxPool1D(pool_size=16, padding="same"))
# model.add(libI.Dropout(0.15))

# model.add(libI.Conv1D(256, 32, padding="same"))
# model.add(libI.MaxPool1D(pool_size=16, padding="same"))
# model.add(libI.Dropout(0.15))

model.add(libI.Flatten())

# model.add(libI.Dense(256))
model.add(libI.Dense(128))
model.add(libI.Dense(64))
model.add(libI.Dense(data_Import.outputSize, activation=None))
model.compile(loss = libI.MeanAbsoluteError(), optimizer = libI.Adam(learning_rate=learningRate, beta_1=0.9, beta_2=0.999, epsilon=10**-8, amsgrad=False))
model.summary()

directory = "D:/Рабочий стол/Models/Model-" + str(random.random())
os.mkdir(directory)
stoppingData = libI.EarlyStopping(monitor="loss", patience=patienceCount, verbose = 0)
checkpointer = libI.ModelCheckpoint(filepath=directory + "/Epoch/Best_{val_loss:5f}_{epoch:1d}.hdf5", verbose=1, save_best_only=True)
History = model.fit(traneX, trainY, validation_data=(testX, testY), epochs=epochCount, verbose=0, callbacks=[checkpointer, stoppingData], batch_size=batchSize, validation_batch_size=len(testX))

m = 10000
prediction1 = model.predict(traneX, batch_size=len(traneX))
errors1 = sklearn.metrics.mean_absolute_error(trainY, prediction1, multioutput="raw_values")
err1 = (str(round(errors1[0] * m, 1)) + " " +
        str(round(errors1[1] * m, 1)) + " " +
        str(round(errors1[2] * m, 1)) + " " +
        str(round(errors1[3] * m, 1)) + " " +
        str(round(sklearn.metrics.mean_absolute_error(trainY, prediction1) * m, 1)))
print(err1)

prediction2 = model.predict(testX, batch_size=len(testX))
errors2 = sklearn.metrics.mean_absolute_error(testY, prediction2, multioutput="raw_values")
err2 = (str(round(errors2[0] * m, 1)) + "\t" +
        str(round(errors2[1] * m, 1)) + "\t" +
        str(round(errors2[2] * m, 1)) + "\t" +
        str(round(errors2[3] * m, 1)) + "\t" +
        str(round(sklearn.metrics.mean_absolute_error(testY, prediction2) * m, 1)))
print(err2)

prediction3 = model.predict(traneX, batch_size=len(traneX))
errors3 = sklearn.metrics.mean_squared_error(trainY, prediction3, multioutput="raw_values")
err3 = (str(round(errors3[0] * m, 5)) + "\t" +
        str(round(errors3[1] * m, 5)) + "\t" +
        str(round(errors3[2] * m, 5)) + "\t" +
        str(round(errors3[3] * m, 5)) + "\t" +
        str(round(sklearn.metrics.mean_squared_error(trainY, prediction3) * m, 4)))
print(err3)

prediction4 = model.predict(testX, batch_size=len(testX))
errors4 = sklearn.metrics.mean_squared_error(testY, prediction4, multioutput="raw_values")
err4 = (str(round(errors4[0] * m, 5)) + "\t" +
        str(round(errors4[1] * m, 5)) + "\t" +
        str(round(errors4[2] * m, 5)) + "\t" +
        str(round(errors4[3] * m, 5)) + "\t" +
        str(round(sklearn.metrics.mean_squared_error(testY, prediction4) * m, 4)))
print(err4)

N = numpy.arange(0, len(History.history["loss"]))
plt.style.use("ggplot")
plt.figure(figsize=(32, 36))
plt.plot(N, History.history["loss"], label="train_loss")
plt.plot(N, History.history["val_loss"], label="val_loss")
setupId = str(validationPercent) + "; " + str(learningRate) + "; " + str(batchSize) + "; " + str(epochCount) + "; " + str(randomState) + "; " + str(patienceCount)
plt.title("Result: " + setupId)
plt.xlabel("Epoch")
plt.ylabel("Loss")
# plt.xlim(0, 1024)
plt.ylim(0, 0.005)
plt.legend()
idNumber = str(round(sklearn.metrics.mean_absolute_error(testY, prediction2) * m, 2))
plt.savefig(directory + "/Loss " + str(idNumber) + " " + str(len(History.history["loss"])) + ".png")
model.save(directory + "/Model " + str(idNumber) + " " + str(len(History.history["loss"])) + ".hdf5")
sleep(1)

pathNew = directory.split('-')[0] + " " + err1
os.rename(directory, pathNew)
sleep(1)

pathList = os.listdir(pathNew + "/Epoch/")
# indexToNewFile = 0
# for i in range(0, len(pathList) - 1):
#     if os.path.getmtime(pathNew + "/Epoch/" + pathList[indexToNewFile]) < pathNew + "/Epoch/" + os.path.getmtime(pathList[i]):
#         indexToNewFile = i

model = load_model(pathNew + "/Epoch/" + pathList[0])

print("\n")
prediction1 = model.predict(traneX, batch_size=len(traneX))
errors1 = sklearn.metrics.mean_absolute_error(trainY, prediction1, multioutput="raw_values")
err1 = (str(round(errors1[0] * m, 1)) + " " +
        str(round(errors1[1] * m, 1)) + " " +
        str(round(errors1[2] * m, 1)) + " " +
        str(round(errors1[3] * m, 1)) + " " +
        str(round(sklearn.metrics.mean_absolute_error(trainY, prediction1) * m, 1)))
print(err1)

prediction2 = model.predict(testX, batch_size=len(testX))
errors2 = sklearn.metrics.mean_absolute_error(testY, prediction2, multioutput="raw_values")
err2 = (str(round(errors2[0] * m, 1)) + "\t" +
        str(round(errors2[1] * m, 1)) + "\t" +
        str(round(errors2[2] * m, 1)) + "\t" +
        str(round(errors2[3] * m, 1)) + "\t" +
        str(round(sklearn.metrics.mean_absolute_error(testY, prediction2) * m, 1)))
print(err2)

prediction3 = model.predict(traneX, batch_size=len(traneX))
errors3 = sklearn.metrics.mean_squared_error(trainY, prediction3, multioutput="raw_values")
err3 = (str(round(errors3[0] * m, 5)) + "\t" +
        str(round(errors3[1] * m, 5)) + "\t" +
        str(round(errors3[2] * m, 5)) + "\t" +
        str(round(errors3[3] * m, 5)) + "\t" +
        str(round(sklearn.metrics.mean_squared_error(trainY, prediction3) * m, 4)))
print(err3)

prediction4 = model.predict(testX, batch_size=len(testX))
errors4 = sklearn.metrics.mean_squared_error(testY, prediction4, multioutput="raw_values")
err4 = (str(round(errors4[0] * m, 5)) + "\t" +
        str(round(errors4[1] * m, 5)) + "\t" +
        str(round(errors4[2] * m, 5)) + "\t" +
        str(round(errors4[3] * m, 5)) + "\t" +
        str(round(sklearn.metrics.mean_squared_error(testY, prediction4) * m, 4)))
print(err4)
