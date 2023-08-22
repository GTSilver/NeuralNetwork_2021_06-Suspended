import keras.callbacks
import numpy
import sklearn.metrics
from matplotlib import pyplot as plt
from numpy import float64
from keras import backend as kb

# def mae(y_true, y_pred):
#     true_value = kb.sum(y_true * kb.arange(0, 100, dtype="float32"), axis=-1)
#     pred_value = kb.sum(y_pred * kb.arange(0, 100, dtype="float32"), axis=-1)
#     mae = kb.mean(kb.abs(true_value - pred_value))
#     return mae
import data_Import

validationPercent = 0.15
learningRate = 0.000033
batchSize = 4
epochCount = 1024
randomState = 64
patienceCount = 32

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
model.add(libI.Conv1D(256, 3, input_shape=(data_Import.inputSize, 1), padding="same"))
model.add(libI.MaxPool1D(pool_size=16, padding="same"))
model.add(libI.Dropout(0.15))

model.add(libI.Conv1D(256, 32, padding="same"))
model.add(libI.MaxPool1D(pool_size=16, padding="same"))
model.add(libI.Dropout(0.15))

model.add(libI.Conv1D(256, 32, padding="same"))
model.add(libI.MaxPool1D(pool_size=16, padding="same"))
model.add(libI.Dropout(0.15))

model.add(libI.Flatten())

model.add(libI.Dense(256))
model.add(libI.Dense(128))
model.add(libI.Dense(64))
model.add(libI.Dense(data_Import.outputSize, activation=None))
model.compile(loss = libI.MeanAbsoluteError(), optimizer = libI.Adam(learning_rate=learningRate, beta_1=0.9, beta_2=0.999, epsilon=10**-8, amsgrad=False))
model.summary()
# print("Тренеруемая = ", model.trainable, "\n")

stoppingData = libI.EarlyStopping(monitor="val_loss", patience=patienceCount, verbose = 0)
checkpointer = libI.ModelCheckpoint(filepath="D:/Рабочий стол/Epochs/Best_{val_loss:5f}_{epoch:3d}.hdf5", verbose=0, save_best_only=True)
# History = model.fit(traneX, trainY, validation_data=(testX, testY), epochs=epochCount, verbose=0, callbacks=[checkpointer, stoppingData], batch_size=batchSize)
History = model.fit(traneX, trainY, validation_data=(testX, testY), epochs=epochCount, verbose=1, callbacks=[checkpointer, stoppingData], batch_size=batchSize, validation_batch_size=len(testX))
prediction = model.predict(testX, batch_size=len(testX))
errors = sklearn.metrics.mean_absolute_error(testY, prediction, multioutput="raw_values")
m = 10000
print("\n" + str(errors[0]*m) + "\n" + str(errors[1]*m) + "\n" + str(errors[2]*m) + "\n" + str(errors[3]*m) + "\n")
print(sklearn.metrics.mean_absolute_error(testY, prediction)*m)
# print(History.history.keys())

N = numpy.arange(0, len(History.history["loss"]))
plt.style.use("ggplot")
plt.figure(figsize=(64, 36))
plt.plot(N, History.history["loss"], label="train_loss")
plt.plot(N, History.history["val_loss"], label="val_loss")
setupId = str(validationPercent) + "; " + str(learningRate) + "; " + str(batchSize) + "; " + str(epochCount) + "; " + str(randomState) + "; " + str(patienceCount)
plt.title("Result: " + setupId)
plt.xlabel("Epoch")
plt.ylabel("Loss")
# plt.xlim(0, len(History.history["loss"]))
plt.xlim(0, 1024)
plt.ylim(0, 0.25)
# plt.ylim(0, 250)
plt.legend()
idNumber = str(round(sklearn.metrics.mean_absolute_error(testY, prediction)*m, 2))
plt.savefig("D:/Рабочий стол/Losses/Loss_" + str(idNumber) + "_" + str(len(History.history["loss"])) + ".png")
model.save("D:/Рабочий стол/Models/Model_" + str(idNumber) + "_" + str(len(History.history["loss"])) + ".model")

# predictList = []
# temporaryArray = "0.8512\t0.8496\t0.8522\t0.8495\t4802".split('\t')
# predictList.append([
# float64(temporaryArray[0].replace(',', '.')),
# float64(temporaryArray[1].replace(',', '.')),
# float64(temporaryArray[2].replace(',', '.')),
# float64(temporaryArray[3].replace(',', '.')),
# float64(temporaryArray[4].replace(',', '.'))])
# predictArray = libI.np.array(predictList)
# predictArray = (predictArray - dataMin) / (dataMax - dataMin)
# pred = model.predict(predictArray)
# print(pred, pred.shape)
