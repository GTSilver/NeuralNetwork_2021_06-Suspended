import os.path
import random
from time import sleep

import numpy
import sklearn.metrics
from keras.models import load_model
from matplotlib import pyplot as plt
import lib_Import as libI
import data_Import as dataI
from sklearn.decomposition import PCA

validationPercent = 0.3
learningRate = 0.0001
batchSize = 100000
epochCount = 2048000
randomState = 16
patienceCount = 128000

print("\nСоздание модели...")

A = dataI.inputArray.copy()
B = dataI.inputArray.copy()
pca = PCA()
pca.fit(A)
msc_new = pca.transform(A)
plt.plot(pca.explained_variance_ratio_[:].cumsum())
plt.savefig('D:/Рабочий стол/w.png')

pca = PCA(2)
pca.fit(B)
msc_new = pca.transform(B)

# dataMax = msc_new.max(axis=0)
# dataMin = msc_new.min(axis=0)
# msc_new = (msc_new - dataMin) / (dataMax - dataMin)


(traneX, testX, trainY, testY) = libI.train_test_split(msc_new,
                                                       dataI.outputArray,
                                                       test_size=validationPercent,
                                                       random_state=randomState)

(valX, testX, valY, testY) = libI.train_test_split(testX,
                                                   testY,
                                                   test_size=0.5,
                                                   random_state=randomState)
print(B.shape)
print(msc_new.shape)
model = libI.Sequential()

# model.add(libI.Conv1D(256, 3, input_shape=(2, 1)))
# model.add(libI.MaxPool1D(pool_size = 2))
# model.add(libI.Flatten())

model.add(libI.Dense(128, input_shape = (None, msc_new.shape[0], msc_new.shape[1])))
model.add(libI.Dense(64))
model.add(libI.Dense(dataI.outputSize, activation=None))

model.compile(loss=libI.MeanAbsoluteError(),
              optimizer=libI.Adam(learning_rate=learningRate,
                                  beta_1=0.9,
                                  beta_2=0.999,
                                  epsilon=10 ** -8,
                                  amsgrad=False))
model.summary()

directory = "D:/Рабочий стол/Models/Model-" + str(random.random())
os.mkdir(directory)
stoppingData = libI.EarlyStopping(monitor="loss", patience=patienceCount, verbose=0)
checkpointer = libI.ModelCheckpoint(filepath=directory + "/Epoch/Best_{val_loss:5f}_{epoch:1d}.hdf5", verbose=1,
                                    save_best_only=True, randomState=randomState)
History = model.fit(traneX, trainY,
                    validation_data=(testX, testY),
                    epochs=epochCount, verbose=0,
                    callbacks=[checkpointer, stoppingData],
                    batch_size=batchSize,
                    validation_batch_size=len(testX))

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

prediction3 = model.predict(valX, batch_size=len(valX))
errors3 = sklearn.metrics.mean_absolute_error(valY, prediction3, multioutput="raw_values")
err3 = (str(round(errors3[0] * m, 1)) + "\t" +
        str(round(errors3[1] * m, 1)) + "\t" +
        str(round(errors3[2] * m, 1)) + "\t" +
        str(round(errors3[3] * m, 1)) + "\t" +
        str(round(sklearn.metrics.mean_absolute_error(valY, prediction3) * m, 1)))
print(err3)

N = numpy.arange(0, len(History.history["loss"]))
plt.style.use("ggplot")
plt.figure(figsize=(32, 36))
plt.plot(N, History.history["loss"], label="train_loss")
plt.plot(N, History.history["val_loss"], label="val_loss")
setupId = str(validationPercent) + "; " + str(learningRate) + "; " + str(batchSize) + "; " + str(
    epochCount) + "; " + str(randomState) + "; " + str(patienceCount)
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

prediction3 = model.predict(valX, batch_size=len(valX))
errors3 = sklearn.metrics.mean_absolute_error(valY, prediction3, multioutput="raw_values")
err3 = (str(round(errors3[0] * m, 1)) + "\t" +
        str(round(errors3[1] * m, 1)) + "\t" +
        str(round(errors3[2] * m, 1)) + "\t" +
        str(round(errors3[3] * m, 1)) + "\t" +
        str(round(sklearn.metrics.mean_absolute_error(valY, prediction3) * m, 1)))
print(err3)
