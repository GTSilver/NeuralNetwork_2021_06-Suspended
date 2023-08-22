import numpy
import sklearn.metrics
from matplotlib import pyplot as plt

validationPercent = 0.3
learningRate = 0.01
batchSize = 128
epochCount = 32
randomState=10
patienceCount = 4

import lib_Import as libI
import data_Import as dataI

print("\nСоздание модели...")

(traneX, testX, trainY, testY) = libI.train_test_split(dataI.inputArray,
                                                       dataI.outputArray,
                                                       test_size = validationPercent,
                                                       random_state = randomState)

print("\t...пример переменных (trainX): {x:.4f} / {y:.4f} / {z:.0f}".format(x = traneX[0][5], y = traneX[1][5], z = len(traneX)))
print("\t...пример переменных (testX): {x:.4f} / {y:.4f} / {z:.0f}".format(x = testX[0][5], y = testX[1][5], z = len(testX)))
print("\t...пример переменных (trainY): {x:.4f} / {y:.4f} / {z:.0f}".format(x = trainY[0][4], y = trainY[1][4], z = len(trainY)))
print("\t...пример переменных (testY): {x:.4f} / {y:.4f} / {z:.0f}\n".format(x = testY[0][4], y = testY[1][4], z = len(testY)))

model = libI.Sequential()
# model.add(libI.Conv1D(6, 3, input_shape=(6,), name="C1"))
model.add(libI.Dense(64, input_shape=(6,), activation="relu", name="d1"))
model.add(libI.Dense(32, activation="relu", name="d2"))
model.add(libI.Dense(5, name="O"))
model.compile(loss = libI.MeanSquaredError(), optimizer = libI.SGD(learning_rate=learningRate))

#print("Тренеруемая = ", model.trainable, "\n")

stoppingData = libI.EarlyStopping(monitor='val_loss', patience=patienceCount, verbose=1)
checkpointer = libI.ModelCheckpoint(filepath="D:/Рабочий стол/Best9.h5", verbose=1, save_best_only=True)
History = model.fit(traneX, trainY, validation_data=(testX, testY), callbacks=[stoppingData, checkpointer], batch_size=batchSize, epochs=epochCount)

prediction = model.predict(testX, batch_size=32)
print(sklearn.metrics.mean_squared_error(testY, prediction))
#print(History.history.keys())
N = numpy.arange(0, len(History.history["loss"]))
plt.style.use("ggplot")
plt.figure()
plt.plot(N, History.history["loss"], label="train_loss")
plt.plot(N, History.history["val_loss"], label="val_loss")
#plt.plot(N, History.history["accuracy"], label="train_acc")
#plt.plot(N, History.history["val_accuracy"], label="val_acc")
plt.title("Result")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.xlim(0, len(History.history["loss"]))
plt.legend()
plt.savefig("D:/Рабочий стол/Loss.png")
model.save("D:/Рабочий стол/Model.model")
