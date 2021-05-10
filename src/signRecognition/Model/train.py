from skimage import transform
from skimage import io
import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow import keras
from time import time
import matplotlib.pyplot as plt

from tensorflow.python.keras.utils.np_utils import to_categorical

NUM_EPOCHS = 100
BATCH_SIZE = 512

labelNames = open("../../../data/signnames.csv").read().strip().split("\n")[1:]
labelNames = [l.split(",")[1] for l in labelNames]


model = keras.models.load_model("../compiled_models/sign_recognition_model_last.h5")
model.summary()

data_for_training = []
labels_for_training = []
rows_for_training = open("../../../data/Train.csv").read().strip().split("\n")[1:]


for (iterator,row) in enumerate(rows_for_training):
    (label, imagePath) = row.strip().split(",")[-2:]
    # derive the full path to the image file and load it
    imagePath = os.path.sep.join(["../../../data/", imagePath])
    image = io.imread(imagePath)
    image = transform.resize(image, (32, 32))
    data_for_training.append(image)
    labels_for_training.append(int(label))

data_for_training = np.array(data_for_training)
labels_for_training = np.array(labels_for_training)

print(data_for_training.shape, labels_for_training.shape)

X_train, X_test, y_train, y_test = train_test_split(data_for_training, labels_for_training, test_size=0.2, random_state=42)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


y_train = to_categorical(y_train, 43)
y_test = to_categorical(y_test, 43)
history = model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS, validation_data=(X_test, y_test))
now = time()
model.save("../trained_models/sign_recognition_model"+str(int(now))+".h5")
model.save("../trained_models/old_models/sign_recognition_model"+str(int(now))+".h5")

plt.figure(0)
plt.plot(history.history['accuracy'], label='training accuracy')
plt.plot(history.history['val_accuracy'], label='val accuracy')
plt.title('Accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()
plt.show()

plt.figure(1)
plt.plot(history.history['loss'], label='training loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.title('Loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.show()


