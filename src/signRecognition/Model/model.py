from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D , MaxPooling2D, Dense , Flatten , Dropout , BatchNormalization
from time import time

height = 30
width = 30
depth = 3
model = Sequential()
shape =(height , width ,depth)

model.add(Conv2D(16, (5, 5), padding="same",activation="relu",input_shape=shape))
model.add(BatchNormalization(axis=-1))
model.add(MaxPooling2D(pool_size=(2, 2),strides=(2, 2)))


model.add(Conv2D(32, (3, 3),padding="same", activation="relu"))
model.add(BatchNormalization(axis=-1))
model.add(Conv2D(32, (3, 3), padding="same",activation="relu"))
model.add(BatchNormalization(axis=-1))
model.add(MaxPooling2D(pool_size=(2, 2),strides=(2, 2)))


model.add(Dropout(rate=0.25))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(BatchNormalization(axis=-1))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(BatchNormalization(axis=-1))
model.add(MaxPooling2D(pool_size=(2, 2),strides=(2, 2)))
model.add(Dropout(rate=0.25))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(BatchNormalization(axis=-1))
model.add(Dropout(rate=0.5))

model.add(Dense(43, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

now = time()
model.save("../compiled_models/old_models/sign_recognition_model_last.h5")
model.save("../compiled_models/sign_recognition_model"+str(int(now))+".h5")

model.summary()

