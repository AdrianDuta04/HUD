import os

from skimage import io , transform
from sklearn.metrics import accuracy_score
import numpy as np
from PIL import Image
from tensorflow import keras
import pandas as pd

model = keras.models.load_model("../trained_models/sign_recognition_model_last.h5")
model.summary()


# y_test = pd.read_csv('../../../data/Test.csv')
# 
# labels = y_test["ClassId"].values
# imgs = y_test["Path"].values
# 
# data=[]
# 
# for img in imgs:
#     image = Image.open(img)
#     image = image.resize((30,30))
#     data.append(np.array(image))
# 
# X_test=np.array(data)

data = []
labels=[]
rows = open("../../../data/Test.csv").read().strip().split("\n")[1:]


for (iterator,row) in enumerate(rows):
    (label, imagePath) = row.strip().split(",")[-2:]
    imagePath = os.path.sep.join(["../../../data/", imagePath])
    image = io.imread(imagePath)
    image = transform.resize(image, (32, 32))
    data.append(image)
    labels.append(int(label))

data = np.array(data)
labels=np.array(labels)


pred = model.predict_classes(data)
print(accuracy_score(labels, pred))

