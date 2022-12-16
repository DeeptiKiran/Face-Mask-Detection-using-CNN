import os
from keras.preprocessing import image
import cv2
import random
import numpy as np
from sklearn.model_selection import train_test_split
from keras.applications.vgg16 import VGG16
from keras import sequential
from keras.layers import Dense

categories = ['with_mask', 'without_mask']

def detect_face_mask(img):
    y_pred=model.predict_classes(img.reshape(1, 224, 224, 3))
    return y_pred[0][0]

def draw_lable(img, text, pos, bg_color):
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_COMPLEX, 1, cv2.FILLED)
    end_x = pos[0] + text_size[0][0] + 2
    end_y = pos[1] + text_size[0][1] - 2
    cv2.rectangle(img, pos, (end_x, end_y), bg_color, cv2.FILLED)
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0),1,cv2.LINE_AA)

data = []
for category in categories:
    path = os.path.join('data', category)
    label = categories.index(category)

    for file in os.listdir(path):
        img_path = os.path.join('path', file)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (224, 224))
        data.append([img, label])

random.shuffle(data)

x = []
y = []
for features, label in data:
    x.append(features)
    y.append(label)

x = np.array(x)
y = np.array(y)
x = x/255

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

vgg = VGG16()
model = sequential()

for layer in vgg.layers[:-1]:
    model.add(layer)

for layer in model.layer:
    layer.trainable = False

model.add(Dense(1,activation = 'sigmoid'))
model.compile(optimizer = 'Adam',loss = 'binay_crossentropy', metrics = ['accuracy'])
model.fit(x_train, y_train, epochs=5, validation_data = (x_test, y_test))

cap = cv2.VidioCapture(0)
while True:
    cat, img = cap.read()
    img=cv2.resize(frame, (224, 224))
    y_pred=detect_face_mask(img)
    if y_pred == 0:
        draw_lable(frame, 'MASK', (30, 30), (0, 255, 0))
    else:
        draw_lable(frame, 'NO MASK', (30, 30), (0, 0, 255))

    cv2.imshow('window', frame)
    if cv2.waitKey(1) & 0xFF == ord('x'):
        break
cv2.destroyAllWindows()





