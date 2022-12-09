import numpy as np
import os
import cv2
import pyautogui
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *

haar = cv2.CascadeClassifier("haarcascade_eye.xml")
webcam = cv2.VideoCapture(0)

def reducer(inp):
  min_, max_ = inp.min(), inp.max()
  return (inp - min_) / (max_ - min_)
  
def scan(img=(32, 32)):
  _, vid = webcam.read()
  grayScale = cv2.cvtColor(vid, cv2.COLOR_BGR2GRAY)
  sections = haar.detectMultiScale(grayScale, 1.3, 10)
  if len(sections) == 2:
    eyeSection = []
    for box in sections:
      x, y, w, h = box
      eye = vid[y:y + h, x:x + w]
      eye = cv2.resize(eye, img)
      eye = reducer(eye)
      eye = eye[10:-10, 5:-5]
      eyeSection.append(eye)
    return (np.hstack(eyeSection) * 255).astype(np.uint8)
  else:
    return None
    
#width and height are monitor specific

width = int(input("Enter the width of the screen..."))
height = int(input("Enter the height of the screen..."))
# width, height = 1366, 768

root = input("Enter the root path of the directory for the pictures...")
# root = "C:\\Users\\tuzoe\\Documents\\Projects\\DS340\\final\\eye_mouse_movement-master\\pics\\"

filepaths = os.listdir(root)
X, Y = [], []
for filepath in filepaths:
  x, y, _ = filepath.split(' ')
  x = float(x) / width
  y = float(y) / height
  print(root +filepath)
  X.append(cv2.imread(root + filepath))
  Y.append([x, y])

# print(X)
X = np.array(X) / 255.0
Y = np.array(Y)
print (X.shape, Y.shape)

model = Sequential()
model.add(Conv2D(32, 3, 2, activation = 'relu', input_shape = (12, 44, 3)))
model.add(Conv2D(64, 2, 2, activation = 'relu'))
model.add(Flatten())
model.add(Dense(32, activation = 'relu'))
model.add(Dense(2, activation = 'sigmoid'))
model.compile(optimizer = "adam", loss = "mean_squared_error")
model.summary()

epochs = 200
for epoch in range(epochs):
  model.fit(X, Y, batch_size = 32)
  
while True:
  eyeSection = scan()
  if not eyeSection is None:
    eyeSection = np.expand_dims(eyeSection / 255.0, axis = 0)
    x, y = model.predict(eyeSection)[0]
    print('Eye location: ')
    print(x)
    print(y)
    pyautogui.moveTo(x * width, y * height)

