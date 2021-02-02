import numpy as np
import pandas as pd
import cv2
import os
from sklearn.neighbors import KNeighborsClassifier as knn

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
path = "./face_data/"
class_id = 0
face_data = []
labels = []
names = {}


for fx in os.listdir(path):
	if fx.endswith(".npy"):
		names[class_id] = fx[:-4]
		print("loaded "+ fx)
		data_item = np.load(path+fx)
		face_data.append(data_item)
		target = class_id * np.ones((data_item.shape[0],))
		labels.append(target)
		class_id += 1

face_dataset = np.concatenate(face_data, axis = 0)
face_labels = np.concatenate(labels, axis = 0)
print(face_dataset.shape)
print(face_labels.shape)

# Train
model = knn(n_neighbors = 5)
X_train = face_dataset
Y_train = face_labels
model.fit(X_train, Y_train)
print("Training Completed")

cap = cv2.VideoCapture(0)

while True:
	ret, frame = cap.read()
	if ret == False:
		continue
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(frame, 1.3, 4)

	for (x,y,w,h) in faces:
		off = 10
		face_section = frame[y-off:y+h+off , x-off:x+w+off]
		face_section = cv2.resize(face_section, (100,100))
		X_test = []
		X_test.append(face_section.flatten())
		out = model.predict(X_test)
		pred_name = names[int(out)]
		font = cv2.FONT_HERSHEY_SIMPLEX 
		cv2.putText(frame, pred_name, (x,y-10), font, 1, (255, 0, 0), 2, cv2.LINE_AA)
		cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 2)

	cv2.imshow("Faces", frame)

	key = cv2.waitKey(1) & 0xFF
	if key == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()