import numpy as np
import pandas as pd
import cv2

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
path = './face_data/'
loop = True
data = []

cnt = 0
name = input("Name: ")

key = 0

while True and key != ord('q'):
	
	ret, frame = cap.read()
	if ret == False:
		continue

	faces = face_cascade.detectMultiScale(frame, 1.3, 4)

	for (x,y,w,h) in faces:
		cv2.rectangle(frame, (x,y), (x+w, y+h), (255, 0, 0), 2)

	cv2.imshow("Frame", frame)
	sorted(faces, key= lambda f: f[2] * f[3])

	if len(faces) > 0:
		x,y,w,h = faces[-1]
		off = 10
		face_section = frame[y-off:y+off+h , x-off:x+off+w]
		face_section = cv2.resize(face_section, (100,100))
		cnt += 1
		if cnt % 20 == 0:
			data.append(face_section)
			print(len(data))
		cv2.imshow("Face_section", face_section)



	key = cv2.waitKey(1) & 0xFF
	if key == ord('q'):
		break

data = np.array(data)
data = np.reshape(data, (data.shape[0], -1))
print(data.shape)

np.save(path+name+".npy", data)
print("Saved")

cap.release()
cv2.destroyAllWindows()