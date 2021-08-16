# -*- coding: utf-8 -*-

import cv2

cascade_src = 'cars.xml'

video_src = 'video.avi'

cap = cv2.VideoCapture(video_src)

video_width = int(cap.get(3))

video_height = int(cap.get(4))

fourcc = cv2.VideoWriter_fourcc(*'MJPG')

output_src = cv2.VideoWriter("output.avi", fourcc, 30,(video_width,video_height))

car_cascade = cv2.CascadeClassifier(cascade_src)


while True:
    ret, img = cap.read()
    if (type(img) == type(None)):
        break
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    cars = car_cascade.detectMultiScale(gray, 1.1, 2)

    for (x,y,w,h) in cars:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,255),2)
    
    cv2.imshow('video', img)

    output_src.write(img)
    
    if cv2.waitKey(1) == ord('q'):
        break

output_src.release()
cap.release()
cv2.destroyAllWindows()
