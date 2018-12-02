import pypyodbc
import pyodbc
import cv2
import bglantı
import numpy as np

#bu alan sadece yuzu algılayıp videoda veya kemerada işareklemeye yarıyor

face_cascade=cv2.CascadeClassifier(r'C:\python\face\haarcascade_frontalface_default.xml')
bady_cascade=cv2.CascadeClassifier(r'C:\python\face\haarcascade_upperbody.xml')
cap=cv2.VideoCapture(0)
while True:
    ret, frame=cap.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    bady=face_cascade.detectMultiScale(gray,1.1,8)
    for(x,y,w,h) in bady:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)
        roi_gray=gray[y:y+h,x:x+w]
        roi_color=frame[y:y+h,x:x+w]
        faces=face_cascade.detectMultiScale(roi_gray)
        for (fx,fy,fw,fh)in faces:
            cv2.rectangle(roi_color, (fx, fy), (fx + fw, fy + fh), (255, 0, 0), 2)




    cv2.imshow('frame',frame)
    if cv2.waitKey(1)&0xFF==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
