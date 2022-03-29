import cv2

#import dataset
facedataset = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eyedataset = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

cap = cv2.VideoCapture(0)
while True:
    success,frame = cap.read()
    grayimg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faceCordinates = facedataset.detectMultiScale(grayimg, 1.1, 4)
    eyeCordinates = eyedataset.detectMultiScale(grayimg ,1.1, 4)

    for x,y,w,h in faceCordinates:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
        cv2.imshow('Win',frame)

    for ex,ey,ew,eh in eyeCordinates:
        cv2.rectangle(frame,(ex,ey),(ex+ew,ey+eh),(0,255,0),1)
        cv2.imshow('Win',frame)

    key = cv2.waitKey(1)
    if (key == ord('q')):
        break
cap.release()
cv2.destroyAllWin()
