import cv2 as cv

face_cap=cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')

capture = cv.VideoCapture(0)
while True:
    ret,frame=capture.read()
    if ret==False:
        break
    #gray_frame=cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces=face_cap.detectMultiScale(frame,scaleFactor=1.1,minNeighbors=5) # detect face
    for (x,y,w,h) in faces:
        cv.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2) # here it draw rectangle on face
    cv.imshow('Video Frame',frame)
    key_pressed=cv.waitKey(1) & 0xFF==ord('q')
    if key_pressed:
        break
capture.release()
cv.destroyAllWindows()
