import cv2


body_classifier = cv2.CascadeClassifier('haarcascade_fullbody.xml')
cap = cv2.VideoCapture('walking.avi')
while (True):   
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    bodies = body_classifier.detectMultiScale(gray, 1.1, 5)
    for (x, y, w, h) in bodies:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255, 0, 255), 4)
    cv2.imshow('frame', frame)
    if cv2.waitKey(25) == 32: #32 is the Space Key
        break

cap.release()
cv2.destroyAllWindows()
