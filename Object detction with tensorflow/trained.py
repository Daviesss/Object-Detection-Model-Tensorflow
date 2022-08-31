import cv2
from cvzone.ClassificationModule import Classifier


cap = cv2.VideoCapture(0)
casacade = Classifier('/home/magnum/Desktop/computervision/scripts/Advanced_opencv/face_detecttion/mymodel/keras_model.h5','/home/magnum/Desktop/computervision/scripts/Advanced_opencv/face_detecttion/mymodel/labels.txt')
while True:
    ret,frame = cap.read()
    prediction,image_reading = casacade.getPrediction(frame,scale= 0.7)
    print(prediction)
    #writing a for loop  to loop on the image and check recognize the image
    # for (x,y,w,h) in image_reading:
    #     cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),thickness=2)
    #     #print(x,y,w,h)

    cv2.imshow('image_reading',frame)
    cv2.waitKey(1)