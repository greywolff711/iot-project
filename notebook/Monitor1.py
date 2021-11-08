import tensorflow as tf
model = tf.keras.models.load_model('model.h5')
import cv2
import numpy as np
cap = cv2.VideoCapture("test_video_2.mp4")
fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
out = cv2.VideoWriter('output_count.avi',fourcc, 20.0,(int(cap.get(3)),int(cap.get(4))))
ret, frame1 = cap.read()
prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)
hsv[...,1] = 255
i= 0
prediction_str = ""
repetitions = 0
up = 0
down = 0
no_move = 0
current_move = 0
initial = -1
while(cap.isOpened()):
    i+=1
    ret, frame2 = cap.read()
    next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
    image = cv2.resize(rgb, (64, 64))
    image = image.reshape((1,) + image.shape)
    image = image/255.0
    prediction = np.argmax(model.predict(image), axis=-1)[0]
    if prediction == 0:
        down +=1 
        if down == 3:
            if initial==-1:
                initial = 0
            if current_move == 2:
                repetitions+=1
            current_move = 0
        elif down > 0:
            up = 0
            no_move = 0
    elif prediction == 2:
        up += 1
        if up == 3 and initial != -1:
            current_move = 2
        elif up > 1:
            down = 0 
            no_move = 0
    else:
        no_move += 1
        if no_move == 15:
            current_move = 1
        elif no_move > 10:
            up = 0
            down = 0 
    font                   = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10,400)
    fontScale              = 1
    fontColor              = (255,255,255)
    lineType               = 5
    cv2.putText(frame2, "Repetitions: "+ str(repetitions),bottomLeftCornerOfText,font, fontScale,fontColor,lineType)
    out.write(frame2)
    prvs = next
print("Video Generated")
out.release()
cap.release()
cv2.destroyAllWindows()