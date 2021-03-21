import cv2
import numpy as np
import dlib


cap = cv2.VideoCapture(0) # camera first was choosed

detector = dlib.get_frontal_face_detector()

predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") # reads necessary data from shape predictor dat file

while True:
    _, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)

    for face in faces: # detects faces edge
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()

        #cv2.rectangle(frame, (x1,y1), (x2,y2), (0, 255, 0), 3)

        landmarks = predictor(gray, face)

        for n in range(0,68):  # works with 69 face landmarks

            # eyes detection
            
            #if 36 <= n <= 47:
                
                #x = landmarks.part(n).x
                #y = landmarks.part(n).y

                #cv2.circle(frame, (x,y),3 , (255, 255, 0), -1)

            ############## mouth detection
            
            #if 48 <= n <= 67:
                
                #x = landmarks.part(n).x
                #y = landmarks.part(n).y

                #cv2.circle(frame, (x,y),3 , (255, 255, 0), -1)


            ######### detects fully 67 face landmarks
                
            x = landmarks.part(n).x
            y = landmarks.part(n).y

            cv2.circle(frame, (x,y),3 , (255, 255, 0), -1)
            

    cv2.imshow("Face", frame)

    key = cv2.waitKey(1)

    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
    
