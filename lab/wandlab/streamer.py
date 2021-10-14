# -*- encoding: utf-8 -*-
#-------------------------------------------------#
# Date created          : 2020. 8. 18.
# Date last modified    : 2020. 8. 19.
# Author                : chamadams@gmail.com
# Site                  : http://wandlab.com
# License               : GNU General Public License(GPL) 2.0
# Version               : 0.1.0
# Python Version        : 3.6+
#-------------------------------------------------#

import time
import cv2
import imutils
import platform
import threading
from threading import Thread
import numpy as np
import mediapipe as mp
from queue import Queue
from tensorflow.keras.models import load_model
from EyesBlink import eyesBlink
from HeadAngle import calcAngles, drawLine, lineLength
from Eyes import *


class Streamer :
    
    def __init__(self ):
        
        if cv2.ocl.haveOpenCL() :
            cv2.ocl.setUseOpenCL(True)
        print('[wandlab] ', 'OpenCL : ', cv2.ocl.haveOpenCL())

        self.capture = None
        self.thread = None
        self.width = 640
        self.height = 360
        self.stat = True
        self.current_time = time.time()
        self.preview_time = time.time()
        self.sec = 0
        self.Q = Queue(maxsize=128)
        self.started = False
        #############
        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(max_num_faces=2, min_detection_confidence=0.7)
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=2)
        self.model = load_model('mp_hand_gesture')
        f = open('gesture.names', 'r')
        self.classNames = f.read().split('\n')
        f.close()
        self.threading_flag1, self.threading_flag2 = False, False
        self.t1, self.t2 = 0, 0
        self.motionName = ''
        self.className = ''
        #############

    def run(self, src = 1 ) :
        
        self.stop()
    
        if platform.system() == 'Windows' :        
            self.capture = cv2.VideoCapture(src, cv2.CAP_DSHOW )
        
        else :
            self.capture = cv2.VideoCapture(scr)

        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        
        if self.thread is None :
            self.thread = Thread(target=self.update, args=())
            self.thread.daemon = False
            self.thread.start()
        
        self.started = True
    
    def stop(self):
        
        self.started = False
        
        if self.capture is not None :
            
            self.capture.release()
            self.clear()
            
    def update(self):
                    
        while True:

            if self.started :
                (grabbed, frame) = self.capture.read()
                
                if grabbed : 
                    self.Q.put(frame)
                          
    def clear(self):
        
        with self.Q.mutex:
            self.Q.queue.clear()
            
    def read(self):

        return self.Q.get()

    def blank(self):
        
        return np.ones(shape=[self.height, self.width, 3], dtype=np.uint8)

    #############
    def warning(self):
        print("warning")
        self.threading_flag1 = False

    def motion(self):
        print(self.motionName)
        self.threading_flag2 = False

    def print_message(self, flag, func, threading_flag, t, tmp=""):
        if threading_flag == False:
            if flag == True:
                threading_flag = flag
                if tmp != "": self.motionName = tmp
                t = threading.Timer(1, func)
                t.start()
        else:
            if flag == False:
                if t != 0:
                    t.cancel()
                    t = 0
                    threading_flag = flag
        return threading_flag, t
    #############

    def bytescode(self):
        
        if not self.capture.isOpened():
            
            frame = self.blank()

        else :
            
            frame = imutils.resize(self.read(), width=int(self.width) )
##############
            frame = cv2.flip(frame, 1)

            Left_hand = frame[10:710, 10:400]
            Right_hand = frame[10:710, 880:1270]
            frame_hands = cv2.hconcat([Left_hand, Right_hand])

            # mp facial landmark detection
            imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results_face = self.faceMesh.process(imgRGB)

            # Get hand landmark prediction
            imgRGB_test = cv2.cvtColor(frame_hands, cv2.COLOR_BGR2RGB)
            results_hand = self.hands.process(imgRGB_test)
            x, y, c = frame_hands.shape

            """ Head Detection """
            try:
                # Detect how many faces in a webcam
                if len(results_face.multi_face_landmarks) == 1:
                    landmarks = results_face.multi_face_landmarks[0].landmark

                    """ Head angle detection in relation to the camera """
                    roll, yaw, pitch = calcAngles(frame, landmarks, visualize=False)

                    # drawing values of roll, yaw and pitch angles
                    cv2.putText(frame, "ROLL    : " + str(int(roll)), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                                (0, 0, 0), 2)
                    cv2.putText(frame, "YAW     : " + str(int(yaw)), (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0),
                                2)
                    cv2.putText(frame, "PITCH   : " + str(int(pitch)), (50, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                                (0, 0, 0), 2)

                    """ Blink detection """

                    cv2.putText(frame, "BLINKING: " + eyesBlink(landmarks), (50, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                                (0, 0, 0), 2)
                    # Set blinking to an error motion
                    if eyesBlink(landmarks) != 'none':
                        error_flag_x, error_flag_y = 50, 50

                    """ Extract eyes """
                    if landmarks[464] or landmarks[33]:
                        eyes = Eyes(frame, landmarks)
                        # Get error values
                        if eyesBlink(landmarks) == 'none':
                            error_flag_x, error_flag_y = eyes.getEyes(show=False)

                        # Drawing values of errors
                        cv2.putText(frame, "ERROR_X: " + str(error_flag_x), (50, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                                    (0, 0, 0), 2)
                        cv2.putText(frame, "ERROR_Y: " + str(error_flag_y), (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                                    (0, 0, 0), 2)

                        # Drawing and printing values of a warning message
                        if error_flag_x > 28 or error_flag_y > 25 or pitch < -25 or abs(yaw) > 25:
                            self.threading_flag1, self.t1 = self.print_message(True, self.warning, self.threading_flag1, self.t1)
                            cv2.putText(frame, "WARNING", (300, 400), cv2.FONT_HERSHEY_DUPLEX, 5, (0, 0, 255), 5)
                        else:  # Only normal input
                            self.threading_flag1, self.t1 = self.print_message(False, self.warning, self.threading_flag1, self.t1)
                else:
                    # Drawing and printing values of a warning message
                    cv2.putText(frame, " FACES", (300, 400), cv2.FONT_HERSHEY_DUPLEX, 5, (0, 0, 255), 5)
                    self.threading_flag1, self.t1 = self.print_message(True, self.warning, self.threading_flag1, self.t1)
            except cv2.error:  # Error from cv2.resize()
                pass
            except TypeError:  # Error from len()
                # Drawing and printing values of a warning message
                cv2.putText(frame, "NOFACE", (300, 400), cv2.FONT_HERSHEY_DUPLEX, 5, (0, 0, 255), 5)
                self.threading_flag1, self.t1 = self.print_message(True, self.warning, self.threading_flag1, self.t1)
                pass

            """ Hands Detection """
            # post process the result
            if results_hand.multi_hand_landmarks:
                landmarks = []
                for handLandmark in results_hand.multi_hand_landmarks:
                    for lm in handLandmark.landmark:
                        lmx = int(lm.x * x)
                        lmy = int(lm.y * y)
                        landmarks.append([lmx, lmy])

                    # Drawing landmarks on frames
                    self.mpDraw.draw_landmarks(frame_hands, handLandmark, self.mpHands.HAND_CONNECTIONS)

                    # Predict gesture
                    prediction = self.model.predict([landmarks])
                    classID = np.argmax(prediction)
                    # Filter motions not to need out
                    if classID in [0, 1, 4, 6, 9]:
                        continue
                    elif classID in [5, 7]:
                        classID = 5
                    self.className = self.classNames[classID]

                    # show the prediction on the frame
                    self.threading_flag2, self.t2 = self.print_message(True, self.motion, self.threading_flag2, self.t2, self.className)
                    cv2.putText(frame, self.className, (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                                1, (0, 0, 255), 2, cv2.LINE_AA, )
            else:
                self.threading_flag2, self.t2 = self.print_message(False, self.motion, self.threading_flag2, self.t2, self.className)

            cv2.rectangle(frame, (880, 10), (1270, 710), (0, 0, 255), 2)
            cv2.rectangle(frame, (10, 10), (400, 710), (0, 0, 255), 2)
            ##############

            if self.stat :  
                cv2.rectangle( frame, (0,0), (120,30), (0,0,0), -1)
                fps = 'FPS : ' + str(self.fps())
                cv2.putText(frame, fps, (10,20), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 1, cv2.LINE_AA)

        return cv2.imencode('.jpg', frame )[1].tobytes()
    
    def fps(self):
        
        self.current_time = time.time()
        self.sec = self.current_time - self.preview_time
        self.preview_time = self.current_time
        
        if self.sec > 0 :
            fps = round(1/(self.sec),1)
            
        else :
            fps = 1
            
        return fps
                   
    def __exit__(self) :
        print( '* streamer class exit')
        self.capture.release()