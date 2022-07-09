import cv2
import cvzone
import winsound
from threading import Thread
from cvzone.FaceMeshModule import FaceMeshDetector
from cvzone.PlotModule import LivePlot

def sound_alarm():
    '''Play a 3s beep sound'''
    winsound.Beep(3300, 3000)
    
cap = cv2.VideoCapture(0)
detector = FaceMeshDetector(maxFaces=1)
plotY = LivePlot(640, 480, [25, 50], invert=True)

EYE_AR_THRESH = 37
EYE_AR_CONSEC_FRAMES = 40
COUNTER = 0
ALARM_ON = False

ratioList = []
counter = 0
# blinkCoutnter = 0
color = (255, 0, 255)
green = (0, 255, 0)
red = (0, 0, 255)

while True:
    success, img = cap.read()
    img, faces = detector.findFaceMesh(img, draw=False)
    
    if faces:
        face = faces[0]
        ''' Outline points of the eyes'''
        # for i in range(len(face)):
            # if i in (159, 23, 130, 173, 385, 374, 414, 249):
            #     cv2.putText(img, f'{i}', (face[i][0], face[i][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0, 0, 255), 1)
    
        leftUp = face[159]
        leftDown = face[23]
        leftLeft = face[130]
        leftRight = face[173]

        rightUp = face[385]
        rightDown = face[374]
        rightLeft = face[414]
        rightRight = face[249]

        lengthVer1, _ = detector.findDistance(leftUp, leftDown)
        lengthtHor1, _ = detector.findDistance(leftLeft, leftRight)
        lengthVer2, _ = detector.findDistance(rightUp, rightDown)
        lengthHor2, _ = detector.findDistance(rightLeft, rightRight)

        lengthVer = (lengthVer1 + lengthVer2) / 2
        lengthHor = (lengthtHor1 + lengthHor2) / 2
        
        ratio = int((lengthVer / lengthHor) * 100)
        ratioList.append(ratio)
        if len(ratioList) > 5:
            ratioList.pop(0)
        ratioAvg = sum(ratioList) / len(ratioList)
        
        '''
        If ratioAvg drops below the threshold, increase the counter and start alarm once counter reaches the threshold number of frames
        otherwise, the eye aspect ratio is not below the blink threshold, so reset the counter and alarm
        '''
        if ratioAvg < EYE_AR_THRESH:
            COUNTER += 1
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                if not ALARM_ON:
                    ALARM_ON = True
                    t = Thread(target=sound_alarm)
                    t.deamon = True
                    t.start()
                cv2.putText(img, "DROWSINESS ALERT!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            COUNTER = 0
            ALARM_ON = False

        # cvzone.putTextRect(img, f'Blink Count: {blinkCounter}', (50, 100), colorR=color)

        imgPlot = plotY.update(ratioAvg, green if ratioAvg > EYE_AR_THRESH else red)
        # img = cv2.resize(img, (640, 360))
        imgStack = cvzone.stackImages([img, imgPlot], 2, 1)
    else:
        imgStack = cvzone.stackImages([img, img], 2, 1)

    cv2.imshow("Drowsiness Detector", imgStack)
    if cv2.waitKey(1) == 27:
        break 

cap.release()
cv2.destroyAllWindows()