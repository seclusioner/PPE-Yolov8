import cv2
import math
import os
from cvzone import putTextRect

from ultralytics import YOLO

videoPath = 'Inputs/ppe1.mp4'
modelPath = 'yolo_model/ppe.pt'

def checkVideo(videoPath):
    if not os.path.exists(videoPath):
        print('Video not found')
        exit()
    else:
        video = cv2.VideoCapture(videoPath)
        return video

# For video
video = checkVideo(videoPath)

# For Webcam
# video = cv2.VideoCapture(0)
# video.set(3, 640)
# video.set(4, 480)

model = YOLO(modelPath)
classNames = ['Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person', 'Safety Cone',
              'Safety Vest', 'machinery', 'vehicle']

# ==============================================
# Get H, W of videos (car.mp4)
frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Output results (video)
FPS = 30
out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), FPS, (frame_width, frame_height))
# ==============================================
    
while(True):
    success, frame = video.read()
    if not success:
        break
    
    results = model(frame, stream=True)
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1

            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            # Class Name
            cls = int(box.cls[0])
            
            currentClass = classNames[cls]
            if conf>0.5:
                if currentClass =='NO-Hardhat' or currentClass =='NO-Safety Vest' or currentClass == "NO-Mask":
                    myColor = (0, 0, 255)
                elif currentClass =='Hardhat' or currentClass =='Safety Vest' or currentClass == "Mask":
                    myColor =(0, 255, 0)
                else:
                    myColor = (255, 0, 0)

                putTextRect(frame, f'{classNames[cls]} {conf}',
                                    (max(0, x1), max(35, y1)), scale=1, thickness=1,colorB=myColor,
                                    colorT=(255,255,255),colorR=myColor, offset=5)
                cv2.rectangle(frame, (x1, y1), (x2, y2), myColor, 3)
   
    # ==============================================
    cv2.imshow('Frame', frame)
    cv2.waitKey(1)

    # Save results (.mp4)
    out.write(frame)
    
    # Break the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# close all windows
cv2.destroyAllWindows()

print('Over.')