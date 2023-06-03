import cv2
import time
import numpy as np
import easyocr
from Funciones import *


salida_0=[]
salida_1=[]
salida_2=[]
salida_3=[]
salida_4=[]
salida_5=[]

valor_f=""

CONFIDENCE_THRESHOLD = 0.3
NMS_THRESHOLD = 0.4
COLORS = [(0, 0, 255), (0, 0, 255), (0, 0, 255), (255, 0, 0)]
tex = easyocr.Reader(['en'], gpu=False) #Vid1 ok Vid2 ok Vid3 ok Vid4 ok Vid5 ok Vid6 Vid7 ok con 0.4 falsa Vid8 ok Vid mal
class_names = []

with open("obj.names", "r") as f:
    class_names = [cname.strip() for cname in f.readlines()]

cap = cv2.VideoCapture()
cap.open('rtsp://admin:Hik12345@192.168.20.212:554/Streaming/channels/01/')

net = cv2.dnn.readNet("custom-yolov4-tiny-detector_best.weights", "custom-yolov4-tiny-detector.cfg")

#net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
#net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)

model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(416, 416), scale=1 / 255, swapRB=True)


while cv2.waitKey(1) < 1:

    (grabbed, frame) = cap.read()
    if not grabbed:
        exit()
    height, width, _ = frame.shape
    start = time.time()

    classes, scores, boxes = model.detect(frame, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    end = time.time()

    blob = cv2.dnn.blobFromImage(frame , 1/255 , (416,416) ,(0,0,0) , swapRB = True , crop = False)

    net.setInput(blob)
    output_layer_names = net.getUnconnectedOutLayersNames()
    layeroutputs = net.forward(output_layer_names)
    #print('im here 1')
    for (classid, score, box) in zip(classes, scores, boxes):
        color = COLORS[int(classid) % len(COLORS)]
        label = "%s : %.2f" % (class_names[0], score)
        cv2.rectangle(frame, box, color, 2)
        cv2.putText(frame, label, (box[0], box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    fps = 'FPS: %.2f ' % (1 / (end - start + 1e-6))

    cv2.putText(frame, fps, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)


    cv2.imshow("output", frame)

    #print('im here 2')

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
