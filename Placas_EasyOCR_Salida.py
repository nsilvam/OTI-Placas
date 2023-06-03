import cv2
import time
import numpy as np
import easyocr
from Funciones import *
from datetime import datetime

salida_0=[]
salida_1=[]
salida_2=[]
salida_3=[]
salida_4=[]
salida_5=[]

valor_f=""

pts = np.array([[535, 440], [1, 862],
                [457, 1076], [1252, 1070],[1238, 514]],
               np.int32)


pts = pts.reshape((-1, 1, 2))
isClosed = True
# Blue color in BGR
color = (10, 255, 170)
# Line thickness of 2 px
tic = 2

CONFIDENCE_THRESHOLD = 0.6
NMS_THRESHOLD = 0.4
COLORS = [(0, 0, 255), (0, 0, 255), (0, 0, 255), (255, 0, 0)]
tex = easyocr.Reader(['en'], gpu=True) #Vid1 ok Vid2 ok Vid3 ok Vid4 ok Vid5 ok Vid6 Vid7 ok con 0.4 falsa Vid8 ok Vid mal
class_names = []

with open("obj.names", "r") as f:
    class_names = [cname.strip() for cname in f.readlines()]

cap = cv2.VideoCapture("video22.mp4 ")
#cap = cv2.VideoCapture("rtmp://127.0.0.1:10000/placas2/")
#cap.open('rtsp://admin:Hik12345@192.168.20.96:554/Streaming/channels/02/')
#cap.open('rtmp://127.0.0.1:10000/placas1/')

net = cv2.dnn.readNet("custom-yolov4-tiny-detector_best.weights", "custom-yolov4-tiny-detector.cfg")

net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)

model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(416, 416), scale=1 / 255, swapRB=True)


while cv2.waitKey(1) < 1:
    (grabbed, frame) = cap.read()
    if not grabbed:
        exit()
    height, width, _ = frame.shape
    start = datetime.utcnow()
    aux = np.copy(frame)
    r = cv2.polylines(aux, [pts],isClosed, color, tic)

    classes, scores, boxes = model.detect(frame, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    blob = cv2.dnn.blobFromImage(frame , 1/255 , (416,416) ,(0,0,0) , swapRB = True , crop = False)

    net.setInput(blob)
    output_layer_names = net.getUnconnectedOutLayersNames()
    layeroutputs = net.forward(output_layer_names)

    for (classid, score, box) in zip(classes, scores, boxes):
        color = COLORS[int(classid) % len(COLORS)]
        label = "%s : %.2f" % (class_names[0], score)
        cv2.rectangle(aux, box, color, 2)
        cv2.putText(aux, label, (box[0], box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (48, 66, 227), 2)

        pos = cv2.pointPolygonTest(pts, np.float32((box[0], box[1])), False)

        if pos == -1:
            if (len(salida_0) != 0):
                time = datetime.utcnow()
                file = open("LOG_Prueba.txt", "a")
                data = [valor_f + ", " + str(time) + "\n"]
                file.writelines(data)
                file.close()
            salida_0 = []
            salida_1 = []
            salida_2 = []
            salida_3 = []
            salida_4 = []
            salida_5 = []
            break

        #Crop de placa
        x, y, w, h = box
        crop_img = frame[y:y + h, x:x + w]

        #Procesamiento
        gray = get_grayscale(crop_img)
        gran = get_resize(gray)

        # Mostrar múltiples
        cv2.imshow("Crop", gran)

        res = tex.readtext(gran, allowlist = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ', rotation_info=[0, 2, 5])
        print('res: ', res)
        if len(res) >= 1 and res[-1][2] > 0.15:
            for n in range(len(res)):
                text = res[-(n+1)][1]

                if  len(text) >= 6:
                    salida_0.append(text[0])
                    salida_1.append(text[1])
                    salida_2.append(text[2])
                    salida_3.append(text[3])
                    salida_4.append(text[4])
                    salida_5.append(text[5])

                    print(salida_0)
                    print(salida_1)
                    print(salida_2)
                    print(salida_3)
                    print(salida_4)
                    print(salida_5)


                valor_f = str(most_freq(salida_0)) + str(most_freq(salida_1)) + str(most_freq(salida_2)) + str(most_freq(salida_3)) + str(most_freq(salida_4)) + str(most_freq(salida_5))
                print("Valor final", valor_f)
                cv2.putText(aux, valor_f, (box[0], box[1] + 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (47, 47, 155), 2)
    end = datetime.utcnow()

    fps = 'FPS: %.2f ' % (1 / (end - start).total_seconds())

    cv2.putText(aux, fps, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

    cv2.imshow("output", aux)

    #print('im here 2')

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()