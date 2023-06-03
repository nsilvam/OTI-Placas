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

pos=0

tiempo_bbox=2668269365.4799159

CONFIDENCE_THRESHOLD = 0.6
NMS_THRESHOLD = 0.4
COLORS = [(0, 0, 255), (0, 0, 255), (0, 0, 255), (255, 0, 0)]
tex = easyocr.Reader(['en'], gpu=True)
class_names = []

with open("obj.names", "r") as f:
    class_names = [cname.strip() for cname in f.readlines()]

#cap = cv2.VideoCapture("Proh1.mp4")#04 y 01 ok return cv2.bilateralFilter(image, 13, 75, 75)
#cap = cv2.VideoCapture("rtmp://127.0.0.1:10000/placas_1")
#04,01,32,34 ok return cv2.bilateralFilter(image, 5, 35, 35)
cap = cv2.VideoCapture('rtsp://admin:Hik12345@192.168.30.35:554/Streaming/channels/101')
#cap.open('rtsp://admin:Hik12345@192.168.30.35:554/Streaming/channels/02/')
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

    classes, scores, boxes = model.detect(frame, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    blob = cv2.dnn.blobFromImage(frame, 1/255, (416,416), (0,0,0), swapRB = True , crop = False)

    net.setInput(blob)
    output_layer_names = net.getUnconnectedOutLayersNames()
    layeroutputs = net.forward(output_layer_names)
    tiempo_frame=time.time()

    for (classid, score, box) in zip(classes, scores, boxes):
        f = True
        color = COLORS[int(classid) % len(COLORS)]
        label = "%s : %.2f" % (class_names[0], score)
        cv2.rectangle(frame, box, color, 2)
        cv2.putText(frame, label, (box[0], box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (48, 66, 227), 2)
        #print(pos-box[0])
        if len(salida_0) != 0 and (pos-box[0])>450:
            salida_0 = []
            salida_1 = []
            salida_2 = []
            salida_3 = []
            salida_4 = []
            salida_5 = []
            f = False
            break

        #Crop de placa
        x, y, w, h = box
        crop_img = frame[y:y + h, x:x + w]

        #Procesamiento
        gray = get_grayscale(crop_img)
        gran = get_resize(gray)
        #gran = remove_noise(gran)

        gran = cv2.GaussianBlur(gran,(17,17),0)
        #gran = get_erode(gran)

        # Mostrar múltiples
        cv2.imshow("Crop", cv2.resize(gran, (533,300)))

        res = tex.readtext(gran, allowlist = '0123456789ABCDEFGHIJKLMNOPQRSTUWXYZV', rotation_info=[0, 5, 10])
        print('res: ', res)
        for n in range(len(res)):
            if res[-(n+1)][2] > 0.15:
                text = res[-(n+1)][1]

                if len(text) >= 6:
                    salida_0.append(text[0])
                    salida_1.append(text[1])
                    salida_2.append(text[2])
                    salida_3.append(text[3])
                    salida_4.append(text[4])
                    salida_5.append(text[5])

                    #print(salida_0)
                    #print(salida_1)
                    #print(salida_2)
                    #print(salida_3)
                    #print(salida_4)
                    #print(salida_5)



                    valor_f = str(most_freq(salida_0)) + str(most_freq(salida_1)) + str(most_freq(salida_2)) + str(
                        most_freq(salida_3)) + str(most_freq(salida_4)) + str(most_freq(salida_5))
                    #print("Valor", valor_f)
                    cv2.putText(frame, valor_f, (box[0], box[1] + 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (47, 47, 155), 2)
        aux=box
        pos = box[0]
        tiempo_bbox = time.time()

    if (tiempo_frame - tiempo_bbox) > 2 and f == True:
        now = datetime.now()
        file = open("LOG_Prueba.txt", "a")
        data = [valor_f + ", " + str(now.strftime("%Y-%m-%d %H:%M:%S")) + "\n"]
        file.writelines(data)
        file.close()
        #print("Bounding box", aux)
        print("Valor final", valor_f)
        f = False

    #print(tiempo_frame - tiempo_bbox)0
    end = datetime.utcnow()

    fps = 'FPS: %.2f ' % (1 / (end - start).total_seconds())

    cv2.putText(frame, fps, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

    #cv2.imshow("output", cv2.resize(frame, (1067,600)))
    cv2.imshow("output", cv2.resize(frame, (1280, 720)))

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()











