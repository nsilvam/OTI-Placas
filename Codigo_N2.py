import cv2
import io, time
import numpy as np
import easyocr
from Funciones import *
from datetime import datetime
import requests
import json

import base64
from PIL import Image

salida_0=[]
salida_1=[]
salida_2=[]
salida_3=[]
salida_4=[]
salida_5=[]

valor_f=""

pos=0

f=False

tiempo_bbox=2668269365.4799159

CONFIDENCE_THRESHOLD = 0.6
NMS_THRESHOLD = 0.4 #Threshold para multiples bboxes que detectan el mismo objeto
COLORS = [(0, 0, 255), (0, 0, 255), (0, 0, 255), (255, 0, 0)]
tex = easyocr.Reader(['en'], gpu=True)
class_names = []

with open("obj.names", "r") as f:
    class_names = [cname.strip() for cname in f.readlines()]

cap = cv2.VideoCapture("video22.mp4")#04 y 01 ok return cv2.bilateralFilter(image, 13, 75, 75)
#cap = cv2.VideoCapture("rtmp://127.0.0.1:9999/placas_1")
#04,01,32,34 ok return cv2.bilateralFilter(image, 5, 35, 35)
#cap.open('rtsp://admin:Hik12345@192.168.20.96:554/Streaming/channels/02/')

net = cv2.dnn.readNet("custom-yolov4-tiny-detector_best.weights", "custom-yolov4-tiny-detector.cfg")

net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)

model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(416, 416), scale=1 / 255, swapRB=True)


ip_servidor = "192.168.52.232"
url = f"http://{ip_servidor}:7896/iot/json?k=beegons.661de995-3456-5b83-abc6-fcb0bb623af8&i=pta5"
headers = {"Content-Type": "application/json; charset=utf-8", "fiware-service": "smartcampus_uni", "fiware-servicepath": "/pta5"}

def encode_and_transmit_numpy_array_in_bytes(numpy_array: np.array) -> str:
    compressed_file = io.BytesIO()
    Image.fromarray(numpy_array).convert('RGB').save(compressed_file, format="JPEG")
    compressed_file.seek(0)
    return json.dumps(base64.b64encode(compressed_file.read()).decode())



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
        f=True
        color = COLORS[int(classid) % len(COLORS)]
        label = "%s : %.2f" % (class_names[0], score)
        cv2.rectangle(frame, box, color, 2)
        cv2.putText(frame, label, (box[0], box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (48, 66, 227), 2)
        #print(pos-box[0])
        if len(salida_0) != 0 and (pos-box[0])>500:
            salida_0 = []
            salida_1 = []
            salida_2 = []
            salida_3 = []
            salida_4 = []
            salida_5 = []
            f=False
            break

        #Crop de placa
        x, y, w, h = box
        crop_img = frame[y:y + h, x:x + w]

        #Procesamiento
        #gray = get_grayscale(crop_img)
        gran = get_resize(crop_img)
        gran = remove_noise(crop_img)

        # Mostrar múltiples
        #cv2.imshow("Crop", gran)

        res = tex.readtext(gran, allowlist = '0123456789ABCDEFGHIJKLMNOPQRSTUWXYZV', rotation_info=[0, 5, 10])
        #print('res: ', res)
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

    if (tiempo_frame-tiempo_bbox)>2 and f==True:
        file = open("LOG_Prueba.txt", "a")
        data = [valor_f + ", " + str(tiempo_bbox) + "\n"]
        file.writelines(data)
        file.close()
        #print("Bounding box", aux)
        #print("Valor final", valor_f)

        x1, y1, x2, y2 = aux[0], aux[1], aux[0]+aux[2], aux[1]+aux[3]
        dict_ = {
            'placa' : valor_f,
            'img' : f'data:image/jpeg;base64,{encode_and_transmit_numpy_array_in_bytes(cv2.resize(gran,(16*50,9*50)))[1:-1]}'
        }

        print(dict_)
        try:
            response = requests.post(url, headers=headers, json=dict_, timeout=1)
            print("Status Code", response.status_code)
            print("JSON Response ", response.json())
        except Exception as e:
            print(f"Error is {e}.")

        f=False

    #print(tiempo_frame - tiempo_bbox)
    end = datetime.utcnow()

    fps = 'FPS: %.2f ' % (1 / (end - start).total_seconds())

    cv2.putText(frame, fps, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)


    cv2.imshow("output", cv2.resize(frame, (1067,600)))

    #print('im here 2')

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()