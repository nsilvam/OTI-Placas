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

#tamaño del bounding box que se dibujara para ROI
upper_left = (941, 643)
bottom_right = (1900, 1080)

CONFIDENCE_THRESHOLD = 0.9
NMS_THRESHOLD = 0.4
COLORS = [(0, 0, 255), (0, 0, 255), (0, 0, 255), (255, 0, 0)]
tex = easyocr.Reader(['en'], gpu=True) #Vid1 ok Vid2 ok Vid3 ok Vid4 ok Vid5 ok Vid6 Vid7 ok con 0.4 falsa Vid8 ok Vid mal
class_names = []

with open("obj.names", "r") as f:
    class_names = [cname.strip() for cname in f.readlines()]

cap = cv2.VideoCapture("video2.mp4")
#cap.open('rtsp://admin:Hik12345@192.168.20.96:554/Streaming/channels/02/')

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
    #2. Rectangle marker
    r = cv2.rectangle(frame, upper_left, bottom_right, (100, 50, 200), 5)
    rect_img = frame[upper_left[1]: bottom_right[1], upper_left[0]: bottom_right[0]]
    cv2.imshow("image", rect_img)
    classes, scores, boxes = model.detect(rect_img, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)

    end = time.time()

    blob = cv2.dnn.blobFromImage(rect_img , 1/255 , (416,416) ,(0,0,0) , swapRB = True , crop = False)

    net.setInput(blob)
    output_layer_names = net.getUnconnectedOutLayersNames()
    layeroutputs = net.forward(output_layer_names)

    for (classid, score, box) in zip(classes, scores, boxes):
        color = COLORS[int(classid) % len(COLORS)]
        label = "%s : %.2f" % (class_names[0], score)
        cv2.rectangle(rect_img, box, color, 2)
        cv2.putText(rect_img, label, (box[0], box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        #Crop de placa
        x, y, w, h = box
        crop_img = rect_img[y:y + h, x:x + w]
        #Procesamiento

        gray = get_grayscale(crop_img)

        median = remove_noise(gray)

        gran = cv2.bilateralFilter(median, 9, 22, 22)

        gran = get_resize(gran)



        #median = remove_noise(gran)

        #th = thresholding(median)

        #erode = get_erode(gran)

        #close = get_close(erode)

        # Mostrar múltiples
        cv2.imshow("Crop", gran)

        res = tex.readtext(gran, allowlist = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ')
        print(res)
        if len(res) >= 1 and res[-1][2] > 0.6:
            text = res[-1][1].upper()

            if  len(text) >= 6:
                salida_0.append(text[0])
                salida_1.append(text[1])
                salida_2.append(text[2])
                salida_3.append(text[3])
                salida_4.append(text[4])
                salida_5.append(text[5])

                # print(c)
                print(salida_0)
                print(salida_1)
                print(salida_2)
                print(salida_3)
                print(salida_4)
                print(salida_5)

                valor_f = str(most_freq(salida_0)) + str(most_freq(salida_1)) + str(most_freq(salida_2)) + str(
                    most_freq(salida_3)) + str(most_freq(salida_4)) + str(most_freq(salida_5))
                print("Valor final", valor_f)
                cv2.putText(frame, valor_f, (box[0], box[1] + 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    fps = "FPS: %.2f " % (1 / (end - start + 1e-6))

    cv2.putText(frame, fps, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)


    cv2.imshow("output", frame)

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
