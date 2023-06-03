import cv2
import time
import pytesseract
from Funciones import *

salida=[]
salida_0=[]
salida_1=[]
salida_2=[]
salida_3=[]
salida_4=[]
salida_5=[]
out=[]
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

CONFIDENCE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.4
COLORS = [(0, 0, 255), (0, 0, 255), (0, 0, 255), (255, 0, 0)]

class_names = []

with open("obj.names", "r") as f:
    class_names = [cname.strip() for cname in f.readlines()]

#cap = cv2.VideoCapture("video1.mp4")
cap = cv2.VideoCapture("video9.mp4")
#cap = cv2.VideoCapture("video5.mp4")
#cap.open('rtsp://admin:Hik12345@192.168.20.96:554/Streaming/channels/02/')

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
    start = time.time()
    classes, scores, boxes = model.detect(frame, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    end = time.time()

    blob = cv2.dnn.blobFromImage(frame , 1/255 , (416,416) ,(0,0,0) , swapRB = True , crop = False)

    net.setInput(blob)
    output_layer_names = net.getUnconnectedOutLayersNames()
    layeroutputs = net.forward(output_layer_names)

    for (classid, score, box) in zip(classes, scores, boxes):
        color = COLORS[int(classid) % len(COLORS)]
        label = "%s : %.2f" % (class_names[0], score)
        cv2.rectangle(frame, box, color, 2)
        cv2.putText(frame, label, (box[0], box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        #Crop de placa
        x, y, w, h = box
        crop_img = frame[y:y + h, x:x + w]
        #Procesamiento

        gray = get_grayscale(crop_img)

        gran = get_resize(gray)

        th = thresholding(gran)

        median = remove_noise(th)

        imgs = np.hstack([gran, th, median])

        # Mostrar múltiples
        cv2.imshow("mutil_pic", imgs)

        custom_config = r'-l eng --oem 3 --psm 9 -c tessedit_char_whitelist="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ" '
        c = pytesseract.image_to_string(median, config=custom_config)

        if  len(c) >= 6:
            salida_0.append(c[0])
            salida_1.append(c[1])
            salida_2.append(c[2])
            salida_3.append(c[3])
            salida_4.append(c[4])
            salida_5.append(c[5])

        # print(c)
        print(salida_0)
        print(salida_1)
        print(salida_2)
        print(salida_3)
        print(salida_4)
        print(salida_5)

        valor_f=str(most_freq(salida_0))+str(most_freq(salida_1))+str(most_freq(salida_2))+str(most_freq(salida_3))+str(most_freq(salida_4))+str(most_freq(salida_5))


        print("Valor final", valor_f)

        cv2.putText(frame,  valor_f, (box[0], box[1] + 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


    fps = "FPS: %.2f " % (1 / (end - start))

    cv2.putText(frame, fps, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)


    cv2.imshow("output", frame)

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
