import cv2 as cv
import numpy as np
import imutils
import imageio

class Detector:
    def __init__(self):
        self.video = cv.VideoCapture(0)
        print("[INFO] loading model...")
        self.net = cv.dnn.readNetFromCaffe('model/MobileNetSSD_deploy.prototxt.txt',
                                            'model/MobileNetSSD_deploy.caffemodel')

    def detectObject(self):
        CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
                   "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
                   "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
                   "sofa", "train", "tvmonitor"]

        COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

        success, frame = self.video.read()

        if success is True:
            fr = imageio.imread('loading.jpeg')
            print("lol", type(np.array(fr)))
            frame = imutils.resize(frame, width=400)

            (h, w) = frame.shape[:2]
            blob = cv.dnn.blobFromImage(cv.resize(frame, (300, 300)),
                                         0.007843, (300, 300), 127.5)

            self.net.setInput(blob)
            detections = self.net.forward()

            for i in np.arange(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]

                if confidence > 0.5:
                    idx = int(detections[0, 0, i, 1])
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")

                    # draw the prediction on the frame
                    label = "{}: {:.2f}%".format(CLASSES[idx],
                                                 confidence * 100)
                    cv.rectangle(frame, (startX, startY), (endX, endY),
                                  COLORS[idx], 2)
                    y = startY - 15 if startY - 15 > 15 else startY + 15
                    cv.putText(frame, label, (startX, y),
                                cv.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
        else:
            frame = imageio.imread('loading.jpeg')
        ret, jpeg = cv.imencode('.jpg', frame)
        return jpeg.tobytes()

