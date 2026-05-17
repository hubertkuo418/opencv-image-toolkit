import cv2
import numpy as np

# 載入模型
net = cv2.dnn.readNetFromCaffe(
    "models/deploy.prototxt",
    "models/res10_300x300_ssd_iter_140000.caffemodel"
)

def detect_faces_dnn(img, conf_threshold=0.3):
    (h, w) = img.shape[:2]

    blob = cv2.dnn.blobFromImage(   # 把圖片轉成 AI 可以吃的格式
        cv2.resize(img, (300, 300)),
        1.0,
        (300, 300),
        (104.0, 177.0, 123.0)
    )

    net.setInput(blob)
    detections = net.forward() # AI看圖

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > conf_threshold:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            label = f"{confidence:.2f}"
            cv2.putText(img, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 0), 2)

    return img