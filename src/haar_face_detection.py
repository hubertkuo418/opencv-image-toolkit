import cv2

# 載入 OpenCV 內建人臉模型（Haar Cascade）
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def detect_faces_haar(img):
    """
    input: BGR image
    output: image with bounding boxes
    """

    # 1️⃣ 轉灰階（模型只吃灰階）
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2️⃣ 偵測人臉
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.05,
        minNeighbors=6,
        minSize=(30, 30)
    )

    # 3️⃣ 畫框框
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 3)

        # 加文字
        cv2.putText(img, "Face", (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8, (0, 255, 0), 2)
    return img