import cv2
import numpy as np
import math
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import pyttsx4
from PIL import Image
import pilgram

# Initialize Hand Detector, Classifier, and TTS Engine
cap = cv2.VideoCapture(0)
detect = HandDetector(maxHands=2)
classifier = Classifier("C:\\Users\\patte\Desktop\\pythonProject5\\pythonProject5\\Model\\keras_model.h5",
                        "C:\\Users\\patte\Desktop\\pythonProject5\\pythonProject5\\Model\\labels.txt")
engine = pyttsx4.init()
engine.setProperty('rate', 150)

labels = ["A", "ALL THE BEST", "B", "C", "Confuse", "D", "E", "Everyone", "F", "G", "Great", "H", "Hii", "I", "I don't Know",
          "J", "K", "Known", "L", "Learn", "M", "N", "No", "O", "P", "Please", "Proud", "Q", "Quiet", "R", "S", "Sorry",
          "T", "Thankyou", "U", "V", "Victory", "W", "X", "Y", "Yes", "Z"]

# Image parameters
offset = 10
imgSize = 300
buffer_size = 5
image_crop1 = 0
image_crop2 = 0

# Moving average for prediction smoothing
def moving_average(predictions, window_size=3):
    return np.convolve(predictions, np.ones(window_size) / window_size, mode='valid')

def camera():
    prev_detected_sign = ""
    delay_frames = 30
    current_delay = 0
    prediction_buffer = []

    while True:
        success, img = cap.read()
        if not success:
            break

        img = cv2.flip(img, 1)
        hands, img = detect.findHands(img)

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255  # Create white blank image

        # Single hand detection
        if hands and len(hands) == 1:
            hand = hands[0]
            x, y, w, h = hand['bbox']

            imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

            if imgCrop.size != 0:  # Ensure imgCrop is valid
                aspectRatio = h / w
                if aspectRatio > 1:
                    k = imgSize / h
                    wCal = math.ceil(k * w)
                    imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                    widthGap = math.ceil((imgSize - wCal) / 2)
                    imgWhite[:, widthGap:wCal + widthGap] = imgResize
                else:
                    k = imgSize / w
                    hCal = math.ceil(k * h)
                    imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                    heightGap = math.ceil((imgSize - hCal) / 2)
                    imgWhite[heightGap:hCal + heightGap, :] = imgResize

        # Two hands detection
        elif hands and len(hands) == 2:
            hand1 = hands[0]
            hand2 = hands[1]
            x1, y1, w1, h1 = hand1['bbox']
            x2, y2, w2, h2 = hand2['bbox']

            if all([y1 - offset >= 0, x1 - offset >= 0, y1 + h1 + offset <= img.shape[0], x1 + w1 + offset <= img.shape[1],
                    y2 - offset >= 0, x2 - offset >= 0, y2 + h2 + offset <= img.shape[0], x2 + w2 + offset <= img.shape[1]]):

                if w1 > 0 and h1 > 0:
                    imgCrop1 = cv2.resize(img[y1 - offset:y1 + h1 + offset, x1 - offset:x1 + w1 + offset], (imgSize, imgSize))
                else:
                    imgCrop1 = np.zeros((imgSize, imgSize, 3), dtype=np.uint8)

                if w2 > 0 and h2 > 0:
                    imgCrop2 = cv2.resize(img[y2 - offset:y2 + h2 + offset, x2 - offset:x2 + w2 + offset], (imgSize, imgSize))
                else:
                    imgCrop2 = np.zeros((imgSize, imgSize, 3), dtype=np.uint8)

                imgWhite = np.concatenate([imgCrop2, imgCrop1], 1)
                imgWhite = cv2.resize(imgWhite, (imgSize, imgSize))

        # Apply Instagram-like filter
        imgWhite_pil = Image.fromarray(imgWhite)
        imgWhite_pil = pilgram.inkwell(imgWhite_pil)
        imgWhite = np.array(imgWhite_pil)

        # Prediction
        prediction, index = classifier.getPrediction(imgWhite, draw=False)
        prediction_buffer.append(index)

        # Prediction stabilization logic
        if len(prediction_buffer) > buffer_size:
            recent_predictions = prediction_buffer[-buffer_size:]
            variance = np.var(recent_predictions)

            if variance < 1.0:
                index = int(np.median(recent_predictions))
                label = labels[index]
                prediction_buffer = []

                if hands and current_delay == 0:
                    if label != prev_detected_sign:
                        prev_detected_sign = label
                        engine.say(label)
                        engine.runAndWait()
                        current_delay = delay_frames

                text_position = (100, 100)
                cv2.putText(img, label, text_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        # Decrease delay for label display
        if current_delay > 0:
            current_delay -= 1

        # Display the original and processed images
        cv2.imshow("Webcam", img)
        cv2.imshow("Processed Image", imgWhite)

        # Add a quit option (Esc key)
        if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to quit
            break

    cap.release()
    cv2.destroyAllWindows()

# Run the camera function
camera()
