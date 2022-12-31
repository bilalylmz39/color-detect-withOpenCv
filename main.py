import cv2
import imutils
import numpy as np


buffer_size = 16
# points = deque(maxlen=buffer_size)

# renk aralığı HSV

redLower = (160,50,50)
redUpper = (180,255,255)

cap = cv2.VideoCapture(0)
cap.set(3, 960)
cap.set(4, 480)

while True:
    success, imgOrig = cap.read()

    if success:
        blured = cv2.GaussianBlur(imgOrig, (11, 11), 0)

        hsv = cv2.cvtColor(blured, cv2.COLOR_BGR2HSV)

        cv2.imshow("HSV", hsv)

        # mavi için maske
        maske = cv2.inRange(hsv, redLower, redUpper)
        cv2.imshow("mask", maske)

        maske = cv2.erode(maske, None, iterations=2)
        maske = cv2.dilate(maske, None, iterations=2)
        cv2.imshow("mask + erozyon", maske)

        # konturleme
        cnts = cv2.findContours(maske, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        print(imutils.is_cv2())
        cnts = cnts[0] if imutils.is_cv4() else cnts[1]

        center = None

        if len(cnts) > 0:
            c = max(cnts, key=cv2.contourArea)
            # dikdörtgene çevirme
            rectangle = cv2.minAreaRect(c)

            ((x, y), (width, height), rotation) = rectangle

            s = "x: {}, width: {}, height: {}, rotation: {}".format(np.round(x), np.round(y), np.round(width),
                                                                    np.round(height), np.round(rotation))
            print(s)
            # kutulama
            box = cv2.boxPoints(rectangle)
            box = np.int64(box)
            # momentler
            moment = cv2.moments(c)
            print("momentmis", moment)
            center = (int(moment["m10"] / moment["m00"]), int(moment["m01"] / moment["m00"]))
            # kontor çizdirme
            cv2.drawContours(imgOrig, [box], 0, (0, 255, 255), 2)

            # merkeze nokta çizimi
            cv2.circle(imgOrig, center, 5, (255, 0, 255), -1)
            # bilgileri ekrana yazdırma
            cv2.putText(imgOrig, s, (25, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 2)

        cv2.imshow("Orijinal Tespit", imgOrig)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        cap.release()
        cv2.destroyAllWindows()
        break