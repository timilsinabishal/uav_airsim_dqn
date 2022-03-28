import cv2
import imutils
import numpy as np

from collections import deque


class HSVDetector:
    pts = deque(maxlen=50)
    counter = 0
    (dX, dY) = (0, 0)
    hsvLower = (140, 100, 80)
    hsvUpper = (160, 255, 255)

    def __init__(self):
        self.counter = 0

    def process(self, frame):
        frame = imutils.resize(frame, width=600)
        blurred = cv2.GaussianBlur(frame, (11, 11), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(hsv, self.hsvLower, self.hsvUpper)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        center = None

        if len(cnts) > 0:
            c = max(cnts, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            M = cv2.moments(c)
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            if radius > 10:
                cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
                cv2.circle(frame, center, 5, (0, 0, 255), -1)
                self.pts.appendleft(center)

            for i in np.arange(1, len(self.pts)):
                if self.pts[i - 1] is None or self.pts[i] is None:
                    continue
                if self.counter >= 10 and i == 10 and self.pts[i - 10] is not None:
                    self.dX = self.pts[-10][0] - self.pts[i][0]
                    self.dY = self.pts[-10][1] - self.pts[i][1]

        self.counter += 1
        return self.dX, self.dY, frame
