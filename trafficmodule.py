import cv2
import numpy as np
from matplotlib import pyplot as plt

cap = cv2.VideoCapture(0)

while 1:

    _, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # red
    lower_red = np.array([30, 0, 240])
    upper_red = np.array([110, 4, 255])
    
    # green
    lower_green = np.array([60, 5, 240])
    upper_green = np.array([100, 20, 255])
    
    mask_red = cv2.inRange(hsv, lower_red, upper_red)
    res_red = cv2.bitwise_and(frame, frame, mask=mask_red)

    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    res_green = cv2.bitwise_and(frame, frame, mask=mask_green)

    kernel = np.ones((5, 5), np.uint8)
    
    closing_red = cv2.morphologyEx(mask_red, cv2.MORPH_CLOSE, kernel)
    blur_red = cv2.GaussianBlur(closing_red, (15, 15), 0)

    closing_green = cv2.morphologyEx(mask_green, cv2.MORPH_CLOSE, kernel)
    blur_green = cv2.GaussianBlur(closing_green, (15, 15), 0)

    circles_red = cv2.HoughCircles(blur_red, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=0, maxRadius=0)

    circles_green = cv2.HoughCircles(blur_green, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=0, maxRadius=0)
    
    if circles_red is not None:
        # convert the (x, y) coordinates and radius of the circles to integers

        circles_red = np.round(circles_red[0, :]).astype("int")

        # loop over the (x, y) coordinates and radius of the circles
        for (x, y, r) in circles_red:
            if r > 10 and r < 30:
                # draw the circle in the output image, then draw a rectangle in the image
                # corresponding to the center of the circle
                cv2.circle(frame, (x, y), r, (255, 0, 0), 2, 8, 0)
                cv2.rectangle(frame, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
                # time.sleep(0.5)
                print "Column Number: "
                print x
                print "Row Number: "
                print y
                print "Radius: "
                print r
                # Display the resulting frame
                cv2.putText(frame, 'Red Light->STOP', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, 155)
                cv2.imshow('gray', frame)

    if circles_green is not None:
        # convert the (x, y) coordinates and radius of the circles to integers

        circles_green = np.round(circles_green[0, :]).astype("int")
        # loop over the (x, y) coordinates and radius of the circles
        for (x, y, r) in circles_green:
            if r > 10 and r < 30:
                # draw the circle in the output image, then draw a rectangle in the image
                # corresponding to the center of the circle
                cv2.circle(frame, (x, y), r, (255, 0, 0), 2, 8, 0)
                cv2.rectangle(frame, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
                print "Column Number: "
                print x
                print "Row Number: "
                print y
                print "Radius is: "
                print r
                # Display the resulting frame
                cv2.putText(frame, 'Green Light->GO', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, 155)
                cv2.imshow('gray', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    cv2.imshow('frame', frame)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()
