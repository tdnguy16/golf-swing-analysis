import cv2

cap = cv2.VideoCapture("Videos/tigerwoods.mp4")

object_detector = cv2.createBackgroundSubtractorMOG2()


while True:
    ret, frame = cap.read()

    mask = object_detector.apply(frame)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:

        area = cv2.contourArea(cnt)
        if area > 20:

            cv2.drawContours(frame, [cnt], -1, (0,255,0), 2)


    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)

    if cv2.waitKey(1) & 0xff == ord('q'):
        break

cap.release()
cv2.destroyAllwindows()