
import cv2

# Start webcam
cap = cv2.VideoCapture(0)

# Background subtractor
background = cv2.createBackgroundSubtractorMOG2(
    history=500,
    varThreshold=25,
    detectShadows=False
)

if not cap.isOpened():
    print("Could not access webcam")
    exit()

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # Resize frame
    frame = cv2.resize(frame, (640, 480))

    # Apply background subtraction
    mask = background.apply(frame)

    # Remove noise
    mask = cv2.medianBlur(mask, 5)

    # Find contours
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    motion_detected = False

    for contour in contours:

        area = cv2.contourArea(contour)

        # Ignore small regions
        if area < 1000:
            continue

        motion_detected = True

        x, y, w, h = cv2.boundingRect(contour)

        # Draw bounding box
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

    # Display status
    status = "Motion Detected" if motion_detected else "No Motion"

    cv2.putText(
        frame,
        status,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255) if motion_detected else (0, 255, 0),
        2
    )

    cv2.imshow("Motion Detection", frame)
    cv2.imshow("Motion Mask", mask)

    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
