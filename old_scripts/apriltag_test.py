import apriltag
import cv2
import numpy as np

# Load the AprilTag detector
detector = apriltag.Detector()

# Load the camera and depth map
cap = cv2.VideoCapture(0)
depth_map = cv2.imread("depth_map.png", cv2.IMREAD_UNCHANGED)

while True:
    # Capture a frame from the camera
    ret, frame = cap.read()

    # Detect AprilTags in the frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    results = detector.detect(gray)

    # Draw AprilTag detections on the frame
    for r in results:
        # Extract the AprilTag ID and pixel position
        tag_id = r.tag_id
        pixel_pos = np.mean(r.corners, axis=0).astype(int)

        # Print the ID and pixel position
        print("AprilTag ID:", tag_id)
        print("AprilTag pixel position:", pixel_pos)

        # Get the depth value from the depth map
        depth = depth_map[pixel_pos[1], pixel_pos[0]]

        # Print the depth value
        print("Depth value:", depth)

        # Draw the AprilTag outline and ID on the frame
        cv2.polylines(frame, [r.corners.astype(int)], True, (0, 255, 0), 2)
        cv2.putText(frame, str(tag_id), tuple(r.corners[0].astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Show the frame
    cv2.imshow("AprilTag Detection", frame)

    # Exit if the user presses 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the window
cap.release()
cv2.destroyAllWindows()

