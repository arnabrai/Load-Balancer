import cv2
from lane_detector import LaneDetector
from controller import LaneController
from visualizer import Visualizer

video_path = "video.mp4"

cap = cv2.VideoCapture(video_path)

# Get video properties
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = cap.get(cv2.CAP_PROP_FPS)

# Create output video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("lane_output.mp4", fourcc, fps, (width, height))

detector = LaneDetector()
controller = LaneController()
vis = Visualizer()

while True:

    ret, frame = cap.read()

    if not ret:
        break

    overlay = detector.process(frame, depth=None)
    controller.compute(detector)

    display = vis.render(
        overlay,
        detector,
        controller,
        None,
        int(fps),
        False
    )

    cv2.imshow("Lane Detection Video", display)

    # Save frame to video
    out.write(display)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
out.release()
cv2.destroyAllWindows()
