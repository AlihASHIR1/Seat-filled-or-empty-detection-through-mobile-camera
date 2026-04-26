import cv2
import os
import urllib.request
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# SETUP MODEL PATH AND DOWNLOAD
model_path = 'efficientdet_lite0.tflite'
if not os.path.exists(model_path):
    print("Downloading AI model...")
    #url_model = "https://googleapis.com"
    url_model = "https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float16/1/efficientdet_lite0.tflite"
    urllib.request.urlretrieve(url_model, model_path)

base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.ObjectDetectorOptions(base_options=base_options, score_threshold=0.3)
detector = vision.ObjectDetector.create_from_options(options)

def main():
    # --- iVCAM CONNECTION ---
    cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        print("Camera index 1 not found, trying index 0...")
        cap = cv2.VideoCapture(0)

    print("Connected! Press 'q' to quit.")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Processing
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        results = detector.detect(mp_image)

        people = []
        chairs = []

        # SORT DETECTIONS (Your original logic)
        for det in results.detections:
            bbox = det.bounding_box
            cat = det.categories[0].category_name
            coords = [bbox.origin_x, bbox.origin_y, bbox.origin_x + bbox.width, bbox.origin_y + bbox.height]

            if cat == 'person':
                people.append(coords)
            elif cat == 'chair':
                chairs.append(coords)

        filled_count = 0
        for c in chairs:
            cx1, cy1, cx2, cy2 = c
            is_occupied = False

            # Intersection Logic
            for p in people:
                px1, py1, px2, py2 = p
                ix1, iy1 = max(cx1, px1), max(cy1, py1)
                ix2, iy2 = min(cx2, px2), min(cy2, py2)

                if ix1 < ix2 and iy1 < iy2:
                    is_occupied = True
                    break

            color = (0, 0, 255) if is_occupied else (0, 255, 0)
            if is_occupied: filled_count += 1

            cv2.rectangle(frame, (int(cx1), int(cy1)), (int(cx2), int(cy2)), color, 2)
            # Labels in BLACK text as requested
            cv2.putText(frame, "Filled" if is_occupied else "Empty", (int(cx1), int(cy1-10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        # BLACK TEXT
        cv2.rectangle(frame, (0, 0), (450, 65), (255, 255, 255), -1)
        cv2.putText(frame, f"CHAIRS: {len(chairs)}", (20, 25),
                    cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(frame, f"OCCUPIED: {filled_count} | VACANT: {len(chairs) - filled_count}",
                    (20, 50), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 0), 2)

        cv2.imshow("iVCam Occupancy Counter", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()