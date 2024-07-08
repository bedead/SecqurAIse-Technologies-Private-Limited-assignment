import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from datetime import datetime
import tensorflow as tf

# Ball class and color mapping
ball_mapping = {
    0: "black ball",
    1: "blue ball",
    2: "green ball",
    3: "orange ball",
    4: "red ball",
    5: "steel ball",
    6: "violet ball",
    7: "white ball",
    8: "yellow ball",
}

# Color mapping for visualization (BGR format)
color_mapping = {
    "black ball": (0, 0, 0),
    "blue ball": (255, 0, 0),
    "green ball": (0, 255, 0),
    "orange ball": (0, 165, 255),
    "red ball": (0, 0, 255),
    "steel ball": (192, 192, 192),
    "violet ball": (238, 130, 238),
    "white ball": (255, 255, 255),
    "yellow ball": (0, 255, 255),
}

# Load the pre-trained YOLO model
model = YOLO("model/best.pt")

# Open the video file
cap = cv2.VideoCapture("video.mp4")

# Load the saved digit recognition model
digit_model = tf.keras.models.load_model("model\digit_recognition_model.h5")


def detect_quadrant(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if (
            20 < w < 50 and 20 < h < 50
        ):  # Adjust these values based on the size of your digits
            roi = gray[y : y + h, x : x + w]
            roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
            roi = roi.reshape(1, 28, 28, 1).astype("float32") / 255

            prediction = digit_model.predict(roi)
            digit = np.argmax(prediction)

            if 1 <= digit <= 4:
                return digit

    return None  # Return None if no valid quadrant number is found


def draw_ball(
    frame, ball_color, x1, y1, x2, y2, quadrant, event_type=None, timestamp=None
):
    color = color_mapping.get(ball_color, (0, 255, 0))
    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
    label = f"{ball_color} Q{quadrant}"
    cv2.putText(
        frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
    )

    if event_type:
        cv2.putText(
            frame,
            f"{event_type} {timestamp:.2f}s",
            (int(x1), int(y2) + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            2,
        )


def track_balls():
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    event_data = []
    ball_positions = {}

    out = cv2.VideoWriter(
        "output/processed_video.mp4",
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)

        if len(results) > 0 and len(results[0].boxes) > 0:
            current_positions = {}

            for box in results[0].boxes:
                cls = int(box.cls.item())
                ball_color = ball_mapping[cls]
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                x, y = (x1 + x2) / 2, (y1 + y2) / 2
                quadrant = detect_quadrant(frame=frame)

                if ball_color in ball_positions:
                    prev_quadrant = ball_positions[ball_color]
                    if prev_quadrant != quadrant:
                        timestamp = frame_count / fps
                        event_data.append(
                            (timestamp, prev_quadrant, ball_color, "Exit")
                        )
                        event_data.append((timestamp, quadrant, ball_color, "Entry"))
                        draw_ball(
                            frame,
                            ball_color,
                            x1,
                            y1,
                            x2,
                            y2,
                            quadrant,
                            "Exit/Entry",
                            timestamp,
                        )
                    else:
                        draw_ball(frame, ball_color, x1, y1, x2, y2, quadrant)
                else:
                    # First appearance of the ball
                    timestamp = frame_count / fps
                    event_data.append((timestamp, quadrant, ball_color, "Entry"))
                    draw_ball(
                        frame, ball_color, x1, y1, x2, y2, quadrant, "Entry", timestamp
                    )

                current_positions[ball_color] = quadrant

            ball_positions = current_positions

        out.write(frame)

        # Display the frame
        # cv2.imshow("Ball Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            save_event_data(event_data)
            print(f"\nProcessing complete. Output saved to: processed_video.mp4")
            print(
                f"Event data saved to: event_data.csv (Total events: {len(event_data)})"
            )
            break

        frame_count += 1
        print(f"Processing frame {frame_count}/{total_frames}", end="\r")

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    save_event_data(event_data)
    print(f"\nProcessing complete. Output saved to: processed_video.mp4")
    print(f"Event data saved to: event_data.csv (Total events: {len(event_data)})")


def save_event_data(event_data):
    if not event_data:
        print("No events recorded.")
        return

    df = pd.DataFrame(event_data, columns=["Time", "Quadrant", "Ball Color", "Type"])
    df.to_csv("output/event_data.csv", index=False)
    print(f"Saved {len(df)} events to event_data.csv")


if __name__ == "__main__":
    track_balls()
