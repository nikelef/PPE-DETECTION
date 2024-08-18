import cv2
from flask import Flask, render_template, Response
from ultralytics import YOLO



app = Flask(__name__)

# Load the trained YOLO model
# model1 = YOLO("/Users/danishbokhari/testinggithub/Construction-Site-Safety-PPE-Detection/models/yolov8n.pt")
model = YOLO("/Users/danishbokhari/testinggithub/Construction-Site-Safety-PPE-Detection/models/best.pt")

# Open the video file using OpenCV
cap = cv2.VideoCapture(0)



# Function to generate video frames for the Flask web app
def generate_frames():
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLO inference on the current frame
        results = model.predict(frame)    

        # Process results
        for result in results:
            boxes = result.boxes  # Get bounding box outputs
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Extract box coordinates
                conf = box.conf.item()  # Convert confidence tensor to a float
                class_id = int(box.cls[0])  # Class ID
                class_name = model.names[class_id]


                # Draw the bounding box and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"Class: {class_name}, Conf: {conf:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)


        # Encode the frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Yield the frame to the web browser
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Route to render the HTML page
@app.route('/')
def index():
    return render_template('index.html')  # HTML page to display video feed

# Route to stream video
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Run the Flask app
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)