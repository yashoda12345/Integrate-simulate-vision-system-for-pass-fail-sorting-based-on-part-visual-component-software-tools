# %%
import cv2
import numpy as np
import os
import streamlit as st
from PIL import Image
import pyttsx3
import threading
from IPython import get_ipython
from IPython.display import display

# IMPORTANT: Replace these files with your custom-trained YOLOv3 model files.
# These files should contain the weights, configuration, and class names
# for the objects you want to detect (pen, spoon, belt, bottle, person,
# cellphone, box, cup, animals, birds, etc.).
YOLO_WEIGHTS = "yolov3.weights"
YOLO_CFG = "yolov3.cfg"
YOLO_NAMES = "coco.names"

# Load YOLO model
if not os.path.exists(YOLO_WEIGHTS) or not os.path.exists(YOLO_CFG) or not os.path.exists(YOLO_NAMES):
    st.error(f"Error: YOLO model files ({YOLO_WEIGHTS}, {YOLO_CFG}, {YOLO_NAMES}) not found!")
    st.stop()

try:
    net = cv2.dnn.readNet(YOLO_WEIGHTS, YOLO_CFG)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU) # You might want to use DNN_TARGET_CUDA for GPU acceleration if available

    with open(YOLO_NAMES, "r") as f:
        classes = [line.strip() for line in f.readlines()]

    layer_names = net.getLayerNames()
    # Get output layer names
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    colors = np.random.uniform(0, 255, size=(len(classes), 3))

except Exception as e:
    st.error(f"Error loading YOLO model: {e}")
    st.stop()

def detect_objects(image):
    """Performs object detection using the loaded YOLO model."""
    try:
        blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        outputs = net.forward(output_layers)

        boxes, confidences, class_ids = [], [], []
        height, width = image.shape[:2]

        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5: # Confidence threshold
                    center_x, center_y = int(detection[0] * width), int(detection[1] * height)
                    w, h = int(detection[2] * width), int(detection[3] * height)
                    x, y = int(center_x - w / 2), int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # Apply Non-Maximum Suppression (NMS) to remove redundant bounding boxes
        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4) # Confidence and NMS thresholds

        return boxes, confidences, class_ids, indices

    except Exception as e:
        st.error(f"Error during object detection: {e}")
        return [], [], [], []

def draw_labels(image, boxes, confidences, class_ids, indices):
    """Draws bounding boxes and labels on the image."""
    detected_objects = []
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            label = f"{classes[class_ids[i]]}: {confidences[i]:.2f}"
            detected_objects.append(classes[class_ids[i]])
            color = colors[class_ids[i]]
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return image, detected_objects

def speak_detected_objects(objects):
    """Speaks the names of detected objects."""
    # Only speak if exactly one object is detected to avoid too much speech
    if len(objects) == 1:
        def speak():
            try:
                engine = pyttsx3.init()
                text = f"Detected object: {objects[0]}"
                engine.say(text)
                engine.runAndWait()
            except Exception as e:
                st.warning(f"Text-to-speech error: {e}")
        # Run speech in a separate thread to avoid blocking the main Streamlit thread
        threading.Thread(target=speak).start()
    elif len(objects) > 1:
        # You could add logic here to speak multiple objects if needed
        pass

# Streamlit UI Configuration
st.set_page_config(page_title="YOLO Object Detection", page_icon="🔍", layout="wide")

# Custom CSS for Sky Blue Theme
st.markdown("""
    <style>
        body { background-color: #E3F2FD; color: black; }
        .stApp { background-color: #E3F2FD; }
        .css-18e3th9 { background-color: #90CAF9; }
        .stMarkdown h1 { text-align: center; color: #0D47A1; }
        .stSidebar { background-color: #64B5F6; color: white; }
        .stButton > button { width: 100%; background-color: #1976D2; color: white; }
        .stFileUploader { border: 2px solid #1976D2; }
    </style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("Upload Image or Start Camera")
choice = st.sidebar.radio("Choose an option:", ["Upload Image", "Live Camera"], index=0)

st.markdown("<h1>Real-Time Object Detection</h1>", unsafe_allow_html=True)

if choice == "Upload Image":
    uploaded_file = st.sidebar.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        try:
            image = Image.open(uploaded_file)
            image = np.array(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Perform object detection
            boxes, confidences, class_ids, indices = detect_objects(image)

            # Draw labels on the image
            detected_image, detected_objects = draw_labels(image.copy(), boxes, confidences, class_ids, indices)

            # Display the image with detections
            st.image(cv2.cvtColor(detected_image, cv2.COLOR_BGR2RGB), caption="Detected Objects", use_column_width=True)

            # Speak detected objects
            speak_detected_objects(detected_objects)

            if len(detected_objects) > 0:
                st.sidebar.subheader("Detected Objects:")
                for obj in detected_objects:
                    st.sidebar.write(obj)
            else:
                st.sidebar.write("No objects detected.")

        except Exception as e:
            st.error(f"Error processing uploaded image: {e}")

elif choice == "Live Camera":
    st.warning("Live camera functionality might have limitations or require specific browser permissions and configurations in a web deployment like Streamlit Share.")
    st.info("In a Colab environment, accessing the webcam directly within Streamlit can be tricky.")

    # This part of the code attempts to use OpenCV's VideoCapture
    # which might not work seamlessly in all Streamlit deployment scenarios,
    # especially in a browser-based environment like Colab's streamed output.
    # You might need to explore Streamlit components for webcam access if this doesn't work.

    cap = None
    stop_camera = False

    if st.sidebar.button("Start Camera"):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Error: Could not open camera.")
            cap = None
        else:
            st.info("Camera started. Press 'Stop Camera' to stop.")
            stop_button = st.sidebar.button("Stop Camera") # Add a stop button after starting

    if cap is not None:
        stframe = st.empty()
        while cap.isOpened() and not stop_camera:
            ret, frame = cap.read()
            if not ret:
                st.error("Error: Failed to capture frame.")
                break

            # Perform object detection on the frame
            boxes, confidences, class_ids, indices = detect_objects(frame)

            # Draw labels on the frame
            detected_frame, detected_objects = draw_labels(frame.copy(), boxes, confidences, class_ids, indices)

            # Display the frame with detections
            stframe.image(cv2.cvtColor(detected_frame, cv2.COLOR_BGR2RGB), channels="RGB", use_column_width=True)

            # Speak detected objects (optional, can be noisy in live feed)
            # speak_detected_objects(detected_objects)

            if stop_button: # Check if stop button is clicked during the loop
                 stop_camera = True

        cap.release()
        cv2.destroyAllWindows()
        st.info("Camera stopped.")

# %%
# Installation commands (can be run in a separate Colab cell or at the beginning)
# !pip install streamlit==1.23.1 # Specify a version for consistency
# !pip install opencv-python==4.7.0.72 # Specify a version
# !pip install numpy==1.24.3 # Specify a version
# !pip install Pillow==9.5.0 # Specify a version
# !pip install pyttsx3==2.90 # Specify a version