import requests
import streamlit as st
import av
import logging
import os
import tempfile
# Set the environment variable
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
logging.basicConfig(level=logging.WARNING)
st.set_page_config(page_title="Ai Object Detection", page_icon="🤖")
from PIL import Image
from ultralytics import YOLO
import cv2
import numpy as np
from io import BytesIO
from streamlit_webrtc import (
    ClientSettings,
    VideoTransformerBase,
    WebRtcMode,
    webrtc_streamer,
)
import supervision as sv
import numpy as np

# Initialize the YOLOv8 model with your custom-trained model
@st.cache_resource
def load_yolo_model():
    return YOLO("C:/Users/hesti/Downloads/Ai_Object_Detection-main/Ai_Object_Detection-main/yolov8n.pt")

# Load the YOLO model (this will be cached)
model = load_yolo_model()

# Define the initial confidence threshold
def main():
    st.title("🤖 Ai Object Detection")
    st.subheader("YOLOv8 & Streamlit WebRTC Integration :)")
    st.sidebar.title("Select an option ⤵️")
    choice = st.sidebar.radio("", ("Live Webcam Predict", "Capture Image And Predict", ":rainbow[Multiple Images Upload -]🖼️🖼️🖼️", "Upload Video"),
                              index = 1)
    conf = st.slider("Score threshold", 0.0, 1.0, 0.3, 0.05)
    if choice == "Live Webcam Predict":
        client_settings = ClientSettings(
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            media_stream_constraints={"video": True, "audio": False},
        )

        class ObjectDetector(VideoTransformerBase):

            def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
                img = Image.fromarray(frame.to_ndarray())
                results = model.predict(img)

                if isinstance(results, list):
                    results1 = results[0]  
                else:
                    results1 = results
                
                detections = sv.Detections.from_ultralytics(results1)
                detections = detections[detections.confidence > conf]

                # Custom labels for spatter and slag inclusion
                labels = [
                    "spatter" if class_id == 0 else "slag inclusion"
                    for class_id in detections.class_id
                ]

                frame_array = frame.to_ndarray(format="bgr24").copy()
                annotated_frame1 = box_annotator.annotate(frame_array, detections=detections)
                annotated_frame1 = label_annotator.annotate(annotated_frame1, detections=detections, labels=labels)
                zone.trigger(detections=detections)
                frame1 = zone_annotator.annotate(scene=annotated_frame1)

                count_text = f"Objects in Zone: {zone.current_count}"
                cv2.putText(frame1, count_text, (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)

                annotated_frame = av.VideoFrame.from_ndarray(frame1, format="bgr24")
                return annotated_frame

        webrtc_streamer(
            key="object-detection",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            media_stream_constraints={"video": True, "audio": False},
            video_processor_factory=ObjectDetector,
        )
    elif choice == "Capture Image And Predict":
        img_file_buffer = st.camera_input("Take a picture")

        if img_file_buffer is not None:
            bytes_data = img_file_buffer.getvalue()
            cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

            results = model.predict(cv2_img)

            if isinstance(results, list):
                results1 = results[0]  
            else:
                results1 = results
    
            detections = sv.Detections.from_ultralytics(results1)
            detections = detections[detections.confidence > conf]

            labels = [
                "spatter" if class_id == 0 else "slag inclusion"
                for class_id in detections.class_id
            ]

            labels1 = [
                f"#{index + 1}: {label} (Accuracy: {detections.confidence[index]:.2f})"
                for index, label in enumerate(labels)
            ]

            annotated_frame1 = box_annotator.annotate(cv2_img, detections=detections)
            annotated_frame1 = label_annotator.annotate(annotated_frame1, detections=detections, labels=labels)

            count_text = f"Objects in Frame: {len(detections)}"
            cv2.putText(annotated_frame1, count_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
            annotated_frame = av.VideoFrame.from_ndarray(annotated_frame1, format="bgr24")
            st.image(annotated_frame.to_ndarray(), channels="BGR")
            st.write(':orange[ Info : ⤵️ ]')
            st.json(labels1)
            st.subheader("", divider='rainbow')

    # Remaining part of the main function
    # ...

if __name__ == '__main__':
    main()
