import os
import cv2
import time
import numpy as np
import torch
import json
import tkinter as tk
from PIL import Image
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from emotion_detector import EmotionDetector
from facenet_pytorch import InceptionResnetV1
from scipy.spatial.distance import cosine
from aggregator import EmotionAggregator

class SessionApp:
    def __init__(self, master, reference_embedding):
        self.master = master
        self.master.title("Background Processes")
        self.master.geometry("400x400")
        self.master.protocol("WM_DELETE_WINDOW", self.on_close)
        
        self.reference_embedding = reference_embedding
        self.minute_aggregates = []  # Stores minute-level aggregates for the current session

        # Set session duration to 5 minutes (300 seconds)
        self.session_duration = 300  # seconds
        self.session_start_time = time.time()
        
        # Create an empty window with a canvas for the dot indicator.
        self.canvas = tk.Canvas(master, bg="black")
        self.canvas.pack(fill="both", expand=True)
        
        # Background processing: initialize video capture.
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise ValueError("Unable to open webcam.")
        
        # Load the emotion detection model.
        model_path = os.path.join("..", "model", "efficientnet_b2_emotion_model.pth")
        self.detector = EmotionDetector(model_path)
        
        # Set up the aggregator for minute-level emotion aggregation.
        self.aggregator = EmotionAggregator(window_seconds=60, save_path=os.path.join("db", "emotion_data.json"))
        self.aggregator.callback = self.minute_callback

        # Initialize FaceNet for face embedding extraction.
        self.facenet = InceptionResnetV1(pretrained="casia-webface").eval().to(self.detector.device)
        
        self.similarity_threshold = 0.6
        self.delay = 1000  # Update every 1000 ms
        self.dot_radius = 20
        
        # Status flag: True if background tasks run normally.
        self.system_status = False
        
        # Start the update loop.
        self.update()

    def minute_callback(self, aggregated_emotion):
        # This callback is triggered every minute by the aggregator.
        print("Minute aggregate:", aggregated_emotion)
        self.minute_aggregates.append(aggregated_emotion)

    def get_face_embedding(self, pil_image):
        try:
            img_cropped = self.detector.mtcnn(pil_image)
        except Exception as e:
            print("[ERROR] MTCNN detection error:", e)
            return None
        if img_cropped is None:
            return None
        if isinstance(img_cropped, list) and len(img_cropped) == 0:
            return None
        if hasattr(img_cropped, "shape") and img_cropped.shape[0] == 0:
            return None
        if img_cropped.ndim == 3:
            img_cropped = img_cropped.unsqueeze(0)
        img_cropped = img_cropped.to(self.detector.device)
        with torch.no_grad():
            embedding = self.facenet(img_cropped).detach().cpu().numpy().flatten()
        return embedding if embedding.shape[0] == 512 else None

    def update(self):
        # Attempt to read a frame from the webcam.
        ret, frame = self.cap.read()
        if not ret:
            self.system_status = False
        else:
            self.system_status = True
            # Run emotion detection in the background.
            results = self.detector.detect_and_predict(frame)
            if results:
                best_face = None
                best_similarity = 0
                for (box, emotion_dict) in results:
                    x1, y1, x2, y2 = box
                    face_roi = frame[y1:y2, x1:x2]
                    if face_roi.size != 0:
                        pil_face = Image.fromarray(cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB))
                        face_embedding = self.get_face_embedding(pil_face)
                        if face_embedding is not None:
                            similarity = 1 - cosine(face_embedding, self.reference_embedding)
                            if similarity > self.similarity_threshold and similarity > best_similarity:
                                best_face = (emotion_dict, similarity)
                                best_similarity = similarity
                if best_face:
                    emotion_dict, similarity = best_face
                    # Merge "Disgust" into "Sad" as per original logic.
                    if "Disgust" in emotion_dict and "Sad" in emotion_dict:
                        emotion_dict["Sad"] += emotion_dict["Disgust"]
                        del emotion_dict["Disgust"]
                    self.aggregator.add_emotion(emotion_dict)
        
        # Update the dot indicator based on system status.
        self.update_dot()
        
        # Check if the session duration has been reached.
        elapsed = time.time() - self.session_start_time
        if elapsed >= self.session_duration:
            self.finalize_session()
            self.minute_aggregates = []
            self.session_start_time = time.time()
        
        # Schedule the next update.
        self.master.after(self.delay, self.update)

    def update_dot(self):
        # Clear the canvas.
        self.canvas.delete("all")
        width = self.canvas.winfo_width()
        height = self.canvas.winfo_height()
        center_x = width // 2
        center_y = height // 2
        
        # Determine dot color: green if the system is functioning; red otherwise.
        dot_color = "green" if self.system_status else "red"
        
        # Draw the dot at the center.
        r = self.dot_radius
        self.canvas.create_oval(center_x - r, center_y - r, center_x + r, center_y + r, fill=dot_color, outline=dot_color)

    def finalize_session(self):
        # Compute session-level aggregate by averaging the minute-level aggregates.
        if self.minute_aggregates:
            session_data = {}
            keys = ['Anger', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
            for key in keys:
                session_data[key] = sum(item.get(key, 0) for item in self.minute_aggregates) / len(self.minute_aggregates)
            print("Session aggregate:", session_data)
            summary_path = os.path.join("db", "session_summary.json")
            if os.path.exists(summary_path):
                with open(summary_path, "r") as f:
                    try:
                        sessions = json.load(f)
                    except json.JSONDecodeError:
                        sessions = []
            else:
                sessions = []
            sessions.append(session_data)
            with open(summary_path, "w") as f:
                json.dump(sessions, f, indent=4)
        else:
            print("No minute-level data to aggregate for this session.")

    def on_close(self):
        # On window close, finalize session if appropriate and release resources.
        elapsed = time.time() - self.session_start_time
        if elapsed >= 0.8 * self.session_duration and self.minute_aggregates:
            self.finalize_session()
        if self.cap.isOpened():
            self.cap.release()
        self.master.destroy()

if __name__ == "__main__":
    import numpy as np
    ref_emb_path = os.path.join("assets", "average_embedding.npy")
    if not os.path.exists(ref_emb_path):
        raise ValueError("Reference embedding not found. Please run the capture process first.")
    reference_embedding = np.load(ref_emb_path)
    
    root = ttk.Window(themename="darkly")
    app = SessionApp(root, reference_embedding)
    root.mainloop()
