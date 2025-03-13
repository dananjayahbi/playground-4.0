import os
import cv2
import time
import numpy as np
import torch
from PIL import Image, ImageTk
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from emotion_detector import EmotionDetector
from facenet_pytorch import InceptionResnetV1
from scipy.spatial.distance import cosine
from aggregator import EmotionAggregator
import json

class SessionApp:
    def __init__(self, master, reference_embedding):
        self.master = master
        self.master.title("Live Emotion Detection - Personalized")
        self.master.geometry("800x800")
        self.master.protocol("WM_DELETE_WINDOW", self.on_close)
        
        self.reference_embedding = reference_embedding
        self.minute_aggregates = []  # Stores minute-level aggregates for the current session

        # Set session duration to 5 minutes (300 seconds)
        self.session_duration = 300  # seconds
        self.session_start_time = time.time()
        # To store completed session summaries in a list, which we later save
        self.completed_sessions = []

        # UI Elements
        self.status_label = ttk.Label(master, text="Emotion detection running...", font=("Helvetica", 14))
        self.status_label.pack(pady=10)
        
        self.video_label = ttk.Label(master)
        self.video_label.pack()
        
        self.current_emotion_label = ttk.Label(master, text="Current Emotion: None", font=("Helvetica", 16))
        self.current_emotion_label.pack(pady=10)
        
        self.quit_button = ttk.Button(master, text="Quit", command=self.on_close, bootstyle="danger")
        self.quit_button.pack(pady=10)
        
        # Video Capture
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise ValueError("Unable to open webcam.")
        
        # Load Emotion Detector
        model_path = os.path.join("..", "model", "efficientnet_b2_emotion_model.pth")
        self.detector = EmotionDetector(model_path)
        
        # Initialize minute-level aggregator with 60-second windows.
        self.aggregator = EmotionAggregator(window_seconds=60, save_path=os.path.join("db", "emotion_data.json"))
        self.aggregator.callback = self.minute_callback

        # Initialize FaceNet for recognition.
        self.facenet = InceptionResnetV1(pretrained="casia-webface").eval().to(self.detector.device)
        
        self.similarity_threshold = 0.6
        self.delay = 30  # milliseconds

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
        ret, frame = self.cap.read()
        if ret:
            results = self.detector.detect_and_predict(frame)
            processed = False
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
                                best_face = (x1, y1, x2, y2, emotion_dict, similarity)
                                best_similarity = similarity
                if best_face:
                    x1, y1, x2, y2, emotion_dict, similarity = best_face
                    # Merge "Disgust" into "Sad"
                    if "Disgust" in emotion_dict and "Sad" in emotion_dict:
                        emotion_dict["Sad"] += emotion_dict["Disgust"]
                        del emotion_dict["Disgust"]
                    current_emotion = max(emotion_dict, key=emotion_dict.get)
                    confidence = emotion_dict[current_emotion]
                    label = f"YOU - {current_emotion} ({confidence*100:.2f}%)"
                    self.current_emotion_label.config(text=f"Current Emotion: {current_emotion} ({confidence*100:.2f}%)")
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    # Add the emotion dictionary to the aggregator.
                    self.aggregator.add_emotion(emotion_dict)
                    processed = True
            if not processed:
                self.current_emotion_label.config(text="Current Emotion: No recognized face")
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb_frame)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.config(image=imgtk)
        # Check if the current session has reached its duration.
        elapsed = time.time() - self.session_start_time
        if elapsed >= self.session_duration:
            self.finalize_session()
            # Restart session: clear minute aggregates and reset timer.
            self.minute_aggregates = []
            self.session_start_time = time.time()
        self.master.after(self.delay, self.update)

    def finalize_session(self):
        # Compute session-level aggregate by averaging minute-level aggregates.
        if self.minute_aggregates:
            session_data = {}
            keys = ['Anger', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
            for key in keys:
                session_data[key] = sum(item.get(key, 0) for item in self.minute_aggregates) / len(self.minute_aggregates)
            print("Session aggregate:", session_data)
            # Save session aggregate to a file. Append to a list of sessions.
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
            # Optionally update status label.
            self.status_label.config(text="Session aggregate saved.")
        else:
            self.status_label.config(text="No minute-level data to aggregate for this session.")

    def on_close(self):
        # When closing, only finalize the current session if its elapsed time is >= 80% of session_duration.
        elapsed = time.time() - self.session_start_time
        if elapsed >= 0.8 * self.session_duration and self.minute_aggregates:
            self.finalize_session()
        if self.cap.isOpened():
            self.cap.release()
        self.master.destroy()

    def quit_app(self):
        self.on_close()

if __name__ == "__main__":
    import numpy as np
    import json
    ref_emb_path = os.path.join("assets", "average_embedding.npy")
    if not os.path.exists(ref_emb_path):
        raise ValueError("Reference embedding not found. Please run the capture process first.")
    reference_embedding = np.load(ref_emb_path)
    
    root = ttk.Window(themename="darkly")
    app = SessionApp(root, reference_embedding)
    root.mainloop()
