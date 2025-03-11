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

class MainApp:
    def __init__(self, master, reference_embedding):
        self.master = master
        self.master.title("Live Emotion Detection - Personalized")
        self.master.geometry("800x800")
        # Ensure that closing the window terminates the app.
        self.master.protocol("WM_DELETE_WINDOW", self.quit_app)
        
        self.reference_embedding = reference_embedding
        
        # UI Elements
        self.status_label = ttk.Label(master, text="Emotion detection running...", font=("Helvetica", 14))
        self.status_label.pack(pady=10)
        
        self.video_label = ttk.Label(master)
        self.video_label.pack()
        
        self.current_emotion_label = ttk.Label(master, text="Current Emotion: None", font=("Helvetica", 16))
        self.current_emotion_label.pack(pady=10)
        
        self.quit_button = ttk.Button(master, text="Quit", command=self.quit_app, bootstyle="danger")
        self.quit_button.pack(pady=10)
        
        # Video Capture
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise ValueError("Unable to open webcam.")
        
        # Load Emotion Detector
        model_path = os.path.join("..", "model", "efficientnet_b2_emotion_model.pth")
        self.detector = EmotionDetector(model_path)
        
        # Initialize Aggregator (30-second window for testing)
        self.aggregator = EmotionAggregator(window_seconds=30, save_path=os.path.join("db", "emotion_data.json"))
        
        # Initialize FaceNet for computing embeddings (for recognition)
        self.facenet = InceptionResnetV1(pretrained="casia-webface").eval().to(self.detector.device)
        
        self.similarity_threshold = 0.6
        self.delay = 30  # milliseconds
        self.update()
    
    def get_face_embedding(self, pil_image):
        try:
            img_cropped = self.detector.mtcnn(pil_image)
        except Exception as e:
            print("[ERROR] MTCNN detection error:", e)
            return None

        # If no face is detected, mtcnn may return None or empty.
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
                    # Pass the merged emotion dictionary to the aggregator.
                    self.aggregator.add_emotion(emotion_dict)
                    processed = True
            if not processed:
                self.current_emotion_label.config(text="Current Emotion: No recognized face")
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb_frame)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.config(image=imgtk)
        self.master.after(self.delay, self.update)
    
    def quit_app(self):
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
    app = MainApp(root, reference_embedding)
    root.mainloop()
