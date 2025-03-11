import os
import cv2
import time
import numpy as np
from PIL import Image, ImageTk
import torch
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from emotion_detector import EmotionDetector
from facenet_pytorch import InceptionResnetV1
from scipy.spatial.distance import cosine
from aggregator import EmotionAggregator

class App:
    def __init__(self, master):
        self.master = master
        self.master.title("Live Emotion Detection - Personalized")
        self.master.geometry("800x800")
        
        # UI Elements
        self.status_label = ttk.Label(master, text="Initializing...", font=("Helvetica", 14))
        self.status_label.pack(pady=10)
        
        self.video_label = ttk.Label(master)
        self.video_label.pack()
        
        self.current_emotion_label = ttk.Label(master, text="Current Emotion: None", font=("Helvetica", 16))
        self.current_emotion_label.pack(pady=10)
        
        self.capture_button = ttk.Button(master, text="Capture Reference Frame", command=self.capture_reference_frame)
        # This button is only visible if no reference embedding is found.
        
        self.quit_button = ttk.Button(master, text="Quit", command=self.quit_app, bootstyle="danger")
        self.quit_button.pack(pady=10)
        
        # Video Capture
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise ValueError("Unable to open webcam.")
        
        # Load Emotion Detector
        model_path = os.path.join("../model", "efficientnet_b2_emotion_model.pth")
        self.detector = EmotionDetector(model_path)
        
        # Initialize Aggregator: Aggregate emotions every 30 seconds (testing window)
        self.aggregator = EmotionAggregator(window_seconds=30, save_path=os.path.join("../db", "emotion_data.json"))
        
        # --- Personalization Setup ---
        # Ensure assets folder exists
        self.assets_dir = os.path.join("assets")
        if not os.path.exists(self.assets_dir):
            os.makedirs(self.assets_dir)
        # Reference frames folder inside assets
        self.reference_folder = os.path.join(self.assets_dir, "reference-face-frames-collect")
        if not os.path.exists(self.reference_folder):
            os.makedirs(self.reference_folder)
        # Average embedding file (stored in assets)
        self.embedding_file = os.path.join(self.assets_dir, "average_embedding.npy")
        
        self.reference_embedding = None
        self.num_reference_frames_required = 5
        self.reference_embeddings = []
        self.similarity_threshold = 0.6
        self.ready_for_detection = False
        
        # Initialize FaceNet (for reference embedding) on the same device as the emotion detector
        self.facenet = InceptionResnetV1(pretrained="casia-webface").eval().to(self.detector.device)
        
        # Check if the average embedding already exists
        if os.path.exists(self.embedding_file):
            self.reference_embedding = np.load(self.embedding_file)
            self.status_label.config(text="Reference embedding loaded. Ready for emotion detection.")
            self.ready_for_detection = True
        else:
            self.status_label.config(text="No reference embedding found. Please capture reference frames.")
            self.capture_button.pack(pady=10)
        
        self.delay = 30  # Milliseconds between frames
        self.update()
        
    def get_face_embedding(self, pil_image):
        """
        Uses MTCNN to crop the face from the provided PIL image
        and computes a 512-dim embedding using FaceNet.
        Returns None if no face is detected.
        """
        img_cropped = self.detector.mtcnn(pil_image)

        if img_cropped is None:  # Face detection failed
            print("[WARNING] No face detected in frame!")
            return None  # Return None to avoid crashing

        if img_cropped.ndim == 3:
            img_cropped = img_cropped.unsqueeze(0)

        img_cropped = img_cropped.to(self.detector.device)

        with torch.no_grad():
            embedding = self.facenet(img_cropped).detach().cpu().numpy().flatten()

        return embedding if embedding.shape[0] == 512 else None
    
    def capture_reference_frame(self):
        """
        Captures a frame from the webcam, computes its face embedding,
        saves the frame to the reference folder, and adds the embedding to a list.
        Once enough frames are captured, computes and saves the average embedding.
        """
        ret, frame = self.cap.read()
        if not ret:
            self.status_label.config(text="Error capturing frame. Try again.")
            return
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        embedding = self.get_face_embedding(pil_image)
        if embedding is not None:
            self.reference_embeddings.append(embedding)
            count = len(self.reference_embeddings)
            self.status_label.config(text=f"Captured {count}/{self.num_reference_frames_required} reference frames.")
            # Save the captured frame to the reference folder
            timestamp = int(time.time())
            filename = os.path.join(self.reference_folder, f"ref_{timestamp}.jpg")
            cv2.imwrite(filename, frame)
            if count >= self.num_reference_frames_required:
                avg_embedding = np.mean(self.reference_embeddings, axis=0)
                np.save(self.embedding_file, avg_embedding)
                self.reference_embedding = avg_embedding
                self.ready_for_detection = True
                self.status_label.config(text="Reference embedding captured. Ready for emotion detection.")
                self.capture_button.pack_forget()
        else:
            self.status_label.config(text="No face detected in captured frame. Try again.")
    
    def update(self):
        ret, frame = self.cap.read()
        if ret:
            if self.ready_for_detection:
                results = self.detector.detect_and_predict(frame)  # Detect faces and predict emotions
                processed = False  # Track if any recognized face was processed

                if results:  # If at least one face is detected
                    best_face = None
                    best_similarity = 0

                    for (box, emotion_dict) in results:
                        x1, y1, x2, y2 = box
                        face_roi = frame[y1:y2, x1:x2]

                        if face_roi.size != 0:  # Ensure a valid face region
                            pil_face = Image.fromarray(cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB))
                            face_embedding = self.get_face_embedding(pil_face)

                            if face_embedding is not None:
                                similarity = 1 - cosine(face_embedding, self.reference_embedding)

                                if similarity > self.similarity_threshold and similarity > best_similarity:
                                    best_face = (x1, y1, x2, y2, emotion_dict, similarity)
                                    best_similarity = similarity  # Keep track of the best-matching face

                    if best_face:  # If we found a valid matching face, process it
                        x1, y1, x2, y2, emotion_dict, similarity = best_face
                        current_emotion = max(emotion_dict, key=emotion_dict.get)
                        confidence = emotion_dict[current_emotion]

                        label = f"YOU - {current_emotion} ({confidence*100:.2f}%)"
                        self.current_emotion_label.config(text=f"Current Emotion: {current_emotion} ({confidence*100:.2f}%)")

                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw box on recognized face
                        cv2.putText(frame, label, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                        # Aggregate the recognized face's emotions
                        self.aggregator.add_emotion(emotion_dict)
                        processed = True

                if not processed:
                    self.current_emotion_label.config(text="Current Emotion: No recognized face")

            # Update the video feed in the GUI
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb_frame)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk  # Prevent garbage collection
            self.video_label.config(image=imgtk)

        self.master.after(self.delay, self.update)  # Schedule the next frame update
    
    def quit_app(self):
        self.cap.release()
        self.master.destroy()

if __name__ == "__main__":
    root = ttk.Window(themename="darkly")
    app = App(root)
    root.mainloop()
