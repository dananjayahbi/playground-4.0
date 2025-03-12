import os
import tkinter as tk
from ttkbootstrap import Window
import numpy as np
from capture import ReferenceCaptureWindow
from personalize import PersonalizationWindow
from session_app import SessionApp

def launch_session_app(avg_embedding):
    main_root = Window(themename="darkly")
    app = SessionApp(main_root, avg_embedding)
    main_root.mainloop()

def start_personalization(avg_embedding):
    root = tk.Tk()
    root.withdraw()
    def on_personalization_complete():
        root.destroy()
        launch_session_app(avg_embedding)
    PersonalizationWindow(root, callback=on_personalization_complete)
    root.mainloop()

def start_reference_capture():
    root = tk.Tk()
    root.withdraw()
    positions = [
        ("Center", 0.5, 0.5),
        ("Top", 0.5, 0.2),
        ("Bottom", 0.5, 0.8),
        ("Left", 0.2, 0.5),
        ("Right", 0.8, 0.5)
    ]
    num_images_per_position = 5
    from emotion_detector import EmotionDetector
    model_path = os.path.join("..", "model", "efficientnet_b2_emotion_model.pth")
    detector = EmotionDetector(model_path)
    from facenet_pytorch import InceptionResnetV1
    facenet = InceptionResnetV1(pretrained="casia-webface").eval().to(detector.device)
    
    def on_capture_complete(avg_embedding):
        root.destroy()
        start_personalization(avg_embedding)
    
    capture_window = ReferenceCaptureWindow(root, detector, facenet, num_images_per_position, positions, callback=on_capture_complete)
    root.mainloop()

if __name__ == "__main__":
    ref_emb_path = os.path.join("assets", "average_embedding.npy")
    if os.path.exists(ref_emb_path):
        print("Reference embedding found. Skipping capture...")
        avg_embedding = np.load(ref_emb_path)
        start_personalization(avg_embedding)
    else:
        print("No reference embedding found. Starting capture process...")
        start_reference_capture()
