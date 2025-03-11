import os
import cv2
import time
import numpy as np
import torch
import tkinter as tk
from PIL import Image, ImageTk

class ReferenceCaptureWindow(tk.Toplevel):
    def __init__(self, master, detector, facenet, num_images_per_position, positions, callback):
        """
        master: Tkinter root window.
        detector: Instance of EmotionDetector (to use its MTCNN for face detection).
        facenet: Instance of InceptionResnetV1 for computing embeddings.
        num_images_per_position: Number of valid images to capture per dot position.
        positions: List of tuples (position_name, rel_x, rel_y) where rel_x, rel_y are between 0 and 1.
        callback: Function to call with the final average embedding once capture is complete.
        """
        super().__init__(master)
        self.title("Capture Reference Images")
        # Maximize the window (without hiding taskbar)
        self.state('zoomed')
        self.configure(bg="black")
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        
        self.detector = detector
        self.facenet = facenet
        self.num_images_per_position = num_images_per_position
        self.positions = positions
        self.callback = callback

        self.current_position_index = 0
        self.current_capture_count = 0
        self.captured_embeddings = []  # list of embeddings from all positions

        # UI Elements
        self.canvas = tk.Canvas(self, bg="black")
        self.canvas.pack(fill="both", expand=True)
        
        self.instruction_label = tk.Label(self, text="", font=("Helvetica", 36), bg="black", fg="white")
        self.instruction_label.pack(pady=20)
        
        self.capture_button = tk.Button(self, text="Capture", font=("Helvetica", 24), command=self.capture_frame)
        self.capture_button.pack(side="bottom", pady=50)
        
        # Setup Video Capture
        self.cap = cv2.VideoCapture(0)
        self.delay = 30
        self.update_video()
        self.update_instructions()

    def on_close(self):
        # Release the camera and destroy this window and the master to exit completely.
        if self.cap.isOpened():
            self.cap.release()
        self.destroy()
        self.master.destroy()

    def update_instructions(self):
        pos_name, rel_x, rel_y = self.positions[self.current_position_index]
        self.instruction_label.config(text=f"Look at the {pos_name} dot.\nCapture image: {self.current_capture_count}/{self.num_images_per_position}")

    def update_video(self):
        ret, frame = self.cap.read()
        if ret:
            # Draw the dot at the current target position
            h, w, _ = frame.shape
            pos_name, rel_x, rel_y = self.positions[self.current_position_index]
            dot_x = int(w * rel_x)
            dot_y = int(h * rel_y)
            cv2.circle(frame, (dot_x, dot_y), 15, (0, 0, 255), -1)
            
            # Convert and display the frame on the canvas
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(frame_rgb))
            self.canvas.create_image(0, 0, image=self.photo, anchor="nw")
        self.after(self.delay, self.update_video)

    def validate_frame(self, frame):
        """
        Validate the frame: ensure a face is detected and image is not blurry.
        """
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        face = self.detector.mtcnn(pil_image)
        if face is None:
            return False
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fm = cv2.Laplacian(gray, cv2.CV_64F).var()
        if fm < 100:  # threshold (tune as needed)
            return False
        return True

    def capture_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return
        if not self.validate_frame(frame):
            self.instruction_label.config(text="Invalid image (no face/blurred). Try again.")
            return
        # Compute embedding using facenet & detector's MTCNN
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        img_cropped = self.detector.mtcnn(pil_image)
        if img_cropped is None:
            self.instruction_label.config(text="No face detected. Try again.")
            return
        if img_cropped.ndim == 3:
            img_cropped = img_cropped.unsqueeze(0)
        img_cropped = img_cropped.to(self.detector.device)
        with torch.no_grad():
            embedding = self.facenet(img_cropped).detach().cpu().numpy().flatten()
        if embedding.shape[0] != 512:
            self.instruction_label.config(text="Invalid embedding. Try again.")
            return
        
        # Save the captured frame to assets for reference (optional)
        timestamp = int(time.time())
        ref_folder = os.path.join("assets", "reference-face-frames-collect")
        if not os.path.exists(ref_folder):
            os.makedirs(ref_folder)
        filename = os.path.join(ref_folder, f"ref_{timestamp}.jpg")
        cv2.imwrite(filename, frame)
        
        # Add embedding to list
        self.captured_embeddings.append(embedding)
        self.current_capture_count += 1
        
        if self.current_capture_count >= self.num_images_per_position:
            # Finished current position; reset counter and move to next
            self.current_position_index += 1
            self.current_capture_count = 0
        if self.current_position_index >= len(self.positions):
            # Finished capturing for all positions; compute average embedding
            avg_embedding = np.mean(self.captured_embeddings, axis=0)
            # Save average embedding to assets
            avg_file = os.path.join("assets", "average_embedding.npy")
            np.save(avg_file, avg_embedding)
            self.cap.release()
            self.destroy()  # Close capture window
            self.callback(avg_embedding)
        else:
            self.update_instructions()
