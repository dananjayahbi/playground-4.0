import os
import cv2
import time
import numpy as np
import torch
import tkinter as tk
from PIL import Image, ImageTk

class ReferenceCaptureWindow(tk.Toplevel):
    def __init__(
        self,
        master,
        detector,
        facenet,
        num_images_per_position,
        positions,
        callback
    ):
        """
        master: Tkinter root window.
        detector: Instance of EmotionDetector (for MTCNN).
        facenet:  Instance of InceptionResnetV1 for embeddings.
        num_images_per_position: Number of valid images to capture per dot position.
        positions: List of (position_name, rel_x, rel_y).
        callback:  Function to call with the final average embedding once capture is complete.
        """
        super().__init__(master)
        self.title("Capture Reference Images")
        self.state('zoomed')  # Maximized window (taskbar remains visible)
        self.configure(bg="black")
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        
        self.detector = detector
        self.facenet = facenet
        self.num_images_per_position = num_images_per_position
        self.positions = positions
        self.callback = callback

        self.current_position_index = 0
        self.current_capture_count = 0
        self.captured_embeddings = []

        # Create a full-window canvas
        self.canvas = tk.Canvas(self, bg="black", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)

        # We'll store the ID of the 'after' job so we can cancel it if needed
        self.video_job = None

        # For the camera image
        self.image_item = self.canvas.create_image(0, 0, anchor="nw")

        # Create text and button as canvas items, so they overlay the video
        # 1) Instruction text
        self.instruction_text = self.canvas.create_text(
            0, 0,
            text="",
            fill="white",
            font=("Helvetica", 14, "normal"),
            anchor="center"
        )
        # 2) Capture button
        self.capture_btn = tk.Button(
            self,
            text="Capture",
            font=("Helvetica", 12),
            command=self.capture_frame,
            bg="#222222",
            fg="white",
            relief="raised"
        )
        self.capture_btn_id = self.canvas.create_window(
            0, 0, anchor="center", window=self.capture_btn
        )

        # Setup Video Capture
        self.cap = cv2.VideoCapture(0)
        self.delay = 30

        # Bind a resize event to reposition text/button
        self.canvas.bind("<Configure>", self.on_canvas_resize)

        # Initial updates
        self.update_instructions()
        self.update_video()

    def on_close(self):
        if self.video_job is not None:
            self.after_cancel(self.video_job)
        if self.cap.isOpened():
            self.cap.release()
        self.destroy()
        self.master.destroy()

    def on_canvas_resize(self, event=None):
        """Reposition text/button items when canvas is resized."""
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()

        # Place the instruction text near the bottom or top
        # e.g. 80% down the canvas
        text_x = canvas_w // 2
        text_y = int(canvas_h * 0.8)

        self.canvas.coords(self.instruction_text, text_x, text_y)

        # Place the capture button below the instruction text
        btn_x = canvas_w // 2
        btn_y = int(canvas_h * 0.9)
        self.canvas.coords(self.capture_btn_id, btn_x, btn_y)

    def update_instructions(self):
        pos_name, _, _ = self.positions[self.current_position_index]
        text_msg = (
            f"Look at the {pos_name} dot.\n"
            f"Capture image: {self.current_capture_count}/{self.num_images_per_position}"
        )
        self.canvas.itemconfig(self.instruction_text, text=text_msg)

    def update_video(self):
        """
        Reads a frame from the webcam, resizes it to fill the canvas,
        draws a red dot at the appropriate scaled position, and displays it.
        """
        ret, frame = self.cap.read()
        if ret:
            canvas_w = self.canvas.winfo_width()
            canvas_h = self.canvas.winfo_height()

            if canvas_w > 0 and canvas_h > 0:
                # Resize the frame to the canvas size
                frame = cv2.resize(frame, (canvas_w, canvas_h), interpolation=cv2.INTER_AREA)

                # Draw the red dot
                _, rel_x, rel_y = self.positions[self.current_position_index]
                dot_x = int(canvas_w * rel_x)
                dot_y = int(canvas_h * rel_y)
                cv2.circle(frame, (dot_x, dot_y), 15, (0, 0, 255), -1)

                # Convert to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                self.photo = ImageTk.PhotoImage(image=img)

                # Update the canvas image
                self.canvas.itemconfig(self.image_item, image=self.photo)

        self.video_job = self.after(self.delay, self.update_video)

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
        return fm >= 100

    def capture_frame(self):
        """
        Captures a frame from the webcam, checks face & clarity,
        then extracts embeddings if valid.
        """
        ret, frame = self.cap.read()
        if not ret:
            return
        
        # Validate on the original frame
        if not self.validate_frame(frame):
            self.canvas.itemconfig(
                self.instruction_text,
                text="Invalid image (no face/blurred). Try again."
            )
            return

        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        img_cropped = self.detector.mtcnn(pil_image)
        if img_cropped is None:
            self.canvas.itemconfig(self.instruction_text, text="No face detected. Try again.")
            return
        if img_cropped.ndim == 3:
            img_cropped = img_cropped.unsqueeze(0)
        img_cropped = img_cropped.to(self.detector.device)
        with torch.no_grad():
            embedding = self.facenet(img_cropped).detach().cpu().numpy().flatten()

        if embedding.shape[0] != 512:
            self.canvas.itemconfig(self.instruction_text, text="Invalid embedding. Try again.")
            return

        # Save the captured frame
        timestamp = int(time.time())
        ref_folder = os.path.join("assets", "reference-face-frames-collect")
        if not os.path.exists(ref_folder):
            os.makedirs(ref_folder)
        filename = os.path.join(ref_folder, f"ref_{timestamp}.jpg")
        cv2.imwrite(filename, frame)

        self.captured_embeddings.append(embedding)
        self.current_capture_count += 1

        if self.current_capture_count >= self.num_images_per_position:
            self.current_position_index += 1
            self.current_capture_count = 0

        if self.current_position_index >= len(self.positions):
            # Compute average embedding
            avg_embedding = np.mean(self.captured_embeddings, axis=0)
            avg_file = os.path.join("assets", "average_embedding.npy")
            np.save(avg_file, avg_embedding)
            self.cap.release()
            self.destroy()
            self.callback(avg_embedding)
        else:
            self.update_instructions()
