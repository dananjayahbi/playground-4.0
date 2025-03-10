# src/realtime.py

import os
import json
import cv2
import time
import numpy as np
from PIL import Image, ImageTk
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from emotion_detector import EmotionDetector
from aggregator import EmotionAggregator

# Matplotlib imports
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ------------------ Scrollable Frame Class ------------------
class ScrollableFrame(ttk.Frame):
    def __init__(self, container, *args, **kwargs):
        super().__init__(container, *args, **kwargs)
        canvas = ttk.Canvas(self)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        self.scrollable_frame = ttk.Frame(canvas)
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        # Bind mouse wheel to scroll
        self.bind_all("<MouseWheel>", lambda event: canvas.yview_scroll(int(-1*(event.delta/120)), "units"))

# ------------------ Plot Drawing Functions ------------------
def draw_bar(ax, data):
    ax.cla()
    emotions = list(data.keys())
    values = [data[emotion] for emotion in emotions]
    ax.bar(emotions, [v * 100 for v in values], color="skyblue")
    ax.set_ylim(0, 100)
    ax.set_ylabel("Percentage")

def draw_line(ax, data):
    ax.cla()
    emotions = list(data.keys())
    values = [data[emotion] for emotion in emotions]
    ax.plot(emotions, [v * 100 for v in values], marker="o", color="green")
    ax.set_ylim(0, 100)
    ax.set_ylabel("Percentage")

def draw_pie(ax, data):
    ax.cla()
    emotions = list(data.keys())
    values = [data[emotion] for emotion in emotions]
    ax.pie([v * 100 for v in values], labels=emotions, autopct='%1.1f%%', startangle=90)
    ax.axis("equal")

def draw_radar(ax, data):
    ax.cla()
    emotions = list(data.keys())
    values = [data[emotion] for emotion in emotions]
    N = len(emotions)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    # Complete the loop
    values += values[:1]
    angles += angles[:1]
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.plot(angles, [v * 100 for v in values], color="magenta", linewidth=2)
    ax.fill(angles, [v * 100 for v in values], color="magenta", alpha=0.25)
    ax.set_thetagrids(np.degrees(angles[:-1]), emotions)
    ax.set_ylim(0, 100)

# Mapping of chart types to their drawing functions
DRAW_FUNCS = {
    "Bar": draw_bar,
    "Line": draw_line,
    "Pie": draw_pie,
    "Radar": draw_radar
}

# ------------------ Main Application Class ------------------
class App:
    def __init__(self, master):
        self.master = master
        self.master.title("Live Emotion Detection")
        self.master.geometry("700x900")
        
        # Use scrollable frame to hold all content
        self.scrollable = ScrollableFrame(master)
        self.scrollable.pack(fill="both", expand=True)
        self.main_frame = self.scrollable.scrollable_frame
        
        # Center-align all content
        self.main_frame.columnconfigure(0, weight=1)
        
        # Video frame (centered)
        self.video_frame = ttk.Frame(self.main_frame)
        self.video_frame.grid(row=0, column=0, pady=10, sticky="nsew")
        self.video_label = ttk.Label(self.video_frame)
        self.video_label.pack()
        
        # Current emotion label (centered)
        self.current_emotion_label = ttk.Label(self.main_frame, text="Current Emotion: None", font=("Helvetica", 16))
        self.current_emotion_label.grid(row=1, column=0, pady=10)
        
        # Aggregated text label (centered)
        self.aggregated_label = ttk.Label(self.main_frame, text="Aggregated Emotion Confidence (Last Window):", font=("Helvetica", 12), justify="center")
        self.aggregated_label.grid(row=2, column=0, pady=10)
        
        # Container for charts
        self.charts_container = ttk.Frame(self.main_frame)
        self.charts_container.grid(row=3, column=0, pady=10)
        
        # Chart types to display simultaneously
        self.chart_types = ["Bar", "Line", "Pie", "Radar"]
        # Dictionary to store chart figures and canvases
        # Structure: {chart_type: {"last": {"fig":, "ax":, "canvas":}, "overall": {...} } }
        self.chart_dict = {}
        current_row = 0
        for ctype in self.chart_types:
            # Create a frame row for each chart type
            row_frame = ttk.Frame(self.charts_container)
            row_frame.pack(pady=10)
            # Label for the chart type
            title_label = ttk.Label(row_frame, text=f"{ctype} Chart", font=("Helvetica", 14))
            title_label.pack(pady=5)
            
            # Create a container for the two charts (Last Window and Overall) side by side
            charts_pair_frame = ttk.Frame(row_frame)
            charts_pair_frame.pack()
            
            # Last Window Chart
            last_frame = ttk.Frame(charts_pair_frame)
            last_frame.pack(side=LEFT, padx=20)
            fig_last = Figure(figsize=(3, 2.5), dpi=100)
            if ctype == "Radar":
                ax_last = fig_last.add_subplot(111, polar=True)
            else:
                ax_last = fig_last.add_subplot(111)
            ax_last.set_title("Last Window")
            canvas_last = FigureCanvasTkAgg(fig_last, master=last_frame)
            canvas_last.get_tk_widget().pack()
            
            # Overall Chart
            overall_frame = ttk.Frame(charts_pair_frame)
            overall_frame.pack(side=LEFT, padx=20)
            fig_overall = Figure(figsize=(3, 2.5), dpi=100)
            if ctype == "Radar":
                ax_overall = fig_overall.add_subplot(111, polar=True)
            else:
                ax_overall = fig_overall.add_subplot(111)
            ax_overall.set_title("Overall")
            canvas_overall = FigureCanvasTkAgg(fig_overall, master=overall_frame)
            canvas_overall.get_tk_widget().pack()
            
            # Save in chart_dict
            self.chart_dict[ctype] = {
                "last": {"fig": fig_last, "ax": ax_last, "canvas": canvas_last},
                "overall": {"fig": fig_overall, "ax": ax_overall, "canvas": canvas_overall}
            }
            current_row += 1
        
        # Quit button (centered)
        self.quit_button = ttk.Button(self.main_frame, text="Quit", command=self.quit_app, bootstyle="danger")
        self.quit_button.grid(row=4, column=0, pady=10)
        
        # Load the emotion detector (model and MTCNN)
        model_path = os.path.join("..", "model", "efficientnet_b2_emotion_model.pth")
        self.detector = EmotionDetector(model_path)
        
        # Initialize the aggregator (window_seconds set to 30 for testing; adjust as needed)
        self.aggregator = EmotionAggregator(window_seconds=30, callback=self.update_aggregated)
        
        # Initialize cumulative overall statistics
        self.overall_emotion_sum = {emotion: 0 for emotion in self.detector.emotion_classes}
        self.overall_count = 0
        
        # Start video capture
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise ValueError("Unable to open webcam.")
        self.delay = 30  # Delay in milliseconds between frames
        
        # Begin the update loop
        self.update()
    
    def update(self):
        ret, frame = self.cap.read()
        if ret:
            # Get detected faces and emotion predictions
            results = self.detector.detect_and_predict(frame)
            if results:
                # Average predictions if more than one face is detected
                emotion_sum = {emotion: 0 for emotion in self.detector.emotion_classes}
                for (_, emotion_dict) in results:
                    for emotion, prob in emotion_dict.items():
                        emotion_sum[emotion] += prob
                n = len(results)
                avg_emotion = {emotion: emotion_sum[emotion] / n for emotion in emotion_sum}
                
                # Determine the highest-probability emotion for display
                current_emotion = max(avg_emotion, key=avg_emotion.get)
                confidence = avg_emotion[current_emotion]
                self.current_emotion_label.config(text=f"Current Emotion: {current_emotion} ({confidence * 100:.2f}%)")
                
                # Add averaged emotion to aggregator (which triggers chart updates)
                self.aggregator.add_emotion(avg_emotion)
                
                # Draw bounding boxes and labels for each detected face
                for (box, emotion_dict) in results:
                    x1, y1, x2, y2 = box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"{max(emotion_dict, key=emotion_dict.get)} ({emotion_dict[max(emotion_dict, key=emotion_dict.get)] * 100:.2f}%)"
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            else:
                self.current_emotion_label.config(text="Current Emotion: No face detected")
            
            # Convert frame (BGR to RGB) and display in GUI
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb_frame)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk  # Prevent garbage collection
            self.video_label.config(image=imgtk)
        self.master.after(self.delay, self.update)
    
    def update_aggregated(self, aggregated_dict):
        """
        Callback for aggregator.
        Updates the aggregated text label and all chart pairs.
        """
        # Update aggregated text label
        with open("emotion_data.json", "r") as file:
            data = json.load(file)

        latest_entry = data[-1]
        text = f"Latest Aggregated Emotions (Time: {latest_entry['timestamp']}):\n"
        for emotion, value in latest_entry["aggregated_emotions"].items():
            text += f"{emotion}: {value * 100:.2f}%\n"
        self.aggregated_label.config(text=text)
        
        # Update overall cumulative stats
        self.overall_count += 1
        for emotion in aggregated_dict:
            self.overall_emotion_sum[emotion] += aggregated_dict[emotion]
        overall_avg = {emotion: self.overall_emotion_sum[emotion] / self.overall_count for emotion in self.overall_emotion_sum}
        
        # For each chart type, update both the last window and overall charts
        for ctype in self.chart_types:
            draw_func = DRAW_FUNCS[ctype]
            # Update last window chart
            ax_last = self.chart_dict[ctype]["last"]["ax"]
            fig_last = self.chart_dict[ctype]["last"]["fig"]
            # For radar charts, ensure polar projection; otherwise standard axis
            if ctype == "Radar":
                fig_last.clf()
                ax_last = fig_last.add_subplot(111, polar=True)
                self.chart_dict[ctype]["last"]["ax"] = ax_last
            draw_func(ax_last, aggregated_dict)
            ax_last.set_title(f"Last Window - {ctype}")
            fig_last.tight_layout()
            self.chart_dict[ctype]["last"]["canvas"].draw()
            
            # Update overall chart
            ax_overall = self.chart_dict[ctype]["overall"]["ax"]
            fig_overall = self.chart_dict[ctype]["overall"]["fig"]
            if ctype == "Radar":
                fig_overall.clf()
                ax_overall = fig_overall.add_subplot(111, polar=True)
                self.chart_dict[ctype]["overall"]["ax"] = ax_overall
            draw_func(ax_overall, overall_avg)
            ax_overall.set_title(f"Overall - {ctype}")
            fig_overall.tight_layout()
            self.chart_dict[ctype]["overall"]["canvas"].draw()
    
    def quit_app(self):
        self.cap.release()
        self.master.destroy()

if __name__ == "__main__":
    # Create a ttkbootstrap window with the dark theme
    root = ttk.Window(themename="darkly")
    app = App(root)
    root.mainloop()
