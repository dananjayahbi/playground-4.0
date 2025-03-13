import time
import json
import os
from datetime import datetime

class EmotionAggregator:
    def __init__(self, window_seconds=60, callback=None, save_path=os.path.join("db", "emotion_data.json")):
        """
        window_seconds: Aggregation window duration (60 seconds for minute-level aggregation).
        callback: Function to call with the aggregated results.
        save_path: Path to save aggregated emotions.
        """
        self.window_seconds = window_seconds
        self.start_time = time.time()
        self.emotion_records = []
        # Using six categories after merging Disgust into Sad.
        self.emotion_labels = ['Anger', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
        self.callback = callback
        self.save_path = save_path

    def add_emotion(self, emotion_dict):
        self.emotion_records.append(emotion_dict)
        if time.time() - self.start_time >= self.window_seconds:
            aggregated = self.compute_average()
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.save_to_json(timestamp, aggregated)
            if self.callback:
                self.callback(aggregated)
            else:
                print("\n=== Aggregated Emotion Confidence (Last Minute) ===")
                print(f"Timestamp: {timestamp}")
                for label, value in aggregated.items():
                    print(f"{label}: {value * 100:.2f}%")
                print("====================================================\n")
            self.start_time = time.time()
            self.emotion_records = []

    def compute_average(self):
        avg_emotions = {label: 0 for label in self.emotion_labels}
        count = len(self.emotion_records)
        if count == 0:
            return avg_emotions
        for record in self.emotion_records:
            for label in self.emotion_labels:
                avg_emotions[label] += record.get(label, 0)
        for label in avg_emotions:
            avg_emotions[label] /= count
        return avg_emotions

    def save_to_json(self, timestamp, aggregated_data):
        new_entry = {
            "timestamp": timestamp,
            "aggregated_emotions": aggregated_data
        }
        directory = os.path.dirname(self.save_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        if os.path.exists(self.save_path):
            with open(self.save_path, "r") as file:
                try:
                    data = json.load(file)
                except json.JSONDecodeError:
                    data = []
        else:
            data = []
        data.append(new_entry)
        with open(self.save_path, "w") as file:
            json.dump(data, file, indent=4)
