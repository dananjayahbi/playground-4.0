# src/aggregator.py

import time
import json
import os
from datetime import datetime

class EmotionAggregator:
    def __init__(self, window_seconds=15 * 60, callback=None, save_path="emotion_data.json"):
        """
        window_seconds: Aggregation window duration (default 15 minutes; use a shorter window for testing).
        callback: A function to call with the aggregated results once the window is over.
        save_path: Path to save aggregated emotions in JSON format.
        """
        self.window_seconds = window_seconds
        self.start_time = time.time()
        self.emotion_records = []
        self.emotion_labels = ['Anger', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
        self.callback = callback
        self.save_path = save_path

    def add_emotion(self, emotion_dict):
        """Add a new prediction (a dictionary of emotion probabilities) to the records."""
        self.emotion_records.append(emotion_dict)
        if time.time() - self.start_time >= self.window_seconds:
            aggregated = self.compute_average()
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.save_to_json(timestamp, aggregated)

            if self.callback:
                self.callback(aggregated)
            else:
                print("\n=== Aggregated Emotion Confidence (Last Window) ===")
                print(f"Timestamp: {timestamp}")
                for label, value in aggregated.items():
                    print(f"{label}: {value * 100:.2f}%")
                print("====================================================\n")

            self.start_time = time.time()
            self.emotion_records = []

    def compute_average(self):
        """Calculate and return the average emotion probabilities over the window."""
        avg_emotions = {label: 0 for label in self.emotion_labels}
        count = len(self.emotion_records)
        if count == 0:
            return avg_emotions
        for record in self.emotion_records:
            for label in self.emotion_labels:
                avg_emotions[label] += record[label]
        for label in avg_emotions:
            avg_emotions[label] /= count
        return avg_emotions

    def save_to_json(self, timestamp, aggregated_data):
        """Save the aggregated data with timestamp to a JSON file."""
        new_entry = {
            "timestamp": timestamp,
            "aggregated_emotions": aggregated_data
        }

        # If file exists, load it; otherwise, create a new one
        if os.path.exists(self.save_path):
            with open(self.save_path, "r") as file:
                try:
                    data = json.load(file)
                except json.JSONDecodeError:
                    data = []
        else:
            data = []

        # Append the new entry and save back to JSON
        data.append(new_entry)
        with open(self.save_path, "w") as file:
            json.dump(data, file, indent=4)
