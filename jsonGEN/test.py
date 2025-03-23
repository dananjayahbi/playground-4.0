import json
import random
import os
from datetime import datetime, timedelta

def generate_emotion_data(start_time, num_minutes):
    """
    Generate minute-level emotion data.
    Each entry has a timestamp and aggregated_emotions (normalized probabilities for 6 emotions),
    ensuring that "Sad" is adjusted to a random value between 40% and 60%.
    Also includes flags for session usage.
    """
    emotions = ["Anger", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
    data = []
    current_time = start_time
    for _ in range(num_minutes):
        # Generate random values and normalize them so they sum to 1
        values = [random.random() for _ in emotions]
        total = sum(values)
        normalized = [v / total for v in values]
        aggregated_emotions = {emotion: round(normalized[i], 3) for i, emotion in enumerate(emotions)}
        
        # Choose a random target for "Sad" between 0.4 and 0.6
        target_sad = round(random.uniform(0.4, 0.6), 3)
        original_sad = aggregated_emotions["Sad"]
        
        # Adjust other emotions proportionally so they sum to (1 - target_sad)
        if original_sad != 1:
            scale = (1 - target_sad) / (1 - original_sad)
        else:
            scale = 1  # Unlikely edge case
        for emotion in aggregated_emotions:
            if emotion != "Sad":
                aggregated_emotions[emotion] = round(aggregated_emotions[emotion] * scale, 3)
        aggregated_emotions["Sad"] = target_sad
        
        # Correct any rounding differences to ensure total equals 1
        total_adjusted = sum(aggregated_emotions.values())
        if total_adjusted != 1:
            difference = round(1 - total_adjusted, 3)
            aggregated_emotions["Sad"] = round(aggregated_emotions["Sad"] + difference, 3)
        
        entry = {
            "timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S"),
            "aggregated_emotions": aggregated_emotions,
            "session_used": True, 
            "session_used_hour": True
        }
        data.append(entry)
        current_time += timedelta(minutes=1)
    return data

def aggregate_sessions(data, group_size):
    """
    Aggregates emotion data in groups.
    :param data: List of minute-level entries.
    :param group_size: Number of entries to aggregate (e.g., 5 for 5-minute, 60 for hourly).
    :return: A list of session aggregate dictionaries.
    """
    sessions = []
    for i in range(0, len(data), group_size):
        group = data[i:i+group_size]
        # Skip incomplete groups.
        if len(group) < group_size:
            break
        aggregated = {}
        emotions = group[0]["aggregated_emotions"].keys()
        for emotion in emotions:
            aggregated[emotion] = round(sum(entry["aggregated_emotions"][emotion] for entry in group) / group_size, 3)
        session_summary = {
            "timestamp": group[-1]["timestamp"],
            "session_aggregate": aggregated,
            "db_status": False
        }
        sessions.append(session_summary)
    return sessions

def adjust_hourly_summary(hourly_summary):
    """
    Adjust the aggregated hourly summaries so that the three "Sad" values are significantly different.
    For this example:
      - The first hour's "Sad" target is randomly chosen from [0.4, 0.5],
      - The second hour's "Sad" target is randomly chosen from [0.55, 0.65],
      - The third hour's "Sad" target is randomly chosen from [0.65, 0.7].
    Other emotion values are scaled proportionally so that the total remains 1.
    """
    if len(hourly_summary) != 3:
        return hourly_summary
    
    # Define target ranges for each session to create a wide spread
    target_ranges = [
        (0.4, 0.5),   # lower "Sad" for the first hour
        (0.55, 0.65), # mid-range for the second hour
        (0.65, 0.7)   # higher "Sad" for the third hour
    ]
    
    for i, session in enumerate(hourly_summary):
        aggregate = session["session_aggregate"]
        original_sad = aggregate["Sad"]
        low, high = target_ranges[i]
        new_sad = round(random.uniform(low, high), 3)
        
        # Adjust other emotions proportionally to account for the new "Sad" value
        for emotion in aggregate:
            if emotion != "Sad":
                # Avoid division by zero if original_sad equals 1 (unlikely)
                aggregate[emotion] = round(aggregate[emotion] * ((1 - new_sad) / (1 - original_sad)) if (1 - original_sad) != 0 else aggregate[emotion], 3)
        aggregate["Sad"] = new_sad
        
        # Correct rounding differences so the total remains 1
        total_adjusted = sum(aggregate.values())
        if total_adjusted != 1:
            diff = round(1 - total_adjusted, 3)
            aggregate["Sad"] = round(aggregate["Sad"] + diff, 3)
            
    return hourly_summary

def main():
    # Create an output directory (change as needed)
    output_dir = "sample_data"
    os.makedirs(output_dir, exist_ok=True)
    
    # Set the start time and number of minutes (3 hours = 180 minutes)
    start_time = datetime(2025, 3, 17, 20, 0, 0)
    num_minutes = 180
    
    # Generate minute-level emotion data
    emotion_data = generate_emotion_data(start_time, num_minutes)
    emotion_data_file = os.path.join(output_dir, "emotion_data.json")
    with open(emotion_data_file, "w") as f:
        json.dump(emotion_data, f, indent=4)
    print(f"Generated {emotion_data_file} with {len(emotion_data)} entries.")
    
    # Generate 5-minute session summary (36 entries)
    session_summary = aggregate_sessions(emotion_data, 5)
    session_summary_file = os.path.join(output_dir, "session_summery.json")
    with open(session_summary_file, "w") as f:
        json.dump(session_summary, f, indent=4)
    print(f"Generated {session_summary_file} with {len(session_summary)} entries.")
    
    # Generate hourly session summary (3 entries)
    hourly_summary = aggregate_sessions(emotion_data, 60)
    # Adjust hourly summary to meet the distinct "Sad" values requirement
    hourly_summary = adjust_hourly_summary(hourly_summary)
    hourly_summary_file = os.path.join(output_dir, "session_summery1h.json")
    with open(hourly_summary_file, "w") as f:
        json.dump(hourly_summary, f, indent=4)
    print(f"Generated {hourly_summary_file} with {len(hourly_summary)} entries.")

if __name__ == "__main__":
    main()
