"""
object_tracking.py
Handles object/person detection and tracking logic for the backend.
"""

import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import os
import logging
from collections import defaultdict
import datetime


logger = logging.getLogger(__name__)

def detect_and_track(video_path, output_path, progress_callback=None, preview_folder=None):
    """
    Run person detection and tracking on a video.
    
    Args:
        video_path: Path to input video file
        output_path: Path to save the processed video
        progress_callback: Optional callback function(progress) to report progress
        preview_folder: Optional folder to save preview images
        
    Returns:
        Tuple of (output_video_path, detections, analytics)
    """
    # Load YOLO model
    model = YOLO('yolov8n.pt')
    
    # Initialize DeepSORT tracker
    tracker = DeepSort(max_age=60, n_init=1)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception("Error opening video file")
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Initialize heatmap
    heatmap = np.zeros((height, width), dtype=np.float32)
    
    detections_for_heatmap = []
    frame_count = 0
    hourly_counts = defaultdict(int)
    total_visitors = 0
    # For advanced analytics
    video_start = datetime.datetime.today().replace(hour=0, minute=0, second=0, microsecond=0)
    frame_to_time = lambda frame: video_start + datetime.timedelta(seconds=frame / fps) if fps else video_start
    day_hour_trackids = defaultdict(lambda: defaultdict(set))  # {day: {hour: set(track_ids)}}
    month_trackids = defaultdict(set)  # {month: set(track_ids)}
    # --- Date handling for video ---
    # Use current date as video date (in real use, extract from metadata or filename)
    video_date = datetime.datetime.today().date()
    video_week = video_date.isocalendar()[1]
    video_month = video_date.strftime('%Y-%m')
    # Aggregate unique track_ids by day, week, month
    daily_trackids = defaultdict(set)
    weekly_trackids = defaultdict(set)
    monthly_trackids = defaultdict(set)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Run YOLO detection
        results = model(frame, classes=[0])  # class 0 is person
        
        # Process detections
        detections = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                if conf > 0.5:  # Confidence threshold
                    detections.append(([x1, y1, x2, y2], conf, 0))  # 0 is class_id for person
        
        # Update tracker
        tracks = tracker.update_tracks(detections, frame=frame)
        
        # Update heatmap and draw tracks
        people_in_frame = 0
        frame_track_ids = set()
        for track in tracks:
            if not track.is_confirmed():
                continue
                
            track_id = track.track_id
            ltrb = track.to_ltrb()
            
            # Update heatmap
            x1, y1, x2, y2 = map(int, ltrb)
            heatmap[y1:y2, x1:x2] += 1
            
            # Add detection for blend_heatmap
            detections_for_heatmap.append({
                'frame': frame_count,
                'bbox': [x1, y1, x2, y2],
                'track_id': track_id
            })
            
            # Draw bounding box and ID with better contrast
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Add black background for text (ID)
            text = f"ID: {track_id}"
            (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
            cv2.rectangle(frame, (x1, y1-text_height-10), (x1+text_width, y1), (0, 0, 0), -1)
            cv2.putText(frame, text, (x1, y1-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

            # Draw a small white dot at the center of the box
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            cv2.circle(frame, (center_x, center_y), 4, (255, 255, 255), -1)
            
            people_in_frame += 1
            frame_track_ids.add(track_id)
        
        # Write frame
        out.write(frame)
        # Save preview every 10 frames
        if preview_folder and frame_count % 10 == 0:
            preview_path = os.path.join(preview_folder, 'preview_detections.jpg')
            cv2.imwrite(preview_path, frame)
        
        # Update progress
        frame_count += 1
        if progress_callback and frame_count % 10 == 0:
            progress = frame_count / total_frames
            progress_callback(progress)
            logger.debug(f"Processing frame {frame_count}/{total_frames} ({progress*100:.1f}%)")
        
        frame_time_seconds = frame_count / fps if fps else 0
        hour = int(frame_time_seconds // 3600)
        hourly_counts[hour] += people_in_frame
        total_visitors += people_in_frame
        # Advanced analytics: aggregate by day/hour and month, using unique track_ids
        dt = frame_to_time(frame_count)
        day = dt.strftime('%A')
        month = dt.strftime('%b')
        for tid in frame_track_ids:
            day_hour_trackids[day][hour].add(tid)
            month_trackids[month].add(tid)
            daily_trackids[str(video_date)].add(tid)
            weekly_trackids[f"{video_date.year}-W{video_date.isocalendar()[1]}"] .add(tid)
            monthly_trackids[video_month].add(tid)
    
    # Release resources
    cap.release()
    out.release()
    
    hourly_traffic = [{"hour": h, "count": c} for h, c in sorted(hourly_counts.items())]
    peak_hour = max(hourly_counts, key=hourly_counts.get) if hourly_counts else None
    # Advanced analytics: peak hour trend and unique visitor count
    peak_hour_trend = []
    for day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']:
        hour_counts = {h: len(tids) for h, tids in day_hour_trackids[day].items()}
        if hour_counts:
            peak_hour_val = max(hour_counts, key=hour_counts.get)
            peak_hour_trend.append({'day': day, 'peakHour': peak_hour_val})
        else:
            peak_hour_trend.append({'day': day, 'peakHour': None})
    # For visitor_count: count unique track IDs per month
    visitor_count = {
        'daily': [{'period': day, 'count': len(tids)} for day, tids in sorted(daily_trackids.items())],
        'weekly': [{'period': week, 'count': len(tids)} for week, tids in sorted(weekly_trackids.items())],
        'monthly': [{'period': month, 'count': len(tids)} for month, tids in sorted(monthly_trackids.items())],
    }
    # For hourly_trend: count unique track IDs per hour
    hour_trackids = defaultdict(set)
    for det in detections_for_heatmap:
        frame = det['frame']
        track_id = det['track_id']
        hour = int((frame / fps) // 3600)
        hour_trackids[hour].add(track_id)
    hourly_trend = [{"hour": h, "count": len(tids)} for h, tids in sorted(hour_trackids.items())]
    # After the detection loop, compute unique_track_ids for this job
    unique_track_ids = set()
    for det in detections_for_heatmap:
        unique_track_ids.add(det['track_id'])
    # --- Dwell time calculation ---
    # Map: track_id -> [first_frame, last_frame]
    track_frames = {}
    for det in detections_for_heatmap:
        tid = det['track_id']
        frame = det['frame']
        if tid not in track_frames:
            track_frames[tid] = [frame, frame]
        else:
            track_frames[tid][1] = frame
    # Dwell time per track (in seconds)
    dwell_times = [(frames[1] - frames[0] + 1) / fps for frames in track_frames.values() if fps > 0 and (frames[1] - frames[0] + 1) / fps > 1]
    # For single video: just use all dwell times for average
    average_dwell_time = np.mean(dwell_times) if dwell_times else 0
    dwell_time_trend = [
        {"day": "This Video", "dwellTime": average_dwell_time, "previousWeek": 0}
    ]
    video_date_str = str(video_date)
    analytics = {
        "total_visitors": total_visitors,
        "hourly_traffic": hourly_traffic,
        "peak_hour": peak_hour,
        "peak_hour_trend": peak_hour_trend,
        "visitor_count": visitor_count,
        "hourly_trend": hourly_trend,
        "unique_track_ids": list(unique_track_ids),
        "dwell_time_trend": dwell_time_trend,
        "average_dwell_time": average_dwell_time,
        "video_date": video_date_str
    }
    print('DEBUG: Number of unique track_ids:', len(unique_track_ids))
    print('DEBUG: Example track_frames:', list(track_frames.items())[:5])
    print('DEBUG: Dwell times (seconds):', dwell_times[:10])
    print('DEBUG: Average dwell time (seconds):', np.mean(dwell_times) if dwell_times else 0)
    return output_path, detections_for_heatmap, analytics

# Add more tracking-related utilities as needed 