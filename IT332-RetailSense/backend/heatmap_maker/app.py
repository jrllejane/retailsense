"""
app.py
Flask entry point for the backend, using refactored modules.
"""

import os
import uuid
import threading
import logging
from flask import Flask, request, jsonify, session, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
from job_manager import init_db, get_db_connection
from video_processing import validate_video_file
from heatmap_maker import blend_heatmap, extract_heatmap_analytics
from utils import hash_password, verify_password
import datetime
import cv2
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
import shutil
import json
from object_tracking import detect_and_track
from collections import defaultdict, Counter
import numpy as np
from sklearn.cluster import DBSCAN
from datetime import timedelta

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = b'supersecretkey'  # Replace with a secure key in production
app.config['JWT_SECRET_KEY'] = 'superjwtsecretkey'  # Change this in production
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(days=7)
jwt = JWTManager(app)

# Configure CORS properly
CORS(app, resources={
    r"/api/*": {
        "origins": ["http://localhost:5173"],  # Frontend URL
        "methods": ["GET", "POST", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"],
        "supports_credentials": True
    }
})

UPLOAD_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../project_uploads'))
RESULTS_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../project_results'))
ALLOWED_EXTENSIONS_VIDEO = {'mp4', 'avi', 'mov'}
ALLOWED_EXTENSIONS_IMAGE = {'png', 'jpg', 'jpeg'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

jobs = {}

def allowed_file(filename, allowed_extensions):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

def process_video_job(job_id):
    """Process a video job in the background (restore backend detection)."""
    try:
        job = jobs[job_id]
        job['status'] = 'processing'
        job['message'] = 'Starting video processing...'
        job['cancelled'] = False

        # Validate video file
        video_path = job['input_files']['video']
        floorplan_path = job['input_files']['floorplan']
        points_path = job['input_files']['points']
        with open(points_path, 'r') as f:
            points_data = json.load(f)
        cap = validate_video_file(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        # Update status for YOLO detection
        job['message'] = 'Running YOLO detection (0%)'
        output_video_path = job['output_files_expected']['video']
        output_video_path, detections, analytics = detect_and_track(
            video_path,
            output_video_path,
            progress_callback=lambda p: update_job_progress(job_id, 'YOLO detection', p),
            preview_folder=job['output_files_expected']['image'] and os.path.dirname(job['output_files_expected']['image'])
        )

        # Save analytics to file
        job_results_folder = os.path.dirname(output_video_path)
        analytics_path = os.path.join(job_results_folder, f"{job_id}_analytics.json")
        with open(analytics_path, "w") as f:
            json.dump(analytics, f)

        # For testing: use static points from Points/floorplan_points.txt
        points = [[768, 204], [690, 200], [655, 305], [793, 309]]

        # Now, generate the blended heatmap using blend_heatmap with real detections and points
        output_heatmap_image_path = job['output_files_expected']['image']
        blend_heatmap(
            detections,
            floorplan_path,
            output_heatmap_image_path,
            output_video_path,
            points,
            preview_folder=os.path.dirname(output_heatmap_image_path)
        )

        # Update status for heatmap generation
        job['message'] = 'Processing completed successfully'
        job['status'] = 'completed'
        job['message'] = 'Processing completed successfully'
        # Update database
        conn = get_db_connection()
        conn.execute('''
            UPDATE jobs 
            SET status = ?, message = ?, updated_at = CURRENT_TIMESTAMP, output_heatmap_path = ?
            WHERE job_id = ?
        ''', (job['status'], job['message'], output_heatmap_image_path, job_id))
        conn.commit()
        conn.close()

    except Exception as e:
        if hasattr(job, 'cancelled') and job['cancelled']:
            job['status'] = 'cancelled'
            job['message'] = 'Job was cancelled by user.'
        else:
            job['status'] = 'error'
            job['message'] = f'Error during processing: {str(e)}'
        logger.error(f"Error processing job {job_id}: {str(e)}", exc_info=True)
        # Update database with error
        conn = get_db_connection()
        conn.execute('''
            UPDATE jobs 
            SET status = ?, message = ?, updated_at = CURRENT_TIMESTAMP
            WHERE job_id = ?
        ''', (job['status'], job['message'], job_id))
        conn.commit()
        conn.close()

def update_job_progress(job_id, stage, progress):
    """Update job progress in both memory and database."""
    job = jobs[job_id]
    job['message'] = f'{stage} ({int(progress * 100)}%)'
    
    # Update database
    conn = get_db_connection()
    conn.execute('''
        UPDATE jobs 
        SET message = ?, updated_at = CURRENT_TIMESTAMP
        WHERE job_id = ?
    ''', (job['message'], job_id))
    conn.commit()
    conn.close()

@app.route('/api/heatmap_jobs', methods=['POST'])
@jwt_required()
def create_heatmap_job():
    try:
        logger.debug("Received job creation request")
        logger.debug(f"Files in request: {request.files}")
        logger.debug(f"Form data: {request.form}")

        if 'videoFile' not in request.files:
            logger.error("Missing required video file")
            return jsonify({"error": "Missing videoFile"}), 400
        
        points_data_str = request.form.get('pointsData')
        if not points_data_str:
            logger.error("Missing points data")
            return jsonify({"error": "Missing pointsData"}), 400
        try:
            points_data = json.loads(points_data_str)
            if not (isinstance(points_data, list) and len(points_data) == 4):
                raise ValueError("pointsData must be a list of 4 points")
        except Exception as e:
            logger.error(f"Invalid pointsData: {e}")
            return jsonify({"error": f"Invalid pointsData: {e}"}), 400

        video_file = request.files['videoFile']
        logger.debug(f"Video file: {video_file.filename}")
        if not (video_file.filename and allowed_file(video_file.filename, ALLOWED_EXTENSIONS_VIDEO)):
            logger.error("Invalid video file type")
            return jsonify({"error": "Invalid video file type"}), 400

        job_id = str(uuid.uuid4())
        logger.debug(f"Generated job ID: {job_id}")

        job_upload_folder = os.path.join(UPLOAD_FOLDER, job_id)
        job_results_folder = os.path.join(RESULTS_FOLDER, job_id)
        os.makedirs(job_upload_folder, exist_ok=True)
        os.makedirs(job_results_folder, exist_ok=True)

        video_filename = secure_filename(video_file.filename)
        points_filename = f"points_{job_id}.json"
        floorplan_filename = f"floorplan_{job_id}.jpg"

        input_video_path = os.path.join(job_upload_folder, video_filename)
        input_points_path = os.path.join(job_upload_folder, points_filename)
        input_floorplan_path = os.path.join(job_upload_folder, floorplan_filename)

        logger.debug(f"Saving files to: {job_upload_folder}")
        video_file.save(input_video_path)
        with open(input_points_path, 'w') as f:
            json.dump(points_data, f)

        # Extract first frame as floorplan
        cap = cv2.VideoCapture(input_video_path)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            logger.error("Failed to extract first frame from video")
            return jsonify({"error": "Failed to extract first frame from video"}), 500
        cv2.imwrite(input_floorplan_path, frame)

        output_heatmap_image_path = os.path.join(job_results_folder, f"video_{job_id}_heatmap.jpg")
        output_processed_video_path = os.path.join(job_results_folder, f"video_{job_id}.mp4")

        # Create job entry
        jobs[job_id] = {
            'status': 'pending',
            'message': 'Job submitted, awaiting processing.',
            'input_files': {
                'video': input_video_path,
                'floorplan': input_floorplan_path,
                'points': input_points_path
            },
            'output_files_expected': {
                'image': output_heatmap_image_path,
                'video': output_processed_video_path
            }
        }

        # Get current user from JWT
        current_user = get_jwt_identity()
        logger.debug(f"Current user: {current_user}")

        # Create database entry
        conn = get_db_connection()
        try:
            logger.debug("Creating database entry")
            conn.execute('''
                INSERT INTO jobs (job_id, user, input_video_name, input_floorplan_name, status, message)
                VALUES (?, ?, ?, ?, ?, ?)''',
                (job_id, current_user, video_filename, floorplan_filename, 'pending', 'Job submitted, awaiting processing.'))
            conn.commit()
            logger.debug("Database entry created successfully")
        except Exception as db_error:
            logger.error(f"Database error: {str(db_error)}")
            raise
        finally:
            conn.close()

        # Start processing in background thread
        processing_thread = threading.Thread(target=process_video_job, args=(job_id,))
        processing_thread.daemon = True
        processing_thread.start()

        return jsonify({"job_id": job_id, "status": "pending", "message": "Job submitted for processing."}), 202
    except Exception as e:
        logger.error(f"Error in create_heatmap_job: {str(e)}", exc_info=True)
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route('/api/heatmap_jobs/<job_id>/status', methods=['GET'])
def get_job_status(job_id):
    job = jobs.get(job_id)
    if job:
        return jsonify({"job_id": job_id, "status": job['status'], "message": job.get('message', '')})
    else:
        conn = get_db_connection()
        db_job = conn.execute("SELECT job_id, status, message FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        conn.close()
        if db_job:
            return jsonify({"job_id": db_job['job_id'], "status": db_job['status'], "message": db_job['message']})
        else:
            return jsonify({"error": "Job not found or not authorized"}), 404

@app.route('/api/heatmap_jobs/<job_id>/result/image', methods=['GET'])
def get_heatmap_image(job_id):
    conn = get_db_connection()
    job_row = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
    conn.close()
    if not job_row or job_row['status'] != 'completed':
        return jsonify({"error": "Job not found or not completed"}), 404

    output_image_path = job_row['output_heatmap_path'] if 'output_heatmap_path' in job_row.keys() else None
    if not output_image_path or not os.path.exists(output_image_path):
        # Try .jpg if .png not found
        jpg_path = output_image_path.replace('.png', '.jpg') if output_image_path else None
        if jpg_path and os.path.exists(jpg_path):
            output_image_path = jpg_path
        else:
            return jsonify({"error": "Result image file not found on server"}), 404
    return send_from_directory(os.path.dirname(output_image_path), os.path.basename(output_image_path))

@app.route('/api/heatmap_jobs/<job_id>/result/video', methods=['GET'])
def get_processed_video(job_id):
    conn = get_db_connection()
    job_row = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
    conn.close()
    if not job_row or job_row['status'] != 'completed':
        return jsonify({"error": "Job not found or not completed"}), 404

    output_video_path = job_row['output_video_path'] if 'output_video_path' in job_row.keys() else None
    if not output_video_path or not os.path.exists(output_video_path):
        return jsonify({"error": "Result video file not found on server"}), 404
    return send_from_directory(os.path.dirname(output_video_path), os.path.basename(output_video_path), as_attachment=True)

@app.route('/api/heatmap_jobs/history', methods=['GET'])
@jwt_required()
def get_job_history():
    conn = get_db_connection()
    history_jobs_cursor = conn.execute('''
        SELECT job_id, input_video_name, input_floorplan_name, status, message, created_at, updated_at
        FROM jobs ORDER BY created_at DESC
    ''')
    history_jobs = [dict(row) for row in history_jobs_cursor.fetchall()]
    conn.close()
    return jsonify(history_jobs)

@app.route('/api/register', methods=['POST'])
def register():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    email = data.get('email')
    if not all([username, password, email]):
        return jsonify({"error": "Missing required fields"}), 400
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT username FROM users WHERE username = ? OR email = ?", (username, email))
        if cursor.fetchone():
            return jsonify({"error": "Username or email already exists"}), 400
        password_hash = hash_password(password)
        cursor.execute(
            "INSERT INTO users (username, password_hash, email) VALUES (?, ?, ?)",
            (username, password_hash, email)
        )
        conn.commit()
        return jsonify({"success": True, "message": "Registration successful"}), 201
    except Exception as e:
        return jsonify({"error": f"Database error: {str(e)}"}), 500
    finally:
        conn.close()

@app.route('/api/login', methods=['POST'])
def login_api():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    if not username or not password:
        return jsonify({"error": "Missing username or password"}), 400
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT password_hash FROM users WHERE username = ?", (username,))
        user = cursor.fetchone()
        if user and verify_password(user['password_hash'], password):
            access_token = create_access_token(identity=username)
            return jsonify({"success": True, "message": "Login successful", "access_token": access_token})
        return jsonify({"error": "Invalid credentials"}), 401
    except Exception as e:
        return jsonify({"error": f"Database error: {str(e)}"}), 500
    finally:
        conn.close()

@app.route('/api/logout', methods=['POST'])
def logout_api():
    # With JWT, logout is handled client-side by deleting the token
    return jsonify({"success": True, "message": "Logged out successfully"})

@app.route('/api/user', methods=['GET'])
@jwt_required()
def get_user_info():
    username = get_jwt_identity()
    if not username:
        return jsonify({"error": "Not logged in"}), 401
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT username, email, created_at FROM users WHERE username = ?", (username,))
        user = cursor.fetchone()
        if user:
            return jsonify({
                "username": user['username'],
                "email": user['email'],
                "created_at": user['created_at']
            })
        return jsonify({"error": "User not found"}), 404
    except Exception as e:
        return jsonify({"error": f"Database error: {str(e)}"}), 500
    finally:
        conn.close()

@app.route('/api/heatmap_jobs/<job_id>', methods=['DELETE'])
@jwt_required()
def delete_heatmap_job(job_id):
    try:
        conn = get_db_connection()
        job_row = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        if not job_row:
            conn.close()
            return jsonify({"error": "Job not found"}), 404
        # Remove from DB
        conn.execute("DELETE FROM jobs WHERE job_id = ?", (job_id,))
        conn.commit()
        conn.close()
        # Remove files (results and uploads)
        results_folder = os.path.join(RESULTS_FOLDER, job_id)
        uploads_folder = os.path.join(UPLOAD_FOLDER, job_id)
        for folder in [results_folder, uploads_folder]:
            if os.path.exists(folder):
                shutil.rmtree(folder)
        return jsonify({"success": True, "message": "Heatmap job deleted."})
    except Exception as e:
        logger.error(f"Error deleting job {job_id}: {str(e)}", exc_info=True)
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route('/api/heatmap_jobs/<job_id>/cancel', methods=['POST'])
@jwt_required()
def cancel_heatmap_job(job_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    job['cancelled'] = True
    return jsonify({"success": True, "message": "Job cancelled."})

@app.route('/api/heatmap_jobs/<job_id>/preview/detections', methods=['GET'])
def get_detection_preview(job_id):
    job_folder = os.path.join(RESULTS_FOLDER, job_id)
    preview_path = os.path.join(job_folder, 'preview_detections.jpg')
    if not os.path.exists(preview_path):
        return jsonify({"error": "No detection preview available yet."}), 404
    return send_from_directory(job_folder, 'preview_detections.jpg')

@app.route('/api/heatmap_jobs/<job_id>/preview/heatmap', methods=['GET'])
def get_heatmap_preview(job_id):
    job_folder = os.path.join(RESULTS_FOLDER, job_id)
    preview_path = os.path.join(job_folder, 'preview_heatmap.jpg')
    if not os.path.exists(preview_path):
        return jsonify({"error": "No heatmap preview available yet."}), 404
    return send_from_directory(job_folder, 'preview_heatmap.jpg')

@app.route('/api/heatmap_jobs/<job_id>/detections', methods=['POST'])
def receive_live_detections(job_id):
    if job_id not in jobs:
        return jsonify({'error': 'Job not found'}), 404
    try:
        data = request.get_json()
        detections = data.get('detections', [])
        if 'live_detections' not in jobs[job_id]:
            jobs[job_id]['live_detections'] = []
        jobs[job_id]['live_detections'].extend(detections)
        # Optionally, trigger heatmap update here
        return jsonify({'success': True, 'count': len(detections)})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/user/username', methods=['PUT'])
@jwt_required()
def update_username():
    username = get_jwt_identity()
    if not username:
        return jsonify({"error": "Not logged in"}), 401
    
    data = request.get_json()
    new_username = data.get('username')
    
    if not new_username:
        return jsonify({"error": "New username is required"}), 400
    
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        # Check if new username already exists
        cursor.execute("SELECT username FROM users WHERE username = ?", (new_username,))
        if cursor.fetchone():
            return jsonify({"error": "Username already exists"}), 400
        
        # Update username
        cursor.execute("UPDATE users SET username = ? WHERE username = ?", 
                      (new_username, username))
        conn.commit()
        
        return jsonify({
            "message": "Username updated successfully",
            "username": new_username
        })
    except Exception as e:
        return jsonify({"error": f"Database error: {str(e)}"}), 500
    finally:
        conn.close()

#new route for analytics
@app.route('/api/heatmap_jobs/<job_id>/analytics', methods=['GET'])
@jwt_required()
def get_heatmap_analytics(job_id):
    job_results_folder = os.path.join(RESULTS_FOLDER, job_id)
    analytics_path = os.path.join(job_results_folder, f"{job_id}_analytics.json")
    if not os.path.exists(analytics_path):
        return jsonify({"error": "Analytics not found"}), 404
    with open(analytics_path, "r") as f:
        analytics = json.load(f)
    return jsonify(analytics)

def get_dashboard_analytics_data():
    import datetime
    from collections import defaultdict, Counter
    import calendar
    processed_videos = 0
    generated_heatmaps = 0
    conn = get_db_connection()
    processed_videos = conn.execute("SELECT COUNT(*) FROM jobs WHERE status = 'completed'").fetchone()[0]
    jobs_with_heatmap = conn.execute("SELECT job_id, output_heatmap_path FROM jobs WHERE status = 'completed'").fetchall()
    for row in jobs_with_heatmap:
        if row['output_heatmap_path'] and os.path.exists(row['output_heatmap_path']):
            generated_heatmaps += 1
    conn.close()
    visitor_count_by_period = defaultdict(int)
    hourly_trend_by_hour = defaultdict(int)
    peak_hour_trend_by_day = {}
    all_unique_track_ids = set()
    for job_id in os.listdir(RESULTS_FOLDER):
        analytics_path = os.path.join(RESULTS_FOLDER, job_id, f"{job_id}_analytics.json")
        if os.path.exists(analytics_path):
            with open(analytics_path, "r") as f:
                try:
                    analytics = json.load(f)
                except Exception:
                    continue
            track_ids = analytics.get("unique_track_ids", [])
            all_unique_track_ids.update(track_ids)
            visitor_count_data = analytics.get("visitor_count", {})
            if isinstance(visitor_count_data, dict):
                for period_type in ["daily", "weekly", "monthly"]:
                    for entry in visitor_count_data.get(period_type, []):
                        period = entry.get("period")
                        count = entry.get("count", 0)
                        visitor_count_by_period[(period_type, period)] += count
            elif isinstance(visitor_count_data, list):
                for entry in visitor_count_data:
                    period = entry.get("period")
                    count = entry.get("count", 0)
                    visitor_count_by_period[("monthly", period)] += count  # fallback
            for entry in analytics.get("hourly_trend", []):
                hour = entry.get("hour")
                count = entry.get("count", 0)
                hourly_trend_by_hour[hour] += count
            for entry in analytics.get("peak_hour_trend", []):
                day = entry.get("day")
                peak_hour = entry.get("peakHour")
                if day not in peak_hour_trend_by_day and peak_hour is not None:
                    peak_hour_trend_by_day[day] = peak_hour
    visitor_count = {
        "daily": [],
        "weekly": [],
        "monthly": []
    }
    for (period_type, period), count in sorted(visitor_count_by_period.items()):
        visitor_count[period_type].append({"period": period, "count": count})
    hourly_trend = [
        {"hour": hour, "count": count}
        for hour, count in sorted(hourly_trend_by_hour.items())
    ]
    peak_hour_trend = []
    for day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']:
        peak_hour = peak_hour_trend_by_day.get(day)
        peak_hour_trend.append({"day": day, "peakHour": peak_hour if peak_hour is not None else None})
    # --- Dwell time trends: daily, weekly, monthly ---
    dwell_time_trend_by_date = defaultdict(list)
    dwell_time_trend_by_week = defaultdict(list)
    dwell_time_trend_by_month = defaultdict(list)
    for job_id in os.listdir(RESULTS_FOLDER):
        analytics_path = os.path.join(RESULTS_FOLDER, job_id, f"{job_id}_analytics.json")
        if os.path.exists(analytics_path):
            with open(analytics_path, "r") as f:
                try:
                    analytics = json.load(f)
                except Exception:
                    continue
            video_date = analytics.get("video_date")
            avg_dwell = analytics.get("average_dwell_time", 0)
            if video_date:
                dwell_time_trend_by_date[video_date].append(avg_dwell)
                # Parse date for week/month
                try:
                    dt = datetime.datetime.strptime(video_date, "%Y-%m-%d")
                    week = f"{dt.isocalendar()[0]}-W{dt.isocalendar()[1]}"
                    month = dt.strftime("%Y-%m")
                    dwell_time_trend_by_week[week].append(avg_dwell)
                    dwell_time_trend_by_month[month].append(avg_dwell)
                except Exception:
                    pass
    dwell_time_trend_daily = [
        {"date": date, "dwellTime": np.mean(times)}
        for date, times in sorted(dwell_time_trend_by_date.items())
    ]
    dwell_time_trend_weekly = [
        {"week": week, "dwellTime": np.mean(times)}
        for week, times in sorted(dwell_time_trend_by_week.items())
    ]
    dwell_time_trend_monthly = [
        {"month": month, "dwellTime": np.mean(times)}
        for month, times in sorted(dwell_time_trend_by_month.items())
    ]
    all_dwell_times = [v for times in dwell_time_trend_by_date.values() for v in times]
    average_dwell_time = np.mean(all_dwell_times) if all_dwell_times else 0
    total_visitors = len(all_unique_track_ids)
    # --- Unique visitor aggregation per period ---
    daily_trackids = defaultdict(set)
    weekly_trackids = defaultdict(set)
    monthly_trackids = defaultdict(set)
    hourly_trackids = defaultdict(set)
    for job_id in os.listdir(RESULTS_FOLDER):
        analytics_path = os.path.join(RESULTS_FOLDER, job_id, f"{job_id}_analytics.json")
        if os.path.exists(analytics_path):
            with open(analytics_path, "r") as f:
                try:
                    analytics = json.load(f)
                except Exception:
                    continue
            video_date = analytics.get("video_date")
            track_ids = set(analytics.get("unique_track_ids", []))
            if video_date:
                # Daily
                daily_trackids[video_date].update(track_ids)
                # Weekly
                try:
                    dt = datetime.datetime.strptime(video_date, "%Y-%m-%d")
                    week = f"{dt.isocalendar()[0]}-W{dt.isocalendar()[1]}"
                    weekly_trackids[week].update(track_ids)
                    month = dt.strftime("%Y-%m")
                    monthly_trackids[month].update(track_ids)
                except Exception:
                    pass
            # Hourly (if available)
            for entry in analytics.get("hourly_trend", []):
                hour = entry.get("hour")
                if hour is not None:
                    hourly_trackids[hour].update(track_ids)
    visitor_count = {
        "daily": [{"period": day, "count": len(tids)} for day, tids in sorted(daily_trackids.items())],
        "weekly": [{"period": week, "count": len(tids)} for week, tids in sorted(weekly_trackids.items())],
        "monthly": [{"period": month, "count": len(tids)} for month, tids in sorted(monthly_trackids.items())],
    }
    hourly_trend = [{"hour": hour, "count": len(tids)} for hour, tids in sorted(hourly_trackids.items())]
    return {
        "total_visitors": total_visitors,
        "visitor_count": visitor_count,
        "hourly_trend": hourly_trend,
        "peak_hour_trend": peak_hour_trend,
        "processed_videos": processed_videos,
        "generated_heatmaps": generated_heatmaps,
        "dwell_time_trend_daily": dwell_time_trend_daily,
        "dwell_time_trend_weekly": dwell_time_trend_weekly,
        "dwell_time_trend_monthly": dwell_time_trend_monthly,
        "average_dwell_time": average_dwell_time,
        "dwell_time_trend_change": 0  # This is a placeholder, actual calculation needed
    }

@app.route('/api/dashboard_analytics', methods=['GET'])
@jwt_required()
def get_dashboard_analytics():
    data = get_dashboard_analytics_data()
    return jsonify(data)

@app.route('/api/dashboard_export', methods=['GET'])
@jwt_required()
def dashboard_export():
    from flask import make_response
    import io
    import csv
    import tempfile
    import datetime
    try:
        print("[DEBUG] Authorization header:", request.headers.get('Authorization'))
        print("[DEBUG] JWT Identity:", get_jwt_identity())
        format = request.args.get('format', 'csv')
        print(f"[DEBUG] Requested export format: {format}")
        data = get_dashboard_analytics_data()
        print("[DEBUG] Got analytics data")
        if format == 'csv':
            output = io.StringIO()
            writer = csv.writer(output)
            writer.writerow(['Metric', 'Value'])
            writer.writerow(['Total Visitors', data['total_visitors']])
            writer.writerow(['Processed Videos', data['processed_videos']])
            writer.writerow(['Generated Heatmaps', data['generated_heatmaps']])
            writer.writerow([])
            writer.writerow(['Visitor Count'])
            writer.writerow(['Date/Period', 'Count'])
            for period_type in ['daily', 'weekly', 'monthly']:
                for row in data['visitor_count'].get(period_type, []):
                    writer.writerow([row.get('period', ''), row.get('count', '')])
            writer.writerow([])
            writer.writerow(['Dwell Time Trend (Daily)'])
            writer.writerow(['Date', 'Avg. Dwell Time (min)'])
            for row in data.get('dwell_time_trend_daily', []):
                writer.writerow([row.get('date', ''), f"{row.get('dwellTime', 0):.2f}"])
            writer.writerow([])
            writer.writerow(['Dwell Time Trend (Weekly)'])
            writer.writerow(['Week', 'Avg. Dwell Time (min)'])
            for row in data.get('dwell_time_trend_weekly', []):
                writer.writerow([row.get('week', ''), f"{row.get('dwellTime', 0):.2f}"])
            writer.writerow([])
            writer.writerow(['Dwell Time Trend (Monthly)'])
            writer.writerow(['Month', 'Avg. Dwell Time (min)'])
            for row in data.get('dwell_time_trend_monthly', []):
                writer.writerow([row.get('month', ''), f"{row.get('dwellTime', 0):.2f}"])
            writer.writerow([])
            writer.writerow(['Hourly Trend'])
            writer.writerow(['Hour', 'Count'])
            for row in data['hourly_trend']:
                writer.writerow([row.get('hour', ''), row.get('count', '')])
            output.seek(0)
            response = make_response(output.getvalue())
            response.headers["Content-Disposition"] = "attachment; filename=dashboard_export.csv"
            response.headers["Content-type"] = "text/csv"
            print("[DEBUG] Returning CSV export response")
            return response
        elif format == 'pdf':
            from fpdf import FPDF
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            pdf.cell(200, 10, txt="RetailSense Dashboard Report", ln=True, align='C')
            pdf.ln(5)
            pdf.cell(200, 10, txt=f"Total Visitors: {data['total_visitors']}", ln=True)
            pdf.cell(200, 10, txt=f"Processed Videos: {data['processed_videos']}", ln=True)
            pdf.cell(200, 10, txt=f"Generated Heatmaps: {data['generated_heatmaps']}", ln=True)
            pdf.ln(5)
            pdf.cell(200, 10, txt="Visitor Count", ln=True)
            for period_type in ['daily', 'weekly', 'monthly']:
                for row in data['visitor_count'].get(period_type, []):
                    pdf.cell(200, 10, txt=f"{row.get('period', '')}: {row.get('count', '')}", ln=True)
            pdf.ln(5)
            pdf.cell(200, 10, txt="Dwell Time Trend (Daily)", ln=True)
            for row in data.get('dwell_time_trend_daily', []):
                pdf.cell(200, 10, txt=f"{row.get('date', '')}: {row.get('dwellTime', 0):.2f} min", ln=True)
            pdf.ln(5)
            pdf.cell(200, 10, txt="Dwell Time Trend (Weekly)", ln=True)
            for row in data.get('dwell_time_trend_weekly', []):
                pdf.cell(200, 10, txt=f"{row.get('week', '')}: {row.get('dwellTime', 0):.2f} min", ln=True)
            pdf.ln(5)
            pdf.cell(200, 10, txt="Dwell Time Trend (Monthly)", ln=True)
            for row in data.get('dwell_time_trend_monthly', []):
                pdf.cell(200, 10, txt=f"{row.get('month', '')}: {row.get('dwellTime', 0):.2f} min", ln=True)
            pdf.ln(5)
            pdf.cell(200, 10, txt="Hourly Trend", ln=True)
            for row in data['hourly_trend']:
                pdf.cell(200, 10, txt=f"{row.get('hour', '')}: {row.get('count', '')}", ln=True)
            pdf_output = pdf.output(dest='S').encode('latin1')
            response = make_response(pdf_output)
            response.headers["Content-Disposition"] = "attachment; filename=dashboard_export.pdf"
            response.headers["Content-type"] = "application/pdf"
            print("[DEBUG] Returning PDF export response")
            return response
        else:
            print("[DEBUG] Invalid format requested")
            return jsonify({"error": "Invalid format. Use 'csv' or 'pdf'."}), 400
    except Exception as e:
        print("[DEBUG] Exception in dashboard_export:", str(e))
        return jsonify({"msg": "Internal error", "error": str(e)}), 500

if __name__ == '__main__':
    init_db()
    app.run(host='0.0.0.0', port=5000, debug=True) 