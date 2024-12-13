'''A Flask application to run the YOLimport pyautogui
Ov7 PPE violation model on a video file or ip cam stream

'''

import os
import validators
from flask import Flask, render_template, request, Response
import hubconfCustom
from hubconfCustom import video_detection
import cv2
import time

# Initialize the Flask application
app = Flask(__name__, static_folder='static')
app.config["VIDEO_UPLOADS"] = "static/video"
app.config["ALLOWED_VIDEO_EXTENSIONS"] = ["MP4", "MOV", "AVI", "WMV"]

# Secret key for the session
app.config['SECRET_KEY'] = 'speed_violation_detection'

# Global variables
frames_buffer = []  # Buffer to store frames from a stream
vid_path = os.path.join(app.config["VIDEO_UPLOADS"], 'vid.mp4')  # Path to uploaded/stored video file
video_frames = cv2.VideoCapture(vid_path)  # Video capture object


def allowed_video(filename: str):
    if "." not in filename:
        return False
    extension = filename.rsplit(".", 1)[1].upper()
    return extension in app.config["ALLOWED_VIDEO_EXTENSIONS"]

def generate_raw_frames():
    global video_frames, frames_buffer
    while True:
        success, frame = video_frames.read()
        if not success:
            break
        frames_buffer.append(frame.copy())
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def generate_processed_frames(conf_=0.50):
    global frames_buffer, vid_path, video_frames

    yolo_output = video_detection(conf_, frames_buffer, vid_path)
         
    for detection_, _ in yolo_output:
        _, buffer = cv2.imencode('.jpg', detection_)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video_raw')
def video_raw():
    return Response(generate_raw_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_processed')
def video_processed():
    conf = 0.75
    return Response(generate_processed_frames(conf_=conf), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/', methods=["GET", "POST"])
def index():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit_form():
    global vid_path, video_frames, frames_buffer

    if request.method == "POST":
        if request.files:
            video = request.files['video']
            if video.filename == '':
                return "That video must have a file name"
            elif not allowed_video(video.filename):
                return "Unsupported video. The video file must be in MP4, MOV, AVI, or WMV format."
            else:
                filename = 'vid.mp4'
                if video.content_length > 200 * 1024 * 1024:
                    return "Error! That video is too large"
                else:
                    try:
                        video.save(os.path.join(app.config["VIDEO_UPLOADS"], filename))
                        return "That video is successfully uploaded"
                    except Exception as e:
                        return "Error! The video could not be saved"
        
        if 'download_button' in request.form:
            with open('static/reports/detections_summary.txt', 'w') as f:
                f.write(hubconfCustom.detections_summary)
            return Response(open('static/reports/detections_summary.txt', 'rb').read(),
                            mimetype='text/plain',
                            headers={"Content-Disposition": "attachment;filename=detections_summary.txt"})
        
        if 'alert_email_checkbox' in request.form:
            email_checkbox_value = request.form['alert_email_checkbox']
            if email_checkbox_value == 'false':
                hubconfCustom.is_email_allowed = False
                return "Alert email is disabled"
            else:
                hubconfCustom.is_email_allowed = True
                hubconfCustom.send_next_email = True
                hubconfCustom.email_recipient = request.form['alert_email_textbox']
                return f"Alert email is enabled at {hubconfCustom.email_recipient}. Violation alert(s) with a gap of 10 minutes will be sent"
        
        if 'inference_video_button' in request.form:
            vid_path = os.path.join(app.config["VIDEO_UPLOADS"], 'vid.mp4')
            video_frames = cv2.VideoCapture(vid_path)
            frames_buffer.clear()
            if not video_frames.isOpened():
                return 'Error in opening video', 500
            else:
                frames_buffer.clear()
                return 'success'
        
        if 'live_inference_button' in request.form:
            vid_ip_path = request.form['live_inference_textbox']
            if validators.url(vid_ip_path):
                vid_path = vid_ip_path.strip()
                video_frames = cv2.VideoCapture(vid_path)
                if not video_frames.isOpened():
                    return 'Error: Cannot connect to live stream', 500
                else:
                    frames_buffer.clear()
                    return 'success'
            else:
                return 'Error: Entered URL is invalid', 500



if __name__ == "__main__":
    #copy file from static/files/vid.mp4 to static/video/vid.mp4
    os.system('cp static/files/vid.mp4 static/video/vid.mp4')
    # app.run(debug=True)
    app.run(host='0.0.0.0',port=5000,debug=True)

