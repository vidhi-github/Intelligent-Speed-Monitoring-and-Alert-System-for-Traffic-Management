import cv2
import torch
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
import time
import sys
from pathlib import Path
from PIL import Image
import threading
# import send_mail
import time
from send_mail import prepare_and_send_email
sys.path.insert(0, str(Path(__file__).resolve().parent / "yolov9"))
from yolov9.models.common import DetectMultiBackend, AutoShape

import yolov9.detect_dual_anpr

# Global Variables
is_email_allowed = False  # When user checks the email checkbox, this variable will be set to True
send_next_email = True  # We have to wait for 10 minutes before sending another email
# detections_summary will be used to store the detections summary report
detections_summary = ''
email_sender = '57224_yogeshtiwari@gbpuat-tech.ac.in'
email_recipient = 'yogeshtiwari733@gmail.com'

# global message
# message = 'Vehicle exceeded the speed limit of 150 km/h'

def violation_alert_generator(im0, message, subject='Speed Violation Detected'):
    '''This function will send an email with attached alert image and then wait for 10 minutes before sending another email
    
    Parameters:
      im0 (numpy.ndarray): The image to be attached in the email
      subject (str): The subject of the email
      message (str): The message text of the email

    Returns:
      None
    '''
    global send_next_email, email_recipient

    message_text = message
    send_next_email = False  # Set flag to False so that another email is not sent
    print('Sending email alert to ', email_recipient)
    prepare_and_send_email(email_sender, email_recipient, subject, message_text, im0)
    print('Mail sent')
    # Wait for 10 minutes before sending another email
    time.sleep(10)
    send_next_email = True


def draw_corner_rect(img, bbox, line_length=30, line_thickness=5, rect_thickness=1,
                     rect_color=(255, 0, 255), line_color=(0, 255, 0)):
    x, y, w, h = bbox
    x1, y1 = x + w, y + h

    if rect_thickness != 0:
        cv2.rectangle(img, bbox, rect_color, rect_thickness)

    # Top Left  x, y
    cv2.line(img, (x, y), (x + line_length, y), line_color, line_thickness)
    cv2.line(img, (x, y), (x, y + line_length), line_color, line_thickness)

    # Top Right  x1, y
    cv2.line(img, (x1, y), (x1 - line_length, y), line_color, line_thickness)
    cv2.line(img, (x1, y), (x1, y + line_length), line_color, line_thickness)

    # Bottom Left  x, y1
    cv2.line(img, (x, y1), (x + line_length, y1), line_color, line_thickness)
    cv2.line(img, (x, y1), (x, y1 - line_length), line_color, line_thickness)

    # Bottom Right  x1, y1
    cv2.line(img, (x1, y1), (x1 - line_length, y1), line_color, line_thickness)
    cv2.line(img, (x1, y1), (x1, y1 - line_length), line_color, line_thickness)

    return img  

def calculate_speed(distance, fps):
    return (distance * fps) * 3.6

def calculate_distance(p1, p2):
    return np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

def read_frames(cap):
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        yield frame 


def video_detection(conf_=0.50, frames_buffer=[], vid_path=None):
    '''This function will detect vehicles, display their numbers and speed in the video file or a live stream.

    Parameters:
        conf_ (float): Confidence threshold for inference
        frames_buffer (list): A list of frames to be processed
        vid_path (str): Path to the video file

    Returns:
        None
    '''
    global send_next_email
    global is_email_allowed
    global email_recipient
    global detections_summary

    # Constants for perspective transformation
    FRAME_WIDTH=30
    FRAME_HEIGHT=100

    # Define source and bird eye view polygons for perspective transformation
    SOURCE_POLYGONE = np.array([[16, 674], [1815, 772],[1383, 294], [578, 295]], dtype=np.float32)
    BIRD_EYE_VIEW = np.array([[0, 0], [FRAME_WIDTH, 0], [FRAME_WIDTH, FRAME_HEIGHT],[0, FRAME_HEIGHT]], dtype=np.float32)

    M = cv2.getPerspectiveTransform(SOURCE_POLYGONE, BIRD_EYE_VIEW)
    
    speed_violation_frames = 0
    
    BLUR_ID = None
    CLASS_ID = 2
    # Initialize dictionaries to track positions and speeds
    prev_positions = {}
    speed_accumulator = {}
    unique_labels = {None, ''}

    while True:
        if len(frames_buffer) > 1:
            _ = frames_buffer.pop(0)
        break
    
    torch.cuda.empty_cache()

    with torch.no_grad():
        torch.backends.cudnn.benchmark = True
        device = torch.device( 'cuda' if torch.cuda.is_available() else 'cpu')
        model = DetectMultiBackend(weights='best.pt', device=device, fuse=True)
        model = AutoShape(model)
        if device == 'cuda':
            torch.cuda.synchronize()

        tracker = DeepSort(max_age=25)
        
        # Load the COCO class labels
        classes_path = "configs/coco.names"
        with open(classes_path, "r") as f:
            class_names = f.read().strip().split("\n")
        
        VEHICLE_CLASS_IDS = {2}
            
        # Create a list of random colors to represent each class
        np.random.seed(42)
        colors = np.random.randint(0, 255, size=(len(class_names), 3)) 
        
        # FPS calculation variables
        # frame_count = 0
        # start_time = time.time()
        prev_positions={}
        speed_accumulator={}

        try:
            cap = cv2.VideoCapture(vid_path)
            if not cap.isOpened():
                print('Error: Unable to open video source.')
                return

            frame_generator = read_frames(cap)
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            print(f"fps value is: ", fps) 
            
            #Create mask to filter detections
            pts = SOURCE_POLYGONE.astype(np.int32) 
            pts = pts.reshape((-1, 1, 2))
            polygon_mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
            cv2.fillPoly(polygon_mask, [pts], 255)
            
            start_time = time.time()
            while True:
                try:
                    img0 = next(frame_generator)
                    #img0 = frames_buffer.pop(0)
                except StopIteration:
                    break
                
                 # Clear the frames buffer if it gets too large
                if len(frames_buffer) > 1:
                    frames_buffer.clear()
                

                # Add the current frame to the buffer
                frames_buffer.append(img0)
                #img0 = cv2.resize(img0, (640, 640))
                results = model(img0)
                detect = []
                for det in results.pred[0]:
                    label, confidence, bbox = det[5], det[4], det[:4]
                    x1, y1, x2, y2 = map(int, bbox)
                    class_id = int(label)
                    # print(class_id)

                    # Filter out weak detections by confidence threshold and class_id
                    # if CLASS_ID == 2:
                    #     if confidence < conf_:
                    #         continue
                    # else:
                    #     if class_id != CLASS_ID or confidence < conf_:
                    #         continue      
                    
                    if class_id in VEHICLE_CLASS_IDS and confidence >= conf_:
                        if polygon_mask[(y1 + y2) // 2, (x1 + x2) // 2] == 255:
                            detect.append([[x1, y1, x2 - x1, y2 - y1], confidence, int(label)]) 

                tracks = tracker.update_tracks(detect, frame=img0)
                
                speed = 0
                
                color = (255, 0, 0)
                B, G, R = map(int, color)
                #frame = draw_corner_rect(img0, (x1, y1, x2 - x1, y2 - y1), line_length=15, line_thickness=3, rect_thickness=1, rect_color=(B, G, R), line_color=(R, G, B))
                #cv2.polylines(frame, [pts], isClosed=True, color=(255, 0, 0), thickness=2)               

                for track in tracks:
                    if not track.is_confirmed():
                       continue

                    track_id = track.track_id
                    ltrb = track.to_ltrb()
                    class_id = track.get_det_class()
                    x1, y1, x2, y2 = map(int, ltrb)
                    
                    if polygon_mask[(y1+y2)//2,(x1+x2)//2] == 0:
                        tracks.remove(track)

                    # color = colors[class_id]
                    color = (255, 0, 0)
                    B, G, R = map(int, color)
                    text = f"{track_id} - {class_names[class_id]}"
                    center_pt = np.array([[(x1+x2)//2, (y1+y2)//2]], dtype=np.float32)
                    transformed_pt = cv2.perspectiveTransform(center_pt[None, :, :], M)
                    
                    #Draw bounding box and text
                    frame = draw_corner_rect(img0, (x1, y1, x2 - x1, y2 - y1), line_length=15, line_thickness=3, rect_thickness=1, rect_color=(B, G, R), line_color=(R, G, B))
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (B, G, R), 2)
                    cv2.rectangle(frame, (x1 - 1, y1 - 20), (x1 + len(text) * 10, y1), (B, G, R), -1)
                    cv2.putText(frame, text, (x1 + 5, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    #cv2.polylines(frame, [pts], isClosed=True, color=(255, 0, 0), thickness=2)

                    #Process distance and speed calculations by using previous predictions
                    if track_id in prev_positions:
                        prev_position = prev_positions[track_id]
                        distance = calculate_distance(prev_position, transformed_pt[0][0])
                        speed = calculate_speed(distance, fps)
                        if track_id in speed_accumulator:
                            speed_accumulator[track_id].append(speed)
                            if len(speed_accumulator[track_id]) > 100:
                                speed_accumulator[track_id].pop(0)
                        else:
                            speed_accumulator[track_id] = []
                            speed_accumulator[track_id].append(speed)
                    prev_positions[track_id] = transformed_pt[0][0]
                    
                    
                    if track_id in speed_accumulator :
                        avg_speed = sum(speed_accumulator[track_id]) / len(speed_accumulator[track_id])
                        cv2.rectangle(frame, (x1 - 1, y1-40 ), (x1 + len(f"Speed: {avg_speed:.0f} km/h") * 10, y1-20), (0, 0, 255), -1)
                        cv2.putText(frame, f"Speed: {avg_speed:.0f} km/h", (x1, y1 - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                        # print(avg_speed)
                        speed = avg_speed
                        if avg_speed > 25 and is_email_allowed:
                            speed_violation_frames += 1

                            if speed_violation_frames >= 5 and send_next_email:
                                speed_violation_frames = 0

                                # Save img0 as an image file
                                img_path = 'detect_img.jpg'
                                cv2.imwrite(img_path, img0)
                                x = x1
                                y = y1
                                w = x2- x1
                                h = y2-y1
                                x1 = x + w
                                y1 = y + h

                                x=int(x) 
                                y=int(y) 
                                x1=int(x1)
                                y1=int(y1)

                                image = Image.open('detect_img.jpg')

                                crop_box = (x,y,x1,y1)
                                cropped_image = image.crop(crop_box)
                                cropped_image.save('path.jpg')
                                cropped_path = 'path.jpg'
                                labels = yolov9.detect_dual_anpr.run(weights='yolov9/anpr_best.pt', source=cropped_path, conf_thres=0.1)
                                message = 'Vehicle exceeded the speed limit of 25 km/h' 
                                if labels not in unique_labels:
                                    unique_labels.add(labels)
                                    message += '\n' + labels
                                    print(message)
                                    print("#######################################")
                                    print(unique_labels)
                                    t = threading.Thread(target=violation_alert_generator, args=(img0, message))
                                    t.start()
                        else:
                            speed_violation_frames = 0
                            
                    # Apply Gaussian Blur
                        if BLUR_ID is not None and class_id == BLUR_ID:
                            if 0 <= x1 < x2 <= frame.shape[1] and 0 <= y1 < y2 <= frame.shape[0]:
                                frame[y1:y2, x1:x2] = cv2.GaussianBlur(frame[y1:y2, x1:x2], (99, 99), 3)

                    # Update the detections summary
                    current_time = time.strftime("%H:%M:%S", time.localtime())
                    detections_summary += f"\n{current_time}\nVehicle {track_id}\nSpeed: {speed:.2f} km/h\n###########\n"

                yield img0, len(tracks)
            
            end_time = time.time()
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            total_time = end_time - start_time
            processing_fps = frame_count/total_time

            print(f"Original FPS: {fps:.2f}")
            print(f"Processed FPS (inference speed): {processing_fps:.2f}")
            print(f"Total inference time for the entire video: {total_time:.2f} seconds")

            cap.release()

        except Exception as e:
            print(e)
