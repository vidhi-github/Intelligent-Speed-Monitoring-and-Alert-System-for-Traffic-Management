# Intelligent Speed Monitoring and Alert System for Traffic Management

## Project Description

The **Intelligent Speed Monitoring and Alert System for Traffic Management** is designed to monitor vehicle speeds, detect violations, and automatically recognize vehicle number plates in real-time. The system aims to capture frames when speed violations occur, and email the concerned authority with the captured frame and vehicle number. This project was developed during the **NVIDIA GRIL training** program, which is a comprehensive AI-focused training by NVIDIA. The program empowers developers with hands-on experience in leveraging high-performance computing resources and deep learning frameworks like PyTorch for solving real-world problems. The project uses NVIDIA DGX systems, which are specifically built for accelerated AI computations, making it an ideal platform for this application.

## Project Aim

The goal of this project is to create an intelligent traffic management system capable of:

- Tracking vehicles and detecting their speed.
- Marking violations when a vehicle exceeds the speed threshold.
- Capturing the frame of the vehicle at the time of violation.
- Automatically recognizing the vehicle’s number plate.
- Sending an email to the concerned authority with the captured frame and vehicle number as an attachment.

## Requirements

To set up the project, the following tools and technologies are required:

- **DGX Server**: This powerful server is equipped with high-end NVIDIA GPUs, ideal for processing large volumes of video data and training AI models quickly.
- **Technologies**:
  - **Python 3.8**: Primary language used for backend development and real-time processing.
  - **Flask**: Lightweight web framework for deploying the application as a web service.
  - **PyTorch**: Deep learning framework used for training and deploying the vehicle number plate recognition model.
  - **OpenCV**: Used for real-time video frame processing to detect vehicles and capture frames of violations.
  - **SCP and SSH**: Secure file transfer and remote access tools for managing files and working on the DGX server.

## Technologies Used

1. **Python 3.8**: Python serves as the core programming language for the system, handling video processing, speed detection, violation marking, and number plate recognition.
2. **Flask**: A web framework that makes it easy to deploy the system as a service, where users can monitor violations and receive alerts.
3. **PyTorch**: Used to train the deep learning model for vehicle number plate recognition. PyTorch’s powerful GPU capabilities make it suitable for AI applications.
4. **OpenCV**: This library enables real-time video processing for vehicle tracking, speed detection, and violation capture.
5. **NVIDIA DGX Server**: With powerful GPUs, the DGX system accelerates the training and deployment of AI models, allowing the system to process high-resolution video frames in real-time.
6. **Email System**: Integrated into the system to send automatic violation alerts to the concerned authorities with the captured image and vehicle number.

## Setup and Running the Project

### 1. Access DGX Server

To begin, access the DGX server by entering its URL in ur browser.

### 2. Login

Log in using the **username** and **password** provided for accessing the DGX system.

### 3. Setting Up the Environment

- RAM: 6GB, CPU: 8 cores, GPU: 1 (with 20GB memory), Environment: **test**
- **PyTorch Image**: Use the appropriate image for PyTorch.
- **Flask Port**: 5000 (default).

### 4. SSH into DGX Server

To interact with the DGX system remotely, use the following SSH command:
ssh -p "ssh port number" username @ "dgx portaddress"

### 5. Transfer Files to DGX Server

Use the **SCP** command to transfer files from your local machine to the DGX server. Example command:

scp -P "port number" -r "user desktop file address" username@"dgx portaddress":"dgx system file address"

This command will send the folder from your local desktop to the `desktop` directory on the DGX server.

### 6. Set Up the Python Virtual Environment

On the DGX server, run the following commands to update the system and set up the virtual environment:
sudo apt update || sudo apt install python3.8-venv || python -m venv newenv

### 7. Install Project Dependencies

Activate the virtual environment and install the necessary dependencies from the `requirements.txt` file:

source newenv/bin/activate || pip install -r requirements.txt

### 8. Run the Application

After the environment is set up and dependencies are installed, start the application by running:

python3 app.py

The system will now process live video streams, detect vehicle speeds, mark violations, and trigger email alerts when a violation is detected.

## Annotations and Data Preparation

### Car Annotations

Annotations are a crucial part of the vehicle detection and number plate recognition process. Annotations involve marking the vehicle's position in the image (bounding box) and labeling the object as a vehicle. This annotated data is used to train the model to recognize vehicles and number plates.

To annotate a car:

1. Draw a bounding box around the vehicle.
2. Label the car with its class (e.g., "car").
3. Annotate multiple images to create a diverse dataset for training the model.

The annotated images are then used to train a vehicle detection model that is capable of recognizing and tracking vehicles in real-time.

## Conclusion

This project demonstrates the ability to monitor and manage traffic more efficiently by detecting speed violations and recognizing vehicle number plates automatically. The system leverages the power of NVIDIA’s DGX servers, equipped with GPUs, to accelerate the processing of video feeds and the training of deep learning models. By implementing OpenCV for video frame processing and PyTorch for number plate recognition, this project successfully automates the task of tracking traffic violations, reducing the need for manual monitoring.

From this project, we gained practical experience in building a real-time application for traffic management, using AI to enhance road safety, and leveraging powerful GPU systems for faster model training and deployment.

## Thank You for going through my project.
Have a great learning!!

Built by: Vidhi Jindal









