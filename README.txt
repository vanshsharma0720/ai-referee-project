# AI NFL Referee

Computer vision system that analyzes football video to detect basic game events such as snap moments, out-of-bounds plays, and potential touchdowns using object detection and motion heuristics.

# Overview:
This project uses YOLOv8 to detect players in video frames and applies rule-based logic to estimate field position and identify events. It is not trained on football-specific data and instead relies on general object detection plus custom tracking rules.

# Features:
Player detection using YOLOv8
Motion-based snap detection
Line of scrimmage estimation from player positions
Out-of-bounds detection using field margins
Endzone (touchdown candidate) detection
Simple movement-based offside check
Real-time video overlay visualization

# Requirements:
Python 3.10+
OpenCV
NumPy
Ultralytics

# Install dependencies:
pip install opencv-python numpy ultralytics

# Run:
python referee.py
Press q to exit.

# Limitations:
No object tracking model (frame-by-frame detection only)
Player identity is not preserved across frames
Event detection is rule-based, not learned from data
Works best on stable broadcast-style footage

# Future Work:
Add multi-object tracking (ByteTrack/DeepSORT)
Improve snap detection accuracy
Add custom-trained football detection model
Export event logs and stats per play
