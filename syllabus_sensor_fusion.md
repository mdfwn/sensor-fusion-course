Below is the syllabus converted to Markdown. It’s a direct OCR-based extraction, so you may spot an occasional typo or odd line-break; feel free to let me know if you’d like any clean-ups or structural tweaks.


# Sensor Fusion  
Syllabus  

---

# Sensor Fusion  
Part of Accenture  

## Overview  
The Sensor Fusion Engineer Nanodegree program consists of four courses that teach the fundamentals of sensor fusion and perception for self-driving cars. The program covers lidar, radar, camera, and Kalman filters, and includes lessons on working with real-world data, filtering, segmentation, clustering, and object tracking. In addition, students will work on a capstone project that involves building a complete sensor fusion pipeline for autonomous vehicles. Upon completing the program, graduates will have the skills and knowledge necessary to design and implement sensor fusion systems for self-driving cars.  

**Nanodegree Program • Advanced • ≈ 63 hours • ★ 4.8 (184 Reviews)**  

### Prerequisites  
- Intermediate C++ and data structures  
- Basic linear algebra & probability  
- Basic calculus  
- Introductory physics  
- Linux & command-line tools  

---

# Course 1 – Lidar  
### Lidar Obstacle Detection  
- Introduction to Lidar and Point Clouds  
- Point clouds • Point cloud data • RANSAC algorithm • Curve fitting • Object clustering • K-d trees • Point Cloud Library • Lidar visualization • Lidar data representation • Lidar simulator • Euclidean clustering • Bounding boxes  

---

# Course 2 – Camera  
### Camera-based Tracking & Detection  
- Sensor fusion • Digital cameras • Digital image key-point descriptors • Object tracking • SIFT algorithm • Autonomous-vehicle fluency • SAE J3016 levels of driving-automation standard • Autonomous-vehicle sensor selection • Object motion models • Computer-vision image filtering • Image-gradient calculation • Object detection • YOLO algorithm • Digital-image key-points • Computer-vision image transformations • Lidar measurement models • Harris detector • Camera measurement models • Feature detection • Collision detection • OpenCV  

---

# Course 3 – Radar  
### Radar Clustering & Tracking  
- Radar basics • Radar clutter thresholding • Fourier transforms • Radar angle of arrival • Cluster models • Multi-target object tracking • Radar range resolution • Radar velocity estimation • Doppler effect • Constant False Alarm Rate (CFAR) • Radar coordinate frames • Radar cross-section • Radar reflection models • Radar data association • Adaptive signal processing • Radar filtering  

---

# Course 4 – Kalman Filters  
### Multi-Sensor Tracking  
- Kalman Filter fundamentals • Extended & Unscented KF • Sensor modeling • Prediction & update steps • Linear vs. nonlinear systems • Multi-sensor fusion • Coordinate-frame transformations • Track-to-track fusion • Process-noise tuning • Measurement-noise modeling • Joint probabilistic data association • Motion-model selection • Imm & Ekf tracking • Sensor mis-alignment compensation  

---

# Capstone Project – Sensor Fusion Pipeline  
Build an end-to-end perception pipeline that fuses lidar, radar, and camera data to detect, classify, and track surrounding objects in real-time. You will:  
- Integrate heterogeneous sensor streams  
- Apply filtering techniques; extract corners, infer features of a scene  
- Implement data association across sensor modalities  
- Maintain object tracks through occlusion and disappearance  
- Evaluate performance with precision, recall, and RMSE metrics  

---

## Why C++?  
(Original syllabus emphasizes C++ for real-time performance; in our Python-based redesign we will lean on optimized libraries such as NumPy, Open3D, OpenCV-Python, and filterpy to achieve near-native speed.)  

## Instructors  
- **David Silver** – Senior Curriculum Lead  
- **Andreas Haja, PhD** – Autonomous-driving researcher & educator  
- **Stephen Welch** – Computer-vision engineer & instructor  

---

## Student Services & Support  
- Access to mentor network for technical questions  
- Project reviews with personalized feedback  
- Career services: résumé review, LinkedIn optimization, GitHub portfolio tips  
- Dedicated community channel for peer discussion and collaboration  

---

## Program Outcomes  
Upon completion, students will be able to:  
1. Design and implement lidar segmentation & clustering algorithms.  
2. Develop camera-based object-tracking systems with feature descriptors.  
3. Model radar returns and perform CFAR detection.  
4. Build and tune multi-sensor Kalman-filter pipelines.  
5. Deploy a real-time sensor-fusion perception stack for autonomous vehicles.  

---

_End of syllabus_


