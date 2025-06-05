# 📋 To-Do List — Build an Interactive **Python-First Sensor Fusion** Course  
_This checklist is meant for a coding-capable LLM that will execute tasks **sequentially from top to bottom**.  
Stick to the order; do **not** skip steps.  
Each [] item may be decomposed into smaller functions/files as needed._  

[]: Undone Tasks
[x]: Done Tasks

Maintain this ToDo yourself - work from top to bottom. Check off tasks as you complete them.

---

## 0 · Project Bootstrap
- [x] **Initialize repo & tooling**
  - Create a Git repo with 2 main folders: `/src` (HTML/CSS/JS) and `/content` (Markdown → HTML).  
  - Use **MyST-Markdown** or **Jupyter Book** for easy math (`$$…$$`) rendering and code execution blocks.  
  - Add `requirements.txt` with: `numpy`, `scipy`, `matplotlib`, `open3d`, `opencv-python`, `filterpy`, `scikit-image`, `plotly`, `jupyter-book`, `myst-parser`, `sphinx-togglebutton`, `sphinx-copybutton`.  
  - Configure **Sphinx** theme = `furo` (or another modern responsive theme) with dark-mode toggle.

- [x] **Global HTML layout**
  - Create a base template (`base.html`) containing:  
    - Mobile-first navbar (hamburger on small screens) linking to all courses & capstone.  
    - Sidebar with progress tracker (JS localStorage).  
    - Built-in code renderer (highlight.js) + "copy ⇩" button.  
    - MathJax config for LaTeX.  
  - Add CSS (`tailwind.css` or vanilla) for a clean, airy design (max-width 920 px, ample line-height).

---

## 1 · Front-Matter Pages
- [x] **`index.md`** — program overview  
  - Brief description, learning goals, estimated hours, prerequisites (Python, linear algebra, probability, calculus, Linux).  
- [x] **`how_to_use.md`** — instructions on interactive notebooks, code sandboxing, quizzes, and progress tracking.  
- [x] **`resources.md`** — links to official docs for NumPy, OpenCV, Open3D, etc.

---

## 2 · Course 1 — **Lidar: Obstacle Detection**
Create folder `/content/lidar/`.

> **For every lesson below:**  
> - Start with a short intuitive explainer.  
> - Present key equation(s) in LaTeX.  
> - Provide runnable Python example(s).  
> - End with 3–5 quiz Q&A + 1 mini-exercise notebook.

1. [x] **Introduction to Lidar & Point Clouds**  
2. [x] **Parsing PCD/PLY Files** (Open3D)  
3. [x] **RANSAC Plane Fitting** (line & plane equations)  
4. [x] **Euclidean Clustering with KD-Tree**  
5. [] **Bounding-Box Generation** (AABB vs OBB)  
6. [] **Interactive Lidar Visualizer** (Plotly 3-D)  
7. [] **Lesson Summary + Quiz**

---

## 3 · Course 2 — **Camera: Tracking & Detection**
Folder `/content/camera/`.

1. [] **Camera Models & Calibration**  
2. [] **Image Gradients & Filtering**  
3. [] **Feature Detectors (Harris, SIFT/SURF/ORB)**  
4. [] **Descriptor Matching & Outlier Rejection (RANSAC)**  
5. [] **Object Detection with YOLOv8 (PyTorch Hub)**  
6. [] **Multi-Object Tracking (SORT/Deep SORT)**  
7. [] **Collision Time-to-Contact Estimation**  
8. [] **Lesson Summary + Quiz**

---

## 4 · Course 3 — **Radar: Clustering & Tracking**
Folder `/content/radar/`.

1. [] **Radar Principles & Range Equation**  
2. [] **FFT-based Range-Doppler Maps (NumPy FFT)**  
3. [] **CFAR Thresholding** — sliding-window implementation  
4. [] **Angle-of-Arrival Estimation (MUSIC/Beamforming)**  
5. [] **Data Association & Gating**  
6. [] **Tracking Multiple Targets (JPDA / GNN)**  
7. [] **Lesson Summary + Quiz**

---

## 5 · Course 4 — **Kalman Filters & Multi-Sensor Fusion**
Folder `/content/kalman_filters/`.

1. [] **Linear KF Mathematics (state, covariance, predict/update)**  
2. [] **Extended & Unscented KF Implementations (filterpy)**  
3. [] **Process & Measurement Noise Tuning**  
4. [] **Coordinate Frame Transforms (Homogeneous matrices)**  
5. [] **Track-to-Track Fusion Strategies**  
6. [] **IMM (Interacting Multiple Model) Tracking**  
7. [] **Sensor Mis-alignment Compensation**  
8. [] **Lesson Summary + Quiz**

---

## 6 · Capstone Project — **End-to-End Sensor Fusion Pipeline**
Folder `/content/capstone/`.

- [] **Project Brief & Rubric**  
  - Goal: fuse lidar + radar + camera to detect, classify, track objects in a recorded urban drive.  
- [] **Dataset Download Script** (Kitti or nuScenes sample)  
- [] **Baseline Notebook** with starter pipeline skeleton  
- [] **Milestones**  
  1. Calibration & synchronization  
  2. Single-sensor perception modules  
  3. Data association & fusion  
  4. Track management & visualization  
- [] **Evaluation Script** — precision, recall, RMSE  
- [] **Submission Checklist & Expected Deliverables**

---

## 7 · Interactive & UX Enhancements
- [] Add toggleable **solution blocks** (Sphinx-togglebutton) for quizzes & exercises.  
- [] Implement **progress persistence** (JS localStorage keys per lesson).  
- [] Provide **dark/light theme switch** retaining user preference.  
- [] Auto-generate **Table of Contents** per page.

---

## 8 · Testing & Continuous Integration
- [] Unit-test all Python utilities with `pytest`.  
- [] Lint code via `ruff` + `black`.  
- [] GitHub Actions workflow: build Jupyter Book, run tests, deploy to GitHub Pages on `main`.

---

## 9 · Deployment
- [] Produce static site in `/build`.  
- [] Deploy to GitHub Pages (or Netlify).  
- [] Verify mobile responsiveness & load times.

---

## 10 · Post-Launch
- [] Set up feedback form (Google Forms or Netlify Forms).  
- [] Schedule monthly content review checklist.  
- [] Collect analytics (privacy-friendly, e.g., Plausible) to gauge engagement.

---

### ✔ Completion Criteria  
The course is considered **done** when:  
1. All lessons render correctly with math, code, and quizzes.  
2. Every code block executes without error in a fresh Python 3.12 environment.  
3. CI pipeline passes, and site is live & publicly accessible.  
4. Capstone rubric & baseline run end-to-end on sample data.  

_Happy building!_