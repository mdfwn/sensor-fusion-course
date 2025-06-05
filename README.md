# 🚗 Interactive Python-First Sensor Fusion Course

Welcome to the comprehensive, hands-on sensor fusion course designed for autonomous vehicle engineers and robotics enthusiasts!

## 🌟 Course Overview

This course transforms traditional sensor fusion education with:
- **Interactive Jupyter Notebooks** — Execute Python code directly in your browser
- **Real-World Applications** — Work with actual autonomous vehicle sensor data
- **Progressive Learning** — Build skills incrementally across 4 specialized courses
- **Modern Python Stack** — Use NumPy, OpenCV, Open3D, and other industry-standard libraries

## 📁 Course Structure

```
sensor-fusion-course/
├── 📚 content/           # Course documentation (Jupyter Book)
│   ├── index.md         # Course overview
│   ├── how_to_use.md    # Usage instructions
│   ├── resources.md     # Reference materials
│   └── lidar/           # Lidar course content
├── 📓 notebooks/        # Interactive Jupyter notebooks
│   └── lidar/           # Hands-on lidar exercises
├── 📦 requirements.txt  # Python dependencies
└── 🔧 generate_notebooks.py  # Notebook generator script
```

## 🚀 Quick Start

### 1. Set Up Your Environment

```bash
# Clone or download the course
git clone <your-repo-url>
cd sensor-fusion-course

# Install Python dependencies
pip install -r requirements.txt

# Generate any missing notebooks
python3 generate_notebooks.py
```

### 2. Start Learning

**Option A: Interactive Notebooks (Recommended)**
```bash
# Launch Jupyter Lab
jupyter lab

# Navigate to: notebooks/lidar/01_introduction_to_lidar.ipynb
# Start with the first lesson!
```

**Option B: Documentation**
```bash
# Build the documentation (optional)
jupyter-book build content/

# Open _build/html/index.html in your browser
```

## 📚 Course Content

### 🔍 Course 1: Lidar Obstacle Detection
**Location:** `notebooks/lidar/`

- 📓 `01_introduction_to_lidar.ipynb` — Sensor principles, time-of-flight calculations
- 📓 `02_parsing_pcd_ply_files.ipynb` — File I/O with Open3D
- 📓 `03_ransac_plane_fitting.ipynb` — Ground plane detection
- 📓 `04_euclidean_clustering.ipynb` — Object segmentation with KD-trees
- 📓 `05_bounding_boxes.ipynb` — Object localization
- 📓 `06_lidar_visualizer.ipynb` — Interactive 3D visualization
- 📓 `07_lidar_summary.ipynb` — Complete pipeline integration

### 📷 Course 2: Camera Tracking & Detection *(Coming Soon)*
**Location:** `notebooks/camera/`

- Camera calibration and distortion correction
- Feature detection (SIFT, SURF, ORB, Harris)
- Object detection with YOLOv8
- Multi-object tracking algorithms
- Time-to-collision estimation

### 📡 Course 3: Radar Processing *(Coming Soon)*
**Location:** `notebooks/radar/`

- FFT-based range-doppler maps
- CFAR (Constant False Alarm Rate) detection
- Angle-of-arrival estimation
- Multi-target tracking

### 🧮 Course 4: Kalman Filters & Fusion *(Coming Soon)*
**Location:** `notebooks/kalman_filters/`

- Linear, Extended, and Unscented Kalman Filters
- Multi-sensor data fusion
- Track-to-track association
- IMM (Interacting Multiple Model) filters

### 🎓 Capstone Project *(Coming Soon)*
**Location:** `notebooks/capstone/`

- End-to-end sensor fusion pipeline
- Real-world dataset processing
- Performance evaluation and metrics

## 🎯 Learning Path

### For Beginners
1. **Start here:** `notebooks/lidar/01_introduction_to_lidar.ipynb`
2. **Follow the sequence** — each notebook builds on previous concepts
3. **Complete exercises** — hands-on practice is essential
4. **Ask questions** — use the community forum for help

### For Experienced Practitioners
1. **Review the overview:** `content/index.md`
2. **Jump to specific topics** — each notebook is self-contained
3. **Focus on implementation** — modify and extend the provided code
4. **Contribute improvements** — help make the course better!

## 🛠️ Prerequisites

### Programming
- **Python 3.8+** — Functions, classes, NumPy basics
- **Jupyter Notebooks** — Basic familiarity helpful
- **Git** — For version control and collaboration

### Mathematics  
- **Linear Algebra** — Vectors, matrices, transformations
- **Probability** — Distributions, Bayes' theorem
- **Basic Calculus** — Derivatives, optimization

### Hardware Understanding
- **Basic Physics** — Wave properties, kinematics
- **Sensor Concepts** — Understanding of measurement principles

## 📦 Dependencies

All required packages are listed in `requirements.txt`:

```txt
# Core scientific computing
numpy>=1.24.0
scipy>=1.10.0
matplotlib>=3.6.0

# Computer vision and 3D processing
open3d>=0.18.0
opencv-python>=4.8.0
scikit-image>=0.20.0
plotly>=5.17.0

# Kalman filtering
filterpy>=1.4.5

# Interactive notebooks
jupyter>=1.0.0
jupyterlab>=4.0.0
ipywidgets>=8.0.0

# Documentation (optional)
jupyter-book>=0.15.0
myst-parser>=2.0.0
```

## 💻 System Requirements

### Minimum
- **OS:** Windows 10, macOS 10.14, or Linux
- **RAM:** 8GB (16GB recommended)
- **Storage:** 2GB free space
- **Python:** 3.8 or newer

### Recommended
- **GPU:** NVIDIA GPU with CUDA support (for deep learning sections)
- **Display:** 1920x1080 or higher (for 3D visualizations)
- **Internet:** For downloading datasets and documentation

## 🎮 Interactive Features

### Jupyter Notebooks
- **Live Code Execution** — Run and modify all examples
- **Interactive Visualizations** — 3D plots with Plotly
- **Progressive Exercises** — Build skills step-by-step
- **Immediate Feedback** — See results instantly

### Visualization Tools
- **3D Point Cloud Viewer** — Rotate, zoom, pan through data
- **Interactive Plots** — Hover for details, zoom regions
- **Real-time Updates** — See changes as you modify code
- **Export Capabilities** — Save plots and results

## 🚦 Getting Help

### Common Issues

**"ModuleNotFoundError" when running notebooks:**
```bash
# Make sure all dependencies are installed
pip install -r requirements.txt
```

**Jupyter Lab won't start:**
```bash
# Try updating Jupyter
pip install --upgrade jupyterlab

# Or use classic Jupyter
jupyter notebook
```

**3D visualizations not working:**
```bash
# Install Plotly extensions
pip install "plotly>=5.0" "ipywidgets>=8.0"
jupyter lab clean
jupyter lab build
```

### Support Channels
- **GitHub Issues** — Bug reports and feature requests
- **Discussion Forum** — Community Q&A
- **Documentation** — Comprehensive guides and references

## 🎯 Success Metrics

By the end of this course, you'll be able to:
- ✅ Process lidar point clouds for obstacle detection
- ✅ Implement camera-based object tracking
- ✅ Build radar signal processing pipelines  
- ✅ Design multi-sensor Kalman filter systems
- ✅ Create complete autonomous vehicle perception stacks
- ✅ Evaluate system performance with industry metrics

## 🤝 Contributing

We welcome contributions! See our contribution guidelines for:
- Adding new lessons or improving existing ones
- Fixing bugs or typos
- Creating additional exercises
- Improving documentation

## 📄 License

This course is open source under the MIT License. Feel free to use, modify, and share!

---

**Ready to start your sensor fusion journey?** 🚀

```bash
jupyter lab
# Open: notebooks/lidar/01_introduction_to_lidar.ipynb
```

*Happy learning!* 🎓✨