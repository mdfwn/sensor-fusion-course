# How to Use This Course

Welcome to your interactive sensor fusion learning journey! This course offers two complementary learning paths: **hands-on Jupyter notebooks** for interactive coding and **comprehensive documentation** for reference and theory.

## 🎮 Two Ways to Learn

### 🚀 Option 1: Interactive Notebooks (Recommended)
**Best for:** Hands-on learners who want to code along and experiment

**Location:** `notebooks/` directory  
**Format:** Jupyter `.ipynb` files with executable code cells

```bash
# Start your interactive learning journey
jupyter lab
# Navigate to: notebooks/lidar/01_introduction_to_lidar.ipynb
```

### 📚 Option 2: Documentation
**Best for:** Reading theory, quick reference, and building the full course website

**Location:** `content/` directory  
**Format:** Markdown files that build into a beautiful website

```bash
# Build the documentation website
jupyter-book build content/
# Open: _build/html/index.html
```

## 🎯 Quick Start Guide

### 1. Set Up Your Environment

```bash
# Install all required packages
pip install -r requirements.txt

# Verify Jupyter Lab installation
jupyter lab --version

# Generate any missing notebooks (if needed)
python3 generate_notebooks.py
```

### 2. Choose Your Learning Path

**For Interactive Learning:**
```bash
# Launch Jupyter Lab
jupyter lab

# You'll see:
# 📁 notebooks/
#   └── 📁 lidar/
#       └── 📓 01_introduction_to_lidar.ipynb  ← Start here!
```

**For Documentation Reading:**
- Read this documentation online or build locally
- Use as reference while working through notebooks
- Perfect for reviewing concepts

## 📓 Working with Interactive Notebooks

### Executing Code Cells

Throughout the course, you'll see executable code blocks:

```python
import numpy as np
import matplotlib.pyplot as plt

# This code runs in the notebook!
data = np.random.randn(100)
plt.hist(data, bins=20)
plt.title("Interactive Histogram")
plt.show()
```

### � Keyboard Shortcuts
- **Shift + Enter** — Run current cell and move to next
- **Ctrl + Enter** — Run current cell and stay in place
- **Alt + Enter** — Run current cell and insert new cell below
- **Tab** — Auto-complete code suggestions
- **Shift + Tab** — View function documentation

### Modifying and Experimenting

The notebooks are designed for experimentation! Try changing parameters:

```python
# 🚀 Experiment with these values!
num_points = 1000  # Change this number
noise_level = 0.1  # Try different values
sensor_height = 1.8  # Modify and see what happens

# Your modifications won't break anything - explore freely!
```

## 💾 Saving Your Work

### Automatic Saving
- **Notebooks:** Auto-save every 2 minutes in Jupyter Lab
- **Manual Save:** `Ctrl+S` or File → Save Notebook

### Best Practices
```bash
# Keep your own copy of modified notebooks
cp notebooks/lidar/01_introduction_to_lidar.ipynb my_notebooks/

# Or create a Git branch for your experiments
git checkout -b my-experiments
```

```{admonition} 💾 Backup Recommendation
:class: warning
While your progress is saved locally, consider backing up important work to your own Git repository or cloud storage.
```

## 🧩 Interactive Features

### Quizzes and Exercises

Notebooks include interactive elements:

**Knowledge Check Questions:**
```python
# Interactive quiz function (run the cell to start)
def quiz_question(question, options, correct_idx):
    # Built-in quiz system with immediate feedback
    pass
```

**Hands-On Exercises:**
```python
# 🚀 YOUR TURN: Complete this function
def analyze_point_cloud(points):
    """
    TODO: Implement point cloud analysis
    
    Your task:
    1. Calculate bounding box
    2. Find center point
    3. Compute statistics
    """
    # Write your code here!
    pass
```

### Interactive Visualizations

Experience rich 3D visualizations:

```python
# Interactive 3D plots with Plotly
import plotly.graph_objects as go

fig = go.Figure(data=go.Scatter3d(...))
fig.show()  # Rotate, zoom, hover for details!
```

**Features:**
- **🔄 Rotate:** Click and drag to change viewpoint
- **🔍 Zoom:** Scroll wheel or pinch to zoom
- **📍 Hover:** Mouse over points for detailed information
- **📷 Export:** Save plots as images or HTML files

## 📊 Progress Tracking

### Notebook Progress
Your progress is automatically tracked:
- ✅ Completed cells are marked with output
- 🏃‍♂️ Current position is highlighted
- 💾 All changes are auto-saved

### Course Progress
- Complete notebooks in sequence for best learning
- Each notebook builds on previous concepts
- Exercise solutions build practical skills

## 🎯 Course Structure

### 🔍 Course 1: Lidar (Available Now)
**Interactive Notebooks:**
- `01_introduction_to_lidar.ipynb` — Sensor principles ✅
- `02_parsing_pcd_ply_files.ipynb` — File I/O *(Coming soon)*
- `03_ransac_plane_fitting.ipynb` — Ground detection *(Coming soon)*
- `04_euclidean_clustering.ipynb` — Object clustering *(Coming soon)*
- `05_bounding_boxes.ipynb` — Object localization *(Coming soon)*
- `06_lidar_visualizer.ipynb` — 3D visualization *(Coming soon)*
- `07_lidar_summary.ipynb` — Complete pipeline *(Coming soon)*

### 📷 Course 2-4: Camera, Radar, Kalman Filters
*(Coming in future updates)*

## 🔧 Code Environment

### Pre-installed Libraries

All notebooks come with these libraries ready to use:

```python
# Core scientific computing
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

# Computer vision and 3D processing
import cv2
import open3d as o3d
from skimage import feature, measure

# Sensor fusion specific
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise

# Interactive visualization
import plotly.express as px
import plotly.graph_objects as go
```

### Local Development (Optional)

To run on your own machine:

```bash
# 1. Install Python 3.8+
# 2. Clone/download the course
# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch Jupyter Lab
jupyter lab

# 5. Open notebooks/lidar/01_introduction_to_lidar.ipynb
```

## � Troubleshooting

### Common Issues

**"Kernel not found" error:**
```bash
# Reinstall Jupyter kernels
python -m ipykernel install --user --name=python3
```

**Plots not showing:**
```bash
# Install visualization extensions
jupyter labextension install @jupyter-widgets/jupyterlab-manager
jupyter labextension install plotlywidget
```

**Module import errors:**
```bash
# Verify all packages are installed
pip install -r requirements.txt

# Check your Python environment
which python
python --version
```

### Performance Tips

**For large datasets:**
- Work with sample data first
- Use numpy vectorization 
- Close unused notebooks to free memory

**For smooth 3D visualizations:**
- Reduce point cloud density for exploration
- Use simplified visualization modes
- Consider your system's graphics capabilities

## 📱 Mobile Learning

### Responsive Experience
- Notebooks work on tablets and phones
- Touch-friendly interface elements
- Optimized for smaller screens

### Limitations
- Complex 3D visualizations work best on desktop
- Code editing easier with physical keyboard
- Large datasets require more powerful devices

## 🎓 Learning Tips

### For Maximum Success

```{admonition} 🚀 Pro Learning Tips
:class: tip
1. **Code Along** — Don't just read, run every code cell
2. **Experiment** — Modify parameters and see what happens
3. **Take Notes** — Use markdown cells to add your insights
4. **Practice** — Complete all exercises and challenges
5. **Connect** — Join the community for discussions and help
```

### Study Schedule Suggestion
- **Week 1-2:** Lidar fundamentals (notebooks 1-3)
- **Week 3-4:** Advanced lidar processing (notebooks 4-7)
- **Week 5+:** Move to camera and radar courses

### Getting Unstuck
1. **Re-read** the cell above your current position
2. **Check docs** — Use `help()` function or `shift+tab`
3. **Search online** — Most concepts have excellent tutorials
4. **Ask for help** — Use the community forum
5. **Take a break** — Sometimes stepping away helps!

---

## 🚀 Ready to Begin?

**Start your interactive learning journey:**

```bash
jupyter lab
# Open: notebooks/lidar/01_introduction_to_lidar.ipynb
```

**Or browse the documentation:**
- Continue reading: [Resources & References](resources.md)
- Jump to content: [Course 1: Lidar](lidar/index.md)

*Let's build the future of autonomous vehicles together!* 🚗💨