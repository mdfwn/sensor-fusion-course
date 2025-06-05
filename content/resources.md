# Resources & References

Your comprehensive guide to documentation, tutorials, and additional learning materials for the sensor fusion course.

## 📚 Core Libraries Documentation

### 🔢 Scientific Computing

#### NumPy - Numerical Computing
- **Official Documentation:** [numpy.org](https://numpy.org/doc/stable/)
- **Quickstart Tutorial:** [NumPy Quickstart](https://numpy.org/doc/stable/user/quickstart.html)
- **Array Programming:** [Array API Standard](https://numpy.org/doc/stable/reference/arrays.html)
- **Performance Tips:** [NumPy Performance](https://numpy.org/doc/stable/user/c-info.ufunc-tutorial.html)

```python
# Essential NumPy for sensor fusion
import numpy as np
# Linear algebra, matrix operations, array processing
# Used in: All courses for data manipulation and calculations
```

#### SciPy - Scientific Library
- **Official Documentation:** [scipy.org](https://scipy.org/doc/scipy/index.html)
- **Statistical Functions:** [scipy.stats](https://docs.scipy.org/doc/scipy/reference/stats.html)
- **Signal Processing:** [scipy.signal](https://docs.scipy.org/doc/scipy/reference/signal.html)
- **Optimization:** [scipy.optimize](https://docs.scipy.org/doc/scipy/reference/optimize.html)

```python
# Key SciPy modules for sensor fusion
from scipy import signal, stats, optimize
# Used in: Kalman filters, signal processing, parameter estimation
```

### 👁️ Computer Vision

#### OpenCV - Computer Vision Library
- **Official Documentation:** [docs.opencv.org](https://docs.opencv.org/4.x/)
- **Python Tutorials:** [OpenCV-Python Tutorials](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html)
- **Camera Calibration:** [Camera Calibration Guide](https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html)
- **Feature Detection:** [Feature Detection Tutorial](https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html)

```python
# OpenCV for image processing
import cv2
# Used in: Course 2 (Camera) for feature detection, calibration, object detection
```

#### scikit-image - Image Processing
- **Official Documentation:** [scikit-image.org](https://scikit-image.org/docs/stable/)
- **User Guide:** [Usage Examples](https://scikit-image.org/docs/stable/user_guide.html)
- **API Reference:** [Module Reference](https://scikit-image.org/docs/stable/api/api.html)

```python
# scikit-image for advanced image processing
from skimage import feature, measure, segmentation
# Used in: Image filtering, edge detection, region analysis
```

### 🎯 3D Processing

#### Open3D - 3D Data Processing
- **Official Documentation:** [open3d.org/docs](http://www.open3d.org/docs/)
- **Getting Started:** [Open3D Tutorial](http://www.open3d.org/docs/release/getting_started.html)
- **Point Cloud Processing:** [Point Cloud Tutorial](http://www.open3d.org/docs/release/tutorial/geometry/pointcloud.html)
- **Visualization:** [Visualization Tutorial](http://www.open3d.org/docs/release/tutorial/visualization/visualization.html)

```python
# Open3D for 3D point cloud processing
import open3d as o3d
# Used in: Course 1 (Lidar) for point cloud processing, visualization, clustering
```

### 🎛️ Sensor Fusion

#### FilterPy - Kalman Filtering
- **Official Documentation:** [filterpy.readthedocs.io](https://filterpy.readthedocs.io/en/latest/)
- **Kalman Filter Book:** [Kalman-and-Bayesian-Filters-in-Python](https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python)
- **Examples:** [FilterPy Examples](https://filterpy.readthedocs.io/en/latest/kalman/KalmanFilter.html)

```python
# FilterPy for advanced filtering
from filterpy.kalman import KalmanFilter, ExtendedKalmanFilter, UnscentedKalmanFilter
# Used in: Course 4 (Kalman Filters) for all filtering implementations
```

### 📊 Visualization

#### Matplotlib - Static Plotting
- **Official Documentation:** [matplotlib.org](https://matplotlib.org/stable/index.html)
- **Tutorials:** [Matplotlib Tutorials](https://matplotlib.org/stable/tutorials/index.html)
- **Gallery:** [Example Gallery](https://matplotlib.org/stable/gallery/index.html)

```python
# Matplotlib for 2D plotting
import matplotlib.pyplot as plt
# Used in: All courses for data visualization and analysis
```

#### Plotly - Interactive Visualization
- **Official Documentation:** [plotly.com/python](https://plotly.com/python/)
- **3D Plotting:** [3D Plots in Python](https://plotly.com/python/3d-charts/)
- **Interactive Widgets:** [Plotly Widgets](https://plotly.com/python/figurewidget/)

```python
# Plotly for interactive 3D visualization
import plotly.express as px
import plotly.graph_objects as go
# Used in: Interactive visualizations, 3D point cloud display
```

---

## 📖 Mathematical Foundations

### Linear Algebra

#### Khan Academy - Linear Algebra
- **Course:** [Linear Algebra Course](https://www.khanacademy.org/math/linear-algebra)
- **Topics:** Vectors, matrices, transformations, eigenvalues
- **Level:** Beginner to intermediate

#### 3Blue1Brown - Essence of Linear Algebra
- **Video Series:** [Essence of Linear Algebra](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab)
- **Visual Approach:** Intuitive understanding of linear algebra concepts
- **Level:** Beginner-friendly with deep insights

### Probability & Statistics

#### Think Stats - Probability and Statistics for Programmers
- **Book (Free):** [Think Stats](https://greenteapress.com/thinkstats2/html/index.html)
- **Python-Based:** Practical approach using Python
- **Level:** Intermediate

#### MIT OpenCourseWare - Probability
- **Course:** [Introduction to Probability](https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-041-probabilistic-systems-analysis-and-applied-probability-fall-2010/)
- **Comprehensive:** Complete university-level course
- **Level:** Advanced

### Calculus

#### Khan Academy - Calculus
- **Course:** [AP Calculus AB/BC](https://www.khanacademy.org/math/calculus-1)
- **Topics:** Derivatives, integrals, optimization
- **Level:** Beginner to intermediate

---

## 🚗 Autonomous Vehicle Context

### Industry Standards

#### SAE International - Autonomous Vehicle Levels
- **Document:** [SAE J3016 Levels of Automation](https://www.sae.org/standards/content/j3016_202104/)
- **Context:** Understanding autonomy levels referenced in the course

#### ISO 26262 - Functional Safety
- **Standard:** [Functional Safety for Road Vehicles](https://www.iso.org/standard/68383.html)
- **Relevance:** Safety requirements for automotive systems

### Datasets

#### KITTI Dataset
- **Website:** [KITTI Vision Benchmark](http://www.cvlibs.net/datasets/kitti/)
- **Content:** Real-world autonomous driving data
- **Usage:** Stereo vision, optical flow, visual odometry, 3D object detection

#### nuScenes Dataset
- **Website:** [nuScenes](https://www.nuscenes.org/)
- **Content:** Full autonomous vehicle sensor suite dataset
- **Usage:** 3D object detection, tracking, prediction

#### Waymo Open Dataset
- **Website:** [Waymo Open Dataset](https://waymo.com/open/)
- **Content:** High-quality sensor data from Waymo vehicles
- **Usage:** 3D detection, tracking, motion prediction

---

## 🛠️ Development Tools

### Python Environment

#### Anaconda - Python Distribution
- **Website:** [anaconda.com](https://www.anaconda.com/)
- **Benefits:** Pre-configured scientific computing environment
- **Installation:** Complete setup for sensor fusion development

#### Poetry - Dependency Management
- **Website:** [python-poetry.org](https://python-poetry.org/)
- **Benefits:** Modern dependency management for Python projects
- **Usage:** Managing project dependencies and virtual environments

### Version Control

#### Git - Version Control System
- **Tutorial:** [Git Tutorial](https://git-scm.com/docs/gittutorial)
- **Interactive Learning:** [Learn Git Branching](https://learngitbranching.js.org/)
- **Cheat Sheet:** [Git Cheat Sheet](https://education.github.com/git-cheat-sheet-education.pdf)

#### GitHub - Code Hosting
- **Guides:** [GitHub Guides](https://guides.github.com/)
- **Usage:** Hosting course projects and collaboration

### IDEs and Editors

#### Jupyter Lab - Interactive Development
- **Documentation:** [JupyterLab Documentation](https://jupyterlab.readthedocs.io/)
- **Extensions:** [JupyterLab Extensions](https://jupyterlab.readthedocs.io/en/stable/user/extensions.html)

#### VS Code - Code Editor
- **Website:** [Visual Studio Code](https://code.visualstudio.com/)
- **Python Extension:** [Python in VS Code](https://code.visualstudio.com/docs/languages/python)
- **Jupyter Support:** [Jupyter Notebooks in VS Code](https://code.visualstudio.com/docs/datascience/jupyter-notebooks)

---

## 📝 Academic Papers & Books

### Foundational Papers

#### Kalman Filter
- **Original Paper:** "A New Approach to Linear Filtering and Prediction Problems" (1960)
- **Author:** Rudolf E. Kalman
- **Significance:** Foundation of modern estimation theory

#### RANSAC Algorithm
- **Paper:** "Random Sample Consensus: A Paradigm for Model Fitting"
- **Authors:** Fischler & Bolles (1981)
- **Application:** Robust parameter estimation in computer vision

### Books

#### "Probabilistic Robotics"
- **Authors:** Sebastian Thrun, Wolfram Burgard, Dieter Fox
- **Publisher:** MIT Press
- **Content:** Comprehensive coverage of probabilistic approaches to robotics
- **Relevance:** Sensor fusion, localization, mapping

#### "Multiple View Geometry in Computer Vision"
- **Authors:** Richard Hartley, Andrew Zisserman
- **Publisher:** Cambridge University Press
- **Content:** Mathematical foundations of computer vision
- **Relevance:** Camera geometry, feature matching, stereo vision

#### "Introduction to Autonomous Mobile Robots"
- **Authors:** Roland Siegwart, Illah Nourbakhsh, Davide Scaramuzza
- **Publisher:** MIT Press
- **Content:** Comprehensive robotics textbook
- **Relevance:** Sensor integration, navigation, perception

---

## 🎓 Online Courses

### Complementary Courses

#### Coursera - Robotics Specialization
- **University:** University of Pennsylvania
- **Content:** Aerial robotics, computational motion planning, estimation
- **Link:** [Robotics Specialization](https://www.coursera.org/specializations/robotics)

#### edX - Autonomous Navigation for Flying Robots
- **University:** Technical University of Munich
- **Content:** Computer vision, state estimation, control
- **Link:** [Autonomous Navigation](https://www.edx.org/course/autonomous-navigation-for-flying-robots)

#### Udacity - Self-Driving Car Engineer Nanodegree
- **Content:** Computer vision, deep learning, sensor fusion, controls
- **Link:** [Self-Driving Car Nanodegree](https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013)

---

## 🔧 Software Tools

### Simulation Platforms

#### CARLA - Open Source Simulator
- **Website:** [carla.org](https://carla.org/)
- **Content:** Realistic autonomous driving simulation
- **Usage:** Testing sensor fusion algorithms in virtual environments

#### AirSim - Drone and Car Simulator
- **Website:** [AirSim](https://microsoft.github.io/AirSim/)
- **Content:** Physics-based simulation for drones and cars
- **Usage:** Sensor simulation, algorithm development

### Robotics Frameworks

#### ROS (Robot Operating System)
- **Website:** [ros.org](https://www.ros.org/)
- **Content:** Middleware for robotics applications
- **Usage:** Sensor integration, algorithm deployment

---

## 💡 Quick Reference Guides

### Cheat Sheets

- [NumPy Cheat Sheet](https://s3.amazonaws.com/assets.datacamp.com/blog_assets/Numpy_Python_Cheat_Sheet.pdf)
- [Matplotlib Cheat Sheet](https://s3.amazonaws.com/assets.datacamp.com/blog_assets/Python_Matplotlib_Cheat_Sheet.pdf)
- [OpenCV Cheat Sheet](https://docs.opencv.org/master/d2/d96/tutorial_py_table_of_contents_imgproc.html)
- [Git Cheat Sheet](https://education.github.com/git-cheat-sheet-education.pdf)

### Formula References

- [Linear Algebra Quick Reference](https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf)
- [Probability Distribution Reference](https://en.wikipedia.org/wiki/List_of_probability_distributions)
- [Kalman Filter Equations](https://www.bzarg.com/p/how-a-kalman-filter-works-in-pictures/)

---

## 🚀 Stay Updated

### Communities & Forums

- **Reddit:** [r/ComputerVision](https://www.reddit.com/r/computervision/)
- **Reddit:** [r/MachineLearning](https://www.reddit.com/r/MachineLearning/)
- **Stack Overflow:** [Computer Vision Tag](https://stackoverflow.com/questions/tagged/computer-vision)
- **GitHub:** [Awesome Computer Vision](https://github.com/jbhuang0604/awesome-computer-vision)

### Conferences & Journals

- **CVPR** - Computer Vision and Pattern Recognition
- **ICCV** - International Conference on Computer Vision
- **ICRA** - International Conference on Robotics and Automation
- **IROS** - International Conference on Intelligent Robots and Systems

---

*These resources will support your learning throughout the course and beyond. Bookmark this page for quick reference!* 🔖