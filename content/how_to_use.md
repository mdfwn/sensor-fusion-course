# How to Use This Course

Welcome to your interactive sensor fusion learning journey! This guide will help you make the most of the course's interactive features and learning tools.

## 🎮 Interactive Features Overview

This course is designed to be **hands-on and interactive**. Unlike traditional textbooks or videos, you'll be actively coding, experimenting, and building throughout your learning experience.

### ✨ Key Features
- 📓 **Live Jupyter Notebooks** — Execute Python code directly in your browser
- 📊 **Progress Tracking** — See your learning journey visualized in real-time
- 🧩 **Interactive Quizzes** — Test your understanding with immediate feedback
- 🎯 **Guided Exercises** — Step-by-step coding challenges with hints
- 📱 **Mobile-Friendly** — Learn on any device, anywhere

## 📓 Working with Interactive Notebooks

### Executing Code Cells

Throughout the course, you'll see code blocks that you can run and modify:

```python
import numpy as np
import matplotlib.pyplot as plt

# This is an interactive code cell!
# Click the "Run" button or press Shift+Enter to execute
data = np.random.randn(100)
plt.hist(data, bins=20)
plt.title("Interactive Histogram")
plt.show()
```

```{admonition} 💡 Pro Tips for Code Cells
:class: tip
- **Shift + Enter** — Run current cell and move to next
- **Ctrl + Enter** — Run current cell and stay in place
- **Alt + Enter** — Run current cell and insert new cell below
- **Tab** — Auto-complete code suggestions
- **Shift + Tab** — View function documentation
```

### Modifying and Experimenting

Feel free to modify any code! The best way to learn is by experimenting:

```python
# Try changing these parameters and see what happens!
num_points = 1000  # Change this number
noise_level = 0.1  # Try different values

# Your modifications won't break anything - experiment freely!
```

### Saving Your Work

Your code modifications and progress are automatically saved in your browser's local storage. However, for important work:

```{admonition} 💾 Backup Recommendation
:class: warning
While your progress is saved locally, consider copying important code to your own Git repository or notebooks for backup purposes.
```

## 📊 Progress Tracking System

### Visual Progress Tracker

Look for the **progress tracker** on the right side of your screen (desktop) or accessible via the hamburger menu (mobile):

- 📚 **Course Progress** — See completion percentage for each course
- ✅ **Completed Lessons** — Green checkmarks for finished content
- 📖 **Current Lesson** — Highlighted in blue
- 🏆 **Quiz Scores** — Track your quiz performance

### How Progress is Tracked

1. **Page Visits** — Automatically tracked when you visit a lesson
2. **Quiz Completion** — Recorded when you answer correctly
3. **Exercise Completion** — Manual check-off for coding exercises
4. **Local Storage** — All progress saved in your browser

### Progress Controls

Use these keyboard shortcuts for quick access:
- **Ctrl/Cmd + /** — Toggle progress tracker visibility
- **Ctrl/Cmd + Shift + R** — View detailed progress (developer mode)

## 🧩 Interactive Quiz System

### Taking Quizzes

Quizzes appear throughout lessons to reinforce key concepts:

<div class="quiz-container" data-quiz-id="sample-quiz" data-correct="b">
  <div class="quiz-question">
    🤔 <strong>Sample Question:</strong> What is the primary advantage of sensor fusion in autonomous vehicles?
  </div>
  <ul class="quiz-options">
    <li data-value="a">Reduced computational requirements</li>
    <li data-value="b">Improved robustness and accuracy</li>
    <li data-value="c">Lower hardware costs</li>
    <li data-value="d">Simplified algorithms</li>
  </ul>
</div>

### Quiz Features

- **Immediate Feedback** — See results instantly upon selection
- **Multiple Attempts** — Learn from mistakes without penalty
- **Progress Tracking** — Completed quizzes marked in your progress
- **Explanations** — Detailed feedback for both correct and incorrect answers

## 🎯 Guided Exercises

### Exercise Structure

Each lesson includes hands-on exercises structured as:

<div class="exercise-notebook">
  <div class="exercise-title">🛠️ Exercise: RANSAC Implementation</div>
  <p><strong>Goal:</strong> Implement the RANSAC algorithm for plane fitting in 3D point clouds.</p>
  
  <p><strong>Tasks:</strong></p>
  <ol>
    <li>Load sample point cloud data</li>
    <li>Implement random sample selection</li>
    <li>Calculate model parameters</li>
    <li>Count inliers and evaluate fit</li>
    <li>Visualize results</li>
  </ol>
  
  <p><strong>Success Criteria:</strong> Your implementation should correctly identify the ground plane with >95% accuracy.</p>
</div>

### Exercise Tips

```{admonition} 🚀 Exercise Success Tips
:class: tip
1. **Read Carefully** — Understand the goal before coding
2. **Start Simple** — Get basic functionality working first
3. **Test Incrementally** — Verify each step before moving on
4. **Use Print Statements** — Debug by examining intermediate results
5. **Ask for Help** — Use the community forum for challenging problems
```

### Marking Exercises Complete

When you finish an exercise, mark it complete to track progress:

```javascript
// Mark exercise as complete (built into each exercise page)
sensorFusionUtils.markExerciseComplete('ransac-implementation');
```

## 🔧 Code Environment Setup

### Required Libraries

All necessary libraries are pre-installed in the interactive environment:

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
from filterpy.kalman import KalmanFilter, ExtendedKalmanFilter
from filterpy.common import Q_discrete_white_noise

# Interactive visualization
import plotly.express as px
import plotly.graph_objects as go
```

### Local Development (Optional)

To run exercises locally on your machine:

```bash
# Clone the course repository
git clone https://github.com/yourusername/sensor-fusion-course.git
cd sensor-fusion-course

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter lab
```

## 📱 Mobile Learning

### Responsive Design

The course is optimized for mobile devices:

- **Touch-Friendly** — Large buttons and touch targets
- **Readable Text** — Optimized font sizes and spacing
- **Collapsible Sections** — Hide/show content as needed
- **Offline Capable** — Core content works without internet

### Mobile-Specific Features

- **Swipe Navigation** — Swipe between lessons on touch devices
- **Zoom Support** — Pinch to zoom on diagrams and equations
- **Portrait Mode** — Optimized layout for phone screens

## 🎨 Customization Options

### Theme Selection

Choose your preferred visual theme:

- **🌞 Light Mode** — Clean, bright interface for day learning
- **🌙 Dark Mode** — Easy on eyes for evening study sessions
- **⚡ Auto** — Switches based on system preferences

Access theme controls via the toolbar or use system settings.

### Accessibility Features

- **High Contrast** — Enhanced visibility for all users
- **Large Text** — Adjustable font sizes
- **Keyboard Navigation** — Full course navigation without mouse
- **Screen Reader** — Compatible with assistive technologies

## 📞 Getting Help

### Community Support

- **💬 Discussion Forum** — Ask questions and help peers
- **🤝 Study Groups** — Form groups with other learners
- **👨‍🏫 Office Hours** — Live Q&A sessions with instructors

### Technical Issues

If you encounter technical problems:

1. **Refresh** — Try reloading the page
2. **Clear Cache** — Clear browser cache and cookies
3. **Different Browser** — Test in Chrome, Firefox, or Safari
4. **Report Bug** — Use the feedback form for persistent issues

### Progress Reset

If you need to reset your progress:

```javascript
// Reset all progress (cannot be undone!)
sensorFusionUtils.resetProgress();
```

```{admonition} ⚠️ Progress Reset Warning
:class: warning
Resetting progress will permanently delete all tracked completion data. This action cannot be undone.
```

---

## 🚀 Ready to Begin?

Now that you understand the interactive features, you're ready to dive into the content!

**Next Steps:**
1. Check out the [Resources](resources.md) for helpful reference materials
2. Start with [Course 1: Lidar](lidar/index.md) to begin your sensor fusion journey
3. Join the community forum to connect with other learners

*Happy learning! 🎓*