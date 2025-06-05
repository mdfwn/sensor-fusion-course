#!/usr/bin/env python3
"""
Generate Interactive Jupyter Notebooks for Sensor Fusion Course

This script creates .ipynb files with executable code cells for hands-on learning.
"""

import json
import os
from pathlib import Path

def create_notebook_cell(cell_type, content, metadata=None):
    """Create a notebook cell dictionary"""
    cell = {
        "cell_type": cell_type,
        "metadata": metadata or {},
        "source": content.split('\n') if isinstance(content, str) else content
    }
    
    if cell_type == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    
    return cell

def create_lidar_intro_notebook():
    """Create the Introduction to Lidar & Point Clouds notebook"""
    
    cells = []
    
    # Title and intro
    cells.append(create_notebook_cell("markdown", """# 🌟 Introduction to Lidar & Point Clouds

Welcome to your first interactive lesson in sensor fusion! In this notebook, you'll learn about lidar technology and work with real 3D point cloud data.

## 🎯 Learning Objectives
By the end of this notebook, you will:
- Understand how lidar sensors work
- Manipulate 3D point cloud data with Python
- Visualize lidar data in 3D
- Perform coordinate transformations
- Analyze real-world data characteristics"""))

    # What is Lidar section
    cells.append(create_notebook_cell("markdown", """## 📚 What is Lidar?

Imagine you're in a dark room and need to understand the layout of furniture around you. You could throw tennis balls in all directions and listen for when they bounce back — measuring both the direction you threw them and how long they took to return. This is essentially how **Lidar** (Light Detection and Ranging) works, except instead of tennis balls, it uses pulses of laser light!

### The Physics Behind Lidar

Lidar sensors use the **time-of-flight** principle:

$$\\text{Distance} = \\frac{c \\times \\Delta t}{2}$$

Where:
- $c$ = speed of light ($3 \\times 10^8$ m/s)
- $\\Delta t$ = round-trip time for the laser pulse
- Division by 2 accounts for the round trip (out and back)"""))

    # Import libraries
    cells.append(create_notebook_cell("code", """# Let's start by importing the libraries we'll need
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import plotly.express as px

# For 3D point cloud processing
try:
    import open3d as o3d
    print("✅ Open3D imported successfully")
except ImportError:
    print("❌ Open3D not found. Install with: pip install open3d")

print("🚀 Libraries loaded! Ready to explore lidar data.")"""))

    # Time-of-flight calculations
    cells.append(create_notebook_cell("markdown", """## 🧮 Time-of-Flight Calculations

Let's calculate some example distances using the time-of-flight principle:"""))

    cells.append(create_notebook_cell("code", """# Speed of light in m/s
SPEED_OF_LIGHT = 3e8

def calculate_distance(time_of_flight):
    \"\"\"
    Calculate distance based on time-of-flight
    
    Args:
        time_of_flight: Round-trip time in seconds
    
    Returns:
        Distance in meters
    \"\"\"
    return (SPEED_OF_LIGHT * time_of_flight) / 2

# Example calculations
examples = [
    (1e-7, "Close object (wall)"),
    (2e-7, "Medium distance (car)"),
    (6.67e-7, "Far object (building)")
]

print("🎯 Lidar Distance Calculations:")
print("=" * 40)

for time_ns, description in examples:
    distance = calculate_distance(time_ns)
    print(f"{description:20} | Time: {time_ns*1e9:.1f} ns | Distance: {distance:.1f} m")

print("\\n💡 Try changing the time values above to see how distance changes!")"""))

    # Interactive exercise
    cells.append(create_notebook_cell("markdown", """## 🎮 Interactive Exercise: Calculate Your Own Distances

Now it's your turn! Modify the code below to calculate distances for different scenarios:"""))

    cells.append(create_notebook_cell("code", """# 🚀 YOUR TURN: Calculate distances for these scenarios

# Scenario 1: A pedestrian 10 meters away - what should the time-of-flight be?
target_distance_1 = 10.0  # meters
expected_time_1 = (2 * target_distance_1) / SPEED_OF_LIGHT

print(f"📍 Pedestrian at {target_distance_1}m:")
print(f"   Expected time-of-flight: {expected_time_1*1e9:.2f} nanoseconds")

# Scenario 2: Your turn! Pick a distance and calculate the time
# TODO: Change this value and see what happens
your_distance = 25.0  # Change this!
your_time = (2 * your_distance) / SPEED_OF_LIGHT

print(f"\\n🎯 Your scenario at {your_distance}m:")
print(f"   Time-of-flight: {your_time*1e9:.2f} nanoseconds")

# Scenario 3: Given a time, calculate the distance
measured_time = 3.33e-7  # seconds
calculated_distance = calculate_distance(measured_time)

print(f"\\n📏 Measured time {measured_time*1e9:.1f} ns:")
print(f"   Calculated distance: {calculated_distance:.1f} meters")"""))

    # Point cloud creation
    cells.append(create_notebook_cell("markdown", """## 📊 Creating Your First Point Cloud

Let's create a synthetic point cloud to understand the data structure:"""))

    cells.append(create_notebook_cell("code", """def create_sample_point_cloud():
    \"\"\"
    Generate a synthetic point cloud representing a simple scene
    \"\"\"
    np.random.seed(42)  # For reproducible results
    
    # Ground plane (z ≈ 0)
    n_ground = 1000
    ground_x = np.random.uniform(-10, 10, n_ground)
    ground_y = np.random.uniform(-10, 10, n_ground)
    ground_z = np.random.normal(0, 0.1, n_ground)  # Small noise for realism
    ground_points = np.column_stack([ground_x, ground_y, ground_z])
    
    # Car obstacle (rectangular box)
    n_car = 200
    car_x = np.random.uniform(3, 5, n_car)
    car_y = np.random.uniform(-1, 1, n_car)
    car_z = np.random.uniform(0, 1.5, n_car)
    car_points = np.column_stack([car_x, car_y, car_z])
    
    # Tree (cylinder)
    n_tree = 100
    theta = np.random.uniform(0, 2*np.pi, n_tree)
    tree_x = -5 + 0.5 * np.cos(theta)
    tree_y = 3 + 0.5 * np.sin(theta)
    tree_z = np.random.uniform(0, 3, n_tree)
    tree_points = np.column_stack([tree_x, tree_y, tree_z])
    
    # Combine all points
    all_points = np.vstack([ground_points, car_points, tree_points])
    
    # Add intensity values (simulating laser reflection strength)
    intensities = np.hstack([
        np.random.uniform(0.1, 0.3, n_ground),  # Dark ground
        np.random.uniform(0.6, 0.9, n_car),     # Reflective car
        np.random.uniform(0.4, 0.7, n_tree)     # Tree bark
    ])
    
    return all_points, intensities

# Generate the point cloud
points, intensities = create_sample_point_cloud()

print(f"🎉 Created point cloud with {len(points):,} points")
print(f"📏 Coordinate ranges:")
print(f"   X: {points[:, 0].min():.1f} to {points[:, 0].max():.1f} meters")
print(f"   Y: {points[:, 1].min():.1f} to {points[:, 1].max():.1f} meters")
print(f"   Z: {points[:, 2].min():.1f} to {points[:, 2].max():.1f} meters")
print(f"🔆 Intensity range: {intensities.min():.2f} to {intensities.max():.2f}")"""))

    # Visualization
    cells.append(create_notebook_cell("markdown", """## 🎨 Visualizing Point Clouds

Let's create beautiful 3D visualizations of our point cloud data:"""))

    cells.append(create_notebook_cell("code", """# Create an interactive 3D plot with Plotly
fig = go.Figure(data=go.Scatter3d(
    x=points[:, 0],
    y=points[:, 1],
    z=points[:, 2],
    mode='markers',
    marker=dict(
        size=2,
        color=points[:, 2],  # Color by height
        colorscale='viridis',
        colorbar=dict(title="Height (m)"),
        opacity=0.8
    ),
    text=[f'X: {x:.1f}<br>Y: {y:.1f}<br>Z: {z:.1f}<br>I: {i:.2f}' 
          for x, y, z, i in zip(points[:, 0], points[:, 1], points[:, 2], intensities)],
    hovertemplate='%{text}<extra></extra>'
))

fig.update_layout(
    title='🚗 Interactive 3D Point Cloud - Simulated Lidar Data',
    scene=dict(
        xaxis_title='X (meters)',
        yaxis_title='Y (meters)',
        zaxis_title='Z (meters)',
        camera=dict(
            eye=dict(x=1.5, y=1.5, z=1.5)
        )
    ),
    width=800,
    height=600
)

fig.show()

print("🎮 Interactive features:")
print("   • Rotate: Click and drag")
print("   • Zoom: Scroll wheel")
print("   • Pan: Hold Shift + click and drag")
print("   • Hover: Move mouse over points for details")"""))

    # Challenge section
    cells.append(create_notebook_cell("markdown", """## 🎯 Hands-On Challenge

Complete the functions below to analyze point cloud data:"""))

    cells.append(create_notebook_cell("code", """# 🚀 CHALLENGE: Implement your own point cloud analysis functions

def analyze_point_cloud_statistics(points):
    \"\"\"
    TODO: Analyze basic statistics of a point cloud
    
    Calculate and return:
    - Number of points
    - Bounding box (min and max coordinates)
    - Center point (centroid)
    - Average distance from origin
    
    Args:
        points: numpy array of shape (N, 3)
    
    Returns:
        dict with statistics
    \"\"\"
    # YOUR CODE HERE
    # Hint: Use np.min(), np.max(), np.mean(), np.linalg.norm()
    
    # Example implementation (remove and write your own!):
    stats = {
        'num_points': len(points),
        'bounding_box': {
            'min': points.min(axis=0).tolist(),
            'max': points.max(axis=0).tolist()
        },
        'centroid': points.mean(axis=0).tolist(),
        'avg_distance_from_origin': np.linalg.norm(points, axis=1).mean()
    }
    
    return stats

def filter_points_by_height(points, min_height, max_height):
    \"\"\"
    TODO: Filter points based on Z coordinate (height)
    
    Args:
        points: numpy array of shape (N, 3)
        min_height: minimum Z value to keep
        max_height: maximum Z value to keep
    
    Returns:
        filtered_points: numpy array of filtered points
        mask: boolean array indicating which points were kept
    \"\"\"
    # YOUR CODE HERE
    # Hint: Use boolean indexing with points[:, 2]
    
    mask = (points[:, 2] >= min_height) & (points[:, 2] <= max_height)
    filtered_points = points[mask]
    
    return filtered_points, mask

# Test your implementations
print("🧪 Testing your implementations...\\n")

# Test 1: Statistics
stats = analyze_point_cloud_statistics(points)
print("📊 Point Cloud Statistics:")
for key, value in stats.items():
    print(f"   {key}: {value}")

# Test 2: Height filtering
filtered_points, height_mask = filter_points_by_height(points, 0.5, 3.0)
print(f"\\n🔍 Height Filtering (0.5m to 3.0m):")
print(f"   Original points: {len(points):,}")
print(f"   Filtered points: {len(filtered_points):,}")
print(f"   Percentage kept: {len(filtered_points)/len(points)*100:.1f}%")

print("\\n✅ Great job! You've completed your first lidar analysis!")"""))

    # Conclusion
    cells.append(create_notebook_cell("markdown", """## 🎉 Congratulations!

You've successfully completed the Introduction to Lidar & Point Clouds lesson! 

### 🌟 What You've Learned:
- ✅ How lidar sensors work using time-of-flight
- ✅ Point cloud data structures and characteristics
- ✅ 3D coordinate systems and transformations
- ✅ Real-world data effects (noise, dropouts)
- ✅ Basic point cloud analysis techniques

### 🚀 Next Steps:
1. **Save your work**: File → Save Notebook
2. **Experiment**: Try modifying the parameters in the code cells
3. **Next lesson**: Open `02_parsing_pcd_ply_files.ipynb`

### 🎯 Challenge Yourself:
- Create your own synthetic scenes with different objects
- Implement additional filtering techniques
- Experiment with different coordinate transformations

Happy learning! 🚗✨"""))

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.8.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

def save_notebook(notebook_dict, filepath):
    """Save notebook dictionary as .ipynb file"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(notebook_dict, f, indent=2, ensure_ascii=False)
    print(f"✅ Created: {filepath}")

def main():
    """Generate all notebooks for the sensor fusion course"""
    print("🔧 Generating Interactive Jupyter Notebooks for Sensor Fusion Course...")
    
    # Create notebooks directory
    os.makedirs("notebooks/lidar", exist_ok=True)
    
    # Generate Lidar Introduction notebook
    lidar_intro = create_lidar_intro_notebook()
    save_notebook(lidar_intro, "notebooks/lidar/01_introduction_to_lidar.ipynb")
    
    print("\n🎉 Notebook generation complete!")
    print("📚 To use the notebooks:")
    print("   1. Install required packages: pip install -r requirements.txt")
    print("   2. Start Jupyter: jupyter lab")
    print("   3. Navigate to: notebooks/lidar/01_introduction_to_lidar.ipynb")
    print("   4. Start learning! 🚀")

if __name__ == "__main__":
    main()