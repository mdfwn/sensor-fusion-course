#!/usr/bin/env python3
"""
Enhanced Interactive Jupyter Notebook Generator for Sensor Fusion Course

Creates professional, well-formatted .ipynb files with:
- Proper markdown formatting
- Progress tracking
- Interactive elements
- Professional styling
- Better visual hierarchy
"""

import json
import os
from pathlib import Path

def create_notebook_cell(cell_type, content, metadata=None):
    """Create a properly formatted notebook cell"""
    if metadata is None:
        metadata = {}
    
    # Handle content formatting
    if isinstance(content, str):
        # Ensure proper line breaks and formatting
        lines = content.strip().split('\n')
        # Remove empty lines at start/end but preserve internal structure
        while lines and not lines[0].strip():
            lines.pop(0)
        while lines and not lines[-1].strip():
            lines.pop()
        source = lines
    else:
        source = content
    
    cell = {
        "cell_type": cell_type,
        "metadata": metadata,
        "source": source
    }
    
    if cell_type == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    
    return cell

def create_progress_cell(lesson_num, total_lessons, lesson_title):
    """Create a progress tracking cell"""
    progress_html = f"""
<div style="background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); 
            padding: 20px; border-radius: 15px; margin: 20px 0; color: white;">
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <div>
            <h2 style="margin: 0; font-size: 24px;">📚 Lesson {lesson_num}/{total_lessons}</h2>
            <h3 style="margin: 5px 0 0 0; font-size: 18px; opacity: 0.9;">{lesson_title}</h3>
        </div>
        <div style="text-align: right;">
            <div style="font-size: 36px;">🎯</div>
        </div>
    </div>
    <div style="background: rgba(255,255,255,0.2); height: 8px; border-radius: 4px; margin-top: 15px;">
        <div style="background: #4ade80; height: 100%; width: {(lesson_num/total_lessons)*100:.1f}%; 
                    border-radius: 4px; transition: width 0.3s ease;"></div>
    </div>
    <p style="margin: 10px 0 0 0; font-size: 14px; opacity: 0.8;">
        Progress: {(lesson_num/total_lessons)*100:.1f}% Complete
    </p>
</div>
"""
    return create_notebook_cell("markdown", progress_html)

def create_styled_header(title, subtitle="", emoji="🎯"):
    """Create a styled header cell"""
    header_content = f"""
<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
            padding: 30px; border-radius: 20px; text-align: center; 
            color: white; margin: 20px 0; box-shadow: 0 10px 30px rgba(0,0,0,0.2);">
    <h1 style="margin: 0; font-size: 36px; font-weight: bold;">
        {emoji} {title}
    </h1>
    {f'<p style="margin: 15px 0 0 0; font-size: 18px; opacity: 0.9;">{subtitle}</p>' if subtitle else ''}
</div>
"""
    return create_notebook_cell("markdown", header_content)

def create_learning_objectives_cell(objectives):
    """Create a styled learning objectives cell"""
    objectives_html = """
<div style="background: linear-gradient(135deg, #4ade80 0%, #22c55e 100%); 
            padding: 25px; border-radius: 15px; margin: 20px 0;">
    <h2 style="color: white; margin: 0 0 15px 0; display: flex; align-items: center;">
        <span style="font-size: 28px; margin-right: 10px;">🎯</span>
        Learning Objectives
    </h2>
    <div style="color: white;">
        <p style="margin: 0 0 10px 0; font-size: 16px; opacity: 0.9;">
            By the end of this lesson, you will be able to:
        </p>
        <ul style="margin: 0; padding-left: 20px;">
"""
    
    for objective in objectives:
        objectives_html += f'            <li style="margin: 8px 0; font-size: 15px;">{objective}</li>\n'
    
    objectives_html += """        </ul>
    </div>
</div>
"""
    return create_notebook_cell("markdown", objectives_html)

def create_section_header(title, emoji="📚", color="#3b82f6"):
    """Create a styled section header"""
    header_html = f"""
<div style="background: linear-gradient(90deg, {color} 0%, {color}dd 100%); 
            padding: 15px 25px; border-radius: 12px; margin: 25px 0 15px 0;">
    <h2 style="color: white; margin: 0; font-size: 22px; display: flex; align-items: center;">
        <span style="font-size: 26px; margin-right: 12px;">{emoji}</span>
        {title}
    </h2>
</div>
"""
    return create_notebook_cell("markdown", header_html)

def create_info_box(content, box_type="info"):
    """Create styled info/tip/warning boxes"""
    colors = {
        "info": {"bg": "#3b82f6", "border": "#1d4ed8"},
        "tip": {"bg": "#10b981", "border": "#047857"},
        "warning": {"bg": "#f59e0b", "border": "#d97706"},
        "success": {"bg": "#22c55e", "border": "#16a34a"}
    }
    
    icons = {
        "info": "💡",
        "tip": "🔥",
        "warning": "⚠️", 
        "success": "✅"
    }
    
    color = colors.get(box_type, colors["info"])
    icon = icons.get(box_type, icons["info"])
    
    box_html = f"""
<div style="border-left: 4px solid {color['border']}; 
            background: {color['bg']}20; 
            padding: 15px 20px; border-radius: 8px; margin: 15px 0;">
    <div style="display: flex; align-items: flex-start;">
        <span style="font-size: 20px; margin-right: 10px; margin-top: 2px;">{icon}</span>
        <div style="flex: 1;">
            {content}
        </div>
    </div>
</div>
"""
    return create_notebook_cell("markdown", box_html)

def create_lidar_intro_notebook():
    """Create an enhanced Introduction to Lidar notebook"""
    
    cells = []
    
    # Progress tracker
    cells.append(create_progress_cell(1, 7, "Introduction to Lidar & Point Clouds"))
    
    # Main header
    cells.append(create_styled_header(
        "Introduction to Lidar & Point Clouds",
        "Master the fundamentals of LiDAR technology and 3D point cloud processing",
        "🌟"
    ))
    
    # Learning objectives
    objectives = [
        "Understand the physics behind LiDAR sensors and time-of-flight measurements",
        "Work with 3D point cloud data structures in Python",
        "Create interactive 3D visualizations using Plotly",
        "Implement coordinate transformations and spatial analysis",
        "Analyze real-world LiDAR data characteristics and limitations"
    ]
    cells.append(create_learning_objectives_cell(objectives))
    
    # Introduction section
    cells.append(create_section_header("What is LiDAR?", "🔬", "#8b5cf6"))
    
    cells.append(create_notebook_cell("markdown", """
Imagine you're in a dark room and need to understand the layout of furniture around you. You could throw tennis balls in all directions and listen for when they bounce back — measuring both the direction you threw them and how long they took to return. 

**LiDAR (Light Detection and Ranging)** works on the same principle, except instead of tennis balls, it uses pulses of laser light!

### The Physics Behind LiDAR

LiDAR sensors operate using the **time-of-flight** principle:

$$\\text{Distance} = \\frac{c \\times \\Delta t}{2}$$

**Where:**
- $c$ = speed of light ($3 \\times 10^8$ m/s)
- $\\Delta t$ = round-trip time for the laser pulse  
- Division by 2 accounts for the round trip (sensor → object → sensor)
"""))
    
    # Info box about LiDAR applications
    cells.append(create_info_box("""
<strong>Real-World Applications:</strong><br>
• <strong>Autonomous Vehicles:</strong> Object detection, navigation, mapping<br>
• <strong>Robotics:</strong> SLAM (Simultaneous Localization and Mapping)<br>
• <strong>Archaeology:</strong> Discovering hidden structures under vegetation<br>
• <strong>Forestry:</strong> Tree height measurement and forest mapping<br>
• <strong>Agriculture:</strong> Crop monitoring and precision farming
""", "info"))

    # Setup section
    cells.append(create_section_header("Environment Setup", "🛠️", "#f59e0b"))
    
    cells.append(create_notebook_cell("code", """# 🚀 Import required libraries for LiDAR processing
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import plotly.express as px
from IPython.display import display, HTML
import warnings
warnings.filterwarnings('ignore')

# Configure matplotlib for better plots
plt.style.use('seaborn-v0_8')
plt.rcParams['figure.figsize'] = (12, 8)

# LiDAR processing libraries
try:
    import open3d as o3d
    print("✅ Open3D imported successfully - Ready for 3D point cloud processing!")
except ImportError:
    print("❌ Open3D not found. Install with: pip install open3d")
    print("   This library is essential for advanced point cloud operations.")

print("🎉 Environment setup complete! Ready to explore LiDAR technology.")"""))

    # Physics calculations section
    cells.append(create_section_header("Time-of-Flight Calculations", "⚡", "#ef4444"))
    
    cells.append(create_notebook_cell("markdown", """
Let's dive into the mathematics behind LiDAR measurements and see how tiny time differences translate to precise distance measurements.
"""))

    cells.append(create_notebook_cell("code", """# Physical constants and helper functions
SPEED_OF_LIGHT = 299_792_458  # m/s (exact value)

def calculate_distance(time_of_flight):
    \"\"\"
    Calculate distance using time-of-flight principle
    
    Args:
        time_of_flight (float): Round-trip time in seconds
        
    Returns:
        float: Distance in meters
    \"\"\"
    return (SPEED_OF_LIGHT * time_of_flight) / 2

def time_for_distance(distance):
    \"\"\"
    Calculate required time-of-flight for a given distance
    
    Args:
        distance (float): Target distance in meters
        
    Returns:
        float: Time-of-flight in seconds
    \"\"\"
    return (2 * distance) / SPEED_OF_LIGHT

# Real-world scenarios
scenarios = [
    (10, "🚶 Pedestrian crossing"),
    (50, "🚗 Car ahead"),
    (100, "🏢 Building"),
    (300, "🌳 Distant tree line"),
    (1000, "🏔️ Mountain/Large structure")
]

print("🎯 LiDAR Distance & Time-of-Flight Analysis")
print("=" * 55)
print(f"{'Distance':>10} | {'Time (ns)':>10} | {'Scenario'}")
print("-" * 55)

for distance, description in scenarios:
    time_ns = time_for_distance(distance) * 1e9  # Convert to nanoseconds
    print(f"{distance:>8}m | {time_ns:>8.1f} ns | {description}")

print(f"\\n💡 Notice how even at 1km distance, the time is only ~6.7 microseconds!")
print(f"   This requires extremely precise timing electronics in LiDAR sensors.")"""))

    # Interactive exercise section
    cells.append(create_section_header("Interactive Exercise: Your Turn!", "🎮", "#22c55e"))
    
    cells.append(create_info_box("""
<strong>Challenge:</strong> Modify the code below to explore different scenarios and understand how LiDAR timing works in practice!
""", "tip"))

    cells.append(create_notebook_cell("code", """# 🚀 Interactive Time-of-Flight Calculator
# TODO: Experiment with different values!

print("🧪 EXPERIMENT 1: Calculate time for your chosen distance")
print("-" * 50)

# Your scenario - change this distance!
your_distance = 25.0  # meters - try different values!
your_time = time_for_distance(your_distance)

print(f"📏 Distance: {your_distance} meters")
print(f"⏱️  Time-of-flight: {your_time*1e9:.2f} nanoseconds")
print(f"⚡ Time-of-flight: {your_time*1e6:.3f} microseconds")

print("\\n🧪 EXPERIMENT 2: Calculate distance from measured time")
print("-" * 50)

# Simulate a LiDAR measurement - change this time!
measured_time = 500e-9  # 500 nanoseconds - try different values!
calculated_distance = calculate_distance(measured_time)

print(f"⏱️  Measured time: {measured_time*1e9:.1f} nanoseconds")
print(f"📏 Calculated distance: {calculated_distance:.2f} meters")

print("\\n🎯 EXPERIMENT 3: Accuracy analysis")
print("-" * 50)

# LiDAR timing precision (typical values)
timing_precision = 0.1e-9  # 0.1 nanosecond precision
distance_precision = calculate_distance(timing_precision)

print(f"⚙️  Timing precision: {timing_precision*1e9:.1f} nanoseconds")
print(f"📐 Distance precision: {distance_precision*100:.1f} centimeters")
print(f"💼 This is why LiDAR can achieve centimeter-level accuracy!")"""))

    # Point cloud creation section
    cells.append(create_section_header("Creating 3D Point Clouds", "📊", "#8b5cf6"))
    
    cells.append(create_notebook_cell("markdown", """
Now let's create and work with actual 3D point cloud data! We'll simulate a realistic driving scenario with multiple objects.
"""))

    cells.append(create_notebook_cell("code", """def create_realistic_scene():
    \"\"\"
    Generate a realistic 3D scene with multiple objects
    
    Returns:
        tuple: (points, intensities, labels) arrays
    \"\"\"
    np.random.seed(42)  # Reproducible results
    
    all_points = []
    all_intensities = []
    all_labels = []
    
    # 1. Ground plane with realistic noise
    print("🛣️  Generating ground plane...")
    n_ground = 2000
    x_ground = np.random.uniform(-20, 20, n_ground)
    y_ground = np.random.uniform(-15, 15, n_ground)
    # Add slight slope and noise to ground
    z_ground = 0.01 * x_ground + np.random.normal(0, 0.05, n_ground)
    
    ground_points = np.column_stack([x_ground, y_ground, z_ground])
    ground_intensities = np.random.uniform(0.1, 0.25, n_ground)  # Dark asphalt
    ground_labels = np.zeros(n_ground, dtype=int)  # Label 0 = ground
    
    # 2. Multiple vehicles
    print("🚗 Adding vehicles...")
    vehicles = [
        {"center": [8, -2, 0.75], "size": [4.5, 1.8, 1.5], "n_points": 300},
        {"center": [-12, 3, 0.75], "size": [4.2, 1.7, 1.4], "n_points": 250},
        {"center": [15, 1, 0.9], "size": [5.2, 2.1, 1.8], "n_points": 350}  # Larger vehicle
    ]
    
    for i, vehicle in enumerate(vehicles):
        cx, cy, cz = vehicle["center"]
        sx, sy, sz = vehicle["size"]
        n_pts = vehicle["n_points"]
        
        # Create box-shaped point cloud for vehicle
        vx = np.random.uniform(cx - sx/2, cx + sx/2, n_pts)
        vy = np.random.uniform(cy - sy/2, cy + sy/2, n_pts)
        vz = np.random.uniform(cz - sz/2, cz + sz/2, n_pts)
        
        vehicle_points = np.column_stack([vx, vy, vz])
        vehicle_intensities = np.random.uniform(0.6, 0.85, n_pts)  # Metallic reflection
        vehicle_labels = np.full(n_pts, i + 1, dtype=int)  # Labels 1, 2, 3
        
        all_points.append(vehicle_points)
        all_intensities.append(vehicle_intensities)
        all_labels.append(vehicle_labels)
    
    # 3. Trees and vegetation
    print("🌳 Adding vegetation...")
    trees = [
        {"center": [-8, -10, 2], "radius": 1.5, "height": 4, "n_points": 400},
        {"center": [12, 8, 1.8], "radius": 1.2, "height": 3.5, "n_points": 350},
        {"center": [-15, 7, 2.2], "radius": 1.8, "height": 4.5, "n_points": 450}
    ]
    
    for i, tree in enumerate(trees):
        cx, cy, base_z = tree["center"]
        radius = tree["radius"]
        height = tree["height"]
        n_pts = tree["n_points"]
        
        # Create cylindrical tree structure
        theta = np.random.uniform(0, 2*np.pi, n_pts)
        r = np.random.uniform(0, radius, n_pts)
        tx = cx + r * np.cos(theta)
        ty = cy + r * np.sin(theta)
        tz = np.random.uniform(base_z, base_z + height, n_pts)
        
        tree_points = np.column_stack([tx, ty, tz])
        tree_intensities = np.random.uniform(0.3, 0.5, n_pts)  # Tree bark/leaves
        tree_labels = np.full(n_pts, 10 + i, dtype=int)  # Labels 10, 11, 12
        
        all_points.append(tree_points)
        all_intensities.append(tree_intensities)
        all_labels.append(tree_labels)
    
    # 4. Building/wall structure
    print("🏢 Adding building structure...")
    # Simple wall
    n_wall = 600
    wall_x = np.random.uniform(-5, 5, n_wall)
    wall_y = np.full(n_wall, 12)  # Fixed Y position
    wall_z = np.random.uniform(0, 6, n_wall)
    
    wall_points = np.column_stack([wall_x, wall_y, wall_z])
    wall_intensities = np.random.uniform(0.4, 0.7, n_wall)  # Concrete/brick
    wall_labels = np.full(n_wall, 20, dtype=int)  # Label 20 = building
    
    # Combine everything
    all_points.extend([ground_points, wall_points])
    all_intensities.extend([ground_intensities, wall_intensities])
    all_labels.extend([ground_labels, wall_labels])
    
    # Concatenate all arrays
    points = np.vstack(all_points)
    intensities = np.concatenate(all_intensities)
    labels = np.concatenate(all_labels)
    
    return points, intensities, labels

# Generate the scene
print("🎬 Creating realistic LiDAR scene...")
points, intensities, labels = create_realistic_scene()

# Display statistics
print(f"\\n📊 Scene Statistics:")
print(f"   Total points: {len(points):,}")
print(f"   Coordinate bounds:")
print(f"     X: {points[:, 0].min():.1f} to {points[:, 0].max():.1f} meters")
print(f"     Y: {points[:, 1].min():.1f} to {points[:, 1].max():.1f} meters") 
print(f"     Z: {points[:, 2].min():.1f} to {points[:, 2].max():.1f} meters")
print(f"   Intensity range: {intensities.min():.2f} to {intensities.max():.2f}")
print(f"   Unique objects: {len(np.unique(labels))}")

print("\\n✅ Scene generation complete! Ready for visualization.")"""))

    # Visualization section
    cells.append(create_section_header("3D Visualization & Exploration", "🎨", "#06b6d4"))
    
    cells.append(create_notebook_cell("code", """# Create stunning 3D visualization
def create_interactive_plot(points, intensities, labels):
    \"\"\"Create an interactive 3D plot with multiple viewing options\"\"\"
    
    # Create color mapping for different objects
    unique_labels = np.unique(labels)
    colors = px.colors.qualitative.Set3
    
    fig = go.Figure()
    
    # Add points colored by height (Z-coordinate)
    scatter = go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1], 
        z=points[:, 2],
        mode='markers',
        marker=dict(
            size=1.5,
            color=points[:, 2],  # Color by height
            colorscale='Viridis',
            colorbar=dict(
                title="Height (m)",
                titleside="right",
                thickness=15
            ),
            opacity=0.7,
            line=dict(width=0)
        ),
        text=[f'Point: ({x:.1f}, {y:.1f}, {z:.1f})<br>Intensity: {i:.2f}<br>Object: {l}' 
              for x, y, z, i, l in zip(points[:, 0], points[:, 1], points[:, 2], 
                                      intensities, labels)],
        hovertemplate='%{text}<extra></extra>',
        name='LiDAR Points'
    )
    
    fig.add_trace(scatter)
    
    # Update layout for professional appearance
    fig.update_layout(
        title={
            'text': '🚗 Interactive 3D LiDAR Point Cloud - Simulated Autonomous Vehicle View',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18}
        },
        scene=dict(
            xaxis=dict(
                title='X - Forward/Backward (meters)',
                backgroundcolor="rgb(240, 240, 240)",
                gridcolor="white",
                showbackground=True,
                zerolinecolor="white"
            ),
            yaxis=dict(
                title='Y - Left/Right (meters)',
                backgroundcolor="rgb(240, 240, 240)",
                gridcolor="white", 
                showbackground=True,
                zerolinecolor="white"
            ),
            zaxis=dict(
                title='Z - Up/Down (meters)',
                backgroundcolor="rgb(240, 240, 240)",
                gridcolor="white",
                showbackground=True,
                zerolinecolor="white"
            ),
            camera=dict(
                eye=dict(x=1.8, y=1.8, z=1.2),
                center=dict(x=0, y=0, z=0)
            ),
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=0.5)
        ),
        width=900,
        height=700,
        margin=dict(l=0, r=0, t=40, b=0)
    )
    
    return fig

# Generate visualization
print("🎨 Creating interactive 3D visualization...")
fig = create_interactive_plot(points, intensities, labels)
fig.show()

print("\\n🎮 Interactive Controls:")
print("   🖱️  Rotate: Click and drag")
print("   🔍 Zoom: Mouse wheel or pinch")
print("   ↔️  Pan: Shift + click and drag")
print("   🎯 Hover: Mouse over points for details")
print("   📷 Reset: Double-click")"""))

    # Add control panel info
    cells.append(create_info_box("""
<strong>Professional Tip:</strong> In real autonomous vehicles, LiDAR data is processed in real-time at 10-20 Hz. 
Each scan can contain 100,000+ points, requiring efficient algorithms for object detection and tracking!
""", "tip"))

    # Analysis section
    cells.append(create_section_header("Hands-On Point Cloud Analysis", "🔬", "#7c3aed"))
    
    cells.append(create_notebook_cell("markdown", """
Now let's implement some fundamental point cloud analysis techniques that are essential in autonomous driving systems.
"""))

    cells.append(create_notebook_cell("code", """# 🎯 Advanced Point Cloud Analysis Functions
# Your mission: Complete these functions for real-world LiDAR processing!

def analyze_scene_statistics(points, intensities):
    \"\"\"
    Comprehensive statistical analysis of LiDAR scene
    
    Args:
        points: numpy array of shape (N, 3) with XYZ coordinates  
        intensities: numpy array of shape (N,) with reflection values
        
    Returns:
        dict: Comprehensive statistics dictionary
    \"\"\"
    # TODO: Implement your analysis here!
    
    stats = {
        # Basic counts
        'total_points': len(points),
        'point_density': len(points) / ((points[:, 0].max() - points[:, 0].min()) * 
                                       (points[:, 1].max() - points[:, 1].min())),
        
        # Spatial extent
        'bounding_box': {
            'x_range': [float(points[:, 0].min()), float(points[:, 0].max())],
            'y_range': [float(points[:, 1].min()), float(points[:, 1].max())],
            'z_range': [float(points[:, 2].min()), float(points[:, 2].max())]
        },
        
        # Central tendency
        'centroid': points.mean(axis=0).tolist(),
        'median_position': np.median(points, axis=0).tolist(),
        
        # Spread measures
        'std_deviation': points.std(axis=0).tolist(),
        'coordinate_variance': points.var(axis=0).tolist(),
        
        # Distance analysis
        'distances_from_origin': np.linalg.norm(points, axis=1),
        'mean_distance': float(np.linalg.norm(points, axis=1).mean()),
        'max_distance': float(np.linalg.norm(points, axis=1).max()),
        
        # Intensity analysis
        'intensity_stats': {
            'mean': float(intensities.mean()),
            'std': float(intensities.std()),
            'min': float(intensities.min()),
            'max': float(intensities.max()),
            'percentiles': {
                '25th': float(np.percentile(intensities, 25)),
                '50th': float(np.percentile(intensities, 50)),
                '75th': float(np.percentile(intensities, 75))
            }
        }
    }
    
    return stats

def filter_ground_points(points, intensities, height_threshold=0.3):
    \"\"\"
    Separate ground points from elevated objects
    
    Args:
        points: numpy array of shape (N, 3)
        intensities: numpy array of shape (N,)
        height_threshold: maximum height to consider as ground (meters)
        
    Returns:
        tuple: (ground_points, ground_intensities, object_points, object_intensities)
    \"\"\"
    # TODO: Implement ground filtering logic
    
    # Simple height-based filtering (you can improve this!)
    ground_mask = points[:, 2] <= height_threshold
    object_mask = ~ground_mask
    
    ground_points = points[ground_mask]
    ground_intensities = intensities[ground_mask]
    object_points = points[object_mask]
    object_intensities = intensities[object_mask]
    
    return ground_points, ground_intensities, object_points, object_intensities

def detect_objects_by_clustering(points, min_cluster_size=20, max_distance=1.0):
    \"\"\"
    Simple object detection using spatial clustering
    
    Args:
        points: numpy array of shape (N, 3)
        min_cluster_size: minimum points to form a cluster
        max_distance: maximum distance between points in same cluster
        
    Returns:
        dict: cluster information
    \"\"\"
    # TODO: Implement a basic clustering algorithm
    # Hint: You could use distance-based grouping or k-means
    
    # Simplified clustering based on spatial proximity
    clusters = {}
    visited = np.zeros(len(points), dtype=bool)
    cluster_id = 0
    
    for i in range(len(points)):
        if visited[i]:
            continue
            
        # Find nearby points (simple euclidean distance)
        distances = np.linalg.norm(points - points[i], axis=1)
        nearby_mask = distances <= max_distance
        nearby_indices = np.where(nearby_mask)[0]
        
        if len(nearby_indices) >= min_cluster_size:
            clusters[cluster_id] = {
                'points': points[nearby_indices],
                'indices': nearby_indices.tolist(),
                'centroid': points[nearby_indices].mean(axis=0).tolist(),
                'size': len(nearby_indices)
            }
            visited[nearby_indices] = True
            cluster_id += 1
    
    return clusters

# Apply your analysis functions
print("🔬 COMPREHENSIVE SCENE ANALYSIS")
print("=" * 50)

# 1. Statistical analysis  
stats = analyze_scene_statistics(points, intensities)
print("\\n📊 Scene Statistics:")
print(f"   Total points: {stats['total_points']:,}")
print(f"   Point density: {stats['point_density']:.1f} points/m²")
print(f"   Scene bounds: X={stats['bounding_box']['x_range']}, Y={stats['bounding_box']['y_range']}, Z={stats['bounding_box']['z_range']}")
print(f"   Average distance from sensor: {stats['mean_distance']:.1f}m")
print(f"   Intensity range: {stats['intensity_stats']['min']:.2f} - {stats['intensity_stats']['max']:.2f}")

# 2. Ground filtering
ground_pts, ground_int, object_pts, object_int = filter_ground_points(points, intensities)
print(f"\\n🛣️  Ground Segmentation:")
print(f"   Ground points: {len(ground_pts):,} ({len(ground_pts)/len(points)*100:.1f}%)")
print(f"   Object points: {len(object_pts):,} ({len(object_pts)/len(points)*100:.1f}%)")

# 3. Object detection
if len(object_pts) > 0:
    clusters = detect_objects_by_clustering(object_pts)
    print(f"\\n🎯 Object Detection:")
    print(f"   Detected objects: {len(clusters)}")
    for cluster_id, cluster_info in clusters.items():
        centroid = cluster_info['centroid']
        print(f"   Object {cluster_id}: {cluster_info['size']} points at ({centroid[0]:.1f}, {centroid[1]:.1f}, {centroid[2]:.1f})")

print("\\n✅ Analysis complete! You've successfully processed real LiDAR data!")"""))

    # Conclusion and next steps
    cells.append(create_section_header("🎉 Lesson Complete!", "🏆", "#22c55e"))
    
    cells.append(create_notebook_cell("markdown", """
### 🌟 Congratulations! You've mastered LiDAR fundamentals!

#### What You've Accomplished:
- ✅ **Physics Mastery**: Understood time-of-flight principles and distance calculations
- ✅ **Data Structures**: Worked with 3D point clouds and intensity data  
- ✅ **Visualization**: Created interactive 3D plots for data exploration
- ✅ **Analysis Skills**: Implemented statistical analysis and object detection
- ✅ **Real-World Application**: Applied techniques used in autonomous vehicles

#### 🚀 Next Steps:
1. **Save Your Work**: `File → Save Notebook` 
2. **Experiment Further**: Try modifying parameters in the analysis functions
3. **Next Lesson**: Open `02_parsing_pcd_ply_files.ipynb` to learn about file formats
4. **Challenge Yourself**: Implement more sophisticated clustering algorithms

#### 🎯 Advanced Challenges:
- Experiment with different ground filtering techniques
- Implement RANSAC plane detection
- Try k-means clustering for object segmentation  
- Add noise simulation for realistic sensor modeling

### 📚 Continue Your Journey
Ready to dive deeper into LiDAR file formats and advanced processing? 
**Next up**: Parsing PCD & PLY Files! 🚀

---
*Happy Learning! You're on your way to becoming a sensor fusion expert!* 🎓✨
"""))

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
            },
            "widgets": {
                "application/vnd.jupyter.widget-state+json": {
                    "state": {},
                    "version_major": 2,
                    "version_minor": 0
                }
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

def save_notebook(notebook_dict, filepath):
    """Save notebook with proper formatting"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(notebook_dict, f, indent=2, ensure_ascii=False)
    print(f"✅ Created: {filepath}")

def main():
    """Generate enhanced notebooks"""
    print("🎨 Generating Enhanced Jupyter Notebooks for Sensor Fusion Course...")
    
    # Create notebooks directory
    os.makedirs("notebooks/lidar", exist_ok=True)
    
    # Generate enhanced LiDAR introduction notebook
    lidar_intro = create_lidar_intro_notebook()
    save_notebook(lidar_intro, "notebooks/lidar/01_introduction_to_lidar.ipynb")
    
    print("\n🎉 Enhanced notebook generation complete!")
    print("✨ Features added:")
    print("   • Professional styling with gradients and colors")
    print("   • Progress tracking throughout lessons")
    print("   • Interactive info boxes and tips")
    print("   • Better code organization and documentation")
    print("   • Enhanced visualizations")
    print("\n📚 To start learning:")
    print("   1. jupyter lab")
    print("   2. Open: notebooks/lidar/01_introduction_to_lidar.ipynb")
    print("   3. Start coding! 🚀")

if __name__ == "__main__":
    main()