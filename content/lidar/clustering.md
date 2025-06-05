# Euclidean Clustering with KD-Tree

## 🎯 From Points to Objects

Imagine you're looking at a crowded parking lot from above. You can see individual cars, but from a computer's perspective, you just have millions of scattered 3D points. How do you determine which points belong to the same car? Which points represent different vehicles?

**Euclidean clustering** solves exactly this problem! After removing the ground plane with RANSAC, we need to group the remaining obstacle points into discrete objects. This is crucial for autonomous vehicles to understand their environment: "There's a car 20 meters ahead, a pedestrian to the left, and a tree on the sidewalk."

```{admonition} 🚗 Why Clustering for Autonomous Vehicles?
:class: tip
Clustering transforms raw point clouds into semantic understanding:
- **Object Detection:** Identify individual cars, pedestrians, cyclists
- **Collision Avoidance:** Track specific objects and predict their motion  
- **Path Planning:** Navigate around discrete obstacles
- **Traffic Understanding:** Distinguish between moving and static objects
- **Safety Systems:** Emergency braking when objects get too close
```

## 🌳 Understanding KD-Trees

Before diving into clustering, we need an efficient way to find nearby points. A **KD-Tree (k-dimensional tree)** is a binary tree data structure that organizes points in k-dimensional space for fast spatial queries.

Think of a KD-Tree like organizing books in a library:
- First, split all books by subject (dimension 1)
- Within each subject, split by author's last name (dimension 2)  
- Within each author, split by publication year (dimension 3)
- And so on...

### KD-Tree Structure

For 3D points, a KD-Tree alternates splitting along x, y, and z axes:

```
Level 0: Split on X-axis
├─ Level 1: Split on Y-axis  
│  ├─ Level 2: Split on Z-axis
│  └─ Level 2: Split on Z-axis
└─ Level 1: Split on Y-axis
   ├─ Level 2: Split on Z-axis
   └─ Level 2: Split on Z-axis
```

### Mathematical Foundation

**Distance Metric:** For Euclidean clustering, we use the standard Euclidean distance:

$$d(\mathbf{p_1}, \mathbf{p_2}) = \sqrt{(x_1-x_2)^2 + (y_1-y_2)^2 + (z_1-z_2)^2}$$

**Nearest Neighbor Search:** Given a query point $\mathbf{q}$ and radius $r$, find all points $\mathbf{p}$ such that:

$$d(\mathbf{q}, \mathbf{p}) \leq r$$

Let's implement a KD-Tree from scratch:

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import deque
import time

class KDNode:
    """Node in a KD-Tree"""
    def __init__(self, point, axis, left=None, right=None):
        self.point = point      # 3D point stored in this node
        self.axis = axis        # Splitting axis (0=x, 1=y, 2=z)
        self.left = left        # Left child (points with smaller values)
        self.right = right      # Right child (points with larger values)

class KDTree:
    """3D KD-Tree implementation for fast spatial queries"""
    
    def __init__(self, points):
        """
        Build KD-Tree from array of 3D points
        
        Args:
            points: numpy array of shape (N, 3)
        """
        self.points = np.array(points)
        self.root = self._build_tree(self.points, depth=0)
    
    def _build_tree(self, points, depth):
        """Recursively build KD-Tree"""
        if len(points) == 0:
            return None
        
        # Select axis to split on (cycle through x, y, z)
        axis = depth % 3
        
        # Sort points by the selected axis and find median
        sorted_points = points[points[:, axis].argsort()]
        median_idx = len(sorted_points) // 2
        
        # Create node with median point
        node = KDNode(
            point=sorted_points[median_idx],
            axis=axis
        )
        
        # Recursively build left and right subtrees
        node.left = self._build_tree(sorted_points[:median_idx], depth + 1)
        node.right = self._build_tree(sorted_points[median_idx + 1:], depth + 1)
        
        return node
    
    def radius_search(self, query_point, radius):
        """
        Find all points within radius of query point
        
        Args:
            query_point: numpy array of shape (3,)
            radius: search radius
            
        Returns:
            list of points within radius
        """
        neighbors = []
        self._radius_search_recursive(self.root, query_point, radius, neighbors)
        return neighbors
    
    def _radius_search_recursive(self, node, query_point, radius, neighbors):
        """Recursive radius search implementation"""
        if node is None:
            return
        
        # Calculate distance to current node's point
        distance = np.linalg.norm(node.point - query_point)
        
        # If within radius, add to neighbors
        if distance <= radius:
            neighbors.append(node.point)
        
        # Determine which child to search first
        axis = node.axis
        if query_point[axis] < node.point[axis]:
            # Query point is on left side
            self._radius_search_recursive(node.left, query_point, radius, neighbors)
            # Check if we need to search right side too
            if query_point[axis] + radius >= node.point[axis]:
                self._radius_search_recursive(node.right, query_point, radius, neighbors)
        else:
            # Query point is on right side  
            self._radius_search_recursive(node.right, query_point, radius, neighbors)
            # Check if we need to search left side too
            if query_point[axis] - radius <= node.point[axis]:
                self._radius_search_recursive(node.left, query_point, radius, neighbors)

# Create test data for KD-Tree
def create_clustered_test_data():
    """Generate synthetic point cloud with distinct clusters"""
    np.random.seed(42)
    
    clusters = []
    
    # Cluster 1: Car at (5, 0, 1)
    car_center = np.array([5.0, 0.0, 1.0])
    car_points = car_center + np.random.normal(0, 0.3, (200, 3))
    clusters.append(car_points)
    
    # Cluster 2: Tree at (-3, 4, 2)
    tree_center = np.array([-3.0, 4.0, 2.0])
    tree_points = tree_center + np.random.normal(0, 0.5, (150, 3))
    clusters.append(tree_points)
    
    # Cluster 3: Building wall at (0, 8, 3)
    building_center = np.array([0.0, 8.0, 3.0])
    building_points = building_center + np.random.normal(0, 0.4, (300, 3))
    clusters.append(building_points)
    
    # Cluster 4: Small obstacle at (2, -3, 0.5)
    obstacle_center = np.array([2.0, -3.0, 0.5])
    obstacle_points = obstacle_center + np.random.normal(0, 0.2, (80, 3))
    clusters.append(obstacle_points)
    
    # Add some noise points
    noise_points = np.random.uniform(-10, 10, (50, 3))
    clusters.append(noise_points)
    
    # Combine all points
    all_points = np.vstack(clusters)
    
    # Create ground truth labels
    labels = []
    for i, cluster in enumerate(clusters):
        labels.extend([i] * len(cluster))
    
    return all_points, np.array(labels)

# Test KD-Tree implementation
test_points, true_labels = create_clustered_test_data()
print(f"Created test dataset with {len(test_points)} points in {len(np.unique(true_labels))} clusters")

# Build KD-Tree
start_time = time.time()
kdtree = KDTree(test_points)
build_time = time.time() - start_time
print(f"Built KD-Tree in {build_time:.4f} seconds")

# Test radius search
query_point = test_points[0]  # Use first point as query
search_radius = 1.0

start_time = time.time()
neighbors = kdtree.radius_search(query_point, search_radius)
search_time = time.time() - start_time

print(f"Radius search found {len(neighbors)} neighbors in {search_time:.4f} seconds")
print(f"Query point: ({query_point[0]:.2f}, {query_point[1]:.2f}, {query_point[2]:.2f})")
```

## 🔗 Euclidean Clustering Algorithm

Now that we have an efficient spatial search structure, we can implement Euclidean clustering. This algorithm groups points that are close to each other, forming connected components.

### Algorithm Steps

1. **Initialize:** Start with all points unmarked
2. **Seed Selection:** Pick an unmarked point as cluster seed
3. **Region Growing:** Find all neighbors within distance threshold
4. **Recursive Expansion:** For each neighbor, find its neighbors
5. **Cluster Completion:** Continue until no more points can be added
6. **Repeat:** Start new cluster with next unmarked point

### Mathematical Analysis

For a cluster to be valid, it must satisfy:
- **Connectivity:** Every point has at least one neighbor within distance $d_{threshold}$
- **Density:** Clusters have minimum number of points $N_{min}$
- **Separation:** Different clusters are separated by distances > $d_{threshold}$

```python
def euclidean_clustering(points, cluster_tolerance=0.5, min_cluster_size=10, max_cluster_size=5000):
    """
    Perform Euclidean clustering on 3D point cloud
    
    Args:
        points: numpy array of shape (N, 3) containing 3D points
        cluster_tolerance: maximum distance between points in same cluster
        min_cluster_size: minimum points required to form a cluster
        max_cluster_size: maximum points allowed in a cluster
        
    Returns:
        cluster_labels: array of cluster assignments (-1 for noise)
        cluster_centers: array of cluster centroids
        num_clusters: number of clusters found
    """
    
    if len(points) == 0:
        return np.array([]), np.array([]), 0
    
    # Build KD-Tree for efficient neighbor search
    print(f"Building KD-Tree for {len(points)} points...")
    kdtree = KDTree(points)
    
    # Initialize cluster tracking
    cluster_labels = np.full(len(points), -1)  # -1 means unassigned
    current_cluster_id = 0
    
    print(f"Starting clustering with tolerance={cluster_tolerance:.2f}m...")
    
    for point_idx in range(len(points)):
        # Skip if point already assigned to a cluster
        if cluster_labels[point_idx] != -1:
            continue
        
        # Start new cluster with this point
        cluster_points = []
        points_to_process = deque([point_idx])
        
        # Region growing: find all connected points
        while points_to_process:
            current_idx = points_to_process.popleft()
            
            # Skip if already processed
            if cluster_labels[current_idx] != -1:
                continue
            
            # Add current point to cluster
            cluster_labels[current_idx] = current_cluster_id
            cluster_points.append(current_idx)
            
            # Find neighbors of current point
            neighbors = kdtree.radius_search(points[current_idx], cluster_tolerance)
            
            # Add unprocessed neighbors to processing queue
            for neighbor in neighbors:
                # Find index of neighbor point
                neighbor_idx = None
                for idx, point in enumerate(points):
                    if np.allclose(point, neighbor, atol=1e-6):
                        neighbor_idx = idx
                        break
                
                if neighbor_idx is not None and cluster_labels[neighbor_idx] == -1:
                    points_to_process.append(neighbor_idx)
        
        # Validate cluster size
        if len(cluster_points) < min_cluster_size:
            # Mark as noise if too small
            for idx in cluster_points:
                cluster_labels[idx] = -1
            print(f"Rejected small cluster with {len(cluster_points)} points")
        elif len(cluster_points) > max_cluster_size:
            # Mark as noise if too large (likely merged clusters)
            for idx in cluster_points:
                cluster_labels[idx] = -1
            print(f"Rejected large cluster with {len(cluster_points)} points")
        else:
            # Valid cluster
            print(f"Cluster {current_cluster_id}: {len(cluster_points)} points")
            current_cluster_id += 1
    
    # Calculate cluster centers
    cluster_centers = []
    for cluster_id in range(current_cluster_id):
        cluster_mask = cluster_labels == cluster_id
        if np.any(cluster_mask):
            center = np.mean(points[cluster_mask], axis=0)
            cluster_centers.append(center)
    
    cluster_centers = np.array(cluster_centers)
    
    # Print summary
    noise_points = np.sum(cluster_labels == -1)
    clustered_points = len(points) - noise_points
    
    print(f"\nClustering Results:")
    print(f"  Total points: {len(points)}")
    print(f"  Clustered points: {clustered_points}")
    print(f"  Noise points: {noise_points}")
    print(f"  Number of clusters: {current_cluster_id}")
    print(f"  Average cluster size: {clustered_points / max(1, current_cluster_id):.1f}")
    
    return cluster_labels, cluster_centers, current_cluster_id

# Run clustering on test data
cluster_labels, cluster_centers, num_clusters = euclidean_clustering(
    test_points,
    cluster_tolerance=0.8,  # 80cm tolerance
    min_cluster_size=20,    # At least 20 points per cluster
    max_cluster_size=1000   # At most 1000 points per cluster
)
```

## 📊 Clustering Evaluation and Visualization

Let's evaluate our clustering performance and create comprehensive visualizations:

```python
def evaluate_clustering(true_labels, predicted_labels):
    """
    Evaluate clustering performance using various metrics
    
    Args:
        true_labels: ground truth cluster assignments
        predicted_labels: algorithm's cluster assignments
        
    Returns:
        dict with evaluation metrics
    """
    
    # Handle noise points (label -1)
    valid_mask = predicted_labels != -1
    valid_true = true_labels[valid_mask]
    valid_pred = predicted_labels[valid_mask]
    
    if len(valid_pred) == 0:
        return {'error': 'No valid clusters found'}
    
    # Calculate basic statistics
    true_clusters = len(np.unique(valid_true))
    pred_clusters = len(np.unique(valid_pred))
    noise_ratio = np.sum(~valid_mask) / len(predicted_labels)
    
    # Adjusted Rand Index (measures clustering similarity)
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    
    ari = adjusted_rand_score(valid_true, valid_pred)
    nmi = normalized_mutual_info_score(valid_true, valid_pred)
    
    # Cluster-level analysis
    cluster_sizes = []
    for cluster_id in np.unique(valid_pred):
        cluster_size = np.sum(valid_pred == cluster_id)
        cluster_sizes.append(cluster_size)
    
    metrics = {
        'true_clusters': true_clusters,
        'predicted_clusters': pred_clusters,
        'noise_ratio': noise_ratio,
        'adjusted_rand_index': ari,
        'normalized_mutual_info': nmi,
        'mean_cluster_size': np.mean(cluster_sizes),
        'std_cluster_size': np.std(cluster_sizes),
        'min_cluster_size': np.min(cluster_sizes),
        'max_cluster_size': np.max(cluster_sizes)
    }
    
    return metrics

def visualize_clustering_results(points, true_labels, predicted_labels, cluster_centers):
    """Create comprehensive visualization of clustering results"""
    
    fig = plt.figure(figsize=(16, 12))
    
    # Plot 1: Original data with true labels
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    
    # Color points by true clusters
    unique_true = np.unique(true_labels)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_true)))
    
    for i, cluster_id in enumerate(unique_true):
        mask = true_labels == cluster_id
        if cluster_id == -1:  # Noise points
            ax1.scatter(points[mask, 0], points[mask, 1], points[mask, 2],
                       c='black', alpha=0.3, s=1, label='Noise')
        else:
            ax1.scatter(points[mask, 0], points[mask, 1], points[mask, 2],
                       c=[colors[i]], alpha=0.7, s=2, label=f'True Cluster {cluster_id}')
    
    ax1.set_title('Ground Truth Clusters')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    
    # Plot 2: Predicted clusters
    ax2 = fig.add_subplot(2, 3, 2, projection='3d')
    
    unique_pred = np.unique(predicted_labels)
    pred_colors = plt.cm.Set1(np.linspace(0, 1, len(unique_pred)))
    
    for i, cluster_id in enumerate(unique_pred):
        mask = predicted_labels == cluster_id
        if cluster_id == -1:  # Noise points
            ax2.scatter(points[mask, 0], points[mask, 1], points[mask, 2],
                       c='black', alpha=0.3, s=1, label='Noise')
        else:
            ax2.scatter(points[mask, 0], points[mask, 1], points[mask, 2],
                       c=[pred_colors[i]], alpha=0.7, s=2, label=f'Pred Cluster {cluster_id}')
    
    # Plot cluster centers
    if len(cluster_centers) > 0:
        ax2.scatter(cluster_centers[:, 0], cluster_centers[:, 1], cluster_centers[:, 2],
                   c='red', s=100, marker='x', linewidth=3, label='Centers')
    
    ax2.set_title('Predicted Clusters')
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_zlabel('Z (m)')
    
    # Plot 3: Cluster size distribution
    ax3 = fig.add_subplot(2, 3, 3)
    
    predicted_valid = predicted_labels[predicted_labels != -1]
    if len(predicted_valid) > 0:
        unique_pred_valid = np.unique(predicted_valid)
        cluster_sizes = [np.sum(predicted_labels == cid) for cid in unique_pred_valid]
        
        ax3.bar(range(len(cluster_sizes)), cluster_sizes)
        ax3.set_xlabel('Cluster ID')
        ax3.set_ylabel('Number of Points')
        ax3.set_title('Cluster Size Distribution')
        ax3.grid(True, alpha=0.3)
    
    # Plot 4: Distance distribution analysis
    ax4 = fig.add_subplot(2, 3, 4)
    
    # Calculate inter-point distances within clusters
    intra_cluster_distances = []
    for cluster_id in np.unique(predicted_labels):
        if cluster_id == -1:
            continue
        cluster_points = points[predicted_labels == cluster_id]
        if len(cluster_points) > 1:
            for i in range(len(cluster_points)):
                for j in range(i+1, len(cluster_points)):
                    dist = np.linalg.norm(cluster_points[i] - cluster_points[j])
                    intra_cluster_distances.append(dist)
    
    if intra_cluster_distances:
        ax4.hist(intra_cluster_distances, bins=30, alpha=0.7, label='Intra-cluster')
    
    ax4.set_xlabel('Distance (m)')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Distance Distribution')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: 2D projection (XY plane)
    ax5 = fig.add_subplot(2, 3, 5)
    
    for i, cluster_id in enumerate(unique_pred):
        mask = predicted_labels == cluster_id
        if cluster_id == -1:
            ax5.scatter(points[mask, 0], points[mask, 1], c='black', alpha=0.3, s=1)
        else:
            ax5.scatter(points[mask, 0], points[mask, 1], c=[pred_colors[i]], alpha=0.7, s=2)
    
    if len(cluster_centers) > 0:
        ax5.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='red', s=100, marker='x')
    
    ax5.set_xlabel('X (m)')
    ax5.set_ylabel('Y (m)')
    ax5.set_title('Top-Down View (XY Plane)')
    ax5.grid(True, alpha=0.3)
    ax5.axis('equal')
    
    # Plot 6: Evaluation metrics
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')
    
    # Calculate and display metrics
    metrics = evaluate_clustering(true_labels, predicted_labels)
    
    metrics_text = []
    for key, value in metrics.items():
        if isinstance(value, float):
            metrics_text.append(f"{key.replace('_', ' ').title()}: {value:.3f}")
        else:
            metrics_text.append(f"{key.replace('_', ' ').title()}: {value}")
    
    ax6.text(0.1, 0.9, "Clustering Metrics:", fontsize=14, fontweight='bold', transform=ax6.transAxes)
    for i, text in enumerate(metrics_text):
        ax6.text(0.1, 0.8 - i*0.08, text, fontsize=10, transform=ax6.transAxes)
    
    plt.tight_layout()
    plt.show()
    
    return fig, metrics

# Evaluate and visualize results
try:
    fig, metrics = visualize_clustering_results(test_points, true_labels, cluster_labels, cluster_centers)
    
    print("\nDetailed Evaluation Metrics:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key.replace('_', ' ').title()}: {value:.3f}")
        else:
            print(f"  {key.replace('_', ' ').title()}: {value}")
            
except ImportError:
    print("Scikit-learn not available for advanced metrics. Computing basic statistics...")
    
    # Basic evaluation without sklearn
    valid_mask = cluster_labels != -1
    noise_ratio = np.sum(~valid_mask) / len(cluster_labels)
    unique_clusters = len(np.unique(cluster_labels[valid_mask])) if np.any(valid_mask) else 0
    
    print(f"Basic Metrics:")
    print(f"  Predicted clusters: {unique_clusters}")
    print(f"  Noise ratio: {noise_ratio:.3f}")
    print(f"  Clustered points: {np.sum(valid_mask)}")
```

## 🚗 Autonomous Vehicle Applications

### Complete Pipeline: Ground Removal + Object Clustering

```python
def autonomous_vehicle_object_detection(points, ground_detection_params=None, clustering_params=None):
    """
    Complete object detection pipeline for autonomous vehicles
    
    Args:
        points: numpy array of shape (N, 3) containing lidar points
        ground_detection_params: dict with RANSAC parameters
        clustering_params: dict with clustering parameters
        
    Returns:
        dict with detection results
    """
    
    # Default parameters
    if ground_detection_params is None:
        ground_detection_params = {
            'distance_threshold': 0.15,
            'max_iterations': 1000,
            'min_inliers': 100
        }
    
    if clustering_params is None:
        clustering_params = {
            'cluster_tolerance': 0.5,
            'min_cluster_size': 30,
            'max_cluster_size': 3000
        }
    
    print("=== Autonomous Vehicle Object Detection Pipeline ===")
    
    # Step 1: Ground plane detection and removal
    print("\n1. Ground Plane Detection...")
    
    # Import RANSAC from previous lesson (simplified version)
    def simple_ransac_ground_detection(points, distance_threshold=0.15):
        """Simplified RANSAC for ground detection"""
        # Filter points likely to be ground
        height_mask = (points[:, 2] > -2.0) & (points[:, 2] < 1.0)
        ground_candidates = points[height_mask]
        
        if len(ground_candidates) < 3:
            return np.array([]), points
        
        # Simple plane fitting (assume roughly horizontal ground)
        ground_z = np.median(ground_candidates[:, 2])
        ground_mask = np.abs(points[:, 2] - ground_z) < distance_threshold
        
        ground_points = points[ground_mask]
        obstacle_points = points[~ground_mask]
        
        return ground_points, obstacle_points
    
    ground_points, obstacle_points = simple_ransac_ground_detection(
        points, ground_detection_params['distance_threshold']
    )
    
    print(f"   Ground points: {len(ground_points):,}")
    print(f"   Obstacle points: {len(obstacle_points):,}")
    
    if len(obstacle_points) == 0:
        return {
            'ground_points': ground_points,
            'obstacle_points': obstacle_points,
            'clusters': [],
            'cluster_centers': np.array([]),
            'bounding_boxes': []
        }
    
    # Step 2: Object clustering
    print("\n2. Object Clustering...")
    
    cluster_labels, cluster_centers, num_clusters = euclidean_clustering(
        obstacle_points, **clustering_params
    )
    
    # Step 3: Extract individual clusters
    clusters = []
    bounding_boxes = []
    
    for cluster_id in range(num_clusters):
        cluster_mask = cluster_labels == cluster_id
        cluster_points = obstacle_points[cluster_mask]
        
        if len(cluster_points) > 0:
            # Calculate bounding box
            min_coords = np.min(cluster_points, axis=0)
            max_coords = np.max(cluster_points, axis=0)
            
            # Calculate dimensions
            length = max_coords[0] - min_coords[0]
            width = max_coords[1] - min_coords[1]
            height = max_coords[2] - min_coords[2]
            
            # Classify object type based on size
            if height < 0.5 and max(length, width) < 1.0:
                object_type = "small_obstacle"
            elif height > 2.5:
                object_type = "building/tree"
            elif 1.0 < max(length, width) < 6.0 and 1.0 < height < 3.0:
                object_type = "vehicle"
            elif height < 2.0 and max(length, width) < 2.0:
                object_type = "pedestrian/cyclist"
            else:
                object_type = "unknown"
            
            cluster_info = {
                'id': cluster_id,
                'points': cluster_points,
                'center': cluster_centers[cluster_id],
                'num_points': len(cluster_points),
                'dimensions': [length, width, height],
                'object_type': object_type,
                'bounding_box': {
                    'min': min_coords,
                    'max': max_coords
                }
            }
            
            clusters.append(cluster_info)
            bounding_boxes.append((min_coords, max_coords))
    
    # Step 4: Summary
    print(f"\n3. Detection Summary:")
    print(f"   Total objects detected: {len(clusters)}")
    
    object_counts = {}
    for cluster in clusters:
        obj_type = cluster['object_type']
        object_counts[obj_type] = object_counts.get(obj_type, 0) + 1
    
    for obj_type, count in object_counts.items():
        print(f"   {obj_type.replace('_', ' ').title()}: {count}")
    
    return {
        'ground_points': ground_points,
        'obstacle_points': obstacle_points,
        'clusters': clusters,
        'cluster_centers': cluster_centers,
        'bounding_boxes': bounding_boxes,
        'object_counts': object_counts
    }

# Test complete pipeline
detection_results = autonomous_vehicle_object_detection(test_points)

# Display detailed results
print("\n=== Detailed Detection Results ===")
for cluster in detection_results['clusters']:
    print(f"\nObject {cluster['id']} ({cluster['object_type']}):")
    print(f"  Points: {cluster['num_points']}")
    print(f"  Center: ({cluster['center'][0]:.2f}, {cluster['center'][1]:.2f}, {cluster['center'][2]:.2f})")
    print(f"  Dimensions: L={cluster['dimensions'][0]:.2f}m, W={cluster['dimensions'][1]:.2f}m, H={cluster['dimensions'][2]:.2f}m")
```

## 🧩 Knowledge Check Quiz

<div class="quiz-container" data-quiz-id="lidar-clustering-1" data-correct="b">
  <div class="quiz-question">
    <strong>Question 1:</strong> What is the primary advantage of using a KD-Tree for spatial queries?
  </div>
  <ul class="quiz-options">
    <li data-value="a">It uses less memory than other data structures</li>
    <li data-value="b">It reduces search time complexity from O(n) to O(log n) on average</li>
    <li data-value="c">It automatically clusters points</li>
    <li data-value="d">It works only with 3D data</li>
  </ul>
</div>

<div class="quiz-container" data-quiz-id="lidar-clustering-2" data-correct="c">
  <div class="quiz-question">
    <strong>Question 2:</strong> In Euclidean clustering, what determines whether two points belong to the same cluster?
  </div>
  <ul class="quiz-options">
    <li data-value="a">They have similar intensity values</li>
    <li data-value="b">They are at the same height</li>
    <li data-value="c">They are within a specified distance threshold</li>
    <li data-value="d">They have the same color</li>
  </ul>
</div>

<div class="quiz-container" data-quiz-id="lidar-clustering-3" data-correct="a">
  <div class="quiz-question">
    <strong>Question 3:</strong> Why is it important to remove ground points before clustering objects?
  </div>
  <ul class="quiz-options">
    <li data-value="a">Ground points would form one large cluster, interfering with object detection</li>
    <li data-value="b">Ground points have different colors</li>
    <li data-value="c">Ground points are always noise</li>
    <li data-value="d">Ground points slow down the algorithm</li>
  </ul>
</div>

<div class="quiz-container" data-quiz-id="lidar-clustering-4" data-correct="d">
  <div class="quiz-question">
    <strong>Question 4:</strong> What happens if you set the cluster tolerance too large in Euclidean clustering?
  </div>
  <ul class="quiz-options">
    <li data-value="a">The algorithm will run faster</li>
    <li data-value="b">You'll get more clusters</li>
    <li data-value="c">Noise points will be eliminated</li>
    <li data-value="d">Separate objects may be merged into single clusters</li>
  </ul>
</div>

<div class="quiz-container" data-quiz-id="lidar-clustering-5" data-correct="b">
  <div class="quiz-question">
    <strong>Question 5:</strong> In the context of autonomous vehicles, what is the main purpose of clustering lidar points?
  </div>
  <ul class="quiz-options">
    <li data-value="a">To reduce data storage requirements</li>
    <li data-value="b">To identify and track individual objects in the environment</li>
    <li data-value="c">To improve lidar sensor accuracy</li>
    <li data-value="d">To colorize the point cloud</li>
  </ul>
</div>

## 🛠️ Hands-On Exercise

<div class="exercise-notebook">
  <div class="exercise-title">🚗 Exercise: Multi-Object Detection and Tracking</div>
  
  <p><strong>Goal:</strong> Build a comprehensive object detection system that can identify and classify multiple types of objects in a lidar point cloud.</p>
  
  <p><strong>Tasks:</strong></p>
  <ol>
    <li>Create a complex urban scene with cars, pedestrians, cyclists, and buildings</li>
    <li>Implement adaptive clustering that adjusts parameters based on object type</li>
    <li>Add object classification based on size, shape, and movement patterns</li>
    <li>Implement temporal tracking to maintain object IDs across frames</li>
    <li>Create safety alerts for objects in collision trajectories</li>
    <li>Optimize the pipeline for real-time performance (>10 Hz)</li>
  </ol>
  
  <p><strong>Success Criteria:</strong></p>
  <ul>
    <li>Detect >95% of vehicles and large objects</li>
    <li>Classify object types with >80% accuracy</li>
    <li>Maintain object tracks for >5 consecutive frames</li>
    <li>Process 50,000 points in <100ms</li>
    <li>Generate appropriate safety alerts</li>
  </ul>
</div>

```python
def multi_object_detection_exercise():
    """
    TODO: Implement comprehensive multi-object detection system
    
    Your implementation should include:
    1. Complex scene generation with multiple object types
    2. Adaptive clustering algorithms
    3. Object classification and tracking
    4. Safety alert system
    5. Real-time performance optimization
    """
    
    # Challenge 1: Complex scene generation
    # TODO: Create realistic urban environment with:
    # - Multiple vehicles of different sizes
    # - Pedestrians and cyclists at crosswalks
    # - Static infrastructure (buildings, signs, barriers)
    # - Moving objects with realistic trajectories
    # - Varying point densities and noise levels
    
    # Challenge 2: Adaptive clustering
    # TODO: Implement clustering that adapts to object characteristics:
    # - Different distance thresholds for different object types
    # - Size-based cluster validation
    # - Shape analysis for object classification
    # - Handling of partial occlusions
    
    # Challenge 3: Object classification
    # TODO: Classify detected clusters:
    # - Size-based classification (vehicle vs. pedestrian vs. bicycle)
    # - Shape analysis (elongated vs. compact objects)
    # - Height distribution analysis
    # - Movement pattern recognition
    
    # Challenge 4: Temporal tracking
    # TODO: Maintain object identities across time:
    # - Data association between consecutive frames
    # - Kalman filter for object state estimation
    # - Handle object appearance/disappearance
    # - Track merging and splitting
    
    # Challenge 5: Safety alert system
    # TODO: Implement collision detection:
    # - Predict object trajectories
    # - Calculate time-to-collision
    # - Generate appropriate warnings
    # - Prioritize alerts by threat level
    
    # Challenge 6: Performance optimization
    # TODO: Optimize for real-time operation:
    # - Spatial indexing for fast neighbor search
    # - Multi-threading for parallel processing
    # - Memory-efficient data structures
    # - Early termination strategies
    
    results = {
        'detection_accuracy': 0.0,      # TODO: Fill in
        'classification_accuracy': 0.0,
        'tracking_success_rate': 0.0,
        'processing_time_ms': 0.0,
        'safety_alerts_generated': 0,
        'objects_tracked': 0
    }
    
    return results

# Run the comprehensive exercise
import time
start_time = time.time()
exercise_results = multi_object_detection_exercise()
total_time = time.time() - start_time

print(f"Multi-object detection exercise completed in {total_time:.2f} seconds")
print("Results:", exercise_results)

# Mark exercise complete when finished
# sensorFusionUtils.markExerciseComplete('lidar-multi-object-detection');
```

## 🎯 Key Takeaways

```{admonition} 📝 What You've Learned
:class: note
1. **KD-Trees:** Efficient spatial data structures for fast neighbor search in 3D space
2. **Euclidean Clustering:** Region-growing algorithm for grouping nearby points into objects
3. **Object Detection:** Complete pipeline from raw points to classified objects
4. **Performance Optimization:** Techniques for real-time processing of large point clouds
5. **Evaluation Metrics:** Methods to assess clustering quality and detection accuracy
6. **Automotive Applications:** Practical object detection for autonomous vehicle perception
```

## 🚀 What's Next?

In the next lesson, we'll learn about **bounding box generation** — creating precise geometric boundaries around detected objects for tracking, collision detection, and path planning applications!

```{admonition} 🎯 Next Lesson Preview
:class: tip
[Lesson 5: Bounding Box Generation](bounding_boxes.md) — Master techniques for generating axis-aligned and oriented bounding boxes around 3D objects. Learn convex hull algorithms, minimum enclosing rectangles, and object orientation estimation.
```

---

*Outstanding work! You've now built a complete object detection pipeline that can identify and cluster individual objects in lidar data. This is a core capability for autonomous vehicle perception systems!* 🎯🚗✨