

# Flight Trajectory Clustering

This project provides a set of tools for loading flight trajectory data, calculating distances between flight paths, and
performing clustering using DBSCAN. The final output includes various clustering metrics such as silhouette score,
Davies-Bouldin index, and more.

## Features

- **Load Dataset**: Load flight trajectory data from a CSV file.
- **Preprocess Data**: Align and normalize flight paths to ensure consistency for distance calculation.
- **Calculate Distance Matrix**: Compute a pairwise distance matrix between flight paths.
- **DBSCAN Clustering**: Perform clustering on flight trajectories using the DBSCAN algorithm.
- **Clustering Metrics**: Calculate various internal and external clustering evaluation metrics, including:
    - Silhouette Coefficient
    - Davies-Bouldin Index
    - Adjusted Rand Index (ARI)
    - Normalized Mutual Information (NMI)
    - Final weighted score combining all metrics

## Requirements

- Python 3.6+
- Required libraries:
    - `numpy`
    - `pandas`
    - `scikit-learn`

To install the necessary dependencies, run:

```bash
pip install -r requirements.txt
```

## Usage

### 1. Loading the Dataset

Make sure your dataset is in a CSV format with columns `x`, `y`, `z`, `label`, and `flight_id`. The `x`, `y`, `z` columns represent the flight trajectory coordinates, `label` contains the true labels for evaluation, and `flight_id` is the unique identifier for each flight.

### 2. Running the Clustering

Use the `cluster()` function to perform the clustering process. Below is an example script that shows how to load the dataset and run clustering:

```python
import pandas as pd
import clustering

# Step 1: Load the dataset
df = pd.read_csv('data/valid_data.csv')

# Step 2: Perform clustering and evaluate metrics
clustering.cluster(df, 'data/distance_df.csv')
```

### 3. Output

The clustering process will output:

- Progress logs for each major step.
- The final clustering metrics including the Silhouette Coefficient, Davies-Bouldin Index, ARI, NMI, and a final
  weighted score.
- The cluster labels for each flight trajectory.

Example output:

```
Begin clustering process.
Feature and label extraction complete.
Reading precomputed distance matrix...
Distance matrix loaded.
DBSCAN clustering finished.
Labels unified across flight groups.
Calculating clustering metrics...
Silhouette Coefficient: 0.6523
Davies-Bouldin Index: 0.3456
Adjusted Rand Index: 0.7802
Normalized Mutual Information: 0.8234
Final Weighted Score: 0.7765
Clustering process completed.
Cluster labels for each sample: [0, 1, 1, 0, ...]
```

### 4. Customizing Parameters

You can modify the DBSCAN parameters directly in the `perform_dbscan_clustering()` function, such as:

- `eps`: The maximum distance between two samples for one to be considered as in the neighborhood of the other.
- `min_samples`: The number of samples in a neighborhood for a point to be considered as a core point.

For example:

```python
pre_labels = perform_dbscan_clustering(distance_matrix, eps=0.03, min_samples=10)
```

## Project Structure

```
flight_clustering/
│
├── flight_clustering.py         # Main logic for clustering and metric calculation
├── requirements.txt             # Required dependencies
└── README.md                    # Project documentation
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

- **slience_me** - *Initial work*


### How to Use:

1. **Install dependencies**:
   - Run `pip install -r requirements.txt` to install the required Python libraries.

2. **Prepare your dataset**:
   - Ensure your data is in CSV format and contains the necessary columns (`x`, `y`, `z`, `label`, `flight_id`).

3. **Run the clustering**:
   - Use the provided example to load your dataset and execute the clustering and evaluation process.

Let me know if you need any adjustments or additional sections for the `README`!