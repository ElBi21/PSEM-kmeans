from collections import Counter

def count_elements_per_cluster(file_path):
    """
    Reads a file with cluster assignments and counts the number of elements in each cluster.
    
    Args:
        file_path (str): Path to the text file containing cluster assignments (one per line).
        
    Returns:
        dict: A dictionary with cluster IDs as keys and the count of points as values.
    """
    try:
        # Read the cluster assignments
        with open(file_path, 'r') as file:
            clusters = [int(line.strip()) for line in file]
        
        # Count the occurrences of each cluster
        cluster_counts = Counter(clusters)
        
        # Print results
        print("Cluster counts:")
        for cluster_id, count in sorted(cluster_counts.items()):
            print(f"Cluster {cluster_id}: {count} points")
        
        return cluster_counts
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    except ValueError:
        print(f"Error: File '{file_path}' contains non-integer values.")
    except Exception as e:
        print(f"Unexpected error: {e}")

# Replace 'output.txt' with the path to your file containing cluster assignments
file_path = '/Users/davide/Desktop/PSEM-kmeans/out100D2.txt '
count_elements_per_cluster(file_path)