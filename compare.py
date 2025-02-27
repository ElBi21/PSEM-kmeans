import os

def compare_files(file1, file2):
    """
    Compare two text files and count the number of mismatched lines (points assigned to different clusters).

    Args:
        file1 (str): Path to the first file.
        file2 (str): Path to the second file.

    Returns:
        tuple: (int, int, int) - Total points in the files, mismatched points, and total points in the larger file.
    """
    mismatch_count = 0
    total_points_file1 = 0
    total_points_file2 = 0

    try:
        with open(file1, 'r') as f1, open(file2, 'r') as f2:
            for line1, line2 in zip(f1, f2):
                total_points_file1 += 1
                total_points_file2 += 1

                # Strip whitespace and compare lines
                if line1.strip() != line2.strip():
                    mismatch_count += 1

            # Count remaining lines in file1
            for _ in f1:
                total_points_file1 += 1
                mismatch_count += 1

            # Count remaining lines in file2
            for _ in f2:
                total_points_file2 += 1
                mismatch_count += 1

        total_points = max(total_points_file1, total_points_file2)
        return total_points, mismatch_count, total_points_file1, total_points_file2

    except FileNotFoundError as e:
        print(f"Error: {e}")
        return -1, -1, -1, -1

# Example usage:
file1 = "out.out"
file2 = "output_clusters_seq.txt"

result = compare_files(file1, file2)
if result[0] == -1:
    print("Error: One or both files were not found.")
elif result[1] == 0:
    print(f"The files are identical. Total points: {result[2]}.")
else:
    print(f"The files have {result[1]} points with mismatched cluster assignments out of {result[0]} total points.\n"
          f"File1 has {result[2]} points and File2 has {result[3]} points.")
