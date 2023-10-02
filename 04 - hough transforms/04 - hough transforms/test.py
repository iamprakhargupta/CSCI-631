import numpy as np

# def euclidean_distance(matrix1, matrix2):
#     # Calculate the element-wise squared differences
#     print(matrix2.shape)
#     squared_diff = (matrix1 - matrix2) ** 2
    
#     # Sum along the appropriate axis (axis=1 for m x 2 matrices)
#     sum_squared_diff = np.sum(squared_diff, axis=1)
    
#     # Take the square root to get the Euclidean distance
#     distance = np.sqrt(sum_squared_diff)
    
#     return distance

# Example usage:
# matrix1 = np.array([[0, 0],
#                    [1, 1]])

# matrix2 = np.array([3, 3])

# distances = euclidean_distance(matrix1, matrix2)
# print(distances)

vector1 = np.array([1, 2, 3])  # Replace with your first vector
vector2 = np.array([4, 5, 6])  # Replace with your second vector

# Merge the vectors into an m x 2 matrix
merged_matrix = np.column_stack((vector1, vector2))

# print(merged_matrix)


def euclidean_distance(matrix1, matrix2):
    # Calculate the element-wise squared differences
    squared_diff = (matrix1 - matrix2) ** 2
    
    # Sum along the appropriate axis (axis=1 for m x 2 matrices)
    sum_squared_diff = np.sum(squared_diff, axis=1)
    
    # Take the square root to get the Euclidean distance
    distance = np.sqrt(sum_squared_diff)
    
    return distance

def distance_to_circles(centers_xy, xy, r):
    distance=euclidean_distance(centers_xy,xy)
    return np.abs(r-distance)


centers_xy = np.array([[-4.0, 12.0], [-8.0, -10.0], [-3.0, 9.0]])
xy = np.array([-5.0, 8.0])
r = 5.0
x=distance_to_circles(centers_xy, xy, r)
print(x)