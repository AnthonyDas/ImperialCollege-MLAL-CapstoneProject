import math


def flatten(xss):
    """
    Flattens a nested list (list of lists) one level into a single list.

    Args:
        xss (list of list): The nested list to be flattened.

    Returns:
        list: A single-dimensional list containing all elements from the sublists.
    """
    return [x for xs in xss for x in xs]


def distances_between_points(x):
    """
    Calculates pairwise Euclidean distances between all points in a set.

    To avoid redundant calculations, this only computes distances for indices i < j.
    The distances are rounded to 6 decimal places to align with BBO submission 
    granularity.

    Args:
        x (list of list or np.ndarray): A collection of points specified by their coordinates.

    Returns:
        tuple: A tuple containing:
            - dists (list of list): A triangular matrix-like structure of distances.
            - min_dist (float): The minimum pairwise distance found.
            - max_dist (float): The maximum pairwise distance found.
    """
    dists = []
    
    for i in range(len(x)):
        row = []
        for j in range(i+1, len(x)): # i < j
            dist = math.dist(x[i], x[j]) # Euclidean distance
            row.append(round(dist, ndigits=6)) # Round to 6 d.p. since that's how granular BBO submissions are
        dists.append(row)

    flattened = flatten(dists)
     
    return dists, min(flattened), max(flattened)


def distance_to_nearest_point(points, x):
    """
    Finds the Euclidean distance between point x and its closest neighbour in 'points'.

    This is primarily used to check how "new" a suggested point is relative to all the already evaluated points.

    Args:
        points (list of list or np.ndarray): List of existing points to check against.
        x (list or np.ndarray): The coordinates of the point being checked.

    Returns:
        float: The distance to the nearest neighbour in `points`.
    """
    min_dist = None

    for pt in points:
        dist = math.dist(pt, x) # Euclidean distance

        if min_dist == None or dist < min_dist:
            min_dist = dist

    return min_dist