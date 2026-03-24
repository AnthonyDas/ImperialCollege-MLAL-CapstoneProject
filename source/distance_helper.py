import math

# TODO
def flatten(xss):
    return [x for xs in xss for x in xs]

# TODO
def distances_between_points(x):
    dists = []
    
    for i in range(len(x)):
        row = []
        for j in range(i+1, len(x)): # i < j
            dist = math.dist(x[i], x[j]) # Euclidean distance
            row.append(round(dist, ndigits=6)) # Round to 6 d.p. since that's how granular BBO submissions are
        dists.append(row)

    flattened = flatten(dists)
     
    return dists, min(flattened), max(flattened)

# TODO
def distance_to_nearest_point(points, x):
    min_dist = None

    for pt in points:
        dist = math.dist(pt, x) # Euclidean distance

        if min_dist == None or dist < min_dist:
            min_dist = dist

    return min_dist