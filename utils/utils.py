import numpy as np

def check_overlap(circle1, circle2, epsilon=1e-6):
    """
    Check if two circles overlap.
    """
    x1, y1, r1 = circle1
    x2, y2, r2 = circle2
    distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    return distance + epsilon <= r1 + r2

def remove_overlapping_circles(circles, epsilon=1e-6):
    """
    Remove circles that overlap with each other.
    """
    # Create a boolean mask to mark non-overlapping circles
    mask = np.ones(len(circles), dtype=bool)

    # Iterate over each circle
    for i in range(len(circles)):
        if mask[i]:
            for j in range(i + 1, len(circles)):
                if check_overlap(circles[i], circles[j], epsilon=epsilon):
                    # Mark overlapping circle for removal
                    mask[j] = False

    # Return non-overlapping circles
    return circles[mask]