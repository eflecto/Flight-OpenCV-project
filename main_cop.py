import cv2
import numpy as np
from itertools import combinations

def ccw(A, B, C):
    return (C[1]-A[1])*(B[0]-A[0]) > (B[1]-A[1])*(C[0]-A[0])

def segments_intersect(A, B, C, D):
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

def line_intersection(A, B, C, D):
    A = np.array(A, dtype=float)
    B = np.array(B, dtype=float)
    C = np.array(C, dtype=float)
    D = np.array(D, dtype=float)

    s = np.vstack([A, B, C, D])
    h = np.hstack((s, np.ones((4, 1))))
    l1 = np.cross(h[0], h[1])
    l2 = np.cross(h[2], h[3])
    x, y, z = np.cross(l1, l2)
    if z == 0:
        return None
    return (x / z, y / z)

def dedupe_points(points, eps=5):
    filtered = []
    for p in points:
        if all(np.hypot(p[0] - q[0], p[1] - q[1]) > eps for q in filtered):
            filtered.append(p)
    return filtered



img = cv2.imread("input.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

edges = cv2.Canny(gray, 50, 150, apertureSize=3)

lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=60,
                        minLineLength=40, maxLineGap=10)

segments = []
if lines is not None:
    for l in lines:
        x1, y1, x2, y2 = l[0]
        segments.append(((x1, y1), (x2, y2)))

intersections = []

for (A, B), (C, D) in combinations(segments, 2):
    if segments_intersect(A, B, C, D):
        p = line_intersection(A, B, C, D)
        if p is not None:
            intersections.append(p)

intersections = dedupe_points(intersections, eps=7)

print(len(intersections))

