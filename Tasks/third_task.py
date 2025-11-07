import cv2
import numpy as np
import math

INPUT_IMAGE = "Examples/vec1.png"
MIN_CONTOUR_AREA = 150
NEIGHBOR_RADIUS = 10


def skeletonize(bin_img):
    img = bin_img.copy()
    img[img > 0] = 1
    skel = np.zeros(img.shape, np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
    while True:
        eroded = cv2.erode(img, kernel)
        temp = cv2.dilate(eroded, kernel)
        temp = img - temp
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()
        if cv2.countNonZero(img) == 0:
            break
    return (skel * 255).astype(np.uint8)

def find_skeleton_endpoints(skel):
    kernel = np.array([[1,1,1],[1,10,1],[1,1,1]], dtype=np.uint8)
    conv = cv2.filter2D((skel>0).astype(np.uint8), -1, kernel)
    ys, xs = np.where(skel > 0)
    endpoints = []
    for x, y in zip(xs, ys):
        if conv[y, x] - 10 == 1:
            endpoints.append((x, y))
    return endpoints


def angular_span_of_neighbors(contour, point, radius=NEIGHBOR_RADIUS):
    px, py = point
    pts = contour.reshape(-1,2)

    d2 = ((pts[:,0]-px)**2 + (pts[:,1]-py)**2)
    mask = d2 <= radius*radius
    neigh = pts[mask]
    if len(neigh) < 6:
        k = min(12, len(pts))
        idx = np.argsort(d2)[:k]
        neigh = pts[idx]
    if len(neigh) < 3:
        return 2*math.pi

    angles = np.arctan2(neigh[:,1]-py, neigh[:,0]-px)
    angles = np.sort(angles)
    gaps = np.diff(np.concatenate([angles, angles[:1] + 2*math.pi]))
    max_gap = np.max(gaps)
    span = 2*math.pi - max_gap
    return span

def farthest_point_on_contour(contour, point):
    px, py = point
    pts = contour.reshape(-1,2)
    d2 = ((pts[:,0]-px)**2 + (pts[:,1]-py)**2)
    idx = np.argmax(d2)
    return (int(pts[idx,0]), int(pts[idx,1]))

img = cv2.imread(INPUT_IMAGE)
if img is None:
    raise FileNotFoundError("Не найден входной файл: " + INPUT_IMAGE)

orig = img.copy()
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, bw = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel, iterations=1)
bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel, iterations=1)

contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

vectors = []
sum_x = 0
sum_y = 0

for cnt in contours:
    if cv2.contourArea(cnt) < MIN_CONTOUR_AREA:
        continue

    x, y, w, h = cv2.boundingRect(cnt)
    roi_bw = bw[y:y+h, x:x+w]
    skel = skeletonize(roi_bw)
    endpoints = find_skeleton_endpoints(skel)
    endpoints_global = [(ex + x, ey + y) for (ex, ey) in endpoints]

    if len(endpoints_global) < 2:
        pts = cnt.reshape(-1,2).astype(np.float32)
        mean = pts.mean(axis=0)
        cov = np.cov((pts - mean).T)
        eigvals, eigvecs = np.linalg.eig(cov)
        principal = eigvecs[:, np.argmax(eigvals)]
        proj = pts.dot(principal)
        idx_min = np.argmin(proj); idx_max = np.argmax(proj)
        e1 = tuple(pts[idx_min].astype(int))
        e2 = tuple(pts[idx_max].astype(int))
        endpoints_global = [e1, e2]

    best_pair = None
    maxd = -1
    for i in range(len(endpoints_global)):
        for j in range(i+1, len(endpoints_global)):
            (x1,y1),(x2,y2)=endpoints_global[i],endpoints_global[j]
            d = (x1-x2)**2 + (y1-y2)**2
            if d > maxd:
                maxd = d
                best_pair = (endpoints_global[i], endpoints_global[j])
    if best_pair is None:
        continue
    p1, p2 = best_pair

    span1 = angular_span_of_neighbors(cnt, p1)
    span2 = angular_span_of_neighbors(cnt, p2)

    if span1 < span2:
        skeleton_tail = p2
        skeleton_head_candidate = p1
    else:
        skeleton_tail = p1
        skeleton_head_candidate = p2

    head = farthest_point_on_contour(cnt, skeleton_tail)
    tail = (int(skeleton_tail[0]), int(skeleton_tail[1]))

    dx = head[0] - tail[0]
    dy = head[1] - tail[1]
    vectors.append((dx, dy))
    sum_x += dx
    sum_y += dy

    cv2.arrowedLine(orig, tail, head, (0,0,255), 2, tipLength=0.2)
    cv2.circle(orig, head, 3, (0,255,0), -1)
    cv2.circle(orig, tail, 3, (255,0,0), -1)

print("Найденные векторы (dx, dy):")
for i,v in enumerate(vectors,1):
    print(f"{i}: {v}")
print(f"\nСуммарный вектор: X = {sum_x}, Y = {sum_y}")

cv2.imshow("Detected", orig)
cv2.imshow("Binary inv", bw)
cv2.waitKey(0)
cv2.destroyAllWindows()
