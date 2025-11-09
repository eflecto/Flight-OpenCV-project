import cv2
import numpy as np

# 1. Загрузка и поиск исходных отрезков (параметры как в main.py)
img = cv2.imread('input.png', cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError("Файл 'input.png' не найден")

_, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

lines = cv2.HoughLinesP(
    image=binary,
    rho=1,
    theta=np.pi / 360,
    threshold=80,
    minLineLength=30,
    maxLineGap=15
)

segments = []
if lines is not None:
    for [[x1, y1, x2, y2]] in lines:
        length = np.hypot(x2 - x1, y2 - y1)
        angle = np.arctan2(y2 - y1, x2 - x1)
        if angle < 0:
            angle += np.pi
        segments.append([x1, y1, x2, y2, length, angle])
segments = np.array(segments) if len(segments) > 0 else np.empty((0, 6))

# 2. ОБЪЕДИНЕНИЕ БЕЗ УДЛИНЕНИЯ (параметры как в main.py)
MAX_ANGLE_DIFF = np.radians(2)
MAX_GAP = 30
MIN_LENGTH = 30

merged_lines = []
used = np.zeros(len(segments), dtype=bool)

for i in range(len(segments)):
    if used[i]:
        continue

    group = [i]
    x1a, y1a, x2a, y2a, _, ang_a = segments[i]

    for j in range(i + 1, len(segments)):
        if used[j]:
            continue

        x1b, y1b, x2b, y2b, _, ang_b = segments[j]

        # 1. Углы
        if abs(ang_a - ang_b) > MAX_ANGLE_DIFF and abs(abs(ang_a - ang_b) - np.pi) > MAX_ANGLE_DIFF:
            continue

        # 2. Близость концов
        ends_a = [(x1a, y1a), (x2a, y2a)]
        ends_b = [(x1b, y1b), (x2b, y2b)]
        close = False
        for xa, ya in ends_a:
            for xb, yb in ends_b:
                if np.hypot(xa - xb, ya - yb) <= MAX_GAP:
                    close = True
                    break
            if close:
                break
        if not close:
            continue

        # 3. На одной прямой (надёжность)
        pts = []
        for idx in group + [j]:
            xa1, ya1, xa2, ya2, _, _ = segments[idx]
            pts.extend([[xa1, ya1], [xa2, ya2]])
        pts = np.array(pts, dtype=np.float32)

        if len(pts) >= 2:
            [vx, vy, x0, y0] = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01)
            distances = np.abs((pts[:, 0] - x0) * vy - (pts[:, 1] - y0) * vx)
            if np.max(distances) <= 15:
                group.append(j)

    for idx in group:
        used[idx] = True

    # точки группы
    pts = []
    for idx in group:
        x1g, y1g, x2g, y2g, _, _ = segments[idx]
        pts.extend([[x1g, y1g], [x2g, y2g]])
    pts = np.array(pts, dtype=np.float32)

    if len(pts) >= 2:
        avg_angle = np.mean([segments[idx][5] for idx in group])
        if abs(avg_angle - np.pi/2) < np.pi/4:
            min_idx = np.argmin(pts[:, 1])
            max_idx = np.argmax(pts[:, 1])
        else:
            min_idx = np.argmin(pts[:, 0])
            max_idx = np.argmax(pts[:, 0])

        x1_f, y1_f = int(pts[min_idx, 0]), int(pts[min_idx, 1])
        x2_f, y2_f = int(pts[max_idx, 0]), int(pts[max_idx, 1])
        length_f = np.hypot(x2_f - x1_f, y2_f - y1_f)

        if length_f >= MIN_LENGTH:
            merged_lines.append((x1_f, y1_f, x2_f, y2_f, len(group)))

# 3. Поиск пересечений
def seg_intersection(a, b, eps=1e-6, tol=0.03):
    x1, y1, x2, y2 = a
    x3, y3, x4, y4 = b
    s1x, s1y = x2 - x1, y2 - y1
    s2x, s2y = x4 - x3, y4 - y3
    # Отбрасываем почти параллельные пары
    n1 = (s1x ** 2 + s1y ** 2) ** 0.5
    n2 = (s2x ** 2 + s2y ** 2) ** 0.5
    if n1 > eps and n2 > eps:
        cosang = abs((s1x * s2x + s1y * s2y) / (n1 * n2))
        if cosang > 0.985:
            return None
    den = s1x * s2y - s1y * s2x
    if abs(den) < eps:
        return None
    t = ((x3 - x1) * s2y - (y3 - y1) * s2x) / den
    u = ((x3 - x1) * s1y - (y3 - y1) * s1x) / den
    if -tol <= t <= 1.0 + tol and -tol <= u <= 1.0 + tol:
        return (x1 + t * s1x, y1 + t * s1y)
    return None

def dist_point_segment(px, py, x1, y1, x2, y2):
    vx, vy = x2 - x1, y2 - y1
    l2 = vx * vx + vy * vy
    if l2 == 0:
        return (px - x1) ** 2 + (py - y1) ** 2, 0.0
    t = ((px - x1) * vx + (py - y1) * vy) / l2
    t = max(0.0, min(1.0, t))
    projx = x1 + t * vx
    projy = y1 + t * vy
    d2 = (px - projx) ** 2 + (py - projy) ** 2
    return d2, t

def endpoint_touch(a, b, eps, margin=0.10):
    x1, y1, x2, y2 = a
    x3, y3, x4, y4 = b
    # фильтр почти параллельных (не считаем T-касание, если линии почти сонаправлены)
    s1x, s1y = x2 - x1, y2 - y1
    s2x, s2y = x4 - x3, y4 - y3
    n1 = (s1x ** 2 + s1y ** 2) ** 0.5
    n2 = (s2x ** 2 + s2y ** 2) ** 0.5
    if n1 > 1e-6 and n2 > 1e-6:
        cosang = abs((s1x * s2x + s1y * s2y) / (n1 * n2))
        if cosang > 0.97:
            return None
    for (px, py) in [(x1, y1), (x2, y2)]:
        d2, t = dist_point_segment(px, py, x3, y3, x4, y4)
        if d2 <= eps * eps and margin <= t <= 1.0 - margin:
            return (px, py)
    for (px, py) in [(x3, y3), (x4, y4)]:
        d2, t = dist_point_segment(px, py, x1, y1, x2, y2)
        if d2 <= eps * eps and margin <= t <= 1.0 - margin:
            return (px, py)
    return None

points = []
for i in range(len(merged_lines)):
    a = merged_lines[i][:4]
    for j in range(i + 1, len(merged_lines)):
        b = merged_lines[j][:4]
        p = seg_intersection(a, b)
        if p is None:
            # eps адаптируем к размеру изображения
            # (чем больше изображение, тем чуть больше допуск)
            eps_touch = max(5.0, 0.0120 * min(img.shape[0], img.shape[1]))
            p = endpoint_touch(a, b, eps_touch)
        if p is not None:
            points.append(p)

# кластеризация (адаптивный радиус)
h, w = img.shape
CLUSTER_RADIUS = max(6.0, 0.02 * min(h, w))
r2 = CLUSTER_RADIUS * CLUSTER_RADIUS
parent = list(range(len(points)))
rank = [0] * len(points)
def find(a):
    while parent[a] != a:
        parent[a] = parent[parent[a]]
        a = parent[a]
    return a
def union(a, b):
    ra, rb = find(a), find(b)
    if ra == rb:
        return
    if rank[ra] < rank[rb]:
        parent[ra] = rb
    elif rank[ra] > rank[rb]:
        parent[rb] = ra
    else:
        parent[rb] = ra
        rank[ra] += 1
for i in range(len(points)):
    xi, yi = points[i]
    for j in range(i + 1, len(points)):
        xj, yj = points[j]
        dx, dy = xi - xj, yi - yj
        if dx * dx + dy * dy <= r2:
            union(i, j)
clusters = set(find(i) for i in range(len(points)))
count = len(clusters)

# вывод только числа
print(count)
