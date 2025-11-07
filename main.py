import cv2
import numpy as np
from itertools import combinations
import math


def normalize_angle_theta(theta):
    """
    Нормализует угол theta в диапазоне [0, pi)
    """
    if theta < 0:
        theta += math.pi
    if theta >= math.pi:
        theta -= math.pi
    return theta


def merge_collinear_segments(segments, angle_thresh_deg=5.0, rho_thresh=20.0):
    """
    segments: list of (x1,y1,x2,y2)
    Возвращает список объединённых сегментов (по кластерам коллинеарности).
    Параметры:
      angle_thresh_deg - максимально допустимая разница углов (в градусах)
      rho_thresh - максимально допустимая разница rho (по нормали), px
    """
    # Преобразуем каждый сегмент в представление (theta_dir, rho, t_min, t_max),
    # где точка на линии: p(t) = d * t + n * rho, d - направление, n - нормаль
    seg_infos = []
    for (x1, y1, x2, y2) in segments:
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0 and dy == 0:
            continue
        theta_dir = math.atan2(dy, dx)  # направление линии
        theta_dir = normalize_angle_theta(theta_dir)  # [0, pi)
        # direction vector d
        dx_dir = math.cos(theta_dir)
        dy_dir = math.sin(theta_dir)
        # normal vector n (perpendicular)
        nx = -dy_dir
        ny = dx_dir
        # rho (distance along normal)
        rho = x1 * nx + y1 * ny
        # project both endpoints onto direction to get scalar coords t
        t1 = x1 * dx_dir + y1 * dy_dir
        t2 = x2 * dx_dir + y2 * dy_dir
        tmin = min(t1, t2)
        tmax = max(t1, t2)
        seg_infos.append({
            'orig': (x1, y1, x2, y2),
            'theta': theta_dir,
            'rho': rho,
            'tmin': tmin,
            'tmax': tmax,
            'd': (dx_dir, dy_dir),
            'n': (nx, ny)
        })

    used = [False] * len(seg_infos)
    merged_segments = []

    angle_thresh = math.radians(angle_thresh_deg)

    for i, si in enumerate(seg_infos):
        if used[i]:
            continue
        cluster = [i]
        used[i] = True
        for j in range(i + 1, len(seg_infos)):
            if used[j]:
                continue
            sj = seg_infos[j]
            # angle difference (shortest)
            da = abs(si['theta'] - sj['theta'])
            da = min(da, math.pi - da)
            if da <= angle_thresh and abs(si['rho'] - sj['rho']) <= rho_thresh:
                # добавляем в кластер
                cluster.append(j)
                used[j] = True

        # Объединяем все t-intervals внутри кластера
        tmin = min(seg_infos[k]['tmin'] for k in cluster)
        tmax = max(seg_infos[k]['tmax'] for k in cluster)
        # усредним theta и rho для стабильности
        theta_avg = sum(seg_infos[k]['theta'] for k in cluster) / len(cluster)
        theta_avg = normalize_angle_theta(theta_avg)
        dx_dir = math.cos(theta_avg)
        dy_dir = math.sin(theta_avg)
        nx = -dy_dir
        ny = dx_dir
        rho_avg = sum(seg_infos[k]['rho'] for k in cluster) / len(cluster)

        # Восстанавливаем 2D точки концов:
        x_start = dx_dir * tmin + nx * rho_avg
        y_start = dy_dir * tmin + ny * rho_avg
        x_end = dx_dir * tmax + nx * rho_avg
        y_end = dy_dir * tmax + ny * rho_avg

        merged_segments.append((x_start, y_start, x_end, y_end))

    return merged_segments


def intersection_point(seg1, seg2, eps=1e-9):
    """
    Возвращает точку пересечения (x,y) если отрезки пересекаются (включая в концевых точках),
    иначе None.
    seg = (x1,y1,x2,y2) - могут быть float
    """
    x1, y1, x2, y2 = seg1
    x3, y3, x4, y4 = seg2

    # Решение векторно: p + t r  and q + u s
    r = (x2 - x1, y2 - y1)
    s = (x4 - x3, y4 - y3)

    r_cross_s = r[0] * s[1] - r[1] * s[0]
    q_p = (x3 - x1, y3 - y1)
    q_p_cross_r = q_p[0] * r[1] - q_p[1] * r[0]

    if abs(r_cross_s) < eps:
        # параллельны или коллинеарны -> не считаем пересечением в этой функции
        return None

    t = (q_p[0] * s[1] - q_p[1] * s[0]) / r_cross_s
    u = (q_p[0] * r[1] - q_p[1] * r[0]) / r_cross_s

    # допускаем небольшие погрешности, поэтому используем -tol..1+tol
    tol = 1e-7
    if -tol <= t <= 1 + tol and -tol <= u <= 1 + tol:
        ix = x1 + t * r[0]
        iy = y1 + t * r[1]
        return (ix, iy)
    return None


def cluster_points(points, cluster_dist=10.0):
    """
    Кластеризация точек по расстоянию, возвращает центр каждого кластера.
    Простейщий greedy: проходим по точкам, если ближе чем cluster_dist к существующему кластеру - добавляем туда.
    """
    clusters = []
    for p in points:
        placed = False
        for c in clusters:
            cx, cy, count = c
            if (p[0] - cx) ** 2 + (p[1] - cy) ** 2 <= cluster_dist ** 2:
                # обновим центр (скользящее среднее)
                new_count = count + 1
                c[0] = (cx * count + p[0]) / new_count
                c[1] = (cy * count + p[1]) / new_count
                c[2] = new_count
                placed = True
                break
        if not placed:
            clusters.append([p[0], p[1], 1])
    # вернуть центры
    return [(c[0], c[1]) for c in clusters]


def count_intersections(image_path,
                        canny_thresh1=50, canny_thresh2=150,
                        hough_threshold=40, minLineLength=0, maxLineGap=10,
                        angle_merge_deg=5.0, rho_merge_px=20.0,
                        cluster_dist=12.0,
                        visualize=True,
                        out_vis_path="intersections_vis.png",
                        apply_clahe=True,
                        apply_blur=True,
                        use_auto_scaling=True,
                        auto_base_diag=1000.0,
                        near_parallel_deg=2.0,
                        return_raw_points=False):
    # Загружаем
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Предобработка контраста и шума
    if apply_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
    if apply_blur:
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # Улучшение контраста/порог при необходимости (необязательно)
    # thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    #                               cv2.THRESH_BINARY,11,2)

    edges = cv2.Canny(gray, canny_thresh1, canny_thresh2, apertureSize=3)

    # Автомасштабирование параметров относительно диагонали изображения
    h, w = gray.shape[:2]
    if use_auto_scaling and auto_base_diag > 0:
        diag = float((h ** 2 + w ** 2) ** 0.5)
        scale = diag / auto_base_diag
        # Масштабируем дистанции
        scaled_cluster_dist = float(cluster_dist) * scale
        scaled_rho_merge = float(rho_merge_px) * scale
        # Если minLineLength не задан (0), подберём от размера
        scaled_min_line_length = int(max(minLineLength, 0.05 * max(h, w)))
    else:
        scaled_cluster_dist = cluster_dist
        scaled_rho_merge = rho_merge_px
        scaled_min_line_length = minLineLength

    lines = cv2.HoughLinesP(edges,
                            rho=1,
                            theta=np.pi / 180,
                            threshold=hough_threshold,
                            minLineLength=scaled_min_line_length,
                            maxLineGap=maxLineGap)

    if lines is None:
        if visualize:
            cv2.imwrite(out_vis_path, img)
        return 0, [], []

    segments = [tuple(map(float, line[0])) for line in lines]

    # Мерджим коллинеарные сегменты
    merged = merge_collinear_segments(segments,
                                      angle_thresh_deg=angle_merge_deg,
                                      rho_thresh=scaled_rho_merge)

    # Находим пересечения пар (точки)
    pts = []
    # Игнорируем пары почти параллельных сегментов, чтобы не собирать ложные пересечения
    near_parallel_rad = math.radians(near_parallel_deg)
    for a, b in combinations(merged, 2):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ang_a = normalize_angle_theta(math.atan2(ay2 - ay1, ax2 - ax1))
        ang_b = normalize_angle_theta(math.atan2(by2 - by1, bx2 - bx1))
        d_ang = abs(ang_a - ang_b)
        d_ang = min(d_ang, math.pi - d_ang)
        if d_ang < near_parallel_rad:
            continue
        p = intersection_point(a, b)
        if p is not None:
            pts.append(p)

    # Кластеризуем близкие точки пересечения
    clustered = cluster_points(pts, cluster_dist=scaled_cluster_dist)

    if visualize:
        vis = img.copy()
        # рисуем объединённые отрезки - толстая линия
        for (x1, y1, x2, y2) in merged:
            cv2.line(vis, (int(round(x1)), int(round(y1))),
                     (int(round(x2)), int(round(y2))), (200, 200, 200), 2)
        # рисуем исходные Hough-отрезки (полупрозрично) - опционально
        # for (x1,y1,x2,y2) in segments:
        #     cv2.line(vis, (int(x1),int(y1)), (int(x2),int(y2)), (80,80,80), 1)

        # рисуем точки пересечения
        for (x, y) in clustered:
            cv2.circle(vis, (int(round(x)), int(round(y))), 6, (0, 0, 255), -1)
        # сохранить
        cv2.imwrite(out_vis_path, vis)

    if return_raw_points:
        return len(clustered), merged, clustered, pts
    return len(clustered), merged, clustered


if __name__ == "__main__":
    img_path = "Examples/example.png"  # замените на свой путь
    cnt, merged_segments, intersections = count_intersections(img_path,
                                                               canny_thresh1=50,
                                                               canny_thresh2=150,
                                                               hough_threshold=40,
                                                               minLineLength=30,
                                                               maxLineGap=10,
                                                               angle_merge_deg=5.0,
                                                               rho_merge_px=20.0,
                                                               cluster_dist=12.0,
                                                               visualize=True,
                                                               out_vis_path="intersections_vis.png",
                                                               apply_clahe=True,
                                                               apply_blur=True,
                                                               use_auto_scaling=True,
                                                               auto_base_diag=1000.0,
                                                               near_parallel_deg=2.0)
    print("Найдено пересечений (после слияния и кластеризации):", cnt)
    print("Визуализация сохранена в intersections_vis.png")
