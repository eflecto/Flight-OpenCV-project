from __future__ import annotations

import math
import sys
from pathlib import Path

import cv2
import numpy as np


def _order_points(pts: np.ndarray) -> np.ndarray:
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = pts[np.argmin(s)]
    ordered[2] = pts[np.argmax(s)]
    ordered[1] = pts[np.argmin(diff)]
    ordered[3] = pts[np.argmax(diff)]
    return ordered


def _largest_quadrilateral(binary: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("Не удалось найти контуры таблицы")

    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4:
            return _order_points(approx.reshape(4, 2))

        hull = cv2.convexHull(cnt)
        peri_hull = cv2.arcLength(hull, True)
        approx_hull = cv2.approxPolyDP(hull, 0.02 * peri_hull, True)
        if len(approx_hull) == 4:
            return _order_points(approx_hull.reshape(4, 2))

        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        if len(box) == 4:
            return _order_points(box.astype(np.float32))

    raise ValueError("Не получилось аппроксимировать контур четырёхугольником")


def _perspective_warp(image: np.ndarray, quad: np.ndarray) -> np.ndarray:
    (tl, tr, br, bl) = quad

    width_top = np.linalg.norm(tr - tl)
    width_bottom = np.linalg.norm(br - bl)
    max_width = int(round(max(width_top, width_bottom)))

    height_right = np.linalg.norm(br - tr)
    height_left = np.linalg.norm(bl - tl)
    max_height = int(round(max(height_right, height_left)))

    max_width = max(max_width, 1)
    max_height = max(max_height, 1)

    dst = np.array(
        [
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1],
        ],
        dtype=np.float32,
    )

    M = cv2.getPerspectiveTransform(quad.astype(np.float32), dst)
    warped = cv2.warpPerspective(image, M, (max_width, max_height))
    return warped


def _count_line_runs(mask: np.ndarray) -> int:
    count = 0
    inside = False
    for val in mask:
        if val and not inside:
            count += 1
            inside = True
        elif not val and inside:
            inside = False
    return count


def _count_grid_lines(binary: np.ndarray, axis: int) -> int:
    if axis == 0:
        profile = binary.mean(axis=1)
    else:
        profile = binary.mean(axis=0)

    if profile.size == 0:
        raise ValueError("Пустой профиль яркости")

    max_val = float(profile.max())
    if max_val <= 1e-6:
        raise ValueError("Не найдено ярко выраженных линий сетки")

    threshold = 0.5 * max_val
    mask = profile > threshold
    return _count_line_runs(mask)


def count_table_cells(image_path: str) -> int:
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(image_path)

    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"Не удалось загрузить изображение: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary_inv = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    quad = _largest_quadrilateral(binary_inv)
    warped = _perspective_warp(binary_inv, quad)

    warped = cv2.medianBlur(warped, 3)

    _, warped = cv2.threshold(warped, 0, 255, cv2.THRESH_BINARY)

    horizontal_count = _count_grid_lines(warped, axis=0)
    vertical_count = _count_grid_lines(warped, axis=1)

    if horizontal_count < 2 or vertical_count < 2:
        raise ValueError("Обнаружено недостаточно линий для вычисления клеток")

    return (horizontal_count - 1) * (vertical_count - 1)


if __name__ == "__main__":
    count = count_table_cells("input.png")
    print(count)


