"""
Phase 3: 画像前処理モジュール (image_preprocessing.py)

OpenCVを用いた画像前処理関数群。
二値化、ノイズ除去、コントラスト適応均等化 (CLAHE)、傾き補正 (Deskew)、シャープニングなどを提供します。
"""

import cv2
import numpy as np

def to_grayscale(image: np.ndarray) -> np.ndarray:
    """RGB/BGR画像をグレースケールに変換"""
    if len(image.shape) == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def denoise_gaussian(gray_img: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """ガウシアンフィルタによるノイズ除去"""
    return cv2.GaussianBlur(gray_img, (kernel_size, kernel_size), 0)


def denoise_bilateral(gray_img: np.ndarray, d: int = 9, sigma_color: int = 75, sigma_space: int = 75) -> np.ndarray:
    """バイラテラルフィルタによるエッジ保存型ノイズ除去"""
    return cv2.bilateralFilter(gray_img, d, sigma_color, sigma_space)


def binarize_otsu(gray_img: np.ndarray) -> np.ndarray:
    """大津の二値化 (Otsu's Binarization)"""
    # 事前平滑化を行ってから二値化
    blur = cv2.GaussianBlur(gray_img, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh


def binarize_adaptive(gray_img: np.ndarray, block_size: int = 11, c: int = 2) -> np.ndarray:
    """適応的二値化 (Adaptive Thresholding)"""
    return cv2.adaptiveThreshold(
        gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, c
    )


def apply_clahe(gray_img: np.ndarray, clip_limit: float = 2.0, tile_grid_size: tuple = (8, 8)) -> np.ndarray:
    """CLAHE (Contrast Limited Adaptive Histogram Equalization) コントラスト強調"""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(gray_img)


def sharpen_image(gray_img: np.ndarray) -> np.ndarray:
    """シャープニング (輪郭・エッジ強調)"""
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]], dtype=np.float32)
    return cv2.filter2D(gray_img, -1, kernel)


def deskew_image(image: np.ndarray) -> tuple[np.ndarray, float]:
    """
    画像の傾き補正 (Deskew)。
    輪郭の最小外接矩形の傾き角度を計算し、アフィン変換で補正します。
    
    Returns:
        (補正後画像, 傾き角度)
    """
    gray = to_grayscale(image)
    # 反転二値化 (背景を黒、文字を白にする)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # 文字領域のすべての白画素の座標を取得
    coords = np.column_stack(np.where(thresh > 0))
    if len(coords) == 0:
        return image, 0.0

    # 最小外接矩形の角度を取得
    angle = cv2.minAreaRect(coords)[-1]
    
    # minAreaRectの仕様調整: 角度は [-90, 0)
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    # アフィン変換行列の作成と回転
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    m = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        image, m, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )

    return rotated, angle
