"""
Phase 3: 前処理 & 精度評価 A/Bテスト実験ランナー (run_phase3_experiments.py)

各種前処理手法（二値化、ノイズ除去、CLAHE、傾き補正）を画像に適用し、
EasyOCRでのテキスト認識結果を CER / WER の観点から定量的に比較評価します。
"""

import os
import cv2
import numpy as np
import easyocr
import json
from evaluate_cer import calculate_cer, calculate_wer
from image_preprocessing import (
    to_grayscale, denoise_gaussian, binarize_otsu, binarize_adaptive,
    apply_clahe, sharpen_image, deskew_image
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_IMAGE_PATH = os.path.join(BASE_DIR, "../01_easy_ocr/images/sample.png")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Ground Truth (正解テキスト)
GROUND_TRUTH_LINES = [
    "AI OCR Learning Journey",
    "日本語の認識テストです。",
    "EasyOCR on M4 Mac GPU/CPU",
    "Date: 2026-07-18 12:34:56"
]
GROUND_TRUTH_TEXT = "\n".join(GROUND_TRUTH_LINES)


def generate_rotated_image(input_path: str, angle: float = 8.0) -> str:
    """実験用に意図的に傾斜(回転ノイズ)を与えた画像を生成"""
    img = cv2.imread(input_path)
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    m = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, m, (w, h), borderValue=(255, 255, 255))
    rotated_path = os.path.join(OUTPUT_DIR, "sample_rotated.png")
    cv2.imwrite(rotated_path, rotated)
    return rotated_path


def run_single_experiment(reader: easyocr.Reader, img_input, exp_name: str, exp_no: str) -> dict:
    """単一の前処理実験を実行して結果を返す"""
    # 認識処理
    # EasyOCR は np.ndarray (BGR/Gray) も受け入れ可能
    results = reader.readtext(img_input)
    
    # 認識テキストの集計 (Y座標順にソート)
    results_sorted = sorted(results, key=lambda r: r[0][0][1])
    pred_lines = [r[1] for r in results_sorted]
    pred_text = "\n".join(pred_lines)
    
    # CER / WER 計測 (改行やスペースを含めた全体の文字列比較)
    cer_res = calculate_cer(GROUND_TRUTH_TEXT, pred_text)
    wer_res = calculate_wer(GROUND_TRUTH_TEXT, pred_text)
    
    # 行ごとの比較詳細
    line_details = []
    for i, gt_line in enumerate(GROUND_TRUTH_LINES):
        hyp_line = pred_lines[i] if i < len(pred_lines) else ""
        l_cer = calculate_cer(gt_line, hyp_line)
        line_details.append({
            "gt": gt_line,
            "hyp": hyp_line,
            "cer": l_cer["cer"],
            "distance": l_cer["distance"]
        })
        
    return {
        "exp_no": exp_no,
        "exp_name": exp_name,
        "pred_text": pred_text,
        "cer": cer_res["cer"],
        "wer": wer_res["wer"],
        "distance": cer_res["distance"],
        "line_details": line_details
    }


def main():
    print("=== Phase 3: OCR精度評価 & 画像前処理 A/Bテスト実験 ===")
    
    if not os.path.exists(INPUT_IMAGE_PATH):
        print(f"エラー: {INPUT_IMAGE_PATH} が存在しません。先に Phase 1 スクリプトを実行してください。")
        return

    # 元画像の読み込み
    orig_img = cv2.imread(INPUT_IMAGE_PATH)
    rotated_img_path = generate_rotated_image(INPUT_IMAGE_PATH, angle=8.0)
    rotated_img = cv2.imread(rotated_img_path)

    # EasyOCR Reader 初期化
    print("EasyOCR Reader を初期化中 (GPU/CPU)...")
    reader = easyocr.Reader(['ja', 'en'], gpu=True)

    experiments_results = []

    # -------------------------------------------------------------
    # 実験 00: 前処理なし（ベースライン）
    # -------------------------------------------------------------
    print("\n--- 実験 00: 前処理なし (Baseline) ---")
    res_00 = run_single_experiment(reader, orig_img, "補正なし (Baseline / デフォルト)", "00")
    experiments_results.append(res_00)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "exp_00_baseline.png"), orig_img)

    # -------------------------------------------------------------
    # 実験 01: グレースケール ＋ 大津の二値化 (Otsu Binarization)
    # -------------------------------------------------------------
    print("\n--- 実験 01: 大津の二値化 (Otsu Binarization) ---")
    gray_img = to_grayscale(orig_img)
    otsu_img = binarize_otsu(gray_img)
    res_01 = run_single_experiment(reader, otsu_img, "グレースケール ＋ 大津の二値化", "01")
    experiments_results.append(res_01)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "exp_01_otsu.png"), otsu_img)

    # -------------------------------------------------------------
    # 実験 02: ガウシアンノイズ除去 ＋ 二値化
    # -------------------------------------------------------------
    print("\n--- 実験 02: ノイズ除去 ＋ 二値化 ---")
    denoised_gray = denoise_gaussian(gray_img, kernel_size=3)
    denoised_otsu = binarize_otsu(denoised_gray)
    res_02 = run_single_experiment(reader, denoised_otsu, "ガウシアンノイズ除去 ＋ 二値化", "02")
    experiments_results.append(res_02)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "exp_02_denoise_otsu.png"), denoised_otsu)

    # -------------------------------------------------------------
    # 実験 03: CLAHE（適応的ヒストグラム均等化）＋ シャープニング
    # -------------------------------------------------------------
    print("\n--- 実験 03: CLAHE コントラスト強調 ＋ シャープニング ---")
    clahe_img = apply_clahe(gray_img, clip_limit=2.0)
    sharp_img = sharpen_image(clahe_img)
    res_03 = run_single_experiment(reader, sharp_img, "CLAHE ＋ シャープニング", "03")
    experiments_results.append(res_03)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "exp_03_clahe_sharp.png"), sharp_img)

    # -------------------------------------------------------------
    # 実験 04-A: 傾斜画像（8度回転）の前処理なし
    # -------------------------------------------------------------
    print("\n--- 実験 04-A: 傾斜画像 (回転8度) 前処理なし ---")
    res_04a = run_single_experiment(reader, rotated_img, "傾斜画像 (回転8度) 補正なし", "04-A")
    experiments_results.append(res_04a)

    # -------------------------------------------------------------
    # 実験 04-B: 傾斜画像に対する Deskew (傾き自動補正) ＋ 二値化
    # -------------------------------------------------------------
    print("\n--- 実験 04-B: 傾斜画像に対する Deskew 傾き補正 ＋ 二値化 ---")
    deskewed_img, estimated_angle = deskew_image(rotated_img)
    deskewed_gray = to_grayscale(deskewed_img)
    deskewed_otsu = binarize_otsu(deskewed_gray)
    print(f" -> 推定傾き角度: {estimated_angle:.2f}度")
    res_04b = run_single_experiment(reader, deskewed_otsu, f"Deskew 傾き補正({estimated_angle:.1f}°補正) ＋ 二値化", "04-B")
    experiments_results.append(res_04b)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "exp_04b_deskewed_otsu.png"), deskewed_otsu)

    # -------------------------------------------------------------
    # 実験結果のサマリー表示とJSON出力
    # -------------------------------------------------------------
    print("\n=============================================================")
    print("           Phase 3: 実験結果サマリー (A/B テスト)")
    print("=============================================================")
    print(f"{'No.':<6} | {'施策内容':<32} | {'CER (%)':<8} | {'WER (%)':<8} | {'編集距離':<6}")
    print("-" * 75)
    
    for r in experiments_results:
        print(f"{r['exp_no']:<6} | {r['exp_name']:<32} | {r['cer']:6.2f}% | {r['wer']:6.2f}% | {r['distance']:<6}")
        
    print("=============================================================")

    # JSON形式で結果を保存
    report_json_path = os.path.join(OUTPUT_DIR, "phase3_experiment_summary.json")
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(experiments_results, f, ensure_ascii=False, indent=2)
        
    print(f"\n実験サマリーJSONを保存しました: {report_json_path}")


if __name__ == "__main__":
    main()
