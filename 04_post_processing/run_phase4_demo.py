"""
Phase 4: エンドツーエンド (E2E) 後処理パイプライン統合デモ (run_phase4_demo.py)

OCR認識結果に対して社内マスタ名寄せ・文字補正を適用し、
LLM (Gemini API / Fallback) により構造化 JSON 形式に変換・出力します。
"""

import os
import sys
import json
import easyocr
import cv2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PHASE3_DIR = os.path.join(BASE_DIR, "../03_data_science")
sys.path.append(PHASE3_DIR)

from evaluate_cer import calculate_cer
from master_matching import MasterMatcher
from llm_structuring import LLMStructurer

INPUT_IMAGE_PATH = os.path.join(BASE_DIR, "../01_easy_ocr/images/sample.png")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

GROUND_TRUTH_TEXT = """AI OCR Learning Journey
日本語の認識テストです。
EasyOCR on M4 Mac GPU/CPU
Date: 2026-07-18 12:34:56"""


def main():
    print("=============================================================")
    print("   Phase 4: E2E OCR後処理 (マスタ名寄せ ＋ LLM構造化) デモ")
    print("=============================================================")

    # 1. OCR テキスト抽出 (EasyOCR)
    if os.path.exists(INPUT_IMAGE_PATH):
        print("\n[Step 1] EasyOCR による画像からのテキスト抽出...")
        reader = easyocr.Reader(['ja', 'en'], gpu=True)
        results = reader.readtext(INPUT_IMAGE_PATH)
        results_sorted = sorted(results, key=lambda r: r[0][0][1])
        raw_ocr_text = "\n".join([r[1] for r in results_sorted])
    else:
        print("\n[Step 1] サンプル画像なし -> デモ認識テキストを使用")
        raw_ocr_text = """AlOCR Learning Journey
日本語の認議テストです。
EasyOCR on M4 Mac GPUICPU
Date: 2026-07-1812.34.56"""

    print("\n--- 【OCR直後の生テキスト (Raw OCR Text)】 ---")
    print(raw_ocr_text)

    # 生OCRテキストのCER計測
    raw_cer_res = calculate_cer(GROUND_TRUTH_TEXT, raw_ocr_text)
    print(f" -> 補正前 CER: {raw_cer_res['cer']:.2f}% (編集距離: {raw_cer_res['distance']})")

    # 2. 社内マスタ自動名寄せ・補正
    print("\n[Step 2] 社内マスタ自動名寄せ・表記ゆれ自動補正の適用...")
    matcher = MasterMatcher()
    match_result = matcher.correct_text_lines(raw_ocr_text)
    corrected_text = match_result["corrected_text"]

    print("\n--- 【マスタ名寄せ補正後のテキスト (Corrected Text)】 ---")
    print(corrected_text)

    if match_result["corrections_made"]:
        print("\n[適用された名寄せ・自動補正一覧]:")
        for idx, corr in enumerate(match_result["corrections_made"], 1):
            print(f"  {idx}. '{corr['original']}'  ==>  '{corr['corrected']}' (CER: {corr['cer_diff']:.1f}%)")

    # 補正後テキストのCER計測
    corrected_cer_res = calculate_cer(GROUND_TRUTH_TEXT, corrected_text)
    print(f"\n -> 補正後 CER: {corrected_cer_res['cer']:.2f}% (編集距離: {corrected_cer_res['distance']})")
    print(f" 🚀 CER 改善率: {raw_cer_res['cer']:.2f}%  ==>  {corrected_cer_res['cer']:.2f}%")

    # 3. LLM (Gemini / Fallback) による JSON 構造化
    print("\n[Step 3] LLM による非構造テキストの JSON 構造化変換...")
    structurer = LLMStructurer()
    structured_json = structurer.structure_text(corrected_text)

    print("\n--- 【生成された構造化 JSON (Structured JSON Output)】 ---")
    print(json.dumps(structured_json, ensure_ascii=False, indent=2))

    # JSON 保存
    json_output_path = os.path.join(OUTPUT_DIR, "structured_result.json")
    with open(json_output_path, "w", encoding="utf-8") as f:
        json.dump(structured_json, f, ensure_ascii=False, indent=2)

    print(f"\n構造化 JSON ファイルを保存しました: {json_output_path}")
    print("=============================================================")


if __name__ == "__main__":
    main()
