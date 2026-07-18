import os
import cv2
import easyocr
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import json

# パスの定義
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(BASE_DIR, "images")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

# ディレクトリの作成
os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_sample_image(filepath):
    """動作確認用のサンプル画像を生成する（日本語と英語の混在）"""
    print("サンプル画像を生成しています...")
    # 600x400の白い画像を作成
    img = Image.new("RGB", (600, 300), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    
    # フォントの選択 (Mac標準 of 日本語フォントパスをいくつか試す)
    font_paths = [
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
        "/System/Library/Fonts/Supplemental/Osaka.ttf",
        "/System/Library/Fonts/ヒラギノ角ゴ Pro W3.otf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",  # 英語のみのフォールバック
    ]
    
    font = None
    for path in font_paths:
        if os.path.exists(path):
            try:
                font = ImageFont.truetype(path, 32)
                print(f"フォントを使用します: {path}")
                break
            except Exception:
                continue
                
    if font is None:
        font = ImageFont.load_default()
        print("デフォルトのフォントを使用します（日本語は文字化けする可能性があります）")

    # テキストの描画
    draw.text((50, 40), "AI OCR Learning Journey", fill=(0, 0, 0), font=font)
    draw.text((50, 100), "日本語の認識テストです。", fill=(0, 0, 128), font=font)
    draw.text((50, 160), "EasyOCR on M4 Mac GPU/CPU", fill=(128, 0, 0), font=font)
    draw.text((50, 220), "Date: 2026-07-18 12:34:56", fill=(50, 50, 50), font=font)
    
    img.save(filepath)
    print(f"サンプル画像を保存しました: {filepath}")

def run_ocr():
    sample_image_path = os.path.join(IMAGE_DIR, "sample.png")
    
    # 画像が存在しない場合は自動生成
    if not os.path.exists(sample_image_path):
        generate_sample_image(sample_image_path)
        
    print("\nOCR処理を開始します。EasyOCRモデルをロード中...")
    # EasyOCRの初期化 (日本語 'ja' と 英語 'en' を指定)
    # Mac M4環境では、EasyOCR(PyTorch)は自動的に最適なデバイスを選びますが、
    # 基本的にCPUまたはMPSを利用します。
    reader = easyocr.Reader(['ja', 'en'], gpu=True)
    
    print("画像を読み込み、テキスト検出・認識を実行しています...")
    results = reader.readtext(sample_image_path)
    
    # 結果の表示と保存データの準備
    output_data = []
    print("\n--- OCR 認識結果 ---")
    for idx, (bbox, text, prob) in enumerate(results):
        # bboxは [[x1, y1], [x2, y2], [x3, y3], [x4, y4]] の形式
        # jsonシリアライズ可能な型に変換
        bbox_list = [[int(pt[0]), int(pt[1])] for pt in bbox]
        print(f"[{idx+1}] Text: \"{text}\" (Confidence: {prob:.4f})")
        print(f"    Bounding Box: {bbox_list}")
        
        output_data.append({
            "index": idx + 1,
            "text": text,
            "confidence": float(prob),
            "box": bbox_list
        })
        
    # 結果をJSONとして保存
    result_json_path = os.path.join(OUTPUT_DIR, "result.json")
    with open(result_json_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)
    print(f"\n認識データを保存しました: {result_json_path}")
    
    # 画像にバウンディングボックスを描画して保存 (OpenCVを使用)
    image = cv2.imread(sample_image_path)
    for item in output_data:
        box = item["box"]
        # 左上と右下の座標を取得
        pts = np.array(box, np.int32)
        pts = pts.reshape((-1, 1, 2))
        # 枠線を描画
        cv2.polylines(image, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
        # インデックス番号を描画 (OpenCVは日本語描画が困難なため、検出番号のみを描画)
        cv2.putText(
            image, 
            str(item["index"]), 
            (box[0][0], box[0][1] - 5), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.6, 
            (0, 0, 255), 
            2
        )
        
    annotated_image_path = os.path.join(OUTPUT_DIR, "annotated_sample.png")
    cv2.imwrite(annotated_image_path, image)
    print(f"結果描画画像を保存しました: {annotated_image_path}")
    print("OCR処理が正常に完了しました。")

if __name__ == "__main__":
    run_ocr()
