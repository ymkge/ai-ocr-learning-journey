import os
import cv2
import easyocr
import numpy as np

# パスの定義
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_IMAGE_PATH = os.path.join(BASE_DIR, "../01_easy_ocr/images/sample.png")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

# 出力ディレクトリの作成
os.makedirs(OUTPUT_DIR, exist_ok=True)

def learn_text_detection():
    print("--- Phase 2: Text Detection (CRAFT) 実践デモ ---")
    
    if not os.path.exists(INPUT_IMAGE_PATH):
        print(f"エラー: 入力画像が見つかりません。パス: {INPUT_IMAGE_PATH}")
        print("先に Phase 1 のスクリプトを実行してサンプル画像を生成してください。")
        return

    # 画像の読み込み
    image = cv2.imread(INPUT_IMAGE_PATH)
    orig_image = image.copy()
    h, w, _ = image.shape
    print(f"画像を読み込みました: {INPUT_IMAGE_PATH} (サイズ: {w}x{h})")

    # EasyOCRのReaderを初期化
    # 検出のみを行う場合でも、Readerの初期化が必要です。
    print("EasyOCR Reader を初期化中...")
    reader = easyocr.Reader(['ja', 'en'], gpu=True)
    
    # 1. テキスト検出 (Detection) のみ実行
    # readtext() ではなく detect() を使用することで、テキストの位置(bbox)のみを高速に検出します。
    # 返り値: (horizontal_list, free_list)
    #   - horizontal_list: 水平な矩形 [x_min, x_max, y_min, y_max] のリスト
    #   - free_list: 傾いた矩形 [[x1,y1], [x2,y2], [x3,y3], [x4,y4]] のリスト
    print("テキスト検出 (CRAFTアルゴリズム) を実行中...")
    horizontal_boxes, free_boxes = reader.detect(INPUT_IMAGE_PATH)
    
    print(f"\n検出結果:")
    print(f" - 水平テキスト領域数 (Horizontal boxes): {len(horizontal_boxes[0])}")
    print(f" - 傾きのあるテキスト領域数 (Free-form boxes): {len(free_boxes[0])}")

    # 2. 検出された各テキスト領域を切り出し (Crop) て保存
    # 実際のE2E OCRシステムでは、このように検出した部分画像を認識器(CRNNなど)に渡します。
    print("\n検出領域のクロップと可視化画像を作成しています...")
    
    # 水平矩形の処理
    # horizontal_boxes[0] には各矩形の座標 [x_min, x_max, y_min, y_max] が入っています
    for idx, box in enumerate(horizontal_boxes[0]):
        x_min, x_max, y_min, y_max = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        
        # 安全のために画像の範囲内にクリップ
        x_min = max(0, x_min)
        x_max = min(w, x_max)
        y_min = max(0, y_min)
        y_max = min(h, y_max)
        
        # 領域のクロップ
        cropped = orig_image[y_min:y_max, x_min:x_max]
        
        # クロップ画像の保存
        crop_path = os.path.join(OUTPUT_DIR, f"cropped_line_{idx+1}.png")
        cv2.imwrite(crop_path, cropped)
        print(f" ➡ クロップ画像を保存しました: {crop_path}")
        
        # 元画像に枠線を描画 (赤色)
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
        cv2.putText(
            image, 
            f"Line {idx+1}", 
            (x_min, y_min - 5), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.5, 
            (0, 0, 255), 
            1
        )

    # 3. 検出結果を描画した画像を保存
    annotated_path = os.path.join(OUTPUT_DIR, "detected_areas.png")
    cv2.imwrite(annotated_path, image)
    print(f"\n検出エリア描画画像を保存しました: {annotated_path}")
    
    print("\n--- 理論ミニ解説: CRAFTとは？ ---")
    print("EasyOCRのテキスト検出器として使われている CRAFT (Character Region Awareness for Text Detection) は、")
    print("文字の領域(Character Region)と、文字と文字のつながり(Affinity)の2つのヒートマップをニューラルネットワークで予測します。")
    print("これにより、文字が斜めに配置されていたり、変形していても高精度にテキスト領域を特定することができます。")
    print("検出された矩形領域(Cropされた画像)は、次のステップである『テキスト認識(CRNN等)』へ入力されます。")

if __name__ == "__main__":
    learn_text_detection()
