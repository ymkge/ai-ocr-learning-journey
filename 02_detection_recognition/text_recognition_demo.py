import os
import torch
import torch.nn as nn
import cv2
import numpy as np
from PIL import Image

# パスの定義
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CROPPED_IMAGE_PATH = os.path.join(BASE_DIR, "outputs/cropped_line_2.png")  # 日本語行の画像

# ==========================================
# 1. CRNN (CNN + RNN) モデルの定義 (PyTorch)
# ==========================================
class MiniCRNN(nn.Module):
    """
    CRNNアーキテクチャの簡易モデル。
    入力画像からCNNで特徴を抽出し、RNNで時系列データとして処理し、文字の確率分布を出力します。
    """
    def __init__(self, num_classes):
        super(MiniCRNN, self).__init__()
        
        # --- CNN Feature Extractor ---
        # 入力画像の高さ(H)を32から1へと圧縮しながら、チャンネル方向の特徴を増やします。
        self.cnn = nn.Sequential(
            # Layer 1: Input (1, 32, W) -> Output (64, 16, W/2)
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Layer 2: Input (64, 16, W/2) -> Output (128, 8, W/4)
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Layer 3: Input (128, 8, W/4) -> Output (256, 8, W/4)
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            
            # Layer 4: Input (256, 8, W/4) -> Output (256, 4, W/8)
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)), # ここで高さ方向が4になる
            
            # Layer 5: Input (256, 4, W/8) -> Output (512, 1, W/8)
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=(4, 1), stride=1),  # 高さを1に潰す(Squeeze)
            nn.ReLU(True)
        )

        # --- Map-to-Sequence (特徴変換) ---
        # CNNの出力をRNNに入力できるように転置します。
        
        # --- RNN Sequence Labeling (Bi-LSTM) ---
        # 特徴マップの幅(W/8)を時系列の長さ(SeqLen)とみなし、左から右へスキャンします。
        # 双方向LSTM(Bi-directional LSTM)を2層重ねます。
        self.rnn = nn.Sequential(
            nn.LSTM(input_size=512, hidden_size=256, bidirectional=True, num_layers=1, batch_first=False),
        )
        # Bi-LSTMの出力次元は hidden_size * 2 (双方向のため512次元)
        
        # --- Transcription Layer (分類器) ---
        # 各タイムステップにおける、定義された文字クラス(漢字、平仮名、英数字、blankなど)の確率を出力します。
        self.fc = nn.Linear(256 * 2, num_classes)

    def forward(self, x):
        # 1. CNNを通過
        # 入力: [Batch, Channel=1, Height=32, Width=W]
        print(f" [Forward] 入力テンソル形状: {x.shape}")
        
        # CNN特徴抽出
        # Conv2d -> MaxPool2d を経て、高さを1に潰します
        features = self.cnn(x)  # [B, 512, 1, W_new]
        print(f" [Forward] CNN特徴抽出後形状: {features.shape}")
        
        # 2. Map-to-Sequence
        # [B, 512, 1, W_new] -> [B, 512, W_new] に次元削除 (Squeeze)
        features = features.squeeze(2)
        # PyTorchのRNN(LSTM)は通常 [Sequence_Length, Batch_Size, Feature_Dimension] の形状を期待するため転置
        features = features.permute(2, 0, 1)  # [W_new, B, 512]
        print(f" [Forward] Sequence変換後形状 (LSTM入力): {features.shape}")
        
        # 3. RNNを通過
        out, _ = self.rnn(features)  # [SeqLen, B, 512]
        print(f" [Forward] RNN (Bi-LSTM) 出力形状: {out.shape}")
        
        # 4. 全結合層(分類器)
        out = self.fc(out)  # [SeqLen, B, num_classes]
        print(f" [Forward] 最終クラス確率出力形状: {out.shape}")
        
        return out

# ==========================================
# 2. CTC (Connectionist Temporal Classification) デコードのシミュレーション
# ==========================================
def ctc_greedy_decode(predictions, char_map):
    """
    CTCのGreedyデコードアルゴリズム。
    各タイムステップで最も確率の高い文字を選択し、重複を排除し、ブランク(空白文字)を除去します。
    """
    # predictions: [SeqLen, Num_Classes] の想定
    # 最も確率の高いインデックスを各タイムステップで取得
    best_path = torch.argmax(predictions, dim=1).numpy()
    print(f"\n [CTC Decode] 予測された生のインデックス列 (Best Path):\n  {list(best_path)}")
    
    # デコードアルゴリズムの適用:
    # 1. 連続して重複する文字インデックスをマージする (例: 1, 1, 0, 0, 2, 2 -> 1, 0, 2)
    # 2. ブランクインデックス (通常は0番) を除去する
    blank_idx = 0
    decoded_indices = []
    prev_idx = -1
    
    for idx in best_path:
        if idx != prev_idx:
            if idx != blank_idx:
                decoded_indices.append(idx)
            prev_idx = idx
            
    # 文字列の復元
    result_text = "".join([char_map[i] for i in decoded_indices])
    return result_text

# ==========================================
# 3. メイン実行処理
# ==========================================
def run_recognition_demo():
    print("--- Phase 2: Text Recognition (CRNN + CTC) 実践デモ ---\n")
    
    if not os.path.exists(CROPPED_IMAGE_PATH):
        print(f"エラー: クロップ画像が見つかりません。パス: {CROPPED_IMAGE_PATH}")
        print("先に text_detection_demo.py を実行してください。")
        return

    # 1. 画像の読み込みと前処理 (OCRの入力として画像高さを32pxに固定するのが一般的)
    # グレースケールで読み込み
    img = cv2.imread(CROPPED_IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
    h, w = img.shape
    
    # 高さを32pxにリサイズし、アスペクト比を維持して幅を計算
    target_h = 32
    target_w = int(w * (target_h / h))
    img_resized = cv2.resize(img, (target_w, target_h))
    
    # テンソルへの変換 [Batch=1, Channel=1, Height=32, Width=target_w]
    # 値を0〜1に正規化
    img_tensor = torch.FloatTensor(img_resized) / 255.0
    img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)  # 次元追加: [1, 1, 32, target_w]
    
    print(f"1. 画像の前処理:")
    print(f"   元の画像サイズ: {w}x{h}")
    print(f"   リサイズ後の画像サイズ: {target_w}x32 (高さ32に正規化)")
    print(f"   入力テンソル形状: {img_tensor.shape}\n")

    # 2. クラス数 (文字の種類) の設定と仮想キャラクターマップの構築
    # 本来は学習データに含まれる全文字（数千〜数万種類）ですが、今回はデモ用に制限します。
    # 0番は「空白文字 (blank)」として予約します。これがCTCデコードの核心です。
    char_map = {
        0: "[blank]", 1: "日", 2: "本", 3: "語", 4: "の", 
        5: "認", 6: "識", 7: "テ", 8: "ス", 9: "ト", 10: "で", 11: "す"
    }
    num_classes = len(char_map)

    # 3. CRNNモデルの初期化
    model = MiniCRNN(num_classes=num_classes)
    model.eval()  # 推論モード

    # 4. モデルへの入力と形状変化の確認 (Forward Pass)
    print("2. モデルの内部フォワードパス:")
    with torch.no_grad():
        output = model(img_tensor)

    # 5. CTCデコーダの動作シミュレーション
    # 実際の予測確率は学習していないためランダムですが、
    # どのように「文字にデコードされるか」をシミュレートするため、
    # 擬似的な予測テンソルを作成してデコーダに入力します。
    print("\n3. CTCデコードのシミュレーション:")
    print("   (画像から文字の並びを復元する仕組みの再現)")
    
    # タイムステップ数 (Sequence Length) = 20
    # クラス数 = 12
    # 擬似的なモデル出力（各タイムステップにおける全クラスの確率分布の対数）
    seq_len = 20
    pseudo_logits = torch.randn(seq_len, num_classes)
    
    # 特定のタイムステップで正しい文字の確率が高くなるように上書き設定します
    # 文字列「日本語の認識テストです」のインデックス: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    # 連続した同じインデックスや、blank (0) を意図的に配置します
    target_indices = [
        0, 0, # [blank], [blank]
        1, 1, # 日, 日 ➡ マージされて「日」
        2,    # 本
        3, 3, # 語, 語 ➡ マージされて「語」
        0,    # [blank]
        4,    # の
        5, 5, # 認, 認
        6,    # 識
        7, 7, # テ, テ
        8,    # ス
        9,    # ト
        10,   # で
        11, 11,# す, す
        0     # [blank]
    ]
    
    # 20ステップ分のインデックスに合わせる
    for step in range(seq_len):
        char_idx = target_indices[step]
        # その文字クラスの値を最大にする
        pseudo_logits[step, :] = -5.0 # 一旦全て低い値にする
        pseudo_logits[step, char_idx] = 5.0 # ターゲット文字だけ値を高くする

    # デコードの実行
    decoded_text = ctc_greedy_decode(pseudo_logits, char_map)
    print(f"\n [CTC Decode] デコード結果テキスト: \"{decoded_text}\"")
    print("\n--- 理論ミニ解説: CTC Lossとデコードの必要性 ---")
    print("画像から文字を読み取る際、『画像のどのピクセルがどの文字に対応するか(アライメント)』を")
    print("事前に正確に定義するのは困難です。また、同じ文字でも書く速度や幅によって画像上のサイズが異なります。")
    print("CTC (Connectionist Temporal Classification) は、文字の重複(日日➡日)や、")
    print("文字と文字の間の『隙間』を表す [blank] という特殊な文字を導入することで、")
    print("アライメント情報なしで、時系列シーケンスから最終的なテキストを復元できるようにしています。")

if __name__ == "__main__":
    run_recognition_demo()
