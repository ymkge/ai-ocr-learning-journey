# TODO List for AI OCR Learning Journey

## 📅 明日（次回のセッション）再開時のタスク

### 1. Phase 3: データサイエンティストとしての精度評価と画像前処理
- [ ] `03_data_science/` ディレクトリの作成
- [ ] **評価指標（CER: 文字エラー率）の自作実装**
  - [ ] レーベンシュタイン距離（編集距離）の計算ロジックの実装
  - [ ] 正解テキストとOCR予測テキストから文字エラー率(CER)を算出する `evaluate_cer.py` の作成
- [ ] **OpenCVを用いた画像前処理の実装と実験**
  - [ ] 二値化 (Binarization)、ノイズ除去、傾き補正 (Deskew) ロジックの実装
  - [ ] 前処理なし（ベースライン）と前処理ありのOCR結果のA/Bテスト実験
  - [ ] 前処理によって、Phase 1/2で発生したエラー（「。」の欠落や英数字の誤認）が改善するか検証
- [ ] **精度実験レポートの更新**
  - [ ] 実験結果を [README.md](file:///Users/ymto/Documents/git/ai-ocr-learning-journey/README.md) の「精度実験レポート」テーブルに追記

### 2. Phase 4: 後処理（NLP）とマスタ名寄せの実装
- [ ] `04_post_processing/` ディレクトリの作成
- [ ] **LLM (Gemini API) を用いた構造化データ変換**
  - [ ] Google AI Studio APIキーを利用したテキストのJSON化スクリプトの実装
- [ ] **台帳名寄せ・自動補正ロジックの実装**
  - [ ] 曖昧マッチングや編集距離を用いた、マスターデータ（台帳）への自動名寄せロジックの実装

---

## 履歴（完了済みタスク）
- [x] **Phase 1**: 簡易OCRスクリプト (`01_easy_ocr/run_ocr.py`) の実装と環境構築
- [x] **Phase 2**: テキスト検出 (CRAFT) と認識 (CRNN+CTC) の分離実行デモ (`02_detection_recognition/text_detection_demo.py`) の実装および検証
- [x] **プロジェクト管理**: `.gitignore` および `.agent/AGENTS.md` の設定完了
