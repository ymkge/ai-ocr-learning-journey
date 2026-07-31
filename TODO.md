# TODO List for AI OCR Learning Journey

## 📅 明日（次回のセッション）再開時のタスク

### Phase 4: 後処理（NLP）とマスタ名寄せの実装
- [ ] `04_post_processing/` ディレクトリの作成
- [ ] **LLM (Gemini API) を用いた構造化データ変換**
  - [ ] Google AI Studio APIキー・`google-genai` を利用したテキストのJSON構造化スクリプトの実装 (`04_post_processing/llm_structuring.py`)
- [ ] **台帳名寄せ・自動補正ロジックの実装**
  - [ ] 曖昧マッチングや編集距離（Phase 3の `evaluate_cer.py` ロジック活用）を用いた、社内マスタデータ（台帳）への自動名寄せ・表記ゆれ補正スクリプトの実装 (`04_post_processing/master_matching.py`)

---

## 履歴（完了済みタスク）
- [x] **Phase 1**: 簡易OCRスクリプト (`01_easy_ocr/run_ocr.py`) の実装と環境構築
- [x] **Phase 2**: テキスト検出 (CRAFT) と認識 (CRNN+CTC) の分離実行デモ (`02_detection_recognition/text_detection_demo.py`, `text_recognition_demo.py`) の実装および検証
- [x] **Phase 3**: 精度評価 (CER/WER) & OpenCV 画像前処理 (二値化, ノイズ除去, CLAHE, Deskew) A/Bテスト実験 (`03_data_science/`) の完了
- [x] **プロジェクト管理**: GitHub Issue投稿用ドキュメント (`03_data_science/phase3_execution_report.md`) の保存、`README.md` 実験レポートテーブルの更新完了

