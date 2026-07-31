# TODO List for AI OCR Learning Journey

## 🚀 すべての Phase (Phase 1 〜 Phase 4) が完了しました！

### 💡 今後の発展的探求テーマ (Optional Future Work)
- [ ] 大量ドキュメント画像に対する前処理・OCRバッチ並行処理の分散化
- [ ] 表組み・複数カラムレイアウトの構文抽出 (Layout Analysis) モジュールの調査
- [ ] カスタム OCR 認識モデル（CRNN / Transformer）のファインチューニング実験

---

## 履歴（完了済みタスク）
- [x] **Phase 1**: 簡易OCRスクリプト (`01_easy_ocr/run_ocr.py`) の実装と環境構築
- [x] **Phase 2**: テキスト検出 (CRAFT) と認識 (CRNN+CTC) の分離実行デモ (`02_detection_recognition/text_detection_demo.py`, `text_recognition_demo.py`) の実装および検証
- [x] **Phase 3**: 精度評価 (CER/WER) & OpenCV 画像前処理 (二値化, ノイズ除去, CLAHE, Deskew) A/Bテスト実験 (`03_data_science/`) の完了
- [x] **Phase 4**: 社内マスタ自動名寄せ (`master_matching.py`) ＋ LLM テキスト構造化 (JSON) (`llm_structuring.py`) ＋ E2E 統合デモ (`run_phase4_demo.py`) の完了およびレポート (`phase4_execution_report.md`) 作成


