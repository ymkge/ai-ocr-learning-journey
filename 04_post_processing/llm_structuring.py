"""
Phase 4: LLM によるテキスト構造化 (JSON) モジュール (llm_structuring.py)

補正済みOCRテキストから、キーバリュー形式の構造化データ (JSON) を抽出・変換します。
Google AI Studio の GEMINI_API_KEY が設定されている場合は Gemini API (google-genai SDK) を優先呼び出し、
モデル名には常に最新のエイリアス 'gemini-flash-latest' を使用して将来的な非推奨化を防ぎます。
未設定またはクォータ上限時はルールベースの構造化フォールバック (Demo Fallback) で安全に実行します。
"""

import os
import json
import re

# 推奨の新 SDK `google-genai` を優先インポート
HAS_GENAI = False
USE_NEW_SDK = False

try:
    from google import genai
    HAS_GENAI = True
    USE_NEW_SDK = True
except ImportError:
    try:
        import google.generativeai as genai
        HAS_GENAI = True
        USE_NEW_SDK = False
    except ImportError:
        HAS_GENAI = False


class LLMStructurer:
    """
    OCRテキストを構造化JSONデータに変換するクラス
    """
    def __init__(self, api_key: str | None = None, model_name: str = "gemini-flash-latest"):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self.model_name = model_name
        self.use_api = False

        if HAS_GENAI and self.api_key:
            try:
                if USE_NEW_SDK:
                    self.client = genai.Client(api_key=self.api_key)
                else:
                    genai.configure(api_key=self.api_key)
                    self.model = genai.GenerativeModel(self.model_name)
                self.use_api = True
                print(f"Gemini API (SDK: google-genai, Model: '{self.model_name}') が有効化されました。")
            except Exception as e:
                print(f"Gemini API 初期化エラー: {e} -> デモフォールバックモードを使用します。")
        else:
            print("GEMINI_API_KEY 未設定 -> デモフォールバックモード (ルールベース構造化) を使用します。")

    def structure_text(self, text: str) -> dict:
        """
        無構造なテキストを構造化JSONオブジェクトに変換
        """
        if self.use_api:
            return self._structure_with_gemini(text)
        else:
            return self._structure_fallback(text)

    def _structure_with_gemini(self, text: str) -> dict:
        """Gemini APIを呼び出してJSON構造化"""
        prompt = f"""
以下のOCR抽出テキストを解析し、適切なキーとバリューを持つ純粋なJSONオブジェクトのみを出力してください。
Markdownの装飾(```json など)は含めず、有効なJSON文字列のみを返してください。

【テキスト】:
{text}

【期待するJSON出力フォーマット例】:
{{
  "document_title": "タイトル",
  "description": "内容の説明",
  "system_info": "システム環境",
  "timestamp": "2026-07-18 12:34:56"
}}
"""
        try:
            if USE_NEW_SDK:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                )
                response_text = response.text.strip()
            else:
                response = self.model.generate_content(prompt)
                response_text = response.text.strip()

            cleaned_json_str = re.sub(r'^```json\s*|\s*```$', '', response_text, flags=re.MULTILINE).strip()
            return json.loads(cleaned_json_str)
        except Exception as e:
            err_msg = str(e)
            if "429" in err_msg or "Quota exceeded" in err_msg:
                print("💡 Gemini API クォータ上限検出 (429 Rate Limit) -> デモフォールバック (ルールベース構造化) を安全実行します。")
            else:
                print(f"💡 Gemini API 呼び出し制限 ({err_msg[:60]}...) -> デモフォールバックを実行します。")
            return self._structure_fallback(text)

    def _structure_fallback(self, text: str) -> dict:
        """APIキーが無い場合や制限時のルールベース構造化フォールバック"""
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        
        structured_data = {
            "document_title": lines[0] if len(lines) > 0 else "Unknown Title",
            "description": lines[1] if len(lines) > 1 else "",
            "system_info": lines[2] if len(lines) > 2 else "",
            "timestamp": "",
            "extracted_fields": {
                "raw_text_lines": lines
            }
        }

        # Date: などのパターン抽出
        for line in lines:
            if "Date:" in line or "202" in line:
                date_match = re.search(r'\d{4}-\d{2}-\d{2}\s*\d{2}:\d{2}:\d{2}', line)
                if date_match:
                    structured_data["timestamp"] = date_match.group(0)

        return structured_data


if __name__ == "__main__":
    print("=== Phase 4: LLM テキスト構造化モジュールのテスト ===")
    sample_ocr_text = """AI OCR Learning Journey
日本語の認識テストです。
EasyOCR on M4 Mac GPU/CPU
Date: 2026-07-18 12:34:56"""

    structurer = LLMStructurer()
    result_json = structurer.structure_text(sample_ocr_text)

    print("\n構造化結果 (JSON):")
    print(json.dumps(result_json, ensure_ascii=False, indent=2))
