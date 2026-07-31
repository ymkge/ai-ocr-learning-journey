"""
Phase 4: 社内マスタ自動名寄せ・表記ゆれ自動補正モジュール (master_matching.py)

OCRの認識結果（「認議」、「AlOCR」、「GPUICPU」など視覚的誤誤認や外来語ノイズ）を、
社内標準マスタ辞書データと照合し、レーベンシュタイン距離/CERに基づき自動名寄せ・補正します。
"""

import sys
import os
import re

# Phase 3 の評価モジュールをインポート
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PHASE3_DIR = os.path.join(BASE_DIR, "../03_data_science")
sys.path.append(PHASE3_DIR)

from evaluate_cer import compute_levenshtein_dp, calculate_cer

class MasterMatcher:
    """
    社内マスタデータとの照合および自動補正を行うクラス
    """
    def __init__(self, master_dictionary: list[str] | None = None):
        # デフォルトの社内標準語彙・台帳辞書
        if master_dictionary is None:
            self.master_dictionary = [
                "AI OCR Learning Journey",
                "日本語の認識テストです。",
                "EasyOCR on M4 Mac GPU/CPU",
                "Date",
                "EasyOCR",
                "M4 Mac",
                "GPU/CPU",
                "日本語",
                "認識テスト",
            ]
        else:
            self.master_dictionary = master_dictionary

    def find_best_match(self, query: str, threshold_cer: float = 40.0) -> tuple[str, float, int]:
        """
        クエリ文字列に最も近いマスタ語彙を検索し、補正テキストを返します。
        
        Args:
            query (str): OCR認識された誤認可能性のあるテキスト
            threshold_cer (float): 名寄せを適用する最大許容CER (%)。これを超える場合は変化なし。
            
        Returns:
            (補正後テキスト, 最小CER %, 編集距離)
        """
        best_candidate = query
        min_cer = 100.0
        best_dist = 999

        for master_word in self.master_dictionary:
            # 長さが極端に違うものはスキップ
            if abs(len(master_word) - len(query)) > max(len(query), 5):
                continue
                
            res = calculate_cer(master_word, query)
            cer = res["cer"]
            dist = res["distance"]

            if cer < min_cer:
                min_cer = cer
                best_candidate = master_word
                best_dist = dist

        # 閾値以下の類似度であればマスタ語彙で置換補正
        if min_cer <= threshold_cer and min_cer > 0:
            return best_candidate, min_cer, best_dist
        else:
            return query, 0.0, 0

    def correct_text_lines(self, input_text: str) -> dict:
        """
        複数行のテキストに対し、各行ごとにマスタ照合を行って自動補正テキストを生成します。
        """
        lines = input_text.splitlines()
        corrected_lines = []
        corrections_made = []

        for line in lines:
            stripped = line.strip()
            if not stripped:
                corrected_lines.append("")
                continue
                
            corrected, cer, dist = self.find_best_match(stripped)
            corrected_lines.append(corrected)
            
            if corrected != stripped:
                corrections_made.append({
                    "original": stripped,
                    "corrected": corrected,
                    "cer_diff": cer,
                    "distance": dist
                })

        return {
            "corrected_text": "\n".join(corrected_lines),
            "corrections_made": corrections_made
        }


if __name__ == "__main__":
    print("=== Phase 4: マスタ自動名寄せモジュールのテスト ===")
    matcher = MasterMatcher()
    
    # 実際に出現した誤認識テキストのサンプル
    test_cases = [
        "AlOCR Learning Journey",      # 'AI' -> 'Al' 誤認知
        "日本語の認議テストです。",      # '認識' -> '認議' 誤認知
        "EasyOCR on M4 Mac GPUICPU",   # '/' -> 'I' 誤認知
    ]

    for original in test_cases:
        corrected, cer, dist = matcher.find_best_match(original)
        print(f"元テキスト  : {original}")
        print(f"名寄せ補正後: {corrected} (CER: {cer:.2f}%, 編集距離: {dist})")
        print("-" * 50)
