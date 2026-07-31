"""
Phase 3: 精度評価モジュール (evaluate_cer.py)

AI OCRの評価指標である CER (Character Error Rate: 文字エラー率) および WER (Word Error Rate: 単語エラー率)
を自作実装し、OCR予測結果と正解データ(Ground Truth)の精度比較を行います。
"""

import Levenshtein
import re

def compute_levenshtein_dp(s1: str, s2: str) -> int:
    """
    動的計画法 (Dynamic Programming) によるレーベンシュタイン距離 (編集距離) の自作実装。
    挿入 (Insertion)、削除 (Deletion)、置換 (Substitution) の最小コストを算出します。
    """
    m, n = len(s1), len(s2)
    # DPテーブルの初期化 ( (m+1) x (n+1) )
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # 初期条件
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    # DP テーブルの更新
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,      # 削除 (Deletion)
                dp[i][j - 1] + 1,      # 挿入 (Insertion)
                dp[i - 1][j - 1] + cost # 置換 (Substitution)
            )

    return dp[m][n]


def calculate_cer(reference: str, hypothesis: str, ignore_spaces: bool = False) -> dict:
    """
    CER (Character Error Rate: 文字エラー率) を算出します。
    CER = (挿入数 + 削除数 + 置換数) / (正解文字列の全文字数) = 編集距離 / len(reference)
    
    Args:
        reference (str): 正解テキスト (Ground Truth)
        hypothesis (str): OCR予測テキスト
        ignore_spaces (bool): スペースを無視して評価するかどうか
        
    Returns:
        dict: CER (%), 編集距離, 正解文字数, 予測文字数
    """
    ref = reference.replace(" ", "") if ignore_spaces else reference
    hyp = hypothesis.replace(" ", "") if ignore_spaces else hypothesis

    if len(ref) == 0:
        return {"cer": 0.0, "distance": 0, "ref_len": 0, "hyp_len": len(hyp)}

    # 自作DPとライブラリ(Levenshtein)の両方で検証
    dist_dp = compute_levenshtein_dp(ref, hyp)
    dist_lib = Levenshtein.distance(ref, hyp)
    
    assert dist_dp == dist_lib, f"DP実装とLevenshteinライブラリの計算結果が一致しません: DP={dist_dp}, Lib={dist_lib}"

    cer = (dist_dp / len(ref)) * 100.0
    return {
        "cer": cer,
        "distance": dist_dp,
        "ref_len": len(ref),
        "hyp_len": len(hyp)
    }


def calculate_wer(reference: str, hypothesis: str) -> dict:
    """
    WER (Word Error Rate: 単語エラー率) を算出します。
    文字列を単語リストに分割し、単語単位での編集距離から算出します。
    """
    ref_words = re.findall(r'\S+', reference)
    hyp_words = re.findall(r'\S+', hypothesis)

    if len(ref_words) == 0:
        return {"wer": 0.0, "distance": 0, "ref_words": 0, "hyp_words": len(hyp_words)}

    # 単語単位の編集距離
    m, n = len(ref_words), len(hyp_words)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1): dp[i][0] = i
    for j in range(n + 1): dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if ref_words[i - 1] == hyp_words[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)

    word_dist = dp[m][n]
    wer = (word_dist / len(ref_words)) * 100.0
    return {
        "wer": wer,
        "word_distance": word_dist,
        "ref_words": len(ref_words),
        "hyp_words": len(hyp_words)
    }


if __name__ == "__main__":
    print("=== CER / WER 評価モジュールの動作確認テスト ===")
    ref_text = "AI OCR Learning Journey"
    hyp_text = "AlOCR Learning Journey"  # 'I' -> 'l' 誤認識 (編集距離: 1)

    cer_res = calculate_cer(ref_text, hyp_text)
    wer_res = calculate_wer(ref_text, hyp_text)

    print(f"正解 (Ref)  : '{ref_text}'")
    print(f"予測 (Hyp)  : '{hyp_text}'")
    print(f"編集距離    : {cer_res['distance']}")
    print(f"CER (文字)  : {cer_res['cer']:.2f}%")
    print(f"WER (単語)  : {wer_res['wer']:.2f}%")
    print("ユニットテスト成功！")
