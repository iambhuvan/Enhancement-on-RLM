import json
import os
import re
import sys
import glob
import statistics
from collections import defaultdict

def extract_abcd(text: str) -> str:
    if not text: return ""
    patterns = [
        r'(?:answer|choice|option)\s+(?:is\s+)?([A-D])\b',
        r'\bFINAL[_\s](?:ANSWER|VAR)[^A-D]*([A-D])\b',
        r'\*\*([A-D])\*\*',
        r'\(([A-D])\)',
        r'\b([A-D])\b(?:\s*$|\s*\n)',
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m: return m.group(1).upper()
    matches = re.findall(r'\b([A-D])\b', text)
    if matches: return matches[-1].upper()
    m = re.search(r'([A-D])', text, re.IGNORECASE)
    if m: return m.group(1).upper()
    return text.strip()

def normalize(text: str) -> str:
    if not text: return ""
    return text.strip().upper()

def aggregate_results(patterns):
    by_method = defaultdict(list)
    
    for p in patterns:
        for f_path in glob.glob(p):
            with open(f_path, 'r') as f:
                for line in f:
                    if not line.strip(): continue
                    data = json.loads(line)
                    method = data['method']
                    
                    pred = extract_abcd(data.get('prediction', ''))
                    gold = normalize(data.get('ground_truth', ''))
                    
                    data['exact_match'] = 1.0 if pred == gold else 0.0
                    by_method[method].append(data)
    
    print("\nAGGREGATED SUMMARY:")
    print(f"{'Method':<20} {'N':>3} {'EM':>6} {'TotalTok':>10} {'InTok':>10} {'OutTok':>10} {'Time(s)':>8}")
    print("-" * 80)
    
    results_summary = {}
    
    for m in sorted(by_method.keys()):
        rows = by_method[m]
        em = statistics.mean(r['exact_match'] for r in rows)
        tok = statistics.mean(r['tokens_used'] for r in rows)
        itok = statistics.mean(r.get('input_tokens', 0) for r in rows)
        otok = statistics.mean(r.get('output_tokens', 0) for r in rows)
        wt = statistics.mean(r['wall_clock_s'] for r in rows)
        
        print(f"{m:<20} {len(rows):>3} {em:>6.3f} {tok:>10.0f} {itok:>10.0f} {otok:>10.0f} {wt:>8.1f}")
        
        results_summary[m] = {
            "em": em,
            "tokens": tok,
            "input_tokens": itok,
            "output_tokens": otok,
            "time": wt,
            "n": len(rows)
        }
    
    # Calculate Redux and Speedup vs rlm_baseline
    if "rlm_baseline" in results_summary:
        base = results_summary["rlm_baseline"]
        print("\nOPTIMIZATION VERIFICATION (vs rlm_baseline):")
        print(f"{'Method':<20} {'TokRedux%':>10} {'Speedup':>10}")
        print("-" * 42)
        for m in sorted(results_summary.keys()):
            res = results_summary[m]
            redux = (base['tokens'] - res['tokens']) / base['tokens'] * 100
            speedup = base['time'] / res['time'] if res['time'] > 0 else 0
            print(f"{m:<20} {redux:>9.1f}% {speedup:>9.2f}x")

if __name__ == "__main__":
    base_path = "RLM/ERLM/EVALS/results"
    # Focus on gemini25_flash as it has the most complete runs
    aggregate_results([f"{base_path}/gemini25_flash/*.jsonl"])
