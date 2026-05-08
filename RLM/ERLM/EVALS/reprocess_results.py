import json
import os
import re
import sys
import glob

def extract_abcd(text: str) -> str:
    """Extract the final A/B/C/D choice from model output."""
    if not text:
        return ""
    # Look for patterns like "answer is A", "correct answer: B", or trailing "(A)"
    patterns = [
        r'(?:answer|choice|option)\s+(?:is\s+)?([A-D])\b',
        r'\bFINAL[_\s](?:ANSWER|VAR)[^A-D]*([A-D])\b',
        r'\*\*([A-D])\*\*',
        r'\(([A-D])\)',
        r'\b([A-D])\b(?:\s*$|\s*\n)',  # lone letter at end of line
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return m.group(1).upper()
    # Fallback: last standalone A/B/C/D in text
    matches = re.findall(r'\b([A-D])\b', text)
    if matches:
        return matches[-1].upper()
    
    # Very fallback: first A/B/C/D
    m = re.search(r'([A-D])', text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
        
    return text.strip()

def normalize(text: str) -> str:
    if not text: return ""
    return text.strip().upper()

def process_file(jsonl_path):
    results = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            if not line.strip(): continue
            data = json.loads(line)
            
            pred_raw = data.get('prediction', '')
            gold_raw = data.get('ground_truth', '')
            
            pred = extract_abcd(pred_raw)
            gold = normalize(gold_raw)
            
            em = 1.0 if pred == gold else 0.0
            data['prediction_extracted'] = pred
            data['exact_match'] = em
            results.append(data)
            
    # Print summary
    methods = {}
    for r in results:
        m = r['method']
        if m not in methods: methods[m] = []
        methods[m].append(r)
        
    print(f"\nResults for {os.path.basename(jsonl_path)}:")
    print(f"{'Method':<20} {'N':>3} {'EM':>6}")
    print("-" * 31)
    for m in sorted(methods.keys()):
        rows = methods[m]
        em = sum(r['exact_match'] for r in rows) / len(rows)
        print(f"{m:<20} {len(rows):>3} {em:>6.3f}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            for f in glob.glob(arg):
                process_file(f)
    else:
        # Process some default directories
        base_path = "RLM/ERLM/EVALS/results"
        patterns = [
            f"{base_path}/gemini25_flash/*.jsonl",
            f"{base_path}/ollama_qwen3_8b/*.jsonl"
        ]
        for p in patterns:
            for f in glob.glob(p):
                process_file(f)
