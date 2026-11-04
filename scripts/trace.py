"""Print a run's trace in a readable form: tool calls with summaries, and the VLM's turns.

    python scripts/trace.py outputs/runs/agent/<scene> [--reasoning]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    parser.add_argument("--reasoning", action="store_true", help="print the VLM's reasoning too")
    parser.add_argument("--width", type=int, default=400)
    args = parser.parse_args()
    for line in (Path(args.run_dir) / "trace.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r.get("source") == "vlm":
            if args.reasoning and r.get("reasoning"):
                print(f"  [think] {r['reasoning'][:2000]}")
            print(f"[vlm] {r.get('content') or ''} -> {[(c['name'], c['arguments']) for c in r.get('tool_calls', [])]}"
                  f"  ({r.get('usage', {}).get('completion_tokens', 0)} tokens)")
        elif "tool" in r:
            s = json.dumps({k: v for k, v in r["summary"].items() if k not in ("renders", "path", "models")}, default=str)
            print(f"[{r['step']:2d}] {r['tool']}({json.dumps(r['params'])[:150]}){' cached' if r.get('cached') else ''} "
                  f"{r['cost'].get('wall', 0):.0f}s -> {r.get('artifact')} {s[:args.width]}")
        else:
            print(f"[{r.get('source')}] {json.dumps({k: v for k, v in r.items() if k not in ('step', 'time', 'source')})[:args.width]}")


if __name__ == "__main__":
    main()
