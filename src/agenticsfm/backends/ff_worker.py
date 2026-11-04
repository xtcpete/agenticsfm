"""Persistent feedforward worker: loads the model once and serves group requests over stdin/stdout.

    <env python> -m agenticsfm.backends.ff_worker <model> <repos.json>

Protocol: one JSON spec per line on stdin (the same spec as the one-shot ``feedforward`` backend);
the worker writes the result JSON to ``spec["result_path"]`` and answers one line on stdout,
"OK <result_path>" or "ERR <message>". The one-shot backend reloads 1 to 4 GB of weights per call
(about 13 s for Pi3X); the solver makes dozens of calls per scene.
"""

from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np


def main() -> None:
    import os

    # the protocol gets its own descriptor; anything the libraries print (to fd 1 or sys.stdout) goes to the log
    proto = os.fdopen(os.dup(1), "w", buffering=1)
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    import torch

    from agenticsfm.backends import feedforward as ff

    which = sys.argv[1]
    repos = json.loads(Path(sys.argv[2]).read_text())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    if which == "pi3x":
        model = ff.load_pi3x(repos["pi3"], device)
    elif which == "vggt":
        model = ff.load_vggt(repos["vggt"], 518, device)
    elif which == "vggt_omega":
        model = ff.load_omega(repos["vggt_omega"], repos["vggt_omega_checkpoint"], device)
    else:
        raise SystemExit(f"unknown model {which}")
    proto.write(f"READY {time.time() - t0:.1f}\n")
    proto.flush()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        if line == "QUIT":
            break
        try:
            spec = json.loads(line)
            ff.serve_one(spec, model, which, device)
            proto.write(f"OK {spec['result_path']}\n")
        except Exception as e:  # report and keep serving
            proto.write("ERR " + json.dumps(f"{type(e).__name__}: {e}\n{traceback.format_exc()[-1500:]}") + "\n")
        proto.flush()


if __name__ == "__main__":
    main()
