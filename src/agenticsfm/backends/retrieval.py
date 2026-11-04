"""Backend (pointglue env): global descriptors and top-k retrieval pairs with hloc.

Spec: {images_dir, image_names, method: "netvlad", k, out_dir, result_path}
Result: {pairs: [[name0, name1], ...] (unordered, sorted names, each once), descriptor_path}
"""

from __future__ import annotations

import time
from pathlib import Path

from agenticsfm.backends import add_repo_paths, read_spec, write_result


def main() -> None:
    spec = read_spec()
    add_repo_paths(spec, "hloc")
    import torch
    from hloc import extract_features, pairs_from_retrieval

    out = Path(spec["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    names = list(spec["image_names"])
    conf = dict(extract_features.confs[spec["method"]])
    descriptors = out / f"global-feats-{spec['method']}.h5"
    start = time.time()
    extract_features.main(conf, Path(spec["images_dir"]), image_list=names, feature_path=descriptors)
    extract_seconds = time.time() - start
    directed = out / f"pairs-{spec['method']}{spec['k']}.txt"
    pairs_from_retrieval.main(descriptors, directed, num_matched=min(int(spec["k"]), len(names) - 1),
                              query_list=names, db_list=names)
    unordered = sorted({tuple(sorted(line.split())) for line in directed.read_text().splitlines() if line})
    write_result(spec, {"pairs": [list(p) for p in unordered], "descriptor_path": str(descriptors),
                        "cost": {"extract_seconds": extract_seconds,
                                 "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None}})


if __name__ == "__main__":
    main()
