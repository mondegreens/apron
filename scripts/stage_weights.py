"""Stage model weights onto Apron's network volume in one datacenter, no GPU.

The cohort run stages weights just before it rents GPUs, and only in a
datacenter that has the GPUs in stock right now.  This stages ahead of time
(e.g. overnight, while GPU stock is short): it grows the existing
``apron-weights-<dc>`` volume to hold everything already on it plus the new
models, accrues its storage so far, and downloads the models from a CPU pod in
the same datacenter.  No GPU pod is created.

    uv run python scripts/stage_weights.py --dc US-CA-2 --tag groupC \
        zai-org/GLM-5.3-Flash deepseek-ai/DeepSeek-V4.1-Flash
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from apron.adapters.backends.runpod_storage import VOLUME_NAME_PREFIX, RunPodStorage
from apron.interfaces.cohort_root import (
    RUN_DIR,
    WeightsSite,
    accrue_storage,
    build_ports,
    stage_site,
    staged_bytes,
    staged_on,
)

HEADROOM = 1.15


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--dc", required=True, help="datacenter of the existing volume")
    parser.add_argument("--tag", required=True, help="suffix of the prestage record")
    parser.add_argument("models", nargs="+")
    args = parser.parse_args(argv)

    storage = RunPodStorage(os.environ["RUNPOD_API_KEY"])
    name = f"{VOLUME_NAME_PREFIX}-{args.dc.lower()}"
    volume = next((v for v in storage.list_volumes() if v.get("name") == name), None)
    if volume is None:
        print(f"no {name} volume; the cohort run creates volumes", file=sys.stderr)
        return 2
    before = WeightsSite(str(volume["id"]), args.dc, int(volume["size"]))

    on_volume = staged_on(before.volume_id) | set(args.models)
    need_gb = int(sum(staged_bytes(m) for m in sorted(on_volume)) * HEADROOM / 1e9) + 10
    ports = build_ports(site=before)
    accrue_storage(before, ports.budget)  # the hours so far, at the size they had
    grown = storage.ensure_volume(args.dc, need_gb)
    site = WeightsSite(before.volume_id, args.dc, int(grown.get("size") or need_gb))
    print(
        f"volume {site.volume_id} in {site.data_center_id}: {before.size_gb} -> {site.size_gb} GB"
    )

    result = stage_site(site, list(args.models), ports)
    record = {
        "site": site.__dict__,
        "staging": {
            "pod_id": result.pod_id,
            "cost": result.cost,
            "seconds": result.seconds,
            "error": result.error,
            "models": [m.__dict__ for m in result.models],
        },
    }
    (RUN_DIR / f"prestage-{args.tag}.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n"
    )
    for m in result.models:
        print(f"{m.model_id}: {'ok' if m.ok else 'FAILED'}")
    return 0 if result.ok else 1


if __name__ == "__main__":
    sys.exit(main())
