"""Layer 0.4 — Spike: GPU memory profiling via torch.cuda.mem_get_info.

GO:   Two distinct memory values, model_weight_memory > 0.
NO-GO: torch.cuda not available → need different profiling approach.

Usage:
    RUNPOD_API_KEY=... python tests/spikes/spike_profiling.py
"""

from __future__ import annotations

import json
import os
import sys
import time

from tests.spikes.spike_runpod_ssh import (
    create_pod,
    terminate_pod,
    wait_for_running,
)

MODEL = "Qwen/Qwen3-8B"

PRE_LOAD_SCRIPT = r"""
import json, torch
free, total = torch.cuda.mem_get_info()
result = {"pre_free": free, "pre_total": total}
with open("/workspace/pre_load.json", "w") as f:
    json.dump(result, f)
print(json.dumps(result))
"""

POST_LOAD_SCRIPT = r"""
import json, torch
free, total = torch.cuda.mem_get_info()
result = {"post_free": free, "post_total": total}
with open("/workspace/post_load.json", "w") as f:
    json.dump(result, f)
print(json.dumps(result))
"""


def ssh_exec(pod: dict, command: str, timeout: int = 300) -> tuple[str, str, int]:
    import paramiko

    ports = pod["runtime"]["ports"]
    ssh_port_info = next(p for p in ports if p["privatePort"] == 22)
    host = ssh_port_info["ip"]
    port = int(ssh_port_info["publicPort"])

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh_key_path = os.environ.get(
        "RUNPOD_SSH_KEY_PATH", os.path.expanduser("~/.ssh/id_ed25519")
    )
    client.connect(host, port=port, username="root", key_filename=ssh_key_path)
    _, stdout, stderr = client.exec_command(command, timeout=timeout)
    exit_code = stdout.channel.recv_exit_status()
    out = stdout.read().decode()
    err = stderr.read().decode()
    client.close()
    return out, err, exit_code


def main() -> None:
    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        print("RUNPOD_API_KEY not set")
        sys.exit(1)

    pod_id = None
    try:
        pod_id = create_pod(api_key)
        pod = wait_for_running(api_key, pod_id)

        # Pre-load measurement
        print("Collecting pre-load memory...")
        out, err, _ = ssh_exec(pod, f"python3 -c {json.dumps(PRE_LOAD_SCRIPT)}")
        print(f"  pre-load: {out.strip()}")

        # Boot vLLM in background
        print(f"Booting vLLM with {MODEL}...")
        serve_cmd = (
            f"nohup vllm serve {MODEL} "
            "--dtype bfloat16 "
            "--gpu-memory-utilization 0.90 "
            "--max-model-len 640 "
            "> /workspace/vllm.log 2>&1 &"
        )
        ssh_exec(pod, serve_cmd)

        # Wait for health
        print("Waiting for vLLM health endpoint...")
        for _ in range(30):
            time.sleep(10)
            out, _, code = ssh_exec(pod, "curl -sf http://localhost:8000/health")
            if code == 0:
                print("  vLLM healthy")
                break
        else:
            log_out, _, _ = ssh_exec(pod, "tail -50 /workspace/vllm.log")
            print(f"  vLLM failed to start:\n{log_out}")
            sys.exit(1)

        # Post-load measurement
        print("Collecting post-load memory...")
        out, err, _ = ssh_exec(pod, f"python3 -c {json.dumps(POST_LOAD_SCRIPT)}")
        print(f"  post-load: {out.strip()}")

        # Compute delta
        pre_out, _, _ = ssh_exec(pod, "cat /workspace/pre_load.json")
        post_out, _, _ = ssh_exec(pod, "cat /workspace/post_load.json")
        pre = json.loads(pre_out)
        post = json.loads(post_out)

        model_weight_memory = pre["pre_free"] - post["post_free"]
        print(f"\n--- Profiling Results ---")
        print(f"  Initial total:  {pre['pre_total'] / 1e9:.2f} GB")
        print(f"  Initial free:   {pre['pre_free'] / 1e9:.2f} GB")
        print(f"  Post-load free: {post['post_free'] / 1e9:.2f} GB")
        print(f"  Model weight:   {model_weight_memory / 1e9:.2f} GB")

        if model_weight_memory > 0:
            print("\nGO: profiling produces valid memory delta")
        else:
            print("\nNO-GO: model_weight_memory <= 0")
            sys.exit(1)

    finally:
        if pod_id:
            terminate_pod(api_key, pod_id)


if __name__ == "__main__":
    main()
