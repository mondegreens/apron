"""Layer 0.3 — Spike: OOM injection via vLLM.

GO:   vLLM fails with parseable error containing "out of memory".
NO-GO: vLLM succeeds → increase pressure or rethink mechanism.

Usage:
    RUNPOD_API_KEY=... python tests/spikes/spike_oom_injection.py
"""

from __future__ import annotations

import os
import sys
import time

from tests.spikes.spike_runpod_ssh import (
    create_pod,
    terminate_pod,
    wait_for_running,
)

MODEL = "Qwen/Qwen3-8B"


def ssh_exec(pod: dict, command: str) -> tuple[str, str, int]:
    import paramiko

    ports = pod["runtime"]["ports"]
    ssh_port_info = next(p for p in ports if p["privatePort"] == 22)
    host = ssh_port_info["ip"]
    port = int(ssh_port_info["publicPort"])

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.WarningPolicy())
    ssh_key_path = os.environ.get("RUNPOD_SSH_KEY_PATH", os.path.expanduser("~/.ssh/id_ed25519"))
    client.connect(host, port=port, username="root", key_filename=ssh_key_path)
    _, stdout, stderr = client.exec_command(command, timeout=300)
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

        print(
            "Attempting OOM injection: gpu_memory_utilization=0.99 + max_num_batched_tokens=65536"
        )
        cmd = (
            f"timeout 120 vllm serve {MODEL} "
            "--dtype bfloat16 "
            "--gpu-memory-utilization 0.99 "
            "--max-num-batched-tokens 65536 "
            "2>&1"
        )
        out, err, exit_code = ssh_exec(pod, cmd)
        combined = out + err

        print(f"\n--- vLLM output (exit={exit_code}) ---")
        print(combined[-2000:] if len(combined) > 2000 else combined)

        oom_markers = ["OutOfMemoryError", "out of memory", "CUDA out of memory", "OOM"]
        if any(m.lower() in combined.lower() for m in oom_markers):
            print("\nGO: OOM detected and parseable")
        else:
            print("\nNO-GO: no OOM detected in output")
            sys.exit(1)

        # Test the good config
        print("\nBooting with safe config: gpu_memory_utilization=0.90")
        safe_cmd = (
            f"timeout 180 vllm serve {MODEL} "
            "--dtype bfloat16 "
            "--gpu-memory-utilization 0.90 "
            "--max-model-len 640 "
            "2>&1 &"
        )
        ssh_exec(pod, safe_cmd)
        time.sleep(60)

        health_out, _, _ = ssh_exec(pod, "curl -s http://localhost:8000/health || echo FAILED")
        if "FAILED" not in health_out:
            print("GO: safe config boots successfully")
        else:
            print("NO-GO: safe config also failed")
            sys.exit(1)

    finally:
        if pod_id:
            terminate_pod(api_key, pod_id)


if __name__ == "__main__":
    main()
