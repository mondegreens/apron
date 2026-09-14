"""Layer 0.1 — Spike: RunPod pod creation + SSH.

GO:   SSH succeeds, nvidia-smi shows RTX 4090 with 24 GB.
NO-GO: pod creation fails or SSH unreachable → investigate RunPod docs.

Usage:
    RUNPOD_API_KEY=... python tests/spikes/spike_runpod_ssh.py
"""

from __future__ import annotations

import json
import os
import sys
import time

import httpx

GRAPHQL_URL = "https://api.runpod.io/graphql"
GPU_TYPE = "NVIDIA GeForce RTX 4090"
IMAGE = "vllm/vllm-openai:v0.29.0"


def _gql(api_key: str, query: str, variables: dict | None = None) -> dict:
    resp = httpx.post(
        GRAPHQL_URL,
        json={"query": query, "variables": variables or {}},
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    if "errors" in data:
        raise RuntimeError(f"GraphQL errors: {json.dumps(data['errors'], indent=2)}")
    return data["data"]


def create_pod(api_key: str) -> str:
    query = """
    mutation {{
      podFindAndDeployOnDemand(input: {{
        name: "apron-spike"
        imageName: "{image}"
        gpuTypeId: "{gpu}"
        cloudType: SECURE
        volumeInGb: 20
        containerDiskInGb: 20
        ports: "22/tcp,8000/http"
        gpuCount: 1
      }}) {{
        id
        desiredStatus
        imageName
        runtime {{ ports {{ ip privatePort publicPort type }} }}
      }}
    }}
    """.format(image=IMAGE, gpu=GPU_TYPE)
    data = _gql(api_key, query)
    pod = data["podFindAndDeployOnDemand"]
    print(f"Pod created: {pod['id']}")
    return pod["id"]


def wait_for_running(api_key: str, pod_id: str, timeout: int = 300) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        query = f"""
        query {{
          pod(input: {{ podId: "{pod_id}" }}) {{
            id
            desiredStatus
            runtime {{
              uptimeInSeconds
              ports {{ ip privatePort publicPort type }}
              gpus {{ id gpuUtilPercent memoryUtilPercent }}
            }}
          }}
        }}
        """
        data = _gql(api_key, query)
        pod = data["pod"]
        runtime = pod.get("runtime")
        if runtime and runtime.get("uptimeInSeconds", 0) > 0:
            print(f"Pod RUNNING after {int(time.monotonic() - (deadline - timeout))}s")
            return pod
        print("  waiting for pod to start...")
        time.sleep(10)
    raise TimeoutError(f"Pod {pod_id} did not reach RUNNING within {timeout}s")


def ssh_nvidia_smi(pod: dict) -> str:
    import paramiko

    ports = pod["runtime"]["ports"]
    ssh_port_info = next((p for p in ports if p["privatePort"] == 22), None)
    if ssh_port_info is None:
        raise RuntimeError("No SSH port mapping found")

    host = ssh_port_info["ip"]
    port = ssh_port_info["publicPort"]
    print(f"SSH connecting to {host}:{port}")

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh_key_path = os.environ.get(
        "RUNPOD_SSH_KEY_PATH", os.path.expanduser("~/.ssh/id_ed25519")
    )
    client.connect(host, port=int(port), username="root", key_filename=ssh_key_path)

    _, stdout, _ = client.exec_command("nvidia-smi")
    output = stdout.read().decode()
    client.close()
    return output


def terminate_pod(api_key: str, pod_id: str) -> None:
    query = f"""
    mutation {{
      podTerminate(input: {{ podId: "{pod_id}" }})
    }}
    """
    _gql(api_key, query)
    print(f"Pod {pod_id} terminated")


def main() -> None:
    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        print("RUNPOD_API_KEY not set — spike cannot run")
        sys.exit(1)

    pod_id = None
    try:
        pod_id = create_pod(api_key)
        pod = wait_for_running(api_key, pod_id)
        output = ssh_nvidia_smi(pod)
        print("\n--- nvidia-smi output ---")
        print(output)

        if "RTX 4090" in output and "24" in output:
            print("\nGO: RTX 4090 detected with ~24 GB")
        else:
            print("\nNO-GO: unexpected GPU or memory")
            sys.exit(1)
    finally:
        if pod_id:
            terminate_pod(api_key, pod_id)


if __name__ == "__main__":
    main()
