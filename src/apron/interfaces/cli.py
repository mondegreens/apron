"""CLI composition root — wires adapters and exposes commands.

Import-linter allows interfaces/ → everything. This is the only place
that imports concrete adapters. Phase 1a: hardwired adapter instances.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.table import Table

import apron.domain.mechanisms.calculator  # noqa: F401 — registers calculators
from apron.adapters.backends.local_store import LocalRecordStore
from apron.adapters.evidence.hf_hub import FixtureHFHubResolver, HFHubResolver
from apron.adapters.renderers.aiconfigurator_export import AIConfiguratorExportRenderer
from apron.adapters.renderers.docker_compose import DockerComposeRenderer
from apron.adapters.renderers.inferencex_export import InferenceXExportRenderer
from apron.adapters.renderers.recipes_export import RecipesExportRenderer
from apron.adapters.renderers.vllm_serve import VllmServeRenderer
from apron.application.orchestration.plan_pipeline import run_plan_pipeline
from apron.domain.ports import UuidIdGenerator, WallClock
from apron.domain.schemas.primitives import HardwareSpec

app = typer.Typer(name="apron", no_args_is_help=True)
console = Console()

_DEFAULT_HARDWARE = HardwareSpec(
    gpu_sku="RTX 4090",
    total_memory_bytes=25_769_803_776,
    compute_capability="8.9",
)


def _build_renderers() -> list[Any]:
    return [
        VllmServeRenderer(),
        DockerComposeRenderer(),
        RecipesExportRenderer(),
        AIConfiguratorExportRenderer(),
        InferenceXExportRenderer(),
    ]


def _default_store() -> LocalRecordStore:
    store_dir = Path.home() / ".apron" / "records"
    store_dir.mkdir(parents=True, exist_ok=True)
    return LocalRecordStore(store_dir)


# ---------------------------------------------------------------------------
# apron plan
# ---------------------------------------------------------------------------


@app.command()
def plan(
    request_file: Path = typer.Argument(..., help="DecisionRequest JSON file"),
    model: str = typer.Option(..., help="Model ID (e.g. Qwen/Qwen3-8B)"),
    target: Path | None = typer.Option(None, help="HardwareSpec JSON file"),
    output_dir: Path = typer.Option(Path("."), help="Output directory"),
    verbose: bool = typer.Option(False, help="Show prediction breakdown"),
    fixture_dir: Path | None = typer.Option(None, hidden=True, help="Use fixture resolver"),
) -> None:
    """Generate a deployment plan for a model."""
    if not request_file.exists():
        console.print(f"[red]Request file not found: {request_file}[/red]")
        raise typer.Exit(1)

    clock = WallClock()
    id_gen = UuidIdGenerator()

    try:
        json.loads(request_file.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        console.print(f"[red]Invalid request file: {exc}[/red]")
        raise typer.Exit(1) from None

    resolver = FixtureHFHubResolver(fixture_dir) if fixture_dir is not None else HFHubResolver()

    hw = _DEFAULT_HARDWARE
    if target is not None and target.exists():
        try:
            hw = HardwareSpec(**json.loads(target.read_text()))
        except (json.JSONDecodeError, OSError, ValueError) as exc:
            console.print(f"[red]Invalid target spec: {exc}[/red]")
            raise typer.Exit(1) from None

    from apron.adapters.planning.calculator_source import CalculatorPlanningSource

    planning_source = CalculatorPlanningSource(clock=clock)

    try:
        pipeline_result = run_plan_pipeline(
            resolver, planning_source, model, hw, clock=clock, id_gen=id_gen
        )
    except Exception as exc:
        console.print(f"[red]Plan pipeline error: {exc}[/red]")
        raise typer.Exit(1) from None

    if not pipeline_result.ok:
        console.print(f"[red]Plan failed: {pipeline_result.error}[/red]")
        raise typer.Exit(1)

    assert pipeline_result.plan is not None
    assert pipeline_result.context is not None

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        console.print(f"[red]Cannot create output directory: {exc}[/red]")
        raise typer.Exit(1) from None
    plan_path = output_dir / "deployment-plan.json"
    plan_path.write_text(json.dumps(pipeline_result.plan.model_dump(mode="json"), indent=2))

    rendered_outputs: dict[str, Any] = {}
    for renderer in _build_renderers():
        rendered = renderer.render(pipeline_result.context)
        rendered_outputs[renderer.target_format] = rendered

    renders_path = output_dir / "rendered-outputs.json"
    renders_path.write_text(json.dumps(rendered_outputs, indent=2))

    console.print(f"[green]Plan saved to {plan_path}[/green]")

    if verbose:
        _print_breakdown(pipeline_result.claim)


def _print_breakdown(claim: Any) -> None:
    config = claim.proposed_configuration
    if config.get("status") == "unknown":
        console.print("[yellow]Unknown mechanism — no breakdown available[/yellow]")
        return

    table = Table(title="Memory Prediction Breakdown")
    table.add_column("Component", style="cyan")
    table.add_column("Bytes", justify="right")
    table.add_column("GB", justify="right")

    for key in (
        "weight_memory_bytes",
        "kv_cache_bytes",
        "activation_estimate_bytes",
        "non_pytorch_overhead_bytes",
        "cuda_graph_estimate_bytes",
        "total_required_bytes",
        "available_kv_cache_bytes",
    ):
        val = config.get(key, 0)
        table.add_row(key, f"{val:,}", f"{val / 1e9:.2f}")

    console.print(table)


# ---------------------------------------------------------------------------
# Verification / deployment orchestration
# ---------------------------------------------------------------------------


def _run_verification(
    plan_data: dict[str, Any],
    *,
    teardown: bool = True,
    timeout: int = 3600,
) -> None:
    """Run the verify/deploy lifecycle on a RunPod target."""
    from apron.adapters.backends.runpod import RunPodTarget
    from apron.adapters.backends.vllm_engine import VllmEngineAdapter
    from apron.domain.schemas.solutions import DeploymentPlan

    target = RunPodTarget(max_uptime=timeout)

    prep = target.prepare()
    if prep["status"] == "hardware_unavailable":
        console.print(f"[yellow]hardware_unavailable: {prep['reason']}[/yellow]")
        return

    plan = DeploymentPlan(**plan_data)
    engine = VllmEngineAdapter()

    success = False
    try:
        console.print("[bold]Provisioning target...[/bold]")
        prov = target.provision()
        console.print(f"  Pod: {prov['pod_id']}")
        console.print(f"  GPU: {target.hardware.gpu_sku}")
        console.print(f"  Memory: {target.hardware.total_memory_bytes / 1e9:.1f} GB")

        validation_errors = engine.validate(plan, target)
        if validation_errors:
            for err in validation_errors:
                console.print(f"[red]Validation: {err}[/red]")
            return

        console.print("[bold]Booting vLLM and profiling memory...[/bold]")
        report = engine.verify(plan, target)

        store = _default_store()
        report_record = {
            **report,
            "claim_scope": "memory",
            "production_mode": False,
            "reason": "Phase 1a verification",
            "lifecycle": "observed",
        }
        digest = store.store(report_record)
        console.print(f"  VerificationReport stored: {digest[:16]}...")

        console.print("[bold]Memory profile:[/bold]")
        for key in (
            "initial_total_memory",
            "model_weight_memory",
            "persistent_consumption",
            "available_kv_cache_memory",
        ):
            val = report.get(key, 0)
            console.print(f"  {key}: {val / 1e9:.2f} GB")

        success = True
        if not teardown:
            console.print(f"\n[green]Endpoint: {target.proxy_url}[/green]")
            console.print("[yellow]Pod left running — teardown manually or via apron[/yellow]")
    except Exception as exc:
        console.print(f"[red]Verification failed: {exc}[/red]")
        raise typer.Exit(1) from None
    finally:
        if teardown or not success:
            console.print("[bold]Tearing down...[/bold]")
            target.teardown()
            console.print("[green]Teardown complete[/green]")


# ---------------------------------------------------------------------------
# apron verify
# ---------------------------------------------------------------------------


@app.command()
def verify(
    plan_file: Path = typer.Argument(..., help="DeploymentPlan JSON file"),
    target_spec: str = typer.Option("runpod-4090", help="Target spec"),
    yes: bool = typer.Option(False, help="Standing authorization"),
    timeout: int = typer.Option(3600, help="Hard deadline seconds"),
) -> None:
    """Verify a deployment plan on a target environment."""
    if not plan_file.exists():
        console.print(f"[red]Plan file not found: {plan_file}[/red]")
        raise typer.Exit(1)

    plan_data = json.loads(plan_file.read_text())

    from apron.application.cost_estimator import estimate_cost

    cost = estimate_cost(plan_data, "runpod")

    console.print("[bold]Verification Summary[/bold]")
    console.print("  Data destination: ~/.apron/records/ (local)")
    console.print(f"  Estimated cost: ${cost['estimated_cost_usd']:.2f}")
    console.print(f"  Hard deadline: {timeout}s")

    if not yes:
        typer.confirm("Proceed with verification?", abort=True)

    _run_verification(plan_data, teardown=True, timeout=timeout)


# ---------------------------------------------------------------------------
# apron deploy
# ---------------------------------------------------------------------------


@app.command()
def deploy(
    plan_file: Path = typer.Argument(..., help="DeploymentPlan JSON file"),
    target_spec: str = typer.Option("runpod-4090", help="Target spec"),
    yes: bool = typer.Option(False, help="Standing authorization"),
    timeout: int = typer.Option(3600, help="Hard deadline seconds"),
) -> None:
    """Deploy a model using a deployment plan (teardown=False)."""
    if not plan_file.exists():
        console.print(f"[red]Plan file not found: {plan_file}[/red]")
        raise typer.Exit(1)

    plan_data = json.loads(plan_file.read_text())

    from apron.application.cost_estimator import estimate_cost

    cost = estimate_cost(plan_data, "runpod")

    console.print("[bold]Deployment Summary[/bold]")
    console.print("  Data destination: ~/.apron/records/ (local)")
    console.print(f"  Estimated cost: ${cost['estimated_cost_usd']:.2f}/hr ongoing")
    console.print(f"  Hard deadline: {timeout}s")

    if not yes:
        typer.confirm("Proceed with deployment?", abort=True)

    _run_verification(plan_data, teardown=False, timeout=timeout)


# ---------------------------------------------------------------------------
# apron run
# ---------------------------------------------------------------------------


@app.command()
def run(command: list[str]) -> None:
    """Run a command with output capture and error classification."""
    if not command:
        console.print("[red]No command provided[/red]")
        raise typer.Exit(1)

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    assert process.stdout is not None
    for line in process.stdout:
        sys.stdout.write(line)
        _classify_line(line)

    process.wait()
    if process.returncode != 0:
        console.print(f"[red]Process exited with code {process.returncode}[/red]")
        raise typer.Exit(process.returncode)


def _classify_line(line: str) -> None:
    if "torch.OutOfMemoryError" in line or "CUDA out of memory" in line:
        console.print("[red]Detected: OOM error — reduce batch size or model size[/red]")
    elif "RuntimeError" in line and "engine" in line.lower():
        console.print("[red]Detected: engine init error[/red]")
    elif "ValueError" in line and "max_model_len" in line:
        console.print("[red]Detected: max_model_len mismatch — check model config[/red]")


# ---------------------------------------------------------------------------
# apron report
# ---------------------------------------------------------------------------


@app.command()
def report(
    record_id: str = typer.Argument(..., help="Record digest or prefix"),
    format: str = typer.Option("json", help="Output format"),
) -> None:
    """Retrieve and display a stored record."""
    store = _default_store()
    record = store.retrieve(record_id)
    if record is None:
        console.print(f"[red]Record not found: {record_id}[/red]")
        raise typer.Exit(1)

    if format == "json":
        console.print_json(json.dumps(record, indent=2))
    else:
        console.print(record)


# ---------------------------------------------------------------------------
# apron submit
# ---------------------------------------------------------------------------


@app.command()
def submit(
    record_id: str = typer.Argument(..., help="Record digest or prefix"),
    destination: str = typer.Option(..., help="Publication destination"),
) -> None:
    """Submit a record to an external destination."""
    store = _default_store()
    record = store.retrieve(record_id)
    if record is None:
        console.print(f"[red]Record not found: {record_id}[/red]")
        raise typer.Exit(1)

    from apron.application.sanitization import sanitize, validate_provenance

    sanitized = sanitize(record)
    errors = validate_provenance(sanitized, store)
    if errors:
        console.print("[yellow]Provenance warnings:[/yellow]")
        for err in errors:
            console.print(f"  - {err}")

    console.print(f"[bold]Will submit to: {destination}[/bold]")
    console.print_json(json.dumps(sanitized, indent=2))
    typer.confirm("Proceed with submission?", abort=True)
    console.print("[yellow]Publication not implemented in Phase 1a[/yellow]")


# ---------------------------------------------------------------------------
# apron mcp
# ---------------------------------------------------------------------------


@app.command("mcp")
def mcp_server() -> None:
    """Start MCP server on stdio."""
    import asyncio

    from mcp.server import Server  # type: ignore[import-untyped]
    from mcp.server.stdio import stdio_server  # type: ignore[import-untyped]
    from mcp.types import TextContent, Tool  # type: ignore[import-untyped]

    server = Server("apron")

    @server.list_tools()  # type: ignore[misc]
    async def list_tools() -> list[Tool]:
        plan_schema: dict[str, Any] = {
            "type": "object",
            "properties": {
                "request_file": {"type": "string"},
                "model": {"type": "string"},
                "target": {"type": "string"},
            },
            "required": ["request_file", "model"],
        }
        report_schema: dict[str, Any] = {
            "type": "object",
            "properties": {
                "record_id": {"type": "string"},
                "format": {"type": "string"},
            },
            "required": ["record_id"],
        }
        return [
            Tool(
                name="apron_plan",
                description="Generate a deployment plan",
                inputSchema=plan_schema,  # pyright: ignore[reportCallIssue]
            ),
            Tool(
                name="apron_report",
                description="Retrieve a stored record",
                inputSchema=report_schema,  # pyright: ignore[reportCallIssue]
            ),
        ]

    @server.call_tool()  # type: ignore[misc]
    async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
        if name == "apron_report":
            store = _default_store()
            record = store.retrieve(arguments["record_id"])
            if record is None:
                return [TextContent(type="text", text=json.dumps({"error": "Record not found"}))]
            return [TextContent(type="text", text=json.dumps(record, indent=2))]
        return [TextContent(type="text", text=json.dumps({"error": f"Unknown tool: {name}"}))]

    async def _run() -> None:
        async with stdio_server() as (read, write):
            await server.run(read, write, server.create_initialization_options())

    asyncio.run(_run())
