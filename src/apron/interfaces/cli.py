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
    task_suite_path: Path | None = None,
    prediction: dict[str, Any] | None = None,
) -> None:
    """Run the verify/deploy lifecycle on a RunPod target.

    Steps: prepare → provision → validate → boot vLLM → profile memory →
    run task suite (if provided) → show prediction delta → store records.
    """
    from apron.adapters.backends.runpod import RunPodTarget
    from apron.adapters.backends.vllm_engine import VllmEngineAdapter
    from apron.domain.schemas.solutions import DeploymentPlan

    target = RunPodTarget(max_uptime=timeout)

    console.print("[bold]Step 1/6: Checking target availability[/bold]")
    prep = target.prepare()
    if prep["status"] == "hardware_unavailable":
        console.print(f"[yellow]hardware_unavailable: {prep['reason']}[/yellow]")
        return
    console.print(f"  RunPod API: authenticated (user {prep.get('user_id', '?')})")

    plan = DeploymentPlan(**plan_data)
    engine = VllmEngineAdapter()

    success = False
    try:
        console.print("[bold]Step 2/6: Provisioning GPU pod[/bold]")
        prov = target.provision()
        console.print(f"  Pod ID:             {prov['pod_id']}")
        console.print(f"  GPU:                {target.hardware.gpu_sku}")
        console.print(f"  GPU memory:         {target.hardware.total_memory_bytes / 1e9:.1f} GB")
        console.print(f"  Compute capability: {target.hardware.compute_capability}")
        console.print(f"  Fingerprint:        {target.execution_fingerprint[:24]}...")

        console.print("[bold]Step 3/6: Validating plan against target[/bold]")
        validation_errors = engine.validate(plan, target)
        if validation_errors:
            for err in validation_errors:
                console.print(f"  [red]FAIL: {err}[/red]")
            return
        console.print("  [green]Validation passed[/green]")

        console.print("[bold]Step 4/6: Booting vLLM and profiling memory[/bold]")
        report = engine.verify(plan, target)

        store = _default_store()
        report_record = {
            **report,
            "claim_scope": "memory",
            "production_mode": False,
            "reason": "verification",
            "lifecycle": "observed",
        }
        digest = store.store(report_record)

        _print_memory_profile(report, prediction)
        console.print(f"  Record: {digest[:24]}...")

        if task_suite_path is not None and task_suite_path.exists():
            console.print("[bold]Step 5/6: Running task suite[/bold]")
            _run_task_suite(task_suite_path, plan_data, target, store)
        else:
            console.print("[dim]Step 5/6: Task suite — skipped (no --task-suite)[/dim]")

        console.print("[bold]Step 6/6: Complete[/bold]")
        success = True
        if not teardown:
            console.print(f"  [green]Endpoint: {target.proxy_url}[/green]")
            console.print("  [yellow]Pod left running — teardown manually or via apron[/yellow]")
    except Exception as exc:
        console.print(f"[red]Verification failed: {exc}[/red]")
        raise typer.Exit(1) from None
    finally:
        if teardown or not success:
            console.print("[bold]Tearing down...[/bold]")
            target.teardown()
            console.print("[green]Teardown complete[/green]")


def _print_memory_profile(
    report: dict[str, Any], prediction: dict[str, Any] | None = None
) -> None:
    """Print a memory profile table, optionally with prediction comparison."""
    table = Table(title="Memory Profile (measured)")
    table.add_column("Component", style="cyan")
    table.add_column("Measured", justify="right")
    if prediction:
        table.add_column("Predicted", justify="right")
        table.add_column("Delta", justify="right")

    rows = [
        ("model_weight_memory", "weight_memory_bytes"),
        ("transient_peak_headroom", "activation_estimate_bytes"),
        ("non_pytorch_increase", "non_pytorch_overhead_bytes"),
        ("cuda_graph_actual", "cuda_graph_estimate_bytes"),
        ("available_kv_cache_memory", "available_kv_cache_bytes"),
        ("persistent_consumption", "total_required_bytes"),
    ]

    for measured_key, predicted_key in rows:
        measured_val = report.get(measured_key, 0)
        measured_str = f"{measured_val / 1e9:.2f} GB"

        if prediction:
            predicted_val = prediction.get(predicted_key, 0)
            delta = measured_val - predicted_val
            sign = "+" if delta >= 0 else ""
            table.add_row(
                measured_key,
                measured_str,
                f"{predicted_val / 1e9:.2f} GB",
                f"{sign}{delta / 1e9:.2f} GB",
            )
        else:
            table.add_row(measured_key, measured_str)

    console.print(table)


def _run_task_suite(
    task_suite_path: Path,
    plan_data: dict[str, Any],
    target: Any,
    store: Any,
) -> None:
    """Execute the task suite against the running vLLM endpoint."""
    from apron.adapters.evaluations.deterministic_scorer import DeterministicScorer

    task_suite = json.loads(task_suite_path.read_text())
    scorer = DeterministicScorer()

    if not scorer.accepts(task_suite):
        console.print("  [yellow]Scorer does not accept this task suite[/yellow]")
        return

    endpoint = getattr(target, "proxy_url", None)
    if endpoint is None:
        console.print("  [yellow]No endpoint URL available — skipping[/yellow]")
        return

    model_id = plan_data.get("resource_allocation", {}).get("model_id", "")
    protocol = scorer.prepare({**task_suite, "model_id": model_id, "endpoint": endpoint})
    attempts = scorer.execute(protocol, endpoint)
    results = scorer.collect(attempts)

    console.print(f"  Cases: {results['total_attempts']}")
    console.print(f"  Passed: {results['passed']}")
    console.print(f"  Failed: {results['failed']}")
    console.print(f"  Pass rate: {results['pass_rate']:.0%}")

    for attempt in attempts:
        status = "[green]PASS[/green]" if attempt.get("score") == 1 else "[red]FAIL[/red]"
        console.print(f"    {attempt.get('case_id', '?')}: {status}")
        if attempt.get("status") == "failed":
            console.print(f"      error: {attempt.get('error', '')}")


# ---------------------------------------------------------------------------
# apron verify
# ---------------------------------------------------------------------------


@app.command()
def verify(
    plan_file: Path = typer.Argument(..., help="DeploymentPlan JSON file"),
    target_spec: str = typer.Option("runpod-4090", help="Target spec"),
    task_suite: Path | None = typer.Option(None, help="Task suite JSON file"),
    prediction_file: Path | None = typer.Option(None, help="PlanningClaim JSON for delta"),
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
    if task_suite:
        console.print(f"  Task suite: {task_suite}")

    if not yes:
        typer.confirm("Proceed with verification?", abort=True)

    prediction = None
    if prediction_file and prediction_file.exists():
        claim = json.loads(prediction_file.read_text())
        prediction = claim.get("proposed_configuration", claim)

    _run_verification(
        plan_data,
        teardown=True,
        timeout=timeout,
        task_suite_path=task_suite,
        prediction=prediction,
    )


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

    from apron.adapters.backends.vllm_engine import VllmEngineAdapter

    engine = VllmEngineAdapter()

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    assert process.stdout is not None
    for line in process.stdout:
        sys.stdout.write(line)
        classification = engine.classify(line)
        if classification["failure_class"] != "unknown":
            console.print(f"[red]Detected: {classification['failure_class']}[/red]")

    process.wait()
    if process.returncode != 0:
        console.print(f"[red]Process exited with code {process.returncode}[/red]")
        raise typer.Exit(process.returncode)


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
