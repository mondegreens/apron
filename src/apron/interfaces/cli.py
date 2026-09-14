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
from apron.application.orchestration.plan_builder import build_plan
from apron.application.orchestration.resolution import ResolutionChain
from apron.domain.mechanisms.model_spec_builder import build_model_spec
from apron.domain.ports import UuidIdGenerator, WallClock
from apron.domain.schemas.primitives import ArtifactLocator, HardwareSpec
from apron.domain.schemas.solutions import RenderContext

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

    json.loads(request_file.read_text())  # validate JSON

    resolver = FixtureHFHubResolver(fixture_dir) if fixture_dir is not None else HFHubResolver()

    chain = ResolutionChain(resolver)
    result = chain.resolve(model)

    if not result.ok:
        console.print(f"[red]Resolution failed: {result.error}[/red]")
        raise typer.Exit(1)

    assert result.observation is not None
    config_content = resolver._download_file(
        model, "config.json", result.observation.resolved_revision
    )
    if config_content is None:
        console.print("[red]Could not download config.json[/red]")
        raise typer.Exit(1)

    config = json.loads(config_content)

    index_content = resolver._download_file(
        model, "model.safetensors.index.json", result.observation.resolved_revision
    )
    total_weight_bytes = 0
    if index_content is not None:
        index_data = json.loads(index_content)
        total_weight_bytes = index_data.get("metadata", {}).get("total_size", 0)

    safetensors_params = result.observation.publisher_metadata or {}
    if total_weight_bytes == 0:
        for key, val in safetensors_params.items():
            if key.startswith("parameters_"):
                total_weight_bytes = int(val) * 2
                break

    model_spec = build_model_spec(
        config,
        repository=model,
        revision=result.observation.resolved_revision,
        license_id=result.observation.license_observed,
    )

    hw = _DEFAULT_HARDWARE
    if target is not None and target.exists():
        hw = HardwareSpec(**json.loads(target.read_text()))

    config["total_weight_bytes"] = total_weight_bytes
    config["components"] = list(model_spec.components)
    config["artifact_metadata"] = config

    from apron.adapters.planning.calculator_source import CalculatorPlanningSource

    planning_source = CalculatorPlanningSource(clock=clock)
    claim = planning_source.predict(config, hw, {"isl": 512, "osl": 128, "max_batch_size": 4})

    deployment_plan = build_plan(
        claim,
        model_spec,
        hw,
        result.execution_spec,
        None,
        clock=clock,
        id_gen=id_gen,
    )

    locator = ArtifactLocator(source_kind="huggingface", uri=model)
    ctx = RenderContext(plan=deployment_plan, locator=locator, hardware=hw)

    output_dir.mkdir(parents=True, exist_ok=True)
    plan_path = output_dir / "deployment-plan.json"
    plan_path.write_text(json.dumps(deployment_plan.model_dump(mode="json"), indent=2))

    rendered_outputs: dict[str, Any] = {}
    for renderer in _build_renderers():
        rendered = renderer.render(ctx)
        rendered_outputs[renderer.target_format] = rendered

    renders_path = output_dir / "rendered-outputs.json"
    renders_path.write_text(json.dumps(rendered_outputs, indent=2))

    console.print(f"[green]Plan saved to {plan_path}[/green]")

    if verbose:
        _print_breakdown(claim)


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

    console.print("[yellow]GPU execution not available in Phase 1a-ii[/yellow]")


# ---------------------------------------------------------------------------
# apron deploy
# ---------------------------------------------------------------------------


@app.command()
def deploy(
    plan_file: Path = typer.Argument(..., help="DeploymentPlan JSON file"),
    target_spec: str = typer.Option("runpod-4090", help="Target spec"),
    yes: bool = typer.Option(False, help="Standing authorization"),
) -> None:
    """Deploy a model using a deployment plan (teardown=False)."""
    if not plan_file.exists():
        console.print(f"[red]Plan file not found: {plan_file}[/red]")
        raise typer.Exit(1)

    if not yes:
        typer.confirm("Proceed with deployment?", abort=True)

    console.print("[yellow]GPU execution not available in Phase 1a-ii[/yellow]")


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
