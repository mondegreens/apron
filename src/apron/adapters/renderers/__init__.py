"""Export renderers — map DeploymentPlan to external ecosystem shapes.

Each renderer implements the RenderTarget Protocol.  The shared-field
table defines which DeploymentPlan fields map to which external fields.
Fields outside the table are format-specific and dropped on export.

Three ecosystem shapes (ADR-002 §4):
- vllm-recipes YAML
- aiconfigurator estimate request JSON
- InferenceX throughput result row
"""

from typing import Any

from apron.domain.schemas.solutions import DeploymentPlan

RECIPES_SHARED_FIELDS = {
    "tensor_parallel": {"external": "tp", "direction": "bidirectional", "conversion": "identity"},
    "dtype": {"external": "precision", "direction": "bidirectional", "conversion": "identity"},
    "batch_size": {
        "external": "max_num_seqs",
        "direction": "bidirectional",
        "conversion": "identity",
    },
    "engine_configuration": {
        "external": "base_args",
        "direction": "export_only",
        "conversion": "identity",
    },
}

AICONFIGURATOR_SHARED_FIELDS = {
    "tensor_parallel": {
        "external": "topology.tensor_parallel_size",
        "direction": "bidirectional",
        "conversion": "identity",
    },
    "pipeline_parallel": {
        "external": "topology.pipeline_parallel_size",
        "direction": "bidirectional",
        "conversion": "identity",
    },
    "dtype": {
        "external": "quantization.dtype",
        "direction": "bidirectional",
        "conversion": "identity",
    },
    "batch_size": {
        "external": "workload.batch_size",
        "direction": "bidirectional",
        "conversion": "identity",
    },
}

INFERENCEX_SHARED_FIELDS = {
    "tensor_parallel": {"external": "tp", "direction": "bidirectional", "conversion": "identity"},
    "pipeline_parallel": {
        "external": "pp",
        "direction": "bidirectional",
        "conversion": "identity",
    },
    "expert_parallel": {"external": "ep", "direction": "bidirectional", "conversion": "identity"},
    "data_parallel": {"external": "dp", "direction": "bidirectional", "conversion": "identity"},
    "dtype": {"external": "precision", "direction": "bidirectional", "conversion": "identity"},
}


def _extract_shared(plan: DeploymentPlan, table: dict[str, dict[str, str]]) -> dict[str, Any]:
    dumped = plan.model_dump(mode="json")
    result = {}
    for apron_field, mapping in table.items():
        if apron_field in dumped and dumped[apron_field] is not None:
            result[mapping["external"]] = dumped[apron_field]
    return result


def export_to_recipes(plan: DeploymentPlan) -> dict[str, Any]:
    return _extract_shared(plan, RECIPES_SHARED_FIELDS)


def export_to_aiconfigurator(plan: DeploymentPlan) -> dict[str, Any]:
    return _extract_shared(plan, AICONFIGURATOR_SHARED_FIELDS)


def export_to_inferencex(plan: DeploymentPlan) -> dict[str, Any]:
    return _extract_shared(plan, INFERENCEX_SHARED_FIELDS)


def import_from_recipes(data: dict[str, Any]) -> dict[str, Any]:
    reverse = {
        v["external"]: k
        for k, v in RECIPES_SHARED_FIELDS.items()
        if v["direction"] != "export_only"
    }
    return {reverse[k]: v for k, v in data.items() if k in reverse}


def import_from_aiconfigurator(data: dict[str, Any]) -> dict[str, Any]:
    reverse = {
        v["external"]: k
        for k, v in AICONFIGURATOR_SHARED_FIELDS.items()
        if v["direction"] != "export_only"
    }
    return {reverse[k]: v for k, v in data.items() if k in reverse}


def import_from_inferencex(data: dict[str, Any]) -> dict[str, Any]:
    reverse = {
        v["external"]: k
        for k, v in INFERENCEX_SHARED_FIELDS.items()
        if v["direction"] != "export_only"
    }
    return {reverse[k]: v for k, v in data.items() if k in reverse}
