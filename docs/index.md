# Apron

Decision and evidence system for LLM inference deployment.

Apron connects model choice and serving configuration with evidence scoped to the artifact, engine, hardware, and workload that produced it.

## Start here

- [Inspect measured deployment records](https://github.com/mondegreens/apron#measured-records): predictions and observations, with links to the stored records.
- [Read the implementation](https://github.com/mondegreens/apron/tree/main/src) and [tests](https://github.com/mondegreens/apron/tree/main/tests).
- [Read the current project status](https://github.com/mondegreens/apron#overview): the repository README describes implemented phases and current work.
- [Explore findings](blog/index.md): technical notes linked to evidence.

## Project stage

Early implementation. The repository contains the implementation, specification, and published measurement records. The PyPI package names are placeholders and are not the installation route for the current code.

A memory prediction, a healthy engine boot, and a serving result establish different things. Records keep their own scope and provenance; support for one environment does not establish support for another.

## Design and contracts

Start with the [product definition](product-definition.md), then the [target architecture](target-architecture.md) and [phase plan](phase-plan.md). The [framework specification](framework-spec.md) defines the extension points, invariants, and metrics. The [architecture decision records](adr/ADR-001-execution-environment.md) record design decisions and their consequences.

For contributions, read [how development works](development-process.md).

Maintained by [Vlad Ryzhkov](https://github.com/data-mission).
