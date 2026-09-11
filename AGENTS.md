# Agent instructions for Apron

## Check command

Run before every commit:

```
prek run --all-files
```

This executes every check CI will run. A commit that fails it will fail the pull request.

## Commit messages

Conventional Commits: `type(scope): subject`

Allowed types: `feat`, `fix`, `refactor`, `test`, `docs`, `chore`, `ci`, `perf`

The subject is imperative, lowercase, no trailing period.

**Forbidden in commit messages:** agent branding, session links, tool-named
`Co-authored-by` trailers, `generated-by` markers, prompt fragments. The
history describes the project, not the tool that wrote it.

## Source layout

```
src/apron/
  domain/        schemas, mechanisms, capabilities, calculator, fingerprints,
                 authorization combination, evidence-promotion policy — no I/O
  application/   qualification graph, orchestration, signals, metrics
  adapters/      engine adapters, artifact resolvers, renderers, evidence
                 sources, planning sources, evaluations, backends, publishers
  interfaces/    cli, mcp, api — composition roots only
```

Layer direction: domain ← application ← adapters ← interfaces. Checked by
import-linter in CI. No layer may import from a layer to its right.

## Scope boundaries

### Contribution-ready (no prior approval needed)

- Adapter against a conformance suite (run the suite in your own repo first)
- Diagnosis rule as `hypothesis` in `rules/` with a source observation
- Bug fix with a failing test
- Documentation correction

### Requires an approved issue or RFC first

- New extension-point implementation
- Schema change
- Any change over ~500 lines
- Architecture decision amendment (propose in Discussions)

## Key invariants for agents

- **INV-11:** No recommendation, calculation, ranking or promotion logic in
  interfaces. Interfaces are composition roots only.
- **INV-42:** Wall-clock time, identifiers and randomness reach domain only
  through injected `Clock`, `IdGenerator` and `Rng` ports.
- **INV-43:** Records serialize to RFC 8785 canonical form before hashing.
  Identity is SHA-256 with multihash prefix.

## MCP server

```
apron mcp
```

Claude Code configuration:

```json
{"command": "apron", "args": ["mcp"]}
```

## Running tests

```
pytest tests/
```

For a specific check by name:

```
prek run <hook-id> --all-files
```

## What not to do

- Do not add vendor MCP configuration files (`.mcp.json`, `.cursor/mcp.json`,
  `.vscode/mcp.json`, `.codex/config.toml`). None is committed.
- Do not add `llms.txt`, `.devcontainer`, `.cursorrules` or `.specify/`.
- Do not commit secrets, credentials or API tokens. Tokens come from
  environment or OS keyring only.
- Do not open pull requests or issues autonomously without a human principal.
  See `AI_POLICY.md`.
