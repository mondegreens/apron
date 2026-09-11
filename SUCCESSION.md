# Succession and continuation path

This records what publishing assets exist, how releases happen, what a successor needs, and which single points of failure remain unmitigated.

## What exists

| asset | state |
|---|---|
| Package index distributions | `apron` and `apron-conformance`, both reserved at 0.0.0 |
| Documentation domain | `apron.dev`, registered |
| GitHub organization | `github.com/mondegreens` |
| Public repositories | `mondegreens/apron`, `mondegreens/.github` |
| Private repository | `mondegreens/apron-research` (separate history) |

## How releases happen

Releases do not require a person holding a credential. The release workflow authenticates to the package index with a short-lived identity token issued by GitHub at run time, exchanges it for an upload token valid for minutes, and publishes. No long-lived secret exists in the repository, in GitHub's secret store, or on anyone's machine.

**The right to publish belongs to the repository, not to a person.** Anyone who can merge to the default branch and push a tag can cut a release. Nothing has to be handed over.

**Nothing can be published from a laptop.** Every artifact on the index provably came from a tagged commit through continuous integration.

The trusted publisher configuration names the repository (`mondegreens/apron`), the workflow file (`release.yml`) and the environment (`pypi`). Both distributions point at the same repository and workflow.

## What a successor needs

1. **Nothing, to continue development.** The specification, the architecture and the evidence are open, and the code will be. A fork can proceed without any credential.

2. **Merge rights on the public repository**, to publish under the existing distribution names. That is the whole of it, once trusted publishing is configured.

3. **The package index account**, only to change who has merge rights, to transfer the projects, or to add a maintainer. This is the sole irreplaceable item.

## Single points of failure

The package index account has one holder. This is not a lockout risk — the account is protected by multiple independent security keys and multiple verified addresses — but an availability risk: work cannot continue through this account if its owner does not.

Trusted publishing has removed releasing from the account entirely, so what the owner account uniquely controls is now narrow: adding or removing maintainers, changing the publisher configuration, deleting releases, and transferring ownership. A lost account would not stop releases; it would stop changes to who may release.

The real mitigation is a second person, which does not exist yet. The package index permits multiple maintainers on a personally owned project. This waits on a second person rather than on a decision.
