# AI contribution policy

This is a scope-and-authority document, not a suspicion document. AI-assisted contributions are welcome. What matters is accountability and engagement, not the tool.

## The contract

1. **Disclose the tool.** A field in the pull request template carries it. It is required, and it is never by itself a reason to reject a contribution.

2. **Explain the change in your own words.** Answer reviewers yourself. A pull request whose author does not engage is closed with a pointer to Discussions and no comment on the quality of the change.

3. **Unattended agents do not open pull requests or issues.** Every pull request has an accountable person who ran the check command. An executor cannot self-authorize publication.

4. **Anything above the RFC threshold links an approved issue or discussion first.** The threshold is roughly 500 lines or any new extension-point implementation. A first contribution is an adapter against the conformance suite or a diagnosis hypothesis, and both sit below it.

## What the maintainer commits to in return

- Review begins when the checks are green. When they are not, one comment carries the local command that reproduces the failure.
- An idle pull request becomes a draft rather than a closed one.
- The commit-message grammar never touches a contributor's own commits, because the default branch takes squash merges and the maintainer writes the commit that lands.
- The per-user open-pull-request cap replaces any list of people.
- Acceptance criteria are mechanical — layer contracts, conformance suites, schema migration, digest reproducibility and invariant checks — not matters of reviewer taste.

## Why this policy exists

The gap behind every AI contribution policy is that AI lowers the cost of producing a contribution and not the cost of trustworthy review. This contract addresses scope and authority so that the review burden stays manageable without judging any individual contribution or contributor.
