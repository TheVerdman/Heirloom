# Skill: code_editing

## Version
1.0.0

## Description
Inspect, modify, test, and explain code changes in an existing repository with minimal targeted patches and truthful verification.

## Triggers
- fix this bug in the repo
- implement this code change
- refactor this module
- add tests for this feature
- review this pull request

## Negative Triggers
- create a spreadsheet dashboard
- summarize a PDF
- send an email

## Required Tools
- filesystem

## Allowed Tools
- filesystem
- cargo
- git
- test-runner

## Forbidden Tools
- destructive-git-reset
- email-send

## Preflight Steps
- Inspect existing project structure before editing.
- Check current git status before making changes.
- Identify relevant tests or validation commands.

## Execution Steps
- Prefer minimal targeted patches.
- Follow existing repo style and local abstractions.
- Run relevant tests when available.

## Hard Constraints
- Inspect existing project structure before editing.
- Prefer minimal, targeted patches.
- Run relevant tests when available.
- Do not claim tests passed unless they were actually run.

## Soft Guidelines
- Explain tradeoffs only when they affect the implementation.
- Avoid broad refactors unrelated to the request.

## Validation Steps
- Run the narrowest relevant test command.
- Report commands that were run and their result.
- If tests cannot be run, state the reason clearly.

## Failure Modes
- Dirty worktree contains unrelated user changes.
- Test command requires unavailable external service.
- Requested change conflicts with existing architecture.

## Examples
- user_request: Fix the failing parser test in this Rust crate.
  expected_behavior: Inspect the repo and failing test, patch the narrow issue, run the relevant cargo test, and report the result.
- user_request: Add validation for this config field.
  expected_behavior: Locate the config parser, add targeted validation, add or update tests, and avoid unrelated refactors.

## Eval Cases
- name: code_test_truthfulness
  user_request: Implement this Rust change and make sure tests pass.
  expected_skills: code_editing
  required_constraints:
    - Do not claim tests passed unless they were actually run.
- name: code_repo_inspection
  user_request: Fix this bug in the repository.
  expected_skills: code_editing
  required_constraints:
    - Inspect existing project structure before editing.

## Metadata
{
  "domain": "software_engineering",
  "artifact_types": ["patch", "source"]
}
