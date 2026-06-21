# Skill Registry Intake Governance

Community `SKILL.md` files are not compiled into trusted runtime artifacts merely because they exist. The first-pass registry is offline and Rust-native: it records provenance, license status, adoption signals, audit status, and trust tier before a skill is compiled.

## Intake Flow

```text
community discovery
  -> provenance capture
  -> adoption/usefulness scoring
  -> static safety audit
  -> license and ownership check
  -> quarantine or review
  -> trusted registry
  -> compiled .hskill
```

## Trust Tiers

- `Untrusted`: recorded but not trusted.
- `LocalDev`: local project-authored or development skill; routable by default.
- `CommunityReviewed`: community skill with passed review; routable and trusted.
- `HeirloomReviewed`: reviewed for Heirloom use; routable and trusted.
- `HeirloomCore`: core maintained skill; routable and trusted.
- `Quarantined`: blocked from routing.
- `Rejected`: blocked from routing.

`--trusted-only` restricts route/eval loading to `CommunityReviewed`, `HeirloomReviewed`, and `HeirloomCore`.

## Registry Shape

```json
{
  "entries": [
    {
      "skill_name": "spreadsheets",
      "source_path": "heirloom-skill/examples/skills/spreadsheets/SKILL.md",
      "source_url": null,
      "source_commit": null,
      "author": "Heirloom",
      "organization": "Heirloom",
      "license": "MIT OR Apache-2.0",
      "license_status": "project_local",
      "adoption_signals": ["example corpus"],
      "trust_tier": "HeirloomCore",
      "audit_status": "Passed",
      "notes": "First-pass local skill corpus."
    }
  ]
}
```

Run:

```text
cargo run -p heirloom-skill --bin heirloom-skill -- registry lint --registry path/to/registry.json
```

The first pass intentionally does not crawl GitHub, Discord, forums, or package registries. Outreach and search should produce registry candidates with evidence; compilation remains a separate reviewed step.
