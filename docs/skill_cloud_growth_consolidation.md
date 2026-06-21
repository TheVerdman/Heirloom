# Skill Cloud Growth And Consolidation

The longer-term training loop treats compiled skills, retrieval, validators, traces, and temporary experts as plastic capacity that can later consolidate into a more stable Heirloom base model.

```text
B0 = stable Heirloom base model
C0 = B0 + temporary skill cloud/adapters/experts/retrieval/tools
D0 = validated traces generated from C0
B1 = consolidated model trained or distilled from B0 + D0 + replay
```

Conceptually:

```text
1B base
  -> 5B skill cloud
  -> 4B consolidated base
  -> 11B skill cloud
  -> 9B consolidated base
  -> ...
```

Here, "cloud" does not require one dense model. It can include:

- temporary experts
- LoRA/adapters
- compiled `.hskill` artifacts
- retrieval memory
- tool policies
- validators
- synthetic traces
- teacher critiques

The core loop is:

```text
plastic expansion -> validated learning -> stable consolidation
```

The first production spine focuses on the compiled artifact and trace path:

```text
SKILL.md
  -> SkillIR
  -> .hskill
  -> routing/context/policy/validation
  -> successful traces
  -> SFT cloud data
```

Later passes can add learned embeddings, signed registries, richer trace privacy gates, expert routing, adapter training, replay selection, and consolidation metrics while preserving the Rust-native compiler boundary.
