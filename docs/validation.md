# Packaging validation

Validated for this source-package revision:

- 19 automated tests passed, including strict Action parsing, bounded context handling, source/test patch separation, repo split isolation, outcome/behavior/termination scoring, repeated-action evidence, deterministic curriculum restoration, exact-context SFT export, reasoning-field exclusion, and clipped GRPO gradient direction.
- A synthetic task passed real Docker and mini-SWE-agent integration: two certification repeats, a three-action scripted repair/test/submit trajectory, fresh-container hidden grading, and generation of the action-review packet.
- Python wheel build and installation into a separate temporary environment succeeded; the installed CLI entry point starts successfully.
- Python source compilation and local documentation-link checks passed.
- A scan of publication files found no credential patterns, private key blocks, or original-machine absolute paths. This is a targeted scan, not a universal guarantee about secrets.
- The source package contains no model weights, task snapshots, private test patches, teacher trajectories, run logs, or checkpoints.

The smoke backend is an explicitly named test double. It is not a Kimi call or a Qwen quality measurement. No paid API call or GPU training/evaluation job was launched for packaging validation. Newly refactored GPU orchestration still needs a small end-to-end acceptance run on the target training environment before large-scale use.
