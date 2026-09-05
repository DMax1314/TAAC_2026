---
name: pcvr-experiment-integration
description: Add TAAC PCVR experiments or change their model, configuration, or checkpoint integration contracts.
---

# PCVR Experiment Integration

For package wiring, start with [experiment integration](../../../docs/guide/contributing.md)
and the affected package's `__init__.py` and `model.py`. For a shared contract
change, use [architecture](../../../docs/architecture.md) to locate its owner.
Routine implementation changes within an existing contract do not need this
integration workflow.

- Trace model/config changes through checkpoint reconstruction in evaluation
  and inference, including the experiment's typed defaults. Successful training
  alone does not establish that the saved model can be reloaded.
- Include new experiments in the existing contract-test cases so discovery,
  model inputs/outputs, and checkpoint reconstruction are exercised.
- For bundle integration, follow [the bundle guide](../../../docs/guide/online-training-bundle.md);
  packaged execution imports the shipped framework and selected experiment.

Select contract tests and the CPU roundtrip from
[the testing guide](../../../docs/guide/testing.md). Keep test paths and commands
there rather than maintaining a second list in this skill.
