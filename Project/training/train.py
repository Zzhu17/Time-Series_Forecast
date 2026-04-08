"""Legacy training module kept as compatibility shim for imports."""
"""Deprecated training entrypoint.

This module is kept only for backward compatibility and should not be used as
an active training entry.

Current training main path:
- services.train_service.run_training_task
- services.pipeline.run_train_predict_pipeline
- models.registry.TRAINER_REGISTRY
"""

# DEPRECATED:
# Please route all training calls through services.train_service.run_training_task,
# which delegates to services.pipeline.run_train_predict_pipeline and resolves
# model trainers/adaptors from models.registry.TRAINER_REGISTRY.
