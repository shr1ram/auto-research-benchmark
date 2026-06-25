"""Parallel batch execution: run many (task × seed × arm) jobs across multiple
GPU boxes, each producing a full traced run bundle on the shared filesystem.

Self-contained: discovers free GPU boxes itself (ssh nvidia-smi), leases them via
atomic files on the shared FS, dispatches `arbench run` over ssh, tracks a
manifest, and is resumable (a job whose run.json exists is skipped).
"""
from arbench.batch.worklist import Job, expand_worklist
from arbench.batch.scheduler import run_batch

__all__ = ["Job", "expand_worklist", "run_batch"]
