"""Fail-closed legacy entry point for distributed target-dump commands.

The contracted launcher rejects ``--pipeline`` and multi-process target
creation because per-rank checkpoint shards cannot attest a complete teacher.
Keeping this file makes old launch commands fail with that explicit provenance
error instead of silently bypassing :mod:`alis_dwq.run`.
"""
from alis_dwq.run import main

main()
