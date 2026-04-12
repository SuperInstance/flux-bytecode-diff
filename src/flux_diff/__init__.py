"""FLUX Bytecode Diff and Migration Tools.

Compare, patch, and migrate FLUX bytecode programs across ISA versions.
"""

from .diff import (
    # Instruction Normalizer
    NormalizedInstruction,
    normalize,
    # Bytecode Differ
    DiffEntry,
    DiffType,
    DiffStats,
    diff,
    diffstat,
    similarity_score,
    # Patch System
    Patch,
    InstructionPatch,
    PatchSet,
    apply_patch,
    create_patch,
    # ISA Migration
    ISAMigration,
    MigrationStep,
    migrate,
    migrate_with_patch,
    migration_plan,
    # Disassembler Integration
    disassemble_insn,
    diff_report,
    color_diff,
    # Bytecode Fingerprint
    fingerprint,
    structural_fingerprint,
    semantic_fingerprint,
)

__all__ = [
    "NormalizedInstruction",
    "normalize",
    "DiffEntry",
    "DiffType",
    "DiffStats",
    "diff",
    "diffstat",
    "similarity_score",
    "Patch",
    "InstructionPatch",
    "PatchSet",
    "apply_patch",
    "create_patch",
    "ISAMigration",
    "MigrationStep",
    "migrate",
    "migrate_with_patch",
    "migration_plan",
    "disassemble_insn",
    "diff_report",
    "color_diff",
    "fingerprint",
    "structural_fingerprint",
    "semantic_fingerprint",
]

__version__ = "0.1.0"
