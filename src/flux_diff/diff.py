"""FLUX Bytecode Diff and Migration Tools.

Provides instruction normalization, diffing, patching, ISA migration,
disassembly, and fingerprinting for FLUX bytecode programs.
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Opcode format table
# ---------------------------------------------------------------------------
# Format A: 1 byte         (0x00-0x07, 0xF0-0xFF)
# Format B: 2 bytes, 1 reg (0x08-0x0F)
# Format C: 2 bytes, imm8  (0x10-0x17)
# Format D: 3 bytes, reg+imm8 (0x18-0x1F)
# Format E: 4 bytes, 3 regs (0x20-0x6F, 0x70-0x9F, 0xA0-0xCF)
# Format F: 4 bytes, reg+imm16 (0x40-0x47, 0xE0-0xEF)
# Format G: 5 bytes, 2 regs+imm16 (0x48-0x4F, 0xD0-0xDF)
#
# Overlaps resolved by priority: G > F > D > C > B > E > A

class _Fmt(Enum):
    A = "A"  # 1 byte
    B = "B"  # 2 bytes, 1 register
    C = "C"  # 2 bytes, imm8
    D = "D"  # 3 bytes, reg + imm8
    E = "E"  # 4 bytes, 3 registers
    F = "F"  # 4 bytes, reg + imm16
    G = "G"  # 5 bytes, 2 regs + imm16


def _opcode_format(opcode: int) -> _Fmt:
    """Determine the instruction format for a given opcode.

    Priority order handles overlapping ranges (G > F > D > C > B > E > A).
    """
    # Format G: 5 bytes (highest priority for overlaps)
    if opcode in range(0x48, 0x50) or opcode in range(0xD0, 0xE0):
        return _Fmt.G
    # Format F: 4 bytes
    if opcode in range(0x40, 0x48) or opcode in range(0xE0, 0xF0):
        return _Fmt.F
    # Format D: 3 bytes
    if opcode in range(0x18, 0x20):
        return _Fmt.D
    # Format C: 2 bytes
    if opcode in range(0x10, 0x18):
        return _Fmt.C
    # Format B: 2 bytes
    if opcode in range(0x08, 0x10):
        return _Fmt.B
    # Format E: 4 bytes
    if opcode in range(0x20, 0x70) or opcode in range(0x70, 0xA0) or opcode in range(0xA0, 0xD0):
        return _Fmt.E
    # Format A: 1 byte
    if opcode in range(0x00, 0x08) or opcode in range(0xF0, 0x100):
        return _Fmt.A
    raise ValueError(f"Unknown opcode format for 0x{opcode:02X}")


def _insn_size(opcode: int) -> int:
    """Return the byte size of an instruction based on its opcode."""
    fmt = _opcode_format(opcode)
    return {_Fmt.A: 1, _Fmt.B: 2, _Fmt.C: 2, _Fmt.D: 3, _Fmt.E: 4, _Fmt.F: 4, _Fmt.G: 5}[fmt]


# ---------------------------------------------------------------------------
# Opcode name table (human-readable)
# ---------------------------------------------------------------------------
_OPCODE_NAMES: dict[int, str] = {
    # Format A opcodes
    0x00: "NOP", 0x01: "HALT", 0x02: "RET", 0x03: "BREAK",
    0x04: "YIELD", 0x05: "SYSCALL", 0x06: "FAULT", 0x07: "RESET",
    0xF0: "PUSH_ALL", 0xF1: "POP_ALL", 0xF2: "INTERRUPT",
    0xF3: "WAIT", 0xF4: "SIGNAL", 0xF5: "LOCK", 0xF6: "UNLOCK",
    0xF7: "FENCE", 0xF8: "PAUSE", 0xF9: "FLUSH", 0xFA: "BARRIER",
    0xFB: "TRAP_IN", 0xFC: "TRAP_OUT", 0xFD: "DEBUG",
    0xFE: "INVALID_A", 0xFF: "INVALID_B",
}

# Default names for ranges we don't explicitly name
for _rng, _prefix in [
    (range(0x08, 0x10), "LDR"),
    (range(0x10, 0x18), "LOADI"),
    (range(0x18, 0x20), "LDRI"),
    (range(0x20, 0x30), "ADD"),
    (range(0x30, 0x40), "SUB"),
    (range(0x40, 0x48), "ADDI"),
    (range(0x48, 0x50), "MULI"),
    (range(0x50, 0x60), "AND"),
    (range(0x60, 0x70), "OR"),
    (range(0x70, 0x80), "XOR"),
    (range(0x80, 0x90), "SHL"),
    (range(0x90, 0xA0), "SHR"),
    (range(0xA0, 0xB0), "CMP"),
    (range(0xB0, 0xC0), "MOV"),
    (range(0xC0, 0xD0), "SWAP"),
    (range(0xD0, 0xE0), "BRANCH"),
    (range(0xE0, 0xF0), "JUMP"),
]:
    for _op in _rng:
        if _op not in _OPCODE_NAMES:
            _OPCODE_NAMES[_op] = _prefix


def _opcode_name(opcode: int) -> str:
    return _OPCODE_NAMES.get(opcode, f"OP_{opcode:02X}")


# ---------------------------------------------------------------------------
# 1. Instruction Normalizer
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class NormalizedInstruction:
    """A decoded, normalized bytecode instruction."""
    offset: int
    opcode: int
    operands: Tuple[int, ...]
    raw_bytes: bytes
    format: str

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, NormalizedInstruction):
            return NotImplemented
        return self.opcode == other.opcode and self.operands == other.operands

    def __hash__(self) -> int:
        return hash((self.opcode, self.operands))


def _sign_extend_8(val: int) -> int:
    """Sign-extend an 8-bit value to Python int."""
    if val >= 0x80:
        return val - 0x100
    return val


def _sign_extend_16(val: int) -> int:
    """Sign-extend a 16-bit value to Python int."""
    if val >= 0x8000:
        return val - 0x10000
    return val


def _canonical_register(reg: int) -> int:
    """Map register aliases to canonical form.

    R8-R15 are aliases for R0-R7 in some contexts.
    For now, we canonicalize R8+ -> R8+ (keep as-is) unless
    alias mapping is provided.
    """
    return reg  # Default: no alias mapping


def _normalize_immediate(opcode: int, imm: int, width: int) -> int:
    """Normalize immediate values (sign-extend for signed operations)."""
    if width == 8:
        return _sign_extend_8(imm & 0xFF)
    if width == 16:
        return _sign_extend_16(imm & 0xFFFF)
    return imm


def _decode_instruction(bytecode: bytes, offset: int) -> Optional[Tuple[NormalizedInstruction, int]]:
    """Decode a single instruction from bytecode at the given offset.

    Returns (instruction, next_offset) or None if at end of bytecode.
    """
    if offset >= len(bytecode):
        return None

    opcode = bytecode[offset]
    fmt = _opcode_format(opcode)

    if fmt == _Fmt.A:
        raw = bytecode[offset:offset + 1]
        insn = NormalizedInstruction(
            offset=offset, opcode=opcode, operands=(),
            raw_bytes=raw, format="A"
        )
        return insn, offset + 1

    elif fmt == _Fmt.B:
        if offset + 2 > len(bytecode):
            return None
        reg = bytecode[offset + 1]
        raw = bytecode[offset:offset + 2]
        insn = NormalizedInstruction(
            offset=offset, opcode=opcode,
            operands=(_canonical_register(reg),),
            raw_bytes=raw, format="B"
        )
        return insn, offset + 2

    elif fmt == _Fmt.C:
        if offset + 2 > len(bytecode):
            return None
        imm = bytecode[offset + 1]
        raw = bytecode[offset:offset + 2]
        insn = NormalizedInstruction(
            offset=offset, opcode=opcode,
            operands=(_normalize_immediate(opcode, imm, 8),),
            raw_bytes=raw, format="C"
        )
        return insn, offset + 2

    elif fmt == _Fmt.D:
        if offset + 3 > len(bytecode):
            return None
        reg = bytecode[offset + 1]
        imm = bytecode[offset + 2]
        raw = bytecode[offset:offset + 3]
        insn = NormalizedInstruction(
            offset=offset, opcode=opcode,
            operands=(_canonical_register(reg), _normalize_immediate(opcode, imm, 8)),
            raw_bytes=raw, format="D"
        )
        return insn, offset + 3

    elif fmt == _Fmt.E:
        if offset + 4 > len(bytecode):
            return None
        r1 = bytecode[offset + 1]
        r2 = bytecode[offset + 2]
        r3 = bytecode[offset + 3]
        raw = bytecode[offset:offset + 4]
        insn = NormalizedInstruction(
            offset=offset, opcode=opcode,
            operands=(
                _canonical_register(r1),
                _canonical_register(r2),
                _canonical_register(r3),
            ),
            raw_bytes=raw, format="E"
        )
        return insn, offset + 4

    elif fmt == _Fmt.F:
        if offset + 4 > len(bytecode):
            return None
        reg = bytecode[offset + 1]
        imm = struct.unpack_from(">H", bytecode, offset + 2)[0]
        raw = bytecode[offset:offset + 4]
        insn = NormalizedInstruction(
            offset=offset, opcode=opcode,
            operands=(_canonical_register(reg), _normalize_immediate(opcode, imm, 16)),
            raw_bytes=raw, format="F"
        )
        return insn, offset + 4

    elif fmt == _Fmt.G:
        if offset + 5 > len(bytecode):
            return None
        r1 = bytecode[offset + 1]
        r2 = bytecode[offset + 2]
        imm = struct.unpack_from(">H", bytecode, offset + 3)[0]
        raw = bytecode[offset:offset + 5]
        insn = NormalizedInstruction(
            offset=offset, opcode=opcode,
            operands=(
                _canonical_register(r1),
                _canonical_register(r2),
                _normalize_immediate(opcode, imm, 16),
            ),
            raw_bytes=raw, format="G"
        )
        return insn, offset + 5

    return None


def normalize(bytecode: bytes) -> List[NormalizedInstruction]:
    """Decode all instructions from bytecode into normalized form.

    Args:
        bytecode: Raw bytecode bytes.

    Returns:
        List of NormalizedInstruction instances.
    """
    instructions: List[NormalizedInstruction] = []
    offset = 0
    while offset < len(bytecode):
        result = _decode_instruction(bytecode, offset)
        if result is None:
            break
        insn, offset = result
        instructions.append(insn)
    return instructions


# ---------------------------------------------------------------------------
# 2. Bytecode Differ
# ---------------------------------------------------------------------------

class DiffType(Enum):
    ADDED = "added"
    REMOVED = "removed"
    MODIFIED = "modified"
    UNCHANGED = "unchanged"


@dataclass
class DiffEntry:
    """A single diff entry comparing old and new instruction sequences."""
    type: DiffType
    old_index: Optional[int] = None
    new_index: Optional[int] = None
    old_insn: Optional[NormalizedInstruction] = None
    new_insn: Optional[NormalizedInstruction] = None


@dataclass
class DiffStats:
    """Aggregate statistics for a diff result."""
    insertions: int = 0
    deletions: int = 0
    modifications: int = 0
    unchanged: int = 0

    @property
    def total_changes(self) -> int:
        return self.insertions + self.deletions + self.modifications

    @property
    def total(self) -> int:
        return self.insertions + self.deletions + self.modifications + self.unchanged

    def __str__(self) -> str:
        parts = []
        if self.insertions:
            parts.append(f"+{self.insertions}")
        if self.deletions:
            parts.append(f"-{self.deletions}")
        if self.modifications:
            parts.append(f"~{self.modifications}")
        parts.append(f"={self.unchanged}")
        return " ".join(parts)


def _lcs_lengths(a: list, b: list) -> list[list[int]]:
    """Compute LCS length matrix for two sequences."""
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp


def _backtrack_diff(
    a: List[NormalizedInstruction],
    b: List[NormalizedInstruction],
    dp: list[list[int]],
) -> List[DiffEntry]:
    """Backtrack through the LCS matrix to produce diff entries.

    Uses a greedy heuristic: when a[i] != b[j], compare the opcodes.
    If opcodes match but operands differ, emit MODIFIED. Otherwise emit
    REMOVED + ADDED.
    """
    entries: List[DiffEntry] = []
    i, j = len(a), len(b)

    # Collect raw backtracked entries (reversed)
    raw: List[DiffEntry] = []

    while i > 0 or j > 0:
        if i > 0 and j > 0 and a[i - 1] == b[j - 1]:
            raw.append(DiffEntry(
                type=DiffType.UNCHANGED,
                old_index=i - 1, new_index=j - 1,
                old_insn=a[i - 1], new_insn=b[j - 1],
            ))
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and a[i - 1].opcode == b[j - 1].opcode:
            # Same opcode, different operands -> MODIFIED
            raw.append(DiffEntry(
                type=DiffType.MODIFIED,
                old_index=i - 1, new_index=j - 1,
                old_insn=a[i - 1], new_insn=b[j - 1],
            ))
            i -= 1
            j -= 1
        elif j > 0 and (i == 0 or dp[i][j - 1] >= dp[i - 1][j]):
            raw.append(DiffEntry(
                type=DiffType.ADDED,
                new_index=j - 1, new_insn=b[j - 1],
            ))
            j -= 1
        else:
            raw.append(DiffEntry(
                type=DiffType.REMOVED,
                old_index=i - 1, old_insn=a[i - 1],
            ))
            i -= 1

    raw.reverse()
    return raw


def diff(
    old_bytecode: bytes,
    new_bytecode: bytes,
) -> List[DiffEntry]:
    """Compare two bytecode programs and produce a list of diff entries.

    Uses LCS-based diff algorithm on decoded instruction sequences.
    Instructions are matched by (opcode, operands). If opcodes match but
    operands differ, a MODIFIED entry is produced.

    Args:
        old_bytecode: Original bytecode.
        new_bytecode: Modified bytecode.

    Returns:
        List of DiffEntry instances.
    """
    old_insns = normalize(old_bytecode)
    new_insns = normalize(new_bytecode)
    dp = _lcs_lengths(old_insns, new_insns)
    return _backtrack_diff(old_insns, new_insns, dp)


def diffstat(diffs: List[DiffEntry]) -> DiffStats:
    """Compute aggregate statistics from a list of diff entries.

    Args:
        diffs: List of DiffEntry from diff().

    Returns:
        DiffStats with counts of insertions, deletions, modifications, unchanged.
    """
    stats = DiffStats()
    for entry in diffs:
        if entry.type == DiffType.ADDED:
            stats.insertions += 1
        elif entry.type == DiffType.REMOVED:
            stats.deletions += 1
        elif entry.type == DiffType.MODIFIED:
            stats.modifications += 1
        elif entry.type == DiffType.UNCHANGED:
            stats.unchanged += 1
    return stats


def similarity_score(old_bytecode: bytes, new_bytecode: bytes) -> float:
    """Compute a similarity score between two bytecode programs.

    Score ranges from 0.0 (completely different) to 1.0 (identical).

    Args:
        old_bytecode: Original bytecode.
        new_bytecode: Modified bytecode.

    Returns:
        Float similarity score between 0.0 and 1.0.
    """
    if old_bytecode == new_bytecode:
        return 1.0
    if not old_bytecode or not new_bytecode:
        return 0.0

    old_insns = normalize(old_bytecode)
    new_insns = normalize(new_bytecode)

    if not old_insns and not new_insns:
        return 1.0

    dp = _lcs_lengths(old_insns, new_insns)
    lcs_len = dp[len(old_insns)][len(new_insns)]
    max_len = max(len(old_insns), len(new_insns))

    if max_len == 0:
        return 1.0

    return lcs_len / max_len


# ---------------------------------------------------------------------------
# 3. Patch System
# ---------------------------------------------------------------------------

@dataclass
class Patch:
    """A byte-level patch: replace old_bytes with new_bytes at offset."""
    offset: int
    old_bytes: bytes
    new_bytes: bytes


@dataclass
class InstructionPatch:
    """An instruction-level patch: replace one instruction with another."""
    offset: int
    old_insn: NormalizedInstruction
    new_insn: NormalizedInstruction


@dataclass
class PatchSet:
    """A collection of patches to apply to bytecode."""
    patches: List[Patch] = field(default_factory=list)
    insn_patches: List[InstructionPatch] = field(default_factory=list)

    def add_patch(self, patch: Patch) -> None:
        self.patches.append(patch)

    def add_insn_patch(self, patch: InstructionPatch) -> None:
        self.insn_patches.append(patch)

    @property
    def is_empty(self) -> bool:
        return len(self.patches) == 0 and len(self.insn_patches) == 0

    def __len__(self) -> int:
        return len(self.patches) + len(self.insn_patches)


class PatchError(Exception):
    """Raised when a patch cannot be applied (verification failure)."""


def _apply_byte_patches(bytecode: bytearray, patches: List[Patch]) -> None:
    """Apply byte-level patches in reverse offset order to preserve offsets."""
    # Sort by offset descending to avoid shifting issues
    sorted_patches = sorted(patches, key=lambda p: p.offset, reverse=True)

    for patch in sorted_patches:
        start = patch.offset
        end = start + len(patch.old_bytes)

        # Verify old bytes match
        if end > len(bytecode):
            raise PatchError(
                f"Patch at offset {start}: old_bytes extend beyond bytecode "
                f"(need {end}, have {len(bytecode)})"
            )
        actual = bytes(bytecode[start:end])
        if actual != patch.old_bytes:
            raise PatchError(
                f"Patch verification failed at offset {start}: "
                f"expected {patch.old_bytes.hex()}, got {actual.hex()}"
            )

        # Apply patch
        replacement = patch.new_bytes
        bytecode[start:end] = replacement


def _apply_insn_patches(bytecode: bytearray, patches: List[InstructionPatch]) -> None:
    """Apply instruction-level patches in reverse offset order."""
    sorted_patches = sorted(patches, key=lambda p: p.offset, reverse=True)

    for patch in sorted_patches:
        start = patch.offset
        old_len = len(patch.old_insn.raw_bytes)
        new_bytes = patch.new_insn.raw_bytes
        end = start + old_len

        if end > len(bytecode):
            raise PatchError(
                f"Instruction patch at offset {start}: extends beyond bytecode"
            )

        actual = bytes(bytecode[start:end])
        if actual != patch.old_insn.raw_bytes:
            raise PatchError(
                f"Instruction patch verification failed at offset {start}: "
                f"expected {patch.old_insn.raw_bytes.hex()}, got {actual.hex()}"
            )

        bytecode[start:end] = new_bytes


def apply_patch(bytecode: bytes, patch_set: PatchSet) -> bytes:
    """Apply a PatchSet to bytecode and return the result.

    Args:
        bytecode: Original bytecode bytes.
        patch_set: Collection of patches to apply.

    Returns:
        Patched bytecode as bytes.

    Raises:
        PatchError: If any patch verification fails.
    """
    result = bytearray(bytecode)
    _apply_byte_patches(result, patch_set.patches)
    _apply_insn_patches(result, patch_set.insn_patches)
    return bytes(result)


def create_patch(old_bytecode: bytes, new_bytecode: bytes) -> PatchSet:
    """Auto-generate a PatchSet from two bytecode versions.

    Produces byte-level patches for each difference found.

    Args:
        old_bytecode: Original bytecode.
        new_bytecode: Modified bytecode.

    Returns:
        PatchSet containing byte-level patches for each change.
    """
    patch_set = PatchSet()

    # Simple byte-level diff: find changed regions
    old = bytearray(old_bytecode)
    new = bytearray(new_bytecode)

    # Find common prefix
    prefix_len = 0
    max_prefix = min(len(old), len(new))
    while prefix_len < max_prefix and old[prefix_len] == new[prefix_len]:
        prefix_len += 1

    # Find common suffix
    suffix_len = 0
    max_suffix = min(len(old) - prefix_len, len(new) - prefix_len)
    while suffix_len < max_suffix and old[len(old) - 1 - suffix_len] == new[len(new) - 1 - suffix_len]:
        suffix_len += 1

    # If there's a changed region, create a patch
    if prefix_len < len(old) or prefix_len < len(new):
        old_end = len(old) - suffix_len
        new_end = len(new) - suffix_len

        if prefix_len <= old_end and prefix_len <= new_end:
            old_region = bytes(old[prefix_len:old_end])
            new_region = bytes(new[prefix_len:new_end])
            patch_set.add_patch(Patch(
                offset=prefix_len,
                old_bytes=old_region,
                new_bytes=new_region,
            ))

    return patch_set


# ---------------------------------------------------------------------------
# 4. ISA Migration Engine
# ---------------------------------------------------------------------------

@dataclass
class MigrationStep:
    """A single step in an ISA migration plan."""
    description: str
    affected_opcodes: List[int]


@dataclass
class ISAMigration:
    """Migration specification between two ISA versions.

    Attributes:
        from_version: Source ISA version string.
        to_version: Target ISA version string.
        opcode_remap: Mapping from old opcode -> new opcode.
        format_changes: Mapping from opcode -> new format (if encoding changed).
        operand_transforms: Mapping from opcode -> transform function.
    """
    from_version: str
    to_version: str
    opcode_remap: dict[int, int] = field(default_factory=dict)
    format_changes: dict[int, str] = field(default_factory=dict)
    operand_transforms: dict[int, callable] = field(default_factory=dict)


def _encode_instruction(insn: NormalizedInstruction, opcode_remap: dict[int, int] | None = None) -> bytes:
    """Re-encode a NormalizedInstruction back to bytes.

    Args:
        insn: The instruction to encode.
        opcode_remap: Optional opcode remapping table.

    Returns:
        Encoded bytes.
    """
    opcode = insn.opcode
    if opcode_remap and opcode in opcode_remap:
        opcode = opcode_remap[opcode]

    result = bytearray([opcode])
    ops = insn.operands

    fmt = _opcode_format(opcode)

    if fmt == _Fmt.A:
        pass
    elif fmt == _Fmt.B:
        result.append(ops[0] & 0xFF)
    elif fmt == _Fmt.C:
        result.append(ops[0] & 0xFF)
    elif fmt == _Fmt.D:
        result.append(ops[0] & 0xFF)
        result.append(ops[1] & 0xFF)
    elif fmt == _Fmt.E:
        result.append(ops[0] & 0xFF)
        result.append(ops[1] & 0xFF)
        result.append(ops[2] & 0xFF)
    elif fmt == _Fmt.F:
        result.append(ops[0] & 0xFF)
        result.extend(struct.pack(">H", ops[1] & 0xFFFF))
    elif fmt == _Fmt.G:
        result.append(ops[0] & 0xFF)
        result.append(ops[1] & 0xFF)
        result.extend(struct.pack(">H", ops[2] & 0xFFFF))

    return bytes(result)


def migrate(bytecode: bytes, migration: ISAMigration) -> bytes:
    """Migrate bytecode from one ISA version to another.

    Applies opcode remapping and re-encodes each instruction.

    Args:
        bytecode: Source bytecode.
        migration: ISAMigration specification.

    Returns:
        Migrated bytecode bytes.
    """
    insns = normalize(bytecode)
    result = bytearray()

    for insn in insns:
        opcode = insn.opcode
        ops = list(insn.operands)

        # Apply operand transforms
        if opcode in migration.operand_transforms:
            ops = migration.operand_transforms[opcode](ops)

        # Create new instruction with potentially modified operands
        migrated_insn = NormalizedInstruction(
            offset=insn.offset,
            opcode=opcode,
            operands=tuple(ops),
            raw_bytes=insn.raw_bytes,
            format=insn.format,
        )

        # Re-encode with opcode remapping
        encoded = _encode_instruction(migrated_insn, migration.opcode_remap)
        result.extend(encoded)

    return bytes(result)


def migrate_with_patch(bytecode: bytes, migration: ISAMigration, patches: PatchSet) -> bytes:
    """Migrate bytecode and then apply patches.

    First applies ISA migration, then applies the patch set.

    Args:
        bytecode: Source bytecode.
        migration: ISAMigration specification.
        patches: PatchSet to apply after migration.

    Returns:
        Migrated and patched bytecode bytes.
    """
    migrated = migrate(bytecode, migration)
    return apply_patch(migrated, patches)


def migration_plan(from_version: str, to_version: str) -> List[MigrationStep]:
    """Generate a migration plan between two ISA versions.

    Returns a list of migration steps describing what changes are needed.

    Args:
        from_version: Source ISA version (e.g., "1.0").
        to_version: Target ISA version (e.g., "2.0").

    Returns:
        List of MigrationStep describing each migration phase.
    """
    steps: List[MigrationStep] = []

    if from_version == to_version:
        return steps

    # Parse version numbers
    try:
        major_from, minor_from = (int(x) for x in from_version.split("."))
        major_to, minor_to = (int(x) for x in to_version.split("."))
    except (ValueError, AttributeError):
        steps.append(MigrationStep(
            description=f"Unknown version format: {from_version} -> {to_version}",
            affected_opcodes=[],
        ))
        return steps

    # Major version bump: opcode renumbering
    if major_to > major_from:
        steps.append(MigrationStep(
            description=f"Opcode renumbering for ISA v{to_version}",
            affected_opcodes=list(range(0x00, 0x100)),
        ))

    # Minor version bumps: specific changes
    if minor_to > minor_from:
        steps.append(MigrationStep(
            description=f"Format changes for ISA v{major_to}.{minor_to}",
            affected_opcodes=list(range(0x40, 0x50)),
        ))
        steps.append(MigrationStep(
            description=f"Operand encoding updates for ISA v{major_to}.{minor_to}",
            affected_opcodes=list(range(0xE0, 0xF0)),
        ))

    return steps


# ---------------------------------------------------------------------------
# 5. Disassembler Integration
# ---------------------------------------------------------------------------

def disassemble_insn(opcode: int, operands: Tuple[int, ...]) -> str:
    """Disassemble a single instruction to human-readable form.

    Args:
        opcode: Instruction opcode.
        operands: Tuple of operand values.

    Returns:
        Human-readable instruction string.
    """
    name = _opcode_name(opcode)

    if not operands:
        return name

    fmt = _opcode_format(opcode)

    if fmt == _Fmt.B:
        return f"{name} R{operands[0]}"
    elif fmt == _Fmt.C:
        return f"{name} #{operands[0]}"
    elif fmt == _Fmt.D:
        return f"{name} R{operands[0]}, #{operands[1]}"
    elif fmt == _Fmt.E:
        return f"{name} R{operands[0]}, R{operands[1]}, R{operands[2]}"
    elif fmt == _Fmt.F:
        return f"{name} R{operands[0]}, #{operands[1]}"
    elif fmt == _Fmt.G:
        return f"{name} R{operands[0]}, R{operands[1]}, #{operands[2]}"

    return name


def _insn_to_str(insn: NormalizedInstruction) -> str:
    """Format an instruction for diff output."""
    addr = f"{insn.offset:04X}"
    dis = disassemble_insn(insn.opcode, insn.operands)
    raw_hex = insn.raw_bytes.hex().upper()
    return f"{addr}:  {dis:<24s}  [{raw_hex}]"


def diff_report(old_bytecode: bytes, new_bytecode: bytes) -> str:
    """Generate a unified diff report for two bytecode programs.

    Args:
        old_bytecode: Original bytecode.
        new_bytecode: Modified bytecode.

    Returns:
        Unified diff format string.
    """
    diffs = diff(old_bytecode, new_bytecode)
    lines: List[str] = []

    lines.append("--- old bytecode")
    lines.append("+++ new bytecode")
    lines.append(f"@@ diff report @@")
    lines.append("")

    old_insns = normalize(old_bytecode)
    new_insns = normalize(new_bytecode)

    # Build index maps
    old_map = {i: old_insns[i] for i in range(len(old_insns))}
    new_map = {i: new_insns[i] for i in range(len(new_insns))}

    for entry in diffs:
        if entry.type == DiffType.UNCHANGED:
            if entry.old_insn is not None:
                lines.append(f"  {_insn_to_str(entry.old_insn)}")
        elif entry.type == DiffType.ADDED:
            if entry.new_insn is not None:
                lines.append(f"+ {_insn_to_str(entry.new_insn)}")
        elif entry.type == DiffType.REMOVED:
            if entry.old_insn is not None:
                lines.append(f"- {_insn_to_str(entry.old_insn)}")
        elif entry.type == DiffType.MODIFIED:
            if entry.old_insn is not None:
                lines.append(f"- {_insn_to_str(entry.old_insn)}")
            if entry.new_insn is not None:
                lines.append(f"+ {_insn_to_str(entry.new_insn)}")

    stats = diffstat(diffs)
    lines.append("")
    lines.append(f"Summary: {stats}")
    return "\n".join(lines)


# ANSI color codes
_RED = "\033[31m"
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_CYAN = "\033[36m"
_RESET = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[2m"


def color_diff(old_bytecode: bytes, new_bytecode: bytes) -> str:
    """Generate a colorized diff report using ANSI escape codes.

    Args:
        old_bytecode: Original bytecode.
        new_bytecode: Modified bytecode.

    Returns:
        Colorized diff string with ANSI escape codes.
    """
    diffs = diff(old_bytecode, new_bytecode)
    lines: List[str] = []

    lines.append(f"{_RED}{_BOLD}--- old bytecode{_RESET}")
    lines.append(f"{_GREEN}{_BOLD}+++ new bytecode{_RESET}")
    lines.append(f"{_CYAN}{_BOLD}@@ diff report @@{_RESET}")
    lines.append("")

    for entry in diffs:
        if entry.type == DiffType.UNCHANGED:
            if entry.old_insn is not None:
                s = _insn_to_str(entry.old_insn)
                lines.append(f"  {_DIM}{s}{_RESET}")
        elif entry.type == DiffType.ADDED:
            if entry.new_insn is not None:
                s = _insn_to_str(entry.new_insn)
                lines.append(f"{_GREEN}+ {s}{_RESET}")
        elif entry.type == DiffType.REMOVED:
            if entry.old_insn is not None:
                s = _insn_to_str(entry.old_insn)
                lines.append(f"{_RED}- {s}{_RESET}")
        elif entry.type == DiffType.MODIFIED:
            if entry.old_insn is not None:
                s = _insn_to_str(entry.old_insn)
                lines.append(f"{_RED}- {s}{_RESET}")
            if entry.new_insn is not None:
                s = _insn_to_str(entry.new_insn)
                lines.append(f"{_GREEN}+ {s}{_RESET}")

    stats = diffstat(diffs)
    lines.append("")
    summary_color = _YELLOW if stats.total_changes > 0 else _GREEN
    lines.append(f"{summary_color}{_BOLD}Summary: {stats}{_RESET}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 6. Bytecode Fingerprint
# ---------------------------------------------------------------------------

def fingerprint(bytecode: bytes) -> str:
    """Compute a content hash of the bytecode.

    Uses SHA-256 on the raw bytes.

    Args:
        bytecode: Raw bytecode bytes.

    Returns:
        Hex-encoded SHA-256 hash string.
    """
    return hashlib.sha256(bytecode).hexdigest()


def structural_fingerprint(bytecode: bytes) -> str:
    """Compute a structural fingerprint ignoring immediate values.

    Hashes only the opcode and register operands (not immediates).

    Args:
        bytecode: Raw bytecode bytes.

    Returns:
        Hex-encoded SHA-256 hash string.
    """
    insns = normalize(bytecode)
    h = hashlib.sha256()

    for insn in insns:
        h.update(struct.pack(">B", insn.opcode))
        fmt = _opcode_format(insn.opcode)

        if fmt in (_Fmt.B, _Fmt.D):
            # First operand is a register
            h.update(struct.pack(">B", insn.operands[0] & 0xFF))
        elif fmt == _Fmt.E:
            # All operands are registers
            for op in insn.operands:
                h.update(struct.pack(">B", op & 0xFF))
        elif fmt == _Fmt.F:
            # First operand is a register
            h.update(struct.pack(">B", insn.operands[0] & 0xFF))
        elif fmt == _Fmt.G:
            # First two operands are registers
            h.update(struct.pack(">B", insn.operands[0] & 0xFF))
            h.update(struct.pack(">B", insn.operands[1] & 0xFF))

    return h.hexdigest()


def semantic_fingerprint(bytecode: bytes) -> str:
    """Compute a fingerprint based on the opcode sequence only.

    Ignores all operands, capturing only the program structure.

    Args:
        bytecode: Raw bytecode bytes.

    Returns:
        Hex-encoded SHA-256 hash string.
    """
    insns = normalize(bytecode)
    h = hashlib.sha256()

    for insn in insns:
        h.update(struct.pack(">B", insn.opcode))

    return h.hexdigest()
