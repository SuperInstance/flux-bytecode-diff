"""Comprehensive tests for flux_diff module — 60 tests covering all 6 components."""

import struct
import pytest

from flux_diff.diff import (
    # Normalizer
    NormalizedInstruction,
    normalize,
    _opcode_format,
    _insn_size,
    _sign_extend_8,
    _sign_extend_16,
    _canonical_register,
    _normalize_immediate,
    _decode_instruction,
    _encode_instruction,
    # Differ
    DiffEntry,
    DiffType,
    DiffStats,
    diff,
    diffstat,
    similarity_score,
    # Patch
    Patch,
    InstructionPatch,
    PatchSet,
    PatchError,
    apply_patch,
    create_patch,
    # Migration
    ISAMigration,
    MigrationStep,
    migrate,
    migrate_with_patch,
    migration_plan,
    # Disassembler
    disassemble_insn,
    diff_report,
    color_diff,
    # Fingerprint
    fingerprint,
    structural_fingerprint,
    semantic_fingerprint,
    _lcs_lengths,
    _backtrack_diff,
)


# ===========================================================================
# 1. Instruction Normalizer Tests
# ===========================================================================

class TestOpcodeFormat:
    """Tests for opcode format determination."""

    def test_format_a_low(self):
        assert _opcode_format(0x00).value == "A"
        assert _opcode_format(0x07).value == "A"

    def test_format_a_high(self):
        assert _opcode_format(0xF0).value == "A"
        assert _opcode_format(0xFF).value == "A"

    def test_format_b(self):
        assert _opcode_format(0x08).value == "B"
        assert _opcode_format(0x0F).value == "B"

    def test_format_c(self):
        assert _opcode_format(0x10).value == "C"
        assert _opcode_format(0x17).value == "C"

    def test_format_d(self):
        assert _opcode_format(0x18).value == "D"
        assert _opcode_format(0x1F).value == "D"

    def test_format_e_low(self):
        assert _opcode_format(0x20).value == "E"
        assert _opcode_format(0x3F).value == "E"

    def test_format_e_mid(self):
        assert _opcode_format(0x70).value == "E"
        assert _opcode_format(0x9F).value == "E"

    def test_format_e_high(self):
        assert _opcode_format(0xA0).value == "E"
        assert _opcode_format(0xCF).value == "E"

    def test_format_f_priority_over_e(self):
        """Format F (0x40-0x47) should take priority over Format E."""
        assert _opcode_format(0x40).value == "F"
        assert _opcode_format(0x47).value == "F"

    def test_format_f_high(self):
        assert _opcode_format(0xE0).value == "F"
        assert _opcode_format(0xEF).value == "F"

    def test_format_g_priority_over_e(self):
        """Format G (0x48-0x4F) should take priority over Format E."""
        assert _opcode_format(0x48).value == "G"
        assert _opcode_format(0x4F).value == "G"

    def test_format_g_high(self):
        assert _opcode_format(0xD0).value == "G"
        assert _opcode_format(0xDF).value == "G"

    def test_unknown_opcode_raises(self):
        # All opcodes 0x00-0xFF are covered, but test boundary behavior
        with pytest.raises(ValueError):
            _opcode_format(256)


class TestInstructionSize:
    """Tests for instruction size calculation."""

    def test_size_format_a(self):
        assert _insn_size(0x00) == 1

    def test_size_format_b(self):
        assert _insn_size(0x08) == 2

    def test_size_format_c(self):
        assert _insn_size(0x10) == 2

    def test_size_format_d(self):
        assert _insn_size(0x18) == 3

    def test_size_format_e(self):
        assert _insn_size(0x20) == 4

    def test_size_format_f(self):
        assert _insn_size(0x40) == 4

    def test_size_format_g(self):
        assert _insn_size(0x48) == 5


class TestSignExtend:
    """Tests for sign extension helpers."""

    def test_sign_extend_8_positive(self):
        assert _sign_extend_8(0x7F) == 127

    def test_sign_extend_8_negative(self):
        assert _sign_extend_8(0x80) == -128
        assert _sign_extend_8(0xFF) == -1

    def test_sign_extend_16_positive(self):
        assert _sign_extend_16(0x7FFF) == 32767

    def test_sign_extend_16_negative(self):
        assert _sign_extend_16(0x8000) == -32768
        assert _sign_extend_16(0xFFFF) == -1


class TestNormalize:
    """Tests for the normalize() function."""

    def test_empty_bytecode(self):
        assert normalize(b"") == []

    def test_single_nop(self):
        """NOP is Format A (0x00), 1 byte."""
        insns = normalize(bytes([0x00]))
        assert len(insns) == 1
        assert insns[0].opcode == 0x00
        assert insns[0].operands == ()
        assert insns[0].format == "A"

    def test_single_halt(self):
        insns = normalize(bytes([0x01]))
        assert len(insns) == 1
        assert insns[0].opcode == 0x01

    def test_format_b_instruction(self):
        """LDR R3: opcode 0x08, register 3."""
        insns = normalize(bytes([0x08, 0x03]))
        assert len(insns) == 1
        assert insns[0].opcode == 0x08
        assert insns[0].operands == (3,)
        assert insns[0].format == "B"

    def test_format_c_instruction(self):
        """LOADI #42: opcode 0x10, imm8 42."""
        insns = normalize(bytes([0x10, 0x2A]))
        assert len(insns) == 1
        assert insns[0].opcode == 0x10
        assert insns[0].operands == (42,)
        assert insns[0].format == "C"

    def test_format_c_negative_immediate(self):
        """LOADI #-1: opcode 0x10, imm8 0xFF should sign-extend to -1."""
        insns = normalize(bytes([0x10, 0xFF]))
        assert insns[0].operands == (-1,)

    def test_format_d_instruction(self):
        """LDRI R2, #10: opcode 0x18, reg 2, imm8 10."""
        insns = normalize(bytes([0x18, 0x02, 0x0A]))
        assert len(insns) == 1
        assert insns[0].opcode == 0x18
        assert insns[0].operands == (2, 10)
        assert insns[0].format == "D"

    def test_format_e_instruction(self):
        """ADD R1, R2, R3: opcode 0x20, 3 registers."""
        insns = normalize(bytes([0x20, 0x01, 0x02, 0x03]))
        assert len(insns) == 1
        assert insns[0].opcode == 0x20
        assert insns[0].operands == (1, 2, 3)
        assert insns[0].format == "E"

    def test_format_f_instruction(self):
        """ADDI R1, #256: opcode 0x40, reg 1, imm16 256."""
        imm = struct.pack(">H", 256)
        insns = normalize(bytes([0x40, 0x01]) + imm)
        assert len(insns) == 1
        assert insns[0].opcode == 0x40
        assert insns[0].operands == (1, 256)
        assert insns[0].format == "F"

    def test_format_g_instruction(self):
        """MULI R1, R2, #1024: opcode 0x48, reg1, reg2, imm16 1024."""
        imm = struct.pack(">H", 1024)
        insns = normalize(bytes([0x48, 0x01, 0x02]) + imm)
        assert len(insns) == 1
        assert insns[0].opcode == 0x48
        assert insns[0].operands == (1, 2, 1024)
        assert insns[0].format == "G"

    def test_multiple_instructions(self):
        """NOP + LDR R0 + HALT."""
        bc = bytes([0x00, 0x08, 0x00, 0x01])
        insns = normalize(bc)
        assert len(insns) == 3
        assert insns[0].opcode == 0x00
        assert insns[1].opcode == 0x08
        assert insns[1].operands == (0,)
        assert insns[2].opcode == 0x01

    def test_truncated_bytecode_ignored(self):
        """Partial instruction at end should be ignored."""
        insns = normalize(bytes([0x00, 0x20]))  # Format E needs 4 bytes
        assert len(insns) == 1
        assert insns[0].opcode == 0x00

    def test_instruction_offsets(self):
        insns = normalize(bytes([0x00, 0x08, 0x00, 0x01]))
        assert insns[0].offset == 0
        assert insns[1].offset == 1
        assert insns[2].offset == 3

    def test_normalized_instruction_equality(self):
        insns = normalize(bytes([0x08, 0x05]))
        same = normalize(bytes([0x08, 0x05]))
        assert insns[0] == same[0]

    def test_raw_bytes_preserved(self):
        insns = normalize(bytes([0x08, 0x05]))
        assert insns[0].raw_bytes == bytes([0x08, 0x05])


# ===========================================================================
# 2. Bytecode Differ Tests
# ===========================================================================

class TestDiff:
    """Tests for the diff() function."""

    def test_identical_bytecode(self):
        bc = bytes([0x00, 0x08, 0x00, 0x01])
        diffs = diff(bc, bc)
        assert all(d.type == DiffType.UNCHANGED for d in diffs)
        assert len(diffs) == 3

    def test_empty_bytecodes(self):
        diffs = diff(b"", b"")
        assert diffs == []

    def test_added_instructions(self):
        old = bytes([0x00, 0x01])
        new = bytes([0x00, 0x08, 0x00, 0x01])
        diffs = diff(old, new)
        types = [d.type for d in diffs]
        assert DiffType.ADDED in types

    def test_removed_instructions(self):
        old = bytes([0x00, 0x08, 0x00, 0x01])
        new = bytes([0x00, 0x01])
        diffs = diff(old, new)
        types = [d.type for d in diffs]
        assert DiffType.REMOVED in types

    def test_modified_instruction(self):
        """Same opcode, different operands -> MODIFIED."""
        old = bytes([0x08, 0x00])  # LDR R0
        new = bytes([0x08, 0x01])  # LDR R1
        diffs = diff(old, new)
        assert len(diffs) == 1
        assert diffs[0].type == DiffType.MODIFIED

    def test_complete_replacement(self):
        old = bytes([0x00])
        new = bytes([0x01])
        diffs = diff(old, new)
        assert len(diffs) == 2
        types = [d.type for d in diffs]
        assert DiffType.REMOVED in types
        assert DiffType.ADDED in types

    def test_complex_diff(self):
        old = bytes([0x00, 0x20, 0x01, 0x02, 0x03, 0x01])  # NOP, ADD R1,R2,R3, HALT
        new = bytes([0x00, 0x20, 0x01, 0x02, 0x04, 0x08, 0x00, 0x01])  # NOP, ADD R1,R2,R4, LDR R0, HALT
        diffs = diff(old, new)
        mod_count = sum(1 for d in diffs if d.type == DiffType.MODIFIED)
        assert mod_count == 1

    def test_diff_entry_has_correct_indices(self):
        old = bytes([0x00, 0x08, 0x00])
        new = bytes([0x00, 0x08, 0x01])
        diffs = diff(old, new)
        mod_entries = [d for d in diffs if d.type == DiffType.MODIFIED]
        assert len(mod_entries) == 1
        assert mod_entries[0].old_index == 1
        assert mod_entries[0].new_index == 1


class TestDiffstat:
    """Tests for diffstat()."""

    def test_empty_diff(self):
        stats = diffstat([])
        assert stats.insertions == 0
        assert stats.deletions == 0
        assert stats.modifications == 0
        assert stats.unchanged == 0

    def test_identical(self):
        bc = bytes([0x00, 0x01])
        diffs = diff(bc, bc)
        stats = diffstat(diffs)
        assert stats.unchanged == 2
        assert stats.total_changes == 0

    def test_total_changes(self):
        stats = DiffStats(insertions=3, deletions=2, modifications=1, unchanged=10)
        assert stats.total_changes == 6
        assert stats.total == 16

    def test_diffstat_str(self):
        stats = DiffStats(insertions=3, deletions=2, modifications=1, unchanged=10)
        s = str(stats)
        assert "+3" in s
        assert "-2" in s
        assert "~1" in s
        assert "=10" in s


class TestSimilarityScore:
    """Tests for similarity_score()."""

    def test_identical(self):
        bc = bytes([0x00, 0x08, 0x00, 0x01])
        assert similarity_score(bc, bc) == 1.0

    def test_empty(self):
        assert similarity_score(b"", b"") == 1.0

    def test_one_empty(self):
        assert similarity_score(bytes([0x00]), b"") == 0.0

    def test_completely_different(self):
        # Different opcode sequences should have low similarity
        old = bytes([0x00, 0x01, 0x02])
        new = bytes([0x08, 0x00, 0x10, 0x2A])
        score = similarity_score(old, new)
        assert 0.0 <= score < 0.5

    def test_mostly_similar(self):
        old = bytes([0x00, 0x08, 0x00, 0x01, 0x02])
        new = bytes([0x00, 0x08, 0x01, 0x01, 0x02])
        score = similarity_score(old, new)
        assert 0.5 < score <= 1.0

    def test_score_between_zero_and_one(self):
        old = bytes([0x00] * 5)
        new = bytes([0x01] * 5)
        score = similarity_score(old, new)
        assert 0.0 <= score <= 1.0


# ===========================================================================
# 3. Patch System Tests
# ===========================================================================

class TestPatch:
    """Tests for byte-level and instruction-level patches."""

    def test_apply_single_byte_patch(self):
        bc = bytes([0x00, 0x01, 0x02])
        ps = PatchSet()
        ps.add_patch(Patch(offset=1, old_bytes=bytes([0x01]), new_bytes=bytes([0x08])))
        result = apply_patch(bc, ps)
        assert result == bytes([0x00, 0x08, 0x02])

    def test_apply_multi_byte_patch(self):
        bc = bytes([0x00, 0x08, 0x00, 0x01])
        ps = PatchSet()
        ps.add_patch(Patch(
            offset=1, old_bytes=bytes([0x08, 0x00]),
            new_bytes=bytes([0x08, 0x01])
        ))
        result = apply_patch(bc, ps)
        assert result == bytes([0x00, 0x08, 0x01, 0x01])

    def test_patch_verification_failure(self):
        bc = bytes([0x00, 0x01, 0x02])
        ps = PatchSet()
        ps.add_patch(Patch(offset=1, old_bytes=bytes([0xFF]), new_bytes=bytes([0x08])))
        with pytest.raises(PatchError, match="verification failed"):
            apply_patch(bc, ps)

    def test_patch_beyond_boundary(self):
        bc = bytes([0x00, 0x01])
        ps = PatchSet()
        ps.add_patch(Patch(offset=1, old_bytes=bytes([0x01, 0x02]), new_bytes=bytes([0x08])))
        with pytest.raises(PatchError, match="beyond bytecode"):
            apply_patch(bc, ps)

    def test_empty_patch_set(self):
        bc = bytes([0x00, 0x01])
        ps = PatchSet()
        result = apply_patch(bc, ps)
        assert result == bc

    def test_instruction_patch(self):
        insns = normalize(bytes([0x08, 0x00]))
        new_insns = normalize(bytes([0x08, 0x05]))
        ps = PatchSet()
        ps.add_insn_patch(InstructionPatch(offset=0, old_insn=insns[0], new_insn=new_insns[0]))
        result = apply_patch(bytes([0x08, 0x00]), ps)
        assert result == bytes([0x08, 0x05])

    def test_instruction_patch_verification(self):
        insns = normalize(bytes([0x08, 0x00]))
        new_insns = normalize(bytes([0x08, 0x05]))
        ps = PatchSet()
        ps.add_insn_patch(InstructionPatch(offset=0, old_insn=insns[0], new_insn=new_insns[0]))
        with pytest.raises(PatchError):
            apply_patch(bytes([0xFF, 0xFF]), ps)

    def test_patch_set_length(self):
        ps = PatchSet()
        assert len(ps) == 0
        assert ps.is_empty
        ps.add_patch(Patch(offset=0, old_bytes=b"", new_bytes=b""))
        assert len(ps) == 1
        assert not ps.is_empty
        ps.add_insn_patch(InstructionPatch(
            offset=0,
            old_insn=NormalizedInstruction(0, 0, (), b"", "A"),
            new_insn=NormalizedInstruction(0, 0, (), b"", "A"),
        ))
        assert len(ps) == 2

    def test_create_patch_identical(self):
        bc = bytes([0x00, 0x01])
        ps = create_patch(bc, bc)
        # For identical bytecodes, prefix_len covers all, no patch generated
        assert ps.is_empty

    def test_create_patch_different(self):
        old = bytes([0x00, 0x01, 0x02])
        new = bytes([0x00, 0xFF, 0x02])
        ps = create_patch(old, new)
        assert len(ps) == 1
        p = ps.patches[0]
        assert p.offset == 1
        assert p.old_bytes == bytes([0x01])
        assert p.new_bytes == bytes([0xFF])

    def test_create_patch_insertion(self):
        old = bytes([0x00, 0x02])
        new = bytes([0x00, 0x01, 0x02])
        ps = create_patch(old, new)
        assert len(ps) == 1
        assert ps.patches[0].old_bytes == bytes([])
        assert ps.patches[0].new_bytes == bytes([0x01])

    def test_apply_created_patch(self):
        old = bytes([0x00, 0x01, 0x02])
        new = bytes([0x00, 0xFF, 0x02])
        ps = create_patch(old, new)
        result = apply_patch(old, ps)
        assert result == new


# ===========================================================================
# 4. ISA Migration Tests
# ===========================================================================

class TestISAMigration:
    """Tests for ISA migration engine."""

    def test_simple_opcode_remap(self):
        bc = bytes([0x00, 0x01])  # NOP, HALT
        migration = ISAMigration(
            from_version="1.0",
            to_version="2.0",
            opcode_remap={0x00: 0xF0, 0x01: 0xF1},
        )
        result = migrate(bc, migration)
        assert result == bytes([0xF0, 0xF1])

    def test_no_remap(self):
        bc = bytes([0x00, 0x08, 0x00, 0x01])
        migration = ISAMigration(from_version="1.0", to_version="1.0")
        result = migrate(bc, migration)
        assert result == bc

    def test_operand_transform(self):
        bc = bytes([0x08, 0x02])  # LDR R2
        migration = ISAMigration(
            from_version="1.0",
            to_version="2.0",
            operand_transforms={0x08: lambda ops: [ops[0] + 1]},
        )
        result = migrate(bc, migration)
        assert result == bytes([0x08, 0x03])  # LDR R3

    def test_migrate_with_format_f(self):
        imm = struct.pack(">H", 256)
        bc = bytes([0x40, 0x01]) + imm  # ADDI R1, #256
        migration = ISAMigration(
            from_version="1.0",
            to_version="2.0",
            opcode_remap={0x40: 0x41},
        )
        result = migrate(bc, migration)
        assert result[0] == 0x41
        assert result[1] == 0x01
        assert struct.unpack(">H", result[2:4])[0] == 256

    def test_migrate_with_format_g(self):
        imm = struct.pack(">H", 1024)
        bc = bytes([0x48, 0x01, 0x02]) + imm  # MULI R1, R2, #1024
        migration = ISAMigration(
            from_version="1.0",
            to_version="2.0",
            opcode_remap={0x48: 0x49},
        )
        result = migrate(bc, migration)
        assert result[0] == 0x49
        assert result[1] == 0x01
        assert result[2] == 0x02

    def test_migrate_with_patch(self):
        bc = bytes([0x00, 0x01])
        migration = ISAMigration(
            from_version="1.0",
            to_version="2.0",
            opcode_remap={0x00: 0xF0},
        )
        ps = PatchSet()
        ps.add_patch(Patch(offset=1, old_bytes=bytes([0x01]), new_bytes=bytes([0xF1])))
        result = migrate_with_patch(bc, migration, ps)
        assert result == bytes([0xF0, 0xF1])

    def test_migrate_with_empty_patch(self):
        bc = bytes([0x00, 0x01])
        migration = ISAMigration(from_version="1.0", to_version="2.0")
        ps = PatchSet()
        result = migrate_with_patch(bc, migration, ps)
        assert result == bc


class TestMigrationPlan:
    """Tests for migration_plan()."""

    def test_same_version(self):
        plan = migration_plan("1.0", "1.0")
        assert plan == []

    def test_minor_upgrade(self):
        plan = migration_plan("1.0", "1.1")
        assert len(plan) == 2
        assert "Format changes" in plan[0].description
        assert "Operand encoding" in plan[1].description

    def test_major_upgrade(self):
        plan = migration_plan("1.0", "2.0")
        assert len(plan) >= 1
        assert "Opcode renumbering" in plan[0].description

    def test_major_and_minor(self):
        plan = migration_plan("1.0", "2.1")
        assert len(plan) == 3

    def test_invalid_version(self):
        plan = migration_plan("abc", "2.0")
        assert len(plan) == 1
        assert "Unknown" in plan[0].description


# ===========================================================================
# 5. Disassembler Integration Tests
# ===========================================================================

class TestDisassembleInsn:
    """Tests for disassemble_insn()."""

    def test_format_a(self):
        assert disassemble_insn(0x00, ()) == "NOP"
        assert disassemble_insn(0x01, ()) == "HALT"

    def test_format_b(self):
        assert disassemble_insn(0x08, (5,)) == "LDR R5"
        assert disassemble_insn(0x0F, (0,)) == "LDR R0"

    def test_format_c(self):
        assert disassemble_insn(0x10, (42,)) == "LOADI #42"
        assert disassemble_insn(0x10, (-1,)) == "LOADI #-1"

    def test_format_d(self):
        result = disassemble_insn(0x18, (2, 10))
        assert result == "LDRI R2, #10"

    def test_format_e(self):
        result = disassemble_insn(0x20, (1, 2, 3))
        assert result == "ADD R1, R2, R3"

    def test_format_f(self):
        result = disassemble_insn(0x40, (1, 256))
        assert result == "ADDI R1, #256"

    def test_format_g(self):
        result = disassemble_insn(0x48, (1, 2, 1024))
        assert result == "MULI R1, R2, #1024"


class TestDiffReport:
    """Tests for diff_report()."""

    def test_identical_report(self):
        bc = bytes([0x00, 0x01])
        report = diff_report(bc, bc)
        assert "--- old bytecode" in report
        assert "+++ new bytecode" in report
        assert "Summary:" in report

    def test_diff_report_shows_changes(self):
        old = bytes([0x00, 0x08, 0x00])
        new = bytes([0x00, 0x08, 0x01])
        report = diff_report(old, new)
        assert "- " in report
        assert "+ " in report

    def test_diff_report_has_addresses(self):
        bc = bytes([0x00, 0x01])
        report = diff_report(bc, bc)
        assert "0000:" in report
        assert "0001:" in report


class TestColorDiff:
    """Tests for color_diff()."""

    def test_has_ansi_codes(self):
        bc = bytes([0x00, 0x01])
        colored = color_diff(bc, bc)
        assert "\033[" in colored

    def test_has_reset_codes(self):
        old = bytes([0x00])
        new = bytes([0x01])
        colored = color_diff(old, new)
        assert "\033[0m" in colored

    def test_has_colored_headers(self):
        colored = color_diff(bytes([0x00]), bytes([0x00]))
        assert "\033[31m" in colored  # Red for ---
        assert "\033[32m" in colored  # Green for +++

    def test_color_diff_unchanged_dim(self):
        colored = color_diff(bytes([0x00]), bytes([0x00]))
        assert "\033[2m" in colored  # Dim for unchanged


# ===========================================================================
# 6. Bytecode Fingerprint Tests
# ===========================================================================

class TestFingerprint:
    """Tests for fingerprinting functions."""

    def test_fingerprint_deterministic(self):
        bc = bytes([0x00, 0x08, 0x00, 0x01])
        assert fingerprint(bc) == fingerprint(bc)

    def test_fingerprint_different_bytecodes(self):
        fp1 = fingerprint(bytes([0x00, 0x01]))
        fp2 = fingerprint(bytes([0x00, 0x02]))
        assert fp1 != fp2

    def test_fingerprint_empty(self):
        fp = fingerprint(b"")
        assert isinstance(fp, str)
        assert len(fp) == 64  # SHA-256 hex

    def test_structural_fingerprint_ignores_immediates(self):
        """Structural fingerprint should be the same when only immediates differ."""
        bc1 = bytes([0x10, 0x0A])  # LOADI #10
        bc2 = bytes([0x10, 0xFF])  # LOADI #-1
        assert structural_fingerprint(bc1) == structural_fingerprint(bc2)

    def test_structural_fingerprint_catches_opcode_change(self):
        bc1 = bytes([0x10, 0x0A])  # LOADI #10
        bc2 = bytes([0x11, 0x0A])  # LOADI_1 #10
        assert structural_fingerprint(bc1) != structural_fingerprint(bc2)

    def test_structural_fingerprint_catches_register_change(self):
        bc1 = bytes([0x08, 0x00])  # LDR R0
        bc2 = bytes([0x08, 0x01])  # LDR R1
        assert structural_fingerprint(bc1) != structural_fingerprint(bc2)

    def test_semantic_fingerprint_same_structure(self):
        """Semantic fingerprint should match for same opcode sequence regardless of operands."""
        bc1 = bytes([0x20, 0x01, 0x02, 0x03])  # ADD R1, R2, R3
        bc2 = bytes([0x20, 0x04, 0x05, 0x06])  # ADD R4, R5, R6
        assert semantic_fingerprint(bc1) == semantic_fingerprint(bc2)

    def test_semantic_fingerprint_different_opcodes(self):
        bc1 = bytes([0x00, 0x01])  # NOP, HALT
        bc2 = bytes([0x01, 0x00])  # HALT, NOP
        assert semantic_fingerprint(bc1) != semantic_fingerprint(bc2)

    def test_semantic_fingerprint_format_f(self):
        imm1 = struct.pack(">H", 100)
        imm2 = struct.pack(">H", 200)
        bc1 = bytes([0x40, 0x01]) + imm1
        bc2 = bytes([0x40, 0x02]) + imm2
        assert semantic_fingerprint(bc1) == semantic_fingerprint(bc2)

    def test_semantic_fingerprint_format_g(self):
        imm1 = struct.pack(">H", 100)
        imm2 = struct.pack(">H", 200)
        bc1 = bytes([0x48, 0x01, 0x02]) + imm1
        bc2 = bytes([0x48, 0x03, 0x04]) + imm2
        assert semantic_fingerprint(bc1) == semantic_fingerprint(bc2)

    def test_fingerprint_differs_from_structural(self):
        bc = bytes([0x10, 0x0A])
        assert fingerprint(bc) != structural_fingerprint(bc)

    def test_structural_differs_from_semantic(self):
        bc = bytes([0x08, 0x05])  # LDR R5
        assert structural_fingerprint(bc) != semantic_fingerprint(bc)


# ===========================================================================
# 7. LCS Algorithm Tests
# ===========================================================================

class TestLCS:
    """Tests for the LCS helper functions."""

    def test_lcs_empty(self):
        dp = _lcs_lengths([], [1, 2, 3])
        assert dp[0][3] == 0

    def test_lcs_identical(self):
        a = [1, 2, 3]
        dp = _lcs_lengths(a, a)
        assert dp[3][3] == 3

    def test_lcs_partial(self):
        a = [1, 2, 3]
        b = [2, 3, 4]
        dp = _lcs_lengths(a, b)
        assert dp[3][3] == 2

    def test_lcs_no_match(self):
        a = [1, 2]
        b = [3, 4]
        dp = _lcs_lengths(a, b)
        assert dp[2][2] == 0


# ===========================================================================
# 8. Encode Instruction Tests
# ===========================================================================

class TestEncodeInstruction:
    """Tests for _encode_instruction round-trip."""

    def test_roundtrip_format_a(self):
        insn = NormalizedInstruction(0, 0x00, (), bytes([0x00]), "A")
        encoded = _encode_instruction(insn)
        assert encoded == bytes([0x00])

    def test_roundtrip_format_b(self):
        insn = NormalizedInstruction(0, 0x08, (3,), bytes([0x08, 0x03]), "B")
        encoded = _encode_instruction(insn)
        assert encoded == bytes([0x08, 0x03])

    def test_roundtrip_format_c(self):
        insn = NormalizedInstruction(0, 0x10, (42,), bytes([0x10, 0x2A]), "C")
        encoded = _encode_instruction(insn)
        assert encoded == bytes([0x10, 0x2A])

    def test_roundtrip_format_d(self):
        insn = NormalizedInstruction(0, 0x18, (2, 10), bytes([0x18, 0x02, 0x0A]), "D")
        encoded = _encode_instruction(insn)
        assert encoded == bytes([0x18, 0x02, 0x0A])

    def test_roundtrip_format_e(self):
        insn = NormalizedInstruction(0, 0x20, (1, 2, 3), bytes([0x20, 0x01, 0x02, 0x03]), "E")
        encoded = _encode_instruction(insn)
        assert encoded == bytes([0x20, 0x01, 0x02, 0x03])

    def test_roundtrip_format_f(self):
        insn = NormalizedInstruction(0, 0x40, (1, 256), bytes([0x40, 0x01, 0x01, 0x00]), "F")
        encoded = _encode_instruction(insn)
        assert encoded == bytes([0x40, 0x01, 0x01, 0x00])

    def test_roundtrip_format_g(self):
        imm = struct.pack(">H", 1024)
        insn = NormalizedInstruction(0, 0x48, (1, 2, 1024), bytes([0x48, 0x01, 0x02]) + imm, "G")
        encoded = _encode_instruction(insn)
        assert encoded == bytes([0x48, 0x01, 0x02]) + imm

    def test_encode_with_remap(self):
        insn = NormalizedInstruction(0, 0x00, (), bytes([0x00]), "A")
        encoded = _encode_instruction(insn, {0x00: 0xF0})
        assert encoded == bytes([0xF0])


# ===========================================================================
# 9. Integration / End-to-End Tests
# ===========================================================================

class TestIntegration:
    """End-to-end tests combining multiple components."""

    def test_diff_then_patch_roundtrip(self):
        old = bytes([0x00, 0x08, 0x00, 0x01])
        new = bytes([0x00, 0x08, 0x05, 0x01])
        ps = create_patch(old, new)
        result = apply_patch(old, ps)
        assert result == new

    def test_diff_migrate_patch_workflow(self):
        # Original program
        old = bytes([0x00, 0x20, 0x01, 0x02, 0x03, 0x01])

        # Modified program
        new = bytes([0x00, 0x20, 0x01, 0x02, 0x04, 0x01])

        # Verify diff shows modification
        diffs = diff(old, new)
        mod_entries = [d for d in diffs if d.type == DiffType.MODIFIED]
        assert len(mod_entries) == 1

        # Create and apply patch
        ps = create_patch(old, new)
        result = apply_patch(old, ps)
        assert result == new

        # Verify similarity (LCS = 2 of 3 instructions since MODIFIED doesn't count in LCS)
        score = similarity_score(old, new)
        assert score > 0.5

    def test_all_fingerprints_on_program(self):
        # Program with immediate values so structural != content fingerprint
        imm = struct.pack(">H", 500)
        bc = bytes([0x00, 0x40, 0x01]) + imm + bytes([0x01])  # NOP, ADDI R1,#500, HALT
        fp = fingerprint(bc)
        sfp = structural_fingerprint(bc)
        sefp = semantic_fingerprint(bc)
        assert isinstance(fp, str)
        assert isinstance(sfp, str)
        assert isinstance(sefp, str)
        assert fp != sfp
        assert sfp != sefp

    def test_migration_preserves_structure(self):
        """After migration, semantic fingerprint of opcode sequence should change per remap."""
        bc = bytes([0x00, 0x01, 0x08, 0x00])
        migration = ISAMigration(
            from_version="1.0",
            to_version="2.0",
            opcode_remap={0x00: 0xF0, 0x01: 0xF1},
        )
        result = migrate(bc, migration)
        assert semantic_fingerprint(bc) != semantic_fingerprint(result)

    def test_complex_program_diff(self):
        """Test with a more complex multi-format program."""
        imm = struct.pack(">H", 500)
        old = bytes([
            0x00,  # NOP
            0x08, 0x01,  # LDR R1
            0x20, 0x01, 0x02, 0x03,  # ADD R1, R2, R3
            0x40, 0x01,  # ADDI R1, #500 (partial)
        ]) + imm + bytes([0x01])  # HALT

        new = bytes([
            0x00,  # NOP
            0x08, 0x02,  # LDR R2
            0x20, 0x01, 0x02, 0x04,  # ADD R1, R2, R4
            0x40, 0x01,  # ADDI R1, #500
        ]) + imm + bytes([0x01])  # HALT

        diffs = diff(old, new)
        stats = diffstat(diffs)
        assert stats.unchanged == 3  # NOP, ADDI, HALT
        assert stats.modifications == 2  # LDR and ADD modified

    def test_full_pipeline(self):
        """Full pipeline: normalize -> diff -> create_patch -> migrate_with_patch -> verify."""
        # Step 1: Define old and new
        old = bytes([0x00, 0x08, 0x00, 0x01])
        new = bytes([0x00, 0x08, 0x01, 0x01])

        # Step 2: Diff
        diffs = diff(old, new)
        assert len(diffs) == 3  # UNCHANGED, MODIFIED, UNCHANGED

        # Step 3: Create patch
        ps = create_patch(old, new)
        assert len(ps) == 1

        # Step 4: Verify patch applies cleanly
        result = apply_patch(old, ps)
        assert result == new

        # Step 5: Migrate and patch
        migration = ISAMigration("1.0", "1.1")
        final = migrate_with_patch(old, migration, ps)
        # Should be migrated (no changes since no remap) then patched
        assert final == new
