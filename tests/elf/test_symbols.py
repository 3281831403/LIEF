import ctypes
import os
import stat
import subprocess
from pathlib import Path
from subprocess import Popen

import lief
import pytest
from utils import check_layout, is_linux, is_x86_64, parse_elf


def test_remove_symbol(tmp_path: Path):
    target = parse_elf("ELF/test_dyn_syms.elf")

    target.remove_dynamic_symbol("puts")

    output = tmp_path / "test_sym_removed.elf"
    target.write(output)

    new = lief.ELF.parse(output)
    assert new is not None

    assert "puts" not in {s.name for s in new.dynamic_symbols}

    if is_linux() and is_x86_64():
        st = os.stat(output)
        os.chmod(output, st.st_mode | stat.S_IEXEC)

        with Popen(
            output.as_posix(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        ) as P:
            assert P.stdout is not None
            stdout = P.stdout.read().decode("utf8")
            lief.logging.info(stdout)
            assert len(stdout) > 0
            assert "Hello world" in stdout

        # Test with bind now
        env = dict(os.environ)
        env["LD_BIND_NOW"] = "1"
        with Popen(
            output.as_posix(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env
        ) as P:
            assert P.stdout is not None
            stdout = P.stdout.read().decode("utf8")
            lief.logging.info(stdout)
            assert len(stdout) > 0
            assert "Hello world" in stdout


@pytest.mark.lief_extended
def test_demangling():
    elf = parse_elf("ELF/ELF64_x86-64_library_libtriton.so")

    assert (
        elf.symbols[80].demangled_name
        == "vtable for std::basic_streambuf<char, std::char_traits<char>>"
    )
    assert (
        elf.symbols[4902].demangled_name
        == "typeinfo name for triton::smt2lib::smtAstIteNode"
    )


@pytest.mark.parametrize("force_relocate", (True, False))
def test_remove_symbol_newline_names(tmp_path: Path, force_relocate: bool):
    """Test related to issue #321"""

    def _create_symbol(name: str, value: int) -> lief.ELF.Symbol:
        sym = lief.ELF.Symbol()
        sym.name = name
        sym.value = value
        sym.type = lief.ELF.Symbol.TYPE.FUNC
        sym.binding = lief.ELF.Symbol.BINDING.GLOBAL
        sym.visibility = lief.ELF.Symbol.VISIBILITY.DEFAULT
        sym.shndx = 1
        return sym

    weird = "\n" * 32

    extra = (
        [weird] * 64
        + ["\n\n\nfoo", "\n\nfoo", "\nfoo", "foo"]
        + ["survivor", "survivor"]
    )

    elf = parse_elf("ELF/ELF64_x86-64_library_libadd.so")
    for i, name in enumerate(extra):
        elf.add_symtab_symbol(_create_symbol(name, 0x1000 + i))

    for sym in list(elf.symtab_symbols):
        assert isinstance(sym.name, str)
        if sym.name.startswith("\n"):
            elf.remove_symtab_symbol(sym)

    expected = sorted(s.name for s in elf.symtab_symbols)
    assert weird not in expected
    assert "foo" in expected
    assert "survivor" in expected

    config = lief.ELF.Builder.config_t()
    config.force_relocate = force_relocate

    output = tmp_path / "newline_symbols.elf"
    elf.write(output, config)

    new = lief.ELF.parse(output)
    assert new is not None

    assert sorted(s.name for s in new.symtab_symbols) == expected

    check_layout(new)


def test_remove_dynamic_symbol_shrink_tables(tmp_path: Path):
    """Removing dynamic symbols must shrink .dynsym / .gnu.version /
    .gnu.version_d in place (default config): the sections keep their
    original offsets and virtual addresses, the re-parsed symbols match
    the in-memory model, and version definitions that are no longer
    referenced are dropped along with DT_VERDEFNUM.
    """
    target = parse_elf("ELF/lib_symbol_versions.so")

    orig_layout = {
        sec.name: (sec.virtual_address, sec.offset, sec.size)
        for sec in target.sections
    }
    nb_verdef = len(target.symbols_version_definition)
    assert nb_verdef == 3

    # 'LIBFOO_1.0' is the only symbol bound to the LIBFOO_1.0 version
    # definition: removing it must drop the definition as well.
    target.remove_dynamic_symbol("LIBFOO_1.0")

    expected_names = [s.name for s in target.dynamic_symbols]
    assert len(target.symbols_version_definition) == nb_verdef - 1
    assert target[lief.ELF.DynamicEntry.TAG.VERDEFNUM].value == nb_verdef - 1

    output = tmp_path / "lib_symbol_versions.stripped.so"
    target.write(output)

    new = lief.ELF.parse(output)
    assert new is not None

    # The re-parsed symbols must exactly match the in-memory model
    # (i.e. no ghost symbols from the previous table)
    assert [s.name for s in new.dynamic_symbols] == expected_names

    # The emptied version definition must be gone from .gnu.version_d
    # and DT_VERDEFNUM
    assert {d.ndx for d in new.symbols_version_definition} == {1, 3}
    assert new[lief.ELF.DynamicEntry.TAG.VERDEFNUM].value == nb_verdef - 1

    for sec in new.sections:
        va, offset, _ = orig_layout[sec.name]
        # All the sections must keep their original location
        assert (sec.virtual_address, sec.offset) == (va, offset)

    # ... while the tables themselves are shrunk in place
    dynsym = new.get_section(".dynsym")
    assert dynsym.size == len(expected_names) * dynsym.entry_size
    assert dynsym.size < orig_layout[".dynsym"][2]

    versym = new.get_section(".gnu.version")
    assert versym.size == len(expected_names) * 2
    assert versym.size < orig_layout[".gnu.version"][2]

    verdef = new.get_section(".gnu.version_d")
    assert verdef.size < orig_layout[".gnu.version_d"][2]

    check_layout(new)

    if is_linux() and is_x86_64():
        st = os.stat(output)
        os.chmod(output, st.st_mode | stat.S_IEXEC)

        lib = ctypes.cdll.LoadLibrary(output.as_posix())

        # The remaining export must still be resolvable. foo is bound to
        # a non-default version (foo@LIBFOO_2.0) so it needs dlvsym.
        libc = ctypes.CDLL("libc.so.6")
        dlvsym = libc.dlvsym
        dlvsym.restype = ctypes.c_void_p
        addr = dlvsym(ctypes.c_void_p(lib._handle), b"foo", b"LIBFOO_2.0")
        assert addr
