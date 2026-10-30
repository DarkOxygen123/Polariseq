"""Griffe extension for the website: the docstrings as ``docs/API.md`` shows them.

``polariseq/_api_reference.py`` fills in what the docstrings leave to convention; this does the
same for the site, on the docstring text, before mkdocstrings renders it:

- Sphinx roles (``:func:`read_h5ad```) become inline code.
- Arguments named together (``min_counts, min_cells: ...``) become one entry each.
- Arguments every plot or step shares (``adata``, ``show``, ``save``, ``ax``, ...) get the
  description of ``_api_reference._COMMON`` when the docstring does not give one.
- The ``ps.`` names under ``See also:`` become links to their pages.
- Functions that hand their work to one of ``_curation`` (the curation steps of ``ps.tl``,
  ``ps.get.markers``) take its arguments' descriptions.
"""

from __future__ import annotations

import ast
import re
import textwrap

import griffe

_ROLE = re.compile(r":(?:func|class|meth|attr|mod|data|obj|exc|py:\w+):`~?([^`]+)`")
_SECTION = re.compile(r"^(Args|Arguments|Returns|Raises|Yields|Example|Examples|Note|Notes|"
                      r"Attributes|See Also|Warns|Warnings):\s*$")
_LITERAL = re.compile(r"^(\S.*?)::\n\n((?:(?:    .*)?\n)+)", re.M)
_ENTRY = re.compile(r"^    (\*{0,2}[A-Za-z_][\w, *]*?)(\s*\([^)]*\))?:(?:\s+(.*))?$")


def _args(lines: list[str]) -> dict[str, list[str]]:
    """The entries of an ``Args:`` section: each name with its description lines, grouped names
    split into one entry each."""
    out: dict[str, list[str]] = {}
    names: list[str] = []
    for line in lines:
        m = _ENTRY.match(line)
        if m:
            names = [n.strip() for n in m.group(1).split(",")]
            for n in names:
                out[n] = [m.group(3) or ""]
        elif names:
            for n in names:
                out[n].append(line.strip())
    return out


def _sections(text: str) -> tuple[list[str], list[str], list[str]]:
    """The docstring's lines before ``Args:``, the section's entry lines, and the lines after."""
    lines = text.splitlines()
    start = next((i for i, ln in enumerate(lines) if ln.strip() in ("Args:", "Arguments:")), None)
    if start is None:
        end = next((i for i, ln in enumerate(lines) if _SECTION.match(ln.strip())), len(lines))
        return lines[:end], [], lines[end:]
    end = next((i for i in range(start + 1, len(lines))
                if lines[i].strip() and not lines[i].startswith(" ")), len(lines))
    return lines[:start], lines[start + 1:end], lines[end:]


def _plain(text: str) -> str:
    """reST markup that Markdown shows as written: roles become inline code, and a paragraph ending
    in ``::`` introduces a Python code block."""
    text = _ROLE.sub(r"`\1`", text)

    def block(m: re.Match) -> str:
        code = textwrap.dedent(m.group(2)).strip("\n")
        return f"{m.group(1)}:\n\n```python\n{code}\n```\n\n"

    return _LITERAL.sub(block, text + "\n").rstrip("\n")


#: Where the ``ps`` namespaces live, for cross-references.
_OWNERS = {"pp": "_Preprocessing", "tl": "_Tools", "get": "_Get", "pl": "plotting"}
_SEE_ALSO = re.compile(r"^See also:\s*$", re.M)
_PS_NAME = re.compile(r"``(ps\.(?:(pp|tl|get|pl)\.)?(\w+))``")


def _see_also_links(text: str) -> str:
    """The ``ps.`` names of a docstring's ``See also:`` section as links to their pages."""
    m = _SEE_ALSO.search(text)
    if not m:
        return text

    def link(n: re.Match) -> str:
        name, ns, fn = n.group(1), n.group(2), n.group(3)
        # ``ps.resources`` too: its page's anchor is ``polariseq.resources``.
        target = f"polariseq.{_OWNERS[ns]}.{fn}" if ns else f"polariseq.{fn}"
        return f"[`{name}`][{target}]"

    return text[:m.end()] + _PS_NAME.sub(link, text[m.end():])


def _rewrite(func: griffe.Function, common: dict[str, str],
             delegated: dict[str, list[str]]) -> None:
    text = func.docstring.value
    before, entries, after = _sections(text)
    documented = _args(entries)
    variadic = (griffe.ParameterKind.var_positional, griffe.ParameterKind.var_keyword)
    params = [p.name for p in func.parameters
              if p.name not in ("self", "cls") and p.kind not in variadic]
    merged: dict[str, list[str]] = {}
    for name in params:
        if name in documented:
            merged[name] = documented[name]
        elif name in delegated:
            merged[name] = delegated[name]
        elif name in common:
            merged[name] = [common[name]]
    merged.update({n: d for n, d in documented.items() if n not in merged})
    if merged:
        while before and not before[-1].strip():
            before.pop()
        section = ["", "Args:"]
        for name, desc in merged.items():
            section.append(f"    {name}: {desc[0]}".rstrip())
            section += [f"        {ln}" for ln in desc[1:] if ln]
        text = "\n".join(before + section + ([""] if after and after[0].strip() else []) + after)
    func.docstring.value = text


def _objects(obj: griffe.Object):
    """Every object defined in ``obj``, recursively (imports skipped)."""
    for member in obj.members.values():
        if member.is_alias:
            continue
        yield member
        if member.is_class or member.is_module:
            yield from _objects(member)


class PolariseqDocstrings(griffe.Extension):
    def __init__(self) -> None:
        super().__init__()
        self._shadowed: dict[str, griffe.Function] = {}

    def on_function_instance(self, *, func: griffe.Function, **kwargs) -> None:
        # ``ps.resources`` is a function and a module of the same name; the module wins the
        # member slot, so keep the function to put back under its own key.
        if func.parent is not None and func.parent.path == "polariseq" and func.name == "resources":
            self._shadowed[func.name] = func

    def on_package(self, *, pkg: griffe.Module, **kwargs) -> None:
        if pkg.name != "polariseq":
            return
        for name, func in self._shadowed.items():
            pkg.set_member(f"{name}_function", func)
        common = ast.literal_eval(str(pkg["_api_reference._COMMON"].value))
        curation = pkg["_curation"]
        objects = [pkg, *_objects(pkg), *self._shadowed.values()]
        for obj in objects:
            if obj.docstring:
                obj.docstring.value = _see_also_links(_plain(obj.docstring.value))
        for obj in objects:
            if not obj.is_function or not obj.docstring:
                continue
            delegated: dict[str, list[str]] = {}
            if obj.name in curation.members and obj.parent is not curation:
                impl = curation.members[obj.name]
                if impl.docstring:
                    delegated = _args(_sections(impl.docstring.value)[1])
            _rewrite(obj, common, delegated)
