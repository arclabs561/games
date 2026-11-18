#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "typst>=0.15,<0.16",
# ]
# ///
"""Compile Typst documents when packages.typst.org is unreachable.

Typst Universe packages are published from the GitHub monorepo typst/packages,
so the exact files the registry serves can be fetched with git instead:

    uv run typst_vendor.py erase.typ          # vendor every @preview import, then compile erase.pdf
    uv run typst_vendor.py erase.typ --png    # also render one PNG per page
    uv run typst_vendor.py erase.typ --all    # also erase-phone.pdf and erase.html

With network access to the registry, plain `typst compile --ignore-system-fonts
--font-path fonts erase.typ` gives the same PDF; add `--input edition=phone` for the phone
edition, or `--features html --format html` for HTML.

It scans the document for "@preview/name:version" imports, sparse-checks-out just
those package directories, scans them for their own imports (recursively), and
copies them into a local package directory that typst is pointed at. Packages
that ship WASM plugins (diagraph, cetz >= 0.4, ...) work unchanged, because the
.wasm files live inside the package directory.
"""

import argparse
import pathlib
import re
import shutil
import subprocess
import sys

import typst

REPO = "https://github.com/typst/packages"
IMPORT = re.compile(r"@preview/([a-z0-9-]+):(\d+\.\d+\.\d+)")


def git(*args, cwd=None):
    subprocess.run(
        ["git", *args], cwd=cwd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
    )


LOCAL = re.compile(r'#(?:import|include)\s+"([^"@][^"]*\.typ)"')


def imports_in(entry):
    """@preview imports reachable from one .typ file by following its local imports.
    Docs, examples and tests inside a package are never followed, so their imports are skipped."""
    seen, stack, found = set(), [entry.resolve()], set()
    while stack:
        f = stack.pop()
        if f in seen or not f.exists():
            continue
        seen.add(f)
        text = f.read_text(errors="ignore")
        found |= set(IMPORT.findall(text))
        stack += [(f.parent / rel).resolve() for rel in LOCAL.findall(text)]
    return found


def entrypoint(pkg_dir):
    toml = (pkg_dir / "typst.toml").read_text()
    m = re.search(r'entrypoint\s*=\s*"([^"]+)"', toml)
    if m is None:
        sys.exit(f"{pkg_dir / 'typst.toml'}: no entrypoint")
    return pkg_dir / m.group(1)


def vendor(doc, cache, out):
    if not (cache / ".git").exists():
        git("clone", "-q", "--depth", "1", "--filter=blob:none", "--sparse", REPO, str(cache))
    todo, done = set(imports_in(doc)), set()
    while todo - done:
        wanted = sorted(todo - done)
        missing = [(n, v) for n, v in wanted if not (out / "preview" / n / v).exists()]
        if missing:
            git(
                "sparse-checkout",
                "add",
                *[f"packages/preview/{n}/{v}" for n, v in missing],
                cwd=cache,
            )
        for name, ver in wanted:
            src = cache / "packages" / "preview" / name / ver
            dst = out / "preview" / name / ver
            if not dst.exists():
                if not src.exists():
                    sys.exit(f"{name}:{ver} is not in the registry")
                shutil.copytree(src, dst)
            print(f"  vendored {name}:{ver}")
            done.add((name, ver))
            todo |= imports_in(entrypoint(dst))
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("doc", type=pathlib.Path)
    cache = pathlib.Path.home() / ".cache"
    ap.add_argument("--packages", type=pathlib.Path, default=cache / "typst-vendored")
    ap.add_argument("--cache", type=pathlib.Path, default=cache / "typst-packages-git")
    ap.add_argument("--png", action="store_true")
    ap.add_argument("--all", action="store_true", help="also build the phone PDF and the HTML page")
    a = ap.parse_args()
    a.packages.mkdir(parents=True, exist_ok=True)
    vendor(a.doc, a.cache, a.packages)
    # Only Typst's own fonts and any in a fonts/ folder beside the document, never the
    # system's, so that a PDF comes out byte for byte the same on every machine.
    fonts = [str(f) for f in [a.doc.parent / "fonts"] if f.is_dir()]
    pkgs = str(a.packages)
    pdf = a.doc.with_suffix(".pdf")
    typst.compile(
        str(a.doc), output=str(pdf), package_path=pkgs, font_paths=fonts, ignore_system_fonts=True
    )
    print(f"wrote {pdf}")
    if a.png:
        pattern = str(a.doc.with_name(a.doc.stem + "-{p}.png"))
        typst.compile(
            str(a.doc),
            output=pattern,
            format="png",
            ppi=110,
            package_path=pkgs,
            font_paths=fonts,
            ignore_system_fonts=True,
        )
        print(f"wrote {pattern}")
    if a.all:
        phone = a.doc.with_name(a.doc.stem + "-phone.pdf")
        typst.compile(
            str(a.doc),
            output=str(phone),
            sys_inputs={"edition": "phone"},
            package_path=pkgs,
            font_paths=fonts,
            ignore_system_fonts=True,
        )
        print(f"wrote {phone}")
        html = a.doc.with_suffix(".html")
        typst.compile(
            str(a.doc),
            output=str(html),
            format="html",
            package_path=pkgs,
            font_paths=fonts,
            ignore_system_fonts=True,
        )
        print(f"wrote {html}")


if __name__ == "__main__":
    main()
