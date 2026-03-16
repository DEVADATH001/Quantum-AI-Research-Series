"""Author: DEVADATH H K

Project: Quantum AI Research Series

Repository authorship validation script."""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

OFFICIAL_AUTHOR = "DEVADATH H K"
TEXT_EXTENSIONS = {
    ".cff",
    ".cfg",
    ".env",
    ".example",
    ".ini",
    ".ipynb",
    ".json",
    ".md",
    ".py",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
TEXT_FILENAMES = {"LICENSE"}
SKIP_DIRS = {".git", ".pytest_cache", "__pycache__", ".venv", "venv", "node_modules"}
SECTION_HEADER_RE = re.compile(r"^\s{0,3}#+")
AUTHORSHIP_PATTERNS = [
    re.compile(r"\bauthor\b", re.IGNORECASE),
    re.compile(r"\bauthors\b", re.IGNORECASE),
    re.compile(r"created by", re.IGNORECASE),
    re.compile(r"developed by", re.IGNORECASE),
    re.compile(r"\bmaintainer\b", re.IGNORECASE),
    re.compile(r"\bcredits\b", re.IGNORECASE),
    re.compile(r"\bcontributors\b", re.IGNORECASE),
    re.compile(r"\bcopyright\b", re.IGNORECASE),
]


@dataclass
class Finding:
    path: Path
    line_number: int
    detected: str


def should_scan(path: Path) -> bool:
    if any(part in SKIP_DIRS for part in path.parts):
        return False
    return path.suffix.lower() in TEXT_EXTENSIONS or path.name in TEXT_FILENAMES


def is_probably_output_noise(line: str) -> bool:
    stripped = line.strip()
    if "\"image/png\":" in stripped:
        return True
    if len(stripped) < 240:
        return False
    printable = sum(ch.isalnum() or ch in "+/=:,_-\"" for ch in stripped)
    return printable / max(len(stripped), 1) > 0.95


def contains_authorship_keyword(line: str) -> bool:
    return any(pattern.search(line) for pattern in AUTHORSHIP_PATTERNS)


def author_context(lines: list[str], index: int) -> list[tuple[int, str]]:
    context: list[tuple[int, str]] = []
    for offset in range(index + 1, min(index + 5, len(lines))):
        candidate = lines[offset]
        stripped = candidate.strip()
        if not stripped:
            break
        if SECTION_HEADER_RE.match(stripped):
            break
        context.append((offset + 1, candidate))
    return context


def scan_file(path: Path) -> list[Finding]:
    if path.name == "check_authorship.py":
        return []

    text = path.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()
    findings: list[Finding] = []

    for index, line in enumerate(lines):
        if is_probably_output_noise(line):
            continue
        lowered = line.lower()
        if "__author__" in line and OFFICIAL_AUTHOR not in line:
            findings.append(Finding(path=path, line_number=index + 1, detected=line.strip()))
            continue
        if re.search(r"^\s*copyright\b", line, re.IGNORECASE):
            if OFFICIAL_AUTHOR not in line and "copyright holders" not in lowered:
                findings.append(Finding(path=path, line_number=index + 1, detected=line.strip()))
            continue
        if not contains_authorship_keyword(line):
            continue
        if OFFICIAL_AUTHOR in line:
            continue
        if line.strip().lower() == "authors:":
            context = author_context(lines, index)
            if any(OFFICIAL_AUTHOR in candidate for _, candidate in context):
                continue
        if SECTION_HEADER_RE.match(line.strip()):
            context = author_context(lines, index)
            if not any(OFFICIAL_AUTHOR in candidate for _, candidate in context):
                findings.append(Finding(path=path, line_number=index + 1, detected=line.strip()))
            continue
        if "=" in line or ":" in line:
            findings.append(Finding(path=path, line_number=index + 1, detected=line.strip()))

    unique: dict[tuple[str, int, str], Finding] = {}
    for finding in findings:
        unique[(str(finding.path), finding.line_number, finding.detected)] = finding
    return list(unique.values())


def iter_scannable_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file() and should_scan(path.relative_to(root)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate repository authorship references.")
    parser.add_argument(
        "--root",
        default=Path(__file__).resolve().parents[1],
        type=Path,
        help="Repository root to scan.",
    )
    parser.add_argument(
        "--audit",
        action="store_true",
        help="Print findings instead of only the summary.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    scanned_files = iter_scannable_files(root)
    findings: list[Finding] = []
    for path in scanned_files:
        findings.extend(scan_file(path))

    if args.audit:
        for finding in findings:
            relative_path = finding.path.relative_to(root)
            print(f"{relative_path}:{finding.line_number}")
            print(f"  detected: {finding.detected}")
            print(f"  expected: Author: {OFFICIAL_AUTHOR}")

    print(f"Scanned files: {len(scanned_files)}")
    print(f"Violations detected: {len(findings)}")

    if findings:
        return 1

    print(f"Official author confirmed: {OFFICIAL_AUTHOR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
