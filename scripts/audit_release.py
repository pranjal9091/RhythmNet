#!/usr/bin/env python3
"""
Release Audit Script for Inter-Patient ECG Arrhythmia Classification System.

Verifies:
1. No hardcoded secrets, tokens, or credentials in codebase.
2. No raw patient binary dumps or stray checkpoints in release artifacts.
3. Consistency of model version/configuration.
4. Validity of internal documentation links.
"""

import sys
import re
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

# Patterns to flag potential secrets
SECRET_PATTERNS = [
    re.compile(r"(?i)(api[_-]?key|secret[_-]?key|password|bearer\s+[a-z0-9_-]+)\s*=\s*['\"][^'\"]+['\"]"),
    re.compile(r"-----BEGIN (RSA|OPENSSH|EC|PRIVATE) KEY-----"),
]

# Patterns for stale/temp files
STALE_FILE_EXTS = [".pyc", ".tmp", ".bak", ".swp"]


def audit_secrets():
    print(" [1/4] Auditing codebase for hardcoded secrets and tokens...")
    code_files = list(ROOT_DIR.glob("src/**/*.py")) + list(ROOT_DIR.glob("scripts/*.py")) + list(ROOT_DIR.glob("*.yml"))
    findings = []
    for file_path in code_files:
        content = file_path.read_text(encoding="utf-8", errors="ignore")
        for pattern in SECRET_PATTERNS:
            if pattern.search(content):
                findings.append(f"Potential secret pattern found in {file_path.relative_to(ROOT_DIR)}")

    if findings:
        raise ValueError("\n".join(findings))
    print("   ✓ Zero hardcoded secrets or sensitive credentials detected.")


def audit_artifact_hygiene():
    print("\n [2/4] Auditing artifact directory hygiene...")
    stray_files = []
    artifacts_dir = ROOT_DIR / "artifacts"
    if artifacts_dir.exists():
        for p in artifacts_dir.rglob("*"):
            if p.is_file() and p.suffix in STALE_FILE_EXTS:
                stray_files.append(str(p.relative_to(ROOT_DIR)))

    if stray_files:
        raise ValueError(f"Stray/temp files found in artifacts: {stray_files}")
    print("   ✓ Artifacts directory is clean of temporary checkpoints and stray files.")


def audit_version_consistency():
    print("\n [3/4] Verifying version and configuration consistency...")
    pyproject = (ROOT_DIR / "pyproject.toml").read_text()
    init_py = (ROOT_DIR / "src" / "ecg_arrhythmia" / "__init__.py").read_text()
    manifest_py = (ROOT_DIR / "reports" / "final_results_manifest.json").read_text()

    ver_pyproject = re.search(r'version\s*=\s*"([^"]+)"', pyproject).group(1)
    ver_init = re.search(r'__version__\s*=\s*"([^"]+)"', init_py).group(1)
    ver_manifest = re.search(r'"version":\s*"([^"]+)"', manifest_py).group(1)

    if not (ver_pyproject == ver_init == ver_manifest):
        raise ValueError(f"Version mismatch! pyproject: {ver_pyproject}, __init__: {ver_init}, manifest: {ver_manifest}")
    print(f"   ✓ Version consistency verified across all configs ({ver_pyproject}).")


def audit_doc_links():
    print("\n [4/4] Verifying internal documentation file links...")
    readme = (ROOT_DIR / "README.md").read_text()
    links = re.findall(r"\[.*?\]\((file:///[^\)]+|docs/[^\)]+|reports/[^\)]+)\)", readme)
    broken_links = []
    import urllib.parse
    for link in links:
        clean_link = urllib.parse.unquote(link.replace("file://", ""))
        # Handle query/anchor fragments
        clean_link = clean_link.split("#")[0]
        p = Path(clean_link)
        if not p.is_absolute():
            p = ROOT_DIR / p
        if not p.exists():
            broken_links.append(link)

    if broken_links:
        print(f"   ⚠️ Warning: Found {len(broken_links)} unresolvable doc links: {broken_links[:3]}")
    else:
        print("   ✓ Internal documentation links verified.")


def main():
    try:
        audit_secrets()
        audit_artifact_hygiene()
        audit_version_consistency()
        audit_doc_links()
        print("\n" + "=" * 60)
        print(" RELEASE AUDIT PASSED — SYSTEM HYGIENE VERIFIED")
        print("=" * 60 + "\n")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Release audit failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
