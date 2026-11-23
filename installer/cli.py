#!/usr/bin/env python3
"""CLI installer for Pale Fire configured from a Semantic Croissant profile (.env)."""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def parse_env_file(path: Path) -> dict[str, str]:
    cfg: dict[str, str] = {}
    if not path.is_file():
        return cfg
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        cfg[k.strip()] = v.strip().strip('"').strip("'")
    return cfg


def resolve_profile_path(profile_arg: str, palefire_dir: Path) -> Path:
    p = Path(profile_arg)
    if p.is_file():
        return p.resolve()
    repo_root = palefire_dir.parent
    candidates = [
        repo_root / profile_arg,
        repo_root / "profiles" / f"{profile_arg}.env",
        repo_root / "profiles" / profile_arg,
    ]
    for c in candidates:
        if c.is_file():
            return c.resolve()
    raise FileNotFoundError(f"Profile file not found for '{profile_arg}'.")


def main():
    palefire_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Configure palefire/.env from a Semantic Croissant profile and build/start Pale Fire."
    )
    parser.add_argument(
        "--profile",
        "-p",
        default="profiles/default.env",
        help="Path or name of Semantic Croissant profile (e.g. profiles/default.env or default)",
    )
    parser.add_argument("--non-interactive", action="store_true", help="Skip prompts and use profile defaults")
    parser.add_argument("--no-build", action="store_true", help="Skip docker compose build")
    parser.add_argument("--no-start", action="store_true", help="Skip docker compose up -d")
    args = parser.parse_args()

    profile_path = resolve_profile_path(args.profile, palefire_dir)
    cfg = parse_env_file(profile_path)
    print(f"Using Semantic Croissant profile: {profile_path}")

    mcp_port = cfg.get("MCP_PORT", "7070")
    api_port = cfg.get("API_PORT", "7013")
    ddi_port = cfg.get("DDI_CDI_PORT", "8083")
    ollama_host = cfg.get("OLLAMA_HOST", "http://host.docker.internal:11434")
    visual_model = cfg.get("OLLAMA_VISUAL_MODEL", "gemma4:31b-cloud")

    if not args.non_interactive:
        val = input(f"OLLAMA_HOST [{ollama_host}]: ").strip()
        if val:
            ollama_host = val
        val = input(f"OLLAMA_VISUAL_MODEL [{visual_model}]: ").strip()
        if val:
            visual_model = val

    pf_env = palefire_dir / ".env"
    pf_env.write_text(
        f"# Generated from {profile_path}\n"
        f"SEMANTIC_CROISSANT_MCP_URL=http://localhost:{mcp_port}/mcp\n"
        f"SEMANTIC_CROISSANT_SSE_URL=http://localhost:{mcp_port}/sse\n"
        f"SEMANTIC_CROISSANT_API_URL=http://localhost:{api_port}\n"
        f"OLLAMA_HOST={ollama_host}\n"
        f"OLLAMA_VISUAL_MODEL={visual_model}\n"
        f"OLLAMA_DEFAULT_MODEL={visual_model}\n"
        f"DDI_CDI_SERVICE=http://localhost:{ddi_port}\n",
        encoding="utf-8",
    )
    print(f"Wrote {pf_env}")

    if not args.no_build:
        res = subprocess.run(
            ["docker", "compose", "-f", "docker-compose.yml", "build", "palefire-api"],
            cwd=str(palefire_dir),
        )
        if res.returncode != 0:
            sys.exit(res.returncode)

    if not args.no_start:
        res = subprocess.run(
            ["docker", "compose", "-f", "docker-compose.yml", "up", "-d", "palefire-api"],
            cwd=str(palefire_dir),
        )
        if res.returncode != 0:
            sys.exit(res.returncode)


if __name__ == "__main__":
    main()
