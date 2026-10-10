"""Explicit entry point for archived history analyses."""
from analysis_scripts.workflow import cli

if __name__ == "__main__":
    raise SystemExit(cli(group="history"))
