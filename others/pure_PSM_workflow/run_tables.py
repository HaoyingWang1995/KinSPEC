"""Explicit entry point for archived tables analyses."""
from analysis_scripts.workflow import cli

if __name__ == "__main__":
    raise SystemExit(cli(group="tables"))
