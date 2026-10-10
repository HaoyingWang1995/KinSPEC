"""Main entry point for the attribution analyses."""
from analysis_scripts.workflow import cli

if __name__ == "__main__":
    raise SystemExit(cli(group="attribution"))
