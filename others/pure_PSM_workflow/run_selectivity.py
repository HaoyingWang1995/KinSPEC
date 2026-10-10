"""Main entry point for the selectivity analyses."""
from analysis_scripts.workflow import cli

if __name__ == "__main__":
    raise SystemExit(cli(group="selectivity"))
