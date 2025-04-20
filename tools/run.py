from datetime import datetime as dt
from pathlib import Path

import click
from loguru import logger

from pipelines import video_etl_pipeline


@click.command(
    help="""Run the Video ETL pipeline.

Example:

  python run.py --config video_etl_pipeline.yaml
"""
)
@click.option("--config", required=True, help="Path to the video ETL config file.")
@click.option("--no-cache", is_flag=True, default=False, help="Disable caching for the pipeline run.")
def main(config: str, no_cache: bool):
    config_path = Path(config).resolve()
    assert config_path.exists(), f"Config not found: {config_path}"

    run_name = f"video_etl_pipeline_run_{dt.now():%Y_%m_%d_%H_%M_%S}"
    pipeline_args = {
        "config_path": config_path,
        "run_name": run_name,
        "enable_cache": not no_cache,
    }

    logger.info(f"Running video_etl_pipeline with config: {config_path}")
    video_etl_pipeline.with_options(**pipeline_args)()


if __name__ == "__main__":
    main()
