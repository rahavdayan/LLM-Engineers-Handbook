from datetime import datetime as dt

import click

from pipelines import video_etl_pipeline


@click.command(
    help="""Run the Video ETL pipeline.

Example:

  poetry run python -m tools.run --no-cache
"""
)
@click.option("--no-cache", is_flag=True, default=False, help="Disable caching for the pipeline run.")
def main(no_cache: bool):
    run_name = f"video_etl_pipeline_run_{dt.now():%Y_%m_%d_%H_%M_%S}"

    video_etl_pipeline.with_options(run_name=run_name, enable_cache=not no_cache)()


if __name__ == "__main__":
    main()
