from datetime import datetime as dt

import click

from pipelines.chunk_subtitles import chunk_subtitles


@click.command(help="Run the subtitle chunking pipeline.")
@click.option("--no-cache", is_flag=True, default=False, help="Disable caching.")
def main(no_cache: bool):
    run_name = f"chunk_subtitles_run_{dt.now():%Y_%m_%d_%H_%M_%S}"
    chunk_subtitles.with_options(run_name=run_name, enable_cache=not no_cache)()


if __name__ == "__main__":
    main()
