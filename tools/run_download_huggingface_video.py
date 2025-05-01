from datetime import datetime as dt

import click

from pipelines.download_huggingface_video import download_huggingface_video


@click.command(help="Run the Hugging Face video download pipeline.")
@click.option("--no-cache", is_flag=True, default=False, help="Disable caching.")
def main(no_cache: bool):
    run_name = f"download_huggingface_video_run_{dt.now():%Y_%m_%d_%H_%M_%S}"
    download_huggingface_video.with_options(run_name=run_name, enable_cache=not no_cache)()


if __name__ == "__main__":
    main()
