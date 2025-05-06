from datetime import datetime as dt

import click

from pipelines.save_vectors import save_vectors


@click.command(help="Run the saving vectors to Qdrant pipeline.")
@click.option("--no-cache", is_flag=True, default=False, help="Disable caching.")
def main(no_cache: bool):
    run_name = f"save_vectors_run_{dt.now():%Y_%m_%d_%H_%M_%S}"
    save_vectors.with_options(run_name=run_name, enable_cache=not no_cache)()


if __name__ == "__main__":
    main()
