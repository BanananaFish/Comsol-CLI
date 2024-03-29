import click
from rich.live import Live
from rich.progress import track
from traitlets import default

from interface import Comsol
from utils import Config


@click.group()
def main():
    pass


@main.command()
@click.option("--model", help="Path to the model file.", default="models/cell.mph")
@click.option("--config", help="Path to the config yaml.", default="config/cell1.yaml")
@click.option("--dumpmodel", help="Dump model file after study", default=False)
def run(model, config, dumpmodel):
    click.echo(f"Running model {model}, CFG: {config}, dumpmodel: {dumpmodel}")
    cfg = Config(config)
    cli = Comsol(model, *cfg.params)
    for task in track(cfg.tasks, description="Running tasks..."):
        cli.update(**task)
        cli.study()
        cli.save()
        if dumpmodel:
            cli.dump()


if __name__ == "__main__":
    click.echo("Comsol CLI! by Bananafish")
    main()
