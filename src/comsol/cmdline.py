import click
from rich import print as rprint
from rich.progress import track


@click.group()
def main():
    pass


@main.command()
@click.option("--model", help="Path to the model file.", default="models/cell.mph")
@click.option("--config", help="Path to the config yaml.", default="config/cell.yaml")
@click.option("--dump", help="Dump model file after study", is_flag=True)
def run(model, config, dump):
    rprint(":baseball: [bold magenta italic]Comsol CLI! by Bananafish[/]")
    from comsol.interface import Comsol
    from comsol.utils import Config

    click.echo(f"Running model {model}, CFG: {config}, dump: {dump}")
    cfg = Config(config)
    cli = Comsol(model, *cfg.params)
    for task in track(cfg.tasks, description="Running tasks..."):
        cli.update(**task)
        cli.study()
        cli.save()
        if dump:
            cli.dump()


@main.command()
@click.option("--saved", help="Path to saved pickles.", default="export/saved")
@click.option("--config", help="Path to the config yaml.", default="config/cell.yaml")
@click.option("--ckpt_path", help="Path to the checkpoint file.", default="ckpt")
def train(saved, config, ckpt_path):
    rprint(":baseball: [bold magenta italic]Comsol CLI! by Bananafish[/]")
    click.echo(
        f"Training model with saved {saved}, CFG: {config}, ckpt_path: {ckpt_path}"
    )
    from comsol.model import MLP
    from comsol.utils import BandDataset, Config, Trainer

    cfg = Config(config)
    dataset = BandDataset(saved)
    model = MLP()
    trainer = Trainer(dataset, model, cfg)
    trainer.train()


@main.command()
@click.option("--ckpt", help="Path to the checkpoint file.", default="ckpt/latest.pth")
@click.option("--saved", help="Path to saved pickles.", default="export/saved")
def ga(ckpt, saved):
    rprint(":baseball: [bold magenta italic]Comsol CLI! by Bananafish[/]")
    from comsol.ga import fit

    fit(ckpt, saved)


if __name__ == "__main__":
    main()
