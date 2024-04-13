import click
from rich.console import Console
from rich.progress import Progress
from traitlets import default

from comsol.console import console


@click.group()
def main():
    pass


@main.command()
@click.option("--model", help="Path to the model file.", default="models/cell.mph")
@click.option("--config", help="Path to the config yaml.", default="config/cell.yaml")
@click.option("--dump", help="Dump model file after study", is_flag=True)
@click.option("--raw", help="Save raw exported data", is_flag=True)
@click.option("--avg", help="Save grid avg exported data, infer in cfg", is_flag=True)
@click.option("--sample", help="sampled frac", default=0.1)
def run(model, config, dump, raw, avg, sample):
    console.log(":baseball: [bold magenta italic]Comsol CLI! by Bananafish[/]")
    from comsol.interface import Comsol
    from comsol.utils import Config

    console.log(
        f"Running model {model}, CFG: {config}, dump: {dump}, raw: {raw}, sample: {sample}, avg: {avg}"
    )
    cfg = Config(config)
    cli = Comsol(model, cfg["export"]["dir"], *cfg.params)

    with Progress(console=console) as progress:
        study_tast = progress.add_task("[cyan]Study", total=len(cfg.tasks))
        for task in cfg.tasks:
            cli.update(**task)
            cli.study()
            if raw or avg or sample:
                if raw:
                    cli.save_raw_data()
                if avg:
                    cli.save_avg_data(cfg["export"]["grid_avg"])
                if sample:
                    cli.save_sampled_data(
                        frac=sample,
                        sample_keys=cfg["export"]["sample_keys"],
                        progress=progress,
                    )
            else:
                console.log("[red]No save option selected")
            if dump:
                cli.dump()
            progress.update(study_tast, advance=1)


@main.command()
@click.option("--saved", help="Path to saved pickles.", default="export/saved")
@click.option("--config", help="Path to the config yaml.", default="config/cell.yaml")
@click.option("--ckpt_path", help="Path to the checkpoint file.", default="ckpt")
def train(saved, config, ckpt_path):
    console.log(":baseball: [bold magenta italic]Comsol CLI! by Bananafish[/]")
    click.echo(
        f"Training model with saved {saved}, CFG: {config}, ckpt_path: {ckpt_path}"
    )
    from comsol.model import MLP
    from comsol.utils import BandDataset, Config, Trainer

    cfg = Config(config)
    dataset = BandDataset(saved, cfg)
    model = MLP(cfg)
    trainer = Trainer(dataset, model, cfg, ckpt_path)
    try:
        trainer.train()
    except (KeyboardInterrupt, Trainer.EarlyStop):
        trainer.save_ckpt(f"earlystop_best_{trainer.best_loss:.6f}", best=True)


@main.command()
@click.option("--ckpt", help="Path to the checkpoint file.", default="ckpt/latest.pth")
@click.option("--config", help="Path to the config yaml.", default="config/cell.yaml")
@click.option("--saved", help="Path to saved pickles.", default="export/saved")
def ga(ckpt, config, saved):
    console.log(":baseball: [bold magenta italic]Comsol CLI! by Bananafish[/]")
    from comsol.ga import fit
    from comsol.utils import Config

    cfg = Config(config)
    fit(ckpt, saved, cfg)


if __name__ == "__main__":
    main()
