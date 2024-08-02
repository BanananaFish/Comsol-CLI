import click
import jpype
from rich.console import Console
from rich.progress import MofNCompleteColumn, Progress, SpinnerColumn

from comsol.console import console
from comsol.datasets import BDDataset
# from rich.traceback import install
# install(show_locals=True)


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

    with Progress(
        SpinnerColumn(),
        *Progress.get_default_columns(),
        MofNCompleteColumn(),
        console=console,
    ) as progress:
        study_task = progress.add_task("[cyan]Study", total=len(cfg.tasks))
        error_count = 0
        for task in cfg.tasks:
            try:
                cli.update(**task)
                cli.study()
                cli.save_cfg(cfg, task)
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
            except jpype.JException as e:
                error_count += 1
                cli.study_count += 1
                console.log(f"[orange1]Java Error: {e}")
                progress.update(
                    study_task, description=f"[cyan]Study (Errors: {error_count})"
                )
            except Exception as e:
                error_count += 1
                cli.study_count += 1
                console.log(f"[red]Error: {e}")
                progress.update(
                    study_task, description=f"[cyan]Study (Errors: {error_count})"
                )
            finally:
                progress.update(study_task, advance=1)


@main.command()
@click.option("--exps", help="Path to saved exps.", default="exp1")
@click.option("--config", help="Path to the config yaml.", default="config/cell.yaml")
@click.option("--ckpt_path", help="Path to the checkpoint file.", default="ckpt")
def train(exps, config, ckpt_path):
    console.log(":baseball: [bold magenta italic]Comsol CLI! by Bananafish[/]")
    click.echo(
        f"Training model with saved {exps}, CFG: {config}, ckpt_path: {ckpt_path}"
    )
    from comsol.model import MLP
    from comsol.utils import Config, Trainer, EarlyStop
    from comsol.datasets import FieldDataset, BDDataset

    cfg = Config(config)
    if cfg["dataset"]["type"] == "field":
        dataset = FieldDataset(exps, cfg)
    elif cfg["dataset"]["type"] == "bd":
        dataset = BDDataset(exps)
    else:
        raise ValueError(f"Unknown dataset type: {cfg['dataset']['type']}")
    model = MLP(cfg)
    trainer = Trainer(dataset, model, cfg, ckpt_path)
    try:
        trainer.train()
    except (KeyboardInterrupt, EarlyStop):
        trainer.save_ckpt(f"earlystop_best_{trainer.best_loss:.6f}", best=True)
        
        
@main.command()
@click.option("--exps", help="Path to saved exps.", default="exp1")
@click.option("--config", help="Path to the config yaml.", default="config/cell.yaml")
@click.option("--ckpt", help="Path to the checkpoint file.")
def test(exps, config, ckpt):
    console.log(":baseball: [bold magenta italic]Comsol CLI! by Bananafish[/]")
    click.echo(
        f"Testing model with saved {exps}, CFG: {config}, ckpt: {ckpt}"
    )
    from comsol.model import MLP
    from comsol.utils import Config, Trainer
    from comsol.datasets import FieldDataset
    import torch

    cfg = Config(config)
    if cfg["dataset"]["type"] == "field":
        dataset = FieldDataset(exps, cfg)
    elif cfg["dataset"]["type"] == "bd":
        dataset = BDDataset(exps)
    model = MLP(cfg)
    model.load_state_dict(torch.load(ckpt))
    trainer = Trainer(dataset, model, cfg, ckpt_path="ckpt/test", test=True)
    trainer.test()


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
