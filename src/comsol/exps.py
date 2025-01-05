from comsol.model import MLP
from comsol.utils import Config, Trainer, EarlyStop
from comsol.datasets import FieldDataset
from comsol.console import console
from itertools import product

if __name__ == "__main__":
    cfg = Config("config/cell.yaml")

    lay_nums = [4, 6]
    ths = [0.6]
    lrs = [1e-5]

    task_nums = len(lay_nums) * len(ths) * len(lrs)
    for i, (lay, th, lr) in enumerate(product(lay_nums, ths, lrs)):
        console.log(f"[{i}/{task_nums}]now: {lay, th, lr}")
        cfg["train"]["lr"] = lr
        cfg["train"]["hidden_layers"] = lay
        cfg["train"]["threshold"] = th
        dataset = FieldDataset("exports/5-8", cfg)
        model = MLP(cfg)
        trainer = Trainer(dataset, model, cfg, "ckpt/5-8-hypertest")
        try:
            trainer.train()
        except EarlyStop:
            trainer.save_ckpt(f"earlystop_best_{trainer.best_loss:.6f}", best=True)
        except KeyboardInterrupt:
            break
        except Exception as e:
            trainer.logging(f"bad things...: {e}")
            trainer.save_ckpt(f"earlystop_best_{trainer.best_loss:.6f}", best=True)
            continue
        finally:
            console.log(f"[{i}/{task_nums}]now: {lay, th, lr}")