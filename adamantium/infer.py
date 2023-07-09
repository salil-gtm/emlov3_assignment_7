from typing import List, Tuple

import hydra
import torch
from lightning import LightningModule
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig
from pprint import pprint

from torchvision import transforms as T
from PIL import Image

from adamantium import utils

log = utils.get_pylogger(__name__)


@utils.task_wrapper
def infer(cfg: DictConfig) -> Tuple[dict, dict]:
    """Infers using checkpoint on a given image.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.

    Returns:
        Tuple[dict, dict]: Dict with metrics and dict with all instantiated objects.
    """

    assert cfg.ckpt_path
    assert cfg.image_path

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating loggers...")
    logger: List[Logger] = utils.instantiate_loggers(cfg.get("logger"))

    object_dict = {"cfg": cfg, "model": model}

    transform = T.Compose(
        [
            T.Resize((32, 32)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    classes = ("cat", "dog")

    if logger:
        log.info("Logging hyperparameters!")
        utils.log_hyperparameters(object_dict)

    if cfg.get("ckpt_path"):
        if cfg.get("image_path"):
            log.info("Starting Infering!")
            checkpoint_path = utils.get_latest_checkpoint(cfg.ckpt_path)
            model = model.load_from_checkpoint(checkpoint_path)
            image = Image.open(cfg.image_path)
            image = transform(image)
            image = image.unsqueeze(0)
            model.eval()
            output = model(image)
            prob = torch.nn.functional.softmax(output, dim=1)
            prob_json = {classes[0]: prob[0][0].item(), classes[1]: prob[0][1].item()}
            print("\n")
            pprint(prob_json)
            print("\n")
            return prob_json, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="infer.yaml")
def main(cfg: DictConfig) -> None:
    infer(cfg)


if __name__ == "__main__":
    main()
