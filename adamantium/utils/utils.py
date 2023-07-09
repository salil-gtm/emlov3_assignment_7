import warnings
from importlib.util import find_spec
from typing import Callable

from omegaconf import DictConfig

from adamantium.utils import pylogger, rich_utils

log = pylogger.get_pylogger(__name__)


def extras(cfg: DictConfig) -> None:
    """Applies optional utilities before the task is started.

    Utilities:
    - Ignoring python warnings
    - Setting tags from command line
    - Rich config printing
    """

    # return if no `extras` config
    if not cfg.get("extras"):
        log.warning("Extras config not found! <cfg.extras=null>")
        return

    # disable python warnings
    if cfg.extras.get("ignore_warnings"):
        log.info("Disabling python warnings! <cfg.extras.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    # prompt user to input tags from command line if none are provided in the config
    if cfg.extras.get("enforce_tags"):
        log.info("Enforcing tags! <cfg.extras.enforce_tags=True>")
        rich_utils.enforce_tags(cfg, save_to_file=True)

    # pretty print config tree using Rich library
    if cfg.extras.get("print_config"):
        log.info("Printing config tree with Rich! <cfg.extras.print_config=True>")
        rich_utils.print_config_tree(cfg, resolve=True, save_to_file=True)


def task_wrapper(task_func: Callable) -> Callable:
    """Optional decorator that controls the failure behavior when executing the task function.

    This wrapper can be used to:
    - make sure loggers are closed even if the task function raises an exception (prevents multirun failure)
    - save the exception to a `.log` file
    - mark the run as failed with a dedicated file in the `logs/` folder (so we can find and rerun it later)
    - etc. (adjust depending on your needs)

    Example:
    ```
    @utils.task_wrapper
    def train(cfg: DictConfig) -> Tuple[dict, dict]:

        ...

        return metric_dict, object_dict
    ```
    """

    def wrap(cfg: DictConfig):
        # execute the task
        try:
            metric_dict, object_dict = task_func(cfg=cfg)

        # things to do if exception occurs
        except Exception as ex:
            # save exception to `.log` file
            log.exception("")

            # some hyperparameter combinations might be invalid or cause out-of-memory errors
            # so when using hparam search plugins like Optuna, you might want to disable
            # raising the below exception to avoid multirun failure
            raise ex

        # things to always do after either success or exception
        finally:
            # display output dir path in terminal
            log.info(f"Output dir: {cfg.paths.output_dir}")

            # always close wandb run (even if exception occurs so multirun won't fail)
            if find_spec("wandb"):  # check if wandb is installed
                import wandb

                if wandb.run:
                    log.info("Closing wandb!")
                    wandb.finish()

        return metric_dict, object_dict

    return wrap


def get_metric_value(metric_dict: dict, metric_name: str) -> float:
    """Safely retrieves value of the metric logged in LightningModule."""

    if not metric_name:
        log.info("Metric name is None! Skipping metric value retrieval...")
        return None

    if metric_name not in metric_dict:
        raise Exception(
            f"Metric value not found! <metric_name={metric_name}>\n"
            "Make sure metric name logged in LightningModule is correct!\n"
            "Make sure `optimized_metric` name in `hparams_search` config is correct!"
        )

    metric_value = metric_dict[metric_name].item()
    log.info(f"Retrieved metric value! <{metric_name}={metric_value}>")

    return metric_value


def get_latest_checkpoint(ckpt_path: str) -> str:
    """Returns the latest checkpoint from the hydra output directory."""

    # get all checkpoints from the hydra output directory
    ckpt_list_dates = os.listdir(ckpt_path)

    # sort the list of checkpoints by creation time
    ckpt_list_dates = sorted(
        ckpt_list_dates, key=lambda x: os.path.getctime(os.path.join(ckpt_path, x))
    )

    # get the latest checkpoint
    ckpt_latest_date = ckpt_list_dates[-1]

    # get the checkpoint name
    ckpt_list_time = os.listdir(os.path.join(ckpt_path, ckpt_latest_date))

    # sort the list of checkpoints by creation time
    ckpt_list_time = sorted(
        ckpt_list_time,
        key=lambda x: os.path.getctime(os.path.join(ckpt_path, ckpt_latest_date, x)),
    )

    # keep only the folders which has checkpoints folder inside it
    ckpt_list_time = [
        x
        for x in ckpt_list_time
        if os.path.isdir(os.path.join(ckpt_path, ckpt_latest_date, x, "checkpoints"))
    ]

    # get the latest checkpoint
    ckpt_latest_time = ckpt_list_time[-1]

    # get the checkpoint name
    ckpt_list_epochs = os.listdir(
        os.path.join(ckpt_path, ckpt_latest_date, ckpt_latest_time, "checkpoints")
    )

    # sort the list of checkpoints by epoch
    ckpt_list_epochs = sorted(
        ckpt_list_epochs, key=lambda x: int(x.split("=")[-1].split(".")[0])
    )

    # get the latest checkpoint
    checkpoint_name = ckpt_list_epochs[-1]

    # get the checkpoint path
    checkpoint_path = os.path.join(
        ckpt_path, ckpt_latest_date, ckpt_latest_time, "checkpoints", checkpoint_name
    )

    return checkpoint_path
