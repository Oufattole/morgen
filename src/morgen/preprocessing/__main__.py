import copy
import logging
import os
import subprocess
from enum import StrEnum
from importlib.resources import files
from pathlib import Path

import hydra
from omegaconf import DictConfig

logger = logging.getLogger(__name__)

CONFIGS = files("morgen") / "preprocessing" / "configs"


class PIPELINE(StrEnum):
    MIMIC = "mimic"
    MIMIC_FILTERED = "mimic_filtered"
    EHRSHOT_FILTERED = "ehrshot_filtered"


@hydra.main(version_base=None, config_path=str(CONFIGS), config_name="_process_data")
def process_data(cfg: DictConfig):
    input_dir = Path(cfg.input_dir)
    intermediate_dir = Path(cfg.intermediate_dir)
    output_dir = Path(cfg.output_dir)
    do_demo = cfg.do_demo

    # 0. Pre-MTD pre-processing
    logger.info("Pre-MTD pre-processing")
    done_fp = intermediate_dir / ".done"
    if done_fp.exists():  # pragma: no cover
        logger.info("Pre-MTD pre-processing already done, skipping")
    else:
        env = copy.deepcopy(os.environ)
        env["RAW_MEDS_DIR"] = str(input_dir)
        env["MTD_INPUT_DIR"] = str(intermediate_dir)

        if do_demo:
            env["MIN_SUBJECTS_PER_CODE"] = "2"
            env["MIN_EVENTS_PER_SUBJECT"] = "1"

        if cfg.pipeline == PIPELINE.MIMIC:
            pipeline_config_fp = CONFIGS / "_data.yaml"
        elif cfg.pipeline == PIPELINE.MIMIC_FILTERED:
            pipeline_config_fp = CONFIGS / "_data_filtered.yaml"
        elif cfg.pipeline == PIPELINE.EHRSHOT_FILTERED:
            pipeline_config_fp = CONFIGS / "_ehrshot_data_filtered.yaml"
        else:
            raise ValueError(f"Unknown pipeline: {cfg.pipeline}")

        cmd = [
            "MEDS_transform-pipeline",
            f"pipeline_config_fp={pipeline_config_fp!s}",
        ]
        logger.info(f"Running command: {' '.join(cmd)}")

        result = subprocess.run(cmd, env=env, capture_output=True, check=False)
        if result.returncode != 0:  # pragma: no cover
            logger.error("Error running MEDS_transform-pipeline")
            logger.error(result.stdout.decode())
            logger.error(result.stderr.decode())
            raise RuntimeError("Error running MEDS_transform-pipeline")

        logger.info("Pre-MTD pre-processing done")
        done_fp.touch()

    # 1. Run MTD pre-processing
    logger.info("Running MTD pre-processing")
    done_fp = output_dir / ".done"

    if done_fp.exists():  # pragma: no cover
        logger.info("MTD pre-processing already done, skipping")
    else:
        env = copy.deepcopy(os.environ)

        cmd = [
            "MTD_preprocess",
            f"MEDS_dataset_dir={intermediate_dir!s}",
            f"output_dir={output_dir!s}",
        ]
        logger.info(f"Running command: {' '.join(cmd)}")

        result = subprocess.run(cmd, env=env, capture_output=True, check=False)
        if result.returncode != 0:  # pragma: no cover
            logger.error("Error running MTD_preprocess")
            logger.error(result.stderr.decode())
            raise RuntimeError("Error running MTD_preprocess")

        logger.info("MTD pre-processing done")
        done_fp.touch()

    return
