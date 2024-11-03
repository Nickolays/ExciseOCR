import hydra
from hydra.utils import instantiate
import pandas as pd
from omegaconf import OmegaConf, DictConfig
import easyocr
import cv2
import numpy as np


import os
print(os.getcwd())
print(os.listdir())
from src.scripts.utils import load_image, preprocess

import logging, os
log = logging.getLogger(__name__)
# dvc exp init -i  This command will guide you to set up a default stage in dvc.yaml.

reader = easyocr.Reader(['en'])

# python3 app.py hydra.run.dir=my_folder
# Посредством декоратора @hydra.main подгружаем конфиги из указанной папки
@hydra.main(version_base=None, config_path='config', config_name='config')
def my_app(cfg: DictConfig):

    # Get data
    print(OmegaConf.to_yaml(cfg))
    print("Reading data ...")
    
    # input_path = OmegaConf.to_container(cfg.['dataset'])
    input_path = cfg['dataset']['path']
    input_image = load_image(input_path)
    input_image = cv2.rotate(input_image, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # Read model
    params = cfg['params']   # Get hyperparams for model, We convert it to dict
    model = instantiate(cfg['model'])
    # Inference
    predict = model.readtext(input_image, **params)
    # Postprocess
    vals, probs = preprocess(predict)
    print(vals)
    assert isinstance(probs, list)
    for val, prob in zip(vals, probs):
        log.info(f'probability(likelihood) of prediction:   {prob}')
        log.info(f'values:   {val}')

    return None

if __name__ == "__main__":
    my_app()