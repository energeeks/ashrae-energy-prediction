import random

import click
import yaml

from src.models.train_ctb_model import train_ctb_model
from src.models.train_lgbm_model import train_lgbm_model
from src.models.train_xgb_model import train_xgb_model


@click.command()
@click.argument('model')
@click.argument('mode')
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def click_main(model, mode, input_filepath, output_filepath):
    main(model, mode, input_filepath, output_filepath)


def main(model, mode, input_filepath, output_filepath):
    """
    Collects prepared data and starts training a model. Parameters can be specified by editing src/config.yml.
    Keep in mind that XGBoost does not accept NA Values. So for this model the corresponding function to set these to
    zero in the preprocessing steps has to be set to true.
    :param model: Specifies the model to run. Options are xgb (XGBoost), lgbm (LightGBM), ctb (CatBoost)
    :param mode: Specifies mode to run. Options are full (no validation set, single fold), cv (cross validation),
    by_meter (training by meter type), by_building (training by building id).
    :param input_filepath: Directory that contains the processed data.
    :param output_filepath: Directory that will contain the trained models.
    """
    random.seed(1337)
    with open("src/config.yml", 'r') as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)

    if model == "xgb":
        train_xgb_model(mode, input_filepath, output_filepath, cfg)

    if model == "lgbm":
        train_lgbm_model(mode, input_filepath, output_filepath, cfg)

    if model == "ctb":
        train_ctb_model(mode, input_filepath, output_filepath, cfg)


if __name__ == '__main__':
    click_main()
