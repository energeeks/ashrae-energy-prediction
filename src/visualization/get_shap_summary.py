# This script is for creating a shap summary plot with a specified model
# and dataset. Result will be saved in reports/figures

import shap
import lightgbm as lgb
import pandas as pd
import click
import matplotlib.pyplot as plt


def main():
    model_path = "models/lgbm/0001.txt"
    data_path = "data/processed/test_data.pkl"

    click.echo("Load Model")
    lgbm_model = lgb.Booster(model_file=model_path)

    click.echo("Load Data")
    test_df = pd.read_pickle(data_path)
    del test_df["row_id"]

    click.echo("Sample Rows for plot")
    test_sample = test_df.sample(n=10000)

    click.echo("Get SHAP Values")
    lgbm_model.params['objective'] = 'regression'
    shap_values = shap.TreeExplainer(lgbm_model).shap_values(test_sample)

    click.echo("Save Plot")
    shap.summary_plot(shap_values, test_sample, show=False)
    plt.tight_layout()
    plt.savefig("reports/figures/shap_values.pdf")


if __name__ == '__main__':
        main()