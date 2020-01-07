ASHRAE - Great Energy Predictor III Challenge
====================================

Who are we?
-------------
We are computer science & statistics students at LMU Munich and this project is happening as part of a Data Science Practical. Our plan was to participate in the associated Kaggle Challenge and subsequently build a product surrounding the trained model.

Have fun checking out our stuff!

Cheers

Challenge Description
------------

- Q: How much does it cost to cool a skyscraper in the summer?
- A: A lot! And not just in dollars, but in environmental impact.

Thankfully, significant investments are being made to improve building efficiencies to reduce costs and emissions. So, are the improvements working? That’s where you come in. Current methods of estimation are fragmented and do not scale well. Some assume a specific meter type or don’t work with different building types.

Developing energy savings has two key elements: Forecasting future energy usage without improvements, and forecasting energy use after a specific set of improvements have been implemented, like the installation and purchase of investment-grade meters, whose prices continue to fall. One issue preventing more aggressive growth of the energy markets are the lack of cost-effective, accurate, and scalable procedures for forecasting energy use.

In this competition, you’ll develop accurate predictions of metered building energy usage in the following areas: chilled water, electric, natural gas, hot water, and steam meters. The data comes from over 1,000 buildings over a three-year timeframe.

With better estimates of these energy-saving investments, large scale investors and financial institutions will be more inclined to invest in this area to enable progress in building efficiencies.

The Data Set
---------------

The provided data consists of ~20 mio. rows for training (one year timespan) and ~40 mio. rows for testing (two years timespan). The target variable are the hourly readings from one of four meters {0: electricity, 1: chilledwater, 2: steam, hotwater: 3}. For building the model the data provides following features out of the box:


- building_id --> Foreign key for the building metadata.
- meter ---> The meter id code. Read as {0: electricity, 1: chilledwater, 2: steam, hotwater: 3}
- timestamp --> Hour of the measurement
- site_id --> Identifier of the site the building is on
- primary_use ---> Primary category of activities for the building 
- square_feet --> Floor area of the building
- year_built ---> build year of the building
- floorcount - Number of floors of the building

Further weather data has been provided, which comes with air_temperature, cloud_coverage, dew_temperature, precip_depth_1_hr, sea_level_pressure, wind_direction and wind_speed.

Getting started
---------------

1. **The Data**

   The raw data has to be placed in `data/raw`. A good practice is to download the data via the Kaggle CLI.
    ```
    kaggle competitions download -c ashrae-energy-prediction
    mkdir -p data/raw
    unzip ashrae-energy-prediction.zip -d data/raw
    ```

2. **Use the configuration file**

   The configuration settings are located in `src/config.yml`. This is important to customize feature engineering and model training.

3. **Prepare data for training**

   Using `make data` the data is loaded and the several `.csv`-files will be joined to a consistent data frame. The result is saved in `data/interim`.
   Next use `make features` to conduct the feature engineering process. The result is saved in `data/processed`.

4. **Train a model**

   The frameworks being uses are LightGBM, CatBoost and XGBoost. We personally had our best experiences with LightGBM, but feel free to try differen frameworks or setting. The default settings are the parameters that have been determined through a hyperparameter search.
   To train a model use `make train MODEL=<framework> MODE=<mode>`. For the `MODEL` parameter you can use lgbm (LightGBM), ctb (CatBoost) or xgb (XGBoost). All Models work with `MODE=cv` (Cross Validation), which is our preferred way that gave us the best results. For LightGBM there are also following options available: full (training on whole dataset w/o validation set), by_meter (training a model by meter type), by_building (training a model by building id).
   The models will be safed in the equally named directory.


Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_lgbm_model.py
    │   │   └── train_ctb_model.py
    |   │   └── train_xgb_model.py
    │   │   └── find_hyperparameter_lgbm.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.testrun.org

