from UberFarePredictionLib import config
from UberFarePredictionLib import data
from UberFarePredictionLib import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor


def train():
    # Get Data
    train_df = preprocessing.preprocess(config.TRAIN_FILE)

    # Generate X and y inputs
    try:
        X = train_df.drop([config.TARGET], axis=1)
        y = train_df[config.TARGET]
    except:
        train_df = train_df[0]
        X = train_df.drop([config.TARGET], axis=1)  # For some reason output train data is coming as tuple of length 2
        y = train_df[config.TARGET]

    # Train model
    rf = RandomForestRegressor(n_estimators=100, random_state=101)
    rf.fit(X, y)

    # Save model
    data.save_pipeline(rf)

    return rf


if __name__ == '__main__':
    model = train()
    print(type(model))
