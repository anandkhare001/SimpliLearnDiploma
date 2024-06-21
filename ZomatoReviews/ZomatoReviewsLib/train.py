from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
from data import load_data, save_pipeline
import NLP
import config


def preprocessing():
    reviews = load_data(config.TRAIN_FILE)
    X = NLP.remove_null_entries(reviews)
    y = X[config.TARGET].values
    X = NLP.text_to_list(X)
    X = NLP.normalise_text(X)
    X = NLP.tokenize_text(X)
    X = NLP.del_stop(X)
    X = NLP.clean_text(X)

    return X, y


def train(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Vectorise the reviews
    vectorizer = TfidfVectorizer(max_features=10)
    X_train_bow = vectorizer.fit_transform(X_train)
    X_test_bow = vectorizer.fit_transform(X_test)

    # Model Training
    model = RandomForestRegressor()

    param_grid = {'max_depth': config.MAX_DEPTH}

    grid_search = GridSearchCV(estimator=model,
                               param_grid=param_grid,
                               cv=5,
                               n_jobs=-1,
                               verbose=1,
                               scoring='neg_mean_squared_error')
    grid_search.fit(X_train_bow, y_train)

    save_pipeline(grid_search)


if __name__ == '__main__':
    xx, yy = preprocessing()
    train(xx, yy)
