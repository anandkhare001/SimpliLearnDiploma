import tensorflow as tf
import pandas as pd
import numpy as np
from keras import optimizers, regularizers
from keras.callbacks import LearningRateScheduler, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.layers import Dense, Dropout, BatchNormalization, PReLU
from keras.models import load_model
from keras.models import Sequential


def step_decay_schedule(initial_lr=1e-3, decay_factor=0.75, step_size=10, verbose=0):
    """ Wrapper function to create a LearningRateScheduler with step decay schedule. """

    def schedule(epoch):
        return initial_lr * (decay_factor ** np.floor(epoch / step_size))

    return LearningRateScheduler(schedule, verbose)


def clean_dataset(df):
    # Remove the entries tending towards infinite value or out of float range
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(axis=1)
    return df[indices_to_keep].astype(np.float64)


def fillInf(df, val):
    numCols = df.select_dtypes(include='number').columns
    cols = numCols[numCols != 'winPlacePerc']
    df[df == np.Inf] = np.NaN
    df[df == np.NINF] = np.NaN
    for c in cols: df[c].fillna(val, inplace=True)


def model_train(X, y, nodes, activation, batchSize, epoch):
    # Sequential of Layers -
    # Sequential class to build sequence of layers
    ann = tf.keras.models.Sequential()

    # Use Dense class to add layers
    # lets try with three layers
    # Try with 10 units/nodes
    # Use relu as activation function

    # Input Layer
    ann.add(tf.keras.layers.Dense(units=nodes, activation=activation))

    # First hidden Layer
    ann.add(tf.keras.layers.Dense(units=nodes, activation=activation))

    # Second hidden layer
    ann.add(tf.keras.layers.Dense(units=nodes, activation=activation))

    # Ad dOutput layer
    ann.add(tf.keras.layers.Dense(units=1, activation=activation))

    # Compile ANN
    ann.compile(optimizer='adam', loss='mse', metrics=['mae'])

    # Train model
    ann.fit(X, y, batch_size=batchSize, epochs=epoch)

    return ann


def createModel(dim):
    dnn = Sequential()
    dnn.add(Dense(512, kernel_initializer='he_normal', input_dim=dim, activation='relu'))
    dnn.add(BatchNormalization())
    dnn.add(Dropout(0.2))

    dnn.add(Dense(256, kernel_initializer='he_normal'))
    dnn.add(PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=None))
    dnn.add(BatchNormalization())
    dnn.add(Dropout(0.2))

    dnn.add(Dense(128, kernel_initializer='he_normal'))
    dnn.add(PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=None))
    dnn.add(BatchNormalization())
    dnn.add(Dropout(0.1))

    dnn.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))

    optimizer = optimizers.Adam(learning_rate=0.005)
    dnn.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

    return dnn


#X = np.array([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]])
#model = createModel(X)
#print(type(model))
