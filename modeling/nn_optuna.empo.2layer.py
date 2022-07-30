#!/usr/bin/env python3

import os
import sys
import sqlite3
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD
import tensorflow as tf
from tensorflow import keras
from keras import metrics
import optuna


DATABASE_NAME = "metagenome_classification.db"


def get_training_observations():
    print(f"Getting all training observations from '{DATABASE_NAME}'...")
    conn = sqlite3.connect("..//data//" + DATABASE_NAME)

    x_train_transposed = pd.read_sql_query("SELECT * FROM x_train", con=conn)

    conn.commit()
    conn.close()

    x_train_transposed.set_index("index", inplace=True)
    x_train = x_train_transposed.T
    x_train_normalized = get_protein_proportions(x_train)

    return x_train_normalized


def get_training_labels():
    print(f"Getting all training labels from '{DATABASE_NAME}'...")
    conn = sqlite3.connect("..//data//" + DATABASE_NAME)

    y_train_transposed = pd.read_sql_query("SELECT * FROM y_train", con=conn)

    conn.commit()
    conn.close()

    y_train_transposed.set_index("index", inplace=True)
    y_train = y_train_transposed.T

    return y_train


def get_test_observations():
    print(f"Getting all test observations from '{DATABASE_NAME}'...")
    conn = sqlite3.connect("..//data//" + DATABASE_NAME)

    x_test_transposed = pd.read_sql_query("SELECT * FROM x_test", con=conn)

    conn.commit()
    conn.close()

    x_test_transposed.set_index("index", inplace=True)
    x_test_normalized = get_protein_proportions(x_test)

    return x_test_normalized


def get_test_labels():
    print(f"Getting all test labels from '{DATABASE_NAME}'...")
    conn = sqlite3.connect("..//data//" + DATABASE_NAME)

    y_test_transposed = pd.read_sql_query("SELECT * FROM y_test", con=conn)

    conn.commit()
    conn.close()

    y_test_transposed.set_index("index", inplace=True)
    y_test = y_test_transposed.T

    return y_test


def get_protein_proportions(df):
    # Each column has counts of "hits" now, but not consistent across observations.
    # Get proportions each protein appears.
    df = df.div(df.sum(axis=1), axis=0)
    return df


def build_model(
    n_classes,
    hidden_layer_sizes=[],
    activation="relu",
    final_layer_activation="softmax",
    dropout=0.0,
    optimizer="Adam",
    learning_rate=0.01,
    kernel_regularizer=1e-5,
    bias_regularizer=1e-5,
    activity_regularizer=1e-5
):
    """Build a multi-class logistic regression model using Keras.

    Args:
      n_classes: Number of output classes in the dataset.
      hidden_layer_sizes: A list with the number of units in each hidden layer.
      activation: The activation function to use for the hidden layers.
      optimizer: The optimizer to use (SGD, Adam).
      learning_rate: The desired learning rate for the optimizer.

    Returns:
      model: A tf.keras model (graph).
    """
    tf.keras.backend.clear_session()
    np.random.seed(0)
    tf.random.set_seed(0)
    model = keras.Sequential()
    model.add(keras.layers.Flatten())

    for hidden_layer_size in hidden_layer_sizes:
        if hidden_layer_size > n_classes:
            model.add(keras.layers.Dense(
                hidden_layer_size,
                activation=activation,
                kernel_regularizer=keras.regularizers.L2(kernel_regularizer),
                bias_regularizer=keras.regularizers.L2(bias_regularizer),
                activity_regularizer=keras.regularizers.L2(activity_regularizer)
            ))
            if dropout > 0:
                model.add(keras.layers.Dropout(dropout))

    model.add(keras.layers.Dense(
        n_classes,
        activation=final_layer_activation,
        kernel_regularizer=keras.regularizers.L2(kernel_regularizer),
        bias_regularizer=keras.regularizers.L2(bias_regularizer),
        activity_regularizer=keras.regularizers.L2(activity_regularizer)
    ))
    opt = None
    if optimizer == "SGD":
        opt = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    elif optimizer == "Adam":
        opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    else:
        raise f"Unsupported optimizer, {optimizer}"
    model.compile(
        loss="sparse_categorical_crossentropy", optimizer=opt, metrics=["accuracy"]
    )
    return model


def train_model(
    X_train,
    Y_train,
    num_classes,
    hidden_layer_sizes=[],
    activation="tanh",
    final_layer_activation="softmax",
    kernel_regularizer=1e-5,
    bias_regularizer=1e-5,
    activity_regularizer=1e-5,
    dropout=0.2,
    optimizer="Adam",
    learning_rate=0.01,
    batch_size=64,
    num_epochs=20,
):

    # Build the model.
    model = build_model(
        num_classes,
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        final_layer_activation=final_layer_activation,
        dropout=dropout,
        optimizer=optimizer,
        learning_rate=learning_rate,
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer,
        activity_regularizer=activity_regularizer
    )

    # Train the model.
    print("Training...")
    history = model.fit(
        x=X_train,
        y=Y_train,
        epochs=num_epochs,
        batch_size=batch_size,
        validation_split=0.1,
        verbose=1,
    )

    model.summary()
    return model


def train_and_evaluate(
    learning_rate: float = 0.01,
    hidden_layer_size1: int = 256,
    hidden_layer_size2: int = 256,
    kernel_regularizer: float = 1e-5,
    bias_regularizer: float = 1e-5,
    activity_regularizer: float = 1e-5,
    dropout: float = 0.0):

    # Load data
    x_train_raw_counts = get_training_observations()
    x_train = get_protein_proportions(x_train_raw_counts)
    print(f"There are {x_train.shape[1]} features")
    y_train = get_training_labels()

    # convert string labels to numeric
    labels3 = [
        "Aerosol (non-saline)",
        "Animal corpus",
        "Animal proximal gut",
        "Hypersaline (saline)",
        "Plant corpus",
        "Plant rhizosphere",
        "Plant surface",
        "Sediment (non-saline)",
        "Sediment (saline)",
        "Soil (non-saline)",
        "Subsurface (non-saline)",
        "Surface (non-saline)",
        "Surface (saline)",
        "Water (non-saline)",
        "Water (saline)",
    ]
    labels3_map = {}
    n_classes = len(labels3)
    for i in range(0, len(labels3)):
        label = labels3[i]
        labels3_map[label] = i
    y_train["EMPO_3_int"] = y_train["EMPO_3"].map(labels3_map)

    # Split into train/validation if not CV
    X_tr, X_val, Y_tr, Y_val = train_test_split(
        x_train, y_train, test_size=0.2
    )  # , random_state=1)

    nn3 = train_model(
        X_tr,
        Y_tr["EMPO_3_int"],
        n_classes,
        hidden_layer_sizes=[hidden_layer_size1, hidden_layer_size2],
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer,
        activity_regularizer=activity_regularizer,
        dropout=dropout,
        optimizer="Adam",
        learning_rate=learning_rate,
        batch_size=128,
        num_epochs=20,
    )

    evaluation = nn3.evaluate(x=X_val, y=Y_val["EMPO_3_int"], verbose=0, return_dict=True)
    accuracy = evaluation["accuracy"]
    loss = evaluation["loss"]

    print(f"Accuracy={accuracy}; Loss={loss}")
    return accuracy


def objective(trial):
    learning_rate = trial.suggest_float('learning_rate', 0.0001, 0.01)
    hidden_layer_size1 = trial.suggest_int('hidden_layer_size1', 0, 1024)
    hidden_layer_size2 = trial.suggest_int('hidden_layer_size2', 0, 1024)
    kernel_regularizer = trial.suggest_float('kernel_regularizer', 1e-10, 1e-4)
    bias_regularizer = trial.suggest_float('bias_regularizer', 1e-10, 1e-4)
    activity_regularizer = trial.suggest_float('activity_regularizer', 1e-10, 1e-4)
    dropout = trial.suggest_float('dropout', 0, 0.1)
    return train_and_evaluate(learning_rate, hidden_layer_size1, hidden_layer_size2, kernel_regularizer, bias_regularizer, activity_regularizer, dropout)


# hyperparameter optimization
study = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(),
    pruner=optuna.pruners.MedianPruner(n_warmup_steps=20)
)
study.optimize(objective, n_trials=50)

print(study.best_params)
with open("./nn_optuna.empo.2layer.txt", 'w') as fh:
    fh.write(str(study.best_params))
