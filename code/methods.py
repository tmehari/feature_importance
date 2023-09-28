import numpy as np
import os
from os.path import join, isdir
import pandas as pd
import shap
from skrebate import ReliefF
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from sklearn.feature_selection import SelectKBest, mutual_info_classif, chi2
from sklearn.metrics import roc_auc_score, mutual_info_score
from lime.lime_tabular import LimeTabularExplainer
from tqdm import tqdm
import GPy
import xgboost
import warnings
import random 
import tensorflow as tf
# tf.config.run_functions_eagerly(True)
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, BatchNormalization


def save_results(output_dir, features, feature_importances, tag, patho, method, acc=None):
    feat_ids = np.argsort(feature_importances)[::-1]
    print(f"top 5 features: {list(features[feat_ids][:5])}")
    feat_order = pd.DataFrame(data=features[feat_ids], columns=["Feature"])
    feat_order.to_csv(join(output_dir, f"{tag}{patho}_{method}.csv"))
    if acc is not None:
        write_acc_txt(acc, join(output_dir, f"{tag}{patho}_{method}_acc.txt"))


def write_acc_txt(data, filepath):
    with open(filepath, "w") as f:
        f.write(str(data))


def order(arr):
    ord = np.argsort(-1 * arr)
    print(ord + 1)
    return ord


def gen_exp_map(model, test_x, test_y, train_x, train_y, feature_selection='lasso_path'):
    lime_explainer = LimeTabularExplainer(
        train_x,
        mode="classification",
        training_labels=train_y,
        feature_selection=feature_selection)
    exp_map_arr = np.zeros(test_x.shape)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i in tqdm(range(len(test_x))):
            exp = lime_explainer.explain_instance(
                test_x[i], model, num_features=117, num_samples=100
            )
            map = list(exp.as_map().values())[0]
            for j in range(test_x.shape[1]):
                try:
                    exp_map_arr[i, map[j][0]] = map[j][1]
                except IndexError:
                    pass
    return exp_map_arr


def lime_maker(model, x, y, train_x, train_y, modelname="lr"):
    if modelname in ["lr", "rf"]:
        exp_array = gen_exp_map(model.predict_proba, x, y, train_x, train_y)
    elif modelname in ["dn", "xgb"]:
        exp_array = gen_exp_map(model, x, y, train_x, train_y, feature_selection='auto' if modelname=='dn' else 'lasso_path')
    else:
        raise ValueError("model not known")
    return np.mean(np.abs(exp_array), 0)


def shap_maker(model, x, y, modelname, train_x=None):
    if modelname == "xgb" or modelname == "rf":
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(x)
        if len(shap_values) == 2:
            shap_values = shap_values[1]
    elif modelname == "dn":
        explainer = shap.Explainer(model, train_x, algorithm='permutation')
        shap_values = explainer(x).values
    elif modelname == "lr":
        explainer = shap.Explainer(model, train_x)
        shap_values = explainer.shap_values(x)
    else:
        raise ValueError("model unknown")

    shap_values = shap_values[np.where(y)[0]]
    return np.mean(np.abs(shap_values), 0)


def calculate_accuracy(y_true, y_pred):
    return 1 - np.mean(np.abs(y_pred - y_true))


def model_maker(modeltyp, train_x, train_y, test_x, test_y):
    print("Takes as input: random forest, logistic regression, deep net, xgb")
    modeltyp = modeltyp.strip().lower()

    if modeltyp == "random forest":
        rf_model = RandomForestClassifier()
        rf_model.fit(train_x, train_y.astype("int"))
        rf_pred = rf_model.predict(test_x)
        acc = calculate_accuracy(test_y, rf_pred)
        print("Random Forest Classification Accuracy:", acc)
        model = rf_model

    elif modeltyp == "logistic regression":
        lr_model = LogisticRegression(solver="liblinear")
        lr_model.fit(train_x, train_y)
        lr_pred = lr_model.predict(test_x)
        acc = calculate_accuracy(test_y, lr_pred)
        print("Logistic Regression Classification Accuracy:", acc)
        model = lr_model

    elif modeltyp == "xgb":
        train = xgboost.DMatrix(train_x, label=train_y)
        test = xgboost.DMatrix(test_x, label=test_y)

        params = {"eta": 0.01, "objective": "binary:logistic"}

        model = xgboost.train(
            params,
            train,
            5000,
            evals=[(test, "test")],
            verbose_eval=True,
            early_stopping_rounds=20,
        )

        acc = calculate_accuracy(test_y, np.round(model.predict(test)))
        print("XGB Classification Accuracy:", acc)
    
    else:
        raise ValueError("Model type not recognized: " + modeltyp)

    return model, acc


def get_model(modelname, X_train, X_test, y_train, y_test, method="lime"):
    if modelname == "lr":
        model, acc = model_maker(
            "logistic regression", X_train, y_train, X_test, y_test
        )
        return model, acc
    elif modelname == "xgb":
        model, acc = model_maker("xgb", X_train, y_train, X_test, y_test)
        if method == "shap":
            return model, acc
        elif method == "lime":

            def pred(d):
                d = xgboost.DMatrix(d)
                o = model.predict(d)
                o2 = 1 - o
                return np.stack([o, o2], axis=1)

            return pred, acc
        elif method == "permutation":

            class XGBModel:
                def __init__(self, model, X_test, y_test):
                    self.model = model
                    self.X_test = X_test
                    self.y_test = y_test

                def score(self, X_test, y_test):
                    test = xgboost.DMatrix(X_test, label=y_test)
                    return 1 - np.mean(
                        np.abs(np.round(self.model.predict(test)) - y_test)
                    )

                def fit(self, X_train, y_train):
                    train = xgboost.DMatrix(X_train, label=y_train)
                    test = xgboost.DMatrix(self.X_test, label=self.y_test)

                    params = {"eta": 0.01, "objective": "binary:logistic"}

                    self.model = xgboost.train(
                        params,
                        train,
                        5000,
                        evals=[(test, "test")],
                        verbose_eval=100,
                        early_stopping_rounds=20,
                    )

            model = XGBModel(model, X_test, y_test)
            acc = model.score(X_test, y_test)
            return model, acc
    elif modelname == "rf":
        model, acc = model_maker("random forest", X_train, y_train, X_test, y_test)
        return model, acc
    elif modelname == "dn":
        # force tensorflow to use cpu
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

        class DenseNet(Model):
            def __init__(self):
                super(DenseNet, self).__init__()
                dim = 256
                self.d1 = Dense(dim, activation="relu")
                self.d2 = Dense(dim, activation="relu")
                self.d3 = Dense(dim, activation="relu")
                self.d4 = Dense(dim, activation="relu")
                self.d5 = Dense(dim, activation="relu")
                self.d6 = Dense(dim, activation="relu")
                self.dfinal = Dense(1, activation="sigmoid")

            def call(self, x):
                layers = 2
                x = self.d1(x)
                if layers > 1:
                    x = self.d2(x)
                if layers > 2:
                    x = self.d3(x)
                if layers > 3:
                    x = self.d4(x)
                if layers > 4:
                    x = self.d5(x)
                if layers > 5:
                    x = self.d6(x)
                
                return self.dfinal(x)

            def score(self, X_test, y_test):
                y_probabilities = self.predict(
                    X_test
                ).flatten()  # get output of network
                y_preds = 1 * (y_probabilities > 0.5)  # get predictions
                test_acc = (y_preds == y_test).sum() / len(y_test)  # get test accuracy
                return test_acc

        optimizer = tf.keras.optimizers.Adam()

        bincro = tf.keras.losses.BinaryCrossentropy()

        train_loss = tf.keras.metrics.Mean(name="train_loss")
        train_accuracy = tf.keras.metrics.BinaryAccuracy(name="train_accuracy")

        @tf.function
        def train_step(model, x, y):
            y = tf.convert_to_tensor(
                tf.expand_dims(tf.cast(y, dtype=float), -1), dtype=float
            )
            with tf.GradientTape() as tape:
                predictions = model(x, training=True)
                loss = bincro(y, predictions)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            train_loss(loss)
            train_accuracy(y, predictions)

        def train_model(model, x, y):
            for _ in range(50):
                train_step(model, x, y)
            return model

        deep_model = DenseNet()
        deep_model = train_model(
            deep_model,
            np.array(X_train),
            np.array(y_train),
        )

        acc = deep_model.score(X_test, y_test)  # get test accuracy
        print("Deep Networks Classification Accuracy:", acc)

        if method == "shap" or method == "permutation":
            return deep_model, acc

        def predict_fn(x, mod=deep_model):
            r = mod(x).numpy()
            return np.concatenate([1 - r, r], 1)

        return predict_fn, acc

    else:
        raise ValueError(modelname + " model not known")


def run_shap(output_dir, X_train, X_test, y_train, y_test, modelname, features, patho, tag):
    print("run shap with", modelname, "model and", patho, "pathology")
    model, acc = get_model(modelname, X_train, X_test, y_train, y_test, method="shap")
    shap_feature_importances = shap_maker(
        model, X_test, y_test, modelname, train_x=X_train
    )
    shapvalues = pd.DataFrame(
        data=list(zip(features, shap_feature_importances)),
        columns=["Features", "Importances"],
    )
    shapvalues.to_csv(join(output_dir, tag + patho + "_svalues_" + modelname + ".csv"))
    save_results(
        output_dir, features, shap_feature_importances, tag, patho, f"shap_{modelname}", acc=acc
    )


def run_lime(output_dir, X_train, X_test, y_train, y_test, modelname, features, patho, tag):
    print("run lime with", modelname, "model and", patho, "pathology")
    model, acc = get_model(modelname, X_train, X_test, y_train, y_test, method="lime")

    lime_feature_importances = lime_maker(
        model, X_test, y_test, X_train, y_train, modelname=modelname
    )
    limevalues = pd.DataFrame(
        data=list(zip(features, lime_feature_importances)),
        columns=["Features", "Importances"],
    )
    limevalues.to_csv("./output/" + tag + patho + "_lvalues_" + modelname + ".csv")
    save_results(
        output_dir, features, lime_feature_importances, tag, patho, f"lime_{modelname}", acc=acc
    )


def run_permutation(output_dir, X_train, X_test, y_train, y_test, modelname, features, patho, tag):
    print("run permutation method with", modelname, "model and", patho, "pathology")
    model, acc = get_model(
        modelname, X_train, X_test, y_train, y_test, method="permutation"
    )
    result = permutation_importance(model, X_test, y_test, n_repeats=5, random_state=0)

    permutation_feature_importances = list(result["importances_mean"])
    save_results(
        output_dir, 
        features,
        permutation_feature_importances,
        tag,
        patho,
        f"permutation_{modelname}",
        acc=acc,
    )


def run_lr_implicit(output_dir, X_train, X_test, y_train, y_test, features, patho, tag):
    model, acc = model_maker("logistic regression", X_train, y_train, X_test, y_test)
    weights = np.abs(model.coef_.flatten())
    feature_importances = weights  # /  np.abs(X_test.mean(axis=0))
    save_results(output_dir, features, feature_importances, tag, patho, f"implicit_lr", acc=acc)


def run_rf_implicit(output_dir, X_train, X_test, y_train, y_test, features, patho, tag):
    model, acc = model_maker("random forest", X_train, y_train, X_test, y_test)
    save_results(
        output_dir, features, model.feature_importances_, tag, patho, f"implicit_rf", acc=acc
    )


def run_gp_implicit(output_dir, X_train, X_test, y_train, y_test, features, patho, tag):
    y_train[np.where(y_train == 0)[0]] = -1
    y_test[np.where(y_test == 0)[0]] = -1
    kernel = GPy.kern.RBF(
        input_dim=X_train.shape[1],
        ARD=True,
        lengthscale=4 * np.ones(X_train.shape[1]),
        variance=1,
    )

    m = GPy.models.GPRegression(
        X_train, y_train[:, None], kernel=kernel, normalizer=True
    )
    m.inference_method = GPy.inference.latent_function_inference.Laplace()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.optimize(max_iters=10)
    feature_importances = -1 * m.kern.lengthscale

    mean, _ = m.predict(X_test)

    y_pred = np.where(mean > 0.0, 1, -1)

    # Calculate accuracy
    acc = np.mean(y_pred.flatten() == y_test)

    save_results(output_dir, features, feature_importances, tag, patho, f"implicit_gp", acc=acc)


def run_nca(output_dir, X_train, X_test, y_train, y_test, features, patho, tag):
    X = np.concatenate([X_train, X_test])
    y = np.concatenate([y_train, y_test])
    nca = NeighborhoodComponentsAnalysis(
        n_components=X.shape[1], random_state=42, tol=1e-7, max_iter=100
    )
    nca.fit(X, y)

    # Compute the feature importances
    A = nca.components_
    # feature_importances = np.linalg.norm(A, axis=1)
    feature_importances = np.sum(np.abs(A), axis=1)

    save_results(output_dir, features, feature_importances, tag, patho, f"_nca", acc=None)


def run_mrmr(output_dir, X_train, X_test, y_train, y_test, features, patho, tag, n_bins=100):
    def select_next_feature(M, mi, selected_features):
        max_mrmr = -np.inf
        next_feature = -1
        for i in range(len(mi)):
            if i in selected_features:
                continue
            relevance = mi[i]
            redundancy = np.mean(M[i, selected_features])
            mrmr = relevance - redundancy
            if mrmr > max_mrmr:
                max_mrmr = mrmr
                next_feature = i
        return next_feature

    X = np.concatenate([X_train, X_test])
    y = np.concatenate([y_train, y_test])
    # Bin the continuous data into categories
    bins = [get_equidistant_bin(X[:, i], n_bins) for i in range(X.shape[1])]

    # Bin the data
    X_binned = np.zeros_like(X)
    for i in range(X.shape[1]):
        X_binned[:, i] = np.digitize(X[:, i], bins[i])

    X = X_binned
    n_features_to_select = X.shape[1]

    # Compute the mutual information between each feature and the target
    mi = mutual_info_classif(X, y)
    M = np.zeros((X.shape[1], X.shape[1]))
    for i in range(n_features_to_select):
        for j in range(n_features_to_select):
            M[i, j] = mutual_info_score(X[:, i], X[:, j])

    # Initialize the selected features and the ranking
    selected_features = []
    # ranking = np.zeros(n_features_to_select, dtype=int)

    first_feature = np.argmax(mi)

    selected_features.append(first_feature)
    for _ in range(X.shape[1] - 1):
        next_feature = select_next_feature(M, mi, selected_features)
        selected_features.append(next_feature)

    selected_features = np.array(selected_features)
    feature_importances = np.zeros(len(selected_features))
    feature_importances[selected_features] = 1.0 / np.arange(
        1, len(selected_features) + 1
    )
    save_results(output_dir, features, feature_importances, tag, patho, f"_mrmr", acc=None)


def run_modified_rocauc(output_dir, X_train, X_test, y_train, y_test, features, patho, tag):
    X_train = np.concatenate([X_train, X_test])
    y_train = np.concatenate([y_train, y_test])
    rocaucs = np.array(
        [roc_auc_score(y_train, X_train[:, i]) for i in range(X_train.shape[1])]
    )
    feature_importances = np.maximum(rocaucs, 1 - rocaucs)

    rocauc_df = pd.DataFrame(
        data=list(zip(features, rocaucs)), columns=["Feature", "Rocauc"]
    )
    modrocauc_df = pd.DataFrame(
        data=list(zip(features, feature_importances)),
        columns=["Feature", "Modifiedrocauc"],
    )
    rocauc_df.to_csv(join(output_dir, f"{tag}{patho}_rocaucs_values.csv"), index=False)
    modrocauc_df.to_csv(join(output_dir, f"{tag}{patho}_modrocaucs_values.csv"), index=False)

    save_results(
        output_dir, features, feature_importances, tag, patho, f"_modifiedrocauc", acc=None
    )


def get_equidistant_bin(arr, num_bins):
    sorted_arr = np.sort(arr)
    bin_values = sorted_arr[
        np.arange(0, sorted_arr.shape[0], int(sorted_arr.shape[0] / num_bins))
    ]
    return bin_values


def run_chi_squared(
    output_dir, X_train, X_test, y_train, y_test, features, patho, tag, n_bins=1000
):
    X_train = np.concatenate([X_train, X_test])
    y_train = np.concatenate([y_train, y_test])
    # Bin the continuous data into categories
    bins = [get_equidistant_bin(X_train[:, i], n_bins) for i in range(X_train.shape[1])]
    # Bin the data
    X_binned = np.zeros_like(X_train)
    for i in range(X_train.shape[1]):
        X_binned[:, i] = np.digitize(X_train[:, i], bins[i])

    selector = SelectKBest(score_func=chi2, k=X_binned.shape[1])
    X_new = selector.fit_transform(X_binned, y_train)

    # Get indices of selected features
    important_features = np.where(selector.get_support())[0]

    # Get scores of selected features
    feature_importances = selector.scores_[important_features]
    save_results(output_dir, features, feature_importances, tag, patho, f"_chisquared", acc=None)


def run_relieff(
    output_dir, X_train, X_test, y_train, y_test, features, patho, tag, n_neighbors=100
):
    X = np.concatenate([X_train, X_test])
    y = np.concatenate([y_train, y_test])
    fs = ReliefF(discrete_threshold=0, n_jobs=-1, n_features_to_select=X.shape[1])
    fs.fit(X, y)
    feature_importances = fs.feature_importances_
    save_results(output_dir, features, feature_importances, tag, patho, f"_relieff", acc=None)


def run_methods(
    output_dir, patho, method, models, X_train, X_test, y_train, y_test, features, tag="multiclass"
):
    # Define mapping between model names and functions
    function_mapping = {
        "shap": run_shap,
        "lime": run_lime,
        "implicit": {
            "lr": run_lr_implicit,
            "rf": run_rf_implicit,
            "gp": run_gp_implicit,
        },
        "permutation": run_permutation,
    }

    filter_methods_function_mapping = {
        "nca": run_nca,
        "chisquared": run_chi_squared,
        "modifiedrocauc": run_modified_rocauc,
        "relieff": run_relieff,
        "mrmr": run_mrmr,
    }
    if not isdir(output_dir):
        os.makedirs(output_dir)

    # Loop through filter methods
    for fmethod, func in filter_methods_function_mapping.items():
        if method is None or method == fmethod:
            print(f"FILTER METHODS: run {fmethod} method")
            func(output_dir, X_train, X_test, y_train, y_test, features, patho, tag)

    # Loop through each model
    for model in models:
        # Check if model is known
        if model not in ["rf", "xgb", "lr", "dn", "gp"]:
            raise ValueError(model + " not known")

        # Loop through each method
        for method_key, func in function_mapping.items():
            # Check if current method should be executed
            if method is None or method == method_key:
                # Special handling for implicit method
                if method_key == "implicit":
                    if model in func.keys():
                        print(f"IMPLICIT METHODS: run {model}")
                        func[model](
                            output_dir, X_train, X_test, y_train, y_test, features, patho, tag
                        )
                else:
                    # we do not evaluate shap nor lime for gps
                    if model == "gp":
                        continue
                    print(
                        f"MODEL-DEPENDENT METHODS: run {method_key} method on {model} model"
                    )
                    func(output_dir, X_train, X_test, y_train, y_test, model, features, patho, tag)
