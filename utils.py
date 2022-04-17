from collections import defaultdict
from dataclasses import dataclass
from types import MappingProxyType
from typing import Dict

import numpy as np
import optuna
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from data_loading import create_train_data, HOUR_COL

EmptyMap = MappingProxyType({})
default_fill_dict = defaultdict(lambda: defaultdict(lambda: "Other"))


class InfrequentReplacer(TransformerMixin, BaseEstimator):
    def __init__(self, threshold, fill_dict=None):
        if fill_dict is None:
            fill_dict = default_fill_dict
        self.threshold = threshold
        self.drop_categories = []
        self.fill_dict = fill_dict

    def fit(self, X: pd.DataFrame, y):
        for c in X.columns:
            counts = X[c].value_counts()
            values_to_drop = [k for k, v in counts.items() if v < self.threshold]
            self.drop_categories.append(values_to_drop)
        return self

    def transform(self, X: pd.DataFrame, y=None):
        res = X.copy()
        for idx, c in enumerate(X.columns):
            for category in self.drop_categories[idx]:
                res[c].replace(
                    inplace=True,
                    to_replace=self.drop_categories[idx],
                    value=self.fill_dict[c][category],
                )
        return res


def get_best_optuna_params_from_study(study: optuna.Study):
    best_params = study.best_params
    res = {}
    for param_name, param in best_params.items():
        try:
            hierarchy, key = param_name.rsplit("__", maxsplit=1)
            curr_level = res
            for comp in hierarchy.split("__"):
                curr_level[comp] = curr_level[comp] if comp in curr_level else {}
                curr_level = curr_level[comp]
            curr_level[key] = param
        except ValueError:
            res[param_name] = param
    return res


@dataclass
class LabeledData:
    data: pd.DataFrame
    labels: pd.Series


def load_data(
        engagements_csv_path, companies_csv_path, contacts_csv_path
) -> LabeledData:
    engagements_df, companies_df, contacts_df = (
        pd.read_csv(engagements_csv_path),
        pd.read_csv(companies_csv_path),
        pd.read_csv(contacts_csv_path),
    )
    x, y = create_train_data(
        engagements_df=engagements_df,
        companies_df=companies_df,
        contacts_df=contacts_df,
    )
    labeled_data = LabeledData(x, y)
    return labeled_data


class PipelineFactory:
    def __init__(self, numeric_cols, ohe_cols, ordinal_cols):
        self.numeric_cols = numeric_cols
        self.ordinal_cols = ordinal_cols
        self.ohe_cols = ohe_cols

    def create_from_args(
            self,
            numeric_imputer_cls,
            ohe_imputer_cls,
            ordinal_imputer_cls,
            ohe_encoder_cls,
            ordinal_encoder_cls,
            scaler_cls,
            model_cls,
            ohe_filter_cls,
            postprocessor_cls=None,
            numeric_imputer_params=EmptyMap,
            ordinal_imputer_params=EmptyMap,
            ordinal_encoder_params=EmptyMap,
            ohe_imputer_params=EmptyMap,
            ohe_encoder_params=EmptyMap,
            scaler_params=EmptyMap,
            model_params=EmptyMap,
            ohe_filter_params=EmptyMap,
            postprocessor_params=EmptyMap,
    ) -> Pipeline:
        pipe_steps = []

        preprocess = Pipeline(
            steps=[
                (
                    "col_transformer",
                    ColumnTransformer(
                        [
                            (
                                "numeric_transformer",
                                Pipeline(
                                    steps=[
                                        (
                                            "imputer",
                                            numeric_imputer_cls(
                                                **numeric_imputer_params
                                            ),
                                        )
                                    ]
                                ),
                                self.numeric_cols,
                            ),
                            (
                                "ohe_transformer",
                                Pipeline(
                                    steps=[
                                        ("filter", ohe_filter_cls(**ohe_filter_params)),
                                        (
                                            "imputer",
                                            ohe_imputer_cls(**ohe_imputer_params),
                                        ),
                                        (
                                            "encoder",
                                            ohe_encoder_cls(**ohe_encoder_params),
                                        ),
                                    ]
                                ),
                                self.ohe_cols,
                            ),
                            (
                                "ordinal_transformer",
                                Pipeline(
                                    steps=[
                                        (
                                            "imputer",
                                            ordinal_imputer_cls(
                                                **ordinal_imputer_params
                                            ),
                                        ),
                                        (
                                            "encoder",
                                            ordinal_encoder_cls(
                                                **ordinal_encoder_params
                                            ),
                                        ),
                                    ]
                                ),
                                self.ordinal_cols,
                            ),
                        ]
                    ),
                ),
                ("scaler", scaler_cls(**scaler_params)),
            ]
        )
        pipe_steps.append(("preprocess", preprocess))

        model = Pipeline(steps=[("model", model_cls(**model_params))])
        pipe_steps.append(("model", model))

        if postprocessor_cls is not None:
            postprocess = Pipeline(
                steps=[("postprocess", postprocessor_cls(**postprocessor_params))]
            )
            pipe_steps.append(("postprocess", postprocess))

        res = Pipeline(steps=pipe_steps)
        return res

    def create_from_optuna(
            self, optuna_params: Dict, trial: optuna.Trial, fixed_params: Dict
    ):
        params = {k: c(trial) for k, c in optuna_params.items()}
        for k in fixed_params:
            if k in params:
                params[k] = {**params[k], **fixed_params[k]}
            else:
                params[k] = fixed_params[k]
        res = self.create_from_args(**params)
        return res


def cyclic_error(y, y_pred, period):
    final_errors = np.min(
        np.column_stack(
            [((y - y_pred) % (period - 1)).values, ((y_pred - y) % (period - 1)).values]
        ),
        axis=1,
    )
    return np.mean(final_errors)


def predict_hours(clf_pipeline, X):
    X_all_hours = pd.concat(24 * [X])
    X_all_hours[HOUR_COL] = [str(i) for i in range(24) for _ in range(len(X))]
    y_pred_all_hours = clf_pipeline.predict_proba(X_all_hours)
    y_pred = []
    for i in range(len(X)):
        curr_probs = y_pred_all_hours.iloc[i::24, :]
        pred_label = str(np.argmax(curr_probs.iloc[:, 1]))
        y_pred.append(pred_label)
    return y_pred