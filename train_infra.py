import json
import pickle
from multiprocessing import Pipe
from uuid import uuid4

from hydra.core.config_store import ConfigStore
from venv import create

# from pydantic.dataclasses import dataclass
from dataclasses import dataclass
from logging.config import dictConfig
from typing import Callable, Dict, List, Optional, Tuple
import hydra
import numpy as np
from omegaconf import DictConfig
from pydantic import BaseModel, validator, dataclasses as pydantic_dataclasses
from sklearn import pipeline
import sklearn
from sklearn import metrics
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import make_scorer, SCORERS, get_scorer
import optuna
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from xgboost import XGBClassifier

from types import MappingProxyType

from data_loading import create_positive_only_train_data, FUNCTION_COL, SENIORITY_COL, STATE_COL, COUNTRY_COL, SIZE_COL, \
    INDUSTRY_COL, create_full_train_data, HOUR_COL

EmptyMap = MappingProxyType({})
CATEGORY_REPLACER = lambda col, category: "Other"


class StudyConfig(BaseModel):
    data_paths: List[str]
    debug_mode: bool
    debug_mode_sample_size: int
    cv_num_folds: int
    scoring: str
    num_trials: int
    objective_optim_direction: str
    test_size: float
    n_estimators_range: Tuple[int, int]
    max_depth_range: Tuple[int, int]
    min_category_freq_range: Tuple[int, int]
    shuffle_data: bool
    stratify_split: bool
    cycle_period: int
    uid: str = str(uuid4())
    test_data_paths: Optional[List[str]] = None

    @validator("test_size")
    def test_size_valid(cls, test_size):
        assert 0 <= test_size <= 1
        return test_size


@dataclass
class LabeledData:
    data: pd.DataFrame
    labels: pd.Series


def split_data(data: LabeledData, conf: StudyConfig) -> Tuple[LabeledData, LabeledData]:
    X_train, X_test, y_train, y_test = train_test_split(
        data.data,
        data.labels,
        test_size=conf.test_size,
        shuffle=conf.shuffle_data,
        stratify=data.labels if conf.stratify_split else None,
    )
    train_data = LabeledData(X_train, y_train)
    test_data = LabeledData(X_test, y_test)
    return train_data, test_data


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
                                Pipeline(steps=[("imputer", numeric_imputer_cls(**numeric_imputer_params))]),
                                self.numeric_cols,
                            ),
                            (
                                "ohe_transformer",
                                Pipeline(steps=[("filter", ohe_filter_cls(**ohe_filter_params)),
                                                ("imputer", ohe_imputer_cls(**ohe_imputer_params)),
                                                ("encoder", ohe_encoder_cls(**ohe_encoder_params))]),
                                self.ohe_cols,
                            ),
                            (
                                "ordinal_transformer",
                                Pipeline(steps=[("imputer", ordinal_imputer_cls(**ordinal_imputer_params)),
                                                ("encoder", ordinal_encoder_cls(**ordinal_encoder_params))]),
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
            postprocess = Pipeline(steps=[("postprocess", postprocessor_cls(**postprocessor_params))])
            pipe_steps.append(("postprocess", postprocess))

        res = Pipeline(steps=pipe_steps)
        return res

    def create_from_optuna(self, optuna_params: Dict, trial: optuna.Trial, fixed_params: Dict):
        res = self.create_from_args(
            **{k: c(trial) for k, c in optuna_params.items()},
            **fixed_params
        )
        return res


def load_data(engagements_csv_path, companies_csv_path, contacts_csv_path) -> LabeledData:
    engagements_df, companies_df, contacts_df = pd.read_csv(engagements_csv_path), \
                                                pd.read_csv(companies_csv_path), \
                                                pd.read_csv(contacts_csv_path)
    x, y = create_full_train_data(engagements_df=engagements_df,
                                  companies_df=companies_df,
                                  contacts_df=contacts_df)
    labeled_data = LabeledData(x, y)
    return labeled_data


def save_result(best_params, best_model, y_test_preds, test_score, conf: StudyConfig):
    with open("./model.pickle", 'wb') as f:
        pickle.dump(best_model, f)
    with open("./metadata.json") as f:
        json.dump({
            "best_params": best_params,
            "y_test_preds": y_test_preds,
            "test_score": test_score,
            "conf": conf.dict()
        }, fp=f)


def cyclic_error(y, y_pred, period):
    final_errors = np.min(np.column_stack([((y - y_pred) % (period - 1)).values,
                                           ((y_pred - y) % (period - 1)).values]),
                          axis=1)
    return np.mean(final_errors)


def create_score_func(study_config: StudyConfig) -> Callable[..., float]:
    if study_config.scoring is None:
        return make_scorer(cyclic_error, greater_is_better=False, period=study_config.cycle_period)
    else:
        return get_scorer(study_config.scoring)


def create_objective_function(
        x_train, y_train, score_func, pipeline_factory: PipelineFactory, study_config: StudyConfig, fixed_params,
        optuna_params: Dict[str, Callable[[optuna.Trial], Dict]]
):
    def objective(trial: optuna.Trial):
        pipeline = pipeline_factory.create_from_optuna(optuna_params=optuna_params, trial=trial,
                                                       fixed_params=fixed_params)
        cv_score = cross_val_score(
            estimator=pipeline,
            X=x_train,
            y=y_train,
            scoring=score_func,
            cv=study_config.cv_num_folds,
            error_score='raise'
        )
        score = cv_score.mean()
        return score

    return objective


def get_best_optuna_params_from_study(study: optuna.Study):
    best_params = study.best_params
    res = {}
    for param_name, param in best_params.items():
        hierarchy, key = param_name.rsplit('__', maxsplit=1)
        curr_level = res
        for comp in hierarchy.split('__'):
            curr_level[comp] = {}
            curr_level = curr_level[comp]
        curr_level[key] = param
    return res


class InfrequentReplacer(TransformerMixin, BaseEstimator):
    def __init__(self, threshold=10, fill_func=CATEGORY_REPLACER):
        self.threshold = threshold
        self.drop_categories = []
        self.fill_func = fill_func

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
                res[c].replace(inplace=True, to_replace=self.drop_categories[idx], value=self.fill_func(c, category))
        return res


@hydra.main(config_name="config", config_path="config")
def main(conf: StudyConfig) -> None:
    data = load_data(*conf.data_paths)
    if conf.debug_mode:
        sample_indices = np.random.choice(len(data.data), size=conf.debug_mode_sample_size)
        data.data = data.data.iloc[sample_indices]
        data.labels = data.labels.iloc[sample_indices]

    if conf.test_data_paths is not None:
        test_data = load_data(*conf.test_data_paths)
        train_data = data
    else:
        train_data, test_data = split_data(data, conf)

    x_train, y_train = train_data.data, train_data.labels
    x_test, y_test = test_data.data, test_data.labels

    pipeline_factory = PipelineFactory(
        numeric_cols=x_train.select_dtypes("number").columns,
        ohe_cols=[STATE_COL, SENIORITY_COL],
        ordinal_cols=[SIZE_COL, HOUR_COL]
    )

    size_categories = sorted(['1,001-5,000',
                              '201-500',
                              '51-200',
                              '501-1,000',
                              '10,000+',
                              '5,001-10,000',
                              '1-10'], key=lambda s: int(s.replace('+', '').replace(',', '').split('-')[0]))
    hour_categories = [str(i) for i in range(24)]

    fixed_params = {
        "numeric_imputer_cls": SimpleImputer,
        "numeric_imputer_params": {"strategy": "mean"},
        "ordinal_imputer_cls": SimpleImputer,
        "ordinal_imputer_params": {"strategy": "most_frequent"},
        "ohe_imputer_cls": SimpleImputer,
        "ohe_imputer_params": {"strategy": "constant", "fill_value": "Unknown"},
        "ordinal_encoder_cls": OrdinalEncoder,
        "ordinal_encoder_params": {"categories": [size_categories, hour_categories],
                                   "handle_unknown": "use_encoded_value",
                                   "unknown_value": -1},
        "ohe_filter_cls": InfrequentReplacer,
        "ohe_encoder_cls": OneHotEncoder,
        "ohe_encoder_params": {"handle_unknown": "ignore"},
        "scaler_cls": StandardScaler,
        "scaler_params": {"with_mean": False},
        "model_cls": XGBClassifier,
    }
    optuna_params = {
        "model_params": lambda trial: {"n_estimators": trial.suggest_int(
            "model_params__n_estimators",
            low=conf.n_estimators_range[0],
            high=conf.n_estimators_range[1],
        ),
            "max_depth": trial.suggest_int(
                "model_params__max_depth",
                low=conf.max_depth_range[0],
                high=conf.max_depth_range[1],
            )},
        "ohe_filter_params": lambda trial: {"threshold": trial.suggest_int(
            "ohe_filter_params__threshold",
            low=conf.min_category_freq_range[0],
            high=conf.min_category_freq_range[1]
        )},

    }
    score_func = create_score_func(conf)
    objective_func = create_objective_function(train_data.data, train_data.labels, score_func, pipeline_factory, conf,
                                               fixed_params, optuna_params=optuna_params)

    study = optuna.create_study(
        direction=conf.objective_optim_direction
    )
    study.optimize(objective_func, n_trials=conf.num_trials)

    best_params = get_best_optuna_params_from_study(study)
    best_model = pipeline_factory.create_from_args(**best_params, **fixed_params)

    best_model.fit(X=x_train, y=y_train)
    test_score = score_func(estimator=best_model, X=x_test, y_true=y_test)

    y_test_preds = best_model.predict(X=x_test)
    save_result(best_params, best_model, y_test_preds, test_score, conf)


if __name__ == "__main__":
    main()
