import json
import dill  # noqa
import pickle
import warnings
from uuid import uuid4

from typing import Callable, Dict, List, Optional, Tuple
import hydra
import numpy as np
from pydantic import BaseModel, validator
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import make_scorer, get_scorer
import optuna
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from xgboost import XGBClassifier

from data_loading import SENIORITY_COL, STATE_COL, HOUR_COL, POSITION_COL
from utils import (
    InfrequentReplacer,
    get_best_optuna_params_from_study,
    load_data,
    LabeledData,
    PipelineFactory,
    cyclic_error,
)


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
    fill_dict: Optional[Dict]
    seed: int
    uid: str = str(uuid4())
    test_data_paths: Optional[List[str]] = None

    @validator("test_size")
    def test_size_valid(cls, test_size):
        assert 0 <= test_size <= 1
        return test_size


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


def save_result(best_model, test_score):
    with open("./model.pickle", "wb") as f:
        dill.dump(best_model, f)
    with open("./metadata.json", "w") as f:
        json.dump(
            {
                "test_score": test_score
            },
            fp=f,
        )


def create_score_func(study_config: StudyConfig) -> Callable[..., float]:
    if study_config.scoring is None:
        return make_scorer(
            cyclic_error, greater_is_better=False, period=study_config.cycle_period
        )
    else:
        return get_scorer(study_config.scoring)


def create_objective_function(
        x_train,
        y_train,
        score_func,
        pipeline_factory: PipelineFactory,
        study_config: StudyConfig,
        fixed_params,
        optuna_params: Dict[str, Callable[[optuna.Trial], Dict]],
):
    def objective(trial: optuna.Trial):
        pipeline = pipeline_factory.create_from_optuna(
            optuna_params=optuna_params, trial=trial, fixed_params=fixed_params
        )
        cv_score = cross_val_score(
            estimator=pipeline,
            X=x_train,
            y=y_train,
            scoring=score_func,
            cv=study_config.cv_num_folds,
            error_score="raise",
        )
        score = cv_score.mean()
        return score

    return objective


@hydra.main(config_name="config", config_path="config")
def main(conf: StudyConfig) -> None:
    np.random.seed(conf.seed)
    data = load_data(*conf.data_paths)
    if conf.debug_mode:
        sample_indices = np.random.choice(
            len(data.data), size=conf.debug_mode_sample_size
        )
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
        ohe_cols=[STATE_COL, SENIORITY_COL, POSITION_COL],
        ordinal_cols=[HOUR_COL],
    )

    size_categories = sorted(
        [
            "1,001-5,000",
            "201-500",
            "51-200",
            "501-1,000",
            "10,000+",
            "5,001-10,000",
            "1-10",
        ],
        key=lambda s: int(s.replace("+", "").replace(",", "").split("-")[0]),
    )
    hour_categories = [str(i) for i in range(24)]
    fixed_params = {
        "numeric_imputer_cls": SimpleImputer,
        "numeric_imputer_params": {"strategy": "mean"},
        "ordinal_imputer_cls": SimpleImputer,
        "ordinal_imputer_params": {"strategy": "most_frequent"},
        "ohe_imputer_cls": SimpleImputer,
        "ohe_imputer_params": {"strategy": "constant", "fill_value": "Unknown"},
        "ordinal_encoder_cls": OrdinalEncoder,
        "ordinal_encoder_params": {
            "categories": [hour_categories],
            "handle_unknown": "use_encoded_value",
            "unknown_value": -1,
        },
        "ohe_filter_cls": InfrequentReplacer,
        "ohe_encoder_cls": OneHotEncoder,
        "ohe_encoder_params": {"handle_unknown": "ignore"},
        "scaler_cls": StandardScaler,
        "scaler_params": {"with_mean": False}
    }

    optuna_params = {
        "model_cls": lambda trial: trial.suggest_categorical(
            "model_cls",
            choices=[RandomForestClassifier, XGBClassifier]
        ),
        "model_params": lambda trial: {
            "n_estimators": trial.suggest_int(
                "model_params__n_estimators",
                low=conf.n_estimators_range[0],
                high=conf.n_estimators_range[1],
            ),
        },
        "ohe_filter_params": lambda trial: {
            "threshold": trial.suggest_int(
                "ohe_filter_params__threshold",
                low=conf.min_category_freq_range[0],
                high=conf.min_category_freq_range[1],
            ),
            "fill_dict": trial.suggest_categorical(
                "ohe_filter_params__fill_dict",
                choices=[conf.fill_dict]
            )
        }
    }

    score_func = create_score_func(conf)
    objective_func = create_objective_function(
        x_train=train_data.data,
        y_train=train_data.labels,
        score_func=score_func,
        pipeline_factory=pipeline_factory,
        study_config=conf,
        fixed_params=fixed_params,
        optuna_params=optuna_params,
    )

    study = optuna.create_study(direction=conf.objective_optim_direction)
    study.optimize(objective_func, n_trials=conf.num_trials)

    best_params = get_best_optuna_params_from_study(study)
    best_model = pipeline_factory.create_from_args(**best_params, **fixed_params)

    best_model.fit(X=x_train, y=y_train)
    test_score = score_func(estimator=best_model, X=x_test, y_true=y_test)

    save_result(best_model, test_score)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
