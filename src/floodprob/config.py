from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator


class AuthConfig(BaseModel):
    type: Literal["none", "bearer", "oauth"] = "none"
    token_env: str = ""


class CollectionsConfig(BaseModel):
    s1: str
    s2: str


class STACConfig(BaseModel):
    api_url: str
    collections: CollectionsConfig
    auth: AuthConfig = Field(default_factory=AuthConfig)
    request_timeout_s: int = 60
    max_items: int = 400


class TimeConfig(BaseModel):
    baseline: tuple[date, date]
    targets: list[tuple[date, date]]
    historical_windows: list[tuple[date, date]] = Field(default_factory=list)

    @field_validator("targets")
    @classmethod
    def validate_targets(cls, value: list[tuple[date, date]]) -> list[tuple[date, date]]:
        if not value:
            raise ValueError("time.targets must include at least one window")
        return value


class FiltersConfig(BaseModel):
    s2_max_cloud: float = 30.0
    s1_orbit_state: Literal["ascending", "descending", "both"] = "both"
    s1_polarizations: list[str] = Field(default_factory=lambda: ["VV", "VH"])
    # Restringe baseline e históricos S1 a las órbitas relativas del evento, para que
    # delta_vv/delta_vh midan cambio real y no diferencias de geometría de adquisición.
    s1_match_target_orbit: bool = True


class DemConfig(BaseModel):
    path: str = ""


class ExternalLabelsConfig(BaseModel):
    water: str = ""
    flood: str = ""


class OutputsConfig(BaseModel):
    data_dir: str = "data"
    cache_dir: str = "cache"
    out_dir: str = "outputs"


class ThresholdChangeConfig(BaseModel):
    vv_db: float = -1.5
    vh_db: float = -1.2


class LogRegConfig(BaseModel):
    c: float = 1.0
    max_iter: int = 400


class RandomForestConfig(BaseModel):
    n_estimators: int = 200
    max_depth: int | None = 10       # regularization: evita overfitting a pseudo-etiquetas
    min_samples_leaf: int = 20       # regularization: mínimo de muestras por hoja


class MlpConfig(BaseModel):
    hidden_layer_sizes: list[int] = Field(default_factory=lambda: [128, 64, 32])
    max_iter: int = 500
    alpha: float = 1e-4


class UNetConfig(BaseModel):
    epochs: int = 30
    lr: float = 1e-3
    batch_size: int = 8
    dropout: float = 0.2
    mc_samples: int = 20


class DeepLabV3PlusConfig(BaseModel):
    encoder: str = "resnet34"
    epochs: int = 20
    lr: float = 5e-4
    batch_size: int = 4


class CompareConfig(BaseModel):
    models_water: list[str] = Field(
        default_factory=lambda: ["threshold_change", "logreg", "random_forest", "xgboost", "mlp"]
    )
    models_flood: list[str] = Field(
        default_factory=lambda: ["logreg", "random_forest", "xgboost", "mlp", "unet", "attention_unet", "deeplabv3plus"]
    )
    random_forest: RandomForestConfig = Field(default_factory=RandomForestConfig)
    mlp: MlpConfig = Field(default_factory=MlpConfig)
    unet: UNetConfig = Field(default_factory=UNetConfig)
    attention_unet: UNetConfig = Field(default_factory=UNetConfig)
    deeplabv3plus: DeepLabV3PlusConfig = Field(default_factory=DeepLabV3PlusConfig)


class ModelConfig(BaseModel):
    water_model: Literal["threshold_change", "logreg"] = "logreg"
    flood_given_water_model: Literal["logreg", "xgboost_if_available"] = "logreg"
    calibration: Literal["isotonic", "sigmoid"] = "isotonic"
    ensemble_n: int = 1
    n_hist_windows: int = 4
    tile_size: int = 128
    # Origen de las etiquetas de ENTRENAMIENTO: "pseudo" (supervisión débil, sin
    # ground truth) o "gt" (etiquetas externas). La evaluación usa siempre el GT si existe.
    train_on: Literal["pseudo", "gt"] = "pseudo"
    # Partición espacial por bloques de tile_size: un fold de test, uno de validación
    # (selección de umbral) y el resto de entrenamiento.
    n_folds: int = 5
    test_fold: int = 0
    val_fold: int = 1
    threshold_change: ThresholdChangeConfig = Field(default_factory=ThresholdChangeConfig)
    logreg: LogRegConfig = Field(default_factory=LogRegConfig)
    compare: CompareConfig = Field(default_factory=CompareConfig)


class AppConfig(BaseModel):
    run_name: str = "floodprob_run"
    stac: STACConfig
    aoi_geojson: str
    crs_salida: str = "EPSG:25830"
    resolution_m: int = 10
    seed: int = 42
    time: TimeConfig
    filters: FiltersConfig = Field(default_factory=FiltersConfig)
    dem: DemConfig = Field(default_factory=DemConfig)
    external_labels: ExternalLabelsConfig = Field(default_factory=ExternalLabelsConfig)
    outputs: OutputsConfig = Field(default_factory=OutputsConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)

    @field_validator("resolution_m")
    @classmethod
    def validate_resolution(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("resolution_m must be > 0")
        return value

    def resolve_path(self, raw_path: str) -> Path:
        return Path(raw_path).expanduser().resolve()


def load_config(path: str | Path) -> AppConfig:
    path_obj = Path(path).expanduser().resolve()
    with path_obj.open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f)
    return AppConfig.model_validate(payload)
