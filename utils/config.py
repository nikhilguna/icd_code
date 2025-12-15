"""Configuration management for ICD code prediction experiments."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional
import yaml


@dataclass
class DataConfig:
    """Data-related configuration."""
    mimic3_database: str = "mimiciii"
    athena_output_bucket: str = "s3://your-bucket/athena-results/"
    raw_data_dir: str = "data/raw/"
    processed_data_dir: str = "data/processed/"
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    random_seed: int = 42


@dataclass
class LabelConfig:
    """Label encoding configuration."""
    top_k_codes: int = 50
    min_code_frequency: int = 10
    use_full_codes: bool = False


@dataclass
class PreprocessingConfig:
    """Preprocessing configuration."""
    max_length_caml: int = 4096
    max_length_longformer: int = 4096
    lowercase: bool = True
    remove_deidentified: bool = True
    remove_numbers: bool = False
    preserve_sections: bool = True


@dataclass
class CAMLConfig:
    """CAML model configuration."""
    embedding_dim: int = 300
    num_filters: int = 256
    filter_sizes: List[int] = field(default_factory=lambda: [3, 5, 7])
    dropout: float = 0.2
    use_pretrained_embeddings: bool = False


@dataclass
class LongformerConfig:
    """Longformer model configuration."""
    model_name: str = "allenai/longformer-base-4096"
    hidden_dim: int = 768
    dropout: float = 0.1
    freeze_layers: int = 6
    use_global_attention: bool = True


@dataclass
class TrainingConfig:
    """Training configuration."""
    batch_size: int = 32
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    epochs: int = 50
    patience: int = 5
    gradient_clip: float = 1.0
    gradient_accumulation_steps: int = 1
    warmup_ratio: float = 0.0
    mixed_precision: bool = True
    num_workers: int = 4
    pin_memory: bool = True


@dataclass
class EvaluationConfig:
    """Evaluation configuration."""
    precision_at_k: List[int] = field(default_factory=lambda: [5, 10])
    compute_roc_auc: bool = True
    stratified_analysis: bool = True
    frequency_bins: Dict[str, int] = field(default_factory=lambda: {
        "head": 1000,
        "medium": 100,
        "tail": 0
    })


@dataclass
class LoggingConfig:
    """Logging and checkpoint configuration."""
    use_wandb: bool = False
    project_name: str = "icd-prediction"
    log_every_n_steps: int = 100
    checkpoint_dir: str = "checkpoints/"
    save_top_k: int = 3


@dataclass
class Config:
    """Main configuration container."""
    data: DataConfig = field(default_factory=DataConfig)
    labels: LabelConfig = field(default_factory=LabelConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    caml: CAMLConfig = field(default_factory=CAMLConfig)
    longformer: LongformerConfig = field(default_factory=LongformerConfig)
    training_caml: TrainingConfig = field(default_factory=lambda: TrainingConfig(
        batch_size=32,
        learning_rate=0.001,
        epochs=50,
        patience=5
    ))
    training_longformer: TrainingConfig = field(default_factory=lambda: TrainingConfig(
        batch_size=4,
        learning_rate=2e-5,
        epochs=10,
        patience=3,
        gradient_accumulation_steps=8,
        warmup_ratio=0.1,
        weight_decay=0.01
    ))
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    device: str = "cuda"
    seed: int = 42


def load_config(config_path: Optional[str] = None) -> Config:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML config file. If None, returns default config.
        
    Returns:
        Config object with all settings.
    """
    config = Config()
    
    if config_path is None:
        return config
    
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, "r") as f:
        yaml_config = yaml.safe_load(f)
    
    # Update data config
    if "data" in yaml_config:
        for key, value in yaml_config["data"].items():
            if hasattr(config.data, key):
                setattr(config.data, key, value)
    
    # Update labels config
    if "labels" in yaml_config:
        for key, value in yaml_config["labels"].items():
            if hasattr(config.labels, key):
                setattr(config.labels, key, value)
    
    # Update preprocessing config
    if "preprocessing" in yaml_config:
        for key, value in yaml_config["preprocessing"].items():
            if hasattr(config.preprocessing, key):
                setattr(config.preprocessing, key, value)
    
    # Update CAML config
    if "caml" in yaml_config:
        for key, value in yaml_config["caml"].items():
            if hasattr(config.caml, key):
                setattr(config.caml, key, value)
    
    # Update Longformer config
    if "longformer" in yaml_config:
        for key, value in yaml_config["longformer"].items():
            if hasattr(config.longformer, key):
                setattr(config.longformer, key, value)
    # Backward compatibility: also check for "led"
    elif "led" in yaml_config:
        for key, value in yaml_config["led"].items():
            if hasattr(config.longformer, key):
                setattr(config.longformer, key, value)
    
    # Update training configs
    if "training" in yaml_config:
        if "caml" in yaml_config["training"]:
            for key, value in yaml_config["training"]["caml"].items():
                if hasattr(config.training_caml, key):
                    setattr(config.training_caml, key, value)
        if "longformer" in yaml_config["training"]:
            for key, value in yaml_config["training"]["longformer"].items():
                if hasattr(config.training_longformer, key):
                    setattr(config.training_longformer, key, value)
        # Backward compatibility: also check for "led"
        elif "led" in yaml_config["training"]:
            for key, value in yaml_config["training"]["led"].items():
                if hasattr(config.training_longformer, key):
                    setattr(config.training_longformer, key, value)
        # Common settings
        if "mixed_precision" in yaml_config["training"]:
            config.training_caml.mixed_precision = yaml_config["training"]["mixed_precision"]
            config.training_longformer.mixed_precision = yaml_config["training"]["mixed_precision"]
        if "num_workers" in yaml_config["training"]:
            config.training_caml.num_workers = yaml_config["training"]["num_workers"]
            config.training_longformer.num_workers = yaml_config["training"]["num_workers"]
        if "pin_memory" in yaml_config["training"]:
            config.training_caml.pin_memory = yaml_config["training"]["pin_memory"]
            config.training_longformer.pin_memory = yaml_config["training"]["pin_memory"]
    
    # Update evaluation config
    if "evaluation" in yaml_config:
        for key, value in yaml_config["evaluation"].items():
            if hasattr(config.evaluation, key):
                setattr(config.evaluation, key, value)
    
    # Update logging config
    if "logging" in yaml_config:
        for key, value in yaml_config["logging"].items():
            if hasattr(config.logging, key):
                setattr(config.logging, key, value)
    
    # Update hardware config
    if "hardware" in yaml_config:
        if "device" in yaml_config["hardware"]:
            config.device = yaml_config["hardware"]["device"]
        if "seed" in yaml_config["hardware"]:
            config.seed = yaml_config["hardware"]["seed"]
    
    return config


def save_config(config: Config, save_path: str) -> None:
    """
    Save configuration to YAML file.
    
    Args:
        config: Config object to save.
        save_path: Path to save YAML file.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    config_dict = {
        "data": {
            "mimic3_database": config.data.mimic3_database,
            "athena_output_bucket": config.data.athena_output_bucket,
            "raw_data_dir": config.data.raw_data_dir,
            "processed_data_dir": config.data.processed_data_dir,
            "train_ratio": config.data.train_ratio,
            "val_ratio": config.data.val_ratio,
            "test_ratio": config.data.test_ratio,
            "random_seed": config.data.random_seed,
        },
        "labels": {
            "top_k_codes": config.labels.top_k_codes,
            "min_code_frequency": config.labels.min_code_frequency,
            "use_full_codes": config.labels.use_full_codes,
        },
        "preprocessing": {
            "max_length_caml": config.preprocessing.max_length_caml,
            "max_length_longformer": config.preprocessing.max_length_longformer,
            "lowercase": config.preprocessing.lowercase,
            "remove_deidentified": config.preprocessing.remove_deidentified,
            "remove_numbers": config.preprocessing.remove_numbers,
            "preserve_sections": config.preprocessing.preserve_sections,
        },
        "caml": {
            "embedding_dim": config.caml.embedding_dim,
            "num_filters": config.caml.num_filters,
            "filter_sizes": config.caml.filter_sizes,
            "dropout": config.caml.dropout,
            "use_pretrained_embeddings": config.caml.use_pretrained_embeddings,
        },
        "longformer": {
            "model_name": config.longformer.model_name,
            "hidden_dim": config.longformer.hidden_dim,
            "dropout": config.longformer.dropout,
            "freeze_layers": config.longformer.freeze_layers,
            "use_global_attention": config.longformer.use_global_attention,
        },
        "training": {
            "caml": {
                "batch_size": config.training_caml.batch_size,
                "learning_rate": config.training_caml.learning_rate,
                "weight_decay": config.training_caml.weight_decay,
                "epochs": config.training_caml.epochs,
                "patience": config.training_caml.patience,
                "gradient_clip": config.training_caml.gradient_clip,
            },
            "longformer": {
                "batch_size": config.training_longformer.batch_size,
                "gradient_accumulation_steps": config.training_longformer.gradient_accumulation_steps,
                "learning_rate": config.training_longformer.learning_rate,
                "weight_decay": config.training_longformer.weight_decay,
                "epochs": config.training_longformer.epochs,
                "patience": config.training_longformer.patience,
                "warmup_ratio": config.training_longformer.warmup_ratio,
                "gradient_clip": config.training_longformer.gradient_clip,
            },
            "mixed_precision": config.training_caml.mixed_precision,
            "num_workers": config.training_caml.num_workers,
            "pin_memory": config.training_caml.pin_memory,
        },
        "evaluation": {
            "precision_at_k": config.evaluation.precision_at_k,
            "compute_roc_auc": config.evaluation.compute_roc_auc,
            "stratified_analysis": config.evaluation.stratified_analysis,
            "frequency_bins": config.evaluation.frequency_bins,
        },
        "logging": {
            "use_wandb": config.logging.use_wandb,
            "project_name": config.logging.project_name,
            "log_every_n_steps": config.logging.log_every_n_steps,
            "checkpoint_dir": config.logging.checkpoint_dir,
            "save_top_k": config.logging.save_top_k,
        },
        "hardware": {
            "device": config.device,
            "seed": config.seed,
        },
    }
    
    with open(save_path, "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
