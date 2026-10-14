"""Initialize Henan stage 2 from stage 1 and the exact teacher used there."""
import hashlib
from pathlib import Path

import torch
import yaml

from SaCMR_ECG_Align.models import _unwrapped_state, ecg_encoders, teacher_from_checkpoint


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_path(value, name, checkpoint_dir=None):
    if not isinstance(value, (str, Path)) or not str(value):
        raise ValueError("Missing %s; supply an explicit path" % name)
    path = Path(value).expanduser()
    if not path.is_absolute() and not path.is_file() and checkpoint_dir is not None:
        path = checkpoint_dir / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError("%s does not exist: %s" % (name, path))
    return path


def _validate_source_context(stage1, teacher):
    # These describe the shared UKB source cohort, never the new Henan cohort.
    for key in ("data_fingerprint", "phenotype_norm"):
        if not isinstance(teacher.get(key), dict) or not teacher[key]:
            raise ValueError("Teacher checkpoint is missing %s" % key)
        if teacher[key] != stage1[key]:
            raise ValueError("Teacher and stage1 %s differ" % key)
    teacher_config = teacher.get("config")
    if not isinstance(teacher_config, dict) or "phenotype_indices" not in teacher_config:
        raise ValueError("Teacher checkpoint is missing config.phenotype_indices")
    if teacher_config["phenotype_indices"] != stage1["config"]["phenotype_indices"]:
        raise ValueError("Teacher and stage1 phenotype_indices differ")


def load_stage2_models(stage1_path, teacher_path=None, ecg_config_path=None, device="cpu"):
    """Return ``(ecg_model, frozen_teacher, source_provenance)``.

    The stage 1 checkpoint supplies ECG weights and binds the teacher by SHA256.
    Its saved paths are defaults; explicit paths support relocating those files.
    The ECG YAML supplies architecture only, so its pretraining and alignment
    paths are never loaded. Stage 1 optimizer and phenotype-head state are not
    carried into stage 2.
    """
    stage1_path = _source_path(stage1_path, "stage1_path")
    checkpoint = torch.load(stage1_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError("Stage1 checkpoint must be a dictionary")
    required = {"model", "config", "teacher_checkpoint_sha256",
                "data_fingerprint", "phenotype_norm"}
    if not required.issubset(checkpoint):
        raise ValueError("Stage1 checkpoint missing keys: %s" % sorted(required - set(checkpoint)))
    config = checkpoint["config"]
    if not isinstance(config, dict) or config.get("stage") != "stage1":
        raise ValueError("Stage1 checkpoint config.stage must be 'stage1'")
    if not isinstance(config.get("phenotype_indices"), list):
        raise ValueError("Stage1 checkpoint is missing config.phenotype_indices")
    for key in ("data_fingerprint", "phenotype_norm"):
        if not isinstance(checkpoint[key], dict) or not checkpoint[key]:
            raise ValueError("Stage1 checkpoint is missing %s" % key)
    if not isinstance(checkpoint["teacher_checkpoint_sha256"], str) or not checkpoint["teacher_checkpoint_sha256"]:
        raise ValueError("Stage1 checkpoint is missing teacher_checkpoint_sha256")
    state = _unwrapped_state(checkpoint["model"])
    stage1 = {key: checkpoint[key] for key in required if key != "model"}
    stage1_epoch = checkpoint.get("epoch")
    # Release the large stage 1 optimizer before constructing either model.
    del checkpoint

    teacher_path = _source_path(teacher_path if teacher_path is not None else config.get("cmr_teacher_path"),
                                "cmr_teacher_path", stage1_path.parent)
    ecg_config_path = _source_path(ecg_config_path if ecg_config_path is not None else config.get("ecg_config_path"),
                                   "ecg_config_path", stage1_path.parent)
    teacher_sha256 = _sha256(teacher_path)
    if teacher_sha256 != stage1["teacher_checkpoint_sha256"]:
        raise ValueError("Stage1 teacher checkpoint SHA256 differs from the selected teacher contents")

    config_bytes = ecg_config_path.read_bytes()
    architecture = yaml.safe_load(config_bytes)
    if not isinstance(architecture, dict) or not isinstance(architecture.get("model"), dict):
        raise ValueError("ECG config must contain a model architecture dictionary")
    model_name = architecture.get("model_name")
    factory = ecg_encoders.__dict__.get(model_name) if isinstance(model_name, str) else None
    if not callable(factory):
        raise ValueError("Unsupported ECG model: %s" % model_name)
    if architecture.get("lora", {}).get("enabled", False):
        raise ValueError("Stage 2 requires the native ECG encoder without LoRA")
    model_config = dict(architecture["model"], num_classes=None, pred_metric=False)
    architecture["model"] = model_config
    ecg_model = factory(**model_config)
    ecg_model.load_state_dict({name: value for name, value in state.items()
                               if not name.startswith("head.")}, strict=True)
    del state

    teacher, teacher_checkpoint = teacher_from_checkpoint(teacher_path, use_checkpoint=False)
    _validate_source_context(stage1, teacher_checkpoint)
    embed_dim = getattr(ecg_model, "width", getattr(ecg_model, "embed_dim", None))
    if embed_dim != teacher.cmr_encoder.embed_dim:
        raise ValueError("ECG and teacher embedding dimensions must agree")
    ecg_model.to(device)
    teacher.to(device).eval()
    provenance = {
        "stage1": {"path": str(stage1_path), "sha256": _sha256(stage1_path), "epoch": stage1_epoch},
        "teacher": {"path": str(teacher_path), "sha256": teacher_sha256,
                    "epoch": teacher_checkpoint.get("epoch")},
        "ecg_config": {"path": str(ecg_config_path), "sha256": hashlib.sha256(config_bytes).hexdigest(),
                       "config": architecture},
    }
    return ecg_model, teacher, provenance


def validate_stage2_provenance(checkpoint, provenance):
    """Reject changed initialization sources on resume; allow path relocation."""
    saved = checkpoint.get("source_provenance")
    if not isinstance(saved, dict):
        raise ValueError("Stage2 checkpoint is missing source_provenance")
    for source in ("stage1", "teacher", "ecg_config"):
        current = provenance.get(source)
        previous = saved.get(source)
        if (not isinstance(current, dict) or not isinstance(previous, dict)
                or not current.get("sha256") or previous.get("sha256") != current["sha256"]):
            raise ValueError("Resume source_provenance changed: %s SHA256" % source)
    architecture = provenance["ecg_config"].get("config")
    if not isinstance(architecture, dict) or saved["ecg_config"].get("config") != architecture:
        raise ValueError("Resume ecg_config architecture changed")
