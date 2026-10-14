"""Short-axis semantic teacher and checkpoint-compatible ST-MEM encoder."""
from functools import partial
from pathlib import Path

import torch
from torch import nn
import yaml
from timm.models.vision_transformer import Block, PatchEmbed

from SaCMR_encoder.models_mae import MaskedAutoencoderSA, _spatial_positions
from models import encoder as ecg_encoders


class SAEncoder(nn.Module):
    """The pretrained early-fusion encoder, without MAE decoder parameters.

    Module names match ``MaskedAutoencoderSA`` exactly. Reusing its encoder
    methods keeps the spatial ordering, slice attention and invalid-slice
    exclusion identical to pretraining. Alignment always uses all 36 patches.
    """

    _validate_input = MaskedAutoencoderSA._validate_input
    _mask_indices = MaskedAutoencoderSA._mask_indices
    _encode = MaskedAutoencoderSA._encode
    forward_encoder = MaskedAutoencoderSA.forward_encoder
    forward_features = MaskedAutoencoderSA.forward_features

    def __init__(self, img_size=96, patch_size=16, in_chans=50, num_slices=10,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.,
                 use_checkpoint=True):
        super().__init__()
        if min(img_size, patch_size, in_chans, num_slices, depth) <= 0 or img_size % patch_size:
            raise ValueError("Image size must be positive and divisible by patch size")
        self.encoder_config = dict(img_size=img_size, patch_size=patch_size,
                                   in_chans=in_chans, num_slices=num_slices,
                                   embed_dim=embed_dim, depth=depth,
                                   num_heads=num_heads, mlp_ratio=mlp_ratio,
                                   use_checkpoint=bool(use_checkpoint))
        self.img_size = img_size
        self.in_chans = in_chans
        self.num_slices = num_slices
        self.embed_dim = embed_dim
        self.num_features = embed_dim
        self.use_checkpoint = bool(use_checkpoint)
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.num_patches = self.patch_embed.num_patches
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(_spatial_positions(img_size // patch_size, embed_dim),
                                      requires_grad=False)
        self.slice_pos_embed = nn.Parameter(torch.zeros(1, num_slices, 1, embed_dim))
        self.slice_attention = nn.Sequential(
            norm_layer(embed_dim), nn.Linear(embed_dim, min(128, embed_dim)),
            nn.Tanh(), nn.Linear(min(128, embed_dim), 1, bias=False))
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for _ in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.apply(MaskedAutoencoderSA._initialize_module)
        nn.init.xavier_uniform_(self.patch_embed.proj.weight.flatten(1))
        nn.init.normal_(self.cls_token, std=.02)
        nn.init.normal_(self.slice_pos_embed, std=.02)

    def forward(self, images, slice_valid_mask):
        return self.forward_features(images, slice_valid_mask)


def _checkpoint_encoder_config(checkpoint, use_checkpoint):
    config = checkpoint.get("config", {})
    if not isinstance(config, dict):
        raise ValueError("Checkpoint config must be a dictionary")
    declared = checkpoint.get("encoder_config", config.get("encoder_config"))
    if declared is not None:
        if not isinstance(declared, dict):
            raise ValueError("Checkpoint encoder_config must be a dictionary")
        result = dict(declared)
    else:
        model_name = config.get("model", "mae_vit_base_patch16")
        if model_name == "mae_vit_base_patch16":
            result = {}
        elif model_name == "mae_vit_tiny_patch16":
            result = dict(embed_dim=64, depth=2, num_heads=4)
        else:
            raise ValueError("Unsupported MAE architecture: {}".format(model_name))
    result["use_checkpoint"] = bool(use_checkpoint)
    return result


def _unwrapped_state(state):
    if not isinstance(state, dict):
        raise ValueError("Checkpoint weights must be a state dictionary")
    if state and all(name.startswith("module.") for name in state):
        return {name[7:]: value for name, value in state.items()}
    return state


def load_sa_encoder(path, use_checkpoint=True):
    """Load every encoder weight strictly, discarding only MAE decoder weights.

    Returns ``(encoder, info)``. ``info`` contains architecture/provenance and
    key names only, so optimizer tensors from the MAE checkpoint can be freed.
    """
    checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise ValueError("MAE checkpoint must contain a model state dictionary")
    model = SAEncoder(**_checkpoint_encoder_config(checkpoint, use_checkpoint))
    state = _unwrapped_state(checkpoint["model"])
    expected = set(model.state_dict())
    decoder_prefixes = ("decoder_embed.", "decoder_blocks.",
                        "decoder_norm.", "decoder_pred.")
    decoder_tokens = {"mask_token", "decoder_pos_embed", "decoder_slice_pos_embed"}
    discarded = [name for name in state if name not in expected and
                 (name in decoder_tokens or name.startswith(decoder_prefixes))]
    unexpected = set(state) - expected - set(discarded)
    if unexpected:
        raise ValueError("Unexpected MAE checkpoint keys: {}".format(sorted(unexpected)))
    encoder_state = {name: value for name, value in state.items() if name in expected}
    model.load_state_dict(encoder_state, strict=True)
    info = {"path": str(Path(path).resolve()), "epoch": checkpoint.get("epoch"),
            "encoder_config": dict(model.encoder_config),
            "loaded_keys": sorted(encoder_state), "ignored_decoder_keys": sorted(discarded)}
    return model, info


def _semantic_head(embed_dim, hidden_dim):
    return nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, hidden_dim),
                         nn.GELU(), nn.Linear(hidden_dim, embed_dim), nn.LayerNorm(embed_dim))


def _phenotype_head(embed_dim, hidden_dim, num_phenotypes):
    return nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, hidden_dim),
                         nn.GELU(), nn.Linear(hidden_dim, num_phenotypes))


class SemanticTeacher(nn.Module):
    def __init__(self, cmr_encoder, sem_hidden_dim=768, pheno_hidden_dim=256,
                 num_phenotypes=27):
        super().__init__()
        self.cmr_encoder = cmr_encoder
        self.sem_head = _semantic_head(cmr_encoder.embed_dim, sem_hidden_dim)
        self.pheno_head = _phenotype_head(cmr_encoder.embed_dim, pheno_hidden_dim, num_phenotypes)

    @property
    def encoder_config(self):
        return dict(self.cmr_encoder.encoder_config)

    def forward(self, images, slice_valid_mask):
        semantic = self.sem_head(self.cmr_encoder(images, slice_valid_mask))
        return semantic, self.pheno_head(semantic)


def set_cmr_trainable(cmr_encoder, train_last_blocks, last_blocks):
    """Freeze the encoder, optionally enabling its final K blocks plus norm."""
    if not isinstance(last_blocks, int) or last_blocks < 0:
        raise ValueError("last_blocks must be a nonnegative integer")
    for parameter in cmr_encoder.parameters():
        parameter.requires_grad_(False)
    if train_last_blocks and last_blocks:
        for block in cmr_encoder.blocks[-last_blocks:]:
            for parameter in block.parameters():
                parameter.requires_grad_(True)
        for parameter in cmr_encoder.norm.parameters():
            parameter.requires_grad_(True)


def teacher_from_checkpoint(path, device="cpu", use_checkpoint=True):
    """Strictly restore a teacher and return it frozen in evaluation mode."""
    checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError("Teacher checkpoint must be a dictionary")
    required = {"cmr_encoder", "sem_head", "pheno_head"}
    if not required.issubset(checkpoint):
        raise ValueError("Missing teacher checkpoint modules: {}".format(
            sorted(required - set(checkpoint))))
    encoder = SAEncoder(**_checkpoint_encoder_config(checkpoint, use_checkpoint))
    sem_state = _unwrapped_state(checkpoint["sem_head"])
    pheno_state = _unwrapped_state(checkpoint["pheno_head"])
    teacher = SemanticTeacher(encoder, sem_hidden_dim=sem_state["1.weight"].shape[0],
                              pheno_hidden_dim=pheno_state["1.weight"].shape[0],
                              num_phenotypes=pheno_state["3.weight"].shape[0])
    encoder.load_state_dict(_unwrapped_state(checkpoint["cmr_encoder"]), strict=True)
    teacher.sem_head.load_state_dict(sem_state, strict=True)
    teacher.pheno_head.load_state_dict(pheno_state, strict=True)
    teacher.to(device)
    teacher.requires_grad_(False)
    teacher.eval()
    return teacher, checkpoint


def build_ecg_encoder(config_path):
    """Build native ST-MEM, requiring all nonclassifier pretrained weights."""
    config_path = Path(config_path)
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    model_name = config["model_name"]
    if model_name not in ecg_encoders.__dict__:
        raise ValueError("Unsupported ECG model: {}".format(model_name))
    if config.get("lora", {}).get("enabled", False):
        raise ValueError("This stage uses the native ST-MEM encoder without LoRA")
    model_config = dict(config["model"])
    model_config["num_classes"] = None
    model_config["pred_metric"] = False
    model = ecg_encoders.__dict__[model_name](**model_config)
    mode = config.get("mode", "finetune")
    if mode == "scratch":
        return model
    if mode not in ("finetune", "align"):
        raise ValueError("Unsupported ECG initialization mode: {}".format(mode))
    path_key = "afterAlign_path" if mode == "align" else "encoder_path"
    if not config.get(path_key):
        raise ValueError("Missing {} in {}".format(path_key, config_path))
    path = Path(config[path_key])
    if not path.is_absolute() and not path.exists():
        path = config_path.parent / path
    checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise ValueError("ECG checkpoint must contain a model state dictionary")
    state = _unwrapped_state(checkpoint["model"])
    state = {name: value for name, value in state.items() if not name.startswith("head.")}
    model.load_state_dict(state, strict=True)
    return model
