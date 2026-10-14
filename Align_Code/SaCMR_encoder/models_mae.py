"""Early slice fusion MAE, following the structure of CMR_encoder's MAE.

Time remains the channel dimension. Independent patch projection is followed
by position-aware pooling across valid slices, before the spatial ViT. MAE
removes the same spatial positions in every slice before that pooling.
"""
import math
from functools import partial

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from timm.models.vision_transformer import Block, PatchEmbed


def _spatial_positions(grid_size, dim):
    """Fixed XY sine/cosine positions, with a zero CLS position (as in MAE)."""
    if dim % 4:
        raise ValueError("Position embedding dimension must be divisible by four")
    frequencies = 1. / (10000 ** (torch.arange(dim // 4).float() / (dim // 4)))
    y, x = torch.meshgrid(torch.arange(grid_size), torch.arange(grid_size), indexing='ij')
    x = x.flatten().float()[:, None] * frequencies
    y = y.flatten().float()[:, None] * frequencies
    positions = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=-1)
    return torch.cat((torch.zeros(1, dim), positions), dim=0).unsqueeze(0)


class MaskedAutoencoderSA(nn.Module):
    """Mask-aware short-axis MAE with 36 fused spatial tokens at 96x96.

    Args use the existing long-axis ViT-B defaults, except for the smaller
    decoder. ``decoder_slice_chunk_size`` caps how many slice positions per
    patient are reconstructed together. Checkpointing recomputes decoder and
    loss intermediates on backward, rather than retaining every slice's graph.
    """

    def __init__(self, img_size=96, patch_size=16, in_chans=50, num_slices=10,
                 embed_dim=768, depth=12, num_heads=12,
                 decoder_embed_dim=256, decoder_depth=2, decoder_num_heads=8,
                 mlp_ratio=4., norm_pix_loss=False, use_checkpoint=True,
                 decoder_slice_chunk_size=2):
        super().__init__()
        if img_size % patch_size or min(img_size, patch_size, in_chans, num_slices) <= 0:
            raise ValueError("Image size must be positive and divisible by patch size")
        if decoder_slice_chunk_size < 1:
            raise ValueError("decoder_slice_chunk_size must be positive")
        self.img_size = img_size
        self.in_chans = in_chans
        self.num_slices = num_slices
        self.embed_dim = embed_dim
        self.num_features = embed_dim
        self.norm_pix_loss = norm_pix_loss
        self.use_checkpoint = use_checkpoint
        self.decoder_slice_chunk_size = decoder_slice_chunk_size
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.num_patches = self.patch_embed.num_patches
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            _spatial_positions(img_size // patch_size, embed_dim), requires_grad=False)
        self.slice_pos_embed = nn.Parameter(torch.zeros(1, num_slices, 1, embed_dim))
        # Content and slice position jointly determine each slice's weight.
        self.slice_attention = nn.Sequential(
            norm_layer(embed_dim), nn.Linear(embed_dim, min(128, embed_dim)),
            nn.Tanh(), nn.Linear(min(128, embed_dim), 1, bias=False))
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for _ in range(depth)])
        self.norm = norm_layer(embed_dim)

        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(
            _spatial_positions(img_size // patch_size, decoder_embed_dim), requires_grad=False)
        self.decoder_slice_pos_embed = nn.Parameter(torch.zeros(num_slices, decoder_embed_dim))
        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio,
                  qkv_bias=True, norm_layer=norm_layer)
            for _ in range(decoder_depth)])
        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size ** 2 * in_chans)
        self.apply(self._initialize_module)
        nn.init.xavier_uniform_(self.patch_embed.proj.weight.flatten(1))
        for token in (self.cls_token, self.mask_token, self.slice_pos_embed, self.decoder_slice_pos_embed):
            nn.init.normal_(token, std=.02)

    @staticmethod
    def _initialize_module(module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def _validate_input(self, images, slice_valid_mask):
        expected = (self.num_slices, self.in_chans, self.img_size, self.img_size)
        if images.ndim != 5 or tuple(images.shape[1:]) != expected or images.shape[0] == 0:
            raise ValueError("Expected images [B,{}], got {}".format(
                ','.join(map(str, expected)), tuple(images.shape)))
        if not images.is_floating_point():
            raise ValueError("images must be floating point tensors")
        if tuple(slice_valid_mask.shape) != tuple(images.shape[:2]):
            raise ValueError("slice_valid_mask must have shape [B,S]")
        if not torch.all((slice_valid_mask == 0) | (slice_valid_mask == 1)):
            raise ValueError("slice_valid_mask must contain only 0 or 1")
        valid = slice_valid_mask.to(device=images.device, dtype=torch.bool)
        if not valid.any(dim=1).all():
            raise ValueError("Each patient must have at least one valid slice")
        return valid

    def _mask_indices(self, batch_size, device, mask_ratio):
        if not math.isfinite(mask_ratio) or not 0 <= mask_ratio < 1:
            raise ValueError("mask_ratio must be in [0,1)")
        keep = int(self.num_patches * (1 - mask_ratio))
        if keep < 1:
            raise ValueError("mask_ratio must leave at least one visible patch")
        if mask_ratio == 0:
            shuffled = torch.arange(self.num_patches, device=device).expand(batch_size, -1)
        else:
            shuffled = torch.rand(batch_size, self.num_patches, device=device).argsort(dim=1)
        restore = shuffled.argsort(dim=1)
        mask = torch.ones(batch_size, self.num_patches, device=device)
        mask[:, :keep] = 0
        return shuffled[:, :keep], mask.gather(1, restore), restore

    def _encode(self, images, valid, mask_ratio):
        batch_size, slices = images.shape[:2]
        keep_ids, mask, restore = self._mask_indices(batch_size, images.device, mask_ratio)
        # Select valid images rather than multiplying by zero: invalid data may
        # be duplicates or even NaN, and must have exactly zero input gradient.
        valid_features = self.patch_embed(images[valid])
        features = valid_features.new_zeros(batch_size, slices, self.num_patches, self.embed_dim)
        features[valid] = valid_features
        visible = features.gather(
            2, keep_ids[:, None, :, None].expand(-1, slices, -1, self.embed_dim))
        visible = visible + self.slice_pos_embed.to(visible.dtype)
        scores = self.slice_attention(visible).float()
        scores = scores.masked_fill(~valid[:, :, None, None], float('-inf'))
        weights = scores.softmax(dim=1).to(visible.dtype)
        x = (visible * weights).sum(dim=1)
        positions = self.pos_embed[:, 1:].expand(batch_size, -1, -1)
        x = x + positions.gather(1, keep_ids[:, :, None].expand(-1, -1, self.embed_dim)).to(x.dtype)
        cls = (self.cls_token + self.pos_embed[:, :1]).expand(batch_size, -1, -1).to(x.dtype)
        x = torch.cat((cls, x), dim=1)
        for block in self.blocks:
            if self.use_checkpoint and self.training and torch.is_grad_enabled():
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        return self.norm(x), mask, restore

    def forward_encoder(self, images, slice_valid_mask, mask_ratio=.75):
        """Return visible volume tokens, spatial hidden-mask and restore indices."""
        return self._encode(images, self._validate_input(images, slice_valid_mask), mask_ratio)

    def forward_features(self, images, slice_valid_mask):
        """Unmasked whole-volume embedding [B,768] with production defaults."""
        latent, _, _ = self.forward_encoder(images, slice_valid_mask, mask_ratio=0)
        return latent[:, 0]

    def patchify(self, images):
        """[B,S,C,H,W] -> [B,S,L,p*p*C], matching long-axis MAE pixel order."""
        p = self.patch_embed.patch_size[0]
        b, s, c, height, width = images.shape
        if height % p or width % p:
            raise ValueError("Spatial dimensions must be divisible by patch size")
        x = images.reshape(b, s, c, height // p, p, width // p, p)
        return x.permute(0, 1, 3, 5, 4, 6, 2).reshape(b, s, (height // p) * (width // p), p*p*c)

    def unpatchify(self, patches):
        p = self.patch_embed.patch_size[0]
        b, s, length, values = patches.shape
        grid = math.isqrt(length)
        if grid * grid != length or values != p*p*self.in_chans:
            raise ValueError("Invalid patch tensor shape")
        x = patches.reshape(b, s, grid, grid, p, p, self.in_chans)
        return x.permute(0, 1, 6, 2, 4, 3, 5).reshape(b, s, self.in_chans, grid*p, grid*p)

    def _decode_features(self, latent, ids_restore, slice_ids):
        x = self.decoder_embed(latent)
        hidden = self.mask_token.expand(x.shape[0], self.num_patches + 1 - x.shape[1], -1)
        patches = torch.cat((x[:, 1:], hidden.to(x.dtype)), dim=1)
        patches = patches.gather(1, ids_restore[:, :, None].expand(-1, -1, x.shape[-1]))
        x = torch.cat((x[:, :1], patches), dim=1)
        x = (x + self.decoder_pos_embed.to(x.dtype)
             + self.decoder_slice_pos_embed[slice_ids, None].to(x.dtype))
        for block in self.decoder_blocks:
            x = block(x)
        return self.decoder_norm(x)[:, 1:]

    def _target(self, images):
        target = self.patchify(images[:, None])[:, 0].float()
        if self.norm_pix_loss:
            target = (target - target.mean(dim=-1, keepdim=True)) / (
                target.var(dim=-1, keepdim=True, unbiased=False) + 1.e-6).sqrt()
        return target

    def _decode_loss(self, latent, restore, images, spatial_mask, slice_ids):
        decoded = self._decode_features(latent, restore, slice_ids)
        # Only materialize pixel predictions for hidden patches during training.
        hidden = spatial_mask.bool()
        patient_ids = hidden.nonzero(as_tuple=True)[0]
        pred = self.decoder_pred(decoded[hidden]).float()
        target = self._target(images)[hidden]
        patch_loss = (pred - target).square().mean(dim=-1)
        sums = patch_loss.new_zeros(images.shape[0]).index_add(0, patient_ids, patch_loss)
        return sums / hidden.sum(dim=1)

    def forward_loss(self, images, pred, reconstruction_mask, slice_valid_mask):
        """Reference full-prediction loss; training uses chunked hidden-only loss."""
        valid = self._validate_input(images, slice_valid_mask)
        if tuple(reconstruction_mask.shape) != (images.shape[0], self.num_patches):
            raise ValueError("reconstruction_mask must have shape [B,L]")
        if not torch.all((reconstruction_mask == 0) | (reconstruction_mask == 1)):
            raise ValueError("reconstruction_mask must be binary")
        if not reconstruction_mask.bool().any(dim=1).all():
            raise ValueError("Loss requires at least one hidden patch per patient")
        patient_ids = valid.nonzero(as_tuple=True)[0]
        target = self._target(images[valid])
        hidden = reconstruction_mask[patient_ids].bool()
        # Boolean selection excludes invalid slices and visible patch pixels.
        errors = (pred[valid][hidden].float() - target[hidden]).square().mean(dim=-1)
        slice_ids = hidden.nonzero(as_tuple=True)[0]
        slice_loss = errors.new_zeros(len(patient_ids)).index_add(0, slice_ids, errors)
        slice_loss = slice_loss / hidden.sum(dim=1)
        patient_loss = errors.new_zeros(images.shape[0]).index_add(0, patient_ids, slice_loss)
        return (patient_loss / valid.sum(dim=1)).mean()

    def forward(self, images, slice_valid_mask, mask_ratio=.75, return_predictions=False):
        valid = self._validate_input(images, slice_valid_mask)
        if not 0 < mask_ratio < 1:
            raise ValueError("MAE loss requires 0 < mask_ratio < 1; use forward_features for inference")
        latent, mask, restore = self._encode(images, valid, mask_ratio)
        losses = latent.new_zeros(images.shape[0], dtype=torch.float32)
        predictions = None
        if return_predictions:
            predictions = latent.new_zeros(
                images.shape[0], self.num_slices, self.num_patches, self.decoder_pred.out_features)
        for start in range(0, self.num_slices, self.decoder_slice_chunk_size):
            batch_ids, slice_ids = valid[:, start:start+self.decoder_slice_chunk_size].nonzero(as_tuple=True)
            if batch_ids.numel() == 0:
                continue
            slice_ids = slice_ids + start
            args = (latent[batch_ids], restore[batch_ids], images[batch_ids, slice_ids],
                    mask[batch_ids], slice_ids)
            if return_predictions:
                pred = self.decoder_pred(self._decode_features(args[0], args[1], slice_ids))
                predictions[batch_ids, slice_ids] = pred.to(predictions.dtype)
                hidden = args[3].bool()
                target = self._target(args[2])
                error = (pred[hidden].float() - target[hidden]).square().mean(dim=-1)
                row_ids = hidden.nonzero(as_tuple=True)[0]
                chunk_loss = error.new_zeros(len(batch_ids)).index_add(0, row_ids, error) / hidden.sum(dim=1)
            elif self.use_checkpoint and self.training and torch.is_grad_enabled():
                chunk_loss = checkpoint(self._decode_loss, *args, use_reentrant=False)
            else:
                chunk_loss = self._decode_loss(*args)
            losses = losses.index_add(0, batch_ids, chunk_loss)
        loss = (losses / valid.sum(dim=1)).mean()
        return loss, predictions, mask

    def load_long_axis_checkpoint(self, path):
        """Initialize matching 2D encoder weights; new slice/decoder weights stay fresh.

        Accepts existing MAE ``model`` or semantic-teacher ``cmr_encoder`` state
        dictionaries. Shape mismatch is reported explicitly, not hidden by
        ``strict=False``. This method does not resume optimizer/training state.
        """
        saved = torch.load(path, map_location='cpu')
        state = saved.get('model', saved.get('cmr_encoder', saved))
        current = self.state_dict()
        compatible, skipped = {}, []
        encoder_prefixes = ('patch_embed.', 'blocks.', 'norm.')
        for name, value in state.items():
            name = name[7:] if name.startswith('module.') else name
            is_encoder = name.startswith(encoder_prefixes) or name in ('cls_token', 'pos_embed')
            if is_encoder and name in current and hasattr(value, 'shape') and value.shape == current[name].shape:
                compatible[name] = value
            else:
                skipped.append(name)
        if 'patch_embed.proj.weight' not in compatible or not any(k.startswith('blocks.') for k in compatible):
            raise ValueError("Checkpoint does not contain a compatible long-axis patch projection and ViT blocks")
        result = self.load_state_dict(compatible, strict=False)
        return {'loaded': sorted(compatible), 'skipped': sorted(skipped), 'missing': list(result.missing_keys)}


def mae_vit_base_patch16(**kwargs):
    """Production model: 768/12 encoder, 256/2 shared decoder, 50 cine channels."""
    return MaskedAutoencoderSA(**kwargs)


def mae_vit_tiny_patch16(**kwargs):
    """Small model for pipeline smoke tests; not the default research backbone."""
    config = dict(embed_dim=64, depth=2, num_heads=4,
                  decoder_embed_dim=32, decoder_depth=1, decoder_num_heads=4)
    config.update(kwargs)
    return MaskedAutoencoderSA(**config)
