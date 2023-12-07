import logging
import math
from dataclasses import dataclass, field
from typing import Optional

from omegaconf import II

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from fairseq.modules import EMAModule, EMAModuleConfig
from fairseq.data.data_utils import compute_mask_indices
from fairseq.models import BaseFairseqModel, register_model
from fairseq.models.wav2vec import (
    ConvFeatureExtractionModel,
    Wav2Vec2Config,
    TransformerEncoder,
)
from fairseq.modules import (
    GradMultiply,
    LayerNorm,
)
from fairseq.utils import index_put


logger = logging.getLogger(__name__)


@dataclass
class DinosrAudioConfig(Wav2Vec2Config):

    discrete: bool = field(default=False)
    codebook_size: int = field(default=256)
    normal_init_codebook: bool = field(default=False)
    codebook_init_decay: float = field(default=0.9)
    codebook_end_decay: float = field(default=0.9)
    codebook_end_decay_step: int = field(default=0)
    freeze_teacher_step: int = field(
        default=200001, metadata={"help": "step to freeze teacher"}
    )
    freeze_pre_enc_modules: bool = field(
        default=True, metadata={"help": "when freezing teacher, freeze the CNN extractor as well"}
    )
    loss_beta: float = field(
        default=0, metadata={"help": "beta for smooth l1 loss. 0 means use l2 loss"}
    )
    loss_scale: Optional[float] = field(
        default=None,
        metadata={
            "help": "scale the reconstruction loss by this constant. if None then scales by 1/sqrt(dim)"
        },
    )
    average_top_k_layers: int = field(
        default=8, metadata={"help": "how many layers to average"}
    )

    layer_norm_target_layer: bool = False
    instance_norm_target_layer: bool = False
    instance_norm_targets: bool = False
    layer_norm_targets: bool = False
    batch_norm_target_layer: bool = False
    group_norm_target_layer: bool = False

    ema_decay: float = field(default=0.999, metadata={"help": "initial ema decay rate"})
    ema_end_decay: float = field(
        default=0.9999, metadata={"help": "final ema decay rate"}
    )

    # when to finish annealing ema decay rate
    ema_anneal_end_step: int = II("optimization.max_update")

    ema_transformer_only: bool = field(
        default=True,
        metadata={"help": "whether to momentum update only the transformer"},
    )
    ema_layers_only: bool = field(
        default=True,
        metadata={"help": "whether to momentum update only the transformer layers"},
    )

    max_update: int = II("optimization.max_update")

    min_target_var: float = field(
        default=0.1, metadata={"help": "stop training if target var falls below this"}
    )
    min_pred_var: float = field(
        default=0.01,
        metadata={"help": "stop training if prediction var falls below this"},
    )


def get_annealed_rate(start, end, curr_step, total_steps):
    r = end - start
    pct_remaining = 1 - curr_step / total_steps
    return end - r * pct_remaining


@register_model("dinosr", dataclass=DinosrAudioConfig)
class DinosrModel(BaseFairseqModel):
    def __init__(self, cfg: DinosrAudioConfig):
        super().__init__()
        self.cfg = cfg
        self.discrete = cfg.discrete

        feature_enc_layers = eval(cfg.conv_feature_layers)
        self.extractor_embed = feature_enc_layers[-1][0]

        self.ema = None
        self.embed = cfg.encoder_embed_dim

        self.average_top_k_layers = cfg.average_top_k_layers
        self.loss_beta = cfg.loss_beta
        self.loss_scale = cfg.loss_scale

        self.feature_extractor = ConvFeatureExtractionModel(
            conv_layers=feature_enc_layers,
            dropout=0.0,
            mode=cfg.extractor_mode,
            conv_bias=cfg.conv_bias,
        )

        self.post_extract_proj = nn.Linear(self.extractor_embed, cfg.encoder_embed_dim)

        self.mask_prob = cfg.mask_prob
        self.mask_selection = cfg.mask_selection
        self.mask_other = cfg.mask_other
        self.mask_length = cfg.mask_length
        self.no_mask_overlap = cfg.no_mask_overlap
        self.mask_min_space = cfg.mask_min_space

        self.mask_channel_prob = cfg.mask_channel_prob
        self.mask_channel_before = cfg.mask_channel_before
        self.mask_channel_selection = cfg.mask_channel_selection
        self.mask_channel_other = cfg.mask_channel_other
        self.mask_channel_length = cfg.mask_channel_length
        self.no_mask_channel_overlap = cfg.no_mask_channel_overlap
        self.mask_channel_min_space = cfg.mask_channel_min_space

        self.dropout_input = nn.Dropout(cfg.dropout_input)
        self.dropout_features = nn.Dropout(cfg.dropout_features)

        self.feature_grad_mult = cfg.feature_grad_mult

        self.mask_emb = nn.Parameter(
            torch.FloatTensor(cfg.encoder_embed_dim).uniform_()
        )

        self.encoder = TransformerEncoder(cfg)
        self.layer_norm = LayerNorm(self.extractor_embed)

        self.pre_encoder_copied = False
        if self.discrete:
            assert cfg.instance_norm_target_layer
            assert not (cfg.layer_norm_targets or cfg.instance_norm_targets)
            self.codebook_size = cfg.codebook_size
            self.n_codebooks = cfg.average_top_k_layers
            self.codebook_decay = cfg.codebook_init_decay
            # Prediction heads
            self.heads = torch.nn.ModuleList([
                    nn.Linear(
                        cfg.encoder_embed_dim,
                        cfg.codebook_size,
                    )
                    for i in range(self.n_codebooks)
                ]
            )
            # Codebook: use dictionary to store so codebooks are always in fp32
            if cfg.normal_init_codebook:
                codebooks = torch.normal(0.0, (1 / self.codebook_size**0.5),
                            size=(self.n_codebooks, self.codebook_size, cfg.encoder_embed_dim))
            else:
                codebooks = torch.randn(self.n_codebooks, cfg.encoder_embed_dim, self.codebook_size)
                codebooks = F.instance_norm(codebooks).transpose(1,2)
            self.codebooks = {
                i:codebooks[i] for i in range(self.n_codebooks)
            }
            self.codebook_cnts = {
                i:torch.ones([self.codebook_size]) for i in range(self.n_codebooks)
            }
            self.shared_module_state_dict = None
        else:
            self.final_proj = nn.Linear(self.embed, self.embed)

        self.num_updates = 0

    def make_ema_teacher(self):
        ema_config = EMAModuleConfig(
            ema_decay=self.cfg.ema_decay,
            ema_fp32=True,
        )
        skip_keys = set()
        if self.cfg.ema_layers_only:
            self.cfg.ema_transformer_only = True
            for k, _ in self.encoder.pos_conv.named_parameters():
                skip_keys.add(f"pos_conv.{k}")

        self.ema = EMAModule(
            self.encoder if self.cfg.ema_transformer_only else self,
            ema_config,
            skip_keys=skip_keys,
        )
    
    def move_codebook_to_gpu(self):
        # Move codebook to GPU
        device = next(self.encoder.parameters()).device
        self.codebooks = {
            i:self.codebooks[i].to(device) for i in range(self.n_codebooks)
        }
        self.codebook_cnts = {
            i:self.codebook_cnts[i].to(device) for i in range(self.n_codebooks)
        }
    
    def freeze_shared_modules(self):
        # Hack to avoid updating any of the shared modules (e.g., Weight Decay from optimizer)
        # using WD=0 + torch.no_grad() for following modules will still result in higher loss somehow
        if self.shared_module_state_dict is None:
            self.shared_module_state_dict = {}
            self.shared_module_state_dict['feature_extractor'] = self.feature_extractor.state_dict()
            self.shared_module_state_dict['layer_norm'] = self.layer_norm.state_dict()
            self.shared_module_state_dict['post_extract_proj'] = self.post_extract_proj.state_dict()
        else:
            self.feature_extractor.load_state_dict(self.shared_module_state_dict['feature_extractor'])
            self.layer_norm.load_state_dict(self.shared_module_state_dict['layer_norm'])
            self.post_extract_proj.load_state_dict(self.shared_module_state_dict['post_extract_proj'])

    def copy_shared_modules(self):
        if not self.pre_encoder_copied:
            ema_config = EMAModuleConfig(
                ema_decay=1,
                ema_fp32=True,
            )
            self.cnn_copy = EMAModule(
                self.feature_extractor,
                ema_config,
                skip_keys=set(),
            )
            self.ln_copy = EMAModule(
                self.layer_norm,
                ema_config,
                skip_keys=set(),
            )
            self.proj_copy = EMAModule(
                self.post_extract_proj,
                ema_config,
                skip_keys=set(),
            )
            self.pre_encoder_copied = True
            logger.debug(f"pre-encoder modules copied for teacher model")

    def set_num_updates(self, num_updates):
        super().set_num_updates(num_updates)

        if self.cfg.freeze_teacher_step!=-1 and num_updates>=self.cfg.freeze_teacher_step:
            if self.cfg.freeze_pre_enc_modules:
                self.freeze_shared_modules()
            else:
                self.copy_shared_modules()
            self.cfg.ema_end_decay = 1

        if self.ema is None and (self.discrete or self.final_proj is not None):
            logger.info(f"making ema teacher")
            self.make_ema_teacher()
        elif self.training and self.ema is not None:
            if self.cfg.ema_decay != self.cfg.ema_end_decay:
                if num_updates >= self.cfg.ema_anneal_end_step:
                    decay = self.cfg.ema_end_decay
                else:
                    decay = get_annealed_rate(
                        self.cfg.ema_decay,
                        self.cfg.ema_end_decay,
                        num_updates,
                        self.cfg.ema_anneal_end_step,
                    )
                self.ema.set_decay(decay)
            if self.ema.get_decay() < 1:
                self.ema.step(self.encoder if self.cfg.ema_transformer_only else self)
        
        if self.cfg.codebook_init_decay == self.cfg.codebook_end_decay:
            self.codebook_decay = self.cfg.codebook_init_decay
        else:
            if num_updates >= self.cfg.codebook_end_decay_step:
                self.codebook_decay = self.cfg.codebook_end_decay
            else:
                self.codebook_decay = get_annealed_rate(
                    self.cfg.codebook_init_decay,
                    self.cfg.codebook_end_decay,
                    num_updates,
                    self.cfg.codebook_end_decay_step,
                )

        self.num_updates = num_updates

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        if self.shared_module_state_dict is not None:
            self.freeze_shared_modules()
        
        state = super().state_dict(destination, prefix, keep_vars)

        if self.ema is not None:
            state[prefix + "_ema"] = self.ema.fp32_params
        
        if self.discrete:
            for i in range(self.n_codebooks):
                state[prefix+f'_codebook{i}'] = self.codebooks[i]
                state[prefix+f'_codebook_cnts{i}'] = self.codebook_cnts[i]
        
        if self.pre_encoder_copied:
            state[prefix+'_pre_encoder_cnn'] = self.cnn_copy.fp32_params
            state[prefix+'_pre_encoder_ln'] = self.ln_copy.fp32_params
            state[prefix+'_pre_encoder_proj'] = self.proj_copy.fp32_params

        return state

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        if self.ema is not None:
            k = prefix + "_ema"
            assert k in state_dict
            self.ema.restore(state_dict[k], True)
            del state_dict[k]
        
        if self.discrete:
            for i in range(self.n_codebooks):
                k = prefix+f'_codebook{i}'
                assert k in state_dict
                self.codebooks[i] = state_dict[k].contiguous()
                del state_dict[k]
                k = prefix+f'_codebook_cnts{i}'
                assert k in state_dict
                self.codebook_cnts[i] = state_dict[k].contiguous()
                del state_dict[k]
        
        k = prefix+'_pre_encoder_cnn'
        if self.pre_encoder_copied:
            assert k in state_dict
            self.cnn_copy.restore(state_dict[k],True)
            del state_dict[k]
            k = prefix+'_pre_encoder_ln'
            self.ln_copy.restore(state_dict[k],True)
            del state_dict[k]
            k = prefix+'_pre_encoder_proj'
            self.proj_copy.restore(state_dict[k],True)
            del state_dict[k]
        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    @classmethod
    def build_model(cls, cfg: DinosrAudioConfig, task=None):
        """Build a new model instance."""

        return cls(cfg)

    def apply_mask(
        self,
        x,
        padding_mask,
        mask_indices=None,
        mask_channel_indices=None,
    ):
        B, T, C = x.shape

        if self.mask_channel_prob > 0 and self.mask_channel_before:
            mask_channel_indices = compute_mask_indices(
                (B, C),
                None,
                self.mask_channel_prob,
                self.mask_channel_length,
                self.mask_channel_selection,
                self.mask_channel_other,
                no_overlap=self.no_mask_channel_overlap,
                min_space=self.mask_channel_min_space,
            )
            mask_channel_indices = (
                torch.from_numpy(mask_channel_indices)
                .to(x.device)
                .unsqueeze(1)
                .expand(-1, T, -1)
            )
            x[mask_channel_indices] = 0

        if self.mask_prob > 0:
            if mask_indices is None:
                mask_indices = compute_mask_indices(
                    (B, T),
                    padding_mask,
                    self.mask_prob,
                    self.mask_length,
                    self.mask_selection,
                    self.mask_other,
                    min_masks=1,
                    no_overlap=self.no_mask_overlap,
                    min_space=self.mask_min_space,
                    require_same_masks=self.cfg.require_same_masks,
                    mask_dropout=self.cfg.mask_dropout,
                )
                mask_indices = torch.from_numpy(mask_indices).to(x.device)
            x = index_put(x, mask_indices, self.mask_emb)
        else:
            mask_indices = None

        if self.mask_channel_prob > 0 and not self.mask_channel_before:
            if mask_channel_indices is None:
                mask_channel_indices = compute_mask_indices(
                    (B, C),
                    None,
                    self.mask_channel_prob,
                    self.mask_channel_length,
                    self.mask_channel_selection,
                    self.mask_channel_other,
                    no_overlap=self.no_mask_channel_overlap,
                    min_space=self.mask_channel_min_space,
                )
                mask_channel_indices = (
                    torch.from_numpy(mask_channel_indices)
                    .to(x.device)
                    .unsqueeze(1)
                    .expand(-1, T, -1)
                )
            x = index_put(x, mask_channel_indices, 0)

        return x, mask_indices

    def _get_feat_extract_output_lengths(self, input_lengths: torch.LongTensor):
        """
        Computes the output length of the convolutional layers
        """

        def _conv_out_length(input_length, kernel_size, stride):
            return torch.floor((input_length - kernel_size) / stride + 1)

        conv_cfg_list = eval(self.cfg.conv_feature_layers)

        for i in range(len(conv_cfg_list)):
            input_lengths = _conv_out_length(
                input_lengths, conv_cfg_list[i][1], conv_cfg_list[i][2]
            )

        return input_lengths.to(torch.long)

    def forward(
        self,
        source,
        padding_mask=None,
        mask=True,
        features_only=False,
        layer=None,
        mask_indices=None,
        mask_channel_indices=None,
        padding_count=None,
    ):
        features = source

        if self.feature_grad_mult > 0:
            features = self.feature_extractor(features)
            if self.feature_grad_mult != 1.0:
                features = GradMultiply.apply(features, self.feature_grad_mult)
        else:
            with torch.no_grad():
                features = self.feature_extractor(features)

        features = features.transpose(1, 2)

        features = self.layer_norm(features)

        orig_padding_mask = padding_mask

        if padding_mask is not None and padding_mask.any():
            input_lengths = (1 - padding_mask.long()).sum(-1)
            # apply conv formula to get real output_lengths
            output_lengths = self._get_feat_extract_output_lengths(input_lengths)

            padding_mask = torch.zeros(
                features.shape[:2], dtype=features.dtype, device=features.device
            )

            # these two operations makes sure that all values
            # before the output lengths indices are attended to
            padding_mask[
                (
                    torch.arange(padding_mask.shape[0], device=padding_mask.device),
                    output_lengths - 1,
                )
            ] = 1
            padding_mask = (1 - padding_mask.flip([-1]).cumsum(-1).flip([-1])).bool()
        else:
            padding_mask = None

        if self.post_extract_proj is not None:
            features = self.post_extract_proj(features)

        pre_encoder_features = None
        if self.pre_encoder_copied:
            # Copied pre-encoder modules used for teacher model
            self.cnn_copy.model.eval()
            self.ln_copy.model.eval()
            self.proj_copy.model.eval()
            with torch.no_grad():
                pre_encoder_features = self.cnn_copy.model(source)
                pre_encoder_features = pre_encoder_features.transpose(1, 2)
                pre_encoder_features = self.ln_copy.model(pre_encoder_features)
                pre_encoder_features = self.proj_copy.model(pre_encoder_features) 
        elif self.cfg.ema_transformer_only:
            pre_encoder_features = features.clone()

        features = self.dropout_input(features)

        if mask:
            x, mask_indices = self.apply_mask(
                features,
                padding_mask,
                mask_indices=mask_indices,
                mask_channel_indices=mask_channel_indices,
            )
        else:
            x = features
            mask_indices = None

        x, layer_results = self.encoder(
            x,
            padding_mask=padding_mask,
            layer=layer,
        )

        if features_only:
            return {
                "x": x,
                "padding_mask": padding_mask,
                "layer_results": layer_results,
            }

        result = {
            "losses": {},
        }

        with torch.no_grad():
            self.ema.model.eval()

            if self.cfg.ema_transformer_only:
                y, layer_results = self.ema.model.extract_features(
                    pre_encoder_features,
                    padding_mask=padding_mask,
                    min_layer=self.cfg.encoder_layers - self.average_top_k_layers,
                )
                y = {
                    "x": y,
                    "padding_mask": padding_mask,
                    "layer_results": layer_results,
                }
            else:
                y = self.ema.model.extract_features(
                    source=source,
                    padding_mask=orig_padding_mask,
                    mask=False,
                )

            target_layer_results = [l[2] for l in y["layer_results"]]

            permuted = False
            if self.cfg.instance_norm_target_layer or self.cfg.batch_norm_target_layer:
                target_layer_results = [
                    tl.permute(1, 2, 0) for tl in target_layer_results  # TBC -> BCT
                ]
                permuted = True

            if self.cfg.batch_norm_target_layer:
                target_layer_results = [
                    F.batch_norm(
                        tl.float(), running_mean=None, running_var=None, training=True
                    )
                    for tl in target_layer_results
                ]

            if self.cfg.instance_norm_target_layer:
                target_layer_results = [
                    F.instance_norm(tl.float()) for tl in target_layer_results
                ]

            if permuted:
                target_layer_results = [
                    tl.transpose(1, 2) for tl in target_layer_results  # BCT -> BTC
                ]

            if self.cfg.group_norm_target_layer:
                target_layer_results = [
                    F.layer_norm(tl.float(), tl.shape[-2:])
                    for tl in target_layer_results
                ]

            if self.cfg.layer_norm_target_layer:
                target_layer_results = [
                    F.layer_norm(tl.float(), tl.shape[-1:])
                    for tl in target_layer_results
                ]
            
            if self.discrete:
                target_layer_results = [
                    tl[mask_indices] for tl in target_layer_results
                ]
            else:
                y = sum(target_layer_results) / len(target_layer_results)

                if self.cfg.layer_norm_targets:
                    y = F.layer_norm(y.float(), y.shape[-1:])

                if self.cfg.instance_norm_targets:
                    y = F.instance_norm(y.float().transpose(1, 2)).transpose(1, 2)

                if not permuted:
                    y = y.transpose(0, 1)

                y = y[mask_indices]

        x = x[mask_indices]

        if self.discrete:
            if self.codebooks[0].device != x.device:
                self.move_codebook_to_gpu()
            
            losses = 0
            target_ppl, pred_ppl = 0,0

            for i,target in enumerate(target_layer_results):
                # Quantize target
                with torch.no_grad():
                    codebook = self.codebooks[i].float() / self.codebook_cnts[i].unsqueeze(1)
                    neg_l2_dist = - (torch.sum(target**2, dim=1, keepdim=True) 
                                    + torch.sum(codebook**2, dim=1)
                                    - 2 * torch.matmul(target, codebook.t()))
                    onehot_target = torch.zeros_like(neg_l2_dist)
                    onehot_target[range(len(neg_l2_dist)),neg_l2_dist.argmax(-1)] = 1.0                    
                # Compute loss
                pred = self.heads[i](x).float()
                pred = F.log_softmax(pred,dim=-1)
                loss = torch.sum(-onehot_target*pred,dim=-1)
                losses = losses + loss
                
                # Compute stats & update codebook
                with torch.no_grad():
                    # Stats
                    target_ppl += self.compute_ppl(onehot_target,input_onehot=True)
                    pred_ppl += self.compute_ppl(pred.float(),input_onehot=False)
                    if self.training and self.codebook_decay<1:
                        # Update codebook
                        # Note: this is done in a per-forward style,
                        #       might wanna consider doing this in set_num_updates
                        count = onehot_target.sum(0)
                        memory = torch.matmul(onehot_target.t(), target)
                        if dist.is_initialized():
                            dist.all_reduce(memory) # Sum of embeddings
                            dist.all_reduce(count) # Total counts
                        alpha = torch.ones_like(count).unsqueeze(1)
                        alpha[count!=0] = self.codebook_decay
                        self.codebook_cnts[i]  = alpha.squeeze(1) * self.codebook_cnts[i] + (1-alpha).squeeze(1) * count
                        self.codebooks[i] = alpha * self.codebooks[i] + (1-alpha) * memory

            result["losses"]["cross_entropy"] = (losses/self.n_codebooks).sum()

        else:
            x = self.final_proj(x)

            sz = x.size(-1)

            if self.loss_beta == 0:
                loss = F.mse_loss(x.float(), y.float(), reduction="none").sum(dim=-1)
            else:
                loss = F.smooth_l1_loss(
                    x.float(), y.float(), reduction="none", beta=self.loss_beta
                ).sum(dim=-1)

            if self.loss_scale is not None:
                scale = self.loss_scale
            else:
                scale = 1 / math.sqrt(sz)

            result["losses"]["regression"] = loss.sum() * scale

        if "sample_size" not in result:
            result["sample_size"] = loss.numel()

        with torch.no_grad():
            if self.discrete:
                result["target_ppl"] = target_ppl/self.n_codebooks
                result["pred_ppl"] = pred_ppl/self.n_codebooks
                result["codebook_decay"] = self.codebook_decay
            else:
                result["target_var"] = self.compute_var(y)
                result["pred_var"] = self.compute_var(x.float())

                if self.num_updates > 5000 and result["target_var"] < self.cfg.min_target_var:
                    logger.error(
                        f"target var is {result['target_var'].item()} < {self.cfg.min_target_var}, exiting"
                    )
                    raise Exception(
                        f"target var is {result['target_var'].item()} < {self.cfg.min_target_var}, exiting"
                    )
                if self.num_updates > 5000 and result["pred_var"] < self.cfg.min_pred_var:
                    logger.error(
                        f"pred var is {result['pred_var'].item()} < {self.cfg.min_pred_var}, exiting"
                    )
                    raise Exception(
                        f"pred var is {result['pred_var'].item()} < {self.cfg.min_pred_var}, exiting"
                    )

        if self.ema is not None:
            result["ema_decay"] = self.ema.get_decay() * 1000

        return result

    @staticmethod
    def compute_ppl(y, input_onehot=False, tokenwise=False):
        # We track the avg. of 1-hot (argmax)
        if not input_onehot:
            y = y.softmax(dim=-1)
        if tokenwise:
            y = 2**(- y * (y+1e-8).log2()).sum(-1)
        y = y.mean(0)
        if dist.is_initialized():
            dist.all_reduce(y)
            y = y /  dist.get_world_size()
        if not tokenwise:
            y = 2**(- y * (y+1e-8).log2()).sum()
        return y
    
    @staticmethod
    def compute_var(y):
        y = y.view(-1, y.size(-1))
        if dist.is_initialized():
            zc = torch.tensor(y.size(0)).cuda()
            zs = y.sum(dim=0)
            zss = (y ** 2).sum(dim=0)

            dist.all_reduce(zc)
            dist.all_reduce(zs)
            dist.all_reduce(zss)

            var = zss / (zc - 1) - (zs ** 2) / (zc * (zc - 1))
            return torch.sqrt(var + 1e-6).mean()
        else:
            return torch.sqrt(y.var(dim=0) + 1e-6).mean()

    def extract_features(
        self, source, padding_mask, mask=False, layer=None
    ):
        res = self.forward(
            source,
            padding_mask,
            mask=mask,
            features_only=True,
            layer=layer,
        )
        return res

    def remove_pretraining_modules(self, last_layer=None):
        self.heads = None
        self.final_proj = None
        self.ema = None
        if last_layer is not None:
            self.encoder.layers = nn.ModuleList(
                l for i, l in enumerate(self.encoder.layers) if i <= last_layer
            )
