"""V5 reading components, initialized independently from public pretraining."""
from __future__ import annotations
from collections.abc import Mapping
from pathlib import Path
import torch
from torch import nn
from seeread.models.recognizer import TrOCRRecognizer


def mapping(value):
    return dict(value) if isinstance(value, Mapping) else vars(value).copy()


def parameter_audit(module):
    named = list(module.named_parameters())
    return dict(total_parameters=sum(p.numel() for _, p in named),
                trainable_parameters=sum(p.numel() for _, p in named if p.requires_grad),
                trainable_names=[name for name, p in named if p.requires_grad])


class ReadingTrOCR(TrOCRRecognizer):
    """Reuse TrOCR preprocessing/decoding, with explicit adapter topology.

    Construct the topology first, then set trainability. Unlike the legacy
    wrapper, the PEFT wrapper is not registered a second time in state_dict.
    """
    def __init__(self, cfg):
        nn.Module.__init__(self)
        cfg = mapping(cfg)
        adapter, trainability = cfg.get('adapter', 'none'), cfg.get('trainability', 'frozen')
        if adapter not in ('none', 'lora'):
            raise ValueError(f'Unknown recognizer adapter: {adapter!r}')
        if trainability not in ('frozen', 'lora', 'full'):
            raise ValueError(f'Unknown recognizer trainability: {trainability!r}')
        if trainability == 'lora' and adapter != 'lora':
            raise ValueError('trainability=lora requires adapter=lora')
        self.adapter, self.trainability = adapter, trainability
        name = cfg.get('name', 'microsoft/trocr-base-printed')
        resolved_name = name
        if not Path(name).is_dir() and (cfg.get('revision') or cfg.get('local_files_only', False)):
            from huggingface_hub import snapshot_download
            resolved_name = snapshot_download(repo_id=name, revision=cfg.get('revision', 'main'),
                                              local_files_only=bool(cfg.get('local_files_only', True)))
        donor = TrOCRRecognizer(name=resolved_name, lora_r=0, freeze=False,
                               max_len=int(cfg.get('max_len', 32)))
        self.base, self.tok, self.max_len = donor.base, donor.tok, donor.max_len
        # The legacy donor overwrites the training start with CLS (0), while
        # the public TrOCR generation config starts with EOS (2). Teacher
        # forcing must use the same prefix as the untouched public decoder.
        generation_start = getattr(self.base.generation_config, 'decoder_start_token_id', None)
        if generation_start is None:
            generation_start = getattr(self.base.generation_config, 'bos_token_id', None)
        if generation_start is None:
            raise ValueError('Public generation config has no decoder start or BOS token')
        self.base.config.decoder_start_token_id = generation_start
        self.pretrained_name = name
        commit = getattr(self.base.config, '_commit_hash', None)
        if commit is None and Path(resolved_name).parent.name == 'snapshots':
            commit = Path(resolved_name).name
        self.provenance = dict(public_pretrained=name, requested_revision=cfg.get('revision'),
                               resolved_source=str(resolved_name), actual_commit_hash=commit,
                               model_config=self.base.config.to_dict(),
                               generation_config=self.base.generation_config.to_dict(),
                               decoder_start_token_ids=dict(training=self.base.config.decoder_start_token_id,
                                                            generation=generation_start,
                                                            policy='training follows unchanged public generation prefix'),
                               explicit_generation_overrides=dict(max_length=self.max_len),
                               initialization='public pretraining; no legacy SeeRead checkpoint')
        if adapter == 'lora':
            from peft import LoraConfig, get_peft_model
            rank = int(cfg.get('lora_r', 16))
            if rank <= 0:
                raise ValueError('LoRA rank must be positive')
            target = cfg.get('lora_targets', ['query', 'value', 'q_proj', 'v_proj'])
            if not isinstance(target, list) or not target or not all(isinstance(t, str) and t for t in target):
                raise ValueError('lora_targets must be a nonempty list of module names')
            # get_peft_model injects layers in place into self.base.
            get_peft_model(self.base, LoraConfig(r=rank, lora_alpha=int(cfg.get('lora_alpha', 32)),
                                               target_modules=target, lora_dropout=float(cfg.get('lora_dropout', .05))))
        self.requires_grad_(trainability == 'full')
        if trainability == 'lora':
            for param_name, parameter in self.named_parameters():
                parameter.requires_grad_('lora_' in param_name)
        audit = parameter_audit(self)
        if trainability != 'frozen' and not audit['trainable_parameters']:
            raise ValueError('Requested trainability produced no trainable parameters')
        if trainability == 'lora' and any('lora_' not in n for n in audit['trainable_names']):
            raise ValueError('Non-LoRA parameter unexpectedly trainable')
        self.train(False)

    def train(self, mode=True):
        return super().train(False if self.trainability == 'frozen' else mode)

    def forward_loss(self, crops, texts):
        labels = self.tok(list(texts), padding='longest', truncation=False, return_tensors='pt').input_ids
        if labels.shape[1] > self.max_len:
            raise ValueError(f'Transcription exceeds max_len={self.max_len}; silent label truncation is disabled')
        labels = labels.to(crops.device).masked_fill(labels.to(crops.device) == self.tok.pad_token_id, -100)
        return self.base(pixel_values=self._px(crops), labels=labels).loss


def build_recognizer(cfg):
    cfg = mapping(cfg)
    kind = cfg.get('type', 'trocr')
    if kind != 'trocr':
        raise ValueError(f'Unsupported V5 recognizer: {kind!r}; available: trocr')
    return ReadingTrOCR(cfg)


class ReadingPipeline(nn.Module):
    def __init__(self, enhancer, recognizer):
        super().__init__()
        self.enhancer, self.recognizer = enhancer, recognizer

    def forward_loss(self, crops, texts):
        return self.recognizer.forward_loss(self.enhancer(crops), texts)

    @torch.no_grad()
    def predict(self, crops):
        return self.recognizer.predict(self.enhancer(crops))


def build_pipeline(cfg):
    from seeread.v5.enhancers import build_enhancer
    cfg = mapping(cfg)
    # A loaded enhancer's temporary random Conv initializers must not change
    # the recognizer's LoRA initialization in the paired identity/joint arms.
    # Both components are constructed on CPU before the caller moves them.
    with torch.random.fork_rng(devices=[]):
        enhancer = build_enhancer(cfg.get('enhancer', {'type': 'identity'}))
    recognizer = build_recognizer(cfg['recognizer'])
    objective_cfg = cfg.get('objective', {'type': 'ocr_only'})
    objective_cfg = mapping(objective_cfg)
    kind = objective_cfg.pop('type', 'ocr_only')
    if kind == 'ocr_only':
        return ReadingPipeline(enhancer, recognizer)
    if kind == 'reading_joint':
        from seeread.v5.reading_objective import ReadingJointObjective, PaperReadingPipeline
        objective = ReadingJointObjective(source_root=mapping(cfg['enhancer'])['source_root'], **objective_cfg)
        return PaperReadingPipeline(enhancer, recognizer, objective)
    raise ValueError(f'Unknown Reading objective: {kind}')
