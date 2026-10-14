"""Recognizers behind one interface.
  * CRNNRecognizer  -- small CTC model, multilingual via charset file; fast ablations / smoke tests (no downloads).
  * TrOCRRecognizer -- HF TrOCR + LoRA (as in *Reading in the Dark*). English-only tokenizer! LATeD also contains
                       Chinese -> report TrOCR on the English subset, use CRNN/SVTR-style (or PARSeq) for multilingual.
All take crops in [0,1], shape (N,3,H,W)."""
from __future__ import annotations
import torch, torch.nn as nn, torch.nn.functional as F

DEFAULT_CHARSET = "0123456789abcdefghijklmnopqrstuvwxyz"


class Charset:
    def __init__(self, chars: str = DEFAULT_CHARSET):
        self.chars = chars; self.idx = {c: i + 1 for i, c in enumerate(chars)}  # 0 = CTC blank

    @classmethod
    def from_file(cls, path):
        return cls("".join(dict.fromkeys(open(path, encoding="utf-8").read().replace("\n", ""))))

    def encode(self, s):
        return [self.idx[c] for c in s.lower() if c in self.idx]

    def decode(self, ids):
        out, prev = [], 0
        for i in ids:
            if i != prev and i != 0:
                out.append(self.chars[i - 1])
            prev = i
        return "".join(out)


class BaseRecognizer(nn.Module):
    def forward_loss(self, crops, texts): raise NotImplementedError
    def predict(self, crops): raise NotImplementedError
    def encode_features(self, crops): raise NotImplementedError   # (N,T,C) tokens for the CIC loss


class CRNNRecognizer(BaseRecognizer):
    def __init__(self, charset: Charset | None = None, hidden=256, in_hw=(32, 128)):
        super().__init__()
        self.cs = charset or Charset(); self.in_hw = in_hw
        def blk(i, o, pool):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(True), nn.MaxPool2d(pool))
        self.cnn = nn.Sequential(blk(3, 64, (2, 2)), blk(64, 128, (2, 2)), blk(128, 256, (2, 1)), blk(256, 256, (2, 1)))
        self.rnn = nn.LSTM(256, hidden, num_layers=2, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(2 * hidden, len(self.cs.chars) + 1)

    def _seq(self, crops):
        x = F.interpolate(crops, size=self.in_hw, mode="bilinear", align_corners=False)
        f = self.cnn((x - 0.5) / 0.5).mean(2).transpose(1, 2)       # (N,T,256)
        return self.rnn(f)[0]

    def encode_features(self, crops): return self._seq(crops)

    def forward_loss(self, crops, texts):
        logits = self.fc(self._seq(crops)).log_softmax(-1).transpose(0, 1)  # (T,N,C)
        tg = [torch.tensor(self.cs.encode(t) or [1], dtype=torch.long) for t in texts]
        T, N = logits.shape[:2]
        return F.ctc_loss(logits, torch.cat(tg).to(crops.device), torch.full((N,), T, dtype=torch.long),
                          torch.tensor([len(t) for t in tg]), zero_infinity=True)

    @torch.no_grad()
    def predict(self, crops):
        ids = self.fc(self._seq(crops)).argmax(-1).cpu().tolist()
        return [self.cs.decode(i) for i in ids]


class TrOCRRecognizer(BaseRecognizer):
    def __init__(self, name="microsoft/trocr-base-printed", lora_r=16, lora_alpha=32, freeze=False, max_len=32):
        super().__init__()
        from transformers import VisionEncoderDecoderModel, TrOCRProcessor   # lazy: optional dependency
        self.base = VisionEncoderDecoderModel.from_pretrained(name)
        self.tok = TrOCRProcessor.from_pretrained(name).tokenizer
        self.max_len = max_len
        if freeze:
            for p in self.base.parameters(): p.requires_grad = False
        elif lora_r > 0:
            from peft import LoraConfig, get_peft_model
            cfg = LoraConfig(r=lora_r, lora_alpha=lora_alpha, target_modules=["query", "value", "q_proj", "v_proj"], lora_dropout=0.05)
            self.peft = get_peft_model(self.base, cfg)   # injects LoRA in place; self.base then contains LoRA layers

    def _px(self, crops):
        return (F.interpolate(crops, size=(384, 384), mode="bilinear", align_corners=False) - 0.5) / 0.5

    def encode_features(self, crops): return self.base.encoder(pixel_values=self._px(crops)).last_hidden_state

    def forward_loss(self, crops, texts):
        lab = self.tok(list(texts), padding="longest", truncation=True, max_length=self.max_len, return_tensors="pt").input_ids.to(crops.device)
        lab = lab.masked_fill(lab == self.tok.pad_token_id, -100)
        return self.base(pixel_values=self._px(crops), labels=lab).loss

    @torch.no_grad()
    def predict(self, crops):
        ids = self.base.generate(pixel_values=self._px(crops), max_length=self.max_len)
        return self.tok.batch_decode(ids, skip_special_tokens=True)


def build_recognizer(cfg) -> BaseRecognizer:
    if cfg.type == "crnn":
        cs = Charset.from_file(cfg.charset_file) if getattr(cfg, "charset_file", None) else Charset()
        return CRNNRecognizer(cs)
    return TrOCRRecognizer(cfg.name, cfg.lora_r, cfg.lora_alpha, cfg.freeze)
