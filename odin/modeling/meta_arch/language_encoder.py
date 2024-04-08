import torch
import torch.nn as nn

from transformers import RobertaModel, RobertaTokenizerFast

import ipdb
st = ipdb.set_trace


class LanguageEncoder(nn.Module):
    def __init__(self, cfg, d_model):
        super().__init__()
        self.cfg = cfg
        self.device = torch.device(cfg.MODEL.DEVICE)

        t_type = "roberta-base"
        self.tokenizer = RobertaTokenizerFast.from_pretrained(t_type)
        self.text_encoder = RobertaModel.from_pretrained(t_type)

        if cfg.MODEL.LANG_FREEZE_BACKBONE:
            for param in self.text_encoder.parameters():
                param.requires_grad = False

        self.text_projector = nn.Sequential(
            nn.Linear(self.text_encoder.config.hidden_size, d_model),
        )

    def forward(self, text):
        tokenized = self.tokenizer.batch_encode_plus(
            text, padding="max_length", return_tensors="pt", 
            max_length=self.cfg.MODEL.MAX_SEQ_LEN,
            truncation=True
        ).to(self.device)
        
        encoded_text = self.text_encoder(**tokenized)
        text_attention_mask = tokenized.attention_mask.ne(1).bool()
        text_feats = encoded_text.last_hidden_state

        text_feats = self.text_projector(text_feats)
        
        return text_feats, text_attention_mask


