"""
bert_encoder.py
================
BERT/DistilBERT wrapper for text encoding (tags, captions, lyrics).

Used standalone for Task 1 (BERT tag classifier) and as the text branch
feeding into fusion_model.py (Task 3) and contrastive.py (Task 4).
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel


class BertTextEncoder(nn.Module):
    """
    Wraps a HuggingFace BERT-family model. Returns both:
      - t:      the pooled [CLS] vector, shape (batch, hidden_dim)
      - tokens: the full sequence of token embeddings, shape (batch, seq_len, hidden_dim)
                (needed for cross-attention fusion in Task 3)
    """

    def __init__(self, model_name: str = "distilbert-base-uncased", finetune: bool = False, max_length: int = 128):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)
        self.max_length = max_length
        self.hidden_dim = self.bert.config.hidden_size

        if not finetune:
            for param in self.bert.parameters():
                param.requires_grad = False

    def tokenize(self, texts):
        """texts: List[str] -> dict of input_ids / attention_mask tensors."""
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        tokens = out.last_hidden_state              # (batch, seq_len, hidden_dim)

        # DistilBERT has no pooler_output; use the [CLS] token (position 0) for both cases
        t = tokens[:, 0, :]                          # (batch, hidden_dim)
        return t, tokens

    def encode_texts(self, texts, device="cpu"):
        """Convenience method: raw strings in, (t, tokens, attention_mask) out."""
        enc = self.tokenize(texts)
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        t, tokens = self.forward(input_ids, attention_mask)
        return t, tokens, attention_mask


class BertTagClassifier(nn.Module):
    """
    Task 1 (Easy): BERT baseline for multi-label tag / mood / genre classification.

    yhat_k = sigmoid(w_k^T * t + b_k)
    """

    def __init__(self, model_name: str = "distilbert-base-uncased", num_labels: int = 50,
                 finetune: bool = False, max_length: int = 128, dropout: float = 0.1):
        super().__init__()
        self.encoder = BertTextEncoder(model_name=model_name, finetune=finetune, max_length=max_length)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.encoder.hidden_dim, num_labels)

    def forward(self, input_ids, attention_mask):
        t, _ = self.encoder(input_ids, attention_mask)
        t = self.dropout(t)
        logits = self.classifier(t)   # raw logits; apply sigmoid + BCEWithLogitsLoss outside
        return logits


if __name__ == "__main__":
    model = BertTagClassifier(num_labels=50, finetune=False)
    texts = ["melancholic piano ballad with soft strings", "upbeat electronic dance track"]
    enc = model.encoder.tokenize(texts)
    logits = model(enc["input_ids"], enc["attention_mask"])
    print(f"logits shape: {logits.shape}")  # (2, 50)
