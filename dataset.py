import os
import json
from collections import Counter

import torch
from torch.utils.data import Dataset
from datasets import load_dataset
import spacy
from tqdm import tqdm


SPECIAL_TOKENS = ["<unk>", "<pad>", "<sos>", "<eos>"]


class Multi30kDataset(Dataset):
    def __init__(self, split='train'):
        # basic setup for split + tokenizers
        self.split = split
        self.ds = load_dataset("bentrevett/multi30k", split=split)
        self.spacy_de = self._load_spacy("de", "de_core_news_sm")
        self.spacy_en = self._load_spacy("en", "en_core_web_sm")

        self.src_vocab = None
        self.tgt_vocab = None
        self.src_stoi = {}
        self.tgt_stoi = {}
        self.src_itos = []
        self.tgt_itos = []
        self.processed_data = []

    @staticmethod
    def _load_spacy(lang_code: str, model_name: str):
        try:
            return spacy.load(model_name)
        except Exception:
            return spacy.blank(lang_code)

    def _extract_text_pair(self, sample) -> tuple[str, str]:
        if "de" in sample and "en" in sample:
            return sample["de"], sample["en"]
        translation = sample.get("translation", {})
        return translation["de"], translation["en"]

    def _tokenize(self, text: str, lang: str) -> list[str]:
        nlp = self.spacy_de if lang == "de" else self.spacy_en
        return [tok.text.lower() for tok in nlp(text)]

    def build_vocab(self):
        # build small vocab maps for src/tgt
        src_counter = Counter()
        tgt_counter = Counter()

        for sample in tqdm(self.ds, desc=f"Building vocab ({self.split})", leave=True):
            de_text, en_text = self._extract_text_pair(sample)
            src_counter.update(self._tokenize(de_text, "de"))
            tgt_counter.update(self._tokenize(en_text, "en"))

        self.src_itos = list(SPECIAL_TOKENS)
        self.tgt_itos = list(SPECIAL_TOKENS)

        self.src_itos.extend([tok for tok, _ in src_counter.items() if tok not in set(SPECIAL_TOKENS)])
        self.tgt_itos.extend([tok for tok, _ in tgt_counter.items() if tok not in set(SPECIAL_TOKENS)])

        self.src_stoi = {tok: idx for idx, tok in enumerate(self.src_itos)}
        self.tgt_stoi = {tok: idx for idx, tok in enumerate(self.tgt_itos)}

        self.src_vocab = {"stoi": self.src_stoi, "itos": self.src_itos}
        self.tgt_vocab = {"stoi": self.tgt_stoi, "itos": self.tgt_itos}

        return self.src_vocab, self.tgt_vocab

    def process_data(self):
        # tokenize + convert to ids
        if not self.src_stoi or not self.tgt_stoi:
            raise RuntimeError("Build or load vocab before calling process_data().")

        unk_idx_src = self.src_stoi["<unk>"]
        unk_idx_tgt = self.tgt_stoi["<unk>"]
        sos_idx_src = self.src_stoi["<sos>"]
        eos_idx_src = self.src_stoi["<eos>"]
        sos_idx_tgt = self.tgt_stoi["<sos>"]
        eos_idx_tgt = self.tgt_stoi["<eos>"]

        self.processed_data = []
        for sample in tqdm(self.ds, desc=f"Processing split ({self.split})", leave=True):
            de_text, en_text = self._extract_text_pair(sample)

            src_tokens = self._tokenize(de_text, "de")
            tgt_tokens = self._tokenize(en_text, "en")

            src_ids = [sos_idx_src] + [self.src_stoi.get(tok, unk_idx_src) for tok in src_tokens] + [eos_idx_src]
            tgt_ids = [sos_idx_tgt] + [self.tgt_stoi.get(tok, unk_idx_tgt) for tok in tgt_tokens] + [eos_idx_tgt]

            self.processed_data.append(
                (
                    torch.tensor(src_ids, dtype=torch.long),
                    torch.tensor(tgt_ids, dtype=torch.long),
                )
            )

        return self.processed_data

    def save_vocab(self, save_dir: str) -> None:
        os.makedirs(save_dir, exist_ok=True)
        torch.save(self.src_vocab, os.path.join(save_dir, "src_vocab.pt"))
        torch.save(self.tgt_vocab, os.path.join(save_dir, "tgt_vocab.pt"))

        assets = {
            "pad_idx": self.src_stoi.get("<pad>", 1),
            "sos_idx": self.src_stoi.get("<sos>", 2),
            "eos_idx": self.src_stoi.get("<eos>", 3),
            "src_lang": "de",
            "tgt_lang": "en",
            "max_infer_len": 100,
        }
        with open(os.path.join(save_dir, "inference_assets.json"), "w", encoding="utf-8") as f:
            json.dump(assets, f)

    def load_vocab(self, load_dir: str) -> None:
        self.src_vocab = torch.load(os.path.join(load_dir, "src_vocab.pt"), map_location="cpu")
        self.tgt_vocab = torch.load(os.path.join(load_dir, "tgt_vocab.pt"), map_location="cpu")
        self.src_stoi = dict(self.src_vocab["stoi"])
        self.tgt_stoi = dict(self.tgt_vocab["stoi"])
        self.src_itos = list(self.src_vocab["itos"])
        self.tgt_itos = list(self.tgt_vocab["itos"])

    def __len__(self) -> int:
        return len(self.processed_data)

    def __getitem__(self, idx):
        return self.processed_data[idx]


def collate_batch(batch, pad_idx: int = 1):
    src_batch, tgt_batch = zip(*batch)
    src_padded = torch.nn.utils.rnn.pad_sequence(src_batch, batch_first=True, padding_value=pad_idx)
    tgt_padded = torch.nn.utils.rnn.pad_sequence(tgt_batch, batch_first=True, padding_value=pad_idx)
    return src_padded, tgt_padded