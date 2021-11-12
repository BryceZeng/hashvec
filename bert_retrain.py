from torch import nn
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers.utils.dummy_pt_objects import BertModel
from transformers import BertTokenizer
from tokenizers import BertWordPieceTokenizer
import bert_modeling as modeling
# from addition_fn import input_fn_builder, model_fn_builder
import nltk
import os
import sys
import json
import nltk
import random
import logging
import tensorflow as tf
import pandas as pd
from tensorflow.keras.utils import Progbar
import numpy as np
import torch

# sys.path.append("bert")
regex_tokenizer = nltk.RegexpTokenizer("\w+")
train = pd.read_excel('1103-1109.xlsx')


def normalize_text(text):
    # lowercase text
    text = str(text).lower()
    # remove non-UTF
    text = text.encode("utf-8", "ignore").decode()
    # remove punktuation symbols
    text = " ".join(regex_tokenizer.tokenize(text))
    return text


train.to_csv('train_1.txt', columns=[
             "detail_list", ], header=False, index=False, sep="\t", encoding='utf-8')


def count_lines(filename):
    count = 0
    with open(filename) as fi:
        for line in fi:
            count += 1
    return count


total_lines = count_lines('train_1.txt')
bar = Progbar(total_lines)

with open('train_1.txt', encoding="utf-8") as fi:
    with open('train_1.txt', "w", encoding="utf-8") as fo:
        for l in fi:
            fo.write(normalize_text(l)+"\n")
            bar.add(1)

# Build Vocab

# initialize
tokenizer = BertWordPieceTokenizer(
    clean_text=True,
    handle_chinese_chars=False,
    strip_accents=False,
    lowercase=False
)
# and train
tokenizer.train(files='train_1.txt', vocab_size=30_000, min_frequency=2,
                limit_alphabet=1000, wordpieces_prefix='##',
                special_tokens=[
                    '[PAD', '[UNK]', '[CLS]', '[SEP]', '[MASK]'])

# os.mkdir('./bert-it')
tokenizer.save_model('./bert-it', 'bert-it')

tokenizer = BertTokenizer.from_pretrained('./bert-it/bert-it-vocab.txt')

tokenizer('jalan jakata')


tokenized = train['detail_list'].apply(
    (lambda x: tokenizer.encode(x, add_special_tokens=True)))

max_len = 0
for i in tokenized.values:
    if len(i) > max_len:
        max_len = len(i)

padded = np.array([i + [0]*(max_len-len(i)) for i in tokenized.values])
np.array(padded).shape

attention_mask = np.where(padded != 0, 1, 0)
attention_mask.shape

input_ids = torch.tensor(padded)
attention_mask = torch.tensor(attention_mask)

d_model = 512


class Embedding(nn.Module):
    def __init__(self):
        super(Embedding, self).__init__()
        self.tok_embed = nn.Embedding(30_000, 512)  # token embedding
        self.pos_embed = nn.Embedding(5, 512)  # position embedding
        # segment(token type) embedding
        self.seg_embed = nn.Embedding(3, 512)
        self.norm = nn.LayerNorm(512)

    def forward(self, x, seg):
        seq_len = x.size(1)
        pos = torch.arange(seq_len, dtype=torch.long)
        # (seq_len,) -> (batch_size, seq_len)
        pos = pos.unsqueeze(0).expand_as(x)
        embedding = self.tok_embed(
            x) + self.pos_embed(pos) + self.seg_embed(seg)
        return self.norm(embedding)


class PoswiseFeedForwardNet(nn.Module):

    def __init__(self, d_model, d_ff):
        super(PoswiseFeedForwardNet, self).__init__()
        self.l1 = nn.Linear(d_model, d_ff)
        self.l2 = nn.Linear(d_ff, d_model)

        self.relu = nn.GELU()
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, inputs):
        residual = inputs
        output = self.l1(inputs)
        output = self.relu(output)
        output = self.l2(output)
        return self.layer_norm(output + residual)


class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = nn.MultiheadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, enc_inputs, enc_self_attn_mask):
        enc_outputs, attn = self.enc_self_attn(
            enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask)  # enc_inputs to same Q,K,V
        # enc_outputs: [batch_size x len_q x d_model]
        enc_outputs = self.pos_ffn(enc_outputs)
        return enc_outputs, attn


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, d_k, attn_mask):
        # scores : [batch_size x n_heads x len_q(=len_k) x len_k(=len_q)]
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)
        # Fills elements of self tensor with value where mask is one.
        scores.masked_fill_(attn_mask, -1e9)
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return scores, context, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v, n_heads):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads)
        self.W_K = nn.Linear(d_model, d_k * n_heads)
        self.W_V = nn.Linear(d_model, d_v * n_heads)

    def forward(self, Q, K, V, attn_mask, n_heads, d_k, d_v):
        # q: [batch_size x len_q x d_model], k: [batch_size x len_k x d_model], v: [batch_size x len_k x d_model]
        residual, batch_size = Q, Q.size(0)
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        # q_s: [batch_size x n_heads x len_q x d_k]
        q_s = self.W_Q(Q).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
        # k_s: [batch_size x n_heads x len_k x d_k]
        k_s = self.W_K(K).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
        # v_s: [batch_size x n_heads x len_k x d_v]
        v_s = self.W_V(V).view(batch_size, -1, n_heads, d_v).transpose(1, 2)

        # attn_mask : [batch_size x n_heads x len_q x len_k]
        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)

        # context: [batch_size x n_heads x len_q x d_v], attn: [batch_size x n_heads x len_q(=len_k) x len_k(=len_q)]
        context, attn = ScaledDotProductAttention()(q_s, k_s, v_s, attn_mask)
        context = context.transpose(1, 2).contiguous().view(
            batch_size, -1, n_heads * d_v)  # context: [batch_size x len_q x n_heads * d_v]
        output = nn.Linear(n_heads * d_v, d_model)(context)
        # output: [batch_size x len_q x d_model]
        return nn.LayerNorm(d_model)(output + residual), attn


class BERT(nn.Module):
    def __init__(self, d_model, n_layers):
        super(BERT, self).__init__()
        self.embedding = Embedding()
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])
        self.fc = nn.Linear(d_model, d_model)
        self.activ1 = nn.Tanh()
        self.linear = nn.Linear(d_model, d_model)
        self.activ2 = nn.GELU()
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, 2)
        # decoder is shared with embedding layer
        embed_weight = self.embedding.tok_embed.weight
        n_vocab, n_dim = embed_weight.size()
        self.decoder = nn.Linear(n_dim, n_vocab, bias=False)
        self.decoder.weight = embed_weight
        self.decoder_bias = nn.Parameter(torch.zeros(n_vocab))

    def forward(self, input_ids, segment_ids, masked_pos):
        output = self.embedding(input_ids, segment_ids)
        enc_self_attn_mask = get_attn_pad_mask(input_ids, input_ids)
        for layer in self.layers:
            output, enc_self_attn = layer(output, enc_self_attn_mask)
        # output : [batch_size, len, d_model], attn : [batch_size, n_heads, d_mode, d_model]
        # it will be decided by first token(CLS)
        h_pooled = self.activ1(self.fc(output[:, 0]))  # [batch_size, d_model]
        logits_clsf = self.classifier(h_pooled)  # [batch_size, 2]

        # [batch_size, max_pred, d_model]
        masked_pos = masked_pos[:, :, None].expand(-1, -1, output.size(-1))

        # get masked position from final output of transformer.
        # masking position [batch_size, max_pred, d_model]
        h_masked = torch.gather(output, 1, masked_pos)
        h_masked = self.norm(self.activ2(self.linear(h_masked)))
        # [batch_size, max_pred, n_vocab]
        logits_lm = self.decoder(h_masked) + self.decoder_bias

        return logits_lm, logits_clsf


with torch.no_grad():
    last_hidden_states = model(input_ids, attention_mask=attention_mask)

lr_clf = LogisticRegression()
lr_clf.fit(train_features, train_labels)
