#import relevant libraries
import os
import sys
import copy
import random
import logging
import argparse
import torch
import torch.nn as nn
import numpy as np
from sklearn import metrics
from time import strftime, localtime
from torch.utils.data import DataLoader
from transformers import BertModel, AdamW
import math
import torch.nn.functional as F
from torch.autograd import Variable
from data_utils import SentenceDataset, build_tokenizer, build_embedding_matrix, Tokenizer4BertGCN, ABSAGCNData
from prepare_vocab import VocabHelp

def attention(q, k, mask=None, dropout=None):
    d_k = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    #masking
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    #dropout
    attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        attn = dropout(attn)
    return attn

class MultiHeadAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = nn.ModuleList([copy.deepcopy(nn.Linear(d_model, d_model)) for _ in range(2)])
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, query, key, mask=None):
        mask = mask[:, :, :query.size(1)]
        if mask is not None:
            mask = mask.unsqueeze(1)   
        nbatches = query.size(0)
        q, k = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2) for l, x in zip(self.linears, (query, key))]
        return attention(q, k, mask=mask, dropout=self.dropout)

#create a dualgcn model
class GCN(nn.Module):
    def __init__(self, embeddings, mem_dim, n_layers):
        super(GCN, self).__init__()
        self.mem_dim = mem_dim
        self.input_dim = 360 # 300+30+30
        self.emb, self.pos_emb, self.post_emb = embeddings
        self.layers = n_layers

        # rnn
        input_size = self.input_dim
        self.rnn = nn.LSTM(input_size, 50, 1, batch_first=True, \
                dropout=0.1, bidirectional=True)
        self.input_dim = 100 #50 * 2 because of bidirection

        # dropout
        self.rnn_drop = nn.Dropout(0.1)
        self.in_drop = nn.Dropout(0.7)
        self.gcn_drop = nn.Dropout(0.1)

        # gcn
        self.W = nn.ModuleList()
        for layer in range(self.layers):
            input_dim = self.input_dim if layer == 0 else self.mem_dim
            self.W.append(nn.Linear(input_dim, self.mem_dim))

        #attention
        self.attention_heads = 1
        self.attn = MultiHeadAttention(self.attention_heads, self.mem_dim*2)
        self.weight_list = nn.ModuleList()
        for j in range(self.layers):
            input_dim = self.input_dim if j == 0 else self.mem_dim
            self.weight_list.append(nn.Linear(input_dim, self.mem_dim))

        #affine
        self.affine1 = nn.Parameter(torch.Tensor(self.mem_dim, self.mem_dim))
        self.affine2 = nn.Parameter(torch.Tensor(self.mem_dim, self.mem_dim))

    def rnn_encode(self, rnn_inputs, seq_lens, batch_size):
        state_shape = (2, batch_size, 50)
        h0 = c0 = Variable(torch.zeros(*state_shape), requires_grad=False).cuda()
        rnn_inputs = nn.utils.rnn.pack_padded_sequence(rnn_inputs, seq_lens.cpu(), batch_first=True, enforce_sorted=False)
        rnn_outputs, (ht, ct) = self.rnn(rnn_inputs, (h0, c0))
        rnn_outputs, _ = nn.utils.rnn.pad_packed_sequence(rnn_outputs, batch_first=True)
        return rnn_outputs

    def forward(self, adj, inputs):
        tok, asp, pos, head, deprel, post, mask, l, _ = inputs           # unpack inputs
        src_mask = (tok != 0).unsqueeze(-2)
        maxlen = max(l.data)
        mask_ = (torch.zeros_like(tok) != tok).float().unsqueeze(-1)[:, :maxlen]

        # embedding
        word_embs = self.emb(tok)
        embs = [word_embs] + [self.pos_emb(pos)] + [self.post_emb(post)]
        embs = torch.cat(embs, dim=2)
        embs = self.in_drop(embs)

        # rnn
        self.rnn.flatten_parameters()
        gcn_inputs = self.rnn_drop(self.rnn_encode(embs, l, tok.size()[0]))
        denom_dep = adj.sum(2).unsqueeze(2) + 1
        attn_tensor = self.attn(gcn_inputs, gcn_inputs, src_mask)
        attn_adj_list = [attn_adj.squeeze(1) for attn_adj in torch.split(attn_tensor, 1, dim=1)]
        outputs_dep = None
        adj_ag = None

        # average multi-head attention matrixes
        for i in range(self.attention_heads):
            if adj_ag is None:
                adj_ag = attn_adj_list[i]
            else:
                adj_ag += attn_adj_list[i]
        adj_ag = adj_ag / self.attention_heads

        for j in range(adj_ag.size(0)):
            adj_ag[j] -= torch.diag(torch.diag(adj_ag[j]))
            adj_ag[j] += torch.eye(adj_ag[j].size(0)).cuda()
        adj_ag = mask_ * adj_ag

        outputs_ag = gcn_inputs
        outputs_dep = gcn_inputs
        denom_ag = adj_ag.sum(2).unsqueeze(2) + 1
        for l in range(self.layers):
            # SynGCN part
            Ax_dep = adj.bmm(outputs_dep)
            AxW_dep = self.W[l](Ax_dep)
            AxW_dep = AxW_dep / denom_dep
            gAxW_dep = F.relu(AxW_dep)

            # SemGCN part
            Ax_ag = adj_ag.bmm(outputs_ag)
            AxW_ag = self.weight_list[l](Ax_ag)
            AxW_ag = AxW_ag / denom_ag
            gAxW_ag = F.relu(AxW_ag)

            # mutual Biaffine
            A1 = F.softmax(torch.bmm(torch.matmul(gAxW_dep, self.affine1), torch.transpose(gAxW_ag, 1, 2)), dim=-1)
            A2 = F.softmax(torch.bmm(torch.matmul(gAxW_ag, self.affine2), torch.transpose(gAxW_dep, 1, 2)), dim=-1)
            gAxW_dep, gAxW_ag = torch.bmm(A1, gAxW_ag), torch.bmm(A2, gAxW_dep)
            if l < self.layers - 1:
                outputs_dep = self.gcn_drop(gAxW_dep)
                outputs_ag = self.gcn_drop(gAxW_ag)
            else:
                outputs_dep = gAxW_dep
                outputs_ag = gAxW_ag
        return outputs_ag, outputs_dep, adj_ag

class GCNAbsaModel(nn.Module):
    def __init__(self, emb_mat):
        super().__init__()
        self.emb_mat = emb_mat
        self.emb = nn.Embedding.from_pretrained(torch.tensor(emb_mat, dtype=torch.float), freeze=True)
        self.pos_emb = nn.Embedding(49, 30, padding_idx=0)
        self.post_emb = nn.Embedding(146, 30, padding_idx=0)
        embeddings = (self.emb, self.pos_emb, self.post_emb)
        self.gcn = GCN(embeddings, 50, 2)

    def forward(self, inputs):
        tok, asp, pos, head, deprel, post, mask, l, adj = inputs           # unpack inputs
        maxlen = max(l.data)
        mask = mask[:, :maxlen]
        adj_dep = adj[:, :maxlen, :maxlen].float()
        h1, h2, adj_ag = self.gcn(adj_dep, inputs)    
        # avg pooling asp feature
        aspect_words_n = mask.sum(dim=1).unsqueeze(-1)
        mask = mask.unsqueeze(-1).repeat(1,1,50)
        outputs1 = (h1*mask).sum(dim=1) / aspect_words_n
        outputs2 = (h2*mask).sum(dim=1) / aspect_words_n
        return outputs1, outputs2, adj_ag, adj_dep

class DualGCNClassifier(nn.Module):
    def __init__(self, emb_mat):
        super().__init__()
        self.gcn_model = GCNAbsaModel(emb_mat=emb_mat)
        self.classifier = nn.Linear(100, 3) #100 is input dim * 2

    def forward(self, inputs):
        outputs1, outputs2, adj_ag, adj_dep = self.gcn_model(inputs)
        final_outputs = torch.cat((outputs1, outputs2), dim=-1)
        logits = self.classifier(final_outputs)
        adj_ag_T = adj_ag.transpose(1, 2)
        identity = torch.eye(adj_ag.size(1)).cuda()
        identity = identity.unsqueeze(0).expand(adj_ag.size(0), adj_ag.size(1), adj_ag.size(1))
        ortho = adj_ag@adj_ag_T
        for i in range(ortho.size(0)):
            ortho[i] -= torch.diag(torch.diag(ortho[i]))
            ortho[i] += torch.eye(ortho[i].size(0)).cuda()
        penal = 0.25 * (torch.norm(ortho - identity) / adj_ag.size(0)).cuda() + 0.25 * (adj_ag.size(0) / torch.norm(adj_ag - adj_dep)).cuda()
        return logits, penal

#create a logger for better interpretability
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))


inputs_cols = ['text', 'aspect', 'pos', 'head', 'deprel', 'post', 'mask', 'length', 'adj']
tokenizer = build_tokenizer(
                fnames=['./DualGCN/dataset/MAMS/train.json', './DualGCN/dataset/MAMS/test.json'], 
                max_length=85, 
                data_file='{}/{}_tokenizer.dat'.format('./DualGCN/dataset/MAMS', 'mams'))
embedding_matrix = build_embedding_matrix(
                vocab=tokenizer.vocab, 
                embed_dim=300, 
                data_file='{}/{}d_{}_embedding_matrix.dat'.format('./DualGCN/dataset/MAMS', str(300), 'mams'))

token_vocab = VocabHelp.load_vocab('./DualGCN/dataset/MAMS/vocab_tok.vocab')
post_vocab = VocabHelp.load_vocab('./DualGCN/dataset/MAMS/vocab_post.vocab')
pos_vocab = VocabHelp.load_vocab('./DualGCN/dataset/MAMS/vocab_pos.vocab')
dep_vocab = VocabHelp.load_vocab('./DualGCN/dataset/MAMS/vocab_dep.vocab')
pol_vocab = VocabHelp.load_vocab('./DualGCN/dataset/MAMS/vocab_pol.vocab')

post_size = len(post_vocab)
pos_size = len(pos_vocab)
            
vocab_help = (post_vocab, pos_vocab, dep_vocab, pol_vocab)
model = DualGCNClassifier(embedding_matrix).to('cuda:0')
trainset = SentenceDataset('./DualGCN/dataset/MAMS/train.json', tokenizer, vocab_help=vocab_help)
testset = SentenceDataset('./DualGCN/dataset/MAMS/test.json', tokenizer, vocab_help=vocab_help)
train_dataloader = DataLoader(dataset=trainset, batch_size=16, shuffle=True)
test_dataloader = DataLoader(dataset=testset, batch_size=16)

criterion = nn.CrossEntropyLoss()
_params = filter(lambda p: p.requires_grad, model.parameters())
optimizer = torch.optim.Adam(_params, lr=0.002, weight_decay=1e-4)

max_test_acc_overall = 0
max_f1_overall = 0

#reset parameters
for p in model.parameters():
    if p.requires_grad:
        if len(p.shape) > 1:
            torch.nn.init.xavier_uniform_(p)
        else:
            stdv = 1. / (p.shape[0]**0.5)
            torch.nn.init.uniform_(p, a=-stdv, b=stdv)

#define evaluate function
def evaluate(model, show_results=False):
    model.eval()
    n_test_correct, n_test_total = 0, 0
    targets_all, outputs_all = None, None
    with torch.no_grad():
        for batch, sample_batched in enumerate(test_dataloader):
            inputs = [sample_batched[col].to('cuda:0') for col in inputs_cols]
            targets = sample_batched['polarity'].to('cuda:0')
            outputs, penal = model(inputs)
            n_test_correct += (torch.argmax(outputs, -1) == targets).sum().item()
            n_test_total += len(outputs)
            targets_all = torch.cat((targets_all, targets), dim=0) if targets_all is not None else targets
            outputs_all = torch.cat((outputs_all, outputs), dim=0) if outputs_all is not None else outputs
    test_acc = n_test_correct / n_test_total
    f1 = metrics.f1_score(targets_all.cpu(), torch.argmax(outputs_all, -1).cpu(), labels=[0, 1, 2], average='macro')

    labels = targets_all.data.cpu()
    predic = torch.argmax(outputs_all, -1).cpu()
    if show_results:
        report = metrics.classification_report(labels, predic, digits=4)
        confusion = metrics.confusion_matrix(labels, predic)
        return report, confusion, test_acc, f1

    return test_acc, f1

#train
max_test_acc = 0
max_f1 = 0
global_step = 0
model_path = ''

for epoch in range(20):
    logger.info('#' * 60)
    logger.info('epoch: {}'.format(epoch))
    n_correct, n_total = 0, 0
    for i_batch, sample_batched in enumerate(train_dataloader):
        global_step += 1
        model.train()
        optimizer.zero_grad()
        inputs = [sample_batched[col].to('cuda:0') for col in inputs_cols]
        outputs, penal = model(inputs)
        targets = sample_batched['polarity'].to('cuda:0')
        
        loss = criterion(outputs, targets) + penal
        loss.backward()
        optimizer.step()
                
        if global_step % 5 == 0:
            n_correct += (torch.argmax(outputs, -1) == targets).sum().item()
            n_total += len(outputs)
            train_acc = n_correct / n_total
            test_acc, f1 = evaluate(model)
            if test_acc > max_test_acc:
                max_test_acc = test_acc
                if test_acc > max_test_acc_overall:
                    if not os.path.exists('./DualGCN/state_dict'):
                        os.mkdir('./DualGCN/state_dict')
                    model_path = './DualGCN/state_dict/{}_{}_acc_{:.4f}_f1_{:.4f}'.format('dualgcn', 'mams', test_acc, f1)
                    best_model = copy.deepcopy(model)
                    logger.info('>> saved: {}'.format(model_path))
            if f1 > max_f1:
                max_f1 = f1
            logger.info('loss: {:.4f}, acc: {:.4f}, test_acc: {:.4f}, f1: {:.4f}'.format(loss.item(), train_acc, test_acc, f1))

logger.info('max_test_acc: {0}, max_f1: {1}'.format(max_test_acc, max_f1))
max_test_acc_overall = max(max_test_acc, max_test_acc_overall)
max_f1_overall = max(max_f1, max_f1_overall)
torch.save(best_model.state_dict(), model_path)
logger.info('>> saved: {}'.format(model_path))
logger.info('#' * 60)
logger.info('max_test_acc_overall:{}'.format(max_test_acc_overall))
logger.info('max_f1_overall:{}'.format(max_f1_overall))

model = best_model
model.eval()
test_report, test_confusion, acc, f1 = evaluate(model, show_results=True)
logger.info("Precision, Recall and F1-Score")
logger.info(test_report)
logger.info("Confusion Matrix")
logger.info(test_confusion)
