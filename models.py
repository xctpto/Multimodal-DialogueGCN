import torch.nn as nn
import torch.geometric
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv, GraphConv
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import copy
import time

import torch
from tqdm import tqdm
from sklearn import metrics




class GCN(nn.Module):
    '''
    g_dim: number of features
    h1_dim: hidden size
    '''

    def __init__(self, g_dim, h1_dim, h2_dim, args):
        super(GCN, self).__init__()
        self.num_relations = 2 * args.n_speakers ** 2
        self.conv1 = RGCNConv(g_dim, h1_dim, self.num_relations, num_bases=30)
        self.conv2 = GraphConv(h1_dim, h2_dim)

    def forward(self, node_features, edge_index, edge_norm, edge_type):
        x = self.conv1(node_features, edge_index, edge_type) #edge_norm
        x = self.conv2(x, edge_index)

        return x



class SeqContext(nn.Module):

    def __init__(self, u_dim, g_dim, args):
        super(SeqContext, self).__init__()
        self.input_size = u_dim
        self.hidden_dim = g_dim

        self.rnn = nn.GRU(self.input_size, self.hidden_dim // 2, dropout=args.drop_rate,
                          bidirectional=True, num_layers=2, batch_first=True)

    def forward(self, text_len_tensor, text_tensor):
        packed = pack_padded_sequence(
            text_tensor,
            text_len_tensor,
            batch_first=True,
            enforce_sorted=False
        )
        out, _ = self.rnn(packed, None)

        out, _ = pad_packed_sequence(out, batch_first=True)

        return out




class EdgeAtt(nn.Module):

    def __init__(self, g_dim, args):
        super(EdgeAtt, self).__init__()
        self.device = args.device
        self.wp = args.wp
        self.wf = args.wf

        self.weight = nn.Parameter(torch.zeros((g_dim, g_dim)).float(), requires_grad=True)
        var = 2. / (self.weight.size(0) + self.weight.size(1))
        self.weight.data.normal_(0, var)

    def forward(self, node_features, text_len_tensor, edge_ind):
        batch_size, mx_len = node_features.size(0), node_features.size(1)
        alphas = []

        weight = self.weight.unsqueeze(0).unsqueeze(0)
        att_matrix = torch.matmul(weight, node_features.unsqueeze(-1)).squeeze(-1)  # [B, L, D_g]
        for i in range(batch_size):
            cur_len = text_len_tensor[i].item()
            alpha = torch.zeros((mx_len, 110)).to(self.device)
            for j in range(cur_len):
                s = j - self.wp if j - self.wp >= 0 else 0
                e = j + self.wf if j + self.wf <= cur_len - 1 else cur_len - 1
                tmp = att_matrix[i, s: e + 1, :]  # [L', D_g]
                feat = node_features[i, j]  # [D_g]
                score = torch.matmul(tmp, feat)
                probs = F.softmax(score)  # [L']
                alpha[j, s: e + 1] = probs
            alphas.append(alpha)

        return alphas


class Classifier(nn.Module):
    def __init__(self, input_dim, hidden_size, tag_size, args):
        super(Classifier, self).__init__()
        self.emotion_att = MaskedEmotionAtt(input_dim)
        self.lin1 = nn.Linear(input_dim, hidden_size)
        self.drop = nn.Dropout(args.drop_rate)
        self.lin2 = nn.Linear(hidden_size, tag_size)
        self.lin3 = nn.Linear(200, 100)
        self.lin4 = nn.Linear(100, tag_size)
        self.loss_weights = torch.tensor([1 / 0.086747, 1 / 0.144406, 1 / 0.227883,
                                          1 / 0.160585, 1 / 0.127711, 1 / 0.252668]).to(args.device)
        self.nll_loss = nn.NLLLoss(self.loss_weights)

    def get_prob(self, h, text_len_tensor):  # similarity-based attention mechanism

        hidden = self.drop(F.relu(self.lin1(h)))
        if args.exp1 or args.exp3:
            scores = self.drop(F.relu(self.lin3(hidden)))
            scores = self.lin4(scores)
        else:
            scores = self.lin2(hidden)
        log_prob = F.log_softmax(scores, dim=-1)

        return log_prob

    def forward(self, h, text_len_tensor):
        log_prob = self.get_prob(h, text_len_tensor)
        y_hat = torch.argmax(log_prob, dim=-1)

        return y_hat

    def get_loss(self, h, label_tensor, text_len_tensor):
        log_prob = self.get_prob(h, text_len_tensor)
        loss = self.nll_loss(log_prob, label_tensor)

        return loss


class MaskedEmotionAtt(nn.Module):

    def __init__(self, input_dim):
        super(MaskedEmotionAtt, self).__init__()
        self.lin = nn.Linear(input_dim, input_dim)

    def forward(self, h, text_len_tensor):
        batch_size = text_len_tensor.size(0)
        x = self.lin(h)  # [node_num, H]
        ret = torch.zeros_like(h)
        s = 0
        for bi in range(batch_size):
            cur_len = text_len_tensor[bi].item()
            y = x[s: s + cur_len]
            z = h[s: s + cur_len]
            scores = torch.mm(z, y.t())  # [L, L]
            probs = F.softmax(scores, dim=1)
            out = z.unsqueeze(0) * probs.unsqueeze(-1)  # [1, L, H] x [L, L, 1] --> [L, L, H]
            out = torch.sum(out, dim=1)  # [L, H]
            ret[s: s + cur_len, :] = out
            s += cur_len

        return ret


class GRU(nn.Module):
    def __init__(self, input_dim, output_dim, args):
        super(GRU, self).__init__()
        self.input_size = input_dim
        self.hidden_dim = output_dim

        self.rnn = nn.GRU(self.input_size, self.hidden_dim // 2, dropout=args.drop_rate,
                          bidirectional=True, num_layers=2, batch_first=True)

    def forward(self, textlen_tensor, input_tensor):
        packed = pack_padded_sequence(
            input_tensor,
            textlen_tensor,
            batch_first=True,
            enforce_sorted=False
        )
        out, _ = self.rnn(packed, None)
        out, _ = pad_packed_sequence(out, batch_first=True)

        return out

    def unpad(self, out, lengths, device):
        batch_size = out.size(0)
        features = []

        for j in range(batch_size):
            cur_len = lengths[j].item()
            features.append(out[j, :cur_len, :])

        features = torch.cat(features, dim=0).to(device)  # [E, D_g]

        return features


class DialogueGCN_DL(nn.Module):
    """
    Decision-Level fusion
    """

    def __init__(self, args):
        super(DialogueGCN_DL, self).__init__()
        t_dim = 100  # dimensionality of an input (feature vectors)
        v_dim = 512
        a_dim = 100
        g_dim = 300  # dimensionality of g vectors
        # GCN dimensions
        h1_dim = 300
        h2_dim = 100
        hc_dim = 100
        tag_size = 6

        self.dbl = False
        self.wp = args.wp
        self.wf = args.wf
        self.device = args.device

        self.audiovid = args.audiovid
        self.textvid = args.textvid
        self.audiotext = args.audiotext

        self.AUDIOrnn = SeqContext(a_dim, 100, args)
        self.VIDEOrnn = SeqContext(v_dim, 100, args)
        self.TEXTrnn = SeqContext(t_dim, 100, args)

        self.edge_att = EdgeAtt(g_dim, args)
        self.edge_att_dbl = EdgeAtt(200, args)
        self.gcn = GCN(300, h1_dim, h2_dim, args)
        self.gcn_dbl = GCN(200, h1_dim, h2_dim, args)
        self.ext_context = GRU(612, 200, args)

        self.clf = Classifier(g_dim + h2_dim, hc_dim, tag_size, args)
        self.clf_dbl = Classifier(200 + h2_dim, hc_dim, tag_size, args)

        edge_type_to_idx = {}
        for j in range(args.n_speakers):
            for k in range(args.n_speakers):
                edge_type_to_idx[str(j) + str(k) + '0'] = len(edge_type_to_idx)
                edge_type_to_idx[str(j) + str(k) + '1'] = len(edge_type_to_idx)
        self.edge_type_to_idx = edge_type_to_idx
        log.debug(self.edge_type_to_idx)

    def get_rep(self, data):
        text_features = self.TEXTrnn(data["text_len_tensor"], data["text_tensor"])  # [batch_size, mx_len, D_g]
        audio_features = self.AUDIOrnn(data["text_len_tensor"], data["audio"])  # [batch_size, mx_len, D_g]
        video_features = self.VIDEOrnn(data["text_len_tensor"], data["video"])  # [batch_size, mx_len, D_g]
        if self.audiovid:
            self.dbl = True
            node_features = torch.cat([video_features, audio_features], dim=-1)
        elif self.textvid:
            self.dbl = True
            node_features = torch.cat([text_features, video_features], dim=-1)
        elif self.audiotext:
            self.dbl = True
            node_features = torch.cat([text_features, audio_features], dim=-1)
        else:
            node_features = torch.cat([text_features, video_features, audio_features], dim=-1)
        if self.dbl:
            features, edge_index, edge_norm, edge_type, edge_index_lengths = batch_graphify(
                node_features, data["text_len_tensor"], data["speaker_tensor"], self.wp, self.wf,
                self.edge_type_to_idx, self.edge_att_dbl, self.device)
        else:
            features, edge_index, edge_norm, edge_type, edge_index_lengths = batch_graphify(
                node_features, data["text_len_tensor"], data["speaker_tensor"], self.wp, self.wf,
                self.edge_type_to_idx, self.edge_att, self.device)
        if self.dbl:
            graph_out = self.gcn_dbl(features, edge_index, edge_norm, edge_type)
        else:
            graph_out = self.gcn(features, edge_index, edge_norm, edge_type)

        return graph_out, features

    def forward(self, data):
        graph_out, features = self.get_rep(data)
        if self.dbl:
            out = self.clf_dbl(torch.cat([features, graph_out], dim=-1), data["text_len_tensor"])
        else:
            out = self.clf(torch.cat([features, graph_out], dim=-1), data["text_len_tensor"])
        return out

    def get_loss(self, data):
        graph_out, features = self.get_rep(data)
        if self.dbl:
            loss = self.clf_dbl.get_loss(torch.cat([features, graph_out], dim=-1),
                                         data["label_tensor"], data["text_len_tensor"])
        else:
            loss = self.clf.get_loss(torch.cat([features, graph_out], dim=-1),
                                     data["label_tensor"], data["text_len_tensor"])

        return loss

class DialogueGCN(nn.Module):

    def __init__(self, args):
        super(DialogueGCN, self).__init__()
        u_dim = 100 if args.exp3 else 712 if args.multim else 612 if args.audiovid else 612 if args.textvid else 200 if args.audiotext else 100 #dimensionality of an input (feature vectors)
        g_dim = 200 # dimensionality of g vectors
        # GCN dimensions
        h1_dim = 100
        h2_dim = 100
        hc_dim = 100
        tag_size = 6

        self.wp = args.wp
        self.wf = args.wf
        self.device = args.device
        self.exp1 = args.exp1
        self.exp3 = args.exp3
        self.audio_only = args.audio_only
        self.vid_only = args.vid_only
        self.audiovid = args.audiovid
        self.textvid = args.textvid
        self.audiotext = args.audiotext

        self.rnn = SeqContext(u_dim, g_dim, args)
        self.edge_att = EdgeAtt(g_dim, args)
        self.gcn = GCN(g_dim, h1_dim, h2_dim, args)
        self.ext_context = GRU(512,200, args) if self.textvid else GRU(100,200, args) if self.audiotext else GRU(612,200, args)
        self.rnn_vid = SeqContext(512, g_dim, args)

        if self.exp1:
          hc_dim = 200
          self.clf = Classifier(g_dim + h2_dim + 512 + 100, hc_dim, tag_size, args) # changed to accomodate exp1
        elif self.exp3:
          hc_dim = 200
          self.clf = Classifier(g_dim + h2_dim + 200, hc_dim, tag_size, args ) # accomodating exp3
        else:
          self.clf = Classifier(g_dim + h2_dim, hc_dim, tag_size, args)

        edge_type_to_idx = {}
        for j in range(args.n_speakers):
            for k in range(args.n_speakers):
                edge_type_to_idx[str(j) + str(k) + '0'] = len(edge_type_to_idx)
                edge_type_to_idx[str(j) + str(k) + '1'] = len(edge_type_to_idx)
        self.edge_type_to_idx = edge_type_to_idx
        log.debug(self.edge_type_to_idx)

    def get_rep(self, data):
      if self.exp3:
        node_features = self.rnn(data["text_len_tensor"], data["text_tensor"])
      if self.audio_only:
          node_features = self.rnn(data["text_len_tensor"], data["audio"])
      elif self.vid_only:
          node_features = self.rnn_vid(data["text_len_tensor"], data["video"])
      else:
        node_features = self.rnn(data["text_len_tensor"], data["text_tensor"]) # [batch_size, mx_len, D_g]

      features, edge_index, edge_norm, edge_type, edge_index_lengths = batch_graphify(
            node_features, data["text_len_tensor"], data["speaker_tensor"], self.wp, self.wf,
            self.edge_type_to_idx, self.edge_att, self.device)

      graph_out = self.gcn(features, edge_index, edge_norm, edge_type)

      return graph_out, features

    def forward(self, data):
        graph_out, features = self.get_rep(data)
        if self.exp1:
            dat = torch.cat([features, graph_out, data['video_tensor'], data['audio_tensor']], dim=-1)
            out = self.clf(torch.cat([features, graph_out, data['video_tensor'], data['audio_tensor']], dim=-1), data["text_len_tensor"])

        elif self.exp3:
            rnn_out = self.ext_context(data['text_len_tensor'],data['audiovid_tensor'])
            rnn_features = self.ext_context.unpad(rnn_out, data['text_len_tensor'], self.device)
            out = self.clf(torch.cat([features, graph_out, rnn_features],dim=-1), data["text_len_tensor"])

        else:
            out = self.clf(torch.cat([features, graph_out], dim=-1), data["text_len_tensor"])
        return out

    def get_loss(self, data):
        graph_out, features = self.get_rep(data)

        if self.exp1:
          loss = self.clf.get_loss(torch.cat([features, graph_out, data['video_tensor'], data['audio_tensor']], dim=-1),
                                    data["label_tensor"], data["text_len_tensor"])
        elif self.exp3:
            rnn_out = self.ext_context(data['text_len_tensor'],data["audiovid_tensor"])
            rnn_features = self.ext_context.unpad(rnn_out, data['text_len_tensor'], self.device)
            loss = self.clf.get_loss(torch.cat([features, graph_out, rnn_features], dim=-1),
                                     data['label_tensor'], torch.tensor([612], device=args.device) )
        else:

            loss = self.clf.get_loss(torch.cat([features, graph_out], dim=-1),
                                 data["label_tensor"], data["text_len_tensor"])

        return loss


class Optim:

    def __init__(self, lr, max_grad_value, weight_decay):
        self.lr = lr
        self.max_grad_value = max_grad_value
        self.weight_decay = weight_decay
        self.params = None
        self.optimizer = None

    def set_parameters(self, params, name):
        self.params = list(params)
        if name == "sgd":
            self.optimizer = optim.SGD(self.params, lr=self.lr, weight_decay=self.weight_decay)
        elif name == "rmsprop":
            self.optimizer = optim.RMSprop(self.params, lr=self.lr, weight_decay=self.weight_decay)
        elif name == "adam":
            self.optimizer = optim.Adam(self.params, lr=self.lr, weight_decay=self.weight_decay)

    def step(self):
        if self.max_grad_value != -1:
            clip_grad_value_(self.params, self.max_grad_value)
        self.optimizer.step()

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)




class Coach:

    def __init__(self, trainset, devset, testset, model, opt, args):
        self.trainset = trainset
        self.devset = devset
        self.testset = testset
        self.model = model
        self.opt = opt
        self.args = args
        self.label_to_idx = {'hap': 0, 'sad': 1, 'neu': 2, 'ang': 3, 'exc': 4, 'fru': 5}
        self.best_dev_f1 = None
        self.best_epoch = None
        self.best_state = None
        self.train_hist = {'loss': [], 'f1': [], 'acc': []}
        self.dev_hist = {'loss': [], 'f1': [], 'acc': []}
        self.test_hist = {'loss': [], 'f1': [], 'acc': []}

    def load_ckpt(self, ckpt):
        self.best_dev_f1 = ckpt["best_dev_f1"]
        self.best_epoch = ckpt["best_epoch"]
        self.best_state = ckpt["best_state"]
        self.model.load_state_dict(self.best_state)

    def train(self):
        log.debug(self.model)
        # Early stopping.
        best_dev_f1, best_epoch, best_state = self.best_dev_f1, self.best_epoch, self.best_state

        # Train
        for epoch in range(1, self.args.epochs + 1):
            self.train_epoch(epoch)
            dev_f1, dev_loss, dev_acc = self.evaluate()
            # log.info("[Dev set] [f1 {:.4f}]".format(dev_f1))
            self.dev_hist['f1'].append(dev_f1)
            self.dev_hist['loss'].append(dev_loss)
            self.dev_hist['acc'].append(dev_acc)

            if best_dev_f1 is None or dev_f1 > best_dev_f1:
                best_dev_f1 = dev_f1
                best_epoch = epoch
                best_state = copy.deepcopy(self.model.state_dict())
                log.info("Save the best model.")

            # log.info("[Test set] [f1 {:.4f}]".format(test_f1))
            train_f1, train_loss, train_acc = self.evaluate(train=True)
            self.train_hist['f1'].append(train_f1)
            self.train_hist['loss'].append(train_loss)
            self.train_hist['acc'].append(train_acc)

        # Log statistics of the best model
        self.model.load_state_dict(best_state)
        # log.info("")
        # log.info("Best in epoch {}:".format(best_epoch))
        dev_f1, dev_loss, dev_acc = self.evaluate()
        # log.info("[Dev set] [f1 {:.4f}]".format(dev_f1))

        return best_dev_f1, best_epoch, best_state

    def train_epoch(self, epoch):
        start_time = time.time()
        epoch_loss = 0
        self.model.train()
        # for idx in tqdm(np.random.permutation(len(self.trainset)), desc="train epoch {}".format(epoch)):
        # self.trainset.shuffle()
        for idx in tqdm(range(len(self.trainset)), desc="train epoch {}".format(epoch)):
            self.model.zero_grad()
            data = self.trainset[idx]
            for k, v in data.items():
                data[k] = v.to(self.args.device)
            nll = self.model.get_loss(data)
            epoch_loss += nll.item()
            nll.backward()
            self.opt.step()

        end_time = time.time()
        # log.info("")
        # log.info("[Epoch %d] [Loss: %f] [Time: %f]" %
        # (epoch, epoch_loss, end_time - start_time))

    def evaluate(self, test=False, train=False):
        dataset = self.testset if test else self.trainset if train else self.devset
        self.model.eval()
        with torch.no_grad():
            golds = []
            preds = []
            epoch_loss = 0
            for idx in tqdm(range(len(dataset)), desc="test" if test else "dev"):
                data = dataset[idx]
                golds.append(data["label_tensor"])
                for k, v in data.items():
                    data[k] = v.to(self.args.device)
                y_hat = self.model(data)
                nll = self.model.get_loss(data)
                epoch_loss += nll.item()
                preds.append(y_hat.detach().to("cpu"))

            golds = torch.cat(golds, dim=-1).numpy()
            preds = torch.cat(preds, dim=-1).numpy()
            f1 = metrics.f1_score(golds, preds, average="weighted")
            acc = metrics.accuracy_score(golds, preds)

        return f1, epoch_loss, acc


class arguments():
    def __init__(self, data, from_begin=True, device='cpu', epochs=1, batch_size=32, optimizer='adam',
                 learning_rate=0.0001, weight_decay=1e-8, max_grad_value=1, drop_rate=0.5,
                 wp=10, wf=10, n_speakers=2, hidden_size=100, rnn='lstm', class_weight=False, multim=False, exp1=False,
                 exp3=False, decision_level=False,
                 audio_only=False, vid_only=False, audiovid=False, audiotext=False, textvid=False):
        self.data = data  # data_path
        self.from_begin = from_begin
        self.device = device
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_grad_value = max_grad_value
        self.drop_rate = drop_rate
        self.wp = wp
        self.wf = wf
        self.n_speakers = n_speakers
        self.hidden_size = hidden_size
        self.rnn = rnn
        self.seed = 100
        self.class_weight = class_weight
        self.multim = multim
        self.exp1 = exp1
        self.exp3 = exp3
        self.decision_level = decision_level
        self.audio_only = audio_only
        self.vid_only = vid_only
        self.audiovid = audiovid
        self.audiotext = audiotext
        self.textvid = textvid
