import math
import random


class Dataset:

    def __init__(self, samples, batch_size, multim=False, audiovid=False, audiotext=False, textvid=False, exp3=False,
                 decision_level=False):
        self.samples = samples
        self.batch_size = batch_size
        self.num_batches = math.ceil(len(self.samples) / batch_size)
        self.speaker_to_idx = {'M': 0, 'F': 1}
        self.multim = multim
        self.size = 100 if exp3 else 100 if decision_level else 712 if multim else 200 if audiotext else 612 if textvid else 612 if audiovid else 100
        self.audiovid = audiovid
        self.audiotext = audiotext
        self.textvid = textvid
        self.exp3 = exp3
        self.decision_level = decision_level

    def __len__(self):
        return self.num_batches

    def __getitem__(self, index):
        batch = self.raw_batch(index)
        return self.padding(batch)

    def raw_batch(self, index):
        assert index < self.num_batches, "batch_idx %d > %d" % (index, self.num_batches)
        batch = self.samples[index * self.batch_size: (index + 1) * self.batch_size]

        return batch

    def padding(self, samples):
        batch_size = len(samples)
        text_len_tensor = torch.tensor([len(s.text) for s in samples]).long()
        mx = torch.max(text_len_tensor).item()
        text_tensor = torch.zeros((batch_size, mx, self.size))
        a_tensor = torch.zeros((batch_size, mx, 100))
        v_tensor = torch.zeros((batch_size, mx, 512))
        audiovid_tensor = torch.zeros((batch_size, mx, 100)) if self.audiotext else torch.zeros(
            (batch_size, mx, 512)) if self.textvid else torch.zeros((batch_size, mx, 612))

        speaker_tensor = torch.zeros((batch_size, mx)).long()
        labels = []
        n_utterances = 0
        for i, s in enumerate(samples):
            cur_len = len(s.text)
            n_utterances += cur_len
            tmp = [torch.from_numpy(t).float() for t in s.text]
            tmp_vid = [torch.from_numpy(t).float() for t in s.visual]
            tmp_aud = [torch.from_numpy(t).float() for t in s.audio]
            tmp = torch.stack(tmp)
            tmp_vid = torch.stack(tmp_vid)
            tmp_aud = torch.stack(tmp_aud)
            if i == 0:
                video_tensor = tmp_vid
                audio_tensor = tmp_aud
            else:
                video_tensor = torch.cat((video_tensor, tmp_vid), 0)
                audio_tensor = torch.cat((audio_tensor, tmp_aud), 0)

            if self.multim:
                aud = torch.tensor(s.audio)
                vid = torch.tensor(s.visual)
                tmp = torch.cat((tmp, vid, aud), 1)
            elif self.audiovid and not self.exp3 and not self.decision_level:
                aud = torch.tensor(s.audio)
                vid = torch.tensor(s.visual)
                tmp = torch.cat((vid, aud), 1)
            elif self.audiotext and not self.exp3 and not self.decision_level:
                aud = torch.tensor(s.audio)
                tmp = torch.cat((tmp, aud), 1)
            elif self.textvid and not self.exp3 and not self.decision_level:
                vid = torch.tensor(s.visual)
                tmp = torch.cat((tmp, vid), 1)

            audvid = torch.cat((tmp_vid, tmp_aud), 1)
            if self.textvid:
                audiovid_tensor[i, :cur_len, :] = tmp_vid
            elif self.audiotext:
                audiovid_tensor[i, :cur_len, :] = tmp_aud
            else:
                audiovid_tensor[i, :cur_len, :] = audvid
            text_tensor[i, :cur_len, :] = tmp
            a_tensor[i, :cur_len, :] = tmp_aud
            v_tensor[i, :cur_len, :] = tmp_vid
            speaker_tensor[i, :cur_len] = torch.tensor([self.speaker_to_idx[c] for c in s.speaker])
            labels.extend(s.label)
        # video_tensor = torch.zeros((n_utterances, 512)) #added <- should be size(num_utterances in batch, vec_dim)
        # audio_tensor = torch.zeros((n_utterances, 100)) #added <- same as above

        label_tensor = torch.tensor(labels).long()
        data = {
            "text_len_tensor": text_len_tensor,
            "text_tensor": text_tensor,
            "speaker_tensor": speaker_tensor,
            "label_tensor": label_tensor,
            "video_tensor": video_tensor,
            "audio_tensor": audio_tensor,
            "audiovid_tensor": audiovid_tensor,
            "audio": a_tensor,
            "video": v_tensor

        }

        return data

    def shuffle(self):
        random.shuffle(self.samples)


import numpy as np
import torch

import utils

log = utils.get_logger()


def batch_graphify(features, lengths, speaker_tensor, wp, wf, edge_type_to_idx, att_model, device):
    node_features, edge_index, edge_norm, edge_type = [], [], [], []
    batch_size = features.size(0)
    length_sum = 0
    edge_ind = []
    edge_index_lengths = []

    for j in range(batch_size):
        edge_ind.append(edge_perms(lengths[j].cpu().item(), wp, wf))

    edge_weights = att_model(features, lengths, edge_ind)

    for j in range(batch_size):
        cur_len = lengths[j].item()
        node_features.append(features[j, :cur_len, :])
        perms = edge_perms(cur_len, wp, wf)
        perms_rec = [(item[0] + length_sum, item[1] + length_sum) for item in perms]
        length_sum += cur_len
        edge_index_lengths.append(len(perms))

        for item, item_rec in zip(perms, perms_rec):
            edge_index.append(torch.tensor([item_rec[0], item_rec[1]]))
            edge_norm.append(edge_weights[j][item[0], item[1]])
            # edge_norm.append(edge_weights[j, item[0], item[1]])

            speaker1 = speaker_tensor[j, item[0]].item()
            speaker2 = speaker_tensor[j, item[1]].item()
            if item[0] < item[1]:
                c = '0'
            else:
                c = '1'
            edge_type.append(edge_type_to_idx[str(speaker1) + str(speaker2) + c])

    node_features = torch.cat(node_features, dim=0).to(device)  # [E, D_g]
    edge_index = torch.stack(edge_index).t().contiguous().to(device)  # [2, E]
    edge_norm = torch.stack(edge_norm).to(device)  # [E]
    edge_type = torch.tensor(edge_type).long().to(device)  # [E]
    edge_index_lengths = torch.tensor(edge_index_lengths).long().to(device)  # [B]

    return node_features, edge_index, edge_norm, edge_type, edge_index_lengths


def edge_perms(length, window_past, window_future):
    """
    Method to construct the edges of a graph (a utterance) considering the past and future window.
    return: list of tuples. tuple -> (vertice(int), neighbor(int))
    """

    all_perms = set()
    array = np.arange(length)
    for j in range(length):
        perms = set()

        if window_past == -1 and window_future == -1:
            eff_array = array
        elif window_past == -1:  # use all past context
            eff_array = array[:min(length, j + window_future + 1)]
        elif window_future == -1:  # use all future context
            eff_array = array[max(0, j - window_past):]
        else:
            eff_array = array[max(0, j - window_past):min(length, j + window_future + 1)]

        for item in eff_array:
            perms.add((j, item))
        all_perms = all_perms.union(perms)
    return list(all_perms)