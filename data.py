import argparse

from tqdm import tqdm
import pickle

import utils
from Sample import Sample

log = utils.get_logger()


def split():
    utils.set_seed(100)

    video_ids, video_speakers, video_labels, video_text, \
        video_audio, video_visual, video_sentence, trainVids, \
        test_vids = pickle.load(open('/.../IEMOCAP_features.pkl', 'rb'), encoding='latin1')

    train, dev, test = [], [], []
    dev_size = int(len(trainVids) * 0.1)
    train_vids, dev_vids = trainVids[dev_size:], trainVids[:dev_size]

    for vid in tqdm(train_vids, desc="train"):
        train.append(Sample(vid, video_speakers[vid], video_labels[vid],
                                 video_text[vid], video_audio[vid], video_visual[vid],
                                 video_sentence[vid]))
    for vid in tqdm(dev_vids, desc="dev"):
        dev.append(Sample(vid, video_speakers[vid], video_labels[vid],
                               video_text[vid], video_audio[vid], video_visual[vid],
                               video_sentence[vid]))
    for vid in tqdm(test_vids, desc="test"):
        test.append(Sample(vid, video_speakers[vid], video_labels[vid],
                                video_text[vid], video_audio[vid], video_visual[vid],
                                video_sentence[vid]))
    log.info("train vids:")
    log.info(sorted(train_vids))
    log.info("dev vids:")
    log.info(sorted(dev_vids))
    log.info("test vids:")
    log.info(sorted(test_vids))

    return train, dev, test


train, dev, test = split()
data = {"train": train, "dev": dev, "test": test}
with open('/.../data_split.pickle', 'wb') as handle:
    pickle.dump(data, handle)
