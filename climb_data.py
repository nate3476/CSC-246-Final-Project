import csv
import os
import boardlib.db.aurora
import sqlite3
import pandas as pd
import torch

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from climb_util import make_climb_set


# Preprocessing to get the average difficulty of each hold
# TODO think about if we want to factor this in
def get_hold_difficulties(filename='climbset.csv'):
    hold_difficulty = dict()
    hold_count = dict()
    dataset = pd.read_csv(filename)
    for row in dataset.itertuples():
        holds = row.frames.split('p')
        for h in holds[1:]:
            label = h.split('r')[0]
            if label not in hold_difficulty:
                hold_difficulty[label] = row.difficulty_average
                hold_count[label] = 1
            else:
                hold_difficulty[label] += row.difficulty_average
                hold_count[label] += 1
    print(dataset.columns)
    for label in hold_difficulty.keys():
        hold_difficulty[label] /= hold_count[label]
    return hold_difficulty


class KilterDataset(Dataset):
    def __init__(self, root, download=True):
        """
        :param root: The directory to store the Kilter database, token dictionary, and csv file for the dataset
        :param download: Whether to try to download the database/csv if it is not present
        """
        # TODO add train and test here
        self.root = root
        self.database = os.path.join(root, 'KilterDatabase.db')
        climb_data_path = os.path.join(root, 'climb_set.csv')
        token_dict_path = os.path.join(root, 'token_dict.csv')

        # Make sure we have the Kilterboard database
        if download and not os.path.exists(self.database):
            boardlib.db.aurora.download_database('kilter', self.database)
        # Make sure we have the dataset of climbs and token labels
        if download and not os.path.exists(climb_data_path):
            make_climb_set(self.root)

        token_df = pd.read_csv(token_dict_path, header=None, names=["label", "token"])
        self.token_dict = dict(zip(token_df.label, token_df.token))

        # read in the dataset of climbs TODO here is probably where to do train vs test
        rows = []
        with open(climb_data_path, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                rows.append(parts)
        # normalize length so pandas can hold it
        max_len = max(len(r) for r in rows)
        climb_df = pd.DataFrame([r + [None] * (max_len - len(r)) for r in rows])

        # convert to tensors
        self.uuids = climb_df.iloc[:, 0].tolist()
        self.diffs = climb_df.iloc[:, 1].astype(float).tolist()
        self.seqs = [
            torch.tensor([int(x) for x in row.dropna().iloc[2:]])
            for _, row in climb_df.iterrows()
        ]

    def __len__(self):
        return len(self.uuids)

    def __getitem__(self, idx):
        """
        Gets one item from the dataset
        :param idx: The index to get
        :return: The climb at that index, giving the climb uuid, difficulty, and a sequence of pairs (hold_id, role_id)
        """
        uuid = self.uuids[idx]
        diff = self.diffs[idx]
        seq = self.seqs[idx]

        return uuid, diff, seq


def climb_collate_fn(batch):
    """
    Custom collate_fn which pads the hold sequences to the same length
    :param batch: input batch of climb_uuids, difficulties, and hold sequences
    :return: batch with the hold sequences padded to the max length with EOS tokens
    """
    uuids = [item[0] for item in batch]
    diffs = [item[1] for item in batch]
    seqs = [item[2] for item in batch]

    # convert difficulties to a tensor
    diffs = torch.tensor(diffs, dtype=torch.float)
    # pad sequences to the same length with -infty
    padded_seqs = torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=-1)

    return {'uuids': uuids, 'diffs': diffs, 'seqs': padded_seqs}


def main():
    dataset = KilterDataset(root='data', download=True)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, collate_fn=climb_collate_fn)
    for batch in dataloader:
        uuids = batch['uuids']
        diffs = batch['diffs']
        seqs = batch['seqs']
        print(uuids[0], diffs[0], seqs[0])
        break


if __name__ == '__main__':
    main()
