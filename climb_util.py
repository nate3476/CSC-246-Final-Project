import os
import sqlite3
import csv
import matplotlib.markers
import pandas as pd
import matplotlib.pyplot as plt
import boardlib.api.aurora
import torch
from PIL import Image
from sklearn.model_selection import train_test_split


def download_database(root):
    """
    Downloads the Kilterboard database to root
    :param root: the directory to store the data in
    """
    database = os.path.join(root, 'KilterDatabase.db')
    boardlib.db.aurora.download_database('kilter', database)


def make_climb_set(root):
    """
    Generates the full dataset of climbs, as well as the token dictionary. Assumes the Kilterboard database is present in root
    :param root: the directory with the Kilterboard database
    """
    database = os.path.join(root, 'KilterDatabase.db')
    climb_data_path = os.path.join(root, 'climb_set.csv')
    token_dict_path = os.path.join(root, 'token_dict.csv')

    with sqlite3.connect(database) as conn:
        # get the specified dataset from the Kilterboard database
        climb_df = pd.read_sql_query(
            r'SELECT climb_uuid, difficulty_average, frames FROM climbs JOIN climb_stats on climbs.uuid = climb_stats.climb_uuid WHERE climbs.angle=40 AND layout_id=1 AND ascensionist_count >= 5 AND quality_average > 1',
            conn
        )
        with open(climb_data_path, 'w', newline='') as climb_set_file:
            writer = csv.writer(climb_set_file)
            # Convert hold labels to tokens
            token_dict = dict()
            token_dict['BOSr12'] = 0
            token_dict['EOSr14'] = 1
            token = 2
            for row in climb_df.itertuples():
                frames = row.frames.split(',')[0]
                label_seq = frames.split('p')[1:]
                for label in label_seq:
                    if label not in token_dict:
                        token_dict[label] = token
                        token += 1
                seq = [token_dict[label] for label in label_seq]
                seq.insert(0, 0)
                seq.append(1)
                writer.writerow([row.climb_uuid, row.difficulty_average] + seq)
            pd.DataFrame.from_dict(data=token_dict, orient='index').to_csv(token_dict_path, header=False)


def get_label_dict(token_dict_path):
    """
    Gets the dictionary mapping tokens to labels and colors. Basically inverts the token_dict
    :param token_dict_path: path of the csv mapping label,color: token
    :return: the dictionary mapping each token to the string 'xxxryy', where xxxx is the label and yy is the color
    """
    token_df = pd.read_csv(token_dict_path, header=None, names=["label", "token"])
    token_dict = dict(zip(token_df.label, token_df.token))
    return {token: label for label, token in token_dict.items()}


# get the dictionary mapping hold labels to their x,y positions
def get_point_dict(database):
    with sqlite3.connect(database) as conn:
        hole_to_pos = {row[0]: (row[1], row[2]) for row in conn.execute("SELECT id, x, y FROM holes")}
        label_to_hole = {row[0]: row[1] for row in conn.execute("SELECT id, hole_id FROM placements")}
        point_map = {label: hole_to_pos[hole] for label, hole in label_to_hole.items()}
    return point_map


def show_climb(climb, root, download=True):
    """
    Generates an image of the given climb. Note it is assumed that token_dict.csv is present in root
    :param climb: a sequence of tokens
    :param root: the directory containing the Kilterboard database, token_dict csv, and board images
    :param download: whether to download the Kilterboard database, token_dict, and board images if they are not present
    """

    # make sure the Kilterboard database is present
    database = os.path.join(root, 'KilterDatabase.db')
    if download and not os.path.exists(database):
        download_database(root)

    # make sure the token_dict is present
    token_dict_path = os.path.join(root, 'token_dict.csv')
    if download and not os.path.exists(token_dict_path):
        make_climb_set(root)
    # get the mapping of labels to positions, and tokens to holds
    point_dict = get_point_dict(database)
    label_dict = get_label_dict(token_dict_path)

    # make sure the board image files are present in root
    image_path = os.path.join(root, 'product_sizes_layouts_sets')
    if download and not os.path.exists(image_path):
        boardlib.api.aurora.download_images('kilter', database_path=database, output_directory=root)
    bolt_on_path = os.path.join(image_path, 'original-16x12-bolt-ons-v2.png')
    screw_on_path = os.path.join(image_path, 'original-16x12-screw-ons-v2.png')

    # set the background to be all black
    fig, ax = plt.subplots()
    fig.patch.set_facecolor('dimgrey')
    ax.set_facecolor('dimgrey')

    # mapping from color labels to the display color
    color_map = {
        12: (0, 0.87, 0),  # start
        13: (0, 1, 1),  # middle
        14: (1, 0, 1),  # finish
        15: (1, 0.65, 0)  # foot only
    }

    # iterate through the climb and get the point position and color of each hold in the climb
    points = []
    colors = []
    index = 1
    while climb[index] > 1:
        hold = label_dict[climb[index]]
        label, color = hold.split('r')
        points.append(point_dict[int(label)])
        colors.append(color_map[int(color)])
        ax.annotate(climb[index], point_dict[int(label)], color='red')
        index += 1
    x, y = zip(*points)
    marker_style = matplotlib.markers.MarkerStyle(marker='o', fillstyle='none')
    plt.scatter(x, y, s=200, marker=marker_style, color=colors)

    # show the images of the holds
    hands_img = Image.open(bolt_on_path)
    feet_img = Image.open(screw_on_path)
    plt.imshow(hands_img, origin='upper', extent=(-24, 168, 0, 156))
    plt.imshow(feet_img, origin='upper', extent=(-24, 168, 0, 156))

    plt.show()


def climb_one_hot(batch, root, download=True):
    """
    Given a batch of climbs (encoded as sequences of tokens), return the same batch with each climb one-hot-encoded
    :param batch: a torch tensor of size [batch_size, max_seq_length]
    :param root: the directory containing the Kilterboard database and token_dict csv
    :param download: whether to download the Kilterboard database or token_dict csv if they aren't present
    :return: a torch tensor of size [batch_size, total_num_tokens] which is the one-hot encoding of each climb
    """
    # make sure the Kilterboard database is present
    database = os.path.join(root, 'KilterDatabase.db')
    if download and not os.path.exists(database):
        download_database(root)

    # make sure the token_dict is present
    token_dict_path = os.path.join(root, 'token_dict.csv')
    if download and not os.path.exists(token_dict_path):
        make_climb_set(root)
    # get the mapping of tokens to holds
    label_dict = get_label_dict(token_dict_path)

    seqs = batch.tolist()
    one_hots = []

    for climb in seqs:
        one_hot = [0] * len(label_dict)
        for token in climb:
            if token > 0:
                one_hot[token] = 1
        one_hots.append(one_hot)
    return torch.tensor(one_hots)


def main():
    climb = [0, 495, 491, 219, 496, 11, 124, 486, 497, 1]
    show_climb(climb, 'data')


if __name__ == '__main__':
    main()
