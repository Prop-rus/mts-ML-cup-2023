import sys
import torch
import cornac
import pandas as pd
import numpy as np

from recommenders.utils.timer import Timer
from recommenders.utils.constants import SEED

print("System version: {}".format(sys.version))
print("PyTorch version: {}".format(torch.__version__))
print("Cornac version: {}".format(cornac.__version__))

# top k items to recommend
TOP_K = 20

# Model parameters
LATENT_DIM = 300
ENCODER_DIMS = [400]
ACT_FUNC = "tanh"
LIKELIHOOD = "pois"
NUM_EPOCHS = 20
BATCH_SIZE = 32
LEARNING_RATE = 0.001


def train_BiVAE():
    train = pd.read_pickle('./context_data/data_cf_gr_coded.pkl')

    train.rename(columns={'user_id': 'userID', 'url_host': 'itemID', 'request_cnt': 'rating'}, inplace=True)

    # train, test = python_chrono_split(data, 0.75, col_timestamp='eventTime')

    train_set = cornac.data.Dataset.from_uir(train.itertuples(index=False), seed=SEED)

    print('Number of users: {}'.format(train_set.num_users))
    print('Number of items: {}'.format(train_set.num_items))

    bivae = cornac.models.BiVAECF(
        k=LATENT_DIM,
        encoder_structure=ENCODER_DIMS,
        act_fn=ACT_FUNC,
        likelihood=LIKELIHOOD,
        n_epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        seed=SEED,
        use_gpu=torch.cuda.is_available(),
        verbose=True
    )

    with Timer() as t:
        bivae.fit(train_set)
    print("Took {} seconds for training.".format(t))
    bivae.bivae.cpu()
    bivae.save('./data/saves')

    u_factors = bivae.bivae.mu_theta

    np.save('./context_data/biv_f.npy', u_factors.cpu())
    print('users factors saves')


if __name__ == '__main__':
    train_BiVAE()
