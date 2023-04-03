import pandas as pd
import numpy as np


def merge_features(full='full', feature_list=['domains_mean',
                                              'reqs',
                                              'part_day',
                                              'price',
                                              'manuf',
                                              'region',
                                              'demo']):
    '''merging most important features in one dataframe'''

    biv_factors = np.load('./context_data/biv_f.npy')
    biv_factors = pd.DataFrame(biv_factors, columns=['als'+str(x) for x in range(biv_factors.shape[1])])
    biv_factors['user_id'] = list(range(biv_factors.shape[0]))

    if full == 'full':
        features = biv_factors
    else:
        features = pd.read_parquet('./context_data/feats/vect2000_sep.parquet')
        features = features.merge(biv_factors, how='inner', on='user_id')

    for feat in feature_list:
        features = features.merge(pd.read_parquet(f'./context_data/feats/{feat}.parquet'))

    features.to_parquet(f'./context_data/features_merged_{full}.parquet')
    print('all features are merged and saved')


if __name__ == '__main_':
    merge_features()
