import pandas as pd


def age_bucket(x):
    if x >= 19 and x <= 25:
        return 1
    elif x >= 26 and x <= 35:
        return 2
    elif x >= 36 and x <= 45:
        return 3
    elif x >= 46 and x <= 55:
        return 4
    elif x >= 56 and x <= 65:
        return 5
    elif x >= 66:
        return 6


def prepare_train_gender(features, targets, full='full'):
    train_gender = targets.merge(features, how='inner', on='user_id')
    train_gender = train_gender[train_gender['is_male'] != 'NA']
    train_gender = train_gender.query('is_male.notnull()', engine='python')

    train_gender['is_male'] = train_gender['is_male'].map(int)
    train_gender.to_parquet(f'./context_data/train_gender_{full}.parquet')
    print('train_gender saved')


def prepare_train_age(features, targets, full='full'):
    train_age = targets.merge(features, how='inner', on=['user_id'])

    train_age = train_age[(train_age['age'] != 'NA')]
    train_age = train_age.query('age > 18')
    train_age = train_age.query('age.notnull()', engine='python')
    train_age['age'] = train_age['age'].map(age_bucket)
    train_age.to_parquet(f'./context_data/train_age_{full}.parquet')
    print('train_age saved')


def prepare_submission(features, id_to_submit, full='full'):
    submission = id_to_submit.merge(features, how='inner', on='user_id')
    submission.to_parquet(f'./contex_data/submission_{full}.parquet')


def prepare_all_datasets(full='full'):

    features = pd.read_parquet(f'./context_data/features_merged_{full}.parquet')
    targets = pd.read_parquet('./context_data/public_train.pqt')
    id_to_submit = pd.read_parquet('./context_data/submit_2.pqt')

    prepare_train_gender(features, targets, full=full)
    prepare_train_age(features, targets, full=full)
    prepare_submission(features, id_to_submit, full=full)


if __name__ == '__main__':
    prepare_all_datasets(full='full')
