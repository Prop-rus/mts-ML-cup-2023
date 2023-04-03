import pandas as pd


def final_submission():

    gender_short = pd.read_csv('./context_data/gender_short_predicts.csv')
    gender_full = pd.read_csv('./context_data/gender_full_predicts.csv')

    users_to_add = gender_full.query('user_id not in @gender_short.user_id')

    gender_fin = pd.concat((gender_short, users_to_add))

    age_short = pd.read_csv('./context_data/age_short_predicts.csv')
    age_full = pd.read_csv('./context_data/age_full_predicts.csv')

    users_to_add = age_full.query('user_id not in @age_short.user_id')

    age_fin = pd.concat((age_short, users_to_add))

    fin = gender_fin.merge(age_fin, how='inner', on='user_id')
    fin.to_csv('./output/prediction.csv', index=False)
    print('final prediction saved')
