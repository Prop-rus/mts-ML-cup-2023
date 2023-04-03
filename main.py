from urls_vectorize import feature_vectorized_urls
from generate_rating import prepare_ratings
from feature_generation import make_all_features
from datasets import prepare_all_datasets
from DANet_fit_predict import fit_predict_DANet
from BiVAECF import train_BiVAE
from final_submission import final_submission
from merge_features import merge_features


def main():

#     generate and save features of vectorized urls
    feature_vectorized_urls()

#     generate ratings and train BiVAE based on them. Then user factors of trained model are saved
    prepare_ratings()
    train_BiVAE()

#     generate and save features based on meta information
    make_all_features()

     # for the reason that some users do not have enough count of visiting urls to include
     # them in predicting model, we generate different sets of features: with all users ('full') and only with 
     # users who have enough actions ('short'). We train and predict them separatly, and the append predictions of
     # 'full' model to 'short' results
    merge_features('full')
    merge_features('short')

#     form datasets to train models and submission
    prepare_all_datasets('full')
    prepare_all_datasets('short')

#     train and predict by DANet model separatly: for gender and age
    fit_predict_DANet(full='full', mode='gender')
    fit_predict_DANet(full='short', mode='gender')

    fit_predict_DANet(full='full', mode='age')
    fit_predict_DANet(full='short', mode='age')

#     merge all files with prediociotns
    final_submission()
