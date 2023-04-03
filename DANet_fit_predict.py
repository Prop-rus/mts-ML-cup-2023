import gc
import numpy as np
import pandas as pd
import torch

from tqdm import tqdm
import pickle
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from DANet.DANetClassifier import DANetClassifier


def save(obj, path, verbose=True):
    if verbose:
        print("Saving object to {}".format(path))

    with open(path, "wb") as obj_file:
        pickle.dump(obj, obj_file, protocol=pickle.HIGHEST_PROTOCOL)

    if verbose:
        print("Object saved to {}".format(path))
    pass


def load(path, verbose=True):
    if verbose:
        print("Loading object from {}".format(path))
    with open(path, "rb") as obj_file:
        obj = pickle.load(obj_file)
    if verbose:
        print("Object loaded from {}".format(path))
    return obj


def gender_score(model, x, y):
    y_pred = model.predict_proba(x)
    if len(y_pred.shape) == 2:
        y_pred = y_pred[:, 1]
    gini_score = 2.0 * roc_auc_score(y, y_pred) - 1.0
    return gini_score


def fit_predict_DANet(mode='gender', cv=10, full='full'):
    if mode == 'gender':
        y_column = 'is_male'
        score_f = gender_score
    else:
        y_column = 'age'
        score_f = f1_score
    train_df = pd.read_parquet(f'./context_data/features_{mode}_{full}.parquet')
    submission_df = pd.read_parquet(f'./context_data/submission_{full}.parquet')

    y = train_df[y_column].values
    y = y.astype(np.float64)

    train_df.drop(columns=["user_id", "age", "is_male"], inplace=True)
    feature_names = np.array(train_df.columns)
    x = train_df.values
    del train_df

    submission_ids = submission_df["user_id"].values
    del submission_df["user_id"]
    submission_features = submission_df.values

    selected_feature_ids = []
    for i in range(len(feature_names)):
        selected_feature_ids.append(i)
    selected_feature_ids = np.array(selected_feature_ids)

    for i in tqdm(range(len(selected_feature_ids)), desc="Scaling selected features"):
        current_feature_id = selected_feature_ids[i]

        # contaminated version (using features from unclassified samples to improve scaling)
        scaler = StandardScaler()
        scaler.partial_fit(x[:, current_feature_id].reshape((-1, 1)))
        scaler.partial_fit(submission_features[:, current_feature_id].reshape((-1, 1)))
        x[:, current_feature_id] = scaler.transform(x[:, current_feature_id].reshape((-1, 1))).reshape(-1,)
        submission_features[:, current_feature_id] = scaler.transform(submission_features[:, current_feature_id].reshape((-1, 1))).reshape(-1,)

    # training
    print('training')
    i = 0
    val_scores = []
    k_fold = StratifiedKFold(n_splits=cv, shuffle=True, random_state=45)
    for train_ids, val_ids in tqdm(k_fold.split(x, y), desc="Fitting cv classifiers"):

        x_t, y_t = x[train_ids], y[train_ids]
        x_v, y_v = x[val_ids], y[val_ids]

        model = DANetClassifier(input_dim=len(x_t[0]),
                                num_classes=len(np.unique(y)),
                                layer_num=32, base_outdim=64, k=5,
                                virtual_batch_size=256, drop_rate=0.1,
                                device="cuda")
        model.fit(x_t, y_t, x_v, y_v, start_lr=0.008, end_lr=0.0001, batch_size=2048, epochs=20)
        save(model, f"./saves/danet/danet_{mode}_{full}_{i}.pkl".format(i))
        val_score_i = score_f(model, x_v, y_v)
        val_scores.append(val_score_i)

        del model
        gc.collect()
        torch.cuda.empty_cache()
        i += 1
    print(val_scores)
    print("Mean val score: {}".format(np.mean(val_scores)))

#     predicting
    print('predicting')
    probas = []
    for i in tqdm(range(cv), desc="Predicting probas"):
        model = load(f"./saves/danet/danet_{mode}_{full}_{i}.pkl")
        probas_i = model.predict_proba(submission_features)[:, 1]
        probas_i = probas_i.reshape((-1, 1))
        probas.append(probas_i)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    if mode == 'gender':
        probas = np.hstack(probas)
        mean_probas = np.mean(probas, axis=1)

    else:
        y_unique = np.unique(y)
        submission_predicts = []
        final_probas = []
        for i in tqdm(range(len(submission_features)), desc="Building predicts"):
            probas_i = []
            for j in range(len(probas)):
                probas_j = probas[j][i]
                probas_j = probas_j.reshape((-1, 1))
                probas_i.append(probas_j)
            probas_i = np.hstack(probas_i)
            mean_probas_i = np.mean(probas_i, axis=1)
            final_probas.append(mean_probas_i)
            max_proba_id = np.argmax(mean_probas_i)
            predicted_label = y_unique[max_proba_id]
            submission_predicts.append(predicted_label)
        submission_predicts = np.array(submission_predicts)

    submission_ids = submission_ids.reshape((-1, 1))
    submission_predicts = mean_probas.reshape((-1, 1))
    submission_data = np.hstack([submission_ids, submission_predicts])

    my_submission_df = pd.DataFrame(data=submission_data, columns=["user_id", y_column])
    my_submission_df["user_id"] = my_submission_df["user_id"].astype(int)
    my_submission_df.to_csv(f"{mode}_{full}_predicts.csv", index=False)
    print("Submission builded and saved")
    print("done")


if __name__ == '__main__':
    fit_predict_DANet()
