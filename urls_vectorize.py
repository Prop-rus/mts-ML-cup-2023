import pandas as pd
import pyarrow.parquet as pq
from sklearn.feature_extraction.text import TfidfVectorizer


def feature_vectorized_urls():
    # loading by each file to save resources (whole dataset crashed the kernel)
    for i in range(10):
        data = pq.read_table(f'./context_data/competition_data_final_pqt/part-0000{i}-aba60f69-2b63-4cc1-95ca-542598094698-c000.snappy.parquet')
        data = data.select(['user_id', 'url_host']).to_pandas()
        data['url_host'] = data['url_host'].apply(lambda x: x.split('.')[:-1])
        data = data.explode('url_host')
        data.to_parquet(f'./context_data/urls/{i}.parquet')

    from collections import Counter
    cnt = Counter([])
    for i in range(10):
        data = pd.read_parquet(f'./context_data/urls/{i}.parquet')
        cnt += Counter(data['url_host'].tolist())

    data = pd.DataFrame([], columns=['user_id', 'url_host'])
    for i in range(10):
        data_tmp = pd.read_parquet(f'./context_data/urls/{i}.parquet')
        data_tmp = data_tmp.query('url_host in @df_cnt.url')
        data_tmp = data_tmp.groupby('user_id')['url_host'].apply(lambda x: ' '.join(x)).reset_index()
        data = pd.concat((data, data_tmp))

    data_flat = data.groupby('user_id')['url_host'].apply(lambda x: ' '.join(x)).reset_index()

    vectorizer = TfidfVectorizer()
    vect = vectorizer.fit_transform(data_flat['url_host'].tolist())
    vect = vect.todense()
    vect = pd.DataFrame(vect, columns=['tf' + str(x) for x in range(vect.shape[1])])
    vect = pd.concat((vect, data_flat[['user_id']].reset_index(drop=True)), axis=1, ignore_index=True)

    for col in vect.columns.tolist():
        if isinstance(col, int):
            vect.rename(columns={col: 'tf' + str(col)}, inplace=True)

    vect.to_parquet('./context_data/feats/vect2000_sep.parquet')
    print('feature vectorized urls saved')


if __name__ == '__main__':
    feature_vectorized_urls()
