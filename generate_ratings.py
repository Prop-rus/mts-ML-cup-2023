import pyarrow.parquet as pq


DATA_FILE = 'competition_data_final_pqt'
LOCAL_DATA_PATH = './context_data/'

def prepare_ratings():
    data = pq.read_table(f'{LOCAL_DATA_PATH}{DATA_FILE}')

    data = data.to_pandas()
#     saving only url to save time next time
    data.to_parquet(f'./{LOCAL_DATA_PATH}/df_urls.parquet')
    print('data with url saved') 

    data = data[['user_id', 'url_host']]
    data = data.sort_values('user_id')

    data_agg = data.select(['user_id', 'url_host', 'request_cnt']).\
        group_by(['user_id', 'url_host']).aggregate([('request_cnt', "sum")])

    url_set = set(data_agg.select(['url_host']).to_pandas()['url_host'])
    print(f'{len(url_set)} urls')
    url_dict = {url: idurl for url, idurl in zip(url_set, range(len(url_set)))}

    data_agg['url_host'] = data_agg['url_host'].map(url_dict)
#      saving counts to use as ratings in ALS or BiVAECF
    data_agg.to_parquet(f'./{LOCAL_DATA_PATH}/data_cf_gr_coded.parquet')
    print(f'ratings saved in {LOCAL_DATA_PATH}')


if __name__ == '__main__':
    prepare_ratings()
