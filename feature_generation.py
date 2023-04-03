import pandas as pd
import numpy as np
import pyarrow.parquet as pq

LOCAL_DATA_PATH = './context_data/'
SPLIT_SEED = 42
DATA_FILE = 'competition_data_final_pqt'
TARGET_FILE = 'public_train.pqt'
SUBMISSION_FILE = 'submit.pqt'


def feature_request_cnt(data):
    '''make statistic feature of request counts'''
    data = data.select(['user_id', 'request_cnt']).to_pandas()
    df_req = data.groupby('user_id').agg([np.sum, np.mean, np.min, np.max, np.std])
    df_req.columns = [x[0]+x[1] for x in df_req.columns]
    data_sum = data.request_cnt.sum()
    df_req['request_cnmean_tot'] = df_req.request_cntsum / data_sum
    df_req = df_req[['request_cntsum',
                     'request_cntmean',
                     'request_cntamin',
                     'request_cntamax',
                     'request_cnmean_tot']]
    df_req.to_parquet('./context_data/feats/reqs.parquet')
    print('feature request counts saved')


def feature_domains(data):
    '''make statistic feature of domains in urls (.ru, .com and etc)'''
    data = data.select(['user_id', 'url_host']).to_pandas()
    data['url_host'] = data['url_host'].apply(lambda x: '.' + x.split('.')[-1])

#     read files with all domains. This info was parsed from external resources
    all_domains = []
    with open('./context_data/all_domains.txt') as f:
        lines = f.readlines()
        for line in lines:
            all_domains.append(line[:-1])

    data.loc[~data.url_host.isin(all_domains), 'url_host'] = 'unk'
    val_cnt = data['url_host'].value_counts()
    data.loc[~data['url_host'].isin(val_cnt[val_cnt > 10000].index), 'url_host'] = 'rare'

    data['fake'] = 1
    data_piv = pd.pivot_table(data,
                              values='fake',
                              index=['user_id'],
                              columns=['url_host'],
                              aggfunc=np.mean)
    data_agg = data.groupby('user_id').sum()

    data_piv.div(data_agg.values, axis=0).to_parquet('./context_data/feats/domains_mean.parquet')
    print('feature domains mean saved')


def feature_part_of_day(data):
    '''make statistics with part of a day of visiting url'''
    data = data.select(['user_id', 'part_of_day']).to_pandas()
    data['fake'] = 1
    df_part_day = pd.pivot_table(data,
                                 values='fake',
                                 index=['user_id'],
                                 columns=['part_of_day'],
                                 aggfunc=[np.sum])
    df_part_day.columns = [x[0]+x[1] for x in df_part_day.columns]
    df_part_day_sum = df_part_day.sum(axis=1)
    df_part_day = df_part_day.div(df_part_day_sum.values, axis=0)
    df_part_day = df_part_day.fillna(0)
    df_part_day.to_parquet('./context_data/feats/part_day.parquet')
    print('feature part of a day saved')


def feature_day_of_week(data):
    '''make statistics with day of week, that we take from date of vizit'''
    data = data.select(['user_id', 'date']).to_pandas()
    data['date'] = pd.to_datetime(data.date).dt.dayofweek
    data['fake'] = 1
    df_dayw = pd.pivot_table(data,
                             values='fake',
                             index=['user_id'],
                             columns=['date'],
                             aggfunc=[np.sum])
    df_dayw.columns = ['dw'+x[0]+str(x[1]) for x in df_dayw.columns]
    df_dayw_sum = df_dayw.sum(axis=1)
    df_dayw = df_dayw.div(df_dayw_sum.values, axis=0)
    df_dayw = df_dayw.fillna(0)
    df_dayw.to_parquet('./context_data/feats/dayw.parquet')
    print('feature day of the week saved')


def feature_os(data):
    '''make stat of operation system of device'''
    data = data.select(['user_id', 'cpe_model_os_type']).to_pandas()
    data.loc[data.cpe_model_os_type == "Apple iOS", 'cpe_model_os_type'] = 'iOS'
    df_os = pd.pivot_table(data,
                           values='fake',
                           index=['user_id'],
                           columns=['cpe_model_os_type'],
                           aggfunc=[np.mean])
    df_os.columns = [x[0]+x[1] for x in df_os.columns]
    df_os = df_os.fillna(0)
    df_os.to_parquet('./context_data/feats/os.parquet')
    print('feature os saved')


def feature_price(data):
    '''make feature of mean price of users device'''
    data = data.select(['user_id', 'price']).to_pandas()
    df_price = data.groupby('user_id').agg([np.mean])
    df_price.columns = [x[0] + x[1] for x in df_price.columns]
    df_price = df_price.fillna(0)
    df_price.to_parquet('./context_data/feats/price.parquet')
    print('feature of mean price of device saved')


def feature_device_type(data):
    '''make stat of users device type'''
    data = data.select(['user_id', 'cpe_type_cd']).to_pandas()
    data['fake'] = 1
    dev_type = pd.pivot_table(data,
                              values='fake',
                              index=['user_id'],
                              columns=['cpe_type_cd'],
                              aggfunc=[np.mean])
    dev_type.columns = [x[0]+x[1] for x in dev_type.columns]
    dev_type = dev_type.fillna(0)
    dev_type.to_parquet('./context_data/feats/dev_type.parquet')
    print('feature of device type saved')


def feature_manufacturer_name(data):
    '''make stat of users devices manufacturer name'''
    data = data.select(['user_id', 'cpe_manufacturer_name']).to_pandas()
    data.loc[data.cpe_manufacturer_name == 'Huawei Device Company Limited', 'cpe_manufacturer_name'] = 'Huawei'
    data['fake'] = 1
    manuf = pd.pivot_table(data,
                           values='fake',
                           index=['user_id'],
                           columns=['cpe_manufacturer_name'],
                           aggfunc=[np.mean])
    manuf = manuf.fillna(0)
    manuf.to_parquet('./context_data/feats/manuf.parquet')
    print('feature of manufacturer name of the users devices saved')


def feature_region(data):
    '''make stat of region of the users'''
    data = data.select(['user_id', 'region_name']).to_pandas()
    data['fake'] = 1
    region = pd.pivot_table(data,
                            values='fake',
                            index=['user_id'],
                            columns=['region_name'],
                            aggfunc=[np.mean])
    region = region.fillna(0)
    region.to_parquet('./context_data/feats/region.parquet')


def feature_demography(data):
    '''make stat of demography features of regions and cities of users
    the data of statistics by the regions and cities was parsed from external sources'''
    data = data.select(['user_id', 'region_name', 'city_name']).to_pandas()
    data = data.groupby('user_id').head(1)

#    city population
    pop = pd.read_csv('./context_data/stat/population.csv', delimiter=';')
    pop.rename(columns={'city': 'city_name'}, inplace=True)
    data = data.merge(pop, how='left', on='city_name')
    data.loc[data.population.isna(), 'population'] = 50

#     mean income by cities
    zp = pd.read_csv('./context_data/stat/zp.csv', delimiter=';')
    zp.columns = ['place', 'city_name', '1', 'zp']
    zp = zp[['city_name', 'zp']]
    data = data.merge(zp, how='left', on='city_name')

#     mean income by regions
    zp_reg = pd.read_csv('./context_data/stat/zp_reg.csv', delimiter=';')
    zp_reg.rename(columns={'reg': 'region_name'}, inplace=True)
    data.region_name = data.region_name.str.replace('область', 'обл.')
    data.region_name = data.region_name.str.replace('Республика ', '')
    data = data.merge(zp_reg, how='left', on='region_name')
    data.loc[data.zp.isnull(), 'zp'] = data.loc[data.zp.isnull(), 'zp_reg']
    data.zp = data.zp.str.replace(',', '.')
    data.zp = data.zp.astype('float')

#     mean age of population
    vozr = pd.read_csv('./context_data/stat/vozr.csv', delimiter=';')
    vozr.rename(columns={'region': 'region_name'}, inplace=True)
    data = data.merge(vozr, how='left', on='region_name')

    data.to_parquet('./context_data/feats/demo.parquet')
    print('demography saved')


def make_all_features():
    data = pq.read_table(f'{LOCAL_DATA_PATH}/{DATA_FILE}')
    feature_request_cnt(data)
    feature_domains(data)
    feature_part_of_day(data)
    feature_day_of_week(data)
    feature_os(data)
    feature_price(data)
    feature_device_type(data)
    feature_manufacturer_name(data)
    feature_region(data)
    feature_demography(data)


if __name__ == '__main__':
    make_all_features()
