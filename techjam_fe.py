import os
import datetime

import numpy as np
import pandas as pd


# Percentile function from https://stackoverflow.com/questions/19894939/calculate-arbitrary-percentile-on-pandas-groupby
def percentile(n):
    def percentile_(x):
        return np.percentile(x, n)
    percentile_.__name__ = 'percentile_%s' % n
    return percentile_


def get_prep_data(DATA_DIR):
    # Read files
    cc = pd.read_csv(os.path.join(DATA_DIR,'cc.csv'),parse_dates=['pos_dt'])
    demo = pd.read_csv(os.path.join(DATA_DIR,'demographics.csv'))
    kplus = pd.read_csv(os.path.join(DATA_DIR,'kplus.csv'),parse_dates=['sunday'])

    train = pd.read_csv(os.path.join(DATA_DIR,'train.csv'))
    test = pd.read_csv(os.path.join(DATA_DIR,'test.csv'))

    # Some file setups NOTE: test data will have income of 0, for the sake of data preparation.
    cc_mapper = demo[['id','cc_no']].copy()
    demo = demo.drop('cc_no',axis=1).drop_duplicates().reset_index(drop=True)
    label = pd.concat([train,test],axis=0,ignore_index=True)
    demo = demo.merge(label, on='id')
    demo['ocp_cd'] = demo['ocp_cd'].fillna(0).astype(int)
    demo.set_index('id',inplace=True)
    kplus.set_index('id',inplace=True)
    joined_cc = cc.merge(cc_mapper, on='cc_no', how='inner').drop('cc_no', axis=1)

    # Add month tab
    kplus['month'] = kplus['sunday'].dt.month
    kplus['month'] = 'month'+ kplus['month'].astype(str)
    joined_cc['month'] = joined_cc.pos_dt.dt.month
    joined_cc['month'] = 'month'+ joined_cc['month'].astype(str)

    # Bank's holidays from Bank of Thailand, and a bunch of features
    bank_holidays = ['2018-01-01','2018-01-02','2018-03-01','2018-04-06','2018-04-13',
                    '2018-04-14','2018-04-15','2018-04-16','2018-05-01','2018-05-29']
    joined_cc['is_holiday'] = joined_cc['pos_dt'].isin([datetime.datetime.strptime(i, '%Y-%m-%d') for i in bank_holidays]).astype(int)
    joined_cc['is_weekend'] = joined_cc['pos_dt'].dt.weekday.isin([0,6]).astype(int)
    joined_cc['is_holiday'] = 'holiday'+joined_cc['is_holiday'].astype(str)
    joined_cc['is_weekend'] = 'weekend'+joined_cc['is_weekend'].astype(str)
    joined_cc['quarter'] = 'q'+((joined_cc['pos_dt'].dt.month>=4)+1).astype(str)

    # Generated features
    demo = demo.reset_index()
    demo['cc_cnt'] = demo['id'].map(cc_mapper.groupby('id').cc_no.count())
    demo['has_kp'] = demo['id'].isin(kplus.index).astype(int)

    # Crossing categorical features with another feature [Modified from group 374 / 336's submission code]
    demo['age_gnd'] = demo['gender'].astype(str)+demo['age'].astype(str)
    demo['gnd_ocp'] = demo['gender'].astype(str)+demo['ocp_cd'].astype(str)
    demo['age_ocp'] = demo['age'].astype(str)+demo['ocp_cd'].astype(str)

    categorical_features = ['gender','ocp_cd','age_gnd','gnd_ocp','age_ocp','age']

    # Target Encoding, code modified from group 374's submission code
    def target_encode(df, target_col, cat_cols):
        for feature in cat_cols:
            df[feature+'_targetmean'] = df[feature].map(df.groupby(feature)[target_col].mean())

    def freq_encode(df, target_col, cat_cols):
        for feature in cat_cols:
            df[feature+'_freq'] = df[feature].map(df.groupby(feature)[target_col].count())
    
    target_encode(demo, 'income', categorical_features)
    freq_encode(demo, 'income', categorical_features)
    
    
    demo.set_index('id',inplace=True)
    
    # Preparing Training data
    
    train = demo.copy()
    # Normal Total Groupby
    kplus_tot = kplus.groupby('id').agg({'kp_txn_count':'sum','kp_txn_amt':'sum'}).copy()
    kplus_mm_tot = kplus.groupby(['id','month']).agg({'kp_txn_count':'sum','kp_txn_amt':'sum'}).unstack(level=1).copy()
    kplus_mm_tot.columns = ['_'.join([str(c) for c in lst]) for lst in kplus_mm_tot.columns]

    # CreditCard Total Groupby
    cc_tot = joined_cc.groupby('id').agg({'cc_txn_amt':['count','sum']}).copy()
    cc_tot.columns = ['_'.join(i) for i in cc_tot.columns]

    # CreditCard Monthly Groupby
    combined_cc = pd.pivot_table(joined_cc, index= 'id', columns= 'month', values= 'cc_txn_amt', aggfunc= [np.mean, min, max, np.sum, 'count', np.var, percentile(10), percentile(90)])
    combined_cc.columns = ['cc_'+'_'.join([str(c) for c in lst]) for lst in combined_cc.columns]


    # CreditCard Pompous Features
    combined_cc_holiday = pd.pivot_table(joined_cc, index= 'id', columns= 'is_holiday', values= 'cc_txn_amt', aggfunc= [np.mean, min, max, np.sum, 'count', np.var, percentile(10), percentile(90)])
    combined_cc_weekend = pd.pivot_table(joined_cc, index= 'id', columns= 'is_weekend', values= 'cc_txn_amt', aggfunc= [np.mean, min, max, np.sum, 'count', np.var, percentile(10), percentile(90)])
    combined_cc_quarter = pd.pivot_table(joined_cc, index= 'id', columns= 'quarter', values= 'cc_txn_amt', aggfunc= [np.mean, min, max, np.sum, 'count', np.var, percentile(10), percentile(90)])
    combined_cc_holiday.columns = ['cc_'+'_'.join([str(c) for c in lst]) for lst in combined_cc_holiday.columns]
    combined_cc_weekend.columns = ['cc_'+'_'.join([str(c) for c in lst]) for lst in combined_cc_weekend.columns]
    combined_cc_quarter.columns = ['cc_'+'_'.join([str(c) for c in lst]) for lst in combined_cc_quarter.columns]

    # Joining all together
    train = train.join(kplus_tot).join(kplus_mm_tot).join(cc_tot).join(combined_cc).join(combined_cc_holiday).join(combined_cc_weekend).join(combined_cc_quarter).fillna(0)

    X_train = train[train['income']>0].drop('income',axis=1).copy()
    y_train = pd.DataFrame(train[train['income']>0]['income']).copy()
    X_test = train[train['income']<=0].drop('income',axis=1).copy()

    return X_train, y_train, X_test