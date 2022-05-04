"""
Counterfactual Fairness (Kusner et al. 2017) Replication in Python 3 by Philip Ball

NB: Stan files courtesy of Matt Kusner

Options
-do_l2: Performs the replication of the L2 (Fair K) model, which can take a while depending on computing power
-save_l2: Saves the resultant models (or not) for the L2 (Fair K) model, which produces large-ish files (100s MBs)

Dependencies (shouldn't really matter as long as it's up-to-date Python >3.5):
Python 3.5.5
NumPy 1.14.3
Pandas 0.23.0
Scikit-learn 0.19.1
PyStan 2.17.1.0
StatsModels 0.9.0
"""

import pystan
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import pickle
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
from argparse import ArgumentParser, ArgumentTypeError
from pathlib import Path
import os

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ArgumentTypeError('Boolean value expected.')

parser = ArgumentParser(description='Train CF model.')
parser.add_argument('-do_l2', type=str2bool, nargs='?', const=True, default=False, help="Perform L2 Train/Test (Warning: TAKES TIME)")
parser.add_argument('-save_l2', type=str2bool, nargs='?', const=True, default=False, help="Save L2 Train/Test Models (Warning: LARGE FILES)")
parser.add_argument('--model_dir', '-m', default='models', help='directory to save l2 models')
# TODO: change default to correct CSV
parser.add_argument('--data_dir', '-d', default='Data/covid_cleaned.csv', help='CSV with cleaned data.')
parser.add_argument('--a_cols', '-a', nargs='*', default=['oxygen_therapy'], help='Columns of protected classes')
parser.add_argument('--y_col', '-y', default='covid_severity', help='Column of Y output')
parser.add_argument('--random_state', 'r', default=1234, help='Random seed for TT split')
parser.add_argument('--test_size', 't', default=0.2, help='Test proportion')
# parser.add_argument('--previous_k', '-k', default=4, type=int,
#                     help='k-previous steps to use for prediction.')
# parser.add_argument('--ablation', '-a', default=-1, type=int,
#                     help='Ablates a feature for training')

# parser.add_argument('--lr', '-l', default=1e-01, type=float,
#                     help='Learning Rate')


args = parser.parse_args()

# TODO: add logistic regression
# wrapper class for statsmodels linear regression (more stable than SKLearn)
class SM_LinearRegression():
    def __init__(self):
        pass
        
    def fit(self, X, y):
        N = X.shape[0]
        self.LRFit = sm.OLS(y, np.hstack([X,np.ones(N).reshape(-1,1)]),hasconst=True).fit()
        
    def predict(self,X):
        N = X.shape[0]
        return self.LRFit.predict(np.hstack([X,np.ones(N).reshape(-1,1)]))

# This produces a dictionary for the stan models, so change it to our covid variables
# k is length of protected features
# N is number of samples
# a is protected features
# function to convert to a dictionary for use with STAN train-time model
def get_pystan_train_dic(pandas_df, sense_cols):
    dic_out = {}
    dic_out['N'] = len(pandas_df)
    dic_out['K'] = len(sense_cols)
    dic_out['a'] = np.array(pandas_df[sense_cols])
    if dic_out['a'].ndim < 2:
        dic_out['a'] = np.atleast_2d(dic_out['a']).reshape(-1, 1)

    vars = list(set(pandas_df.columns.tolist()) - set(sense_cols))
    for var in vars:
        dic_out[var] = list(pandas_df[var])
    # dic_out['ugpa'] = list(pandas_df['UGPA'])
    # dic_out['lsat'] = list(pandas_df['LSAT'].astype(int))
    # dic_out['zfya'] = list(pandas_df['ZFYA'])
    return dic_out, vars

# TODO: Change this to OUR SEM
# function to convert to a dictionary for use with STAN test-time model
def get_pystan_test_dic(fit_extract, test_dic):
    dic_out = {}
    for key in fit_extract.keys():
        if key not in ['sigma_g_Sq', 'u', 'eta_a_zfya', 'eta_u_zfya', 'lp__']:
            dic_out[key] = np.mean(fit_extract[key], axis = 0)
    
    need_list = ['N', 'K', 'a', 'ugpa', 'lsat']
    for data in need_list:
        dic_out[data] = test_dic[data]

    return dic_out

# TODO: Safyan take a look at this and either get rid of it, change it for our needs or parametrize it as needed
# I assume you've already done most of this in preprocessing the data, i.e. sklearn multilabel binarizer for symptoms,
# binary / ordinal for severity etc. etc.
# Preprocess data for all subsequent experiments
def get_data_preprocess(sense_cols=['oxygen_therapy'],
                        random_state = 1234,
                        test_size = 0.2):
    """Loads pandas Dataframe, returns test and train dictionaries.
    """
    law_data = pd.read_csv(args.data_dir, index_col=0) 
    # law_data = pd.get_dummies(law_data,columns=['race'],prefix='',prefix_sep='')

    # law_data['male'] = law_data['sex'].map(lambda z: 1 if z == 2 else 0)
    # law_data['female'] = law_data['sex'].map(lambda z: 1 if z == 1 else 0)
    
    # law_data['LSAT'] = law_data['LSAT'].apply(lambda x: int(np.round(x)))

    # law_data = law_data.drop(axis=1, columns=['sex'])

    # sense_cols = ['Amerindian','Asian','Black','Hispanic','Mexican','Other','Puertorican','White','male','female']

    law_train,law_test = train_test_split(law_data, random_state=random_state, test_size=test_size)

    law_train_dic, vars = get_pystan_train_dic(law_train, sense_cols)
    law_test_dic, _ = get_pystan_train_dic(law_test, sense_cols)

    return law_train_dic, law_test_dic, vars

# Get the Unfair Model predictions
def Unfair_Model_Replication(law_train_dic, law_test_dic, X_vars, y_var):
    lr_unfair = SM_LinearRegression()

    X_train = np.hstack([law_train_dic['a']]+[np.array(law_train_dic[X_var]).reshape(-1,1) for X_var in X_vars])
    X_test = np.hstack([law_test_dic['a']]+[np.array(law_test_dic[X_var]).reshape(-1,1) for X_var in X_vars])
    lr_unfair.fit(X_train,law_train_dic[y_var])
    
    preds = lr_unfair.predict(X_test)
    
    # Return Results:
    return preds

# Get the FTU Model predictions
def FTU_Model_Replication(law_train_dic, law_test_dic, X_vars, y_var):
    lr_unaware = SM_LinearRegression()

    X_train = np.hstack([np.array(law_train_dic[X_var]).reshape(-1,1) for X_var in X_vars])
    X_test = np.hstack([np.array(law_test_dic[X_var]).reshape(-1,1) for X_var in X_vars])
    lr_unaware.fit(X_train,law_train_dic[y_var]); 

    preds = lr_unaware.predict(X_test)
    
    # Return Results:
    return preds

# Get the Fair K/L2 Model predictions
def L2_Model_Replication(law_train_dic, law_test_dic, X_vars, y_var, save_models = False):

    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)

    check_fit = Path(os.path.join(args.model_dir, "model_fit.pkl"))

    if check_fit.is_file():
        print('File Found: Loading Fitted Training Model Samples...')
        if save_models:
            print('No models will be trained or saved')
        with open(check_fit, "rb") as f:
            post_samps = pickle.load(f)
    else:
        print('File Not Found: Fitting Training Model...\n')
        # Compile Model
        model = pystan.StanModel(file = os.path.join('src', 'covid_l2_train.stan'))
        print('Finished compiling model!')
        # Commence the training of the model to infer weights (500 warmup, 500 actual)
        fit = model.sampling(data = law_train_dic, iter=1000, chains = 1)
        post_samps = fit.extract()
        # Save parameter posterior samples if specified
        if save_models:
            with open(check_fit, "wb") as f:
                pickle.dump(post_samps, f, protocol=-1)
            print('Saved fitted model!')

    # Retreive posterior weight samples and take means
    law_train_dic_final = get_pystan_test_dic(post_samps, law_train_dic)
    law_test_dic_final = get_pystan_test_dic(post_samps, law_test_dic)

    check_train = Path(os.path.join(args.model_dir, "model_fit_train.pkl"))
    
    if check_train.is_file():
        # load posterior training samples from file
        print('File Found: Loading Test Model with Train Data...')
        if save_models:
            print('No models will be trained or saved')
        with open(check_train, "rb") as f:
            fit_train_samps = pickle.load(f)
    else:
        # Obtain posterior training samples from scratch
        print('File Not Found: Fitting Test Model with Train Data...\n')
        model_train = pystan.StanModel(file = os.path.join('src', 'covid_l2_only_u.stan'))
        fit_train = model_train.sampling(data = law_train_dic_final, iter=2000, chains = 1)
        fit_train_samps = fit_train.extract()
        if save_models:
            with open(check_train, "wb") as f:
                pickle.dump(fit_train_samps, f, protocol=-1)
            print('Saved train samples!')
    
    train_K = np.mean(fit_train_samps['u'],axis=0).reshape(-1,1)

    check_test = Path(os.path.join(args.model_dir, "model_fit_test.pkl"))

    if check_test.is_file():
        # load posterior test samples from file
        print('File Found: Loading Test Model with Test Data...')
        if save_models:
            print('No models will be trained or saved')
        with open(check_test, "rb") as f:
            fit_test_samps = pickle.load(f)
    else:
        # Obtain posterior test samples from scratch
        print('File Not Found: Fitting Test Model with Test Data...\n')
        model_test = pystan.StanModel(file = os.path.join('src', 'covid_l2_only_u.stan'))
        fit_test = model_test.sampling(data = law_test_dic_final, iter=2000, chains = 1)
        fit_test_samps = fit_test.extract()
        if save_models:
            with open(check_test, "wb") as f:
                pickle.dump(fit_test_samps, f, protocol=-1)
            print('Saved test samples!')
    
    test_K = np.mean(fit_test_samps['u'],axis=0).reshape(-1,1)

    # Train L2 Regression
    smlr_L2 = SM_LinearRegression()
    smlr_L2.fit(train_K,law_train_dic[y_var])

    # Predict on test
    preds = smlr_L2.predict(test_K)

    # Return Results:
    return preds

# TODO: Change this to OUR SEM
# Get the Fair All/L3 Model Predictions
def L3_Model_Replication(law_train_dic, law_test_dic, X_vars, y_var,):

    # abduct the epsilon_G values
    linear_eps_g = SM_LinearRegression()
    linear_eps_g.fit(np.vstack((law_train_dic['a'],law_test_dic['a'])),law_train_dic['ugpa']+law_test_dic['ugpa'])
    eps_g_train = law_train_dic['ugpa'] - linear_eps_g.predict(law_train_dic['a'])
    eps_g_test = law_test_dic['ugpa'] - linear_eps_g.predict(law_test_dic['a'])
    
    # abduct the epsilon_L values
    linear_eps_l = SM_LinearRegression()
    linear_eps_l.fit(np.vstack((law_train_dic['a'],law_test_dic['a'])),law_train_dic['lsat']+law_test_dic['lsat'])
    eps_l_train = law_train_dic['lsat'] - linear_eps_l.predict(law_train_dic['a'])
    eps_l_test = law_test_dic['lsat'] - linear_eps_l.predict(law_test_dic['a'])

    # predict on target using abducted latents
    smlr_L3 = SM_LinearRegression()
    smlr_L3.fit(np.hstack((eps_g_train.reshape(-1,1),eps_l_train.reshape(-1,1))),law_train_dic['zfya'])

    # predict on test epsilons
    preds = smlr_L3.predict(np.hstack((eps_g_test.reshape(-1,1),eps_l_test.reshape(-1,1))))

    # Return Results:
    return preds

def main():

    # Get the data, split train/test
    law_train_dic, law_test_dic, vars = get_data_preprocess(args.a_cols, args.random_state, args.test_size)

    X_vars = vars.copy()
    X_vars.remove( args.y_col)
    # Get the predictions
    unfair_preds = Unfair_Model_Replication(law_train_dic, law_test_dic, X_vars, args.y_col)
    ftu_preds = FTU_Model_Replication(law_train_dic, law_test_dic, X_vars, args.y_col)
    if args.do_l2:
        l2_preds = L2_Model_Replication(law_train_dic, law_test_dic, X_vars, args.y_col, args.save_l2)
    l3_preds = L3_Model_Replication(law_train_dic, law_test_dic, X_vars, args.y_col)

    # Print the predictions
    print('Unfair RMSE: \t\t\t%.3f' % np.sqrt(mean_squared_error(unfair_preds,law_test_dic['zfya'])))
    print('FTU RMSE: \t\t\t%.3f' % np.sqrt(mean_squared_error(ftu_preds,law_test_dic['zfya'])))
    if args.do_l2:
        print('Level 2 (Fair K) RMSE: \t\t%.3f' % np.sqrt(mean_squared_error(l2_preds,law_test_dic['zfya'])))
    print('Level 3 (Fair Add) RMSE: \t%.3f' % np.sqrt(mean_squared_error(l3_preds,law_test_dic['zfya'])))

if __name__ == '__main__':
    main()