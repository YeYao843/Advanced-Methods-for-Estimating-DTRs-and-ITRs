#Please run Mlearn_KW.py first to get the MatchOLearn_KW class
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import ensemble
from Mlearn_KW import MatchOLearn_KW
from joblib import Parallel, delayed
import multiprocessing
from collections import Counter
from sklearn.preprocessing import scale
import warnings
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")
### Parameters ###
strat = 'benefit'
ran_p = 3; obs_p = 3;
unknown = 'unknown'
whole = 'sub'
ran_size = 500

if whole == 'sub':
    crit_point1 = -0.4;
    crit_point2 = 0.4;
else:
    crit_point1 = -9999;
    crit_point2 = 9999;

if unknown == 'unknown':
    unobs_latent = obs_p -1;
elif unknown == 'known':
    unobs_latent = obs_p;

my_size = 90
### Parameters ###  (p: number of features, sample_size: number of samples)
def phi_eta(Frame_X):

    phi = 0.5 * (Frame_X[:, 0] + Frame_X[:, 1] > 0) -  (Frame_X[:, 0] + Frame_X[:, 1] <= 0) * (0.5 + 0.5 * (Frame_X[:, 1] <= - 0.5)) + (Frame_X[:, 2]**2  - Frame_X[:,1]**2)
    eta = Frame_X[:, 0] - 0.5 * Frame_X[:, 1]

    return phi,eta

### Generate data ###
def generate_data(sample_size,p,seed=0,obs=True,prob=0.5):
    r = np.random.RandomState(seed);
    Val_X = r.standard_normal((sample_size,p));

    try:
        if obs == True:
            z = 1 + 2 * Val_X[:,0] + Val_X[:,1];
            pr = 1 / (1+np.exp(-z));
        elif obs == False:
            pr = np.repeat(prob,sample_size);
        else:
            raise ValueError;
    except ValueError:
        print("Invalid setting!")

    r2 = np.random.RandomState(seed+1)
    Val_A = r2.binomial(1,pr) * 2 - 1


    r3 = np.random.RandomState(seed+2)
    Val_N = r3.standard_normal((sample_size)) / 2

    phi,eta = phi_eta(Val_X)

    Val_Q = eta + phi * Val_A + Val_N
    Val_A_True = np.sign(phi)
    Potential = eta + phi * Val_A_True + Val_N

    return Val_Q, Potential, Val_X, Val_N, Val_A, Val_A_True

Cs = np.array([0.01,0.1, 0.5, 1, 2, 5, 10,20,50])
gammas = np.array([0.0001, 0.001, 0.01, 0.1, 1, 10,20,50])
kernels = ['rbf']

### Learn observational data ###
def learn_obs(X,Q,A, seed=0):
    max_est = -9999


    for C in Cs:
        for gamma in gammas:
            for kernel in kernels:

                model_obs = MatchOLearn_KW(C=C, gamma=gamma, kernel=kernel,metric='mahalanobis',propensity='obs')
                model_obs.fit(X,Q,A, match=np.array(range(X.shape[1] )), learn=np.array(range(X.shape[1])), bandC=1, size=my_size)
                est_obs = model_obs.estimate(X, Q, A, learn=np.array(range(X.shape[1])), normalize=True)

                if est_obs > (max_est + 1e-5):
                    max_est = est_obs
                    param_est = (C, gamma, kernel)


    model_obs1 = MatchOLearn_KW(C=param_est[0], gamma=param_est[1], kernel=param_est[2], metric='mahalanobis',propensity='obs')
    model_obs1.fit(X, Q, A, match=np.array(range(X.shape[1] )),learn=np.array(range(X.shape[1])),bandC=1, size=my_size)

    print ('OBS Learning Completed!')
    pred_obs_O1 = model_obs1.predict(X)

    lm = LogisticRegression()
    lm.fit(X,A)
    prob = lm.predict_proba(X)[:,1]

    ### Predict optimal outcome
    X_optimal = X[np.where(pred_obs_O1==A)]
    A_optimal = A[np.where(pred_obs_O1==A)]
    Q_optimal = Q[np.where(pred_obs_O1==A)]
    PSwt = 1/prob[np.where(pred_obs_O1==A)]

    model_out = ensemble.RandomForestRegressor(n_estimators=50,max_features='sqrt',random_state=seed)
    model_out.fit(X_optimal,Q_optimal,sample_weight=PSwt)

    ### Predict nonoptimal outcome
    X_nonoptimal = X[np.where(pred_obs_O1 != A)]
    A_nonoptimal = A[np.where(pred_obs_O1 != A)]
    Q_nonoptimal = Q[np.where(pred_obs_O1 != A)]
    PSwt_non = 1 / prob[np.where(pred_obs_O1 != A)]

    model_out2 = ensemble.RandomForestRegressor(n_estimators=50, max_features='sqrt', random_state=seed)
    model_out2.fit(X_nonoptimal, Q_nonoptimal, sample_weight=PSwt_non)


    ### Predict optimal TRT
    model_trt = ensemble.RandomForestClassifier(n_estimators=50,max_features='sqrt',random_state=seed)
    model_trt.fit(X_optimal,A_optimal,sample_weight=PSwt)

    return model_out, model_out2, model_trt



## Predict Obs-related scores in RCT data
def RCT_scores(RCT_X, model_out,model_out2,model_trt):

    RCT_popt_out1 = model_out.predict(RCT_X);
    RCT_popt_out2 = model_out2.predict(RCT_X);
    RCT_benefit = RCT_popt_out1 - RCT_popt_out2

    RCT_prob = 1 - model_trt.predict_proba(RCT_X)[:, 0];
    return RCT_benefit, RCT_prob





########## Simulation ###########
import time
start_time = time.time()
num_cores = multiprocessing.cpu_count()
# generate test data
sim_Q_TEST, sim_Potential_TEST, sim_X_TEST, sim_N_TEST, sim_A_TEST, sim_A_True_TEST = generate_data(100000, ran_p, seed=9999, obs=False)
subgroup = np.where((sim_X_TEST[:, 0] >= crit_point1) & (sim_X_TEST[:, 0] <= crit_point2))
np.random.seed(111)
ran_index = np.random.choice(len(subgroup[0]), 10000);

sim_Q_TEST = sim_Q_TEST[subgroup][ran_index]
sim_Potential_TEST = sim_Potential_TEST[subgroup][ran_index]
sim_A_TEST = sim_A_TEST[subgroup][ran_index]
sim_N_TEST = sim_N_TEST[subgroup][ran_index]
sim_X_TEST = sim_X_TEST[subgroup][ran_index]


#########################
def Learning(t):
    from sklearn.model_selection import KFold
    K = 2
    kf = KFold(n_splits=K, random_state=t, shuffle=True)
    # generate train data
    sim_Q_obs, sim_Potential_obs, sim_X_obs, sim_N_obs, sim_A_obs, sim_A_True_obs = generate_data(sample_size, obs_p, seed=t)
    sim_Q_ran, sim_Potential_ran, sim_X_ran, sim_N_ran, sim_A_ran, sim_A_True_ran = generate_data(10000, ran_p, seed=t,obs=False)

    ## inclusion criterion

    np.random.seed(t)
    subgroup = np.where((sim_X_ran[:, 0] >= crit_point1) & (sim_X_ran[:, 0] <= crit_point2))
    ran_index = np.random.choice(len(subgroup[0]),ran_size);
    sim_Q_ran = sim_Q_ran[subgroup][ran_index]
    sim_Potential_ran = sim_Potential_ran[subgroup][ran_index]
    sim_A_ran = sim_A_ran[subgroup][ran_index]
    sim_N_ran = sim_N_ran[subgroup][ran_index]
    sim_X_ran = sim_X_ran[subgroup][ran_index]

    ## Prognositic score
    from sklearn import linear_model
    X_1 = sim_X_ran[np.where(sim_A_ran == -1)]
    Q_1 = sim_Q_ran[np.where(sim_A_ran == -1)]
    lin = linear_model.LinearRegression()
    lin.fit(X_1, Q_1)
    prog = lin.predict(sim_X_ran)
    prog = scale(prog)


    ####### strategy 3:
    My_model_out1, My_model_out2, My_model_trt1 = learn_obs(sim_X_obs[:,0:(unobs_latent)], sim_Q_obs,sim_A_obs, seed=t*t);
    My_RCT_benefit, My_RCT_prob1 = RCT_scores(sim_X_ran[:, 0:(unobs_latent)], My_model_out1, My_model_out2, My_model_trt1);
    np.random.seed(t)
    My_RCT_prob1 = My_RCT_prob1 + np.random.uniform(-1e-10, 1e-10, size=ran_size)

    if strat == 'prob':
        My_RCT_scores1 = np.column_stack( (scale(My_RCT_benefit),prog,scale(My_RCT_prob1)) );
    elif strat == 'benefit':
        My_RCT_scores1 = np.column_stack((scale(My_RCT_prob1), prog, scale(My_RCT_benefit)));

    cutoff = 0
    My_RCT_X3 = np.append(sim_X_ran[:, 0:(unobs_latent)], My_RCT_scores1, axis=1)


    My_RCT_X3_hi = My_RCT_X3[np.where(My_RCT_scores1[:, 2] >= cutoff)]
    sim_Q_ran_hi = sim_Q_ran[np.where(My_RCT_scores1[:, 2] >= cutoff)]
    sim_A_ran_hi = sim_A_ran[np.where(My_RCT_scores1[:, 2] >= cutoff)]

    My_RCT_X3_lo = My_RCT_X3[np.where(My_RCT_scores1[:, 2] < cutoff)]
    sim_Q_ran_lo = sim_Q_ran[np.where(My_RCT_scores1[:, 2] < cutoff)]
    sim_A_ran_lo = sim_A_ran[np.where(My_RCT_scores1[:, 2] < cutoff)]

    My_RCT_X3_hi = My_RCT_X3_hi[:, :-1]
    My_RCT_X3_lo = My_RCT_X3_lo[:, :-1]


    TEST_RCT_benefit, TEST_RCT_prob1 = RCT_scores(sim_X_TEST[:, 0:(unobs_latent)], My_model_out1, My_model_out2, My_model_trt1);
    np.random.seed(t*t)
    TEST_RCT_prob1 = TEST_RCT_prob1 + np.random.uniform(-1e-10, 1e-10, size=10000)
    if strat == 'prob':
        TEST_RCT_scores1 = np.column_stack(  (scale(TEST_RCT_benefit), scale(TEST_RCT_prob1)) );
    elif strat == 'benefit':
        TEST_RCT_scores1 = np.column_stack((scale(TEST_RCT_prob1), scale(TEST_RCT_benefit)));
    TEST_RCT_X3 = np.append(sim_X_TEST[:, 0:(unobs_latent)], TEST_RCT_scores1, axis=1)


    TEST_RCT_X3_hi = TEST_RCT_X3[np.where(TEST_RCT_scores1[:, 1] >= cutoff)]
    sim_N_TEST_hi = sim_N_TEST[np.where(TEST_RCT_scores1[:, 1] >= cutoff)]
    sim_X_TEST_hi = sim_X_TEST[np.where(TEST_RCT_scores1[:, 1] >= cutoff)]

    TEST_RCT_X3_lo = TEST_RCT_X3[np.where(TEST_RCT_scores1[:, 1] < cutoff)]
    sim_N_TEST_lo = sim_N_TEST[np.where(TEST_RCT_scores1[:, 1] < cutoff)]
    sim_X_TEST_lo = sim_X_TEST[np.where(TEST_RCT_scores1[:, 1] < cutoff)]

    TEST_RCT_X3_hi = TEST_RCT_X3_hi[:, :-2]
    TEST_RCT_X3_lo = TEST_RCT_X3_lo[:, :-2]

    ### HIGH GROUP
    MAX_EST3 = -9999
    for C in Cs:
        for gamma in gammas:
            for kernel in kernels:
                cv_res = []
                for train_index, test_index in kf.split(My_RCT_X3_hi):
                    # print("TRAIN:", train_index, "TEST:", test_index)
                    X_train, X_test = My_RCT_X3_hi[train_index], My_RCT_X3_hi[test_index]
                    Q_train, Q_test = sim_Q_ran_hi[train_index], sim_Q_ran_hi[test_index]
                    A_train, A_test = sim_A_ran_hi[train_index], sim_A_ran_hi[test_index]
                    model_all = MatchOLearn_KW(C=C, gamma=gamma, kernel=kernel, metric='mahalanobis', propensity=0.5);
                    model_all.fit(X_train, Q_train, A_train,match=np.array([X_train.shape[1] - 2, X_train.shape[1] - 1]),
                                  learn=np.array(range(X_train.shape[1] - 2)), bandC=1,size=my_size)
                    est_all = model_all.estimate(X_test, Q_test, A_test, learn=np.array(range(X_test.shape[1] - 2)), normalize=True)

                    cv_res.append(est_all)
                cv_res_all = np.mean(cv_res)

                if cv_res_all > (MAX_EST3 + 1e-5):
                    MAX_EST3 = cv_res_all
                    PARAM_EST3 = (C, gamma, kernel)

    best_model_s3_hi = MatchOLearn_KW(C=PARAM_EST3[0], gamma=PARAM_EST3[1], kernel=PARAM_EST3[2], metric='mahalanobis',propensity=0.5);
    best_model_s3_hi.fit(My_RCT_X3_hi, sim_Q_ran_hi, sim_A_ran_hi,match=np.array([ My_RCT_X3_hi.shape[1] - 2, My_RCT_X3_hi.shape[1] - 1]),
                      learn=np.array(range(My_RCT_X3_hi.shape[1] - 2)), bandC=1, size=my_size)

    TEST_Pred_ML_s3_hi = best_model_s3_hi.predict(TEST_RCT_X3_hi)
    phi_TEST3_hi , eta_TEST3_hi = phi_eta(sim_X_TEST_hi)

    TEST_Potential_ML_s3_hi = np.mean(eta_TEST3_hi + phi_TEST3_hi * TEST_Pred_ML_s3_hi + sim_N_TEST_hi)
    Ben3_hi = np.mean(phi_TEST3_hi * TEST_Pred_ML_s3_hi) * len(TEST_RCT_X3_hi) / 10000

    TEST_Potential_ML_s3_hi = TEST_Potential_ML_s3_hi * len(TEST_RCT_X3_hi) / 10000

    ### LOW GROUP
    MAX_EST4 = -9999
    for C in Cs:
        for gamma in gammas:
            for kernel in kernels:
                cv_res = []
                for train_index, test_index in kf.split(My_RCT_X3_lo):
                    # print("TRAIN:", train_index, "TEST:", test_index)
                    X_train, X_test = My_RCT_X3_lo[train_index], My_RCT_X3_lo[test_index]
                    Q_train, Q_test = sim_Q_ran_lo[train_index], sim_Q_ran_lo[test_index]
                    A_train, A_test = sim_A_ran_lo[train_index], sim_A_ran_lo[test_index]
                    model_all = MatchOLearn_KW(C=C, gamma=gamma, kernel=kernel, metric='mahalanobis', propensity=0.5);
                    model_all.fit(X_train, Q_train, A_train,match=np.array([X_train.shape[1] - 2, X_train.shape[1] - 1]),
                                  learn=np.array(range(X_train.shape[1] - 2)), bandC=1, size=my_size)
                    est_all = model_all.estimate(X_test, Q_test, A_test, learn=np.array(range(X_test.shape[1] - 2)),
                                                 normalize=True)

                    cv_res.append(est_all)
                cv_res_all = np.mean(cv_res)

                if cv_res_all > (MAX_EST4 + 1e-5):
                    MAX_EST4 = cv_res_all
                    PARAM_EST4 = (C, gamma, kernel)

    best_model_s3_lo = MatchOLearn_KW(C=PARAM_EST4[0], gamma=PARAM_EST4[1], kernel=PARAM_EST4[2], metric='mahalanobis', propensity=0.5);
    best_model_s3_lo.fit(My_RCT_X3_lo, sim_Q_ran_lo, sim_A_ran_lo,match=np.array([My_RCT_X3_lo.shape[1] - 2, My_RCT_X3_lo.shape[1] - 1]),
                         learn=np.array(range(My_RCT_X3_lo.shape[1] - 2)), bandC=1, size=my_size)

    TEST_Pred_ML_s3_lo = best_model_s3_lo.predict(TEST_RCT_X3_lo)
    phi_TEST3_lo, eta_TEST3_lo = phi_eta(sim_X_TEST_lo)

    TEST_Potential_ML_s3_lo = np.mean(eta_TEST3_lo + phi_TEST3_lo * TEST_Pred_ML_s3_lo + sim_N_TEST_lo)
    TEST_Potential_ML_s3_lo = TEST_Potential_ML_s3_lo * len(TEST_RCT_X3_lo) / 10000
    Ben3_lo = np.mean(phi_TEST3_lo * TEST_Pred_ML_s3_lo) * len(TEST_RCT_X3_lo) / 10000

    TEST_Potential_ML_s3 = TEST_Potential_ML_s3_hi + TEST_Potential_ML_s3_lo
    Ben3 = Ben3_hi + Ben3_lo


    ####### strategy 1:
    My_model_out1, My_model_out2, My_model_trt1 = learn_obs(sim_X_obs[:,0:(unobs_latent)], sim_Q_obs,sim_A_obs, seed=t*t);
    My_RCT_benefit, My_RCT_prob1 = RCT_scores(sim_X_ran[:, 0:(unobs_latent)], My_model_out1, My_model_out2, My_model_trt1);

    My_RCT_scores0 =np.column_stack( ( prog, scale(My_RCT_benefit),scale(My_RCT_prob1)) );
    My_RCT_X1 = np.append(sim_X_ran[:, 0:(unobs_latent)], My_RCT_scores0, axis=1)
    My_RCT_X1 = My_RCT_X1[:,:-2]
    TEST_RCT_X1 = sim_X_TEST[:, 0:(unobs_latent)]

    MAX_EST1 = -9999
    for C in Cs:
        for gamma in gammas:
            for kernel in kernels:
                cv_res = []
                for train_index, test_index in kf.split(My_RCT_X1):
                    # print("TRAIN:", train_index, "TEST:", test_index)
                    X_train, X_test = My_RCT_X1[train_index], My_RCT_X1[test_index]
                    Q_train, Q_test = sim_Q_ran[train_index], sim_Q_ran[test_index]
                    A_train, A_test = sim_A_ran[train_index], sim_A_ran[test_index]
                    model_all = MatchOLearn_KW(C=C, gamma=gamma, kernel=kernel, metric='mahalanobis', propensity=0.5);
                    model_all.fit(X_train, Q_train, A_train,match=np.array([X_train.shape[1] - 1]),
                                  learn=np.array(range(X_train.shape[1] - 1)), bandC=1,size=my_size)
                    est_all = model_all.estimate(X_test, Q_test, A_test, learn=np.array(range(X_test.shape[1] - 1)),
                                                 normalize=True)

                    cv_res.append(est_all)
                cv_res_all = np.mean(cv_res)

                if cv_res_all > (MAX_EST1 + 1e-5):
                    MAX_EST1 = cv_res_all
                    PARAM_EST1 = (C, gamma, kernel)

    best_model_s1 = MatchOLearn_KW(C=PARAM_EST1[0], gamma=PARAM_EST1[1], kernel=PARAM_EST1[2], metric='mahalanobis',
                                   propensity=0.5);
    best_model_s1.fit(My_RCT_X1, sim_Q_ran, sim_A_ran, match=np.array([My_RCT_X1.shape[1] - 1]),
                      learn=np.array(range(My_RCT_X1.shape[1] - 1)), bandC=1,size=my_size)
    TEST_Pred_ML_s1 = best_model_s1.predict(TEST_RCT_X1)

    phi_TEST1 , eta_TEST1 = phi_eta(sim_X_TEST)
    TEST_Potential_ML_s1 = np.mean(eta_TEST1 + phi_TEST1 * TEST_Pred_ML_s1 + sim_N_TEST)
    Ben1 = 2 * np.mean(phi_TEST1 * TEST_Pred_ML_s1)

    ####### Strategy 2: Add extra features:

    My_RCT_X2 = np.append(sim_X_ran[:, 0:(unobs_latent)], My_RCT_scores0, axis=1)
    TEST_RCT_X1 = sim_X_TEST[:, 0:(unobs_latent)]

    MAX_EST2 = -9999
    for C in Cs:
        for gamma in gammas:
            for kernel in kernels:
                cv_res = []
                for train_index, test_index in kf.split(My_RCT_X2):
                    # print("TRAIN:", train_index, "TEST:", test_index)
                    X_train, X_test = My_RCT_X2[train_index], My_RCT_X2[test_index]
                    Q_train, Q_test = sim_Q_ran[train_index], sim_Q_ran[test_index]
                    A_train, A_test = sim_A_ran[train_index], sim_A_ran[test_index]
                    model_all = MatchOLearn_KW(C=C, gamma=gamma, kernel=kernel, metric='mahalanobis', propensity=0.5);
                    model_all.fit(X_train, Q_train, A_train, match=np.array([X_train.shape[1] - 3,X_train.shape[1] - 2,X_train.shape[1] - 1]),
                                  learn=np.array(range(X_train.shape[1] - 3)), bandC=1, size=my_size)
                    est_all = model_all.estimate(X_test, Q_test, A_test, learn=np.array(range(X_test.shape[1] - 3)),normalize=True)

                    cv_res.append(est_all)
                cv_res_all = np.mean(cv_res)

                if cv_res_all > (MAX_EST2 + 1e-5):
                    MAX_EST2 = cv_res_all
                    PARAM_EST2 = (C, gamma, kernel)

    best_model_s2 = MatchOLearn_KW(C=PARAM_EST2[0], gamma=PARAM_EST2[1], kernel=PARAM_EST2[2], metric='mahalanobis',propensity=0.5);
    best_model_s2.fit(My_RCT_X2, sim_Q_ran, sim_A_ran, match=np.array([My_RCT_X2.shape[1] - 3, My_RCT_X2.shape[1] - 2,My_RCT_X2.shape[1] - 1]),
                      learn=np.array(range(My_RCT_X2.shape[1] - 3)), bandC=1, size=my_size)
    TEST_Pred_ML_s2 = best_model_s2.predict(TEST_RCT_X1)

    TEST_Potential_ML_s2 = np.mean(eta_TEST1 + phi_TEST1 * TEST_Pred_ML_s2 + sim_N_TEST)
    Ben2 = 2 * np.mean(phi_TEST1 * TEST_Pred_ML_s2)

    print('iteration_time: ', t, ' results: ', np.mean(sim_Potential_ran)),
    print("strategy 1: ", MAX_EST1, TEST_Potential_ML_s1, PARAM_EST1, Counter(TEST_Pred_ML_s1)),
    print("strategy 2: ", MAX_EST2, TEST_Potential_ML_s2, PARAM_EST2, Counter(TEST_Pred_ML_s2)),
    print("strategy 3: " , TEST_Potential_ML_s3_hi, TEST_Potential_ML_s3_lo, Counter(TEST_Pred_ML_s3_hi),Counter(TEST_Pred_ML_s3_lo), len(TEST_RCT_X3_hi),len(TEST_RCT_X3_lo)),
    print("----------------------------------------------------------------------------------")


    return TEST_Potential_ML_s1, TEST_Potential_ML_s2, TEST_Potential_ML_s3, Ben1, Ben2, Ben3

results = {}
iters = 100
sim_size = [1000]


for sample_size in sim_size:
    results[str(sample_size)] = Parallel(n_jobs= (num_cores-1))(delayed(Learning)(t) for t in range(iters))

cost = time.time() - start_time
print('Time Consumed: ', cost)



print(np.median(np.array(results[str(sample_size)])[:,0]),np.mean(np.array(results[str(sample_size)])[:,0]), np.max(np.array(results[str(sample_size)])[:,0]), np.std(np.array(results[str(sample_size)])[:,0])),
print(np.median(np.array(results[str(sample_size)])[:,1]),np.mean(np.array(results[str(sample_size)])[:,1]), np.max(np.array(results[str(sample_size)])[:,1]), np.std(np.array(results[str(sample_size)])[:,1]))
print(np.median(np.array(results[str(sample_size)])[:,2]),np.mean(np.array(results[str(sample_size)])[:,2]), np.max(np.array(results[str(sample_size)])[:,2]), np.std(np.array(results[str(sample_size)])[:,2]))

import pandas as pd
res = pd.DataFrame(results[str(sample_size)],columns=['s1','s2','s3','s1_ben','s2_ben','s3_ben'])
res.to_csv('Sim_M_s1_' + whole+ '_' + unknown +'_' + str(ran_size)+ '_'+ str(sim_size[0]) + '_' + strat +'_res_rbf.csv',index=None)
