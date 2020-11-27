"""
.. module:: prediction
    :synopsis Predicting drug response using kernel features

.. moduleauthor:: Nok <suphavilaic@gis.a-star.edu.sg>

"""

import pandas as pd
import numpy as np
import os, pickle, time

import tensorflow as tf
from tensorflow.python.framework import ops
import tensorflow.python.util.deprecation as deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

def load_model(model_fname):

    """Load a pre-trained model

	:param model_fname: File name of the model
	:return: model_dict contains model information

    """

    model_dict = pickle.load(open(model_fname, 'br'))

    return model_dict


# def get_model_param(pg_space):

    """Get model paramters
    """

# def get_training_info(pg_space):

    """Get training information
    """

def predict_from_model(model_dict, test_kernel_df, model_spec_name='cadrres-wo-sample-bias'):

    """Make a prediction of testing samples. Only for the model without sample bias.
    """

    # TODO: add other model types and update the scrip accordingly
    if model_spec_name not in ['cadrres-wo-sample-bias', 'cadrres-wo-sample-bias-weight']:
        return None

    sample_list = list(test_kernel_df.index)
    
    # Read drug list from model_dict
    drug_list = model_dict['drug_list']
    kernel_sample_list = model_dict['kernel_sample_list']

    # Prepare input
    X = np.matrix(test_kernel_df[kernel_sample_list])

    # Make a prediction
    b_q = model_dict['b_Q']
    WP = model_dict['W_P']
    WQ = model_dict['W_Q']

    n_dim = WP.shape[1]

    pred = b_q.T + (X * WP) * WQ.T
    pred = pred * -1 # convert sensitivity score to IC50
    pred_df = pd.DataFrame(pred, sample_list, drug_list)

    # Projections
    P_test = X * WP
    P_test_df = pd.DataFrame(P_test, index=sample_list, columns=range(1,n_dim+1))  
    
    return pred_df, P_test_df

def calculate_baseline_prediction(obs_resp_df, train_sample_list, drug_list, test_sample_list):

    """Calculate baseline prediction, i.e., for each drug, predict the average response.
    """

    repeated_val = np.repeat([obs_resp_df.loc[train_sample_list, drug_list].mean().values], len(test_sample_list), axis=0)
    return pd.DataFrame(repeated_val, index=test_sample_list, columns=drug_list)


##########################
##### Model training #####
##########################

##### Utility functions #####

def create_placeholders(n_x_features, n_y_features, sample_weight=False):

    """
    Create placeholders for model inputs
    """

    # gene expression
    X = tf.placeholder(tf.float32, [None, n_x_features])
    # drug response
    Y = tf.placeholder(tf.float32, [None, n_y_features])
    if sample_weight:
        # for logistic weight based on maximum drug dosage
        O = tf.placeholder(tf.float32, [None, None])
        # for indication-specific weight
        D = tf.placeholder(tf.float32, [None, None])
        return X, Y, O, D
    else:
        return X, Y

def initialize_parameters(n_samples, n_drugs, n_x_features, n_y_features, n_dimensions, seed):

    """
    Initialize parameters
    Depending on the objective function, b_P might not be used in the later step.
    """

    parameters = {}

    parameters['W_P'] = tf.Variable(tf.truncated_normal([n_x_features, n_dimensions], stddev=0.2, mean=0, seed=seed), name="W_P")
    parameters['W_Q'] = tf.Variable(tf.truncated_normal([n_y_features, n_dimensions], stddev=0.2, mean=0, seed=seed), name="W_Q")
    parameters['b_P'] = tf.get_variable('b_P', [n_samples, 1], initializer = tf.zeros_initializer())
    parameters['b_Q'] = tf.get_variable('b_Q', [n_drugs, 1], initializer = tf.zeros_initializer())

    return parameters

def inward_propagation(X, Y, parameters, n_samples, n_drugs, model_spec_name):

    """
    Define base objective function
    """

    W_P = parameters['W_P']
    W_Q = parameters['W_Q']
    P = tf.matmul(X, W_P)
    Q = tf.matmul(Y, W_Q)

    b_P_mat = tf.matmul(parameters['b_P'], tf.convert_to_tensor(np.ones(n_drugs).reshape(1, n_drugs), np.float32))
    b_Q_mat = tf.transpose(tf.matmul(parameters['b_Q'], tf.convert_to_tensor(np.ones(n_samples).reshape(1, n_samples), np.float32)))

    if model_spec_name == 'cadrres':
        S = tf.add(b_Q_mat, tf.add(b_P_mat, tf.matmul(P, tf.transpose(Q))))
    elif model_spec_name in ['cadrres-wo-sample-bias', 'cadrres-wo-sample-bias-weight']:
        S = tf.add(b_Q_mat, tf.matmul(P, tf.transpose(Q)))
    # TODO: add the model without both drug and sample biases
    # elif model_spec_name == 'cadrres-wo-bias':
    #     S = tf.matmul(P, tf.transpose(Q))
    else:
        S = None

    return S

def get_latent_vectors(X, Y, parameters):

    """
    Get latent vectors of cell line (P) and drug (Q) on the pharmacogenomic space
    """

    W_P = parameters['W_P']
    W_Q = parameters['W_Q']
    P = tf.matmul(X, W_P)
    Q = tf.matmul(Y, W_Q)
    return P, Q

##### Predicting function #####

def predict(X, Y, S_obs, parameters_trained, X_sample_list, model_spec_name, is_train):

    """
    Make a prediction and calculate cost. This function is used in the training step.
    """

    n_samples = len(X_sample_list)
    n_drugs = Y.shape[1]
    
    W_P = parameters_trained['W_P']
    W_Q = parameters_trained['W_Q']
    P = np.matmul(X, W_P)
    Q = np.matmul(Y, W_Q)

    if model_spec_name in ['cadrres-wo-sample-bias', 'cadrres-wo-sample-bias-weight']:
    
        b_Q = parameters_trained['b_Q']

        b_Q_mat = np.transpose(np.matmul(b_Q, np.ones(n_samples).reshape(1, n_samples)))
        S = b_Q_mat + np.matmul(P, np.transpose(Q))

        cost = np.nanmean(np.square(S - S_obs))/2.0

    elif model_spec_name == 'cadrres':
 
        b_Q = parameters_trained['b_Q']
        b_P = parameters_trained['b_P']

        if is_train:

            b_P_est = b_P

        else:
            # estimate sample bias
            b_P_est = np.matmul(X, b_P) 
            # copy bias for seen samples
            for u, s_name in enumerate(X_sample_list):
                if s_name in parameters_trained['sample_list_train']:
                    s_idx = sample_list_train.index(s_name)
                    b_P_est[u, 0] = b_P[s_idx, 0]

        b_P_mat = np.matmul(b_P_est, np.ones(n_drugs).reshape(1, n_drugs))
        b_Q_mat = np.transpose(np.matmul(b_Q, np.ones(n_samples).reshape(1, n_samples)))
        S = b_Q_mat + b_P_mat + np.matmul(P, np.transpose(Q))        

        cost = np.nanmean(np.square(S - S_obs))/2.0

    return S, cost

##### Training function #####

def train_model(train_resp_df, train_feature_df, test_resp_df, test_feature_df, n_dim, lda, max_iter, l_rate, model_spec_name='cadrres-wo-sample-bias', flip_score=True, seed=1, save_interval=1000, output_dir='output'):

    """
    Train a model. This is for the original cadrres and cadrres-wo-sample-bias

    :param train_resp_df: drug response training data
    :param train_feature_df: kernel feature training data
    :param test_resp_df: drug response testing data
    :param test_feature_df: kernel feature testing data
    :param n_dim: number of dimension of the latent space
    :param lda: regularization factor
    :param max_iter: maximum iteration
    :param l_rate: learning rate
    :param model_spec_name: model specification to define an objective function
    :param flip_score: if `True` then multiple by -1. This is used for converting IC50 to sensitivity score.
    :param seed: random seed for parameter initialization
    :param save_interval: interval for saving results
    :param output_dir: output directory

    :returns: `parameters_trained` contains trained paramters and `output_dict` contains predictions

    """

    print ('Initializing the model ...')

    # Reset TensorFlow graph
    ops.reset_default_graph()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # TODO: save the model configuration

    ################################
    ##### Setting up the model #####
    ################################

    # TODO: add other model types and update the scrip accordingly
    if model_spec_name not in ['cadrres', 'cadrres-wo-sample-bias']:
        return None

    n_drugs = train_resp_df.shape[1]
    drug_list = train_resp_df.columns

    n_samples = train_resp_df.shape[0]
    sample_list_train = train_resp_df.index
    sample_list_test = test_resp_df.index

    n_x_features = train_feature_df.shape[1]
    n_y_features = n_drugs

    X_train_dat = np.array(train_feature_df)
    Y_train_dat = np.identity(n_drugs)

    X_test_dat = np.array(test_feature_df)
    Y_test_dat = np.identity(n_drugs)

    ##### Convert log(IC50) to sensitivity scores #####
    if flip_score:
        S_train_obs = np.array(train_resp_df) * -1
        S_test_obs = np.array(test_resp_df) * -1
    else:
        S_train_obs = np.array(train_resp_df)
        S_test_obs = np.array(test_resp_df)

    ##### Initialize placeholders and parameters #####
    X_train, Y_train = create_placeholders(n_x_features, n_y_features)
    parameters = initialize_parameters(n_samples, n_drugs, n_x_features, n_y_features, n_dim, seed)

    ##### Extract only prediction of only observed drug response #####
    train_known_idx = np.where(~np.isnan(S_train_obs.reshape(-1)))[0]
    n_train_known = len(train_known_idx)
    print ("Train:", len(train_known_idx), "out of", n_drugs * n_samples)

    S_train_pred = inward_propagation(X_train, Y_train, parameters, n_samples, n_drugs, model_spec_name)
    S_train_pred_resp = tf.gather(tf.reshape(S_train_pred, [-1]), train_known_idx, name="S_train_pred_resp")
    S_train_obs_resp = tf.convert_to_tensor(S_train_obs.reshape(-1)[train_known_idx], np.float32, name="S_train_obs_resp")

    #### Calculate the difference between the predicted sensitivity and the actual #####
    diff_op_train = tf.subtract(S_train_pred_resp, S_train_obs_resp, name="raw_training_error")

    with tf.name_scope("train_cost") as scope:
        base_cost = tf.reduce_sum(tf.square(diff_op_train, name="squared_diff_train"), name="sse_train")
        regularizer = tf.multiply(tf.add(tf.reduce_sum(tf.square(parameters['W_P'])), tf.reduce_sum(tf.square(parameters['W_Q']))), lda, name="regularize")
        cost_train = tf.math.divide(tf.add(base_cost, regularizer), n_train_known * 2.0, name="avg_error_train")

    # TODO: add different kinds of regulalization (the current version uses ridge; see CaDRReS2_tf_matrix_factorization_wo_bp_lasso.py)
    
    ##### Use an exponentially decaying learning rate #####
    # learning_rate = tf.train.exponential_decay(l_rate, global_step, 10000, 0.96, staircase=True)

    ##################################################
    ##### Initialize session and train the model #####
    ##################################################

    print ('Starting model training ...')

    global_step = tf.Variable(0, trainable=False)

    with tf.name_scope("train") as scope:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=l_rate)
        train_step = optimizer.minimize(cost_train, global_step=global_step)
        mse_summary = tf.summary.scalar("mse_train", cost_train)

    sess = tf.Session()

    # TODO: save model every save_interval
    # summary_op = tf.summary.merge_all()
    # writer = tf.summary.FileWriter("{}/tf_matrix_factorization_logs".format(output_dir), sess.graph)

    sess.run(tf.global_variables_initializer())
    parameters_init = sess.run(parameters)

    cost_train_vals = []
    cost_test_vals = []

    start = time.time()
    for i in range(max_iter):    
        _ = sess.run(train_step, feed_dict={X_train: X_train_dat, Y_train: Y_train_dat})
        
        if i % save_interval == 0:

            # training step
            res = sess.run(cost_train, feed_dict={X_train: X_train_dat, Y_train: Y_train_dat})
            cost_train_vals += [res]

            time_used = (time.time() - start)
            print("MSE train at step {}: {:.3f} ({:.2f}m)".format(i, cost_train_vals[-1], time_used/60))

            # save parameter
            parameters_trained = sess.run(parameters)
            parameters_trained['sample_list_train'] = sample_list_train
            parameters_trained['sample_list_test'] = sample_list_test

            # make a prediction
            test_pred, test_cost = predict(X_test_dat, Y_test_dat, S_test_obs, parameters_trained, sample_list_test, model_spec_name, False)

            cost_test_vals += [test_cost]
            # summary_str = res[0]
            # writer.add_summary(summary_str, i)

    parameters_trained, train_pred = sess.run([parameters, S_train_pred], feed_dict={X_train: X_train_dat, Y_train: Y_train_dat})
    parameters_trained['sample_list_train'] = sample_list_train
    parameters_trained['sample_list_test'] = sample_list_test

    test_pred, test_cost = predict(X_test_dat, Y_test_dat, S_test_obs, parameters_trained, sample_list_test, model_spec_name, False)
    train_pred, train_cost = predict(X_train_dat, Y_train_dat, S_train_obs, parameters_trained, sample_list_train, model_spec_name, True)
    parameters_trained['mse_train_vals'] = cost_train_vals
    parameters_trained['mse_test_vals'] = cost_test_vals

    P_train, Q_train = get_latent_vectors(X_train, Y_train, parameters)
    P, Q = sess.run([P_train, Q_train], feed_dict={X_train: X_train_dat, Y_train: Y_train_dat})

    sess.close()

    ############################
    ##### Saving the model #####
    ############################

    print ('Saving model parameters and predictions ...')

    # Save model configurations

    parameters_trained['n_dim'] = n_dim
    parameters_trained['lda'] = lda
    parameters_trained['max_iter'] = max_iter
    parameters_trained['l_rate'] = l_rate
    parameters_trained['model_spec_name'] = model_spec_name
    parameters_trained['seed'] = seed

    parameters_trained['drug_list'] = drug_list
    parameters_trained['train_sample_list'] = sample_list_train
    parameters_trained['kernel_sample_list'] = list(train_feature_df.columns)

    # Save the prediction and processed data

    pred_df = pd.DataFrame(test_pred, index=sample_list_test, columns=drug_list) * -1
    obs_df = pd.DataFrame(S_test_obs, index=sample_list_test, columns=drug_list) * -1

    pred_train_df = pd.DataFrame(train_pred, index=sample_list_train, columns=drug_list) * -1
    obs_train_df = pd.DataFrame(S_train_obs, index=sample_list_train, columns=drug_list) * -1

    output_dict = {}
    output_dict['pred_test_df'] = pred_df
    output_dict['obs_test_df'] = obs_df
    output_dict['pred_train_df'] = pred_train_df
    output_dict['obs_train_df'] = obs_train_df

    bq = parameters_trained['b_Q']
    bq_df = pd.DataFrame([drug_list, list(bq.flatten())]).T
    bq_df.columns = ['drug_name', 'drug_bias']
    bq_df = bq_df.set_index('drug_name')

    P_df = pd.DataFrame(P, index=sample_list_train, columns=range(1, n_dim+1))
    Q_df = pd.DataFrame(Q, index=drug_list, columns=range(1, n_dim+1))

    output_dict['b_Q_df'] = bq_df
    output_dict['P_df'] = P_df
    output_dict['Q_df'] = Q_df

    print ('DONE')

    return parameters_trained, output_dict

def train_model_logistic_weight(train_resp_df, train_feature_df, test_resp_df, test_feature_df, weights_logistic_x0_df, weights_indication_df, n_dim, lda, max_iter, l_rate, model_spec_name='cadrres-wo-sample-bias-weight', flip_score=True, seed=1, save_interval=1000, output_dir='output', device='CPU:0'):

    """
    Train a model. This is for CaDRReS-Sc, i.e. cadrres-wo-sample-bias-weight

    :param train_resp_df: drug response training data
    :param train_feature_df: kernel feature training data
    :param test_resp_df: drug response testing data
    :param test_feature_df: kernel feature testing data
    :param weights_logistic_x0_df: logistic weight based on the maximum concentration
    :param weights_indication_df: indication-specific weight
    :param n_dim: number of dimension of the latent space
    :param lda: regularization factor
    :param max_iter: maximum iteration
    :param l_rate: learning rate
    :param model_spec_name: model specification to define an objective function
    :param flip_score: if `True` then multiple by -1. This is used for converting IC50 to sensitivity score.
    :param seed: random seed for parameter initialization
    :param save_interval: interval for saving results
    :param output_dir: output directory
    :param device: select device for tensorflow

    :returns: `parameters_trained` contains trained paramters and `output_dict` contains predictions

    """

    print ('Getting data ...')

    ################################
    ##### Setting up the model #####
    ################################

    if model_spec_name not in ['cadrres-wo-sample-bias-weight']:
        return None

    n_drugs = train_resp_df.shape[1]
    drug_list = train_resp_df.columns

    n_samples = train_resp_df.shape[0]
    sample_list_train = train_resp_df.index
    sample_list_test = test_resp_df.index

    n_x_features = train_feature_df.shape[1]
    n_y_features = n_drugs

    X_train_dat = np.array(train_feature_df)
    Y_train_dat = np.identity(n_drugs)

    X_test_dat = np.array(test_feature_df)
    Y_test_dat = np.identity(n_drugs)

    ##### Convert log(IC50) to sensitivity scores #####
    if flip_score:
        S_train_obs = np.array(train_resp_df) * -1
        S_test_obs = np.array(test_resp_df) * -1
        logistic_x0_dat = np.array(weights_logistic_x0_df) * -1
    else:
        S_train_obs = np.array(train_resp_df)
        S_test_obs = np.array(test_resp_df)
        logistic_x0_dat = np.array(weights_logistic_x0_df)

    weight_indication_dat = np.array(weights_indication_df)

    with tf.device(device):

        print ('Initializing the model ...')

        ##### Reset TensorFlow graph #####
        ops.reset_default_graph()

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # TODO: save the model configuration

        ##### Initialize placeholders and parameters #####
        X_train, Y_train, logistic_x0, weight_indication = create_placeholders(n_x_features, n_y_features, sample_weight=True)
        parameters = initialize_parameters(n_samples, n_drugs, n_x_features, n_y_features, n_dim, seed)

        ##### Extract only prediction of only observed drug response #####
        train_known_idx = np.where(~np.isnan(S_train_obs.reshape(-1)))[0]
        n_train_known = len(train_known_idx)
        print ("Train:", len(train_known_idx), "out of", n_drugs * n_samples)
        ones = tf.convert_to_tensor(np.ones(n_train_known), np.float32)

        S_train_obs_resp = tf.convert_to_tensor(S_train_obs.reshape(-1)[train_known_idx], np.float32, name="S_train_obs_resp")

        S_train_pred = inward_propagation(X_train, Y_train, parameters, n_samples, n_drugs, model_spec_name)
        S_train_pred_resp = tf.gather(tf.reshape(S_train_pred, [-1]), train_known_idx, name="S_train_pred_resp")

        ##### Assign weights #####
        # TODO: tune parameters (slope and shift) for sigmoid function for assigning weight
        O_per_sample = tf.gather(tf.reshape(logistic_x0, [-1]), train_known_idx, name="weight_logistic_x0")
        O_weight_pred = tf.math.sigmoid(tf.math.scalar_mul(10.0, tf.subtract(S_train_pred_resp, O_per_sample) + 0.5))
        O_weight_obs = tf.math.sigmoid(tf.math.scalar_mul(10.0, tf.subtract(S_train_obs_resp, O_per_sample) + 0.5))
        C_per_sample = tf.math.maximum(O_weight_pred, O_weight_obs, name="C_per_sample")

        D_per_sample = tf.gather(tf.reshape(weight_indication, [-1]), train_known_idx, name="weight_indication")

        #### Calculate the difference between the predicted sensitivity and the actual #####
        diff_op_train = tf.subtract(S_train_pred_resp, S_train_obs_resp, name="raw_training_error")
        sqrt_err_per_sample = tf.square(diff_op_train, name="sqrt_err_per_sample")

        with tf.name_scope("train_cost") as scope:
            indication_weighted_se_per_sample = tf.multiply(D_per_sample, sqrt_err_per_sample)
            weighted_se_per_sample = tf.multiply(C_per_sample, indication_weighted_se_per_sample)
            base_cost = tf.reduce_sum(weighted_se_per_sample, name="base_cost")
            # base_cost = tf.reduce_sum(sqrt_err_per_sample, name="base_cost")
            
            regularizer = tf.multiply(tf.add(tf.reduce_sum(tf.square(parameters['W_P'])), tf.reduce_sum(tf.square(parameters['W_Q']))), lda, name="regularize")
            cost_train = tf.math.divide(tf.add(base_cost, regularizer), n_train_known * 1.0, name="avg_error_train")
        
        # TODO: add different kinds of regulalization (the current version uses ridge; see CaDRReS2_tf_matrix_factorization_wo_bp_lasso.py)
        
        ##### Use an exponentially decaying learning rate #####
        # learning_rate = tf.train.exponential_decay(l_rate, global_step, 10000, 0.96, staircase=True)

        ##################################################
        ##### Initialize session and train the model #####
        ##################################################

        print ('Starting model training ...')

        global_step = tf.Variable(0, trainable=False)

        with tf.name_scope("train") as scope:
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=l_rate)
            train_step = optimizer.minimize(cost_train, global_step=global_step)
            mse_summary = tf.summary.scalar("mse_train", cost_train)

        sess = tf.Session()
        # sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

        print ('TF session started ...')

        # summary_op = tf.summary.merge_all()
        # writer = tf.summary.FileWriter("{}/tf_matrix_factorization_logs".format(output_dir), sess.graph)

        # TODO: update MSE in log to be the weighted version

        sess.run(tf.global_variables_initializer())
        parameters_init = sess.run(parameters)    

        cost_train_vals = []
        cost_test_vals = []

        O_weight_pred_list = []

        # temp1 = {}
        # temp1['parameters'], temp1['diff_op_train'], temp1['O_per_sample'], temp1['O_weight_pred'], temp1['O_weight_obs'], temp1['C_per_sample'], temp1['S_train_pred_resp'] = sess.run([parameters, diff_op_train, O_per_sample, O_weight_pred, O_weight_obs, C_per_sample, S_train_pred_resp], feed_dict={X_train: X_train_dat, Y_train: Y_train_dat, logistic_x0: logistic_x0_dat, weight_indication: weight_indication_dat})

        print ('Starting 1st iteration ...')

        start = time.time()
        for i in range(max_iter):    

            _ = sess.run(train_step, feed_dict={X_train: X_train_dat, Y_train: Y_train_dat, logistic_x0: logistic_x0_dat, weight_indication: weight_indication_dat})

            # temp2 = {}
            # temp2['parameters'], temp2['diff_op_train'], temp2['O_per_sample'], temp2['O_weight_pred'], temp2['O_weight_obs'], temp2['C_per_sample'], temp2['S_train_pred_resp'] = sess.run([parameters, diff_op_train, O_per_sample, O_weight_pred, O_weight_obs, C_per_sample, S_train_pred_resp], feed_dict={X_train: X_train_dat, Y_train: Y_train_dat, logistic_x0: logistic_x0_dat, weight_indication: weight_indication_dat})
            
            if i % save_interval == 0:

                # training step
                res, O_weight_pred_vals = sess.run([cost_train, O_weight_pred], feed_dict={X_train: X_train_dat, Y_train: Y_train_dat, logistic_x0: logistic_x0_dat, weight_indication: weight_indication_dat})
                cost_train_vals += [res]
                O_weight_pred_list += [O_weight_pred_vals]

                time_used = (time.time() - start)
                print("MSE train at step {}: {:.3f} ({:.2f}m)".format(i, cost_train_vals[-1], time_used/60))

                # save parameter
                parameters_trained = sess.run(parameters)

                # make a prediction
                test_pred, test_cost = predict(X_test_dat, Y_test_dat, S_test_obs, parameters_trained, sample_list_test, model_spec_name, False)

                cost_test_vals += [test_cost]
                # summary_str = res[0]
                # writer.add_summary(summary_str, i)

                if np.isnan(res):
                    break

        parameters_trained, train_pred, O_weight_pred_vals, O_weight_obs_vals = sess.run([parameters, S_train_pred, O_weight_pred, O_weight_obs], feed_dict={X_train: X_train_dat, Y_train: Y_train_dat, logistic_x0: logistic_x0_dat, weight_indication: weight_indication_dat})
        test_pred, test_cost = predict(X_test_dat, Y_test_dat, S_test_obs, parameters_trained, sample_list_test, model_spec_name, False)
        _, train_cost = predict(X_train_dat, Y_train_dat, S_train_obs, parameters_trained, sample_list_train, model_spec_name, True)
        parameters_trained['mse_train_vals'] = cost_train_vals
        parameters_trained['mse_test_vals'] = cost_test_vals

        O_weight_pred_list += [O_weight_pred_vals]
        parameters_trained['O_weight_pred_vals'] = np.array(O_weight_pred_list)
        parameters_trained['O_weight_obs_vals'] = O_weight_obs_vals
        parameters_trained['train_known_idx'] = train_known_idx

        # parameters_trained['temp1'] = temp1
        # parameters_trained['temp2'] = temp2

        P_train, Q_train = get_latent_vectors(X_train, Y_train, parameters)
        P, Q = sess.run([P_train, Q_train], feed_dict={X_train: X_train_dat, Y_train: Y_train_dat})

        sess.close()

    ############################
    ##### Saving the model #####
    ############################

    print ('Saving model parameters and predictions ...')

    # Save model configurations

    parameters_trained['n_dim'] = n_dim
    parameters_trained['lda'] = lda
    parameters_trained['max_iter'] = max_iter
    parameters_trained['l_rate'] = l_rate
    parameters_trained['model_spec_name'] = model_spec_name
    parameters_trained['seed'] = seed

    parameters_trained['drug_list'] = drug_list
    parameters_trained['train_sample_list'] = sample_list_train
    parameters_trained['kernel_sample_list'] = list(train_feature_df.columns)

    parameters_trained['weights_logistic_x0'] = weights_logistic_x0_df
    parameters_trained['weights_indication'] = weights_indication_df

    # Save the prediction and processed data

    pred_df = pd.DataFrame(test_pred, index=sample_list_test, columns=drug_list) * -1
    obs_df = pd.DataFrame(S_test_obs, index=sample_list_test, columns=drug_list) * -1

    pred_train_df = pd.DataFrame(train_pred, index=sample_list_train, columns=drug_list) * -1
    obs_train_df = pd.DataFrame(S_train_obs, index=sample_list_train, columns=drug_list) * -1

    output_dict = {}
    output_dict['pred_test_df'] = pred_df
    output_dict['obs_test_df'] = obs_df
    output_dict['pred_train_df'] = pred_train_df
    output_dict['obs_train_df'] = obs_train_df

    bq = parameters_trained['b_Q'] * -1
    bq_df = pd.DataFrame([drug_list, list(bq.flatten())]).T
    bq_df.columns = ['drug_name', 'drug_bias']
    bq_df = bq_df.set_index('drug_name')

    P_df = pd.DataFrame(P, index=sample_list_train, columns=range(1, n_dim+1))
    Q_df = pd.DataFrame(Q, index=drug_list, columns=range(1, n_dim+1))

    output_dict['b_Q_df'] = bq_df
    output_dict['P_df'] = P_df
    output_dict['Q_df'] = Q_df

    print ('DONE')

    return parameters_trained, output_dict

def get_sample_weights_logistic_x0(drug_df, log2_max_conc_col_name, sample_list):

    """
    Calculate weights_logistic_x0_df, which is an input of train_model_logistic_weight. The logistic weight is assigned to each drug-sample pair with respect to maximum drug dosage.
    """

    drug_list = drug_df.index
    max_conc = np.array(drug_df[[log2_max_conc_col_name]])
    n_samples = len(sample_list)
    weights_logistic_x0 = np.repeat(max_conc.T, n_samples, axis=0)
    weights_logistic_x0_df = pd.DataFrame(weights_logistic_x0, columns=drug_list, index=sample_list)

    return weights_logistic_x0_df

# TODO:
# 1) update to tensorflow v.2 and using standard model.save(h5)
# 2) support original cadrres version with an improved bias estimation
