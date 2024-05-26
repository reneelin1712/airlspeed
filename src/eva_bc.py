from utils.evaluation import evaluate_model, evaluate_log_prob, evaluate_train_edit_dist
import time
import torch
from utils.load_data import ini_od_dist, load_path_feature, load_link_feature, \
    minmax_normalization, load_train_sample, load_test_traj
from network_env import RoadWorld
from utils.torch import to_device
import numpy as np
import pandas as pd
from model.policy import PolicyCNN, PolicyCNNWrapper

cv = 0  # cross validation process [0, 1, 2, 3, 4]
size = 1000  # size of training data [100, 1000, 10000]
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model_p = "../trained_models/bc_CV%d_size%d.pt" % (cv, size)
test_p = "../data/cross_validation/test_CV%d.csv" % cv

"""environment"""
edge_p = "../data/edge.txt"
network_p = "../data/transit.npy"
path_feature_p = "../data/feature_od.npy"
train_p = "../data/cross_validation/train_CV%d_size%d.csv" % (cv, size)
test_p = "../data/cross_validation/test_CV%d.csv" % cv
model_p = "../trained_models/bc_CV%d_size%d.pt" % (cv, size)

"""inialize road environment"""
od_list, od_dist = ini_od_dist(train_p)
env = RoadWorld(network_p, edge_p, pre_reset=(od_list, od_dist))
"""load path-level and link-level feature"""
path_feature, path_max, path_min = load_path_feature(path_feature_p)
edge_feature, link_max, link_min = load_link_feature(edge_p)
path_feature = minmax_normalization(path_feature, path_max, path_min)
path_feature_pad = np.zeros((env.n_states, env.n_states, path_feature.shape[2]))
path_feature_pad[:path_feature.shape[0], :path_feature.shape[1], :] = path_feature
edge_feature = minmax_normalization(edge_feature, link_max, link_min)
edge_feature_pad = np.zeros((env.n_states, edge_feature.shape[1]))
edge_feature_pad[:edge_feature.shape[0], :] = edge_feature

"""define policy network"""
policy_net = PolicyCNN(env.n_actions, env.policy_mask, env.state_action,
                       path_feature_pad, edge_feature_pad,
                       path_feature_pad.shape[-1] + edge_feature_pad.shape[-1] + 1 + 1,
                       env.pad_idx).to(device)

def load_model(model_path):
    policy_net.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    print("Policy Model loaded Successfully")


if __name__ == '__main__':
    # Load the trained model
    load_model(model_p)
    
    # Evaluate the model on test data
    test_trajs, test_od_weather = load_test_traj(test_p)
    test_od = test_od_weather[:, :2].astype(int)  # Extract the origin and destination columns
    test_weather = test_od_weather[:, 2]  # Extract the weather column
    
    start_time = time.time()
    evaluate_model(test_od, test_trajs, test_weather, policy_net, env)
    print('Test time:', time.time() - start_time)
    
    # Evaluate log probabilities
    test_trajs = env.import_demonstrations_step(test_p)
    test_weather = [traj[0].speed for traj in test_trajs]
    evaluate_log_prob(test_trajs, test_weather, policy_net)