from utils.evaluation import evaluate_model, evaluate_log_prob, evaluate_train_edit_dist
import time
import torch
from utils.load_data import ini_od_dist, load_path_feature, load_link_feature, \
    minmax_normalization, load_train_sample, load_test_traj
from network_env import RoadWorld
from utils.torch import to_device
import numpy as np
import pandas as pd
from model.policy import PolicyCNN,PolicyCNNWrapper
from model.value import ValueCNN
from model.discriminator import DiscriminatorAIRLCNN

import shap

def load_model(model_path):
    model_dict = torch.load(model_path)
    policy_net.load_state_dict(model_dict['Policy'])
    print("Policy Model loaded Successfully")
    value_net.load_state_dict(model_dict['Value'])
    print("Value Model loaded Successfully")
    discrim_net.load_state_dict(model_dict['Discrim'])
    print("Discrim Model loaded Successfully")

cv = 0  # cross validation process [0, 1, 2, 3, 4]
size = 100  # size of training data [100, 1000, 10000]
gamma = 0.99  # discount factor
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model_p = "C:/AI/rcm-airl-my-data/synthetic/trained_models/airl_CV%d_size%d.pt" % (cv, size)
test_p = "C:/AI/rcm-airl-my-data/synthetic/data/cross_validation/test_CV%d.csv" % cv

"""environment"""
edge_p = "C:/AI/rcm-airl-my-data/synthetic/data/edge.txt"
network_p = "C:/AI/rcm-airl-my-data/synthetic/data/transit.npy"
path_feature_p = "C:/AI/rcm-airl-my-data/synthetic/data/feature_od.npy"
train_p = "C:/AI/rcm-airl-my-data/synthetic/data/cross_validation/train_CV%d_size%d.csv" % (cv, size)
test_p = "C:/AI/rcm-airl-my-data/synthetic/data/cross_validation/test_CV%d.csv" % cv
# test_p = "../data/cross_validation/train_CV%d_size%d.csv" % (cv, size)
model_p = "C:/AI/rcm-airl-my-data/synthetic/trained_models/airl_CV%d_size%d.pt" % (cv, size)
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


"""define actor and critic"""
policy_net = PolicyCNN(env.n_actions, env.policy_mask, env.state_action,
                    path_feature_pad, edge_feature_pad,
                    path_feature_pad.shape[-1] + edge_feature_pad.shape[-1] + 1 +1,
                    env.pad_idx).to(device)
value_net = ValueCNN(path_feature_pad, edge_feature_pad,
                    path_feature_pad.shape[-1] + edge_feature_pad.shape[-1] + 1).to(device)
discrim_net = DiscriminatorAIRLCNN(env.n_actions, gamma, env.policy_mask,
                                env.state_action, path_feature_pad, edge_feature_pad,
                                path_feature_pad.shape[-1] + edge_feature_pad.shape[-1] + 1 + 1 ,
                                path_feature_pad.shape[-1] + edge_feature_pad.shape[-1] + 1,
                                env.pad_idx).to(device)

"""Evaluate model"""
load_model(model_p)
test_trajs, test_od_weather = load_test_traj(test_p)
test_od = test_od_weather[:, :2]  # Extract the origin and destination columns
test_weather = test_od_weather[:, 2]  # Extract the weather column
start_time = time.time()
evaluate_model(test_od, test_trajs, test_weather, policy_net, env)
print('test time', time.time() - start_time)
"""Evaluate log prob"""
test_trajs = env.import_demonstrations_step(test_p)
test_weather = [traj[0].weather for traj in test_trajs]
evaluate_log_prob(test_trajs, test_weather, policy_net)


target_traj, target_od_weather = load_test_traj(test_p)
target_od = target_od_weather[:, :2]
target_weather = target_od_weather[:, 2]

target_od = torch.from_numpy(target_od).long().to(device)
target_weather = torch.from_numpy(target_weather).long().to(device)

def create_background_data(train_path, sample_size=100):
    df = pd.read_csv(train_path)
    sample_df = df.sample(n=sample_size)
    ori = torch.LongTensor(sample_df['ori'].values)
    des = torch.LongTensor(sample_df['des'].values)
    weather = torch.LongTensor(sample_df['speed'].values)
    return ori, des, weather

background_ori, background_des, background_weather = create_background_data(train_p)

def prepare_input_data(target_od, target_traj, target_weather):
    ori = torch.LongTensor(target_od[:, 0])
    des = torch.LongTensor(target_od[:, 1])
    weather = torch.LongTensor(target_weather)
    return ori, des, weather

input_ori, input_des, input_weather = prepare_input_data(target_od, target_traj, target_weather)

policy_net_wrapper = PolicyCNNWrapper(policy_net).to(device)

# explainer = shap.DeepExplainer(policy_net, background_data)
explainer = shap.DeepExplainer(policy_net_wrapper, (background_ori, background_des, background_weather))

shap_values = explainer.shap_values((input_ori, input_des, input_weather))
shap.summary_plot(shap_values, (input_ori, input_des, input_weather), plot_type="bar")