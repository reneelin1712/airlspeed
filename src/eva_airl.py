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
from model.value import ValueCNN
from model.discriminator import DiscriminatorAIRLCNN

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

model_p = "../trained_models/airl_CV%d_size%d.pt" % (cv, size)
test_p = "../data/cross_validation/test_CV%d.csv" % cv

"""environment"""
edge_p = "../data/edge.txt"
network_p = "../data/transit.npy"
path_feature_p = "../data/feature_od.npy"
train_p = "../data/cross_validation/train_CV%d_size%d.csv" % (cv, size)
test_p = "../data/cross_validation/test_CV%d.csv" % cv
# test_p = "../data/cross_validation/train_CV%d_size%d.csv" % (cv, size)
model_p = "../trained_models/airl_CV%d_size%d.pt" % (cv, size)
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
                       path_feature_pad.shape[-1] + edge_feature_pad.shape[-1] + 1 + 1,
                       env.pad_idx).to(device)
value_net = ValueCNN(path_feature_pad, edge_feature_pad,
                     path_feature_pad.shape[-1] + edge_feature_pad.shape[-1] + 1).to(device)
discrim_net = DiscriminatorAIRLCNN(env.n_actions, gamma, env.policy_mask,
                                   env.state_action, path_feature_pad, edge_feature_pad,
                                   path_feature_pad.shape[-1] + edge_feature_pad.shape[-1] + 1 + 1,
                                   path_feature_pad.shape[-1] + edge_feature_pad.shape[-1] + 1,
                                   env.pad_idx).to(device)


def evaluate_rewards(test_traj, test_weather, policy_net, discrim_net, env):
    device = torch.device('cpu')  # Use CPU device
    policy_net.to(device)  # Move policy_net to CPU
    discrim_net.to(device)  # Move discrim_net to CPU

    reward_data = []
    for episode_idx, (episode, weather) in enumerate(zip(test_traj, test_weather)):
        des = torch.LongTensor([episode[-1].next_state]).long().to(device)
        weather_var = torch.LongTensor([weather]).long().to(device)
        for step_idx, x in enumerate(episode):
            state = torch.LongTensor([x.cur_state]).to(device)
            next_state = torch.LongTensor([x.next_state]).to(device)
            action = torch.LongTensor([x.action]).to(device)
            
            action_rewards = []
            for a in env.get_action_list(x.cur_state):
                action_tensor = torch.LongTensor([a]).to(device)
                with torch.no_grad():
                    log_prob = policy_net.get_log_prob(state, des, weather_var, action_tensor).squeeze()
                    reward = discrim_net.calculate_reward(state, des, action_tensor, log_prob, next_state, weather_var).item()
                action_rewards.append((a, reward))
            
            max_reward_action = max(action_rewards, key=lambda x: x[1])
            
            reward_data.append({
                'episode': episode_idx + 1,
                'step': step_idx + 1,
                'state': x.cur_state,
                'action': x.action,
                'next_state': x.next_state,
                'chosen_action_reward': next(r for a, r in action_rewards if a == x.action),
                'max_reward_action': max_reward_action[0],
                'max_reward': max_reward_action[1]
            })
            
            for a, r in action_rewards:
                reward_data.append({
                    'episode': episode_idx + 1,
                    'step': step_idx + 1,
                    'state': x.cur_state,
                    'action': a,
                    'next_state': x.next_state,
                    'chosen_action_reward': None,
                    'max_reward_action': None,
                    'max_reward': None,
                    'action_reward': r
                })
    
    reward_df = pd.DataFrame(reward_data)
    return reward_df


if __name__ == '__main__':
    # Load the trained models
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
    
    # Evaluate rewards
    reward_df = evaluate_rewards(test_trajs, test_weather, policy_net, discrim_net, env)
    reward_df.to_csv('reward_data.csv', index=False)