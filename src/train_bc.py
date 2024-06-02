import time
import torch
import numpy as np
from utils.load_data import load_path_feature, load_link_feature, minmax_normalization, load_test_traj
from utils.evaluation import evaluate_model, evaluate_log_prob
from network_env import RoadWorld
from model.policy import PolicyCNN
torch.backends.cudnn.enabled = False
import pandas as pd

if __name__ == '__main__':
    learning_rate = 1e-3
    n_iters = 500  # number of training iterations
    batch_size = 32  # training batch size
    max_length = 50  # maximum path length for recursively generating next link until destination is reached
    cv = 0  # cross validation process [0, 1, 2, 3, 4]
    size = 1000  # # size of training data [100, 1000, 10000]
    max_iter_num = 100
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    edge_p = "../data/edge.txt"
    network_p = "../data/transit.npy"
    path_feature_p = "../data/feature_od.npy"
    train_p = "../data/cross_validation/train_CV%d_size%d.csv" % (cv, size)
    test_p = "../data/cross_validation/test_CV%d.csv" % cv
    # test_p = "../data/cross_validation/train_CV%d_size%d.csv" % (cv, size)
    model_p = "../trained_models/bc_CV%d_size%d.pt" % (cv, size)
    loss_csv_p = "../trained_models/loss_CV%d_size%d.csv" % (cv, size)

    """initialize road environment"""
    env = RoadWorld(network_p, edge_p)
    """load path-level and link-level feature"""
    path_feature, path_max, path_min = load_path_feature(path_feature_p)
    print('path_feature',path_feature)
    print('path_max',path_max)
    print('path_max',path_min)

    edge_feature, edge_max, edge_min = load_link_feature(edge_p)
    print('edge_feature',edge_feature)
    print('edge_feature',edge_feature)
    print('edge_feature',edge_feature)

    path_feature = minmax_normalization(path_feature, path_max, path_min)
    print('path_feature.shape[2]',path_feature.shape[2])
    path_feature_pad = np.zeros((env.n_states, env.n_states, path_feature.shape[2]))
    print('path_feature_pad',path_feature_pad)
    path_feature_pad[:path_feature.shape[0], :path_feature.shape[1], :] = path_feature
    print('path_feature_pad',path_feature_pad)

    edge_feature = minmax_normalization(edge_feature, edge_max, edge_min)
    print('edge_feature norm', edge_feature)
    edge_feature_pad = np.zeros((env.n_states, edge_feature.shape[1]))
    edge_feature_pad[:edge_feature.shape[0], :] = edge_feature
    print('edge_feature_pad', edge_feature_pad)
    """initiate model"""
    CNNMODEL = PolicyCNN(env.n_actions, env.policy_mask, env.state_action, path_feature_pad, edge_feature_pad,
                         path_feature_pad.shape[-1] + edge_feature_pad.shape[-1] + 1 +1, env.pad_idx).to(device)
    CNNMODEL.to_device(device)
    criterion = torch.nn.NLLLoss()
    optimizer = torch.optim.Adam(CNNMODEL.parameters(), lr=learning_rate)
    """load train data"""
    train_trajs = env.import_demonstrations_step(train_p)
    # train_state_list, train_des_list, train_action_list = [], [], []
    train_state_list, train_des_list, train_weather_list, train_action_list = [], [], [], []
    for episode in train_trajs:
        for x in episode:
            train_state_list.append(x.cur_state)
            train_des_list.append(episode[-1].next_state)
            train_weather_list.append(x.speed)
            train_action_list.append(x.action)
    x_state_train = torch.LongTensor(train_state_list).to(device)
    x_des_train = torch.LongTensor(train_des_list).to(device)
    x_weather_train = torch.LongTensor(train_weather_list).to(device)
    y_train = torch.LongTensor(train_action_list).to(device)
    """train model"""
    start_time = time.time()
    loss_records = []

    for it in range(max_iter_num):
        epoch_loss = 0
        for i in range(0, x_state_train.shape[0], batch_size):
            batch_right = min(i + batch_size, x_state_train.shape[0])
            sampled_x_state_train = x_state_train[i:batch_right]
            sampled_x_des_train = x_des_train[i:batch_right]
            sampled_x_weather_train = x_weather_train[i:batch_right]
            sampled_y_train = y_train[i:batch_right].contiguous().view(-1)
            y_est = CNNMODEL.get_action_log_prob(sampled_x_state_train, sampled_x_des_train, sampled_x_weather_train)
            loss = criterion(y_est.view(-1, y_est.size(1)), sampled_y_train)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        # Average loss for the epoch
        epoch_loss /= (x_state_train.shape[0] / batch_size)
        print(f'Epoch [{it + 1}/{max_iter_num}], Loss: {epoch_loss:.4f}')
        
        # Record loss
        loss_records.append([it + 1, epoch_loss])
    torch.save(CNNMODEL.state_dict(), model_p)

    # Save loss records to CSV
    loss_df = pd.DataFrame(loss_records, columns=['Iteration', 'Loss'])
    loss_df.to_csv(loss_csv_p, index=False)
    """evaluate model"""
    CNNMODEL.load_state_dict(torch.load(model_p, map_location=torch.device('cpu')))
    print('training time', time.time() - start_time)
    # target_traj, target_od = load_test_traj(test_p)
    target_traj, target_od_weather = load_test_traj(test_p)
    target_od = target_od_weather[:, :2]
    target_weather = target_od_weather[:, 2]

    target_od = torch.from_numpy(target_od).long().to(device)
    target_weather = torch.from_numpy(target_weather).long().to(device)
    # evaluate_model(target_od, target_traj, CNNMODEL, env)
    evaluate_model(target_od, target_traj, target_weather, CNNMODEL, env)
    print('test time', time.time() - start_time)
    """evaluate log prob"""
    test_trajs = env.import_demonstrations_step(test_p)
    # evaluate_log_prob(test_trajs, CNNMODEL)
    test_weather = [x[0].weather for x in test_trajs]
    evaluate_log_prob(test_trajs, test_weather, CNNMODEL)
