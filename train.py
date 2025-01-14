import yaml
from data.loaders.dataset_loader import *
from models.sgpn_model import *
from evaluate import eval_visualization
import torch
from torch.utils.data import DataLoader
import os

# Load config
with open('config/config.yaml', 'r') as file:
    config = yaml.safe_load(file)

""" Data 로더 부분  """
path_object = config['data']['path_object']
scenario_lists, total_feature, scenario_relationships = init_datasets(path_object)
train_datasets = GCMDataset(item_list=scenario_lists[:config['data']['train_split']], object_features=total_feature,
                             relationships_list=scenario_relationships)
test_datasets = GCMDataset(item_list=scenario_lists[config['data']['train_split']:], object_features=total_feature,
                            relationships_list=scenario_relationships)

train_loader = DataLoader(dataset=train_datasets, batch_size=config['train']['batch_size'], shuffle=True)
test_loader = DataLoader(dataset=test_datasets, batch_size=config['eval']['batch_size'], shuffle=False)

""" Device 설정  """
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
print('device = ', device)

"""하이퍼 파라미터 설정 """
hy_param = config['hyperparameters']

# Ensure pretrained folder exists
pretrained_folder = 'pretrained'
os.makedirs(pretrained_folder, exist_ok=True)

def train_v2():
    model = SGPNModel('SGPNModel', hy_param['num_obj_cls'], hy_param['num_rel_cls'], hy_param).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=hy_param['lr'])
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=hy_param['step_size'], gamma=hy_param['gamma'])

    model.train()
    for epoch_num in range(1, hy_param['epochs'] + 1):
        result_loss = []
        for data in train_loader:
            optimizer.zero_grad()
            scenario = data['scenario']

            obj_feature = data['obj_feature']
            obj_feature = obj_feature.squeeze(0).to(device)
            obj_feature = obj_feature[:, :3]
            edges_index = data['edge_index']
            edges_index = torch.Tensor(edges_index).type(torch.LongTensor).t().to(device)

            edge_label = data['edge_label']
            edge_label = edge_label.squeeze(dim=0)[:, 1:6].to(device)

            node2dic = data['node2dic']
            rel_feature = data['node2vec'].type(torch.FloatTensor).squeeze(0)
            rel_feature = torch.Tensor(MinMaxScaler().fit_transform(rel_feature)).to(device)  # 데이터 정규화

            """forward"""

            logs, predict = model.process(
                obj_feature=obj_feature,
                rel_feature=rel_feature,
                edges_index=edges_index,
                gt_rel_cls=edge_label.type(torch.FloatTensor),
                gt_obj_cls=None
            )
            result_loss.append(logs[1])

        if lr_scheduler is not None:
            lr_scheduler.step()

        print('{} epoch = train_loss_avg = {}'.format(epoch_num, sum(result_loss) / len(result_loss)))

        # Save model weights
        if epoch_num % 100 == 0:
            checkpoint_path = os.path.join(pretrained_folder, f'model_epoch_{epoch_num}.pth')
            torch.save(model.state_dict(), checkpoint_path)
            print(f'Model saved at {checkpoint_path}')

        eval_visualization(model, test_loader)

def train_v1():
    model = TripletGCNModel(num_layers=hy_param['num_layer'], rel_num=hy_param['num_rel_cls'],
                            dim_node=hy_param['dim_obj'], dim_edge=hy_param['dim_edge'],
                            dim_hidden=hy_param['gcn_hidden_feature_size']).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=hy_param['lr'])
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=hy_param['step_size'], gamma=hy_param['gamma'])

    model.train()
    for epoch_num in range(1, hy_param['epochs'] + 1):
        result_loss = []
        for data in train_loader:
            optimizer.zero_grad()
            scenario = data['scenario']

            obj_feature = data['obj_feature'].to(device)
            obj_feature = obj_feature.squeeze(0)
            edge_index = data['edge_index']
            edge_index = torch.Tensor(edge_index).type(torch.LongTensor).t().to(device)
            edge_label = data['edge_label']
            edge_label = edge_label.squeeze(dim=0)[:, 1:].to(device)
            node2dic = data['node2dic']

            """Edge Feature 생성"""
            edge_feature = torch.rand([edge_index.size()[1], 40], dtype=torch.float).to(device)

            """GCM 신경망 모델 입력"""
            predict = model(obj_feature, edge_feature, edge_index)

            result = predict['relation_result'].cpu()
            gt_result = edge_label.type(torch.FloatTensor).detach().cpu()
            loss = F.binary_cross_entropy(result, gt_result)
            print(loss.item())
            loss.backward()
            optimizer.step()
        if lr_scheduler is not None:
            lr_scheduler.step()

        # Save model weights every 100 epochs
        if epoch_num % 100 == 0:
            checkpoint_path = os.path.join(pretrained_folder, f'model_epoch_{epoch_num}.pth')
            torch.save(model.state_dict(), checkpoint_path)
            print(f'Model checkpoint saved at {checkpoint_path}')

if __name__ == '__main__':
    train_v2()
