import os
import json
from torch.utils.data import Dataset, DataLoader
import torch
import sys
import numpy as np
from torch_geometric.data import Data
import torch
import re
import collections
from torch._six import string_classes
from sklearn.preprocessing import MinMaxScaler


path = './datasets'

'''
관계 종류
none
supported by
left
right
front
behind
close by
inside
bigger than
smaller than
higher than
lower than
same symmetry as
same as
attached to
standing on
lying on
hanging on
connected to
leaning against
part of
belonging to
build in
standing in
cover
lying in
hanging in
'''

class HobeDataset(Dataset):
    def __init__(self,semseg_file_list,path,relation_dataset):
        self.semseg_files = semseg_file_list
        self.path = path
        self.items = []

        # relation_dataset 이건 --> 'scan_id' : 관계들로 구성
        self.relation_dataset = relation_dataset

        for semseg_path in semseg_file_list:
            temp_path = '{}/{}'.format(path, semseg_path)
            with open(temp_path, "r") as semseg_json:
                self.items.append(json.load(semseg_json))

        f = open("./relationships.txt", 'r')
        lines = f.readlines()
        self.total_rel_num = len(lines)
        f.close()

    def __len__(self):
        return len(self.items)


    def __getitem__(self, index):
        item = self.items[index]
        output = {}
        scan_id = item['scan_id']
        obj_feature,hobe_dic = self.node_feature_get(item)

        node_num = len(hobe_dic.values())
        rel_num = len(self.relation_dataset[scan_id]['relationships'])
        # print(self.relation_dataset[scan_id])
        # print('------Node 개수->', len(hobe_dic.values()))
        # print('-------rel_num 개수 ->',rel_num)
        obj_feature = obj_feature.view(node_num, -1)
        edge_index = self.edge_index_get(rel_num=rel_num,hobe_dic=hobe_dic,scan_id=scan_id)
        # print(edge_index)
        data = Data()
        data.x = obj_feature
        data.edge_index = edge_index
        # print(data)

        node2vec, node2dic = self.node2vec_get(obj_feature)
        edge_label = self.edge_label(node2vec, node2dic, scan_id,hobe_dic)
        #     def edge_label(self,node2vec,node2node_dic,scan_id):
        output['obj_feature'] = obj_feature
        output['edge_index'] = edge_index
        output['node2vec'] = node2vec
        output['edge_label'] = edge_label
        output['node2dic'] = node2dic
        return output



    def node_feature_get(self,obj):
        hobe_dic = {}
        object_feature = torch.FloatTensor([])
        id = []
        k = 0
        for nodes in obj['segGroups']:
            feature_list = []
            for fe_key, fe_value in nodes.items():
                if fe_key in ['id', 'dominantNormal', 'obb']:
                    if fe_key == 'id':
                        hobe_dic[fe_value] = k
                        id.append(fe_value)
                        k += 1
                        # print(fe_value)
                    if isinstance(nodes[fe_key], (list)):

                        for i in nodes[fe_key]:
                            feature_list.append(i)
                    elif isinstance(nodes[fe_key], dict):
                        for i in list(nodes[fe_key].keys()):
                            if i == 'normalizedAxes': continue
                            for j in nodes[fe_key][i]:
                                feature_list.append(j)

            feature = torch.tensor(feature_list)
            # labels = torch.tensor(label_list,dtype=torch.long)
            object_feature = torch.cat([object_feature, feature])
            # y = torch.cat([y,labels])
        a = hobe_dic
        # print(a)
        return object_feature,hobe_dic

    def edge_index_get(self,rel_num,hobe_dic,scan_id):
        edge_index = torch.zeros((2, rel_num), dtype=torch.long)
        for index,relationships in enumerate(self.relation_dataset[scan_id]['relationships']):
            # print(hobe_dic)
            # sys.exit()
            edge_index[0, index] = hobe_dic[relationships[0]]
            edge_index[1, index] = hobe_dic[relationships[1]]
            #relationships[2] 는 관계다.
            '''
            {'relationships': [[2, 1, 14, 'attached to'], [3, 2, 17, 'hanging on'], 
                [5, 16, 14, 'attached to'], 
            '''
        return edge_index
            # edge_index[0, index] = hobe_dic[node[0]]
            # edge_index[1, index] = hobe_dic[node[1]]

    def node2vec_get(self,node_feature):
        node2node_vec = torch.FloatTensor([])
        node2node_dic = {}
        count_idx = 0
        for i in range(node_feature.size()[0]):
            for j in range(node_feature.size()[0]):
                # sourceNode_idx = edge_index.squeeze(dim=0)[0][i]
                # targetNode_idx = edge_index.squeeze(dim=0)[1][i]
                if i==j:
                    continue
                te = '{}_{}'.format(i,j)
                node2node_dic[te] = count_idx
                sourceNode_feature = node_feature[i,:]
                targetNode_feature = node_feature[j,:]
                count_idx += 1
                temp = torch.concat([sourceNode_feature, targetNode_feature], dim=0)
                node2node_vec = torch.cat([node2node_vec,temp])
        node2node_vec = node2node_vec.view(-1, 18)
        return node2node_vec,node2node_dic


    def edge_label(self,node2vec,node2node_dic,scan_id,hobe_dic):
        edge_label = torch.zeros(node2vec.size()[0],self.total_rel_num).long()
        # print(node2node_dic.keys())
        # print(hobe_dic)
        # print(node2vec.size())
        # print(scan_id)
        for index, relationships in enumerate(self.relation_dataset[scan_id]['relationships']):
            # print('rel[0]',relationships[0],'\trel[1]',relationships[1])
            node2node_idx = node2node_dic['{}_{}'.format(hobe_dic[relationships[0]],hobe_dic[relationships[1]])]
            edge_label[node2node_idx][relationships[2]-1] = 1
            # print('{}_{} -> 1 '.format(hobe_dic[relationships[0]],hobe_dic[relationships[1]]))
        # print(edge_label.size())
        return edge_label

# def collate_fn(data):
#     images, instances = zip(*data)
#     batch = Batch(instances)
#     td = batch.as_tensor_dict()
#     return td


def hobe_test(model):
    model.eval()

    model = model
    test_loss_result = {'right_left':[],'infront_behind':[]}
    test_acc_result = []
    with torch.no_grad():

        for data in test_loader:
            obj_feature = data['obj_feature']
            obj_feature = obj_feature.squeeze(0).type(torch.LongTensor)
            obj_feature = torch.Tensor(MinMaxScaler().fit_transform(obj_feature)).to(device)

            edge_index = data['edge_index']
            edge_index = edge_index.squeeze(0).type(torch.LongTensor).to(device)

            edge_label = data['edge_label'].to(device)
            edge_label = edge_label.squeeze(dim=0) #배치사이즈 죽이기

            node2dic = data['node2dic']
            #나중에 이걸로 최종 결과 출력할 때 사용할 듯

            predict_dic =model(obj_feature, edge_index)

            i_b_predict = predict_dic['i_b'].squeeze(dim=0) #i_b = infront - behind
            r_l_predict = predict_dic['r_l'].squeeze(dim=0) #r_l = right - left

            #관계 하나하나씩 가져오기
            #라벨 데이터임
            right_left = edge_label[:,1]
            infront_behind = edge_label[:,3]

            r_l_loss = F.cross_entropy(r_l_predict, right_left) #4는 임의의 수임. edge_label에 적절한 관계 번호로 바꿔야됌.
            i_b_loss = F.cross_entropy(i_b_predict, infront_behind)

            #Loss 저장소
            test_loss_result['right_left'].append(r_l_loss.item())
            test_loss_result['infront_behind'].append(i_b_loss.item())

            #정확도 저장소
            r_l_max = torch.max(r_l_predict, dim=1)[1]
            r_l_score = r_l_max[r_l_max == right_left]
            r_l_acc = r_l_score.size()[0]/r_l_max.size()[0]

            i_b_max = torch.max(i_b_predict, dim=1)[1]
            i_b_score = i_b_max[i_b_max == infront_behind]
            i_b_acc = i_b_score.size()[0] / i_b_max.size()[0]

            test_acc_result.append(r_l_acc)
            test_acc_result.append(i_b_acc)

    # r_l_loss = np.nan_to_num(np.array(test_loss_result['right_left']))
    # i_b_loss = np.nan_to_num(np.array(test_loss_result['infront_behind']))

    print('right_left_loss_avg = {:.3f}%\n infront_behind_loss_avg = {:.3f}%\ntest_acc_avg = {:.3f}%'.format(
        sum(test_loss_result['right_left'])/len(test_loss_result['right_left']),
        sum(test_loss_result['infront_behind']) / len(test_loss_result['infront_behind']),
         sum(test_acc_result)/len(test_acc_result)))


if __name__ =='__main__':

    relation_scan_id = {}
    total_semseg = sorted(os.listdir(path))
    # print('semseg 개수 : ',len(total_semseg))

    train_semseg = total_semseg[:-300]
    test_semseg = total_semseg[-50:]
    # print(train_semseg)
    # print(test_semseg)

    # 전처리 아닌 전처리.
    # relationships.json 파일 먼저 읽기.
    with open("./relationships.json", "r") as rel_json:
        rel_json = json.load(rel_json)
    for data in rel_json['scans']:
        relation_scan_id[data['scan']] = data
    # print(relation_scan_id)

    # Train 파일 전처리
    for semseg_path in train_semseg:
        temp_path = '{}/{}'.format(path,semseg_path)
        with open(temp_path, "r") as semseg_json:
            sem_json = json.load(semseg_json)

        if sem_json['scan_id'] not in relation_scan_id:
            print('없어 ..-> 삭제 : ',temp_path,' : ',sem_json['scan_id'],':',semseg_path)
            train_semseg.remove(semseg_path)
    # print(len(train_semseg))
    # print(train_semseg)

    #Test 파일 전처리
    for semseg_path in test_semseg:
        temp_path = '{}/{}'.format(path, semseg_path)
        with open(temp_path, "r") as semseg_json:
            sem_json = json.load(semseg_json)

        if sem_json['scan_id'] not in relation_scan_id:
            print('없어 ..-> 삭제 : ', temp_path, ' : ', sem_json['scan_id'], ':', semseg_path)
            test_semseg.remove(semseg_path)
    # print(len(test_semseg))
    # print(test_semseg)

    #relationships.json 파일  scan_id !=  semseg.json 파일  scan_id 일 경우에는 파일 리스트에서 지움.

    train_datasets = HobeDataset(semseg_file_list=train_semseg,path=path,relation_dataset=relation_scan_id)
    test_datasets = HobeDataset(semseg_file_list=test_semseg,path=path,relation_dataset=relation_scan_id)

    train_loader = DataLoader(dataset=train_datasets,batch_size=1,shuffle=True)
    test_loader = DataLoader(dataset=test_datasets,batch_size=1,shuffle=True)

    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! 여기서 부터 학습 관련한 코드 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    from gcn_model import *
    from torch.nn import functional as F
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print('device = ',device)
    # model = GCN_network(in_channel=18, node_feature_num=18, out_channel=18)
    model = Net(in_channel=9,node_feature_num=9).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    # criterion = F.cross_entropy() #nn.CrossEntropyLoss()
    for epoch_num in range(3):
        for data in train_loader:

            optimizer.zero_grad()

            obj_feature = data['obj_feature']
            obj_feature = obj_feature.squeeze(0).type(torch.LongTensor)
            obj_feature = torch.Tensor(MinMaxScaler().fit_transform(obj_feature)).to(device)


            edge_index = data['edge_index']
            edge_index = edge_index.squeeze(0).type(torch.LongTensor).to(device)

            edge_label = data['edge_label'].to(device)
            print("edge_label", edge_label)
            edge_label = edge_label.squeeze(dim=0) 
            # batch size 줄이기
            node2dic = data['node2dic']
            '''
            
            data['node2dic'] = > '58_55': tensor([3419]), '58_56': tensor([3420]), '58_57': tensor([3421] 
            
            '''

            obj_size = obj_feature.size()
            # print(obj_feature.size())
            # print(edge_index.size())
            # print(obj_feature[:,:10,:].size())

            # print(obj_feature)
            # print(edge_index)
            # sys.exit()
            temp_predict = model(obj_feature,edge_index)
            # print(temp_predict)
            # 예측 값 : (batch,1600,3) or (batch,1600,2)
            # 정답 값 : (batch,1600)

            # 각 결과에 따른 (1600, 2또는3또는4 ) 중에 dim=1중에 가장 큰 녀석을 뽑아서 --> 40x40으로 바꿈.
            # 그럼 40개의 노드들간의 관계를 매핑할 수 있음
            temp1 = temp_predict['i_b'].squeeze(dim=0)
            a = torch.max(temp1,dim=1)[1]

            temp2 = temp_predict['r_l'].squeeze(dim=0)
            b = torch.max(temp2,dim=1)[1]
            
            # print('temp1 .size = ', temp1.size())
            # print('temp1 .size = ', temp2.size())
            # print('정답 edge _label 값 size = ',edge_label[:,:,4].squeeze().size())


            # print("temp1-------->", temp1.size(), temp1)
            # print("temp_edge_label-------->", temp_edge_label[:, 3].size(), temp_edge_label[:, 3])
            # print("temp2-------->", temp2.size(), temp2)
            # print("temp_edge_label-------->", temp_edge_label[:, 1].size(), temp_edge_label[:, 1])
            r_l_loss = F.cross_entropy(temp1,edge_label[:,1])
            i_b_loss = F.cross_entropy(temp2, edge_label[:, 3])

            total_loss = r_l_loss + i_b_loss

            print('-->total_loss : {:.3f}'.format(total_loss.item()))
            total_loss.backward()
            optimizer.step()
        print('---- test 중이야 .... ---- \n')
        hobe_test(model)
        print('---- test 끝  ---- \n')
        if lr_scheduler is not None:
            lr_scheduler.step()

