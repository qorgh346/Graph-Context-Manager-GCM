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


def init_datasets(path_object):
    hobe_dic = {}
    id = []
    k = 0
    total_feature = {}
    object_path = os.path.join(path_object,'objects.json')
    rel_path = os.path.join(path_object, 'relationships.json')

    with open(object_path, "r") as object_json:
        object_list = json.load(object_json)

    for nodes in object_list['scenarios']:
        feature_list = []
        object_feature = torch.FloatTensor([])
        node_number = 0
        temp_ = 1.2
        for idx,group in enumerate(nodes['groups']):
            feature_list.append(group['pose'][0]+temp_)
            feature_list.append(group['pose'][1]+temp_)
            feature_list.append(group['vertext']+temp_)
            feature_list.append(group['battery'])
            feature_list.append(group['speed']*100+temp_)
            feature_list.append(group['maxPayLoad'])
            #feature_list.append(temp_emb[group['status']]) 
            node_number += 1
            temp_ += 0.001
        feature = torch.tensor(feature_list)
        object_feature = torch.cat([object_feature, feature])
        total_feature[nodes['scenario']] = object_feature.view(node_number,-1)

    # edge index 만들기
    with open(rel_path, "r") as rel_json:
        rel_list = json.load(rel_json)

    scenario_relationships = {}
    for rel in rel_list['scenarios']:
        rel_num = len(rel['relationships'])
        edge_index = torch.zeros((2, rel_num), dtype=torch.long)
        for index, relationships in enumerate(rel['relationships']):
            if relationships[2] == 0:
                continue #none 이라는 관계 Edge Index 제외외
            edge_index[0, index] = relationships[0]-1 
            #index 맞추기기 0번은 ARM_LIFT01, 1번은 AMR_LIFT02 ...
            edge_index[1, index] = relationships[1]-1
        scenario_relationships[rel['scenario']] = (edge_index,rel['relationships'])


    scenario_lists = list(total_feature.keys())

    return scenario_lists,total_feature,scenario_relationships



class GCMDataset(Dataset):
    def __init__(self,item_list,object_features,relationships_list):
        self.object_features = object_features
        self.relationships_list = relationships_list
        self.items = item_list


        f = open("./data/raw/relations.txt", 'r')
        lines = f.readlines()
        self.total_rel_num = len(lines)
        f.close()

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        scenario = self.items[index]
        output = {}

        obj_feature = self.node_feature_get(scenario)
        node_num = obj_feature.size()[0] #노드 개수
        edge_index = self.edge_index_get(node_num)
        node2vec, node2dic = self.node2vec_get(obj_feature)
        edge_label = self.edge_label(node2vec, node2dic,scenario)

        output['obj_feature'] = obj_feature
        output['edge_index'] = edge_index #(nodeNum*nodeNum-1 , 2)
        output['node2vec'] = node2vec
        output['edge_label'] = edge_label
        output['node2dic'] = node2dic
        output['scenario'] = scenario
        return output

    def node_feature_get(self,scenario):
        object_feature = self.object_features[scenario]
        return object_feature

    def edge_index_get(self,node_num):
        edge_indices = list()
        for i in range(node_num):
            for j in range(node_num):
                if i==j: continue
                edge_indices.append([i,j])
        return edge_indices

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
                sourceNode_feature = node_feature[i,:3]
                targetNode_feature = node_feature[j,:3]
                count_idx += 1
                temp = torch.concat([sourceNode_feature, targetNode_feature], dim=0)
                node2node_vec = torch.cat([node2node_vec,temp])
        node2node_vec = node2node_vec.view(node_feature.size()[1]*2,-1)
        return node2node_vec,node2node_dic


    def edge_label(self,node2vec,node2node_dic,scenario):
        edge_label = torch.zeros(node2vec.size()[0],self.total_rel_num).long()
        for index, relationships in enumerate(self.relationships_list[scenario][1]):
            node2node_idx = node2node_dic['{}_{}'.format(relationships[0]-1,relationships[1]-1)]
            edge_label[node2node_idx][relationships[2]] = 1
        return edge_label
