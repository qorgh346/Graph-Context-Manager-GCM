from Mos_custom_datasets import Mos_HobeDataset
from torch.utils.data import DataLoader
import os
import json
import graphviz

''' 관계 총 7개
none
inFrontOf
behind
toTheRightOf
toTheLeftOf
near
far
'''

rel_lists = {0:'inFrontOf',1:'behind',2:'toTheRightOf',3:'toTheLeftOf',4:'near',5:'far'}

def init_datasets(path_object):
    hobe_dic = {}
    id = []
    k = 0
    total_feature = {}
    object_path = os.path.join(path_object,'objects.json')
    rel_path = os.path.join(path_object, 'relationships.json')
    # print(object_path)
    # sys.exit()
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
            #feature_list.append(temp_emb[group['status']]) 현재 에러 있음.
            node_number += 1
            temp_ += 0.001
        # print(node_number)
        # sys.exit()
        feature = torch.tensor(feature_list)
        object_feature = torch.cat([object_feature, feature])
        total_feature[nodes['scenario']] = object_feature.view(node_number,-1)

    # edge index 만들기
    with open(rel_path, "r") as rel_json:
        rel_list = json.load(rel_json)

    scenario_relationships = {}
    for rel in rel_list['scenarios']:
        # print(rel['relationships'])
        rel_num = len(rel['relationships'])
        edge_index = torch.zeros((2, rel_num), dtype=torch.long)
        for index, relationships in enumerate(rel['relationships']):
            if relationships[2] == 0:
                continue #none 이라는 관계는 edge_index에 담지도 않아~
            edge_index[0, index] = relationships[0]-1 #index 맞춰줄려고 0번은 ARM_LIFT01, 1번은 AMR_LIFT02 ...
            edge_index[1, index] = relationships[1]-1
        scenario_relationships[rel['scenario']] = (edge_index,rel['relationships'])


    scenario_lists = list(total_feature.keys())
    # print(scenario_lists)
    return scenario_lists,total_feature,scenario_relationships




def visual_graph(data,scenario_id,mode='Test'):
    #visual 시자악.
    # print(data)
    node_color_list = ['aliceblue', 'antiquewhite', 'cornsilk3', 'lightpink']
    # visual 시자악.
    obj_list = ['AMR_LIFT01','AMR_LIFT02','TOW_LIFT01','TOW_LIFT02']
    digraph1  = graphviz.Digraph(comment='The Scene Graph')

    for i, node in enumerate(obj_list):
        digraph1.attr('node', fillcolor=node_color_list[i], style='filled')
        digraph1.node(str(i), node)

    digraph1.attr('edge', fontname='Sans', color='black', style='filled')
    for i, edge in enumerate(data):
        temp_data = data[edge].split('_')
        source_node = temp_data[0]
        target_node = temp_data[1]
        edge_label = temp_data[2]

        digraph1.edge(str(source_node), str(target_node), str(edge_label))
    # save_graph_as_svg(digraph1,file_name)
    try:
        digraph1.render('./result_graph/visual_{}.gv'.format(scenario_id), view=True)  # 이름 바꾸기
    except:
        pass

def hobe_visual(model):
    #평가 아님 비쥬얼임
    model.eval()
    model = model
    temp = 0
    with torch.no_grad():
        for data in test_loader:

            obj_feature = data['obj_feature'].to(device)
            obj_feature = obj_feature.squeeze(0)

            edge_index = data['edge_index']
            edge_index = edge_index.squeeze(0).type(torch.LongTensor).to(device)

            edge_label = data['edge_label'].to(device)
            edge_label = edge_label.squeeze(dim=0)[:,1:] #배치사이즈 죽이기

            node2dic = data['node2dic']
            node2dic = {v.item():k for k,v in node2dic.items()}
            # {0: '0_1', 1: '0_2', 2: '0_3', 3: '1_0', 4: '1_2', 5: '1_3', 6: '2_0', 7: '2_1', 8: '2_3', 9: '3_0', 10: '3_1', 11: '3_2'}
            #나중에 이걸로 최종 결과 출력할 때 사용할 듯
            g_t_visual_data = {}
            count = 0
            for i in range(edge_label.size()[0]):
                for j in range(edge_label.size()[1]):
                    if edge_label[i, j] == 1:
                        node_node = node2dic[i]
                        result = '{}_{}'.format(node_node, rel_lists[j])
                        g_t_visual_data[count] = result
                        count += 1
            visual_graph(g_t_visual_data,'{}_{}'.format(data['scenario'],'GT'),mode='GT')

            predict_dic =model(obj_feature, edge_index)
            i_b_predict = predict_dic['i_b'].squeeze(dim=0) #i_b = infront - behind

            predict_result = torch.max(i_b_predict,dim=1)[1]
            visual_data = {}
            for idx,i in enumerate(predict_result):
                Node_to_Node = node2dic[idx]
                result = '{}_{}'.format(Node_to_Node,rel_lists[i.item()])
                visual_data[idx] = result

            visual_graph(visual_data,data['scenario'],mode='predicet') #visual_graph에서 이제 그래프 그려주기
                # print('{}번 ♥ {}번 = {}관계'.format(sourceNode,targetNode,rel_lists[i.item()]))
            # print('\n ------------------------------------------ \n ')
            temp +=1


if __name__ == '__main__':
    #데이터셋 준비과정
    path_object = 'bae_datasets'
    scenario_lists,total_feature,scenario_relationships = init_datasets(path_object)
    # print(scenario_lists)
    # print(total_feature)
    # print(scenario_relationships)


    train_datasets = Mos_HobeDataset(item_list=scenario_lists[:300], object_features=total_feature, relationships_list=scenario_relationships)
    test_datasets = Mos_HobeDataset(item_list=scenario_lists[300:], object_features=total_feature, relationships_list=scenario_relationships)

    train_loader = DataLoader(dataset=train_datasets, batch_size=1, shuffle=True)
    test_loader = DataLoader(dataset=test_datasets, batch_size=1, shuffle=True)

    a = next(iter(train_loader))
    print('노드 피쳐들 size -> ',a['obj_feature'].size())
    print('Edge_index size -> ',a['edge_index'].size())
    print('edge_label 정답 GT 파일 -> ',a['edge_label'].size())
    print('나중에 필요한 0_1 이런거 -> ',a['node2dic'])

    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! 여기서 부터 학습 관련한 코드 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

    from base.gcn_model import *
    from torch.nn import functional as F

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print('device = ', device)

    model = Net(in_channel=6, node_feature_num=6,num_layers=5).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.2)

    for epoch_num in range(20):
        result_loss = []
        for data in train_loader:
            optimizer.zero_grad()
            scenario = data['scenario']

            # sys.exit()
            obj_feature = data['obj_feature'].to(device)
            obj_feature = obj_feature.squeeze(0)

            edge_index = data['edge_index']
            edge_index = edge_index.squeeze(0).type(torch.LongTensor).to(device)

            edge_label = data['edge_label']
            edge_label = edge_label.squeeze(dim=0)[:,1:].to(device)


            temp_edge_label = torch.randint(0,2,size=(1,12,6)).long() # 0이랑 1 사이의 값이 너무 적어서..
            temp_edge_label = temp_edge_label.squeeze(dim=0).to(device)
            node2dic = data['node2dic']
            # print(node2dic)

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
            temp_predict = model(obj_feature, edge_index)
            # print(type(temp_predict))
            # print(temp_predict)
            # print(temp_predict)
            # 예측 값 : (batch,1600,3) or (batch,1600,2)
            # 정답 값 : (batch,1600)

            # 각 결과에 따른 (1600, 2또는3또는4 ) 중에 dim=1중에 가장 큰 녀석을 뽑아서 --> 40x40으로 바꿈.
            # 그럼 40개의 노드들간의 관계를 매핑할 수 있음
            temp1 = temp_predict['i_b'].squeeze(dim=0).type(torch.FloatTensor)
            # print(type(temp1.type(torch.FloatTensor)))
            # sys.exit()
            # print(temp1)
            # a = torch.max(temp1, dim=1)
            # print(a)
            # sys.exit()

            # temp2 = temp_predict['r_l'].squeeze(dim=0)
            # b = torch.max(temp2, dim=1)[1]

            # print('temp1 .size = ', temp1.size())
            # print('temp1 .size = ', temp2.size())
            # print('정답 edge _label 값 size = ',edge_label[:,:,4].squeeze().size())

            #원래 코드
            # r_l_loss = F.cross_entropy(temp1, edge_label[:, 3])
            # i_b_loss = F.cross_entropy(temp2, edge_label[:, 1])

            #테스트 코드
            # print(edge_label[:, 3].view(-1,1).size())
            # print(temp1.size())
            # sys.exit()
            # r_l_loss = F.cross_entropy(temp1, edge_label[:, 3])
            i_b_loss = F.binary_cross_entropy(temp1, edge_label.type(torch.FloatTensor))
            # print(r_l_loss.item())
            # print(i_b_loss)
            total_loss =  i_b_loss
            result_loss.append(total_loss)
            # print('-->total_loss : {:.3f}'.format(total_loss.item()))

            total_loss.backward()
            optimizer.step()
        print('{} epoch = train_loss_avg = {}'.format(epoch_num,sum(result_loss)/len(result_loss)))

        if lr_scheduler is not None:
            lr_scheduler.step()
    hobe_visual(model) #visual 하는 부분임.
