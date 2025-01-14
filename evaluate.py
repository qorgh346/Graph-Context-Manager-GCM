import torch
import graphviz

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
rel_lists = {0:'inFrontOf',1:'behind',2:'toTheRightOf',3:'toTheLeftOf',4:'near',5:'far'}

def eval_visualization(model, test_loader):

    model.eval()
    model = model
    temp = 0
    with torch.no_grad():
        for data in test_loader:

            obj_feature = data['obj_feature'].to(device)
            obj_feature = obj_feature.squeeze(0)

            edge_index = data['edge_index']
            print(edge_index)
            edge_index = edge_index.type(torch.LongTensor).to(device)

            edge_label = data['edge_label'].to(device)
            edge_label = edge_label.squeeze(dim=0)[:,1:] #배치사이즈 1 고정정

            node2dic = data['node2dic']
            node2dic = {v.item():k for k,v in node2dic.items()}
            # {0: '0_1', 1: '0_2', 2: '0_3', 3: '1_0', 4: '1_2', 5: '1_3', 6: '2_0', 7: '2_1', 8: '2_3', 9: '3_0', 10: '3_1', 11: '3_2'}

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

            predict_dic =model.process(obj_feature, edge_index)
            i_b_predict = predict_dic['i_b'].squeeze(dim=0) #i_b = infront - behind

            predict_result = torch.max(i_b_predict,dim=1)[1]
            visual_data = {}
            for idx,i in enumerate(predict_result):
                Node_to_Node = node2dic[idx]
                result = '{}_{}'.format(Node_to_Node,rel_lists[i.item()])
                visual_data[idx] = result

            visual_graph(visual_data,data['scenario'],mode='predicet') 
            temp +=1

def visual_graph(data,scenario_id,mode='Test'):

    node_color_list = ['aliceblue', 'antiquewhite', 'cornsilk3', 'lightpink']

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
    try:
        digraph1.render('./result_graph/visual_{}.gv'.format(scenario_id), view=True)  # 이름 바꾸기
    except:
        pass