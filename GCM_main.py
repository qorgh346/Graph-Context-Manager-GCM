import sys

import torch
import torch_geometric
from models.CloudGCM_Network import *
from Datasets.GCMDataLoader import GCMDataset
from torch.utils.data import DataLoader
import copy

def main():

    # Argument 설정
    parser = argparse.ArgumentParser(description="GCM Main")
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'eval'],
                        help="Default is 'train'")
    parser.add_argument('--lr', type=float, default=0.0001, help="Learning rate")
    parser.add_argument('--epochs', type=int, default=150, help="Epoch")
    parser.add_argument('--batch_size', type=int, default=8, help="Batch size")
    parser.add_argument('--gcn_layers', type=int, default=2, help="GCN layers Num")
    parser.add_argument('--pretrained', type=str, default='best_model.pt', help="Best Model")
    
    args = parser.parse_args()

    # 하이퍼파라미터 초기화
    hy_param = {
        'lr': args.lr,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'gcn_layers': args.gcn_layers
        'best_model': args.pretrained
    }

    # 모델 초기화
    network = GCMModel('GCMModel', hy_param, norm_flag=True)
    print("Model:\n", network)

    run_process(mode=args.mode, hy_param=hy_param)


def build_datasets(hy_param):
    # Update Temporal Dataset 2022.05.19
    # path = './mos_timestamp_jsons'
    #     train_test_path = '../split_dataset_list'
    #     a = MosDataset(root=path,split_path=train_test_path,mode='train')
    #     b = DataLoader(dataset=a,batch_size=1)
    #     for i in b:
    #         print(i["nodes"].size())
    #         print(i['edge_index'].size())
    #         print(i['meta_data']['GT'].size())
    #         sys.exit()
    if hy_param['temporal']:
        # build Datasets
        train_datasets = MosDataset(root=hy_param['path'], split_path=hy_param['train_test_path'], mode='train',
                                    Normalization=True)
        test_datasets = MosDataset(root=hy_param['path'], split_path=hy_param['train_test_path'], mode='test',
                                   Normalization=True)
        val_datasets = MosDataset(root=hy_param['path'], split_path=hy_param['train_test_path'], mode='val',
                                  Normalization=True)
        # build Dataloader
        trainDataLoader = DataLoader(dataset=train_datasets, batch_size=1, shuffle=True)
        testDataLoader = DataLoader(dataset=test_datasets, batch_size=1, shuffle=False)
        valDataLoader = DataLoader(dataset=val_datasets, batch_size=1, shuffle=True)
    else:
        # build Datasets
        train_datasets = MosDataset(root=hy_param['path'], split_path=hy_param['train_test_path'], mode='train',
                                    Normalization=True)
        test_datasets = MosDataset(root=hy_param['path'], split_path=hy_param['train_test_path'], mode='test',
                                   Normalization=True)
        val_datasets = MosDataset(root=hy_param['path'], split_path=hy_param['train_test_path'], mode='val',
                                  Normalization=True)
        # build Dataloader
        trainDataLoader = DataLoader(dataset=train_datasets, batch_size=1, shuffle=True)
        testDataLoader = DataLoader(dataset=test_datasets, batch_size=1, shuffle=False)
        valDataLoader = DataLoader(dataset=val_datasets, batch_size=1, shuffle=True)

    return trainDataLoader,testDataLoader,valDataLoader

def run_process(mode,hy_param,model_path='./save_models'):

    #build dataset
    trainDataLoader,testDataLoader,valDataLoader =  build_datasets(hy_param)

    if mode == 'train':
        best_acc = 0.0
        for epoch in range(0, hy_param['epochs']):
            running_loss = 0.0
            running_corrects = 0
            num_cnt = 0

            #Train Start
            network.train()
            for item in trainDataLoader:
                x = item['nodes'].squeeze(dim=0) #[1,4,6,18]
                edge_index = item['edge_index'].squeeze(dim=0) #[1,2,12]
                gt_label = item['meta_data']['GT'].squeeze(dim=0) #[1,12,3,18]
                # print('id : ', item['meta']['id'])

                logs, predict_value = network.process('train', x, edge_index, gt_label)

                running_loss += logs[1]
                running_corrects += logs[3]
                num_cnt += 1
                # print(logs)
                # print(predict_value)
            epoch_train_loss = running_loss / num_cnt
            epoch_train_acc = running_corrects / num_cnt
            # 에폭 loss & acc 계산
            print('{}/{} train(loss) = {:.5f} \t train(acc) = {:.5f}'.format(epoch, hy_param['epochs'], epoch_train_loss,
                                                               epoch_train_acc))

            #Validation Start
            # print('############ Start Val ############## ')
            running_loss = 0.0
            running_corrects = 0
            num_cnt = 0
            for item in valDataLoader:
                with torch.no_grad():
                    network.eval()
                    x = item['nodes'].squeeze(dim=0)  # [1,4,6,18]
                    edge_index = item['edge_index'].squeeze(dim=0)  # [1,2,12]
                    gt_label = item['meta_data']['GT'].squeeze(dim=0)  # [1,12,3,18]
                    # print('id : ', item['meta']['id'])
                    
                    logs, predict_value = network.process('val', x, edge_index, gt_label)
                    running_loss += logs[1]
                    running_corrects += logs[3]
                    num_cnt += 1
            epoch_val_loss = running_loss / num_cnt
            epoch_val_acc = running_corrects / num_cnt


            #에폭 loss & acc 계산
            print('{}/{} val(loss) = {:.5f} \t val(acc) = {:.5f}'.format(epoch, hy_param['epochs'], epoch_val_loss,
                                                               epoch_val_acc))

            if epoch_val_acc > best_acc:
                best_idx = epoch
                best_acc = epoch_val_acc
                best_model_wts = copy.deepcopy(network.state_dict())
                print('==> best model saved - %d / %.1f' % (best_idx, best_acc))
                check_point = torch.save({'epoch': epoch,'model_state_dict': network.state_dict(),
                'loss': epoch_val_loss}, './last_checkPoint')
            if epoch % 10 == 0:
                torch.save(network.state_dict(),
                       '{}/{}'.format('./save_models', 'model_epoch{}.pt'.format(epoch)))
        network.load_state_dict(best_model_wts)

        torch.save(network.state_dict(),
                   '{}/{}'.format('./save_models','bestmodel.pt'))
        print('model saved')


    else:
        # Test Start
        #load_model
        model_file_name = hy_param['best_model'] #'bestmodel.pt'
        network.load_state_dict(torch.load(os.path.join(model_path,model_file_name)))
        # print(network)
        #
        running_loss = 0.0
        running_corrects = 0
        num_cnt = 0

        # 그래프 시각화 viz 객체 생성
        from utils import Graph_Vis
        import time
        for item in testDataLoader:
            gv = Graph_Vis.GraphVIS('C:/Users/ailab/Desktop/graphVis')
            gv.init_json()
            [gv.set_Node(robot) for robot in ['AMR_LIFT1', 'AMR_LIFT2', 'AMR_TOW1', 'AMR_TOW2']]
            print('Start')
            with torch.no_grad():
                network.eval()
                x = item['x'].squeeze(dim=0)
                # print('id : ', item['meta']['id'])


                edge_index = item['edge_index'].squeeze(dim=0)
                gt_label = item['meta']['GT'].squeeze(dim=0)
                logs, predict_value = network.process('test', x, edge_index, gt_label)

                running_loss += logs[1]
                running_corrects += logs[3]
                num_cnt += 1

                relation_mapping_idx ={ v.item():k for k,v in item['meta']['relation_mapping_idx'].items()}
                edge_mapping_idx = {v.item(): k for k, v in item['meta']['edge_mapping_idx'].items()}
                robot_mappint_idx = {v.item(): k for k, v in item['meta']['robot_mappint_idx'].items()}
                # print(robot_mappint_idx)
                # print(relation_mapping_idx)
                threshold = 0.5
                predict = predict_value['pred_rel'] >= threshold

                for pos in predict.nonzero():
                    row = pos[0]
                    col = pos[1]
                    src_obj = edge_mapping_idx[row.item()].split('_')
                    subject = robot_mappint_idx[int(src_obj[0])]
                    object = robot_mappint_idx[int(src_obj[1])]
                    predicate = relation_mapping_idx[col.item()]
                    print('예측 : \t {}_{}_{}'.format(subject,predicate,object))
                    gv.set_Edge(subject,predicate,object)

                    # sys.exit()
            gv.write_json()

        test_loss = running_loss / num_cnt
        test_acc = running_corrects / num_cnt

        # Test loss & acc 계산
        print('test(loss) = {:.5f} \t test(acc) = {:.5f}'.format(test_loss,
                                                                     test_acc))

if __name__ == '__main__':
    main()