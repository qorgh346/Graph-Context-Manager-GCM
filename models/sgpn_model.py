from networks.network_TripletGCN import TripletGCNModel
from networks.network_RelNet import RelNetFeat, RelNetCls
import torch
from torch import nn
import utils. op_utils as op_utils
import torch.optim as optim
import torch.nn.functional as F
import sys
class SGPNModel(nn.Module):
    def __init__(self, name:str, num_obj_cls, num_rel_cls, hy_param):
        super(SGPNModel, self).__init__()

        self.name = name

        # Build model
        models = dict()

        # Relationship Encoder
        models['rel_encoder'] = RelNetFeat(
            input_size=hy_param['dim_obj'] * 2,
            output_size=hy_param['edge_feature_size'],
            batch_norm=True,
            init_weights=True
        )

        #공간적 맥락 추론(Spatial Context Reasoning) 단계계
        # Triplet GCN
        models['triplet_gcn'] = TripletGCNModel(
            num_layers=hy_param['num_layer'],
            dim_node=hy_param['dim_obj'],
            dim_edge=hy_param['edge_feature_size'],
            dim_hidden=hy_param['gcn_hidden_feature_size']
        )

        #맥락 서술자 예측(Context Predicate Prediction) 단계계

        models['rel_predictor'] = RelNetCls(
            rel_num=num_rel_cls,
            in_size=hy_param['edge_feature_size'],
            batch_norm=True,
            drop_out=True
        )
        params = list()
        print('==trainable parameters==')
        for name, model in models.items():
            self.add_module(name, model)
            params += list(model.parameters())
            print(name,op_utils.pytorch_count_params(model))
        print('')
        self.optimizer = optim.Adam(params=params, lr=hy_param['lr'],)
        self.optimizer.zero_grad()

    def forward(self, obj_feature, rel_feature, edges_index):

        edge_feature = self.rel_encoder(rel_feature)  # 특징 추출 과정

        gcn_obj_feature, gcn_rel_feature = self.triplet_gcn(obj_feature, edge_feature, edges_index)
        gcn_rel_feature = gcn_rel_feature.squeeze(0)

        rel_cls = self.rel_predictor(gcn_rel_feature)

        return rel_cls, obj_feature, edge_feature, gcn_rel_feature

    def process(self, obj_feature, rel_feature, edges_index, gt_obj_cls, gt_rel_cls,mode='Train',weights_obj=None, weights_rel=None):
        if mode == 'Train':
            rel_pred, obj_feature, edge_feature, gcn_rel_feature = self(obj_feature, rel_feature, edges_index)
            gt_rel_cls = gt_rel_cls.detach().cpu()
            rel_pred = rel_pred.cpu()
            loss_rel = F.binary_cross_entropy(rel_pred, gt_rel_cls)

            self.backward(loss_rel)
            logs = ("Loss/rel_loss", loss_rel.detach().item())
            return logs, rel_pred.detach(),
        else: #mode == 'Test'
            rel_pred, obj_feature, edge_feature, gcn_rel_feature = self(obj_feature, rel_feature, edges_index)
            rel_pred = rel_pred.cpu()
            return 'Test',rel_pred

    def backward(self, loss):
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

if __name__ == '__main__':
    use_dataset = False

    if not use_dataset:
        num_obj_cls = 4
        num_rel_cls = 26
    # else:
    #     from src.dataset_builder import build_dataset
    #
    #     config.dataset.dataset_type = 'rio_graph'
    #     dataset = build_dataset(config, 'validation_scans', True, multi_rel_outputs=True, use_rgb=False,
    #                             use_normal=False)
    #     num_obj_cls = len(dataset.classNames)  # 160
    #     num_rel_cls = len(dataset.relationNames)  # 8

    # build model
    # network = SGPNModel('SGPNModel', num_obj_cls, num_rel_cls)

    if not use_dataset:

        num_node = 4
        dim_obj = 6
        num_rel = num_node * num_node - num_node
        dim_rel = 256
        obj_feat = torch.rand([num_node, dim_obj])
        rel_feat = torch.rand([num_rel, dim_rel])
        edges_index = torch.zeros(num_rel, 2, dtype=torch.long)
        counter = 0
        for i in range(num_node):
            if counter >= edges_index.shape[0]: break
            for j in range(num_node):
                if i == j: continue
                if counter >= edges_index.shape[0]: break
                edges_index[counter, 0] = i
                edges_index[counter, 1] = j
                counter += 1
        print(edges_index)
        obj_gt = torch.randint(0, num_obj_cls - 1, (num_node,))
        rel_gt = torch.randint(0, num_rel_cls - 1, (num_rel,))

        # rel_gt
        adj_rel_gt = torch.rand([num_node, num_node, num_rel_cls])
        rel_gt = torch.zeros(num_rel, num_rel_cls, dtype=torch.float)
        for e in range(edges_index.shape[0]):
            i, j = edges_index[e]
            for c in range(num_rel_cls):
                if adj_rel_gt[i, j, c] < 0.5: continue
                rel_gt[e, c] = 1

        edges_index = edges_index.t().contiguous()
        network.process(obj_feat, rel_feat, edges_index, obj_gt, rel_gt)

    for i in range(100):
        # if use_dataset:
        #     scan_id, instance2mask, obj_points, rel_points, obj_gt, rel_gt, edges = dataset.__getitem__(i)
        #     obj_points = obj_points.permute(0, 2, 1)
        #     rel_points = rel_points.permute(0, 2, 1)
        logs, rel_pred = network.process(obj_feat, rel_feat, edges_index, obj_gt, rel_gt)
        # logs += network.calculate_metrics([obj_pred, rel_pred], [obj_gt, rel_gt])
        # pred_cls = torch.max(obj_pred.detach(),1)[1]
        # acc_obj = (obj_gt == pred_cls).sum().item() / obj_gt.nelement()

        # rel_pred = rel_pred.detach() > 0.5
        # acc_rel = (rel_gt==(rel_pred>0)).sum().item() / rel_pred.nelement()
        #
        # print('{0:>3d} acc_obj: {1:>1.4f} acc_rel: {2:>1.4f} loss: {3:>2.3f}'.format(i,acc_obj,acc_rel,logs[0][1]))
        print('{:>3d} '.format(i), end='')
        for log in logs:
            print('{0:} {1:>2.3f} '.format(log[0], log[1]), end='')
        print('')