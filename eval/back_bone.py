from final_future_prediction_with_memory_spatial_sumonly_weight_ranking_top1 import convAE
import torch
import torch.nn as nn
from model.final_future_prediction_with_memory_spatial_sumonly_weight_ranking_top1 import flow_convAE

device = 'cuda'

class Backbone(nn.Module):
    def __init__(self):
        super(Backbone, self).__init__()
        self.model_frame = convAE().to(device)
        self.model_flow = flow_convAE().to(device)
        self.model_frame.load_state_dict(torch.load('./check/ped2_remove_skip12_.pth'))
        self.model_flow.load_state_dict(torch.load('./check/flows_dic_model.pth'))

        frames_m_items = torch.load('./check/ped2_remove_skip12_keys_.pt')
        self.frames_m_items_test = frames_m_items.clone()

        flows_m_items = torch.load('./check/flows_keys.pt')
        self.flows_m_items_test = flows_m_items.clone()
        print('load model weights done ! ! !')
                                                                                                     # forward(self, x, keys, flow_fea, update_fea_flow, train=True):

    def forward(self, image, flow, train=True):
        flow_output, feas, updated_fea, keys, softmax_score_query, softmax_score_memory, query, top1_keys, keys_ind, compactness_loss, skip3  = self.model_flow(
                                                                                                                                            flow, self.flows_m_items_test, train)

        output, att_weight, fea, updated_fea_, keys, softmax_score_query, softmax_score_memory, query, top1_keys, keys_ind, compactness_loss = self.model_frame(
                                                                                                                                            image, self.frames_m_items_test, feas, skip3,  updated_fea,
                                                                                                                                 train)

        return output,att_weight, fea, updated_fea_, keys, softmax_score_query, softmax_score_memory, query, top1_keys, keys_ind, compactness_loss, flow_output








