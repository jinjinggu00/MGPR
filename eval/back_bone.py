from final_future_prediction_with_memory_spatial_sumonly_weight_ranking_top1 import convAE
import torch
import torch.nn as nn
from model.final_future_prediction_with_memory_spatial_sumonly_weight_ranking_top1 import flow_convAE
#from M1_unet.final_future_prediction_with_memory_spatial_sumonly_weight_ranking_top1 import Unet
from M1_unet.Unet import Unet


device = 'cuda'

class Backbone(nn.Module):
    def __init__(self):
        super(Backbone, self).__init__()
        self.model_frame = convAE().to(device).eval()
        self.model_flow = flow_convAE().to(device).eval()
        self.recon_unet = Unet(3, 5, 512, 512).to(device).eval()

        self.model_frame.load_state_dict(torch.load('./check/freeway_check/freeway_model_60.pth'))
        self.model_flow.load_state_dict(torch.load('./check/freeway_check/freeway_flows_dic_model.pth'))
        self.recon_unet.load_state_dict(torch.load('./check/freeway_unet_120_4_9.pth'))

        frames_m_items = torch.load('./check/freeway_check/freeway_keys_60.pt')
        self.frames_m_items_test = frames_m_items.clone()

        flows_m_items = torch.load('./check/freeway_check/freeway_flows_keys.pt')
        self.flows_m_items_test = flows_m_items.clone()

        #recon_m_items = torch.load('./check/freeway_check/freeway_unet_M1_keys_60.pt')
        #self.recon_m_item_test = recon_m_items.clone()
        print('load model weights done ! ! !')
                                                                                                     # forward(self, x, keys, flow_fea, update_fea_flow, train=True):

    def forward(self, image, flow, train=True):
        flow_output, feas, updated_fea, keys, softmax_score_query, softmax_score_memory, query, top1_keys, keys_ind, compactness_loss, skip3  = self.model_flow(
                                                                                                                                            flow, self.flows_m_items_test, train)


        output, att_weight, fea, updated_fea_, keys, softmax_score_query, softmax_score_memory, query, top1_keys, keys_ind, compactness_loss = self.model_frame(
                                                                                                                                            image, self.frames_m_items_test, feas, skip3,  updated_fea,
                                                                                                                                            train)

        recom_out, _ =  self.recon_unet(output, train)

        return output, recom_out, att_weight, fea, updated_fea_, keys, softmax_score_query, softmax_score_memory, query, top1_keys, keys_ind, compactness_loss, flow_output








