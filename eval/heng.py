import torch
from final_future_prediction_with_memory_spatial_sumonly_weight_ranking_top1 import convAE
import torch.nn.functional as F


model = convAE(n_channel =3,  t_length = 5, memory_size = 10, feature_dim = 512, key_dim = 512, temp_update = 0.1, temp_gather=0.1)


device = 'cuda'

m_items = F.normalize(torch.rand((10, 512), dtype=torch.float), dim=1).to(device)

model = model.to(device)

x = torch.rand(1, 12, 240, 360).to(device)
outputs, feas, m_items_test,updated_feas,  softmax_score_query, softmax_score_memory, _, _, _, compactness_loss, skip2 = model.forward(x , m_items, False)

print('asafaf',updated_feas.shape)

