# from functools import partial

import sys
import torch
import torch.nn as nn

from torch.utils.data import Dataset
# from tqdm import tqdm
import numpy as np
import ptwt 

class MyDataSet(Dataset):
    def __init__(self, mat_set: list, mat_property: list):
        self.mat_set = mat_set
        self.mat_property = mat_property

    def __len__(self):
        return len(self.mat_set)

    def __getitem__(self, item):
        com = self.mat_set[item]
        label = self.mat_property[item]
        return com, label

    @staticmethod
    def collate_fn(batch):
        mat_com, mat_label = tuple(zip(*batch))
        mat_label = np.array(mat_label)
        mat_com_tensor = [torch.as_tensor(i) for i in mat_com]
        mat_com = torch.nn.utils.rnn.pad_sequence(mat_com_tensor, batch_first=True).type(torch.float32)
        mat_label = torch.as_tensor(mat_label).type(torch.float32)
        return mat_com, mat_label

class Module_1(nn.Module):
    def __init__(self,
                 embed_dim,
                 norm_layer=nn.LayerNorm):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 64, bias=True),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 64, bias=True),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, embed_dim, bias=True)
        )
    
    def forward(self, x):
        x = x.type(torch.float32)
        x = self.mlp(x)
        return x
    
class Module_2(nn.Module):
    def __init__(self,
                 embed_dim):
        super().__init__()
        self.scale = embed_dim ** -0.5
        self.q_fc = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_fc = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_fc = nn.Linear(embed_dim, embed_dim, bias=True)
        self.fc = nn.Linear(embed_dim, embed_dim)
    def forward(self, x):
        x = x.type(torch.float32)
        q = self.q_fc(x)
        k = self.k_fc(x)
        v = self.v_fc(x)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v)
        x = self.fc(x)

        return x


class Module_3(nn.Module):
    def __init__(self,
                 embed_dim,):
        super().__init__()
        self.scale = embed_dim ** -0.5        
        self.q_fc = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_fc = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_fc = nn.Linear(embed_dim, embed_dim, bias=True)
        self.fc = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        x = x.type(torch.float32)
        q = self.q_fc(x)
        k = self.k_fc(x)
        v = self.v_fc(x)
        q = q.unsqueeze(-2)
        k = k.unsqueeze(-2)
        v = v.unsqueeze(-2)
        attn = (q.transpose(-1, -2)) @ k * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ (v.transpose(-1, -2)))
        x = x.squeeze(dim=-1)
        x = self.fc(x)
        return x

class emb_update(nn.Module):
    def __init__(self,
                 embed_dim,
                 nnlist_length,):
        super().__init__()
        self.embed_dim = embed_dim
        self.nnlist_length = nnlist_length
        self.layernorm = nn.LayerNorm(self.embed_dim)
        self.batchnorm = nn.LayerNorm(self.nnlist_length)  
        self.fc_lay = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.fc_bat = nn.Linear(self.nnlist_length, self.nnlist_length, bias=True)   
        self.rl = nn.ReLU()
        self.m1 = Module_1(embed_dim=embed_dim)
        self.m2 = Module_2(embed_dim=embed_dim)
        self.m3 = Module_3(embed_dim=embed_dim)
        self.fc_attn_emb = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.fc_attn_nnlist = nn.Linear(self.nnlist_length, self.nnlist_length, bias=True)    
        self.for_explanation = nn.Parameter(torch.ones(12)/12, requires_grad=True)

    def forward(self, x):
        x = x.type(torch.float32)
        x_layernorm = self.layernorm(x)
        x_layernorm = self.fc_lay(x_layernorm)
        x_layernorm = self.rl(x_layernorm)
        x_batchnorm = self.batchnorm(x.transpose(2,3))  
        x_batchnorm = self.fc_bat(x_batchnorm) 
        x_batchnorm = self.rl(x_batchnorm)
        x_batchnorm = x_batchnorm.transpose(2,3)
        x1 = self.m1(x)
        x2 = self.m1(x_layernorm)
        x3 = self.m1(x_batchnorm)
        x4 = self.m2(x)
        x5 = self.m2(x_layernorm)
        x6 = self.m2(x_batchnorm)
        x7 = self.m3(x)
        x8 = self.m3(x_layernorm)
        x9 = self.m3(x_batchnorm)
        x10 = x * self.fc_attn_emb(x_layernorm.softmax(dim=-1))
        x11_attn = self.fc_attn_nnlist(x_batchnorm.softmax(dim=-2).transpose(2,3))   
        x11 = x * x11_attn.transpose(2,3)
        scaling_factor = self.for_explanation.softmax(dim=-1)
        x = x  * scaling_factor[0] + \
            x1 * scaling_factor[1] + \
            x2 * scaling_factor[2] + \
            x3 * scaling_factor[3] + \
            x4 * scaling_factor[4] + \
            x5 * scaling_factor[5] + \
            x6 * scaling_factor[6] + \
            x7 * scaling_factor[7] + \
            x8 * scaling_factor[8] + \
            x9 * scaling_factor[9] + \
            x10 * scaling_factor[10] + \
            x11 * scaling_factor[11]
        
        return x

class nn_feature_cal(nn.Module):
    def __init__(self,
                 embed_dim,
                 power_exponent, 
                 nnlist_length,
                 feature_update_depth, 
                 ):
        super().__init__()
        self.power_exponent = power_exponent
        self.embed_dim = embed_dim
        self.nnlist_length = nnlist_length
        self.poly_fc = nn.Linear(self.power_exponent, 1, bias=True)

        self.feature_update_blocks = nn.Sequential(*[emb_update(embed_dim=embed_dim,
                                                                nnlist_length=nnlist_length) for i in range(feature_update_depth)])
    def forward(self, x):
        x = x.type(torch.float32)
        x_nnlist_index, x_tot_r, x_tot_emb = torch.split(x, [1, self.power_exponent, self.embed_dim], dim=3)
        x_center_emb, tmp = torch.split(x_tot_emb, [1, self.nnlist_length-1], dim=2)
        x_center_emb = x_center_emb.transpose(1,2)
        x_center_emb = x_center_emb.expand(-1, x_nnlist_index.shape[1], -1, -1)
        x_tot_emb = x_nnlist_index.to(torch.int64)
        x_tot_emb = x_tot_emb.expand(-1, -1, -1, self.embed_dim)
        x_tot_emb = torch.gather(x_center_emb, dim=2, index=x_tot_emb)
        x_tot_r_kan = x_tot_r.unsqueeze(3)
        x_tot_r_kan = x_tot_r_kan.expand(-1, -1, -1, self.embed_dim, -1)
        x_tot_emb = x_tot_emb.unsqueeze(dim=-1)
        x_tot_emb = x_tot_emb.repeat(1, 1, 1, 1, self.power_exponent)
        x_tot_emb = x_tot_emb * x_tot_r_kan
        x_tot_emb = self.poly_fc(x_tot_emb)
        x_tot_emb = x_tot_emb.squeeze(dim=-1)
        x_tot_emb = self.feature_update_blocks(x_tot_emb)
        x = torch.cat((x_nnlist_index, x_tot_r, x_tot_emb), dim=3)

        return x

class Wavelet_model(nn.Module):
    def __init__(self,
                embed_dim=20, 
                power_exponent=10, 
                poly_degree=10, 
                nnlist_length=100, 
                feature_update_depth=15, 
                wavelet_type='haar',
                wavelet_level=7, 
                ):
        super(Wavelet_model, self).__init__()

        self.embed_dim = embed_dim
        self.embeddings = nn.Embedding(119, self.embed_dim, padding_idx=0, max_norm=100)
        self.nnlist_length = nnlist_length 
        self.atom_ee_add = nn.Parameter(torch.ones(1, 1, self.embed_dim), requires_grad=True)
        self.atom_ee_multi = nn.Parameter(torch.ones(1, 1, self.embed_dim), requires_grad=True)
        self.power_exponent = power_exponent 
        self.blocks = nn.Sequential(*[nn_feature_cal(embed_dim=self.embed_dim, 
                                                    power_exponent=self.power_exponent,
                                                    nnlist_length=nnlist_length+1,
                                                    feature_update_depth=feature_update_depth) for i in range(poly_degree)])
        self.mlp_emb = nn.Sequential(
            nn.Linear(self.embed_dim, 32, bias=True),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1, bias=True)
        )
        self.tot_eng_add = nn.Parameter(torch.ones(1, 1, self.embed_dim), requires_grad=True)
        self.tot_eng_multi = nn.Parameter(torch.ones(1, 1, self.embed_dim), requires_grad=True)
        self.mlp1 = nn.Sequential(
            nn.Linear(self.embed_dim+9, 64, bias=True),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 128, bias=True),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64, bias=True),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1, bias=True)
        )
        self.rr_fc = nn.Linear(power_exponent, power_exponent, bias=True)
        self.mlp_emb = nn.Sequential(
            nn.Linear(self.embed_dim, 32, bias=True),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1, bias=True)
        )
        self.mlp_wavelet = nn.Sequential(
            nn.Linear(self.nnlist_length+8, 32, bias=True),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1, bias=True)
        )
        self.wavelet_level = wavelet_level
        self.apply(_init_vit_weights)

    def forward_features(self, x):
        x = x.type(torch.float32)
        x_lattice_abc_angle, x_nn_tot = torch.split(x, [1, x.shape[1]-1], dim=1)
        x_nn_tot, tmp = torch.split(x_nn_tot, [self.nnlist_length+1, x_nn_tot.shape[2]-self.nnlist_length-1], dim=2)
        x_center_atom_index_type, x_nnlist_index_r = torch.split(x_nn_tot, [1, x_nn_tot.shape[2]-1], dim=2)
        x_center_atom_index, x_center_atom_type = torch.split(x_center_atom_index_type, [1, 1], dim=3)
        x_center_atom_type = x_center_atom_type.squeeze(dim=3).squeeze(dim=2)
        x_center_atom_emb_1 = self.embeddings(torch.as_tensor(x_center_atom_type, dtype=torch.int32))
        x_nnlist_index, x_nnlist_r = torch.split(x_nnlist_index_r, [1, 1], dim=3)
        x_tot_index = torch.cat((x_center_atom_index, x_nnlist_index), dim=2)
        x_center_atom_index_zeros = torch.zeros_like(x_center_atom_index)
        x_tot_r = torch.cat((x_center_atom_index_zeros, x_nnlist_r), dim=2)
        x_tot_r = x_tot_r.repeat(1,1,1,self.power_exponent)
        x_tot_r = self.rr_fc(x_tot_r)
        x_tot_r = torch.cos(x_tot_r)
        x_center_atom_emb_1 = x_center_atom_emb_1.unsqueeze(dim=2)
        tmp_zeros_nnlist = torch.zeros_like(x_nnlist_r)
        tmp_zeros_nnlist = tmp_zeros_nnlist.repeat(1,1,1,x_center_atom_emb_1.shape[3])
        x_tot_init_emb = torch.cat((x_center_atom_emb_1, tmp_zeros_nnlist), dim=2)
        x_tot_init = torch.cat((x_tot_index, x_tot_r, x_tot_init_emb), dim=3)
        x_set_0_1 = torch.where(x_tot_index!=0, 1, 0)
        x_set_0_1 = x_set_0_1.expand(-1, -1, -1, x_tot_init.shape[3])
        x_tot_init = x_tot_init * x_set_0_1
        x_tot_enr = self.blocks(x_tot_init)
        x_tot_enr = x_tot_enr * x_set_0_1
        x_nnlist_index, x_tot_r, x_tot_emb = torch.split(x_tot_enr, [1, self.power_exponent, self.embed_dim], dim=3)
        x_tot_emb = self.mlp_emb(x_tot_emb).squeeze(dim=3)
        x_center_atom_index = x_center_atom_index.squeeze(dim=3)
        x_set_0_1 = torch.where(x_center_atom_index!=0, 1, 0)
        x_set_0_1 = x_set_0_1.expand(-1, -1, x_tot_emb.shape[2])
        x_tot_emb = x_tot_emb * x_set_0_1
        x_tot_eng_wavelet = ptwt.wavedec(x_tot_emb, 'haar', mode='zero', level=self.wavelet_level)
        x_tot_eng_wavelet_0 = x_tot_eng_wavelet[0]
        x_tot_eng_wavelet_1 = x_tot_eng_wavelet[1]
        x_tot_eng_wavelet_2 = x_tot_eng_wavelet[2]
        x_tot_eng_wavelet_3 = x_tot_eng_wavelet[3]
        x_tot_eng_wavelet_4 = x_tot_eng_wavelet[4]
        x_tot_eng_wavelet_5 = x_tot_eng_wavelet[5]
        x_tot_eng_wavelet_6 = x_tot_eng_wavelet[6]
        x_tot_eng_wavelet_7 = x_tot_eng_wavelet[7]
        x_tot_eng_wavelet = torch.cat((x_tot_eng_wavelet_0, x_tot_eng_wavelet_1,
                                       x_tot_eng_wavelet_2, x_tot_eng_wavelet_3,
                                       x_tot_eng_wavelet_4, x_tot_eng_wavelet_5,
                                       x_tot_eng_wavelet_6, x_tot_eng_wavelet_7), dim=2)
        x_set_0_1 = torch.where(x_center_atom_index!=0, 1, 0)
        x_set_0_1 = x_set_0_1.expand(-1, -1, x_tot_eng_wavelet.shape[2])
        x_tot_eng_wavelet = x_tot_eng_wavelet * x_set_0_1
        x_tot_eng_wavelet = self.mlp_wavelet(x_tot_eng_wavelet).squeeze(dim=2)
        x_set_0_1 = torch.where(x_center_atom_index!=0, 1, 0)
        x_set_0_1 = x_set_0_1.squeeze(dim=2)
        x_tot_eng_wavelet = x_tot_eng_wavelet * x_set_0_1
        x_tot_eng_wavelet = torch.sum(x_tot_eng_wavelet, dim=1)
        return x_tot_eng_wavelet

    def forward(self, x):
        x = x.type(torch.float32)
        x = self.forward_features(x)
        return x

def _init_vit_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)


def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    loss_function = torch.nn.MSELoss()
    accu_loss = torch.zeros(1).to(device) 
    optimizer.zero_grad()
    res_tot = []
    sample_num = 0
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]
        pred = model(images.to(device))
        pred = torch.reshape(pred, (-1,))
        labels = torch.reshape(labels, (-1,))
        loss = loss_function(pred, labels.to(device))
        loss.backward()
        res_tot.append(loss.item())
        if len(res_tot) > 101:
            del res_tot[0]
        accu_loss = np.mean(res_tot)
        data_loader.desc = "[train epoch {}] tmp_loss: {:.3f}, tot_loss: {:.3f}\n".format(epoch,
                                                                               loss.item(),
                                                                               np.mean(res_tot))
        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)
        optimizer.step()
        optimizer.zero_grad()
    return accu_loss.item() / sample_num


@torch.no_grad()
def evaluate(model, data_loader, device, epoch=0):

    model.eval()
    nlp_pre = []
    nlp_ture = []
    sample_num = 0
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]
        pred = model(images.to(device))
        pred = torch.reshape(pred, (-1,))
        pred_array = pred.cpu().detach().tolist()
        labels_array = labels.cpu().detach().tolist()
        for i in range(len(pred_array)):
            nlp_pre.append(pred_array[i])
        for i in range(len(labels_array)):
            nlp_ture.append(labels_array[i])

    return nlp_pre, nlp_ture



