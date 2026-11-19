from copy import deepcopy
import torch
from torch import nn
import torch.nn.functional as F
class BaseLinearModel(nn.Module):
    def __init__(self, configs):
        ## ignore lin_prob since this is linear model
        super(BaseLinearModel, self).__init__()
        self.IN = configs.instance_norm
        self.indv = configs.individual
        if self.indv:
            self.l1 = nn.ModuleList(
                [nn.Linear(configs.seq_len, configs.pred_len, bias=False) for _ in range(configs.enc_in)]
            )
        else:
            self.l1 = nn.Linear(configs.seq_len, configs.pred_len, bias=False)

    def forward(self, x, *args):
        if self.IN:
            x_mean = x.mean(dim = 1, keepdim = True)
            x = x-x_mean
        x = x.permute(0,2,1) # x: [B, C, T]
        if self.indv:
            y = [l_c(x[:,[idx],:]) for idx, l_c in enumerate(self.l1)]
            y = torch.cat(y,dim=1).permute(0,2,1)
        else:
            y = self.l1(x).permute(0,2,1)
        
        if self.IN:
            y = y + x_mean
        return y

class SpecLinear(nn.Module):
    def __init__(self, configs):
        super(SpecLinear, self).__init__()
        self.IN = configs.instance_norm
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.individual = configs.individual
        self.channels = configs.enc_in

        self.length_ratio=(self.seq_len + self.pred_len)/self.seq_len

        if self.individual:
            self.freq_lin = nn.ModuleList()
            for i in range(self.channels):
                self.freq_lin.append(nn.Linear(self.seq_len//2+1, (self.seq_len + self.pred_len)//2+1, bias=False).to(torch.cfloat))
        else:
            self.freq_lin = nn.Linear(self.seq_len//2+1, (self.seq_len + self.pred_len)//2+1, bias=False).to(torch.cfloat) # complex layer for frequency upcampling]


    def forward(self, x, *args):
        if self.IN:
            x_mean = torch.mean(x, dim=1, keepdim=True)
            x = x - x_mean

        low_specx = torch.fft.rfft(x, dim=1)
        # print(low_specx.permute(0,2,1))
        if self.individual:
            low_specy = torch.zeros(
                [low_specx.size(0), (self.seq_len + self.pred_len)//2+1, low_specx.size(2)],
                dtype=low_specx.dtype
            ).to(low_specx.device)
            for i in range(self.channels):
                low_specy[:,:,i]=self.freq_lin[i](low_specx[:,:,i].permute(0,1)).permute(0,1)
        else:
            low_specy = self.freq_lin(low_specx.permute(0,2,1)).permute(0,2,1)
        y = torch.fft.irfft(low_specy, dim=1)
        y = y * self.length_ratio # energy compemsation for the length change
        
        if self.IN:
            y = y + x_mean
        return y[:,-self.pred_len:,:]

class RootPurgeModel(nn.Module):
    def __init__(self, configs, Model):
        super(RootPurgeModel, self).__init__()

        self.IN = configs.instance_norm
        self.lambda_ = configs.regu_coef
        self.order = configs.purge_order
    
        assert configs.residual_gt in ['zero', 'model_zero']
        self.residual_gt = configs.residual_gt
        self.padding_compensator = max(configs.seq_len/configs.pred_len,1.0)
        
        # set it to false so that base_model does not redo it
        configs_base_model = deepcopy(configs)
        configs_base_model.instance_norm = False
        self.base_model = Model(configs_base_model)


    @staticmethod
    def sample_residual(x):
        B, C, T, nS = x.shape
        indices = torch.randint(0, nS, (B, C), device=x.device)
        indices = indices.view(B, C, 1, 1).expand(-1, -1, T, -1)
        x_sampled = torch.gather(x, dim=3, index=indices)
        return x_sampled.squeeze(-1)

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        return self.base_model.load_state_dict(state_dict, strict, assign)

    def state_dict(self):
        return self.base_model.state_dict()

    def forward(self, *args):
        if self.training:
            return self.forward_train(*args)
        else:
            return self.forward_test(*args)

    def parameters(self, recurse: bool = True):
        return self.base_model.parameters(recurse)

    def forward_test(self, x, *args):
        if self.IN:
            x_mean = x.mean(dim = 1, keepdim = True)
            x = (x-x_mean)
        
        y = self.base_model(x)
        
        if self.IN:
            y = y + x_mean
        return y

    def forward_train(self, x, y, *args):
        if self.IN:
            x_mean = x.mean(dim = 1, keepdim = True)
            x = (x-x_mean)
            y = (y-x_mean)
        B, Tx, C = x.shape
        _, Ty, _ = y.shape
        # forward data
        yf_hat = self.base_model(x)
        residual_ = (y-yf_hat).detach() # B, Ty, C

        for i in range(self.order):
            if Tx == Ty:
                pass
            elif Tx > Ty:
                residual_ = F.pad(residual_.permute(0,2,1),  pad=(Tx-Ty, Tx-Ty)).unfold(-1, Tx, 1).permute(0,1,3,2) # B,C,nS,Tx -> B,C,Tx,nS
                residual_ = self.sample_residual(residual_).permute(0, 2, 1) # B, C, Tx -> B, Tx, C
            elif Tx < Ty:
                residual_ = residual_[:,:Tx,:]
            
            residual_ = self.base_model(residual_)

        val_y = residual_
        if self.residual_gt == "zero":
            val_gt = torch.zeros_like(val_y)
        elif self.residual_gt == "model_zero":
            val_gt = self.base_model(torch.zeros_like(x)).detach()
        floss = F.mse_loss(yf_hat, y)
        val_floss = F.mse_loss(val_y,val_gt)

        loss_ = floss+(val_floss * self.lambda_ * self.padding_compensator**self.order)
        return {"loss":loss_, "mse":floss, "checker":val_floss}
