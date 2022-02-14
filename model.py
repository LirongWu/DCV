import scipy
import numpy as np
import torch
from torch import nn
from torch.nn import Parameter
from multiprocessing import Pool

class LISV2_MLP(nn.Module):
    def __init__(self, data, device, args):
        super(LISV2_MLP, self).__init__()

        with torch.no_grad():
            self.device = device
            self.args = args
            self.cluster_centers = Parameter(torch.Tensor(args['n_cluster'], args['clu_dim']))
  
            self.n_points = data.shape[0]
            self.perplexity = args['perplexity']
            self.args['NetworkStructure'][0] = data.shape[1]
            self.NetworkStructure = args['NetworkStructure']
            self.data = data.float().to(self.device)

            self.epoch = 0
            self.vList = [100] + [1] * (len(args['NetworkStructure']) - 1)
            self.gammaList = self.CalGammaF(self.vList)

            print('start claculate sigma')
            dist = self.Distance_squared(data, data).float()
            rho, self.sigmaListlayer = self.InitSigmaSearchEverySample(self.gammaList, self.vList, data, dist)
            self.P = self.CalPt(dist, rho, self.sigmaListlayer[0].cpu(), gamma=self.gammaList[0], v=self.vList[0]).float().detach().to(self.device)

            # np.save("../InitDis/{}_InputDis_perplexity{}.npy".format(args['data_name'], args['perplexity']), self.P.detach().cpu().numpy())
            # self.P = torch.tensor(np.load("../InitDis/{}_InputDis_perplexity{}.npy".format(args['data_name'], args['perplexity']))).to(self.device)

            s_out = np.log10(self.args['vtrace_out'][0])
            e_out = np.log10(self.args['vtrace_out'][1])
            self.vListForEpoch_out = np.concatenate([np.zeros((self.args['epochs']//2+1, )) + 10**s_out, np.logspace(s_out, e_out, self.args['epochs']//2)])

            s_in = np.log10(self.args['vtrace_in'][0])
            e_in = np.log10(self.args['vtrace_in'][1])
            self.vListForEpoch_in = np.concatenate([np.zeros((self.args['epochs']//2+1, )) + 10**s_in, np.logspace(s_in, e_in, self.args['epochs']//2)])

            print('start Init network')
            self.InitNetwork()
            
            torch.cuda.empty_cache()

    def InitNetwork(self, ):
        self.encoder = nn.ModuleList()
        for i in range(len(self.NetworkStructure) - 1):
            self.encoder.append(
                nn.Linear(self.NetworkStructure[i],
                          self.NetworkStructure[i + 1]))
            if i != len(self.NetworkStructure) - 2:
                self.encoder.append(nn.LeakyReLU(0.1))

        self.decoder = nn.ModuleList()
        for i in range(len(self.NetworkStructure) - 1, 0, -1):
            self.decoder.append(
                nn.Linear(self.NetworkStructure[i],
                          self.NetworkStructure[i - 1]))
            if i != 1:
                self.decoder.append(nn.LeakyReLU(0.1))

    def CalGammaF(self, vList):
        
        out = []
        for v in vList:
            a = scipy.special.gamma((v + 1) / 2)
            b = np.sqrt(v * np.pi) * scipy.special.gamma(v / 2)
            out.append(a / b)

        return out

    def Distance_squared(self, x, y):

        m, n = x.size(0), y.size(0)
        xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
        yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
        dist = xx + yy
        dist.addmm_(1, -2, x, y.t())
        d = dist.clamp(min=1e-12)
        d[torch.eye(d.shape[0]) == 1] = 1e-12
        
        return d

    def InitSigmaSearchEverySample(self, gammaList, vList, data, dist):

        distC = torch.clone(dist)
        distC[distC.le(1e-11)] = 1e16
        rho, _ = torch.min(distC, dim=1)

        sigmaListlayer = [0] * len(self.args['NetworkStructure'])
        
        r = PoolRunner(self.n_points, self.perplexity, dist.detach().cpu().numpy(), rho.detach().cpu().numpy(), gammaList[0], vList[0])
        sigmaListlayer[0] = torch.tensor(r.Getout()).to(self.device)

        std_dis = torch.std(rho) / np.sqrt(data.shape[1])
        print('std', std_dis)

        if std_dis > 0.2:
            for i in range(1, len(self.args['NetworkStructure'])):
                sigmaListlayer[i] = torch.zeros(data.shape[0], device=self.device) + 1
        else:
            for i in range(0, len(self.args['NetworkStructure'])):
                sigmaListlayer[i] = torch.zeros(data.shape[0], device=self.device) + sigmaListlayer[0].mean() * 5
                
        return rho, sigmaListlayer

    def CalPt(self, dist, rho, sigma_array, gamma, v=100):

        if torch.is_tensor(rho):
            dist_rho = (dist - rho.reshape(-1, 1)) / sigma_array.reshape(-1, 1)
        else:
            dist_rho = dist
        dist_rho[dist_rho < 0] = 0

        Pij = torch.pow(gamma * torch.pow((1 + dist_rho / v), -1 * (v + 1) / 2) * torch.sqrt(torch.tensor(2 * 3.14)), 2)
        P = Pij + Pij.t() - torch.mul(Pij, Pij.t())

        return P

    def CE(self, P, Q):

        EPS = 1e-12
        losssum1 = (P * torch.log(Q + EPS)).mean()
        losssum2 = ((1-P) * torch.log(1-Q + EPS)).mean()
        losssum = -1 * (losssum1 + losssum2)
        
        if torch.isnan(losssum):
            input('stop and find nan')
        return losssum

    def Loss(self, latentList, input_data_index):

        Q0 = self.CalPt(dist=self.Distance_squared(latentList[0], latentList[0]), rho=0, sigma_array=1, gamma=self.gammaList[-1], v=self.vList[-1])
        Q1 = self.CalPt(dist=self.Distance_squared(latentList[1], latentList[1]), rho=0, sigma_array=1, gamma=self.gammaList[-2], v=self.vList[-2])
        loss_ce = self.CE(P=self.P[input_data_index][:, input_data_index],Q=Q0) * self.args['ratio'][0]
        if self.args['ratio'][1] > 0:
            loss_ce += self.CE(P=self.P[input_data_index][:, input_data_index],Q=Q1) * self.args['ratio'][1]
        if self.args['ratio'][2] > 0:
            loss_ce += self.CE(P=Q1,Q=Q0) * self.args['ratio'][2]

        ReconstructionLoss = nn.MSELoss()
        loss_rc = ReconstructionLoss(self.reconstruct(self.data[input_data_index])[0], self.data[input_data_index])

        return [loss_ce, loss_rc * self.args['ratio'][3]]

    def ChangeVList(self):

        epoch = self.epoch
        self.vCurent = self.vListForEpoch_out[epoch]
        newVList = [100]
        for i in range(len(self.args['NetworkStructure']) - 1):
            newVList.append(self.vCurent)
        self.vList = newVList
        self.vList[-2] = self.vListForEpoch_in[epoch]
        self.gammaList = self.CalGammaF(newVList)

    def forward(self, input_data_index):

        self.ChangeVList()
        x = self.data[input_data_index]

        for _, layer in enumerate(self.encoder):
            x = layer(x)
            if x.shape[1] == self.args['clu_dim']:
                y = x

        return [x, y]

    def test(self, input_data):

        x = input_data.to(self.device)
        
        for _, layer in enumerate(self.encoder):
            x = layer(x)
            if x.shape[1] == self.args['clu_dim']:
                y = x

        return [x, y]

    def reconstruct(self, input_data):

        x_noise = torch.randn(input_data.size()).to(self.device) * 0.2
        x = input_data + x_noise

        for i, layer in enumerate(self.encoder):
            x = layer(x)
            if x.shape[1] == self.NetworkStructure[self.args['rec_index']]:
                break

        for i, layer in enumerate(self.decoder):
            if i > 2 * (len(self.NetworkStructure) - 1 - self.args['rec_index']) - 1:
                x = layer(x)

        return [x]

    def cluster(self, x):

        x = x.to(self.device)
        for _, layer in enumerate(self.encoder):
            x = layer(x)
            if x.shape[1] == self.args['clu_dim']:
                hidden = x

        p_squared = torch.sum((hidden.unsqueeze(1) - self.cluster_centers)**2, 2)
        p = 1.0 / (1.0 + (p_squared / self.args['sigma']))
        power = float(self.args['sigma'] + 1) / 2
        p = p ** power
        p_dist = (p.t() / torch.sum(p, 1)).t()

        return p_dist

    def target_distribution(self, q):
        weight = (q ** 2.0) / torch.sum(q, 0)
        return (weight.t() / torch.sum(weight, 1)).t()


class PoolRunner(object):
    def __init__(self, n, N_NEIGHBOR, dist, rho, gamma, v):
        pool = Pool(processes=16)

        result = []
        for dist_row in range(n):
            result.append(pool.apply_async(sigma_binary_search, (N_NEIGHBOR, dist[dist_row], rho[dist_row], gamma, v)))

        pool.close()
        pool.join()

        sigma_array = []
        for i in result:
            sigma_array.append(i.get())
        self.sigma_array = np.array(sigma_array)

        print("Mean sigma = " + str(np.mean(sigma_array)))
        print('finish calculate sigma')

    def Getout(self):
        return self.sigma_array


def sigma_binary_search(fixed_k, dist_row_line, rho_line, gamma, v):

    sigma_lower_limit = 0
    sigma_upper_limit = 100

    for i in range(20):
        approx_sigma = (sigma_lower_limit + sigma_upper_limit) / 2
        k_value = func(approx_sigma, dist_row_line, rho_line, gamma, v)

        if k_value < fixed_k:
            sigma_lower_limit = approx_sigma
        else:
            sigma_upper_limit = approx_sigma
        if np.abs(fixed_k - k_value) <= 1e-4:
            break

    return approx_sigma


def func(sigma, dist_row_line, rho_line, gamma, v):

    d = (dist_row_line - rho_line) / sigma
    d[d < 0] = 0

    p = np.power(gamma * np.power((1 + d / v), -1 * (v + 1) / 2) * np.sqrt(2 * 3.14), 2)
    return np.power(2, np.sum(p))
