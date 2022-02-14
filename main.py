import argparse
import warnings
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score

import torch
import torch.optim as optim
import torch.nn.functional as F

import tool
import dataloader
import model as model_DCV


warnings.filterwarnings("ignore", category=Warning)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def Train(args, Model, data, target, optimizer, epoch):

    Model.train()
    BATCH_SIZE = args['batch_size']

    num_train_sample = data.shape[0]
    num_batch = (num_train_sample - 0.5) // BATCH_SIZE + 1
    rand_index_i = torch.randperm(num_train_sample, device=Model.device).long()
    train_loss_sum = [0, 0, 0]

    q = Model.cluster(data)
    p = Model.target_distribution(q).detach()

    for batch_idx in torch.arange(num_batch):
        start = (batch_idx * BATCH_SIZE).int().to(Model.device)
        end = torch.min(torch.tensor([batch_idx * BATCH_SIZE + BATCH_SIZE, num_train_sample])).to(Model.device)
        sample_index_i = rand_index_i[start:end.int()]

        optimizer.zero_grad()
        output = Model(sample_index_i)
        loss_list = Model.Loss(output, sample_index_i)

        if ((epoch > args['pre_epochs'] and args['pretrain'] == 0) or args['pretrain'] == 2) and args['ratio'][4]:
            q = Model.cluster(data[sample_index_i])
            loss_cluster = F.kl_div(q.log(), p[sample_index_i]) * args['ratio'][4]
            loss_cluster.backward()
            train_loss_sum[2] += loss_cluster.item()

        loss_list[0].backward(retain_graph=True)
        loss_list[1].backward()
        train_loss_sum[0] += loss_list[0].item()
        train_loss_sum[1] += loss_list[1].item()
        optimizer.step()

    if epoch % 10 == 0:
        print('Train Epoch: {} [{}/{} ({:.0f}%)] \t Loss: {}'.format(
            epoch, batch_idx * BATCH_SIZE, num_train_sample,
            100. * batch_idx * BATCH_SIZE / num_train_sample, train_loss_sum))


def Test(args, Model, data, target, optimizer, epoch, torch_format=False):

    Model.eval()
    BATCH_SIZE = args['batch_size']

    num_train_sample = data.shape[0]
    num_batch = (num_train_sample - 0.5) // BATCH_SIZE + 1
    rand_index_i = torch.arange(num_train_sample)

    for batch_idx in torch.arange(num_batch):
        start = (batch_idx * BATCH_SIZE).int()
        end = torch.min(torch.tensor([batch_idx * BATCH_SIZE + BATCH_SIZE, num_train_sample]))
        sample_index_i = rand_index_i[start:end.int()]

        datab = data.float()[sample_index_i]
        em = Model.test(datab)

        em1 = em[-2]
        em2 = em[-1]

        if batch_idx == 0:
            outem1 = em1
            outem2 = em2
        else:
            outem1 = torch.cat((outem1, em1), 0)
            outem2 = torch.cat((outem2, em2), 0)

    if torch_format is False:
        outem1 = outem1.detach().cpu().numpy()
        outem2 = outem2.detach().cpu().numpy()


    return [outem1, outem2]


def main(args):

    path = tool.GetPath(args['name'])
    tool.SaveParam(path, args)
    tool.SetSeed(args['seed'])

    data_train, label_train = dataloader.GetData(args)

    gifPloterLatentTrain = tool.GIFPloter()
    DataSaver = tool.DataSaver()

    Model = model_DCV.LISV2_MLP(data_train, device=device, args=args).to(device)
    if args['pretrain'] == 2:
        Model.load_state_dict(torch.load("../model/model_{}.pkl".format(args['data_name'])))
    optimizer = optim.Adam(Model.parameters(), lr=args['lr'])

    for epoch in range(args['epochs'] + 1):

        Model.epoch = epoch

        if epoch > 0:
            Train(args, Model, data_train, label_train, optimizer, epoch)
        
            if (epoch > args['pre_epochs'] and args['pretrain'] == 0) or args['pretrain'] == 2:
                em_train = Test(args, Model, data_train, label_train, optimizer, epoch, torch_format=True)
                pred = Model.cluster(data_train).argmax(1).detach().cpu().numpy()
                for i in range(args['n_cluster']):
                    if em_train[1][pred == i].shape[0] > 0:
                        Model.cluster_centers.data[i] = args['alpha'] * Model.cluster_centers.data[i] + (1-args['alpha']) * torch.mean(em_train[1][pred == i], 0)


        if epoch == args['pre_epochs'] and (args['pretrain'] == 1 or args['pretrain'] == 0):
            em_train = Test(args, Model, data_train, label_train, optimizer, epoch)
            kmeans = KMeans(n_clusters=args['n_cluster'], random_state=0, n_init=20).fit(em_train[0])
            y_pred = kmeans.predict(em_train[0])

            cluster_centers = np.zeros((args['n_cluster'], args['clu_dim']))
            for i in range(args['n_cluster']):
                cluster_centers[i] = np.mean(em_train[1][y_pred == i], 0)
            Model.cluster_centers.data = torch.tensor(cluster_centers).to(device)

            gifPloterLatentTrain.AddNewFig(em_train[0], y_pred, path=path, name='train_epoch{}_cluster_{}.png'.format(epoch, tool.cluster_acc(label_train.detach().cpu().numpy(), y_pred)), args=args)
            if args['pretrain'] == 1:
                torch.save(Model.state_dict(), "../model/model_{}.pkl".format(args['data_name']))


        if epoch % args['log_interval'] == 0:

            em_train = Test(args, Model, data_train, label_train, optimizer, epoch)
            y_pred = Model.cluster(data_train).argmax(1).detach().cpu().numpy()

            kmeans = KMeans(n_clusters=args['n_cluster'], random_state=0, n_init=20).fit(em_train[0])
            y_pred_vis = kmeans.predict(em_train[0])
            kmeans = KMeans(n_clusters=args['n_cluster'], random_state=0, n_init=20).fit(em_train[1])
            y_pred_clu = kmeans.predict(em_train[1])

            vis_acc = tool.cluster_acc(label_train.detach().cpu().numpy(), y_pred_vis)
            clu_acc = tool.cluster_acc(label_train.detach().cpu().numpy(), y_pred_clu)
            dec_acc = tool.cluster_acc(label_train.detach().cpu().numpy(), y_pred)
            dec_nmi = nmi_score(label_train.detach().cpu().numpy(), y_pred)
            print('Train Epoch {} : {}, {}, {}, {}'.format(epoch, vis_acc, clu_acc, dec_acc, dec_nmi))

            gifPloterLatentTrain.AddNewFig(em_train[0], label_train.detach().cpu().numpy(), path=path, name='train_epoch{}_em2.png'.format(epoch), args=args)
            gifPloterLatentTrain.AddNewFig(em_train[0], y_pred_vis, path=path, name='train_epoch{}_em2_clu.png'.format(epoch), args=args)
            if (args['pretrain'] == 1 and epoch >= args['pre_epochs']) or (args['pretrain'] == 2 and epoch % 100 == 0 and epoch > 0):
                gifPloterLatentTrain.AddNewFig(em_train[1], label_train.detach().cpu().numpy(), path=path, name='train_epoch{}_em{}.png'.format(epoch, args['clu_dim']), args=args, cluster = Model.cluster_centers.detach().cpu().numpy())
                gifPloterLatentTrain.AddNewFig(em_train[1], y_pred, path=path, name='train_epoch{}_em{}_clu.png'.format(epoch, args['clu_dim']), args=args, cluster = Model.cluster_centers.detach().cpu().numpy())
            DataSaver.SaveData(data_train, em_train, label_train, epoch, path=path, name='train_epoch{}_'.format(str(epoch).zfill(5)))
            
            with open(path+"results.txt","a+") as files:
                files.write('Train Epoch {} : {}, {}, {}, {}\n'.format(epoch, vis_acc, clu_acc, dec_acc, dec_nmi))
                files.flush() 

            with open("../log/results.txt","a+") as files:
                files.write('{} {} {} | {} {} {} {} | Train Epoch {} : {}, {}, {}, {}\n'.format(args['perplexity'], args['batch_size'], args['ratio'][3], args['alpha'], args['sigma'], args['ratio'][4], args['vtrace_out'][1], epoch, vis_acc, clu_acc, dec_acc, dec_nmi))
                files.flush() 


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='author: Lirong Wu')
    
    # data set param
    parser.add_argument('--name', type=str, default='USPS')
    parser.add_argument('--data_name', type=str, default='usps')
    parser.add_argument('--n_cluster', type=int, default=10)
    parser.add_argument('--pretrain', type=int, default=0)

    # model param
    parser.add_argument('--perplexity', type=int, default=3)
    parser.add_argument('--vtrace_in', type=float, nargs='+', default=[0.001, 0.001])
    parser.add_argument('--vtrace_out', type=float, nargs='+', default=[0.001, 0.001])
    parser.add_argument('--NetworkStructure', type=int, nargs='+', default=[-1, 1000, 500, 300, 100, 50, 30, 2])
    parser.add_argument('--clu_dim', type=int, default=100)
    parser.add_argument('--rec_index', type=int, default=4)
    parser.add_argument('--ratio', type=float, nargs='+', default=[1.0, 0.0, 0.0, 0.01, 0.01])
    parser.add_argument('--alpha', type=float, default=0.99)
    parser.add_argument('--sigma', type=float, default=1.0)

    # train param
    parser.add_argument('--batch_size', type=int, default=500)
    parser.add_argument('--pre_epochs', type=int, default=500)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--log_interval', type=int, default=100)
    args = parser.parse_args().__dict__

    path = main(args)

