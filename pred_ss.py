#!/usr/bin/env python
import os
import numpy as np
import torch
import argparse


def read_data(data_dir):
    X_pro_file = data_dir + "/Xdata_pro.npy"
    X_res_file = data_dir + "/Xdata_res.npy"
    Xdata_pro = np.load(X_pro_file)
    Xdata_res = np.load(X_res_file)
    Xdata_pro = Xdata_pro.reshape([1, 7])
    len_res = Xdata_res.shape[0]
    input_dim = Xdata_res.shape[1]
    Xdata_res = Xdata_res.reshape([1, len_res, input_dim])
    Xdata_pro = torch.from_numpy(Xdata_pro)
    Xdata_res = torch.from_numpy(Xdata_res)
    return (Xdata_pro, Xdata_res)


def main():

    parser = argparse.ArgumentParser(description='pred_ss')
    parser.add_argument('-s', '--save_dir', type=str, required=True,
                        help='weight parameter dir')

    parser.add_argument('-d', '--data_dir', type=str, required=False,
                        default='.', help='input data dir')
    parser.add_argument('-o', '--output_dir', type=str, required=False,
                        default='.', help='output dir')

    args = parser.parse_args()

    save_dir = args.save_dir
    data_dir = args.data_dir
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    path = save_dir + '/save_ss.pth'

    num_cpu = 1
#    use_cuda = torch.cuda.is_available()
    use_cuda = False
    if use_cuda:
        device_num = torch.cuda.current_device()
#        print(device_num)
        device = torch.device("cuda:%d" % device_num)
        torch.set_num_threads(num_cpu)
    else:
        device = torch.device("cpu")
        torch.set_num_threads(num_cpu)

#    print(device)

    model = torch.load(path)
    model.to(device)

    model.eval()

    batch_xpro, batch_xres = read_data(data_dir)
    batch_xpro = batch_xpro.to(device)
    batch_xres = batch_xres.to(device)
    num_res = batch_xres.shape[1]
    num_fea_pro = batch_xpro.shape[1]
    Xpro = batch_xpro.reshape(1, 1, num_fea_pro).repeat(1, num_res, 1)
    batch_x = torch.cat([batch_xres, Xpro], dim=2)
    out_ss8 = model(batch_x)
    Ypred_ss8 = torch.softmax(out_ss8, dim=2).cpu().detach().numpy()[0]

    Ypred_ss8_file = output_dir + '/Ypred_ss8.npy'
    np.save(Ypred_ss8_file, Ypred_ss8)


if __name__ == "__main__":
    main()
