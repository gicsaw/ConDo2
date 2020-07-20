#!/usr/bin/env python
import os
import numpy as np
import torch
import argparse


def read_data(data_dir, prepare_dir):
    X_pro_file = data_dir + "/Xdata_pro.npy"
    X_res_file = data_dir + "/Xdata_res.npy"
    X_pred_ss8_file = prepare_dir + "/Ypred_ss8.npy"
    X_pred_sa3_file = prepare_dir + "/Ypred_sa3.npy"
    X_pred_rsa_file = prepare_dir + "/Ypred_rsa.npy"

    Xdata_pro = np.load(X_pro_file)
    Xdata_res = np.load(X_res_file)
    Xdata_ss8 = np.load(X_pred_ss8_file)
    Xdata_sa3 = np.load(X_pred_sa3_file)
    Xdata_rsa = np.load(X_pred_rsa_file)

    Xdata_res = np.concatenate([Xdata_res, Xdata_ss8,
                                Xdata_sa3, Xdata_rsa], axis=1)
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
    prepare_dir = output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    path = save_dir + '/save_dom.pth'

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

    batch_xpro, batch_xres = read_data(data_dir, prepare_dir)
    batch_xpro = batch_xpro.to(device)
    batch_xres = batch_xres.to(device)
    num_res = batch_xres.shape[1]
    num_fea_pro = batch_xpro.shape[1]
    Xpro = batch_xpro.reshape(1, 1, num_fea_pro).repeat(1, num_res, 1)
    batch_x = torch.cat([batch_xres, Xpro], dim=2)
    out_dom = model(batch_x)

    Ypred_dom = torch.sigmoid(out_dom).cpu().detach().numpy()[0]

    Ypred_dom_file = output_dir + '/Ypred_dom.npy'
    np.save(Ypred_dom_file, Ypred_dom)


if __name__ == "__main__":
    main()
