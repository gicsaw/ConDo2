#!/usr/bin/env python
import sys
import numpy as np
import matplotlib.pyplot as plt

# Secondary Structure
COIL = ['S', 'T', ' ', '_']  # ' ' == '_'
HELIX = ['H', 'G', 'I']
STRAND = ['E', 'B']
SS8toSS3 = {'S': 'C', 'T': 'C', ' ': 'C', '_': 'C',
            'H': 'H', 'G': 'H', 'I': 'H', 'E': 'E', 'B': 'E'}

# SS: C, H, E
SS3_dict = {'C': 0, 'H': 1, 'E': 2}

SS8_dict = {'S': 0, 'T': 1, ' ': 2, '_': 2,
            'H': 3, 'G': 4, 'I': 5, 'E': 6, 'B': 7}
SS8_str = 'ST_HGIEB'
SS3_str = 'CHE'
SA3_str = 'BIE'
# SA: B, I, E
SA3_dict = {'B': 0, 'I': 1, 'E': 2}


def read_results_ss(target, seq, pred_dir):

    Ypred_ss8_file = pred_dir + '/Ypred_ss8.npy'
    Ypred_ss8 = np.load(Ypred_ss8_file)

    num_res = Ypred_ss8.shape[0]

    Ypred_ss3 = np.zeros([num_res, 3], dtype=np.float32)
    Ypred_ss3[:, 0] = Ypred_ss8[:, 0] + Ypred_ss8[:, 1] + Ypred_ss8[:, 2]
    Ypred_ss3[:, 1] = Ypred_ss8[:, 3] + Ypred_ss8[:, 4] + Ypred_ss8[:, 5]
    Ypred_ss3[:, 2] = Ypred_ss8[:, 6] + Ypred_ss8[:, 7]

    predfile = pred_dir + '/' + target+'.ss8'
    fp_ss8 = open(predfile, 'w')
    line_out = '#resudie_idx AA SS8 S T _ H G I E B \n'
    fp_ss8.write(line_out)

    for i in range(0, num_res):
        j = Ypred_ss8[i].argmax()
        ss8 = SS8_str[j]
        line_out = '%4d %s %s' % (i+1, seq[i], ss8)
        for j in range(0, 8):
            line_out += ' %6.4f' % (Ypred_ss8[i][j])
        line_out += '\n'
        fp_ss8.write(line_out)
    fp_ss8.close()

    predfile = pred_dir + '/' + target+'.ss3'
    fp_ss3 = open(predfile, 'w')
    line_out = '#resudie_idx AA SS3 C H E \n'
    fp_ss3.write(line_out)

    for i in range(0, num_res):
        j = Ypred_ss3[i].argmax()
        ss3 = SS3_str[j]
        line_out = '%4d %s %s' % (i+1, seq[i], ss3)
        for j in range(0, 3):
            line_out += ' %6.4f' % (Ypred_ss3[i][j])
        line_out += '\n'
        fp_ss3.write(line_out)
    fp_ss3.close()

    return Ypred_ss8, Ypred_ss3


def read_results_sa(target, seq, pred_dir):

    Ypred_sa3_file = pred_dir + '/Ypred_sa3.npy'
    Ypred_rsa_file = pred_dir + '/Ypred_rsa.npy'
    Ypred_sa3 = np.load(Ypred_sa3_file)
    Ypred_rsa = np.load(Ypred_rsa_file)

    num_res = Ypred_sa3.shape[0]

    predfile = pred_dir + '/' + target+'.sa'
    fp_sa = open(predfile, 'w')
    line_out = '#resudie_idx AA RSA SA3 B I E \n'
    fp_sa.write(line_out)

    for i in range(0, num_res):
        j = Ypred_sa3[i].argmax()
        sa3 = SA3_str[j]
        line_out = '%4d %s %6.4f %s' % (i+1, seq[i], Ypred_rsa[i][0], sa3)
        for j in range(0, 3):
            line_out += ' %6.4f' % (Ypred_sa3[i][j])
        line_out += '\n'
        fp_sa.write(line_out)
    fp_sa.close()

    return Ypred_sa3, Ypred_rsa


def read_results_dom(target, seq, pred_dir, conf_cut=1.3, NC=40):

    Ypred_dom_file = pred_dir + '/Ypred_dom.npy'

    Ypred_dom = np.load(Ypred_dom_file)

    num_res = Ypred_dom.shape[0]
    predfile = pred_dir + '/' + target+'.dom'

    fp_dom = open(predfile, 'w')
    line_out = '#residue_idx amino_acid score_sum score_5 score_10 score_15 score_20\n'
    fp_dom.write(line_out)
    scores = []
    for i in range(0, num_res):
        score = Ypred_dom[i].sum()
        line_out = '%4d %s %6.4f' % (i+1, seq[i], score)
        for j in range(0, 4):
            line_out += ' %6.4f' % (Ypred_dom[i][j])
        line_out += '\n'
        fp_dom.write(line_out)
        scores += [score]
    npscores = np.array(scores)
    arg_scores = np.argsort(-npscores)
    fp_dom.close()

    boundary = []
    boundary2 = []
    bd2 = []
    for i in range(0, num_res):
        k = arg_scores[i]
        score = scores[k]
        if score < conf_cut:
            break
        if (k < 1 or k >= num_res-1):
            continue
        check = 0
        for bb in boundary:
            if abs(bb-(k+1)) < 40:
                check += 1
        if check == 0:
            boundary += [k+1]
            if scores[k+1] < scores[k-1]:
                boundary2 += [[k, k+1]]
            else:
                boundary2 += [[k+1, k+2]]
            bd2 += [[k+1, score]]

    Ncount = 0
    line_db = ''
    if len(bd2) == 0:
        print(target+' is single domain')
        line_out = 'predicted boundary: None '
        return Ypred_dom, boundary2, bd2

    arg = np.array(bd2)[:, 0].argsort()
    for j in range(0, len(bd2)):
        k = arg[j]
        i = bd2[k][0]
        score = bd2[k][1]
        line_db += '%d %6.4f\n' % (i+1, score)
        if (score >= conf_cut) and (i > NC and i < num_res-NC-1):
            Ncount += 1

    if Ncount == 0:
        print(target+' is single domain')
        line_out = 'predicted boundary: '
        print(line_out)
        print(line_db)
        return Ypred_dom, boundary2, bd2

    print('#' + target + ' is multi domain')
    line_out = '#predicted boundary: residue_index score '
    print(line_out)
    print(line_db)
    return Ypred_dom, boundary2, bd2


def read_fasta(file_name):
    fp = open(file_name)
    lines = fp.readlines()
    fp.close()

    seq = ""
    check = 0
    for line in lines:
        if line.startswith(">"):
            if check == 1:
                break
            check = 0
            continue
        seq += line.strip()
    return seq


def write_output(out_file, seq, Ypred_ss8, Ypred_ss3, Ypred_sa3, boundary2, line_len):
    num_res = len(seq)

    line_ss8 = ''
    line_ss3 = ''
    line_sa3 = ''
    line_db = ''

    be = []
    bs = []
    for b in boundary2:
        be += [b[0]]
        bs += [b[1]]

    for i in range(num_res):
        j_ss8 = Ypred_ss8[i].argmax()
        ss8 = SS8_str[j_ss8]
        line_ss8 += ss8
        j_ss3 = Ypred_ss3[i].argmax()
        ss3 = SS3_str[j_ss3]
        line_ss3 += ss3

        j_sa3 = Ypred_sa3[i].argmax()
        sa3 = SA3_str[j_sa3]
        line_sa3 += sa3

        if i in be:
            line_db += '>'
        elif i in bs:
            line_db += '<'
        else:
            line_db += '-'

    num_lines = int(np.ceil(num_res/line_len))
    fp_out = open(out_file, 'w')
    for k in range(num_lines):
        ini = k * line_len
        fin = (k+1) * line_len
        if fin > num_res:
            fin = num_res
        ll = fin-ini
        blank = ' '*(ll-5)
        line_out = '          %-4d%s%d\n' % (ini+1, blank, fin)
        line_out += 'query   : %s\n' % seq[ini:fin]
        line_out += 'ss8     : %s\n' % line_ss8[ini:fin]
        line_out += 'ss3     : %s\n' % line_ss3[ini:fin]
        line_out += 'sa3     : %s\n' % line_sa3[ini:fin]
        line_out += 'boundary: %s\n' % line_db[ini:fin]
        line_out += '\n'

        fp_out.write(line_out)
    fp_out.close()


def draw_pas(pas_file, pas_fig, num_res):
    fp = open(pas_file)
    lines = fp.readlines()
    fp.close()
    x_list = []
    y_list = []
    z_list = []
    for i, line in enumerate(lines):
        lis = line.strip().split()
        x_list += [int(lis[0])]
        y_list += [int(lis[1])]
        z_list += [float(lis[2])]

    x = np.array(x_list).reshape(num_res, num_res)
    y = np.array(y_list).reshape(num_res, num_res)
    z = np.array(z_list).reshape(num_res, num_res)

    plt.figure(figsize=(6, 6))
    plt.rc('font', size=12)
    plt.rc('axes', titlesize=16, labelsize=16)
    plt.rc('xtick', labelsize=12)
    plt.rc('ytick', labelsize=12)
    plt.rc('legend', fontsize=12)
    plt.rc('figure', titlesize=24)

    plt.xlabel('Sequence')
    plt.ylabel('Sequence')
    plt.title('PAS')
    plt.pcolormesh(x, y, z)
    plt.tight_layout()
    plt.savefig(pas_fig, dpi=300)


def draw_ccm(ccm_file, ccm_fig, num_res):
    fp = open(ccm_file)
    lines = fp.readlines()
    fp.close()
    x_list = []
    y_list = []
    z_list = []
    for i, line in enumerate(lines):
        lis = line.strip().split()
        x = int(lis[0])
        y = int(lis[1])
        z = float(lis[2])
        if z <= 0.00000001:
            continue
        x_list += [x]
        y_list += [y]
        z_list += [z]

    x = np.array(x_list)
    y = np.array(y_list)
    z = np.array(z_list)

    plt.figure(figsize=(6, 6))
    plt.rc('font', size=12)
    plt.rc('axes', titlesize=16, labelsize=16)
    plt.rc('xtick', labelsize=12)
    plt.rc('ytick', labelsize=12)
    plt.rc('legend', fontsize=12)
    plt.rc('figure', titlesize=24)

    plt.xlabel('Sequence')
    plt.ylabel('Sequence')
    plt.title('CCMpred')
    plt.scatter(x, y, s=z, c=z, marker='o')
    plt.tight_layout()
#    plt.show()
    plt.savefig(ccm_fig, dpi=300)


def draw_com(com_file, com_fig, num_res):

    fp = open(com_file)
    lines = fp.readlines()
    fp.close()
    x_list = []
    y_list = []
    for i, line in enumerate(lines):
        lis = line.strip().split()
        x = int(lis[0])
        y = float(lis[1])
        x_list += [x]
        y_list += [y]

    x = np.array(x_list)
    y = np.array(y_list)

    plt.figure(figsize=(6, 4))
    plt.rc('font', size=12)
    plt.rc('axes', titlesize=16, labelsize=16)
    plt.rc('xtick', labelsize=12)
    plt.rc('ytick', labelsize=12)
    plt.rc('legend', fontsize=12)
    plt.rc('figure', titlesize=24)

    plt.xlabel('Sequence')
    plt.ylabel('Modularity')
    plt.title('')
    plt.plot(x, y, '-')
    plt.tight_layout()
#    plt.show()
    plt.savefig(com_fig, dpi=300)


def main():

    if len(sys.argv) < 2:
        print('gen_results.py target')
        sys.exit()

    target = sys.argv[1]
    fasta_file = './' + target+'.fasta'
    seq = read_fasta(fasta_file)

    num_res = len(seq)
    pas_file = './' + target + '_PAS4.txt'
    pas_fig = './' + target + '_PAS4.png'
    draw_pas(pas_file, pas_fig, num_res)

    ccm_file = './result_ccm.txt'
    ccm_fig = './result_ccm.png'
    draw_ccm(ccm_file, ccm_fig, num_res)

    com_file = './community_ccm.txt'
    com_fig = './community_ccm.png'
    draw_com(com_file, com_fig, num_res)

    pred_dir = '.'
    Ypred_ss8, Ypred_ss3 = read_results_ss(target, seq, pred_dir)
    Ypred_sa3, Ypred_rsa = read_results_sa(target, seq, pred_dir)
    conf_cut = 1.3
    Ypred_dom, boundary2, bd2 = read_results_dom(
        target, seq, pred_dir, conf_cut)

    line_len = 60
    out_file = './' + target + '_result.txt'
    write_output(out_file, seq, Ypred_ss8, Ypred_ss3,
                 Ypred_sa3, boundary2, line_len)


if __name__ == '__main__':
    main()
