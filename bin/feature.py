#!/usr/bin/env python
import numpy as np
import sys
import os

Res31 = {'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
         'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
         'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
         'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y',
         'ASX': 'N', 'GLX': 'Q', 'UNK': 'X', 'INI': 'K', 'AAR': 'R',
         'ACE': 'X', 'ACY': 'G', 'AEI': 'T', 'AGM': 'R', 'ASQ': 'D',
         'AYA': 'A', 'BHD': 'D', 'CAS': 'C', 'CAY': 'C', 'CEA': 'C',
         'CGU': 'E', 'CME': 'C', 'CMT': 'C', 'CSB': 'C', 'CSD': 'C',
         'CSE': 'C', 'CSO': 'C', 'CSP': 'C', 'CSS': 'C', 'CSW': 'C',
         'CSX': 'C', 'CXM': 'M', 'CYG': 'C', 'CYM': 'C', 'DOH': 'D',
         'EHP': 'F', 'FME': 'M', 'FTR': 'W', 'GL3': 'G', 'H2P': 'H',
         'HIC': 'H', 'HIP': 'H', 'HTR': 'W', 'HYP': 'P', 'KCX': 'K',
         'LLP': 'K', 'LLY': 'K', 'LYZ': 'K', 'M3L': 'K', 'MEN': 'N',
         'MGN': 'Q', 'MHO': 'M', 'MHS': 'H', 'MIS': 'S', 'MLY': 'K',
         'MLZ': 'K', 'MSE': 'M', 'NEP': 'H', 'NPH': 'C', 'OCS': 'C',
         'OCY': 'C', 'OMT': 'M', 'OPR': 'R', 'PAQ': 'Y', 'PCA': 'Q',
         'PHD': 'D', 'PRS': 'P', 'PTH': 'Y', 'PYX': 'C', 'SEP': 'S',
         'SMC': 'C', 'SME': 'M', 'SNC': 'C', 'SNN': 'D', 'SVA': 'S',
         'TPO': 'T', 'TPQ': 'Y', 'TRF': 'W', 'TRN': 'W', 'TRO': 'W',
         'TYI': 'Y', 'TYN': 'Y', 'TYQ': 'Y', 'TYS': 'Y', 'TYY': 'Y',
         'YOF': 'Y', 'FOR': 'X', '---': '-', 'PTR': 'Y', 'LCX': 'K',
         'SEC': 'D', 'MCL': 'K', 'LDH': 'K'}

Res20 = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
         'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL']

Res13 = {'A': 'ALA', 'R': 'ARG', 'N': 'ASN', 'D': 'ASP', 'C': 'CYS',
         'Q': 'GLN', 'E': 'GLU', 'G': 'GLY', 'H': 'HIS', 'I': 'ILE',
         'L': 'LEU', 'K': 'LYS', 'M': 'MET', 'F': 'PHE', 'P': 'PRO',
         'S': 'SER', 'T': 'THR', 'W': 'TRP', 'Y': 'TYR', 'V': 'VAL',
         'X': 'UNK'}

seqcode1 = "-ARNDCQEGHILKMFPSTWYVX"
code_dict = {'-': 0, 'A': 1, 'R': 2, 'N': 3, 'D': 4, 'C': 5, 'Q': 6,
             'E': 7, 'G': 8, 'H': 9, 'I': 10, 'L': 11, 'K': 12,
             'M': 13, 'F': 14, 'P': 15, 'S': 16, 'T': 17, 'W': 18,
             'Y': 19, 'V': 20, 'X': 21}
Ntype = 21

# Secondary Structure
COIL = ['S', 'T', ' ', '_']  # ' ' == '_'
HELIX = ['H', 'G', 'I']
STRAND = ['E', 'B']
SS8toSS3 = {'S': 'C', 'T': 'C', ' ': 'C', '_': 'C',
            'H': 'H', 'G': 'H', 'I': 'H', 'E': 'E', 'B': 'E'}

# Solvent Accessibility
MAXACC = {'X': 180, 'A': 115, 'R': 225, 'N': 160, 'D': 150, 'C': 135, 'Q': 180,
          'E': 190, 'G': 75, 'H': 195, 'I': 175, 'L': 170, 'K': 200, 'M': 185,
          'F': 210, 'P': 145, 'S': 115, 'T': 140, 'W': 255, 'Y': 230, 'V': 155,
          'B': 155, 'Z': 185}
THR3 = [0.09, 0.36]

# Amino Acid hydrophobicity
# ref http://stevegallik.org/cellbiologyolm_Ex02_P03.html
# Eisenberg and Weiss; Dngleman; Kyte and Doolittle; Hoop and Woods; Janin
AA_hydrophobicity = {
    'I': [0.73,  3.1,  4.5, -1.8,  0.7],
    'F': [0.61,  3.7,  2.8, -2.5,  0.5],
    'V': [0.54,  2.6,  4.2, -1.5,  0.6],
    'L': [0.53,  2.8,  3.8, -1.8,  0.5],
    'W': [0.37,  1.9, -0.9, -3.4,  0.3],
    'M': [0.26,  3.4,  1.9, -1.3,  0.4],
    'A': [0.25,  1.6,  1.8, -0.5,  0.3],
    'G': [0.16,  1.0, -0.4,  0.0,  0.3],
    'C': [0.04,  2.0,  2.5, -1.0,  0.9],
    'Y': [0.02, -0.7, -1.3, -2.3, -0.4],
    'P': [-0.07, -0.2, -1.6,  0.0, -0.3],
    'T': [-0.18,  1.2, -0.7, -0.4, -0.2],
    'S': [-0.26,  0.6, -0.8,  0.3, -0.1],
    'H': [-0.40, -3.0, -3.2, -0.5, -0.1],
    'E': [-0.62, -8.2, -3.5,  3.0, -0.7],
    'N': [-0.64, -4.8, -3.5,  0.2, -0.5],
    'Q': [-0.69, -4.1, -3.5,  0.2, -0.7],
    'D': [-0.72, -9.2, -3.5,  3.0, -0.6],
    'K': [-1.10, -8.8, -3.9,  3.0, -1.8],
    'R': [-1.80, -12.3, -4.5,  3.0, -1.4],
    'X': [0.0, 0.0, 0.0, 0.0, 0.0]
}
# Amino Acid charge -1, 0, 1
AA_charge = {
    'A': 0, 'V': 0, 'I': 0, 'L': 0, 'M': 0, 'F': 0, 'Y': 0, 'W': 0,
    'C': 0, 'G': 0, 'P': 0,
    'S': 0, 'T': 0, 'N': 0, 'Q': 0,
    'D': -1, 'E': -1,
    'R': 1, 'H': 1, 'K': 1,
    'X': 0
}

# SS: C, H, E
SS3_dict = {'C': 0, 'H': 1, 'E': 2}

SS8_dict = {'S': 0, 'T': 1, ' ': 2, '_': 2,
            'H': 3, 'G': 4, 'I': 5, 'E': 6, 'B': 7}

# SA: B, I, E
SA3_dict = {'B': 0, 'I': 1, 'E': 2}


def set_ss3(ss):

    #    if   ss in COIL  :
    #        return 'C'
    #    elif ss in HELIX :
    #        return 'H'
    #    elif ss in STRAND:
    #        return 'E'
    return SS8toSS3[ss]


def set_sa3(rsa):
    if rsa < THR3[0]:
        return 'B'
    elif rsa < THR3[1]:
        return 'I'
    else:
        return 'E'


def creator(q, data, Nproc):
    for d in data:
        idx = d[0]
        da = d[1]
        q.put((idx, da))

    for i in range(0, Nproc):
        q.put('DONE')


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


def read_msa0(file_name, Nres):

    fp = open(file_name)
    lines = fp.readlines()
    fp.close()
    msa = list()
    msa2 = list()
    pasinfo = list()
    Npasinfo = 0

    for line in lines:
        if line[0] == ">":
            mm2 = np.zeros(Nres, dtype=int)
            lis = line.strip().split("/")
            inifin = lis[1]
            inifin_list = inifin.strip().split(",")
            pcheck = 0
            for inifin2 in inifin_list:
                inifin3 = inifin2.strip().split("-")
                ini = int(inifin3[0])
                fin = int(inifin3[1])
                if ((ini > 20) or (fin < Nres-20)):
                    pcheck = 1
                for i in range(ini-1, fin):
                    mm2[i] = 1

                msa2 += [mm2]
                pasinfo += [pcheck]
                Npasinfo += 1

            continue
        else:
            mm = np.zeros(Nres, dtype=int)
            for i in range(0, Nres):
                cc = line[i]
                if cc not in code_dict:
                    j = 21
                else:
                    j = code_dict[cc]
                mm[i] = j
            msa += [mm]
    msa = np.concatenate([msa])
    msa2 = np.concatenate([msa2])
    pasinfo = np.array(pasinfo)

    return msa, msa2, pasinfo


def read_msa(file_name, Nres):

    fp = open(file_name)
    lines = fp.readlines()
    fp.close()
    Npasinfo = 0
    Nmsa = 0

    PAS = np.zeros([Nres, Nres], dtype=np.float32)
# Nsig[i][0]: left_gap &right_gap
# Nsig[i][1]: left_gap
# Nsig[i][2]: right_gap
    Nsig = np.zeros([Nres, 3], dtype=np.long)

    for line in lines:
        if line[0] == ">":
            Nmsa += 1
            lis = line.strip().split("/")
            inifin = lis[1]
            inifin_list = inifin.strip().split(",")
#            pcheck = 0
            for inifin2 in inifin_list:
                inifin3 = inifin2.strip().split("-")
                ini = int(inifin3[0])
                fin = int(inifin3[1])
#                if ((ini > 20) or (fin < Nres-20)):
#                    pcheck = 1
                PAS[ini-1:fin, ini-1:fin] += 1
                Npasinfo += 1
                Nsig[ini-1][0] += 1
                Nsig[fin-1][0] += 1
                Nsig[ini-1][1] += 1
                Nsig[fin-1][2] += 1

    if Npasinfo != 0:
        PAS = PAS/Npasinfo

    return PAS, Nsig, Npasinfo, Nmsa


def cal_sig(Nsig, Nres, Nmsa):

    Nsite5 = np.zeros([Nres, 3], np.long)
    Nsig5 = np.zeros([Nres, 3], np.long)
    psig5 = np.zeros([Nres, 3], np.float32)
    msig5 = np.zeros([Nres, 3], np.float32)
    fsite5 = np.zeros([Nres, 3], np.float32)

    if Nmsa < 1:
        return fsite5, psig5, msig5

    for i in range(0, Nres):
        ini = i - 5
        fin = i + 5
        if(ini < 0):
            ini = 0
        if(fin > Nres-1):
            fin = Nres-1
        Nsig5[i][1] = Nsig[ini:fin+1, 1].sum()
        Nsig5[i][2] = Nsig[ini:fin+1, 2].sum()
        Nsig5[i][0] = Nsig5[i][1] + Nsig5[i][2]
        Nsite5[i][1] = Nsig[ini:fin+1, 1].astype(np.bool).sum()
        Nsite5[i][2] = Nsig[ini:fin+1, 2].astype(np.bool).sum()
        Nsite5[i][0] = Nsite5[i][1] + Nsite5[i][2]

    fsite5[:, 1:3] = Nsite5[:, 1:3] / 11.0
    fsite5[:, 0] = Nsite5[:, 0] / 22.0

    ggap = 20 * Nres // 100
    if (ggap < 20):
        ggap = 20
    if ggap > 100:
        ggap = 100
    ini = ggap
    fin = Nres - ggap
    max_sig5 = Nsig5[ini:fin].max(axis=0)
    for k in range(0, 3):
        if max_sig5[k] == 0:
            max_sig5[k] = Nsig5[:, k].max()

    psig5[:, 1:3] = Nsig5[:, 1:3] / float(Nmsa)
    psig5[:, 0] = Nsig5[:, 0] / (2.0 * float(Nmsa))
    msig5[:, 0] = Nsig5[:, 0] / float(max_sig5[0])
    msig5[:, 1] = Nsig5[:, 1] / float(max_sig5[1])
    msig5[:, 2] = Nsig5[:, 2] / float(max_sig5[2])

    psig5 = psig5.clip(-1.0, 1.0)
    msig5 = msig5.clip(-1.0, 1.0)

    return fsite5, psig5, msig5


def read_ck2(file_name):

    fp = open(file_name)
    lines = fp.readlines()
    fp.close()

    Nres = int(lines[0].strip())
#    seq = lines[1].strip()
    Nat = 20
    profile = np.zeros([Nres, Nat], dtype=np.float32)
    for i, line in enumerate(lines[2:]):
        lis = line.strip().split()
        for j in range(Nat):
            profile[i][j] = float(lis[j])

    return profile


def read_ccmpred(file_name, Nres):

    fp = open(file_name)
    lines = fp.readlines()
    fp.close()

    ccmpred = np.zeros([Nres, Nres], dtype=np.float32)
    for i, line in enumerate(lines):
        lis = line.strip().split()
        for j, ll in enumerate(lis):
            ccmpred[i][j] = ll

    for i in range(Nres):
        for j in range(i, i+12):
            if j < Nres:
                ccmpred[i][j] = 0.0
                ccmpred[j][i] = 0.0

    return ccmpred


def cal_ccm_cut(ccmpred, Nres):

    sum1 = 0
    sum2 = 0
    Nedge = 0

    for i in range(Nres):
        for j in range(i+12, Nres):
            Nedge += 1
            sum1 += ccmpred[i][j]
            sum2 += pow(ccmpred[j][i], 2)

    avg1 = sum1/Nedge
    avg2 = sum2/Nedge
    dd = avg2-pow(avg1, 2)
    stddev = np.sqrt(dd)

    ccm_cut = avg1+2*stddev
    return ccm_cut


def cal_ccm_community(ccmpred, ccm_cut, Nres):

    community = np.zeros([Nres], dtype=np.float32)
    community0 = np.zeros([Nres], dtype=np.float32)

    sum_con = 0
    for i in range(Nres):
        for j in range(i+1, Nres):
            if ccmpred[i][j] >= ccm_cut:
                sum_con += ccmpred[i][j]

    M = sum_con

    if M == 0.0:
        return community0
    ddd1 = 0
    ddd2 = M
    dm1 = 0
    dm2 = 2*M

    for k in range(Nres):
        dd1 = 0
        dd2 = 0
        dd0 = 0
        for i in range(0, k):
            if(ccmpred[k][i] >= ccm_cut):
                dd1 += ccmpred[k][i]
                dd0 += ccmpred[k][i]
        for i in range(k+1, Nres):
            if(ccmpred[k][i] >= ccm_cut):
                dd2 += ccmpred[k][i]
                dd0 += ccmpred[k][i]

        ddd1 += dd1
        ddd2 -= dd2
        dm1 += dd0
        dm2 -= dd0
        community1 = ddd1/M+ddd2/M
        community2 = pow(dm1/(2*M), 2)+pow(dm2/(2*M), 2)
        community[k] = community1-community2

    for i in range(Nres):
        if i == 0:
            community0[i] = community[i]/2.0
            continue
        community0[i] = (community[i-1]+community[i])/2.0

    return community0


def write_pas(out_file, PAS, Nres):
    fp = open(out_file, "w")
    for i in range(Nres):
        for j in range(Nres):
            line_out = "%4d %4d %6.4f\n" % (i+1, j+1, PAS[i][j])
            fp.write(line_out)
    fp.close()


def cal_pas_sum(PAS, Nres):

    PASsumN = np.zeros(Nres, dtype=np.float32)
    PASsumC = np.zeros(Nres, dtype=np.float32)
    PASdiag = np.zeros(Nres, dtype=np.float32)

    for i in range(0, Nres):
        PASsumN[i] += PAS[i, :i+1].sum()/(i+1)
        PASsumC[i] += PAS[i, i+1:].sum()/(Nres-i)
        PASdiag[i] = PAS[i, i]

    return PASsumN, PASsumC, PASdiag


def write_pas_sum(out_file, PASsumN, PASsumC, PASdiag, Nres):
    fp = open(out_file, "w")
    for i in range(Nres):
        line_out = "%4d %6.4f %6.4f %6.4f\n" % (
            i+1, PASsumN[i], PASsumC[i], PASdiag[i])
        fp.write(line_out)
    fp.close()


def write_ccm(out_file, ccmpred, Nres, ccm_cut):

    fp = open(out_file, "w")
    for i in range(Nres):
        for j in range(Nres):
            if ccmpred[i][j] >= ccm_cut:
                line_out = "%4d %4d %6.4f\n" % (i+1, j+1, ccmpred[i][j])
            else:
                line_out = "%4d %4d %6.4f\n" % (i+1, j+1, 0.0)
            fp.write(line_out)
    fp.close()

    return


def write_community(out_file, community, Nres):
    fp = open(out_file, "w")
    for i in range(Nres):
        line_out = "%4d %6.4f\n" % (i+1, community[i])
        fp.write(line_out)
    fp.close()

    return


def read_pdb(pdb_file):

    n_file = open(pdb_file)
    lines = n_file.readlines()
    n_file.close()

    chain = dict()

    old_chain_id = ""
    for line in lines:
        if line[:6] == 'ATOM  ':
            chain_id = line[21]
            if chain_id != old_chain_id:
                CA_dict = dict()
                CB_dict = dict()
                res_dict = dict()
                atom_dict = dict()
                atom_map = dict()
                old_chain_id = chain_id
                chain[chain_id] = dict()
                idx = 0
                chain[chain_id]['CA_dict'] = CA_dict
                chain[chain_id]['CB_dict'] = CB_dict
                chain[chain_id]['res_dict'] = res_dict
                chain[chain_id]['atom_dict'] = atom_dict
                chain[chain_id]['atom_map'] = atom_map
                chain[chain_id]['seq'] = ""

            iseq = line[22:27].strip()
            atom = line[12:16].strip()
            res = line[17:20]
            coor = np.array([float(line[30:38]), float(
                line[38:46]), float(line[46:54])])

            if iseq not in res_dict:
                res_dict[iseq] = res
                chain[chain_id]['seq'] += Res31[res]
                idx += 1
                atom_map[idx] = iseq
                if res not in Res20:
                    print(iseq, res)

            if iseq not in atom_dict:
                atom_dict[iseq] = [line]
            else:
                atom_dict[iseq] += [line]

            if atom == "CA":
                CA_dict[iseq] = coor
            if res != "GLY":
                if atom == "CB":
                    CB_dict[iseq] = coor
            elif atom == "CA":
                CB_dict[iseq] = coor

    return chain


def read_amap(amap_file):
    map_dict = dict()
    map_dict2 = dict()
    fa_map = dict()

    fp = open(amap_file)
    lines = fp.readlines()
    fp.close()

    for line in lines:
        line_arr = line.split()
        fa_map[int(line_arr[0])] = line_arr[1]
        map_dict[int(line_arr[0])] = (line_arr[6], line_arr[2])
        if line_arr[6] == "-":
            continue
        map_dict2[line_arr[6]] = int(line_arr[0])

    return map_dict, map_dict2, fa_map


def gen_mask(map_dict, Nres):

    keys = sorted(map_dict.keys())
    mask = np.ones([Nres], dtype=np.float32)
    for i in keys:
        j0 = map_dict[i][1]
#        j = map_dict[i][0]
        if j0 == "-":
            ini = i-2
            if ini < 0:
                ini = 0
            fin = i+1
            mask[ini:fin] = 0

    mask = np.array(mask, dtype=np.float32)
    return mask


def read_ss4(ss4_file, map_dict, fa_map, Nres):

    fp = open(ss4_file)
    lines = fp.readlines()
    fp.close()

#    mask_ss4 = np.ones([Nres], dtype=np.float32)

# C, H, E
    SS3_dict = {'C': 0, 'H': 1, 'E': 2}
    data = []
    for line in lines:
        if line.startswith("#"):
            continue
        if line.strip() == "":
            continue
        lis = line.strip().split()
        data += [[int(lis[0]), lis[1], lis[2]]]

    keys = sorted(map_dict.keys())
    ss4 = []
    k = -1
    for i in keys:
        amino0 = fa_map[i]
        j = map_dict[i][0]
        j0 = map_dict[i][1]
#        if j!="-" and j0!="-":
        if j0 != "-":
            k += 1
            dd = data[k]
#            print(i,j,j0,k,dd)
            amino = dd[1]
            if amino != amino0:
                print(i, j, k, amino, amino0)
            ss4 += [SS3_dict[dd[2]]]
        else:
            ss4 += [3]
#            ini=i-2
#            if ini<0:
#                ini=0
#            fin=i+1
#            mask_ss4[ini:fin]=0
    ss4 = np.array(ss4, dtype=np.long)
    return ss4  # , mask_ss4


def read_sa4(sa4_file, map_dict, fa_map, Nres):

    fp = open(sa4_file)
    lines = fp.readlines()
    fp.close()
#    mask_sa4 = np.ones([Nres], dtype=np.float32)

# C, H, E
    SA3_dict = {'B': 0, 'I': 1, 'E': 2}
    data = []
    for line in lines:
        if line.startswith("#"):
            continue
        if line.strip() == "":
            continue
        lis = line.strip().split()
        data += [[int(lis[0]), lis[1], lis[2], float(lis[6])]]

    keys = sorted(map_dict.keys())
    sa4 = []
    rsa4 = []

    k = -1
    for i in keys:
        amino0 = fa_map[i]
        j = map_dict[i][0]
        j0 = map_dict[i][1]
        if j0 != "-":
            #        if j!="-" and j0!="-":
            k += 1
            dd = data[k]
            amino = dd[1]
            if amino != amino0:
                print(i, j, k, amino, amino0)
            sa4 += [SA3_dict[dd[2]]]
            rsa4 += [dd[3]]
        else:
            sa4 += [3]
            rsa4 += [0.00]
#            ini=i-2
#            if ini<0:
#                ini=0
#            fin=i+1
#            mask_sa4[ini:fin]=0

    sa4 = np.array(sa4, dtype=np.long)
    rsa4 = np.array(rsa4, dtype=np.float32)

    return sa4, rsa4  # , mask_sa4


def read_dssp(dssp_file, map_dict, fa_map, Nres):

    fp = open(dssp_file)
    lines = fp.readlines()
    fp.close()
    k = -1

    dssp_dict = dict()
    header_check = True
    for line in lines:
        if line.startswith('  #  RESIDUE AA'):
            header_check = False
            continue
        if header_check:
            continue

        k += 1
        if line[13] == '!':
            continue
        ires = line[6:10].strip()
        ins = line[10].strip()
        resna = str(ires)+ins
        amino = line[13]
        ss8 = line[16]
        ss3 = set_ss3(ss8)
        sol = float(line[35:38])
        rsa = sol/MAXACC[amino]
        sa3 = set_sa3(rsa)
        if rsa > 1.0:
            rsa = 1.0

        dssp_dict[resna] = [amino, ss8, ss3, sa3, rsa]

    keys = sorted(map_dict.keys())

    mask = np.ones([Nres], dtype=np.bool)
    ss8 = []
    ss3 = []
    sa3 = []
    rsa = []

    for i in keys:
        amino0 = fa_map[i]
        resna = map_dict[i][0]
        j0 = map_dict[i][1]
#        print(i,resna)
        if resna != "-" and j0 != "-":
            if resna not in dssp_dict:
                #                print("ddd", i, resna)
                ss8 += [8]
                ss3 += [3]
                sa3 += [3]
                rsa += [0.00]
                ini = i-2
                if ini < 0:
                    ini = 0
                fin = i+1
                mask[ini:fin] = 0
                continue
            dd = dssp_dict[resna]
            amino = dd[0]
            if amino != amino0:
                print(i, resna, amino, amino0)
            ss8 += [SS8_dict[dd[1]]]
            ss3 += [SS3_dict[dd[2]]]
            sa3 += [SA3_dict[dd[3]]]
            rsa += [dd[4]]
        else:
            ss8 += [8]
            ss3 += [3]
            sa3 += [3]
            rsa += [0.00]

            ini = i-2
            if ini < 0:
                ini = 0
            fin = i+1
            mask[ini:fin] = 0
    ss8 = np.array(ss8, dtype=np.long)
    ss3 = np.array(ss3, dtype=np.long)
    sa3 = np.array(sa3, dtype=np.long)
    rsa = np.array(rsa, dtype=np.float32)

    return ss8, ss3, sa3, rsa, mask


def cal_freq_aa(seq, Nres):
    freq_aa = np.zeros(20, dtype=np.float32)

    for i in range(Nres):
        k = code_dict[seq[i]] - 1
        if k < 20:
            freq_aa[k] += 1.0

    return freq_aa/Nres


def cal_res_hydrophobicity(seq, Nres):

    res_hydro = np.zeros([Nres, 5], dtype=np.float32)
    for i in range(Nres):
        res = seq[i]
        res_hydro[i] = np.array(AA_hydrophobicity[res], dtype=np.float32)

    res_hydro[:, 0] = res_hydro[:, 0]/1.8
    res_hydro[:, 1] = res_hydro[:, 1]/12.3
    res_hydro[:, 2] = res_hydro[:, 2]/4.5
    res_hydro[:, 3] = res_hydro[:, 3]/3.4
    res_hydro[:, 4] = res_hydro[:, 4]/1.4

    return res_hydro


def cal_res_charge(seq, Nres):
    res_charge = np.zeros([Nres, 1], dtype=np.float32)
    for i in range(Nres):
        res = seq[i]
        res_charge[i][0] = AA_charge[res]

    return res_charge


def cal_res_max_acc(seq, Nres):
    res_max_acc = np.zeros([Nres, 1], dtype=np.float32)
    for i in range(Nres):
        res = seq[i]
        res_max_acc[i][0] = MAXACC[res]

    res_max_acc = res_max_acc/255.0
    return res_max_acc


def gen_feature(target, seq, Nres, Nproc):

    #    freq_aa = cal_freq_aa(seq, Nres)
    #    res_hydro = cal_res_hydrophobicity(seq, Nres)
    #    res_charge = cal_res_charge(seq, Nres)
    #    res_max_acc = cal_res_max_acc(seq, Nres)

    ck2_file = target+".ck2"
    if not os.path.isfile(ck2_file):
        print(ck2_file, " is not exist")
    profile = read_ck2(ck2_file)
#    print(profile)

    msa_file = target+".msa"
    if not os.path.isfile(msa_file):
        print(msa_file, " is not exist")
    PAS, Nsig, Npasinfo, Nmsa = read_msa(msa_file, Nres)

    pas_out_file = target+"_PAS4.txt"
    write_pas(pas_out_file, PAS, Nres)

    PASsumN, PASsumC, PASdiag = cal_pas_sum(PAS, Nres)
    pas_sum_file = target+"_PAS4_sum.txt"
    write_pas_sum(pas_sum_file, PASsumN, PASsumC, PASdiag, Nres)

    fsite5, psig5, msig5 = cal_sig(Nsig, Nres, Nmsa)

    ccm_file = target+".ccmpred"
#    if Nmsa>5:
    if os.path.isfile(ccm_file):
        ccmpred = read_ccmpred(ccm_file, Nres)
        ccm_cut = cal_ccm_cut(ccmpred, Nres)
        community = cal_ccm_community(ccmpred, ccm_cut, Nres)
        ccm_out_file = "result_ccm.txt"
        write_ccm(ccm_out_file, ccmpred, Nres, ccm_cut)
        com_file = "community_ccm.txt"
        write_community(com_file, community, Nres)
    else:
        community = np.zeros([Nres], dtype=np.float32)

    comm_min = community.min()
    comm_max = community.max()
    PASN_max = PASsumN.max()
    PASC_max = PASsumC.max()
    PASD_max = PASdiag.max()
    Dpasinfo = 0
    if Npasinfo > 0:
        Dpasinfo = np.log10(Npasinfo)/5.0
    Dmsa = 0
    if Nmsa > 0:
        Dmsa = min(np.log10(Nmsa)/5.0, 1.0)

    fea_protein = [comm_min, comm_max, PASN_max, PASC_max, PASD_max,
                   Dpasinfo, Dmsa]
    fea_protein = np.array(fea_protein, dtype=np.float32)
#    fea_protein = np.concatenate([fea_protein, freq_aa])
    pos = []
    for i in range(0, Nres):
        Nter = i/1000.0
        Cter = (Nres-i-1)/1000
        NCter = Nres/1000.0
        pos += [[Nter, Cter, NCter]]

    pos = np.array(pos, dtype=np.float32)
    fea_long = np.concatenate([community.reshape(Nres, 1),
                               PASsumN.reshape(Nres, 1),
                               PASsumC.reshape(Nres, 1),
                               PASdiag.reshape(Nres, 1)], axis=1)
    fea_residue = np.concatenate(
        [profile, fea_long, pos], axis=1).astype(np.float32)
#    fea_residue = np.concatenate(
#        [profile, res_hydro, res_charge, res_max_acc, fea_long,
#            pos, fsite5, psig5, msig5], axis=1).astype(np.float32)

    fea_res_file = "Xdata_res.npy"
    np.save(fea_res_file, fea_residue)
    fea_pro_file = "Xdata_pro.npy"
    np.save(fea_pro_file, fea_protein)


def gen_label(target, Nres, Nproc):
    amap_file = target+".amap"
    map_dict, map_dict2, fa_map = read_amap(amap_file)

#    print(map_dict)
#    print(map_dict2)
#    print(fa_map)

#    mask = gen_mask(map_dict,Nres)
#    print(mask)

    dssp_file = target+".dssp"
    ss8, ss3, sa3, rsa, mask = read_dssp(dssp_file, map_dict, fa_map, Nres)

#    ss4_file=target+".ss4"
#    ss4 = read_ss4(ss4_file,map_dict,fa_map,Nres)
#    print(ss4)

#    sa4_file=target+".sa4"
#    sa4, rsa4 = read_sa4(sa4_file,map_dict,fa_map,Nres)
#    print(sa4)
#    print(rsa4)

#    print((ss3==ss4).all())
#    print((sa3==sa4).all())
#    print((rsa==rsa4).all())

#    print(ss3)
#    print(sa3)
#    print(mask)

#    print(np.concatenate([ss3.reshape(Nres,1),sa3.reshape(Nres,1),mask.reshape(Nres,1)],axis=1))

    if Nres != ss3.shape[0]:
        print("ss", Nres, ss3.shape)
    if Nres != sa3.shape[0]:
        print("sa", Nres, sa3.shape)

    label_ss8_file = "Ydata_ss8.npy"
    np.save(label_ss8_file, ss8)

    label_ss3_file = "Ydata_ss3.npy"
    np.save(label_ss3_file, ss3)
    label_sa3_file = "Ydata_sa3.npy"
    np.save(label_sa3_file, sa3)
    label_rsa_file = "Ydata_rsa.npy"
    np.save(label_rsa_file, rsa)

    mask_file = "Mdata.npy"
    np.save(mask_file, mask)

#    mask_ss_file = "Mdata_ss.npy"
#    np.save(mask_ss_file, mask_ss4)
#    mask_sa_file = "Mdata_sa.npy"
#    np.save(mask_sa_file, mask_sa4)

#    print(ss8)
#    print(ss3)


def main():

    if len(sys.argv) < 2:
        print("feature.py target Nproc")
        sys.exit()
    target = sys.argv[1]
    Nproc = 1
    if len(sys.argv) >= 3:
        Nproc = int(sys.argv[2])

#    print(target, Ncpu)

    fasta_file = target+".fasta"
    if not os.path.isfile(fasta_file):
        print(fasta_file, " is not exist")

    seq = read_fasta(fasta_file)
    Nres = len(seq)

# seq, feature
    gen_feature(target, seq, Nres, Nproc)


# structure, label
#    gen_label(target, Nres, Nproc)


if __name__ == '__main__':
    main()
