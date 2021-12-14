import numpy as np 

def incrimination(C: list, X: np.ndarray, b: int):
    S = []
    p = np.copy(X[:])
    pre_f_S =  0
    while len(S) < b:
        f_S = 0
        max_index = -1
        delta_f = -100
        for i in C:
            temp = np.argmax(p[i])
            S.append(temp)
            f_S_temp = 0
            for j in C:
                f_S_temp += np.max(X[j][S])
            if delta_f < f_S_temp - pre_f_S:
                max_index = np.argmax(p[j])
                delta_f = f_S_temp - pre_f_S
                f_S = f_S_temp
            S = S[:-1]
        S.append(max_index)
        p[:, S] = -100
        delta_f = f_S - pre_f_S
        pre_f_S = f_S
    #print("incrimination: ", pre_f_S)
    return f_S

def get_incrimination(C: list, X: np.ndarray, n: int, W_t: np.ndarray, fp_num):
    max_index = []
    for i in range(W_t.shape[0]):
        max_index.append(np.argsort(W_t[i])[-fp_num:])
    max_index = np.array(max_index)
    oss_fp_j_j_weights = []
    oss_fp_j_j_normals = []
    for index ,clu_i in enumerate(C):
        oss_fp_j_j_weight = []
        oss_fp_j_j_normal = []
        f_S_max = np.array(incrimination(clu_i, X, n))
        for b in range(1, fp_num + 1):
            f_S_max_wei = np.array(incrimination(clu_i, X[:, max_index[index, -b: ]], b))
            f_S_normal = np.array(incrimination(clu_i, X, b ))
            oss_fp_j_j_weight.append(f_S_max_wei / f_S_max)
            oss_fp_j_j_normal.append(f_S_normal / f_S_max)
        print("------------------")
        print(oss_fp_j_j_weight)
        print(oss_fp_j_j_normal)
        oss_fp_j_j_weights.append(oss_fp_j_j_weight)
        oss_fp_j_j_normals.append(oss_fp_j_j_normal)
    return np.array(oss_fp_j_j_weights), np.array(oss_fp_j_j_normals)