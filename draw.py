import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import os

mode = 1
iters_list = [200, 500, 1000, 1500, 2000]
#iters_list = [100, 200, 300, 400, 500]
#iters_list = [200, 400, 600]
#iters_list = [1000]
csv_paths = ['melting_golden_3d_rendering_WithoutStyleEdit_Neg_new1', 'melting_golden_3d_rendering_WithoutStyleEdit_new1']
names = ['lora+neg', 'lora']
pic_name = 'Van_Gogh_WithoutStyleEdit_mode0' 
flag = 0
print_ = False

colors = ['blue', 'red', 'green']
cols = ['img score', 'text prompt score']
titles = ['style score', 'text score']
save_path = 'outcomes/figures'
save_path_csv = 'outcomes/datas'

csv_paths = ['outcomes/clip_score/' + m +'/'+m + '.csv' for m in csv_paths]

def iter_score(data):
    n_iter = len(iters_list)
    ret = [[0.0, 0.0] for m in range(n_iter)]
    n_data = 0
    name, s1_, s2_ = data['name'].values.tolist(), data[cols[0]].values.tolist(), data[cols[1]].values.tolist()
    for iters, s1, s2 in zip(name, s1_, s2_):
        index = iters.index('_')
        iters = iters[index+1:]
        index = iters.index('_')
        iters = iters[:index]
        iters = int(iters[4:])
        
        if iters in iters_list:
            n_data+=1
            it = iters_list.index(iters)
            #it = int(iters/iter_gap)-1
            ret[it][0] += s1
            ret[it][1] += s2
        else:
            continue
    a = (n_data/n_iter)
    for i in range(len(ret)):
        ret[i][0] /= a 
        ret[i][1] /= a
        
    return pd.DataFrame(ret, columns=[cols[0], cols[1]])

def get_name(col):
    name = pic_name + '_' + col + '.png'
    return os.path.join(save_path, name)

def save_data(data, name):
    file_name = pic_name+'_'+name +'.csv'
    path = os.path.join(save_path_csv, file_name)
    data.to_csv(path)

col = cols[flag]
title = titles[flag]
if __name__ == '__main__':
    if mode == 1:
        #compare
        x = iters_list
        #x = [(m+1) * iter_gap for m in range(n_iter)]
        datas = [iter_score(pd.read_csv(m)) for m in csv_paths]
        n_data = len(datas)
        
        #draw
        fig = plt.figure(1)
        plt.title(title)
        plt.xlabel('iters')
        plt.ylabel('CLIP Score')
        for i in range(n_data):
            save_data(datas[i], names[i])
            y = datas[i][col].values.tolist()
            plt.plot(x, y, color = colors[i], label = names[i])
            if print_ == True:
                print(names[i],datas[i])
        save_name = get_name(col)
        plt.legend()
        plt.savefig(save_name)
            
            