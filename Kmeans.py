import warnings
warnings.filterwarnings("ignore")
import os, click, math, collections
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
from scipy.stats.mstats import zscore
from yellowbrick.cluster import KElbowVisualizer
from yellowbrick.cluster import SilhouetteVisualizer

import matplotlib.pyplot as plt
import seaborn as sns
plt.switch_backend('agg')
sns.set_style( "white" )

@click.group()
def cli():
    pass

@cli.command(short_help='determine k by elbow')
@click.option('-f', help='input file')
@click.option('-g', default='r', show_default=True,help='r: cluster by rows\nc: cluster by columns')
@click.option('-krange', nargs=2,default=(2,5),show_default=True,help='min and max number of cluster, separated by commas')
def elbow(f,g,krange):
    df = pd.read_table(f,index_col=0,header=0)
    if g == 'c':
        df = df.T
    model = KMeans(random_state=1)
    min_k = krange[0]
    max_k = krange[1]
    visualizer = KElbowVisualizer(model, k=(min_k, max_k+1))
    visualizer.fit(df)  
    visualizer.show(outpath=os.path.basename(f).replace('.txt', '_elbow.pdf'),clear_figure=True)

@cli.command(short_help='determine k by  silhouette coefficient')
@click.option('-f', help='input file')
@click.option('-g', default='r', show_default=True, help='r: cluster by rows\nc: cluster by columns')
@click.option('-krange', nargs=2,default=(2,5), show_default=True, help='min and max number of cluster, separated by commas')
def sc(f,g,krange):
    df = pd.read_table(f,index_col=0,header=0)
    if g == 'c':
        df = df.T
    min_k = krange[0]
    max_k = krange[1]
    num_fig = max_k - min_k + 1
    num_row = math.ceil(num_fig/3.0)
    plt.figure(figsize=(12,num_row*4))
    for k in range(min_k, max_k+1):
        model = KMeans(k, random_state=1)
        visualizer = SilhouetteVisualizer(model,colors='yellowbrick')
        plt.subplot(num_row,3,k-min_k+1)
        visualizer.fit(df)
        plt.title('k=%d silhouette score %0.2f'%(k, visualizer.silhouette_score_))
        plt.grid(0)
        plt.tight_layout()
    plt.savefig(os.path.basename(f).replace('.txt', '_silhouette.pdf'))

@cli.command(short_help='run k-means clustering')
@click.option('-f', help='input file')
@click.option('-g', default='r',show_default=True, help='r: cluster by rows\nc: cluster by columns')
@click.option('-k', type=int, help='number of clusters')
@click.option('-heatmap', is_flag=True, help='if plot heatmap')
def kmeans(f,k,g,heatmap):
    df = pd.read_table(f,index_col=0,header=0)
    mat_f = os.path.basename(f).replace('.txt', '_{}K{}.txt'.format(g,k))
    cluster_f = os.path.basename(f).replace('.txt', '_{}K{}_cluster.txt'.format(g,k))
    if g == 'c':
        df = df.T
    kmeans = KMeans(n_clusters=k, random_state=1).fit(df)
    sort_ids = sorted(zip(df.index, kmeans.labels_), key=lambda x:x[1])
    ###################
    ###for heatmap divider
    global id_dict
    id_dict = collections.defaultdict(list)
    for ele in sort_ids:
        id_dict[ele[1]].append(ele[0])
    ###################    
    with open(cluster_f, 'w') as o:
        for x in sort_ids:
            o.write('\t'.join(map(str, x))+'\n')
    sort_ids = [x[0] for x in sort_ids]
    df = df.loc[sort_ids]
    if g == 'c':
        df = df.T
    df.to_csv(mat_f, sep='\t')
    if heatmap:
        plot_heatmap(mat_f,g,df)

def add_divider(df):
    ### how many line as black line
    if df.shape[0]<50:
        num_line = 1
    elif 500>df.shape[0]>50:
        num_line = 2
    else:
        num_line = int(df.shape[0]/200)
    new_ids = []
    sort_cluster = sorted(id_dict.keys())
    tmp = set([0])
    for c in sort_cluster:
        if c not in tmp:
            tmp.add(c)
        for idx in id_dict[c]:
            new_ids.append(idx)
        for n in range(num_line):
            line_idx = 'line_{}_{}'.format(c,n)
            new_ids.append(line_idx)  
    for c in sort_cluster[:-1]:
        for n in range(num_line):
            line_name = 'line_{}_{}'.format(c,n)
            df.loc[line_name] = None
    df = df.reindex(new_ids)
    return df  

def plot_heatmap(f,g,df):
    plt.rcParams[ "font.size" ] = 4.0
    plt.rcParams[ "figure.dpi" ] = 100
    plt.rcParams['axes.linewidth'] = 1
    plt.rcParams['axes.labelsize'] = 6 
    plt.rcParams[ "figure.figsize" ] = ( 2*0.8,2.75*0.8 )
    plt.rcParams[ "font.serif" ] = 'Arial'
    df = df.T.apply(zscore,0).T
    if g == 'c':
        df = df.T
    ### add blank line
    df = add_divider(df)
    if g =='c':
        df = df.T
    ### figure size
    x_l,y_l = (12,16*df.shape[0]/200.0) 
    if y_l>20:
        y_l = 20 
    plt.figure(figsize=(x_l, y_l))
    sns.heatmap(df,yticklabels=False,xticklabels=False,cmap='vlag',cbar_kws = {'shrink':0.2},center=0)
    fo = os.path.basename(f).replace('.txt','_heatmap.pdf')
    plt.savefig(fo)


cli()




