from mpl_toolkits.mplot3d import axes3d
import matplotlib
# matplotlib.use('QT5Agg')
import matplotlib.pyplot as plt
# plt.switch_backend('agg')
import numpy as np
import json
import networkx as nx
from itertools import combinations

from sklearn import linear_model
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.preprocessing import scale, normalize, minmax_scale, StandardScaler
import datetime


def calculate_logreg_score(inmat):
    std_scaler = StandardScaler()
    std_scaler.fit(inmat)
    inmat = std_scaler.transform(inmat)

    X = np.array(inmat)[:, :-1]
    y = np.array(inmat)[:, -1]

    logreg = linear_model.LogisticRegression(C=300.5, verbose=True, tol=1e-8, fit_intercept=True)
    logreg.fit(X, y)

    var = logreg.score(X, y)

    return var

def plot3d(inmat,inputcirc=None,title=""):

    fig = plt.figure()
    plt.title(datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y"))
    ax = fig.add_subplot(235, projection='3d')
    std_scaler = StandardScaler()
    std_scaler.fit(inmat)
    inmat = std_scaler.transform(inmat)
    for cords in inmat:
        ax.scatter(cords[0], cords[1], cords[2], c='r' if cords[3] < 1. else 'b', marker='o' if cords[3] < 1. else '^')

    X = np.array(inmat)[:, :-1]
    y = np.array(inmat)[:, -1]

    logreg = linear_model.LogisticRegression(C=300.5, verbose=True, tol=1e-8, fit_intercept=True)
    logreg.fit(X, y)

    zlr = lambda x, y: (-logreg.intercept_[0] - logreg.coef_[0][0] * x - logreg.coef_[0][1] * y) / logreg.coef_[0][2]

    tmp = np.linspace(-1, 1, 10)
    xlg, ylg = np.meshgrid(tmp, tmp)
    ax.plot_surface(xlg, ylg, zlr(xlg, ylg),color='yellow')

    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')

    var = logreg.score(X,y)
    ax.set_title("Score: {}".format(str(var)))

    pca = PCA(n_components=3)
    X_r = pca.fit(X).transform(X)
    print(X_r)

    ax2 = fig.add_subplot(231)
    ax3 = fig.add_subplot(232)
    ax4 = fig.add_subplot(233)
    colors = ['red', 'blue']
    for color, i in zip(colors, [min(y), max(y)]):
        ax2.scatter(X_r[y == i, 0], X_r[y == i, 1], color=color)
        ax3.scatter(X_r[y == i, 1], X_r[y == i, 2], color=color)
        ax4.scatter(X_r[y == i, 2], X_r[y == i, 0], color=color)

    if inputcirc!=None:
        ax2 = fig.add_subplot(234)
        col_map, edgelist = json2edgelist(inputcirc)

        G, colors, edges = edgelist2graph(col_map, edgelist)

        pos = nx.spring_layout(G)

        nx.draw(G, with_labels=False, ax=ax2, edgelist=edges, pos=pos, edge_color=colors, node_size=10, linewidth=5.,
                font_size=8,title=title)


    plt.savefig("aggregated.png")
    plt.show()
    return ax

def pca_plotter(input,savepath=""):
    fig = plt.figure()

    std_scaler = StandardScaler()
    std_scaler.fit(input)
    input = std_scaler.transform(input)
    X = np.array(input)[:, :-1]
    y = np.array(input)[:, -1]

    # X = np.array(input)[:, :-1]
    # y = np.array(input)[:, -1]
    features=np.shape(X)[1]

    pca = PCA(n_components=features)
    X_r = pca.fit(X).transform(X)

    # ax2 = fig.add_subplot(231)
    # ax3 = fig.add_subplot(232)
    # ax4 = fig.add_subplot(233)
    # colors = ['red', 'blue']
    # for color, i in zip(colors, [min(y), max(y)]):
    #     ax2.scatter(X_r[y == i, 0], X_r[y == i, 1], color=color)
    #     ax3.scatter(X_r[y == i, 1], X_r[y == i, 2], color=color)
    #     ax4.scatter(X_r[y == i, 2], X_r[y == i, 0], color=color)
    # pca = PCA(n_components=features)
    # X_r = pca.fit(X).transform(X)
    #
    combs=list(combinations(range(features),2))

    #create axes
    axs=[]
    pn=int(np.ceil(np.sqrt(features)))
    for comb,feat in zip(combs,range(features)):
        axs.append(fig.add_subplot(pn,pn,feat+1))

    colors = ['red', 'blue']
    for color, i in zip(colors, [min(y), max(y)]):
        for ax, comb in zip(axs, combs):
            ax.scatter(X_r[y == i, comb[0]], X_r[y == i, comb[1]], color=color)

    if savepath!="":
        plt.savefig(savepath)
    else:
        plt.show()

def plot_json_graph(dictdata,imagepath=""):

    col_map, edgelist = json2edgelist(dictdata)

    G, colors, edges = edgelist2graph(col_map, edgelist)

    pos = nx.spring_layout(G)

    nx.draw(G, with_labels=True,edgelist=edges,pos=pos,edge_color=colors,node_size=10,linewidth=5.,font_size=8)

    if imagepath!="":
        plt.savefig(imagepath)
    else:
        plt.show()


def edgelist2graph(col_map, edgelist):
    G = nx.Graph()
    for e in edgelist:
        G.add_edges_from([e[1:]], color=col_map.get(e[0], 'cyan'))
    edges, colors = zip(*nx.get_edge_attributes(G, 'color').items())
    return G, colors, edges


def json2edgelist(dictdata):
    jsondata = json.loads(dictdata)
    edgelist = []
    for k in jsondata.keys():
        if k != '0':
            edgelist.append(jsondata[k][0:3])
    col_map = {'m': 'blue',  # internal
               'g': 'green',  # output
               'r': 'green',  # output
               'R': 'red'}  # input
    return col_map, edgelist


def main():

    a = [[-0.0034, -0.0001, -0.0001, 0.],
         [-0.0001, -0., -0.0001, 1.],
         [-0.0033, -0.0001, -0.0001, 1.],
         [0., 0., 0.0001, 0.]]

    plot3d(a)

    with open(r'/home/nifrick/PycharmProjects/ResSymphony/results/n100_p0.045_k4_testxor_eqt0_5_date01-14-18-16_03_44_id35.json','r') as f:
        jdat=f.read()
        plot_json_graph(jdat)

if __name__ == "__main__":
    main()

