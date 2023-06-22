import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.decomposition import PCA
import matplotlib.lines as mlines

#############################################################################################################################
before_path = "Thesis_datasets/HEP_data/Z_open_data_data_cut.npz"
#before_path = "Thesis_datasets/HEP_data/cut_cms_data.npz"


after_paths = "Thesis_datasets/HEP_data/HEP2-MSE-4x-decompressed.npz"

after_6x = "Thesis_datasets/HEP_data/HEP2-EMD-4x-decompressed.npz"
#############################################################################################################################
z = True

PCA_dim = 2


Box_plots = False
Distributions = True
response = False
residual = False

remove_nans = False
pt_energy_mass_response = True
#############################################################################################################################
before = np.transpose(np.load(before_path)["data"])
names = np.load(before_path)["names"]
after1 = np.transpose(np.load(after_paths)["data"])
after2 = np.transpose(np.load(after_6x)["data"])

type_list = [
    "float64",
    "float64",
    "float64",
    "float64",
    "float64",
    "float64",
    "float64",
    "float64",
    "float64",
    "float64",
    "float64",
    "float64",
    "int",
    "int",
    "int",
    "int",
    "int",
    "int",
    "int",
    "float64",
    "float64",
    "float64",
    "int",
    "int",
]
unit_list = ["[GeV]", "[arb.]" ,"[rad.]", "[GeV]", "[GeV]", "[GeV]", "[GeV]", "[GeV]", "[GeV]", "[GeV]" ,"[GeV]", "[GeV]", "[#]", "[#]", "[#]", "[#]", "[#]", "[#]" , "[#]" ,"[GeV]", "[GeV]", "[GeV]", "[#]", "[#]"]

try:
    print("Converting floats to ints")

    #data1, data2 = np.transpose(data1), np.transpose(data2)
    for index, column in enumerate(after1):
        if type_list[index] == "int":
            after1[index] = (np.rint(after1[index])).astype("int")
            after2[index] = np.rint(after2[index])
            before[index] = (np.rint(before[index])).astype("int")

    #data1, data2 = np.transpose(data1), np.transpose(data2)
except AttributeError:
    pass
column_names = [i.split(".")[-1] for i in names]
try:
    column_names = [i.split("_")[1] for i in column_names]
except:
    pass
try:
    column_names = [i.split("f")[-1] for i in column_names]
except:
    pass
def plot_distributions(names, before, after, after2, after3):
    fig, ax = plt.subplots(figsize=(15,12),sharex=True,sharey=True,layout="constrained") #15,12 || 12,5
    fig.suptitle('Variable distributions for PCA vs Baler at R = 4', fontsize=18, x=0.3) #x = 0.3

    for i, name in enumerate(names):
        if z:
            ax = plt.subplot(4,2,i+1)
        else:
            ax = plt.subplot(2,2,i+1)

        y,x,_ = ax.hist(before[i],bins=100,label="Before")
        ax.set_yscale("log")

        ax.set_title(name +" " +unit_list[i],fontsize=16)
        ax.set_xlabel("")
        ax.set_ylim(ymin=1,ymax=2*y.max())
        ax.set_ylabel("Counts",fontsize=15)

        y1,x1,_ = ax.hist(after[i],bins=100,histtype="step",label="After (PCA)")
        y2,x2,_ = ax.hist(after2[i],bins=100,histtype="step",label=r"After (Baler w. $\mathcal{L}_1$)")
        y3,x3,_ = ax.hist(after3[i],bins=100,histtype="step",label=r"After (Baler w. $\mathcal{L}_2$)")

        #if i == 0:
        #    ax.legend(fontsize="12")

        #ax.legend(loc="best")

        #plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        #if i == 3: break
    #fig.subplots_adjust(bottom=0.03, wspace=0.33)
    blue_line = mlines.Line2D([], [], color='#1f77b4')
    reds_line = mlines.Line2D([], [], color='#2ca02c')
    orange_line = mlines.Line2D([], [], color='#ff7f0e')
    green_line = mlines.Line2D([], [], color='#d62728')

    fig.legend(handles = [blue_line,orange_line,reds_line,green_line] , labels=['Before', 'After (PCA)', r"After (Baler w. $\mathcal{L}_1$)",r"After (Baler w. $\mathcal{L}_2$)"],
               bbox_to_anchor=(0.75, 0.97),loc="lower center",fancybox=False, shadow=False, ncol=4)


    #plt.legend(ncol= 4)#, bbox_to_anchor=[-0.5,-3])
    #plt.show()
    plt.savefig("PCAvsBaler_beforevsafter_4x.pdf")

def plot_box_and_whisker(names, after1, after2, before):
    if response: 
        response1 = np.divide(np.subtract(after1, before), before)
        response2 = np.divide(np.subtract(after2, before), before)
        data1, data2 = response1, response2
        if remove_nans:
            # Replace infs with NaNs
            data1[data1 > 1e308] = np.nan
            data2[data2 > 1e308] = np.nan

            # Remove NaNs
            data1 = data1[~np.isnan(data1)]
            data2 = data2[~np.isnan(data2)]

    elif residual:    
        residual1 = np.subtract(after1, before)
        residual2 = np.subtract(after2, before)
        data1, data2 = residual1, residual2

    elif pt_energy_mass_response:
        energy_names = [column_names[0],column_names[1],column_names[2],column_names[3]]
        rest_names = list((Counter(column_names) - Counter(energy_names)).elements())

        energy_data1, energy_data2 = (after1.T[:, [0,1,2,3]]).T, (after2.T[:, [0,1,2,3]]).T
        if z:
            rest_data1, rest_data2 = (after1.T[:, [4,5,6,7]]).T, (after2.T[:, [4,5,6,7]]).T
        else:
            rest_data1, rest_data2 = (after1.T[:, [4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]]).T, (after2.T[:, [4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]]).T

        before_energy_data = (before.T[:, [0,1,2,3]]).T
        if z:
            before_rest_data =  (before.T[:, [4,5,6,7]]).T        
        else:
            before_rest_data =  (before.T[:, [4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]]).T

        response_values_1, response_values_2 = np.divide(np.subtract(energy_data1, before_energy_data),before_energy_data), np.divide(np.subtract(energy_data2, before_energy_data),before_energy_data)





        data1, data2 = np.subtract(rest_data1, before_rest_data), np.subtract(rest_data2, before_rest_data)

    fig1, (ax1,ax2) = plt.subplots(1,2,sharey=False, figsize=(12,7))

    # First boxplot
    if pt_energy_mass_response:
        boxes = ax1.boxplot(list(response_values_1), vert=False,showfliers=False,showmeans=True,meanline=True)
    else:
        boxes = ax1.boxplot(list(data1), vert=False,showfliers=False,showmeans=True,meanline=True)#, whis=[0.1,99.9])

    whiskers = np.concatenate([item.get_xdata() for item in boxes["whiskers"]])
    edges = max([abs(min(whiskers)), max(whiskers)])

    ax1.grid()
    ax1.set_xlim(-edges - edges * 0.1, edges + edges * 0.1)
    ax1.set_xlim(-4,4)
    if pt_energy_mass_response:
        ax1.set_yticks(np.arange(1, len(energy_names) + 1, 1))
        ax1.set_yticklabels(energy_names,fontsize=13)
    #else:
    #    ax1.set_yticks(np.arange(1, len(column_names) + 1, 1))
    #    ax1.set_yticklabels(column_names,fontsize=13)



    # Second boxplot
    if pt_energy_mass_response:
        boxes = ax2.boxplot(list(response_values_2), vert=False,showfliers=False,showmeans=True,meanline=True)
    else:
        boxes = ax2.boxplot(list(data2), vert=False,showfliers=False,showmeans=True,meanline=True)#, whis=[0.1,99.9])
    
    whiskers = np.concatenate([item.get_xdata() for item in boxes["whiskers"]])
    edges = max([abs(min(whiskers)), max(whiskers)])

    ax2.grid()
    ax2.set_xlim(-edges - edges * 0.1, edges + edges * 0.1)
    ax2.set_xlim(-4,4)

    ax2.set_yticklabels([])    

    if pt_energy_mass_response:

        ax1.set_yticklabels(energy_names,fontsize=13)
        ax2.set_yticklabels([])

        divider = make_axes_locatable(ax1)
        if z:
            ax3 = divider.append_axes("top", size="100%", pad=0.6) 
        else:  
            ax3 = divider.append_axes("top", size="350%", pad=0.6)
        ax1.figure.add_axes(ax3)
        boxes = ax3.boxplot(list(data1), vert=False,showfliers=False,showmeans=True,meanline=True)#, whis=[0.1,99.9])
        whiskers = np.concatenate([item.get_xdata() for item in boxes["whiskers"]])
        edges = max([abs(min(whiskers)), max(whiskers)])

        ax3.grid()
        ax3.set_xlim(-edges - edges * 0.1, edges + edges * 0.1)
        ax3.set_xlim(-150,150)
        ax3.set_yticks(np.arange(1, len(rest_names) + 1, 1))
        ax3.set_yticklabels(rest_names,fontsize=13)

        divider = make_axes_locatable(ax2)
        if z:
            ax4 = divider.append_axes("top", size="100%", pad=0.6) # 
        else:
            ax4 = divider.append_axes("top", size="350%", pad=0.6) # 100%
        ax2.figure.add_axes(ax4)
        boxes = ax4.boxplot(list(data2), vert=False,showfliers=False,showmeans=True,meanline=True)#, whis=[0.1,99.9])
        whiskers = np.concatenate([item.get_xdata() for item in boxes["whiskers"]])
        edges = max([abs(min(whiskers)), max(whiskers)])

        ax4.grid()
        ax4.set_xlim(-edges - edges * 0.1, edges + edges * 0.1)
        ax4.set_xlim(-150,150)
        ax4.set_yticklabels([])
        if z:
            ax3.set_title("Baler at R = 4",fontsize=18)
        else:
            ax3.set_title("Baler at R = 6",fontsize=18)

        if z:
            ax4.set_title("PCA at R = 4",fontsize=18)
        else:
            ax4.set_title("PCA at R = 6",fontsize=18)
        
        #ax1.set_title("Compression Ratio = 1.6",fontsize=18)
       

    if response:
        ax1.set_xlabel("Response",fontsize=16)
        ax2.set_xlabel("Response",fontsize=16)
    elif residual: 
        ax1.set_xlabel("Residual",fontsize=16)
        ax2.set_xlabel("Residual",fontsize=16)
    elif pt_energy_mass_response:
        ax1.set_xlabel("Response",fontsize=16)
        ax2.set_xlabel("Response",fontsize=16)
        ax3.set_xlabel("Residual",fontsize=16)
        ax4.set_xlabel("Residual",fontsize=16)




    fig1.tight_layout()
    fig1.subplots_adjust(wspace=0.1)
    plt.savefig("boxplot-comparison_PCAvBal_4x_samex.pdf")


    #plt.show()


def compute_PCA(data):
    data = data.T 
    #setting the algorithm to compress data to 15 dimensions
    principal=PCA(n_components=PCA_dim)

    #training the algorithm on the train data
    principal.fit(data)

    #compressing the test data
    reduced_data = principal.transform(data)
    print("\nCompressed test data shape:\n",reduced_data.shape)

    #reconstructing the test data
    reconstructed_data = principal.inverse_transform(reduced_data)
    print("Reconstructed test data shape:\n",reconstructed_data.shape)
    return reconstructed_data.T

PCA_data = compute_PCA(before)

try:
    print("Converting floats to ints")

    #data1, data2 = np.transpose(data1), np.transpose(data2)
    for index, column in enumerate(PCA_data):
        if type_list[index] == "int":
            PCA_data[index] = (np.rint(PCA_data[index])).astype("int")
    #data1, data2 = np.transpose(data1), np.transpose(data2)
except AttributeError:
    pass

plot_distributions(column_names, before, PCA_data,after1, after2)

#plot_box_and_whisker(column_names, after1, PCA_data, before)
