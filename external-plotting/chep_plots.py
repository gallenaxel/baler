import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

original = "CHEP_data/new_cms_data_85.npz"
after_1x = "CHEP_data/decompressed_1x.npz"
after_6x = "CHEP_data/decompressed_6x.npz"

#############################################################################################################################

response = True
residual = False
dist_response = True
remove_nans = True

#############################################################################################################################
before = np.transpose(np.load(original)["data"])
names = np.load(original)["names"]
after1x = np.transpose(np.load(after_1x)["data"])
after6x = np.transpose(np.load(after_6x)["data"])

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
unit_list_chep = ["[GeV]","[arb.]","[GeV]","[GeV]"]
name_list_chep = ["pt","eta","mass","Neutral Hadron Energy"]

def get_index_to_cut(column_index, cut, array):
    indices_to_cut = np.argwhere(array[column_index] < cut).flatten()
    return indices_to_cut


def cut(before, after1x,after6x):
    index_to_cut = get_index_to_cut(3, 1e-3, before)
    before = np.delete(before, index_to_cut, axis=1)
    after1x = np.delete(after1x, index_to_cut, axis=1)
    after6x = np.delete(after6x,index_to_cut,axis=1)
    return before, after1x, after6x

before, after1x, after6x = cut(before,after1x,after6x)

try:
    print("Converting floats to ints")
    for index, column in enumerate(after1x):
        if type_list[index] == "int":
            after1x[index] = (np.rint(after1x[index])).astype("int")
            after6x[index] = np.rint(after6x[index])
            before[index] = (np.rint(before[index])).astype("int")

except AttributeError:
    pass

def dist_and_response(names, before, after1x,unit_list):
    column_names = [i.split(".")[-1] for i in names]
    column_names = [i.split("_")[0] for i in column_names]


    index_to_cut = get_index_to_cut(3, 1e-3, before)
    before = np.delete(before, index_to_cut, axis=1)

    after1x = np.delete(after1x, index_to_cut, axis=1)

    if response:
        response1 = np.divide(np.subtract(after1x, before), before)
    else:
        print("Plotting residual")
        response1 = np.subtract(after1x, before)

    ## SELECTING THE VARIABLES CHOSEN FOR CHEP ##
    before = np.array([before[0],before[1],before[3],before[6]])
    after1x = np.array([after1x[0],after1x[1],after1x[3],after1x[6]])
    response1 = np.array([response1[0],response1[1],response1[3],response1[6]])
    ##############################################

    fig = plt.figure(figsize=(15,8),constrained_layout=True)
    subfigs = fig.subfigures(2, 2)

    for outerind, subfig in enumerate(subfigs.flat):
        subfig.suptitle(f'{name_list_chep[outerind]} {unit_list_chep[outerind]}',fontsize=18)
        axs = subfig.subplots(1, 2)
        
        for innerind, ax in enumerate(axs.flat):
            if innerind == 0:
                ax.set_title("Variable distribution",fontsize=16)
                y,x,_ = ax.hist(before[outerind],bins=100,label="Before")
                y1,x1,_ = ax.hist(after1x[outerind],bins=100,label="After",histtype="step")
                ax.set_yscale("log")
                ax.set_ylim(ymin=1,ymax=1.8*y.max())
                if outerind == 1:
                    ax.set_ylim(1,1e5)
                if outerind == 3:
                    ax.set_ylim(1,1e6)
                ax.set_ylabel("Counts",fontsize=16)
                ax.legend()



            if innerind == 1:
                infs_nans = np.count_nonzero(np.isinf(response1[outerind])) + np.count_nonzero(np.isnan(response1[outerind]))
                response_list = list(filter(lambda res: -1e300 <= res <= 1e300, response1[outerind])) # All values inside of this range are used to find the RMS and Mean
                percentage = 100*(len(response1[outerind]) - len(response_list))/len(response1[outerind])
                print(f"Amount of values in the response calculation range: {len(response1[outerind])} or {100 - percentage:2f}% of the original")
                rms = np.sqrt(np.mean(np.square(response_list)))
                if response:
                    ax.set_title("Response distribution",fontsize=16)
                elif residual:
                    ax.set_title("Residual distribution",fontsize=16)
                    
                counts_response, bins_response = np.histogram(
                response1[outerind], bins=np.arange(-0.6, 0.6, 0.005))
                if outerind == 1:
                    counts_response, bins_response = np.histogram(response1[outerind], bins=np.arange(-0.003, 0.003,0.00005))
                ax.hist(
                    bins_response[:-1],
                    bins_response,
                    weights=counts_response,
                    label="Response",
                )
                ax.axvline(
                    np.mean(response_list),
                    color="k",
                    linestyle="dashed",
                    linewidth=1,
                    label=f"Mean:\n{np.mean(response_list):0.4e}",
                )
                ax.plot([], [], " ", label=f"RMS:\n{rms:0.4e}")
                if outerind == 0 or outerind == 2:
                    ax.set_ylim(0,250000)
                    ax.set_xlim(-0.2,0.2)
                if outerind == 1:
                    ax.set_xlim(-0.002,0.002)
                ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

                ax.legend()

    plt.savefig(f"chep_plot.pdf")
dist_and_response(names,before,after1x,unit_list_chep)


