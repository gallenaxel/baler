import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec


#############################################################################################################################
#before_path = "new_cms_data_85.npz"

#after_paths = "paper_runs/paper_ts0_1/decompressed_output/decompressed.npz"

#after_6x = "paper_runs_6x2/paper_6x_ts0_2/decompressed_output/decompressed.npz"
#############################################################################################################################
#before_path = "cut_cms_data.npz"

#after_paths = "thesis_jobs_1/thesis_cms_cut_ts0/decompressed_output/decompressed.npz"

after_6x = "thesis_jobs_1/thesis_cms_cut_ts0/decompressed_output/decompressed.npz"
#############################################################################################################################
before_path = "Thesis_datasets/HEP_data/cut_cms_data.npz"

after_paths = "Thesis_datasets/HEP_data/HEP1-MSE-1x-decompressed.npz"
#after_paths =  "Final_paper_1x/decompressed_ts0_1x_pratik_1.npz"
after_6x = "Thesis_datasets/HEP_data/HEP1-MSE-6x-decompressed.npz"
#############################################################################################################################
z = False

Box_plots = True
Distributions = False
response = False
residual = False
dist_response = False

remove_nans = True
pt_energy_mass_response = False
unit_separation = True
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
unit_list = ["[GeV]", "[arb.]","[rad.]","[GeV]", "[Area]", "[GeV]", "[GeV]", "[GeV]", "[GeV]", "[GeV]" ,"[GeV]", "[GeV]", "[#]", "[#]", "[#]", "[#]", "[#]", "[#]" , "[#]" ,"[GeV]", "[GeV]", "[GeV]", "[#]", "[#]"]
print(len(unit_list))
unit_list_z = ["[GeV]", "[arb.]" ,"[rad.]", "[GeV]", "[GeV]", "[GeV]", "[GeV]", "[GeV]"]
Energy_list_HEP1 = [0,2,5,6,7,8,9,10,11,19,20,21]
Energy_list_HEP2 = [0,3,4,5,6,7]

Rest_list_HEP1 = [1,3,4,12,13,14,15,16,17,18,22,23]
Rest_list_HEP2 = [1,2]
def get_index_to_cut(column_index, cut, array):
    indices_to_cut = np.argwhere(array[column_index] < cut).flatten()
    return indices_to_cut


def cut(before, after1,after2):
    index_to_cut = get_index_to_cut(3, 1e-3, before)
    before = np.delete(before, index_to_cut, axis=1)
    after1 = np.delete(after1, index_to_cut, axis=1)
    after2 = np.delete(after2,index_to_cut,axis=1)
    return before, after1, after2

before, after1, after2 = cut(before,after1,after2)

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



def plot_distributions(names, before, after):
    fig, axs = plt.subplots(figsize=(15,12),sharex=True,sharey=True,layout="constrained")
    fig.suptitle('Variable distributions for R = 1.6', fontsize=18)

    for i, names in enumerate(names):
        split_name = names.split(".")[-1]
        try:
            split_name = split_name.split("f")[-1]
        except:
            pass
        try: 
            split_name = split_name.split("_")[0]
        except:
            pass


        ax = plt.subplot(4,2,i+1) # 6,4

        y,x,_ = ax.hist(before[i],bins=100,label="Before")
        ax.set_yscale("log")

        ax.set_title(split_name +" " +unit_list[i],fontsize=16)
        ax.set_xlabel("")
        ax.set_ylim(ymin=1,ymax=1.4*y.max())
        ax.set_ylabel("Counts",fontsize=15)

        y1,x1,_ = ax.hist(after[i],bins=100,histtype="step",color="#ff7f0e",label="After")
        #plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)

    #plt.legend(loc= "upper center", ncol= 2, bbox_to_anchor=[-1.4,7.8])
    #plt.show()
    plt.savefig("Thesis_z_distributions-16x.pdf")

def plot_box_and_whisker(names, after1, after2,before):
    column_names = [i.split(".")[-1] for i in names]
    try:
        column_names = [i.split("_")[1] for i in column_names]
    except:
        pass

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

    elif unit_separation:
        energy_names = []
        rest_names = []
        for i,j in enumerate(unit_list):
            if j == "[GeV]":
                energy_names.append(column_names[i])
            if j != "[GeV]":
                rest_names.append(column_names[i])
        energy_data1, energy_data2 = (after1.T[:,Energy_list_HEP1]).T , (after2.T[:,Energy_list_HEP1]).T
        rest_data1, rest_data2 = (after1.T[:, Rest_list_HEP1]).T, (after2.T[:, Rest_list_HEP1]).T
        before_energy_data = (before.T[:, Energy_list_HEP1]).T
        before_rest_data = (before.T[:,Rest_list_HEP1]).T
        if z:
            energy_data1, energy_data2 = (after1.T[:,Energy_list_HEP2]).T , (after2.T[:,Rest_list_HEP2]).T
            rest_data1, rest_data2 = (after1.T[:, Rest_list_HEP2]).T, (after2.T[:, Rest_list_HEP2]).T
            before_energy_data = (before.T[:, Energy_list_HEP2]).T
            before_rest_data = (before.T[:,Rest_list_HEP2]).T


        response_values_1  = np.divide(np.subtract(energy_data1, before_energy_data),before_energy_data) 
        response_values_2   = np.divide(np.subtract(energy_data2, before_energy_data),before_energy_data)
        data1, data2 = np.subtract(rest_data1, before_rest_data), np.subtract(rest_data2, before_rest_data)

        #if remove_nans:
            # Replace infs with NaNs
        #    response_values_1[response_values_1 > 1e308] = np.nan
        #    response_values_2[response_values_2 > 1e308] = np.nan

            # Remove NaNs
        #    response_values_1 = response_values_1[~np.isnan(response_values_1)]
        #    response_values_2 = response_values_2[~np.isnan(response_values_2)]



    fig1, (ax1,ax2) = plt.subplots(1,2,sharey=False, figsize=(12,7))

    # First boxplot
    if pt_energy_mass_response:
        boxes = ax1.boxplot(list(response_values_1), vert=False,showfliers=False,showmeans=True,meanline=True)
    elif unit_separation:
        boxes = ax1.boxplot(list(response_values_1), vert=False,showfliers=False,showmeans=True,meanline=True)
    else:
        boxes = ax1.boxplot(list(data1), vert=False,showfliers=False,showmeans=True,meanline=True)#, whis=[0.1,99.9])

    whiskers = np.concatenate([item.get_xdata() for item in boxes["whiskers"]])
    edges = max([abs(min(whiskers)), max(whiskers)])

    ax1.grid()
    ax1.set_xlim(-edges - edges * 0.1, edges + edges * 0.1)
    if pt_energy_mass_response:
        ax1.set_yticks(np.arange(1, len(energy_names) + 1, 1))
        ax1.set_yticklabels(energy_names,fontsize=13)
    elif unit_separation:
        ax1.set_yticks(np.arange(1, len(energy_names) + 1, 1))
        ax1.set_yticklabels(energy_names,fontsize=13)
    else:
        ax1.set_yticks(np.arange(1, len(column_names) + 1, 1))
        ax1.set_yticklabels(column_names,fontsize=13)



    # Second boxplot
    if pt_energy_mass_response:
        boxes = ax2.boxplot(list(response_values_2), vert=False,showfliers=False,showmeans=True,meanline=True)
    elif unit_separation:
        boxes = ax2.boxplot(list(response_values_2), vert=False,showfliers=False,showmeans=True,meanline=True)
    else:
        boxes = ax2.boxplot(list(data2), vert=False,showfliers=False,showmeans=True,meanline=True)#, whis=[0.1,99.9])
    
    whiskers = np.concatenate([item.get_xdata() for item in boxes["whiskers"]])
    edges = max([abs(min(whiskers)), max(whiskers)])

    ax2.grid()
    ax2.set_xlim(-edges - edges * 0.1, edges + edges * 0.1)
    ax2.set_yticks(np.arange(1,len(rest_names)+1,1))
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
        boxes = ax3.boxplot(list(data1),vert=False, showmeans=True,showfliers=False,meanline=True)#, whis=[0.1,99.9])
        #whiskers = np.concatenate([item.get_xdata() for item in boxes["whiskers"]])
        #edges = max([abs(min(whiskers)), max(whiskers)])

        ax3.grid()
        #ax3.set_xlim(-edges - edges * 0.1, edges + edges * 0.1)
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
        ax4.set_yticklabels([])
        ax3.set_title("Compression Ratio = 1.6",fontsize=18)
        if z:
            ax4.set_title("Compression Ratio = 4",fontsize=18)
        else:
            ax4.set_title("Compression Ratio = 6",fontsize=18)

    elif unit_separation:
        ax1.set_yticklabels(energy_names,fontsize=13)
        ax2.set_yticklabels([])

        divider = make_axes_locatable(ax1)
        if z:
            ax3 = divider.append_axes("top", size="100%", pad=0.6) 
        else:  
            ax3 = divider.append_axes("top", size="100%", pad=0.6) # 350% for four-vec separation
        ax1.figure.add_axes(ax3)
        boxes = ax3.boxplot(list(data1),vert=False, showmeans=True,showfliers=False,meanline=True)#, whis=[0.1,99.9])
        whiskers = np.concatenate([item.get_xdata() for item in boxes["whiskers"]])
        edges = max([abs(min(whiskers)), max(whiskers)])

        ax3.grid()
        ax3.set_xlim(-edges - edges * 0.1, edges + edges * 0.1)
        #ax3.set_yticks(np.arange(1, len(rest_names) + 1, 1))
        ax3.set_yticklabels(rest_names,fontsize=13)
        ax3.set_title("Compression Ratio = 1.6",fontsize=18)

        divider = make_axes_locatable(ax2)
        if z:
            ax4 = divider.append_axes("top", size="100%", pad=0.6) # 
        else:
            ax4 = divider.append_axes("top", size="100%", pad=0.6) # 100%
       
        ax2.figure.add_axes(ax4)
        boxes = ax4.boxplot(list(data2), vert=False,showfliers=False,showmeans=True,meanline=True)#, whis=[0.1,99.9])
        whiskers = np.concatenate([item.get_xdata() for item in boxes["whiskers"]])
        edges = max([abs(min(whiskers)), max(whiskers)])

        ax4.grid()
        ax4.set_xlim(-edges - edges * 0.1, edges + edges * 0.1)
        ax4.set_yticklabels([])
        if z:
            ax4.set_title("Compression Ratio = 4",fontsize=18)
        else:
            ax4.set_title("Compression Ratio = 6",fontsize=18)
    else:
        ax1.set_title("Compression Ratio = 1.6",fontsize=18)
        if z:
            ax2.set_title("Compression Ratio = 4",fontsize=18)
        else:
            ax2.set_title("Compression Ratio = 6",fontsize=18)

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
    elif unit_separation:
        ax1.set_xlabel("Response",fontsize=16) # Bottom left
        ax2.set_xlabel("Response",fontsize=16) # Bottom Right 
        ax3.set_xlabel("Residual",fontsize=16) # Top Left
        ax4.set_xlabel("Residual",fontsize=16) # Top Right



    fig1.tight_layout()
    fig1.subplots_adjust(wspace=0.1)
    plt.savefig("boxplot-thesis-HEP1-MSE-1v6x_redo.pdf")


    #plt.show()

def dist_and_response(names, before, after1,unit_list):
    column_names = [i.split(".")[-1] for i in names]
    #print(column_names[0].split("_"))
    column_names = [i.split("_")[0] for i in column_names]


    index_to_cut = get_index_to_cut(3, 1e-3, before)
    before = np.delete(before, index_to_cut, axis=1)

    after1 = np.delete(after1, index_to_cut, axis=1)


    value_to_split = 0 # 0, 4, 8, 12, 16, 20


    if value_to_split:
        before = before[value_to_split:,:]
        after1 = after1[value_to_split:,:]
        column_names = column_names[value_to_split:]
        unit_list = unit_list[value_to_split:]

    RPD = False
    #if RPD:
    #    response1 = 2*np.divide((np.add(before,after1)),(np.add(np.abs(before),np.abs(after1))))
    if value_to_split == 0:
        before[[2,3]] = before[[3,2]]
        after1[[2,3]] = after1[[3,2]]
        a, b = column_names.index('mass'), column_names.index('phi')
        column_names[b], column_names[a] = column_names[a], column_names[b]

    #response1 = np.divide(np.subtract(after1, before), before)
    #response1 = np.subtract(after1, before)
    response1 = np.divide(after1,before)
    fig = plt.figure(figsize=(15,8),constrained_layout=True)

    subfigs = fig.subfigures(2, 2)

    for outerind, subfig in enumerate(subfigs.flat):
        subfig.suptitle(f'{column_names[outerind]} {unit_list[outerind]}',fontsize=18)
        axs = subfig.subplots(1, 2)
        
        for innerind, ax in enumerate(axs.flat):
            if innerind == 0:
                ax.set_title("Variable distribution",fontsize=16)
                y,x,_ = ax.hist(before[outerind],bins=100,label="Before")
                y1,x1,_ = ax.hist(after1[outerind],bins=100,label="After",histtype="step")
                ax.set_yscale("log")
                ax.set_ylim(ymin=1,ymax=1.4*y.max())
                if value_to_split == 0:
                    if outerind == 1 or outerind == 3:
                       ax.set_ylim(1,1e5)
                ax.set_ylabel("Counts",fontsize=16)
                ax.legend()


            if innerind == 1:
                #print(len(response1[outerind]))
                infs_nans = np.count_nonzero(np.isinf(response1[outerind])) + np.count_nonzero(np.isnan(response1[outerind]))

                response_list = list(filter(lambda res: -1e300 <= res <= 1e300, response1[outerind])) # All values inside of this range are used to find the RMS and Mean
                percentage = 100*(len(response1[outerind]) - len(response_list))/len(response1[outerind])
                print(f"Amount of values in the response calculation range: {len(response1[outerind])} or {100 - percentage:2f}% of the original")
                rms = np.sqrt(np.mean(np.square(response_list)))
                ax.set_title("Residual distribution",fontsize=16)
                counts_response, bins_response = np.histogram(
                response1[outerind], bins=np.arange(-2, 2, 0.05)) # -0.2, 0.2, 0.005
                #if value_to_split == 0:
                #    if outerind == 1 or outerind == 3:
                #        counts_response, bins_response = np.histogram(
                #    response1[outerind], bins=np.arange(-0.003, 0.003,0.00005))
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
                if infs_nans == 0:
                        ax.plot([],[], " ", label=f"NaNs & infs\n{infs_nans} = {percentage:0.1f}%")
                else:
                    ax.plot([],[], " ", label=f"NaNs & infs\n{infs_nans} \n= {percentage:0.1f}%")
                ax.set_xlim(-2,2)
                #ax.set_ylim(0,250000)
                #if value_to_split == 0:
                #    if outerind == 0 or outerind == 2:
                #        ax.set_ylim(0,250000)
                #    if outerind == 1 or outerind == 3:
                #        ax.set_xlim(-0.002,0.002)
                ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

                ax.legend()


    #plt.yticks(fontsize=14)
    plt.savefig(f"dist_vs_ratio_{value_to_split}_6x.pdf")
    #plt.show()
        



if dist_response:
    #after_paths = [
    #"Final_paper_1x/decompressed_ts0_1x_pratik_1.npz",
    #"Final_paper_1x/decompressed_ts0_2.npz",
    #"Final_paper_1x/decompressed_ts0_3.npz",
    #"Final_paper_1x/decompressed_ts0_4.npz",
    #"Final_paper_1x/decompressed_ts0_5.npz"]
    #for i in after_paths:
    #    after99 = np.transpose(np.load(i)["data"])
    dist_and_response(names,before,after2,unit_list)

elif Box_plots:
    plot_box_and_whisker(names,after1,after2,before)
elif Distributions:
    plot_distributions(names,before,after2)
    #a = [item for item, count in Counter(after2[14]).items() if count > 1]
    #print(np.sort(a))


