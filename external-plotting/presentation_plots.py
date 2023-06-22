import numpy as np
#before_path = "new_cms_data_85.npz"
import matplotlib.pyplot as plt

#after_paths =  "Final_paper_1x/decompressed_ts0_1x_pratik_1.npz"
#before_path = "Thesis_datasets/HEP_data/cut_cms_data.npz"
before_path = "Thesis_datasets/HEP_data/Z_open_data_data_cut.npz"
after_paths = "Thesis_datasets/HEP_data/HEP2-MSE-1x-decompressed.npz"
after_paths2 = "Thesis_datasets/HEP_data/HEP2-MSE-4x-decompressed.npz"

before = np.transpose(np.load(before_path)["data"])
names = np.load(before_path)["names"]
after1 = np.transpose(np.load(after_paths)["data"])
after2 = np.transpose(np.load(after_paths2)["data"])


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
unit_list = ["[GeV]", "[arb.]" ,"[GeV]", "[rad.]", "[Area]", "[GeV]", "[GeV]", "[GeV]", "[GeV]", "[GeV]" ,"[GeV]", "[GeV]", "[#]", "[#]", "[#]", "[#]", "[#]", "[#]" , "[#]" ,"[GeV]", "[GeV]", "[GeV]", "[#]", "[#]"]
unit_list_z = ["[GeV]", "[arb.]" ,"[rad.]", "[GeV]", "[GeV]", "[GeV]", "[GeV]", "[GeV]"]

def get_index_to_cut(column_index, cut, array):
    indices_to_cut = np.argwhere(array[column_index] < cut).flatten()
    return indices_to_cut


def compute_response_residual(before, after):
    index_to_cut = get_index_to_cut(3, 1e-6, before)
    before = np.delete(before, index_to_cut, axis=1)
    after = np.delete(after, index_to_cut, axis=1)

try:
    print("Converting floats to ints")

    #data1, data2 = np.transpose(data1), np.transpose(data2)
    for index, column in enumerate(after1):
        if type_list[index] == "int":
            after1[index] = (np.rint(after1[index])).astype("int")
            before[index] = (np.rint(before[index])).astype("int")

    #data1, data2 = np.transpose(data1), np.transpose(data2)
except AttributeError:
    pass

def dist_and_response(names, before, after1, after2, unit_list):
    column_names = [i.split(".")[-1] for i in names]
    #print(column_names[0].split("_"))
    column_names = [i.split("_")[0] for i in column_names]

    index_to_cut = get_index_to_cut(3, 1e-3, before)
    before = np.delete(before, index_to_cut, axis=1)

    after1 = np.delete(after1, index_to_cut, axis=1)
    after2 = np.delete(after2, index_to_cut, axis=1)

    response = np.divide(np.subtract(after1, before),before)
    response2 = np.divide(np.subtract(after2, before),before)

    residual = np.subtract(after1, before)
    residual2 = np.subtract(after2,before)
    for index, variable in enumerate(column_names):
        fig, axs = plt.subplots(2, 2,figsize=(12,7))
        index = index + 3

        infs_nans = np.count_nonzero(np.isinf(response[index])) + np.count_nonzero(np.isnan(response[index]))
        #print(len(residual[index]))
        response_list = list(filter(lambda res: -1e300 <= res <= 1e300, response[index])) # All values inside of this range are used to find the RMS and Mean

        infs_nans2 = np.count_nonzero(np.isinf(response2[index])) + np.count_nonzero(np.isnan(response2[index]))
        #print(len(residual[index]))
        response_list2 = list(filter(lambda res: -1e300 <= res <= 1e300, response2[index])) # All values inside of this range are used to find the RMS and Mean


        # DISTRIBUTIONS
        hist_before = axs[0,0].hist(before[index],bins=100, label="Original")
        hist_after = axs[0,0].hist(after1[index],bins=100,label="Reconstructed 1.6x",histtype="step",color="Red") # alpha for opacity
        hist_after2 = axs[0,0].hist(after2[index],bins=100,label="Reconstructed 4x",histtype="step",color="Orange") # alpha for opacity

        axs[0,0].set_yscale("log")
        axs[0,0].set_ylabel("Counts",rotation=90,fontsize=14)
        axs[0,0].set_title('Variable Distributions',fontsize=14)
        #axs[0,0].set_ylim(ymin=1,ymax=23000)
        axs[0,0].set_xlabel(r"mass [GeV]",fontsize=14)
        #axs[0,0].set_xticks([])
        a = axs[0,0].get_xticks().tolist()
        a = [int(i) for i in a]
        a = a[1:]
        print((a))

        #print(a)
        axs[0,0].set_xticks([])


        axs[0,0].legend(fontsize="14",loc="best")

        # RATIO
        axs[1, 0].plot(hist_before[0]/hist_after[0], marker=".",linestyle=" ",label="1.6x")
        axs[1, 0].plot(hist_before[0]/hist_after2[0], marker="x",linestyle=" ",label="4x")

        axs[1, 0].axhline(y=1, linewidth=0.2, color="black")
        axs[1,0].set_ylim(-0.5,2.5)        
        axs[1, 0].set_ylabel("Ratio",rotation=90,fontsize=14)
        axs[1,0].legend()
        axs[1,0].set_xticks(ticks=[0,14,28,42,56,70,84,100],labels=a) #phi & eta [0,20,35,50,65,80,95] # pt Mass[0,12.5,25,37.5,50,62.5,75,87.5,100] [0,14,28,42,56,70,84,100]

        # RESPONSE
        axs[0, 1].hist(response_list, bins=50000,label="1.6x") #Phi: 50000 Eta: 500000 Mass: 1000 pt: 100
        axs[0, 1].hist(response_list2, bins=500000, histtype="step",label="4x")
        axs[0, 1].set_xlim(-3,3)
        axs[0, 1].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        #axs[0, 1].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        axs[0, 1].set_ylabel("Counts", rotation=90, fontsize=14)

        axs[0, 1].axvline(np.mean(response_list),color="k",linestyle="dashed",linewidth=1,label=f"Mean:\n{np.mean(response_list):0.4e}",)
        axs[0, 1].plot([],[], " ", label=f"NaNs & infs\n{infs_nans} = {100*(infs_nans/(len(residual[index]))):0.1f}%")

        axs[0, 1].axvline(np.mean(response_list2),color="r",linestyle="dashed",linewidth=1,label=f"Mean:\n{np.mean(response_list2):0.4e}",)
        axs[0, 1].plot([],[], " ", label=f"NaNs & infs\n{infs_nans} = {100*(infs_nans2/(len(residual2[index]))):0.1f}%")


        axs[0, 1].legend(fontsize="10")
        axs[0, 1].set_title('Relative Difference',fontsize=14)

        # RESIDUAL
        axs[1, 1].hist(residual[index], bins=5000,label="1.6x") # Phi: 100, Eta: 500
        axs[1, 1].hist(residual2[index], bins=5000,label="4x",histtype="step") # Phi: 100, Eta: 500

        axs[1, 1].set_xlim(-10,10)
        axs[1, 1].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        #axs[1, 1].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        axs[1, 1].set_ylabel("Counts", rotation=90, fontsize=14)
        axs[1, 1].axvline(np.mean(residual[index]),color="k",linestyle="dashed",linewidth=1,label=f"Mean:\n{np.mean(residual[index]):0.4e}",)
        axs[1, 1].axvline(np.mean(residual2[index]),color="r",linestyle="dashed",linewidth=1,label=f"Mean:\n{np.mean(residual2[index]):0.4e}",)

        axs[1, 1].legend(fontsize="10")
        axs[1, 1].set_title('Difference',fontsize=14)




        plt.tight_layout
        plt.savefig(f'MSC_Presentation_plot_mass_HEP2_1x.pdf')
        if index == 3:
            print("Done!")
            break






    exit()
    subfigs = fig.subfigures(2, 2)

dist_and_response(names, before, after1,after2, unit_list)