import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
import scipy.optimize as opt
import pickle
from sklearn.decomposition import PCA

# data = "http://opendata.cern.ch/record/21856#"

project_name = "example_CMS"
data_path_before = "Z_open_data_data_cut.npz"
data_path_after = "Thesis_datasets/HEP_data/HEP2-MSE-4x-decompressed.npz"

def compute_PCA(data):
    data = data.T 
    #setting the algorithm to compress data to 2 dimensions (R=4 -> n_comp = 2) (R=1.6 -> n_comp = 5)
    principal=PCA(n_components=2)

    #training the algorithm on the train data
    principal.fit(data)

    #compressing the test data
    reduced_data = principal.transform(data)
    print("\nCompressed test data shape:\n",reduced_data.shape)

    #reconstructing the test data
    reconstructed_data = principal.inverse_transform(reduced_data)
    print("Reconstructed test data shape:\n",reconstructed_data.shape)
    return reconstructed_data.T

PCA_data = compute_PCA(np.load(data_path_before)["data"])

def analysis(project_name, data_path_before, data_path_after, PCA_data):
    #print(data_path_before, data_path_after)

    before, before_names = np.load(data_path_before)["data"], np.load(data_path_before)["names"]
    after, after_names = np.load(data_path_after)["data"], np.load(data_path_after)["names"]
    #print(before_names,after_names)

    before = pd.DataFrame(before,columns=before_names)
    after = pd.DataFrame(after,columns=after_names)
    PCA_df = pd.DataFrame(PCA_data, columns=after_names)


    # plot_all(project_path, before, after)
    variable = "recoGenJets_slimmedGenJets__PAT.obj.m_state.p4Polar_.fCoordinates.fM"
    plt.hist(before[variable],bins=200)
    plt.hist(after[variable],histtype="step",bins=200)
    plt.hist(PCA_df[variable],histtype="step",bins=200)
    plt.yscale("log")
    #plt.show()
    plot_peak("baler", before[variable], after[variable], PCA_df[variable])


def fit(x, a, b, c, k, m):
    return a * np.exp(-((x - b) ** 2) / (2 * c**2)) + m * x + k


def plot_peak(project_path, before, after, PCA):
    fig, ([ax1, ax3], [ax2, ax4], [ax5, ax6]) = plt.subplots(
        ncols=2, nrows=3, figsize=(12,7), sharex=False
    )

    x_min = min(before + after)
    x_max = max(before + after)
    x_diff = abs(x_max - x_min)

    with PdfPages("peak_plot_combined_4x_redo.pdf") as pdf:

        # Before Histogram
        counts_before, bins_before = np.histogram(
            before, bins=np.linspace(60, 100, 200)
        )
        hist1 = ax1.hist(
            bins_before[:-1],
            bins_before,
            weights=counts_before,
            label="Before",
            histtype="step",
            color="black",
        )
        before_bin_centers = hist1[1][:-1] + (hist1[1][1:] - hist1[1][:-1]) / 2
        before_bin_centers_error = (hist1[1][1:] - hist1[1][:-1]) / 2
        before_bin_counts = hist1[0]
        before_count_error = np.sqrt(hist1[0])
        ax1.errorbar(
            before_bin_centers,
            before_bin_counts,
            yerr=before_count_error,
            xerr=None,
            marker="",
            linewidth=0.75,
            markersize=1,
            linestyle="",
            color="black",
        )
        optimizedParameters1, pcov1 = opt.curve_fit(
            fit, before_bin_centers, before_bin_counts, p0=[1, 80.4, 1, 1, 1], maxfev=50000000
        )
        perr1 = np.sqrt(np.diag(pcov1))
        ax1.plot(
            before_bin_centers,
            fit(before_bin_centers, *optimizedParameters1),
            linewidth=1,
            label="Fit",
            color="red",
        )
        leg1 = ax1.legend(
            borderpad=0.5,
            loc="lower center",
            ncol=2,
            frameon=False,
            facecolor="white",
            framealpha=1,
            fontsize="small",
        )
        leg1._legend_box.align = "left"
        leg1.set_title(
            fr"Mass  : {round(optimizedParameters1[1],2)} $\pm$ {round(perr1[1],2)}"+"\n"
            + fr"Width : {round(optimizedParameters1[2],2)} $\pm$ {round(perr1[2],2)}"
        )
        ax1.set_ylabel("Counts", fontsize=14, ha="right", y=1.0)
        #ax1.set_xlabel("Mass [GeV]", fontsize=14, ha="right", x=1.0)
        #ax1.set_title("Before Compression")
        ax1.set_yscale("log")

        ax1.set_xticklabels([])
        ax1.set_ylim(ymin=1,ymax=250)
        ax1.set_xlim(65,95)
        print(f"Before compression:")
        print(f"Mass  : {round(optimizedParameters1[1],2)} +/- {round(perr1[1],2)}")
        print(f"Width : {round(optimizedParameters1[2],2)} +/- {round(perr1[2],2)}")

        # After Histogram
        counts_after, bins_after = np.histogram(after, bins=np.linspace(60, 100, 200))
        hist2 = ax2.hist(
            bins_after[:-1],
            bins_after,
            weights=counts_after,
            label="After",
            histtype="step",
            color="black",
        )
        after_bin_centers = hist2[1][:-1] + (hist1[1][1:] - hist1[1][:-1]) / 2
        after_bin_centers_error = (hist2[1][1:] - hist1[1][:-1]) / 2
        after_bin_counts = hist2[0]
        after_count_error = np.sqrt(hist2[0])
        ax2.errorbar(
            after_bin_centers,
            after_bin_counts,
            yerr=after_count_error,
            xerr=None,
            marker="",
            linewidth=0.75,
            markersize=1,
            linestyle="",
            color="black",
        )
        optimizedParameters2, pcov2 = opt.curve_fit(
            fit, after_bin_centers, after_bin_counts, p0=[1, 80.4, 1, 1, 1]
        )
        perr2 = np.sqrt(np.diag(pcov2))
        ax2.set_yscale("log")
        ax2.plot(
            after_bin_centers,
            fit(after_bin_centers, *optimizedParameters2),
            linewidth=1,
            label="Fit",
            color="red",
        )
        leg2 = ax2.legend(
            borderpad=0.5,
            loc="lower center",
            ncol=4,
            frameon=False,
            facecolor="white",
            framealpha=1,
            fontsize="small",
            
        )
        leg2._legend_box.align = "left"
        leg2.set_title(
            fr"Mass  : {round(optimizedParameters2[1],2)} $\pm$ {round(perr2[1],2)}"+"\n"
            + fr"Width : {round(optimizedParameters2[2],2)} $\pm$ {round(perr2[2],2)}"
        )
        #ax2.set_ylabel("Counts", fontsize=14, ha="right", y=1.0)
        #ax2.set_xlabel("Mass [GeV]", fontsize=14, ha="right", x=1.0)
        #ax2.set_title("After Decompression")
        ax2.set_xticklabels([])
        ax2.set_ylim(ymin=1,ymax=250)
        ax2.set_xlim(65,95)
        print(f"After compression:")
        print(f"Mass  : {round(optimizedParameters2[1],2)} +/- {round(perr2[1],2)}")
        print(f"Width : {round(optimizedParameters2[2],2)} +/- {round(perr2[2],2)}")

        #############################################################################################
        # Before Histogram
        counts_before, bins_before = np.histogram(
            before, bins=np.linspace(100, 250, 200)
        )
        hist3 = ax3.hist(
            bins_before[:-1],
            bins_before,
            weights=counts_before,
            label="Before",
            histtype="step",
            color="black",
        )
        before_bin_centers = hist3[1][:-1] + (hist3[1][1:] - hist3[1][:-1]) / 2
        before_bin_centers_error = (hist3[1][1:] - hist3[1][:-1]) / 2
        before_bin_counts = hist3[0]
        before_count_error = np.sqrt(hist3[0])
        ax3.errorbar(
            before_bin_centers,
            before_bin_counts,
            yerr=before_count_error,
            xerr=None,
            marker="",
            linewidth=0.75,
            markersize=1,
            linestyle="",
            color="black",
        )
        optimizedParameters3, pcov3 = opt.curve_fit(
            fit, before_bin_centers, before_bin_counts, p0=[1, 173, 1, 1, 1], maxfev=50000000
        )
        perr3 = np.sqrt(np.diag(pcov3))
        ax3.plot(
            before_bin_centers,
            fit(before_bin_centers, *optimizedParameters3),
            linewidth=1,
            label="Fit",
            color="red",
        )
        leg3 = ax3.legend(
            borderpad=0.5,
            loc="best",
            ncol=2,
            frameon=False,
            facecolor="white",
            framealpha=1,
            fontsize="small",
        )
        leg3._legend_box.align = "left"
        leg3.set_title(
            fr"Mass  : {round(optimizedParameters3[1],2)} $\pm$ {round(perr3[1],2)}"+"\n"
            + fr"Width : {round(optimizedParameters3[2],2)} $\pm$ {round(perr3[2],2)}"
        )
        #leg3.set_title(
        #    fr"Mass  : No Peak Found"+"\n"
        #    + fr"Width : No Width Found"
        #)
        #ax3.set_ylabel("Counts", fontsize=14, ha="right", y=1.0)
        #ax3.set_xlabel("Mass [GeV]", fontsize=14, ha="right", x=1.0)
        #ax3.set_title("Before Compression")
        ax3.set_yscale("log")
        ax3.set_xticklabels([])
        ax3.set_yticklabels([])


        ax3.set_ylim(ymin=1,ymax=800)
        ax3.set_xlim(120,225)
        print(f"Before compression:")
        print(f"Mass  : {round(optimizedParameters3[1],2)} +/- {round(perr3[1],2)}")
        print(f"Width : {round(optimizedParameters3[2],2)} +/- {round(perr3[2],2)}")

        # After Histogram
        counts_after, bins_after = np.histogram(after, bins=np.linspace(100, 250, 200))
        hist4 = ax4.hist(
            bins_after[:-1],
            bins_after,
            weights=counts_after,
            label="After",
            histtype="step",
            color="black",
        )
        after_bin_centers = hist4[1][:-1] + (hist4[1][1:] - hist4[1][:-1]) / 2
        after_bin_centers_error = (hist4[1][1:] - hist4[1][:-1]) / 2
        after_bin_counts = hist4[0]
        after_count_error = np.sqrt(hist4[0])
        ax4.errorbar(
            after_bin_centers,
            after_bin_counts,
            yerr=after_count_error,
            xerr=None,
            marker="",
            linewidth=0.75,
            markersize=1,
            linestyle="",
            color="black",
        )
        optimizedParameters4, pcov4 = opt.curve_fit(
            fit, after_bin_centers, after_bin_counts, p0=[1, 165, 1, 1, 1]
        )
        perr4 = np.sqrt(np.diag(pcov4))
        ax4.set_yscale("log")
        ax4.plot(
            after_bin_centers,
            fit(after_bin_centers, *optimizedParameters4),
            linewidth=1,
            label="Fit",
            color="red",
        )
        leg4 = ax4.legend(
            borderpad=0.5,
            loc="best",
            ncol=2,
            frameon=False,
            facecolor="white",
            framealpha=1,
            fontsize="small",
        )
        leg4._legend_box.align = "left"
        leg4.set_title(
            fr"Mass  : {round(optimizedParameters4[1],2)} $\pm$ {round(perr4[1],2)}"+"\n"
            + fr"Width : {round(optimizedParameters4[2],2)} $\pm$ {round(perr4[2],2)}"
        )
        #ax4.set_ylabel("Counts", fontsize=14, ha="right", y=1.0)
        #ax4.set_xlabel("Mass [GeV]", fontsize=14, ha="right", x=1.0)
        #ax4.set_title("After Decompression")
        ax4.set_xticklabels([])

        ax4.set_ylim(ymin=1,ymax=800)
        ax4.set_xlim(120,225)
        ax4.set_yticklabels([])
        print(f"After compression:")
        print(f"Mass  : {round(optimizedParameters4[1],2)} +/- {round(perr4[1],2)}")
        print(f"Width : {round(optimizedParameters4[2],2)} +/- {round(perr4[2],2)}")

        #############################################################################################
        # Before Histogram
        counts_PCA, bins_PCA = np.histogram(
            PCA, bins=np.linspace(60, 100, 200)
        )
        hist5 = ax5.hist(
            bins_PCA[:-1],
            bins_PCA,
            weights=counts_PCA,
            label="PCA",
            histtype="step",
            color="black",
        )
        PCA_bin_centers = hist5[1][:-1] + (hist5[1][1:] - hist5[1][:-1]) / 2
        PCA_bin_centers_error = (hist5[1][1:] - hist5[1][:-1]) / 2
        PCA_bin_counts = hist5[0]
        PCA_count_error = np.sqrt(hist5[0])
        ax5.errorbar(
            PCA_bin_centers,
            PCA_bin_counts,
            yerr=PCA_count_error,
            xerr=None,
            marker="",
            linewidth=0.75,
            markersize=1,
            linestyle="",
            color="black",
        )
        optimizedParameters5, pcov5 = opt.curve_fit(
            fit, PCA_bin_centers_error, PCA_bin_counts, p0=[1, 80.4, 1, 1, 1], maxfev=50000000
        )
        perr5 = np.sqrt(np.diag(pcov5))
        ax5.plot(
            PCA_bin_centers,
            fit(PCA_bin_centers, *optimizedParameters5),
            linewidth=1,
            label="Fit",
            color="red",
        )
        leg5 = ax5.legend(
            borderpad=0.5,
            loc="best",
            ncol=2,
            frameon=False,
            facecolor="white",
            framealpha=1,
            fontsize="small",
        )
        leg5._legend_box.align = "left"
        leg5.set_title(
            fr"Mass  : {round(optimizedParameters5[1],2)} $\pm$ {round(perr5[1],2)}"+"\n"
            + fr"Width : {round(optimizedParameters5[2],2)} $\pm$ {round(perr5[2],2)}"
        )
        leg5.set_title(
            fr"Mass  : No Peak Found"+"\n"
            + fr"Width : No Width Found"
        )
        #ax3.set_ylabel("Counts", fontsize=14, ha="right", y=1.0)
        #ax3.set_xlabel("Mass [GeV]", fontsize=14, ha="right", x=1.0)
        #ax3.set_title("Before Compression")
        ax5.set_yscale("log")
        #ax5.set_xticklabels([])
        #ax5.set_yticklabels([])


        ax5.set_ylim(ymin=1,ymax=250)
        ax5.set_xlim(65,95)
        print(f"Before compression:")
        print(f"Mass  : {round(optimizedParameters5[1],2)} +/- {round(perr5[1],2)}")
        print(f"Width : {round(optimizedParameters5[2],2)} +/- {round(perr5[2],2)}")

        # After Histogram
        counts_PCA, bins_PCA = np.histogram(PCA, bins=np.linspace(100, 250, 200))
        hist6 = ax6.hist(
            bins_PCA[:-1],
            bins_PCA,
            weights=counts_PCA,
            label="PCA",
            histtype="step",
            color="black",
        )
        PCA_bin_centers = hist6[1][:-1] + (hist6[1][1:] - hist6[1][:-1]) / 2
        PCA_bin_centers_error = (hist6[1][1:] - hist6[1][:-1]) / 2
        PCA_bin_counts = hist6[0]
        PCA_count_error = np.sqrt(hist6[0])
        ax6.errorbar(
            PCA_bin_centers,
            PCA_bin_counts,
            yerr=PCA_count_error,
            xerr=None,
            marker="",
            linewidth=0.75,
            markersize=1,
            linestyle="",
            color="black",
        )
        optimizedParameters6, pcov6 = opt.curve_fit(
            fit, PCA_bin_centers, PCA_bin_counts, p0=[1, 173, 1, 1, 1]
        )
        perr6 = np.sqrt(np.diag(pcov6))
        ax6.set_yscale("log")
        ax6.plot(
            PCA_bin_centers,
            fit(PCA_bin_centers, *optimizedParameters6),
            linewidth=1,
            label="Fit",
            color="red",
        )
        leg6 = ax6.legend(
            borderpad=0.5,
            loc="best",
            ncol=2,
            frameon=False,
            facecolor="white",
            framealpha=1,
            fontsize="small",
        )
        leg6._legend_box.align = "left"
        leg6.set_title(
            fr"Mass  : {round(optimizedParameters6[1],2)} $\pm$ {round(perr6[1],2)}"+"\n"
            + fr"Width : {round(optimizedParameters6[2],2)} $\pm$ {round(perr6[2],2)}"
        )
        leg6.set_title(
            fr"Mass  : No Peak Found"+"\n"
            + fr"Width : No Width Found"
        )
        #ax4.set_ylabel("Counts", fontsize=14, ha="right", y=1.0)
        ax6.set_xlabel("Mass [GeV]", fontsize=14, ha="right", x=1.0)
        #ax4.set_title("After Decompression")
        ax6.set_ylim(ymin=1,ymax=800)
        ax6.set_xlim(120,225)
        ax6.set_yticklabels([])
        print(f"After compression:")
        print(f"Mass  : {round(optimizedParameters6[1],2)} +/- {round(perr6[1],2)}")
        print(f"Width : {round(optimizedParameters6[2],2)} +/- {round(perr6[2],2)}")






        plt.subplots_adjust(wspace=0.07,hspace=0.40)
        plt.figtext(0.5,0.92, "Original Distributions", ha="center", va="top", fontsize=14)
        plt.figtext(0.5,0.63, "Reconstructed Distributions (Baler) for R = 4", ha="center", va="top", fontsize=14)
        plt.figtext(0.5,0.35, "Reconstructed Distributions (PCA) for R = 4", ha="center", va="top", fontsize=14)

        pdf.savefig()

analysis(project_name, data_path_before, data_path_after, PCA_data)
