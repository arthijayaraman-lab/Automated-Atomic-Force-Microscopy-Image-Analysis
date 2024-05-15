import numpy as np 
from sklearn.preprocessing import StandardScaler
import porespy as ps
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
    
def processing_normalization(feature_cube_flt):
    """
    Discription: function to normalize feature cube
    Input:
        feature_cube_flt(ndarray) : feature cube flattened 
    return:
        scaled_features(ndarray) : normalized feature cube flattened 
    """
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(feature_cube_flt)
    return scaled_features

def get_transform_list(transform_string):
    transform_list = transform_string.split("_")
    return transform_list
    
def get_diameter_distribution(img, um_pix):
    """
    Function to calculate domain size distribution from index map
    uses porosity simulations on images 
    Input:
        img(ndarray) : index map
    Returns:
        fig(matplotlib object) : domain size distribution 
    #um_pi -> length of one pix in nanometers
    """
    thk = ps.filters.local_thickness(img)
    plt.rcParams.update({'font.size': 14})
    fig, ax = plt.subplots(1, 2, figsize=[10, 4], constrained_layout=True)

    ax[0].tick_params(left = False, right = False , labelleft = False ,
                    labelbottom = False, bottom = False)
    #thk = thk*(2*um_pix)  # gives diameter therefore its 2 * scale of pix to nanometers
    img0 = ax[0].imshow(thk*2*um_pix, cmap='viridis')

    fig.colorbar(img0, ax=ax[0], orientation='vertical')
    #thk = thk//(2*um_pix)
    psd = ps.metrics.pore_size_distribution(im=thk*2*um_pix,log=False,bins=11)
    #*2*um_pix  # gives diameter therefore its 2 * scale of pix to nanometers
    ax[1].plot(psd.bin_centers[:-1], -1*np.diff(psd.cdf), color="black")  # the results from the psd.pdf is very flaky. it gives different scales for different bin sizes and values check this out when free
    ax[1].tick_params('x', top=True)
    ax[1].tick_params('y', right=True)
    ax[1].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    return fig

########## stats for feature extraction and dimension reduction ##########
def get_stats_feature_vector(trans_out, stats_to_use):
    """
    function to multiplex statistics to apply on domain transform outputs
    Input:
        trans_out(ndarray) : domain transform output
    Output:
        feature_vector(ndarray) : feature vector 
    """
    trans_out = np.abs(trans_out)
    trans_out = trans_out.flatten()
    feature_vector = []

    if "a" in stats_to_use:
        mean_trans_out = np.mean(trans_out)
        feature_vector.append(mean_trans_out)

    if "m" in stats_to_use:
        max_trans_out = np.max(trans_out)
        feature_vector.append(max_trans_out)

    if "v" in stats_to_use:
        variance_trans_out = np.var(trans_out)
        feature_vector.append(variance_trans_out)

    if "s" in stats_to_use:
        mean_trans_out = np.mean(trans_out)
        variance_trans_out = np.var(trans_out)
        skewness_trans_out = np.mean((trans_out - mean_trans_out) ** 3) / (variance_trans_out ** (3/2))
        feature_vector.append(skewness_trans_out)

    if "k" in stats_to_use:
        mean_trans_out = np.mean(trans_out)
        variance_trans_out = np.var(trans_out)
        kurtosis_trans_out = np.mean((trans_out - mean_trans_out) ** 4) / (variance_trans_out ** 2)
        feature_vector.append(kurtosis_trans_out)

    return feature_vector
