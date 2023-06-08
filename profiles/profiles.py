# Libraries
import os
from pandas import read_csv
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import re
from scipy import stats

pd.options.mode.chained_assignment = None


# profile class
class Profile:
    # TODO: change data slots to numpy arrays, pos, pos_zero, and signal
    # TODO: dataframe should have multi-index with geno and stain
    """A single profile constructed from metadata and position and signal arrays"""
    def __init__(self,
                 name=None,
                 label=None,
                 animal=None,
                 geno=None,
                 image=None,
                 line=None,
                 stain=None,
                 region=None,
                 zero=None,
                 data=None,
                 # signal_raw=None
                 ):
        self.name = name
        self.label = label  # will be the label string from imagej
        self.animal = animal
        self.geno = geno
        self.image = image  # image that the profile was acquired from
        self.line = line
        self.stain = stain  # protein that was imaged
        self.region = region
        self.zero = zero  # position
        self.data = data
        # self.data = pd.DataFrame(columns=['pos_raw', 'pos',  'signal', 'signal_norm'])


# dataset class

class ProfileDataset:
    """A collection of profiles and methods to output data in useful formats"""
    # TODO: make method to output dataframe without positions containing NaNs
    # TODO: make data have a multi index -- NEED TO TEST
    # TODO: use numpy arrays where possible
    # TODO: concat all profile data as tuples/lists only append df after collection of all
    # TODO: add a slot for cutoff positions (where to crop data)
    # TODO: add cutoff flag for plot functions
    def __init__(self, name=None):
        self.name = name
        self.profiles = []  # list of profiles
        self.subregions = {}
        self.meta = []
        self.data = []
        self.summary = []
        self.cutoff = []
        self.scale = None  # micrometers per pixel

    def append(self, profile):
        """method to a append profile to the set"""
        self.profiles.append(profile)

    def long(self):
        """method to compile all profiles and metadata into a single dataframe.
        format suitable for plotting in seaborn"""
        df = pd.DataFrame(columns=['pos_raw',
                                   'pos',
                                   'signal',
                                   'geno',
                                   'image',
                                   'name',
                                   'stain',
                                   'region'])
        for profile in self.profiles:
            profile_df = profile.data
            profile_df['name'] = profile.name
            profile_df['label'] = profile.label
            profile_df['animal'] = profile.animal
            profile_df['geno'] = profile.geno
            profile_df['region'] = profile.region
            profile_df['image'] = profile.image
            profile_df['stain'] = profile.stain
            df = df.append(profile_df, ignore_index=True)
            # df = df.set_index(['stain', 'geno'])
            # TODO: sort the indexes to improve performance
        # print(df)
        return df

    def create_meta(self):
        meta_list = []
        for profile in self.profiles:
            meta_list.append([profile.name, profile.image, profile.animal, profile.geno, profile.stain, profile.region])
        meta = pd.DataFrame(meta_list, columns=['name', 'image', 'animal', 'geno', 'stain', 'region'])
        self.meta = meta

    def wide(self):
        """method to output profile data in wide format suitable for heatmap"""
        pass

    # make a method to call individual profiles
    # method to output whole dataset as wide or long
    # method to filter data
    # method to output dataset characteristics number of animals, images, etc...


def sample_ij(input, metadata, animal_meta, image_file, sample_number):
    """parse imagej generated spreadsheets into a list of profile objects"""
    dataset = ProfileDataset()

    def get_image_info(image_file_name):
        """extract image metadata"""

        regex = '(C\d)-((\S{4,6})_(\S{2,5})_(\S{2,5})_(\S{2,5})_(\S{2,7})_\S{4}.oif)'
        regex_match = re.search(regex, image_file_name)
        return regex_match

    for label in metadata.sample(n=sample_number)['Label']:
        data = pd.DataFrame(columns=['pos_raw', 'signal'], index=input.index)
        row_index = metadata.loc[metadata['Label'] == label].index.tolist()[0]
        # print(label)
        # print(row_index)
        zero = round(metadata.iloc[row_index]['Length'], 0)
        # print(zero)
        match = get_image_info(image_file)
        animal = match[3]
        # print('animal = ', match[3])
        image = match[2]
        # print('image = ', match[0])
        geno = animal_meta[animal_meta['animal'] == animal]['geno'].values[0]
        if match[1] == "C1":
            stain = match[4]
        elif match[1] == "C2":
            stain = match[5]
        elif match[1] == "C3":
            stain = match[6]
        region = match[7]
        # print(type(input.iloc[:, row_index:row_index + 2]))
        # data.index = input.index
        data['pos_raw'] = input.iloc[:, row_index * 2]
        data['signal'] = input.iloc[:, row_index * 2 + 1]
        data['pos'] = data['pos_raw'] - zero
        dataset.append(Profile(label=label,
                               animal=animal,
                               geno=geno,
                               image=image,
                               line=None,
                               stain=stain,
                               region=region,
                               zero=zero,
                               data=data))
    return dataset


def get_file_info(r, file):
    data_file = os.path.join(r, file)
    data = read_csv(data_file)
    image_file = file.split("_profiles.csv")[0]
    meta_file = r + image_file[3:].split(".oif")[0] + "_meta.csv"
    meta = read_csv(meta_file)
    return data, meta, image_file


def sample_data_folder(path, animal_meta, sample_number):
    # TODO: make this function more efficient. concat tuples not dfs
    """This function will concatenate a sample of the data from a single folder into a single dataset object"""
    dataset = ProfileDataset()
    for r, d, f in os.walk(path):
        for file in f:
            if '_profiles.csv' in file:
                # print(r, d, file)
                data, meta, image_file = get_file_info(r, file)
                image_profiles = sample_ij(data, meta, animal_meta, image_file, sample_number)
                for profile in image_profiles.profiles:
                    dataset.append(profile)
    # TODO: add output messages stating the number of images and profiles
    return dataset


# function to a series of profiles from a single image
def parse_ij(input, metadata, animal_meta, image_file):
    """parse imagej generated spreadsheets into a list of profile objects"""
    dataset = ProfileDataset()

    def get_image_info(image_file):
        """extract image metadata"""

        regex = "(C\d)-((\S{4,6})_(\S{2,5})_(\S{2,5})_(\S{2,5})_(\S{2,7})_\S{4}.oif)"
        regex_match = re.search(regex, image_file)
        return regex_match

    for label in metadata['Label']:
        data = pd.DataFrame(columns=['pos_raw',  'signal'], index=input.index)
        row_index = metadata.loc[metadata['Label'] == label].index.tolist()[0]
        # print(label)
        # print(row_index)
        zero = round(metadata.iloc[row_index]['Length'], 0)
        # print(zero)
        match = get_image_info(image_file)
        animal = match[3]
        # print('animal = ', match[3])
        image = match[2]
        # print('image = ', match[0])
        geno = animal_meta[animal_meta['animal'] == animal]['geno'].values[0]
        if match[1] == "C1":
            stain = match[4]
        elif match[1] == "C2":
            stain = match[5]
        elif match[1] == "C3":
            stain = match[6]
        region = match[7]
        # print(type(input.iloc[:, row_index:row_index + 2]))
        # data.index = input.index
        data['pos_raw'] = input.iloc[:, row_index * 2]
        data['signal'] = input.iloc[:, row_index * 2 + 1]
        data['pos'] = data['pos_raw'] - zero
        dataset.append(Profile(label=label,
                               animal=animal,
                               geno=geno,
                               image=image,
                               line=None,
                               stain=stain,
                               region=region,
                               zero=zero,
                               data=data))
    return dataset


# iterate parse_ij function over all files in folder

def parse_data_folder(path, animal_meta):
    # TODO: make this function more efficient. concat tuples not dfs
    """This function will concatenate all of the data from a single folder into a single dataset object"""
    dataset = ProfileDataset()
    for r, d, f in os.walk(path):
        for file in f:
            if '_profiles.csv' in file:
                data, meta, image_file = get_file_info(r, file)
                image_profiles = parse_ij(data, meta, animal_meta, image_file)
                for profile in image_profiles.profiles:
                    dataset.append(profile)
    # TODO: add output messages stating the number of images and profiles
    return dataset


# plot_heatmap of each profile
def plot_heatmap(data, stain, crop=False):
    # TODO: test crop flag
    """generate a heatmap of the dataset sorted by genotype"""
    if not crop:
        stain_data = data[data['stain'] == stain].sort_values(by=['geno'])
        pivot = stain_data.pivot_table(index='label', columns='pos', values='signal')
    if crop:
        # TODO: test if cutoff values have been added to dataset. return message if not
        cropped_data = data[(pos > data.cutoff[0]) & (pos < data.cutoff[1])]
        stain_data = cropped_data[cropped_data['stain'] == stain].sort_values(by=['geno'])
        pivot = stain_data.pivot_table(index='label', columns='pos', values='signal')
    hm = sns.heatmap(pivot, yticklabels='')
    hm.set_title(stain + " profiles")
    return hm

# function to plot per animal - no ci


def trace_per_animal(data, stain):
    # TODO: make this function annotate regions of the dataset
    spy_start = -50
    spy_end = 50
    trace = sns.lineplot(data=data[(data['stain'] == stain)],
                         x='pos',
                         y='signal',
                         hue='animal',
                         palette=sns.color_palette("colorblind", data[(data['stain'] == stain)]['animal'].nunique()),
                         estimator='mean',
                         ci=None,
                         alpha=0.5)
    trace.set_title("average " + stain + " profile per animal - no ci")

    trace.axvspan((spy_start - 100), spy_start, color=sns.xkcd_rgb['grey'], alpha=0.25, )
    trace.axvspan(spy_end, (spy_end + 100), color=sns.xkcd_rgb['grey'], alpha=0.25)
    trace.set_alpha(0.2)

    return trace


# plot means with sd confidence interval
# need to fix # of n's ---  collapse per animal
def trace_per_geno(data, stain):
    spy_start = -50
    spy_end = 50

    trace = sns.lineplot(data=data[(data['stain'] == stain)],
                         x='pos',
                         y='signal',
                         palette=sns.color_palette("colorblind", data[(data['stain'] == stain)]['geno'].nunique()),
                         hue='geno',
                         ci=68)
    trace.set_title("average " + stain + " profile per genotype - ci")
    trace.axvspan((spy_start - 100), spy_start, color=sns.xkcd_rgb['grey'], alpha=0.25)
    trace.axvspan(spy_end, (spy_end + 100), color=sns.xkcd_rgb['grey'], alpha=0.25)
    return trace


def trace_per_animal_by_geno(data):
    """Creates a sns facet grid of profiles collapsed at the animal and gen levels.
       The input is ProfileDataSet.data"""
    # fig = plt.figure(figsize=(8, 8), dpi= 80, facecolor='w', edgecolor='k')
    spy_start = -50
    spy_end = 50

    sub_agg = data. \
        groupby(['stain', 'region', 'geno', 'animal', 'pos']). \
        agg(mean_signal=pd.NamedAgg(column='signal', aggfunc='mean'))
    sub_agg.reset_index(inplace=True)
    sub_agg = sub_agg.astype({'pos': 'int64'})
    fig = sns.relplot(x="pos",
                      y="mean_signal",
                      palette=sns.color_palette("colorblind", data['geno'].nunique()),
                      hue="geno",
                      col="region",
                      row="stain",
                      height=3, aspect=1.5, linewidth=1,
                      kind="line",
                      data=sub_agg[(sub_agg["pos"] >= -500) & (sub_agg["pos"] <= 1000)],
                      ci=68,
                      facet_kws={'sharey': False, 'sharex': True}
                      )

    fig.map(plt.axvspan, xmin=(spy_start - 100), xmax=spy_start, color=sns.xkcd_rgb['grey'], alpha=0.25)
    fig.map(plt.axvspan, xmin=spy_end, xmax=(spy_end + 100), color=sns.xkcd_rgb['grey'], alpha=0.25)
    return fig


# region analysis function
# This function needs to be made to accept any dataset
"""
def region_analysis(data, region, geno_loc):
    # mean signal across region
    for region in regions
    
        for geno in genos
    region_wt = data[data['geno'] == 'Shh+/+'].iloc[:, regions[region]].mean(axis=1)
    region_cc = data[data['geno'] == 'Shhc/c'].iloc[:, regions[region]].mean(axis=1)
    print("t-test of mean across " + region)
    plot dot plots for each geno in each region
    output summary table
    stat = stats.ttest_rel(region_wt, region_cc, axis=0)
    print(stat)

    # return bar graph with p value and print stats for repeated and mean for the region
    bar = sns.barplot(y=data.iloc[:, regions[region]].mean(axis=1),
                      x=data.iloc[:, geno_loc],
                      ci='sd')
    bar.set_title("mean fluorescence of " + region)
    bar.text(0.5, 0.95,
             "p=" + str(round(stat.pvalue, 3)),
             horizontalalignment='center',
             verticalalignment='center',
             transform=bar.transAxes)
    # repeated measures
    # region_tom_rm = data[data['geno'] == 'Tom'].iloc[:, regions[region]]
    # region_M2_rm = data[data['geno'] == 'SmoM2'].iloc[:, regions[region]]
    # print("t-test of across " + region)
    # print(stats.ttest_rel(region_tom_rm, region_M2_rm, axis=0))
    return bar
"""

# scale data so that x axis is consistent

# Function to compare regions


def trace_per_region(data, stain):
    spy_start = -50
    spy_end = 50
    trace = sns.lineplot(data=data[(data['stain'] == stain)],
                         x='pos',
                         y='signal',
                         palette=sns.color_palette("colorblind", data[(data['stain'] == stain)]['geno'].nunique()),
                         hue='geno',
                         style='region')
    trace.set_title("average " + stain + " profile per genotype")
    trace.axvspan((spy_start - 100), spy_start, color=sns.xkcd_rgb['grey'], alpha=0.25)
    trace.axvspan(spy_end, (spy_end + 100), color=sns.xkcd_rgb['grey'], alpha=0.25)
    return trace


def plot_datapoints(data, subregions, style, save_prefix):
    """Generates a series of plots showing the mean signal for each animal and each subregion.
        data: ProfileDataset.data. A long pandas dataframe.
        subregions: a dict of subregions. names as key. numpy range as values
    """
    for subregion in subregions:
        sub_agg = data[data['pos'].
                        isin(subregions[subregion])].\
                        groupby(['stain', 'region', 'geno', 'animal']).\
                        agg(mean_signal=pd.NamedAgg(column='signal', aggfunc='mean'))
        sns.set_style("whitegrid")
        ax = sns.catplot(data=sub_agg.reset_index(),
                         kind=style,
                         dodge=True,
                         y='mean_signal',
                         x='stain',
                         hue='geno',
                         col='region',
                         ci=68)
        ax.fig.suptitle(subregion, fontsize=24, x=0.2)
        ax.fig.savefig(save_prefix + subregion + ".pdf")
        plt.show()


def grouped_ttests(data, subregions):
    """
    """
    # TODO: Add test for the number of groups. give error if more than 2 groups
    d = []
    for subregion in subregions:
        sub_agg = data[data['pos'].
                       isin(subregions[subregion])]. \
            groupby(['stain', 'region', 'geno', 'animal']). \
            agg(mean_signal=pd.NamedAgg(column='signal', aggfunc='mean'))

        for stain in sub_agg.index.get_level_values(0).unique():

            for region in sub_agg.index.get_level_values(1).unique():
                genos = [geno for geno in sub_agg.index.get_level_values(2).unique()]
                values1 = sub_agg.xs((stain, region, genos[0]), level=('stain', 'region', 'geno')).values
                values2 = sub_agg.xs((stain, region, genos[1]), level=('stain', 'region', 'geno')).values
                t, p = stats.ttest_ind(values1, values2)

                d.append(
                    {
                        "stain": stain,
                        "region": region,
                        "subregion": subregion,
                        "geno1": genos[0],
                        "geno2": genos[1],
                        "geno1_mean": values1.mean(),
                        "geno1_sem": stats.sem(values1)[0],
                        "geno2_mean": values2.mean(),
                        "geno2_sem": stats.sem(values2)[0],
                        "geno1_n": len(values1),
                        "geno2_n": len(values2),
                        "p-value": p[0]
                    }
                )

    return pd.DataFrame(d)
