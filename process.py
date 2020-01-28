import numpy as np
import xarray as xr
import re
from pathlib import Path
import collections

def distance(val, ref):
    return abs(ref - val)
vectDistance = np.vectorize(distance)

def cmap_xmap(function, cmap):
    """ Applies function, on the indices of colormap cmap. Beware, function
    should map the [0, 1] segment to itself, or you are in for surprises.

    See also cmap_xmap.
    """
    cdict = cmap._segmentdata
    function_to_map = lambda x : (function(x[0]), x[1], x[2])
    for key in ('red','green','blue'):
        cdict[key] = map(function_to_map, cdict[key])
#        cdict[key].sort()
#        assert (cdict[key][0]<0 or cdict[key][-1]>1), "Resulting indices extend out of the [0, 1] segment."
    return matplotlib.colors.LinearSegmentedColormap('colormap',cdict,1024)

def getClosest(sortedMatrix, column, val):
    while len(sortedMatrix) > 3:
        half = int(len(sortedMatrix) / 2)
        sortedMatrix = sortedMatrix[-half - 1:] if sortedMatrix[half, column] < val else sortedMatrix[: half + 1]
    if len(sortedMatrix) == 1:
        result = sortedMatrix[0].copy()
        result[column] = val
        return result
    else:
        safecopy = sortedMatrix.copy()
        safecopy[:, column] = vectDistance(safecopy[:, column], val)
        minidx = np.argmin(safecopy[:, column])
        safecopy = safecopy[minidx, :].A1
        safecopy[column] = val
        return safecopy

def convert(column, samples, matrix):
    return np.matrix([getClosest(matrix, column, t) for t in samples])

def valueOrEmptySet(k, d):
    return (d[k] if isinstance(d[k], set) else {d[k]}) if k in d else set()

def mergeDicts(d1, d2):
    """
    Creates a new dictionary whose keys are the union of the keys of two
    dictionaries, and whose values are the union of values.

    Parameters
    ----------
    d1: dict
        dictionary whose values are sets
    d2: dict
        dictionary whose values are sets

    Returns
    -------
    dict
        A dict whose keys are the union of the keys of two dictionaries,
    and whose values are the union of values

    """
    res = {}
    for k in d1.keys() | d2.keys():
        res[k] = valueOrEmptySet(k, d1) | valueOrEmptySet(k, d2)
    return res

def extractCoordinates(filename):
    """
    Scans the header of an Alchemist file in search of the variables.

    Parameters
    ----------
    filename : str
        path to the target file
    mergewith : dict
        a dictionary whose dimensions will be merged with the returned one

    Returns
    -------
    dict
        A dictionary whose keys are strings (coordinate name) and values are
        lists (set of variable values)

    """
    with open(filename, 'r') as file:
#        regex = re.compile(' (?P<varName>[a-zA-Z._-]+) = (?P<varValue>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?),?')
        regex = r"(?P<varName>[a-zA-Z._-]+) = (?P<varValue>[^,]*),?"
        dataBegin = r"\d"
        is_float = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"
        for line in file:
            match = re.findall(regex, line)
            if match:
                return {
                    var : float(value) if re.match(is_float, value)
                        else bool(re.match(r".*?true.*?", value.lower())) if re.match(r".*?(true|false).*?", value.lower())
                        else value
                    for var, value in match
                }
            elif re.match(dataBegin, line[0]):
                return {}

def extractVariableNames(filename):
    """
    Gets the variable names from the Alchemist data files header.

    Parameters
    ----------
    filename : str
        path to the target file

    Returns
    -------
    list of list
        A matrix with the values of the csv file

    """
    with open(filename, 'r') as file:
        dataBegin = re.compile('\d')
        lastHeaderLine = ''
        for line in file:
            if dataBegin.match(line[0]):
                break
            else:
                lastHeaderLine = line
        if lastHeaderLine:
            regex = re.compile(' (?P<varName>\S+)')
            return regex.findall(lastHeaderLine)
        return []

def openCsv(path):
    """
    Converts an Alchemist export file into a list of lists representing the matrix of values.

    Parameters
    ----------
    path : str
        path to the target file

    Returns
    -------
    list of list
        A matrix with the values of the csv file

    """
    regex = re.compile('\d')
    with open(path, 'r') as file:
        lines = filter(lambda x: regex.match(x[0]), file.readlines())
        return [[float(x) for x in line.split()] for line in lines]

if __name__ == '__main__':
    # CONFIGURE SCRIPT
    # Where to find Alchemist data files
    directory = 'data'
    # Where to save charts
    output_directory = 'charts'
    # How to name the summary of the processed data
    pickleOutput = 'data_summary'
    # Experiment prefixes: one per experiment (root of the file name)
    experiments = ['simulation']
    floatPrecision = '{: 0.3f}'
    # Number of time samples 
    timeSamples = 600
    # time management
    minTime = 0
    maxTime = 600
    timeColumnName = 'time'
    logarithmicTime = False
    # One or more variables are considered random and "flattened"
    seedVars = ['seed']
    
    # Setup libraries
    np.set_printoptions(formatter={'float': floatPrecision.format})
    # Read the last time the data was processed, reprocess only if new data exists, otherwise just load
    import pickle
    import os
    newestFileTime = max(os.path.getmtime(directory + '/' + file) for file in os.listdir(directory))
    try:
        lastTimeProcessed = pickle.load(open('timeprocessed', 'rb'))
    except:
        lastTimeProcessed = -1
    shouldRecompute = newestFileTime != lastTimeProcessed
    if not shouldRecompute:
        try:
            means = pickle.load(open(pickleOutput + '_mean', 'rb'))
            stdevs = pickle.load(open(pickleOutput + '_std', 'rb'))
        except:
            shouldRecompute = True
    if shouldRecompute:
        timefun = np.logspace if logarithmicTime else np.linspace
        means = {}
        stdevs = {}
        for experiment in experiments:
            # Collect all files for the experiment of interest
            import fnmatch
            allfiles = filter(lambda file: fnmatch.fnmatch(file, experiment + '_*.txt'), os.listdir(directory))
            allfiles = [directory + '/' + name for name in allfiles]
            allfiles.sort()
            # From the file name, extract the independent variables
            dimensions = {}
            for file in allfiles:
                dimensions = mergeDicts(dimensions, extractCoordinates(file))
            dimensions = {k: sorted(v) for k, v in dimensions.items()}
            # Add time to the independent variables
            dimensions[timeColumnName] = range(0, timeSamples)
            # Compute the matrix shape
            shape = tuple(len(v) for k, v in dimensions.items())
            # Prepare the Dataset
            dataset = xr.Dataset()
            for k, v in dimensions.items():
                dataset.coords[k] = v
            if len(allfiles) == 0:
                print("WARNING: No data for experiment " + experiment)
            else:
                varNames = extractVariableNames(allfiles[0])
                for v in varNames:
                    if v != timeColumnName:
                        novals = np.ndarray(shape)
                        novals.fill(float('nan'))
                        dataset[v] = (dimensions.keys(), novals)
                # Compute maximum and minimum time, create the resample
                timeColumn = varNames.index(timeColumnName)
                allData = { file: np.matrix(openCsv(file)) for file in allfiles }
                computeMin = minTime is None
                computeMax = maxTime is None
                if computeMax:
                    maxTime = float('-inf')
                    for data in allData.values():
                        maxTime = max(maxTime, data[-1, timeColumn])
                if computeMin:
                    minTime = float('inf')
                    for data in allData.values():
                        minTime = min(minTime, data[0, timeColumn])
                timeline = timefun(minTime, maxTime, timeSamples)
                # Resample
                for file in allData:
#                    print(file)
                    allData[file] = convert(timeColumn, timeline, allData[file])
                # Populate the dataset
                for file, data in allData.items():
                    dataset[timeColumnName] = timeline
                    for idx, v in enumerate(varNames):
                        if v != timeColumnName:
                            darray = dataset[v]
                            experimentVars = extractCoordinates(file)
                            darray.loc[experimentVars] = data[:, idx].A1
                # Fold the dataset along the seed variables, producing the mean and stdev datasets
                means[experiment] = dataset.mean(dim = seedVars, skipna=True)
                stdevs[experiment] = dataset.std(dim = seedVars, skipna=True)
        # Save the datasets
        pickle.dump(means, open(pickleOutput + '_mean', 'wb'), protocol=-1)
        pickle.dump(stdevs, open(pickleOutput + '_std', 'wb'), protocol=-1)
        pickle.dump(newestFileTime, open('timeprocessed', 'wb'))
        
    # QUICK CHARTING

    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.cm as cmx
    matplotlib.rcParams.update({'axes.titlesize': 12})
    matplotlib.rcParams.update({'axes.labelsize': 10})
    def make_line_chart(xdata, ydata, title = None, ylabel = None, xlabel = None, colors = None, linewidth = 1, errlinewidth = 0.5, figure_size = (6, 4)):
        fig = plt.figure(figsize = figure_size)
        ax = fig.add_subplot(1, 1, 1)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
#        ax.set_ylim(0)
#        ax.set_xlim(min(xdata), max(xdata))
        index = 0
        for (label, (data, error)) in ydata.items():
            lines = ax.plot(xdata, data, label=label, color=colors(index / (len(ydata) - 1)) if colors else None, linewidth=linewidth)
            index += 1
            if error:
                last_color = lines[-1].get_color()
                ax.plot(xdata, data+error, label=None, color=last_color, linewidth=errlinewidth)
                ax.plot(xdata, data-error, label=None, color=last_color, linewidth=errlinewidth)
        return (fig, ax)
    for experiment in experiments:
        current_experiment_means = means[experiment]
        for comparison_variable in set(current_experiment_means.coords) - {timeColumnName}:
            mergeable_variables = set(current_experiment_means.coords) - {timeColumnName, comparison_variable}
            for current_coordinate in mergeable_variables:
                merge_variables = mergeable_variables - { current_coordinate }
                merge_data_view = current_experiment_means.mean(dim = merge_variables, skipna = True)
                for current_coordinate_value in merge_data_view[current_coordinate].values:
                    for current_metric in merge_data_view.data_vars:
                        title = current_metric + " with " + comparison_variable + " when " + current_coordinate + "=" + str(current_coordinate_value)
                        fig, ax = make_line_chart(
                            title = title,
                            xdata = merge_data_view[timeColumnName],
                            xlabel = timeColumnName,
                            ydata = {
                                label: (merge_data_view.sel({comparison_variable: label, current_coordinate: current_coordinate_value})[current_metric], 0)
                                    for label in merge_data_view[comparison_variable].values
                            },
                            ylabel = current_metric
                        )
                        ax.legend()
                        fig.tight_layout()
                        by_time_output_directory = output_directory + "/" + experiment + "/by-time/" + comparison_variable
                        Path(by_time_output_directory).mkdir(parents=True, exist_ok=True)
                        fig.savefig(by_time_output_directory + "/" + current_metric + "_" + current_coordinate + "_" + str(current_coordinate_value) + ".pdf")
                        plt.close(fig)
    # Prepare the charting system
#    import seaborn as sns
#    from matplotlib.colors import LogNorm
#    def make_heatmap_chart(matrix, vmin = None, vmax = None, ticks = None, show_values = False, norm = None, title = None, ylabel = None, xlabel = None, colors = None, figure_size = (6, 4)):
#        fig = plt.figure(figsize = figure_size)
#        ax = fig.add_subplot(1, 1, 1)
#        ax.set_title(title)
#        sns.heatmap(
#            matrix,
#            vmin = vmin,
#            vmax = vmax,
#            annot = show_values,
#            ax = ax,
#            norm = norm,
#            cmap = colors,
#            cbar_kws = { "ticks": ticks } if ticks else None
#        )
#        ax.invert_yaxis()
#        if xlabel:
#            ax.set_xlabel(xlabel)
#        if ylabel:
#            ax.set_ylabel(ylabel)
#        return (fig, ax)
#    for experiment in experiments:
#        metric = 'taskHops[Mean]'
#        current_experiment_data = means[experiment].mean(dim = timeColumnName)[metric]
#        labelmap = { True: "process-based", False: "gradient-based" }
#        for value in current_experiment_data.coords[comparison_variable].values:
#            heatmap_data = current_experiment_data.sel({comparison_variable: value})
#            axes_names = list({ name for name in heatmap_data.coords } - { comparison_variable })
#            values_x = heatmap_data.coords[axes_names[0]]
#            values_y = heatmap_data.coords[axes_names[1]]
#            extent = (min(values_x), max(values_x), min(values_y), max(values_y))
#            log_norm = LogNorm(vmin=1, vmax=heatmap_data.max())
#            ticks = [1, 10, 20, 40]
##            basecolormap = cmx.seismic
#            fig, ax = make_heatmap_chart(
#                heatmap_data.clip(1, 100).to_pandas().transpose(),
#                title = "Route length with " + labelmap[value] + " discovery",
#                norm = log_norm,
#                ylabel = "Mean route length to cloud provider (hops)",
#                xlabel = "Per-client request creation rate (Hz)",
#                colors = cmx.viridis_r,
#                vmin = 1,
#                vmax = 100,
#                show_values = False,
#                ticks = [1, 10, 20, 100, current_experiment_data.max()]
#            )
#            fig.tight_layout()
#            fig.savefig(output_directory + "/" + labelmap[value] + ".pdf")
#            plt.close(fig)
#
#    # Prepare selected charts
#    # Evaluation of the backoff parameter
#    
#    # CHART set 1: broadcast-time + ccast-time
#    # CHART set 2: performance w.r.t. stage width
#    allwidths = means['corridor']
#    reference_stage_width = 2000
#    data_by_time = allwidths.sel(stage_width=reference_stage_width)
#    data_by_width = allwidths.mean('time')
#    charterrordata = stdevs['corridor']
#    mixcolormap = lambda x: cmx.winter(x * 2) if x < 0.5 else cmx.YlOrRd((x - 0.5) * 2 * 0.6 + 0.3)
#    divergingmixcolormap = lambda x: cmx.winter(1 - x * 2) if x < 0.5 else cmx.YlOrRd((x - 0.5) * 2 * 0.6 + 0.3)
#    for algorithm in ['b', 'c']:
#        # wrt time
#        fig, ax = makechart(
#            xdata = data_by_time['time'],
#            ydata = {
#                primitive + kind : (
#                    data_by_time[label],
#                    stdevs['corridor'].sel(stage_width=reference_stage_width)[label]
#                )
#                for label, primitive, kind in (
#                    (primitive + "-" + algorithm + "cast" + kind + "[Sum]", primitive, kind)
#                    for primitive in ['rep', 'share']
#                    for kind in ["-single", ""]
#                )
#            },
#            ylabel = "Packet delay (s)",
#            xlabel = "Simulation time (s)",
#            figure_size = (6, 3),
#            colors = mixcolormap,
#            linewidth = 1.5,
#            title = "rep vs. share performance, " + ("broadcast" if algorithm == 'b' else 'accumulation')
#        )
#        ax.legend()
#        fig.tight_layout()
#        fig.savefig("delay-" + algorithm + ".pdf")
#        fig, ax = makechart(
#            xdata = data_by_width['stage_width'],
#            ydata = {
#                primitive + kind : (
#                    data_by_width[label],
#                    stdevs['corridor'].mean('time')[label]
#                )
#                for label, primitive, kind in (
#                    (primitive + "-" + algorithm + "cast" + kind + "[Sum]", primitive, kind)
#                    for primitive in ['rep', 'share']
#                    for kind in ["-single", ""]
#                )
#            },
#            ylabel = "Mean packet delay (s)",
#            xlabel = "Distance between source and destination (m)",
#            figure_size = (6, 3),
#            colors = mixcolormap,
#            linewidth = 1.5,
#            title = "rep vs. share performance, " + ("broadcast" if algorithm == 'b' else 'accumulation')
#        )
##        ax.set_xscale('log')
#        ax.legend()
#        fig.tight_layout()
#        fig.savefig("width-" + algorithm + ".pdf")
        