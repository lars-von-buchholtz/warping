
This code for warping in situ to in vivo images accompanies the publication by [von Buchholtz et al.](https://www.cell.com/neuron/pdfExtended/S0896-6273(20)30852-7) in Neuron.


# DRAW WARPING VECTORS

Open FIJI ImageJ and install the `warpingvectors.ijm` macro.

From the Plugins/Macros menu run the align_patch macro for a crude initial alignment of the stacks. As prompted use the multi-point tool to select 3 matching points on the reference stack and the source stack to crudely align the source stack to the reference stack.


Work on the red/green overlay generated above to draw the warping vectors (from the source point to the target point).

Before beginning, make sure the ROI manager and Results table are both empty.

Activate the drawing tool by pressing the 'Record' icon (filled circle) in the toolbar.

Click and drag (typically from red to green) to draw vectors.

Pressing 'd' will delete the last vector.

Pressing 's' will save the data (select .tsv extension to save as tab separated values).

Pressing 'l' will load saved data from a .tsv file back into the Results/ROIs

When done save the Results as .tsv file which is being used as 
[VECTOR_FILENAME] in the next section.



# WARPING

## Setting up required software

### On Windows or Mac using Anaconda

If you dont have it, install [Anaconda](https://docs.anaconda.com/anaconda/install/windows/).
On Windows use the  Anaconda Prompt (you might have to run it as administrator) for the following steps, on Mac you can use the Terminal.
 Go to the warping directory, create a new conda environment named `warping37` with python 3.7 and pip install the required packages from the requirements.txt file.

```bash
conda create -n warping37  python=3.7 anaconda
conda install -n warping37 pip
conda activate warping37
pip install -r requirements.txt
```

Run the warp.py script in the active environment. Use `conda deactivate` to deactivate your environment.


### Running the program

Activate your warping37 environment (`conda activate warping37`; use `conda deactivate` to deactivate your environment when done)

Run the warp.py script with the appropriate parameters.

```bash
 python python/warp.py --src=[SOURCE_FILENAME] --ref=[REFERENCE_FILENAME] --out=[BASENAME_FOR_OUTPUT] --vec=[VECTOR_FILENAME] --dir=[DIRECTORY]
```
  
with the following parameters
  

- `SOURCE_FILENAME` = image stack to be warped (alignment channel in first slice)
- `REFERENCE_FILENAME` = reference image stack that the source is being warped to (alignment channel in first slice)
- `VECTOR_FILENAME` = csv file with source X, source Y, target X, target Y coordintes as columns
- `DIRECTORY` = the directory that these 3 files are in and where the output is generated
- `BASENAME_FOR_OUTPUT` = arbritrary name that is appended by the specific output files

The script generates the following files in `DIRECTORY` as output.

OUTPUT:

- `BASENAME_FOR_OUTPUT-outstack.tif` = warped stack from SOURCE_FILENAME
- `BASENAME_FOR_OUTPUT-input.tif` = alignment channels overlaid before warping
- `BASENAME_FOR_OUTPUT-overlay.tif` = alignment channels overlaid after warping
- `BASENAME_FOR_OUTPUT-arrows.svg` = vector drawing of arrows connecting source and target coordinates (open in Illustrator or Inkscape)
- `BASENAME_FOR_OUTPUT-srctriangles.svg` = vector drawing of Delaunay triangles of source coordinates
- `BASENAME_FOR_OUTPUT-reftriangles.svg` = vector drawing of corresponding triangles of target coordinates


### Platform-independent use of Docker container


If you have trouble installing and running the script on your platform, you can use the provided Docker container. You have to install [Docker](https://docs.docker.com/install/) and
[Docker Compose](https://docs.docker.com/compose/install/).  

Before using the Docker container you need to build it (from the warping directory).

```bash
# before you use the software for the first time
docker-compose build
```

Then run the warp.py script using the Docker container. Note that you have to mount your working directory with the `-v` option.

```bash

docker-compose run -v [DIRECTORY]:[DIRECTORY] app python warp.py --src=[SOURCE_FILENAME] --ref=[REFERENCE_FILENAME] --out=[BASENAME_FOR_OUTPUT] --vec=[VECTOR_FILENAME] --dir=[DIRECTORY]
```

# CALCIUM TRACE EXTRACTION AND BACKGROUND SUBTRACTION

To extract dF/F fluorescence traces for a set of ImageJ ROIs from a stack of timeframes, subtract the surrounding background fluorescence and perform quality control, we are making available a Matlab script that was written by Marcin Szczot and has been first described [here](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5599122/).  
It contains a read function  for large tiff files written by Darcy Peterka (dp2403@columbia.edu) and published in [Pnevmatikakis et al](https://www.ncbi.nlm.nih.gov/pmc/articles/pmid/26774160/)
and requires the [geom2d library](https://www.mathworks.com/matlabcentral/fileexchange/7844-geom2d).  
  
### Installation:

1) Unpack the zip file into any folder you want.
2) In matlab go to home->set path -> add with subfolders -> choose a folder to which you unpacked; choose 'save' at the bottom to not have to do this every time you start matlab. 

### Operation  

1) place a movie and all unpacked ROI files selected from ImageJ (unpacked from a zip file you get when saving ROI in imageJ)
2) In matlab go to open-> find your directory and choose 'main_neuropil_sub.m'
3) on the top go to editor tab and hit Run.
4) in gui choose a folder where your data is stored and where you want to save output click let's get analyzing.
5) after it sucesfully loads you should see gui with various controls, refer to screenshot I attached which describes what is what.
6) after you gho through all the cells gui will close, in the output folder you will find screenshot for every cell and most importantly csv files where every column is a calcium trace for a cell. You can use that for data visualization.

### Output  
  
   - neuro_corr_traces.csv has all the traces with the first column indicating frames
   - neuro_corr_traces_labelled.csv has an extra 2 rows with area in pixel and imageJ ROI name for every cell.
   - there are also matlab variables with a list of accepted ROI and rejected ROIs.
 


# ANALYSIS OF DATA for von Buchholtz et al.

A Jupyter notebook that performs much of the qualitative and quantitative analysis for the manuscript published in [Neuron] (https://www.cell.com/neuron/pdfExtended/S0896-6273(20)30852-7) is provided as part of this repo. A rendered HTML version of this notebook is also available.

