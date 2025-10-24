MFG_convergence_rate.ipynb is essentially the code from MFG_verification.ipynb, put inside a 
for-loop so that runs with different grid sizes h can be compared. The results from runs of the code are placed into the MFG_convergence_rate_runs folder, and the parameters used in each of the runs have been (manually) saved in the MFG_convergence_rate_runs_parameters.ipynb file. 

See first line in each cell of MFG_convergence_rate.ipynb for a quick description of that cell's function.
- To generate new data, use the #RUN SIMULATION cell. Enter the desired grid sizes h and the desired parameters.
- To load data from an old run, go to the #READ FROM FILE cell and enter the correct run number in the file name and the grid sizes h that were used into h_reference and h_list.
- After generating new data or loading data from an old run, to plot the results use the #PLOT RESULTS cell. Choose the same h_list and parameters as were used to generate the data.