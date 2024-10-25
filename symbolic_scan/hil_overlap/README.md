# Hil Overlap for the ISI/LZ scan plot

This directory contains graphical assets, MATLAB scripts, and Julia code for assembling the inter-spike interval (ISI) scan plot with Lempel-Ziv complexities (LZ) of the template for the SiN model overlaid on top, as well as David's, Andrey's, and Hil's MATCONT bifurcation continuation diagram.
No code in this directory generates new scan data.

## Usage
Multiple steps are required to fully assemble the resulting plot, including executing several scripts both in Julia and MATLAB in a particular order:

1. `ISI_and_LZ_plot.jl`: This program generates the background image `ISI_and_LZ_plot.png`, which contains the ISI scan plot with LZ complexities overlaid.
2. `carter_diagram_als_new.m`: This script displays `ISI_and_LZ_plot.png` in MATLAB, as well as plots the MATCONT curves and points generated by David and Andrey overlaid onto the scan background.
3. `to_overlap_carter_hil.m`: This script overlays Hil's MATCONT bifurcation diagram on top of the previous plot.

## Oversized files
Some files in this directory are too large to store on GitHub, so they are included in the `.gitignore` file and must be hosted elsewhere (Dropbox/Telegram):

- `Homoclinic.mat`: 540.30 MB
- `SNPO_2.mat`: 57.37 MB