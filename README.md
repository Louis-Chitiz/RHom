# RHom

Welcome to RHom. 

RHom is a Python-based toolbox for running, visualizing, and testing the robustness of different forms of PCA.

This readme is intended for novices and assumes little-to-no prior knowledge of coding and GitHub. It will take you from installation through your first analysis, with examples.

- [Setting Up Visual-Studio Code](https://github.com/Louis-Chitiz/Rhom/blob/main/tutorials/Installing_VS_Code.md)
- [Setting Up Python in VSCode](https://github.com/Louis-Chitiz/Rhom/blob/main/tutorials/Setting_Up_Python.md)

## Setting Up and Running RHom

To use this repository for your own purposes, you'll have to *fork* off a personal copy and *clone* it to your computer:

- [Fork and Clone RHom](https://github.com/Louis-Chitiz/Rhom/blob/main/tutorials/Fork_and_Clone_RHom.md)

Once you've set up a local fork of the RHom repository on your computer, you can set up a virtual environment for using RHom:

- [Setting Up RHom](https://github.com/Louis-Chitiz/Rhom/blob/main/tutorials/Set_Up_RHom.md)
    - [Updating RHom](https://github.com/Louis-Chitiz/Rhom/blob/main/tutorials/updating_RHom.md)

- [Running Your First PCA Analysis](https://github.com/Louis-Chitiz/Rhom/blob/main/tutorials/First_PCA_Analysis.md)

## The rhom Module: Testing the Robustness of Your Components

After you've generated your components, it's important to get a sense of how robustly they represent your data and how well they  generalize across types of situations (e.g., different sampling environments, different participant populations, etc.).

Usage of rhom primarily involves four functions that assess both component reliability and generalizability in a few different ways. Take a look at the guides below, organized by the questions each function targets, and be sure to look at the example script provided in the /examples folder!

*How robustly do my components represent my data?*
- [Split-Half Reliability](https://github.com/Louis-Chitiz/Rhom/blob/main/tutorials/split-half.md)

*How similar are the components produced by different situations?*
- [Direct-Projection Reproducibility](https://github.com/Louis-Chitiz/Rhom/blob/main/tutorials/direct-project.md)

*How representative are the components I get when I combine data from different situations?*
- [Omnibus-Sample Reproducibility](https://github.com/Louis-Chitiz/Rhom/blob/main/tutorials/omni-sample.md)

Answering these questions requires a metric that provides some indication of the similarity between two components (e.g., generated from different halves of the same dataset, generated from separate datasets with the same measure, etc.). To do so, the rhom module leverages two metrics of component similarity:

- [Tucker's Congruence Coefficient (TCC): Comparing Components by Their Loadings](https://github.com/Louis-Chitiz/Rhom/blob/main/tutorials/tcc.md)
- [R-Homologue: Comparing Components by the Way They Organize Observations](https://github.com/Louis-Chitiz/Rhom/blob/main/tutorials/RHom.md)

