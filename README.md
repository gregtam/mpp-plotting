# mpp-plotting

This repository includes files that illustrate how to create plots in an MPP database such as HAWQ or GPDB. I wrote a blog about this. It can be found [here](http://engineering.pivotal.io/post/mpp-plotting/).

## Files
### Notebooks
- `MPP Decision Tree.ipynb`: This file shows to build a Decision Tree in PL/Python, then apply it to a table in HAWQ or GPDB while retaining the actual leaf number. This is useful if we want to determine the distribution of labels in a given leaf.

- `MPP Plotting.ipynb`: This file gives examples of how to plot on the order of millions and billions of data from HAWQ or GPDB. This uses functions from `mpp_plotting.py` to summarize the data into manageable pieces. We then use matplotlib to plot these.  
- `MPP ROC Curve.ipynb`: This file shows how to plot an ROC curve from data in HAWQ or GPDB.

### Python FIles
- `credentials.py`: This file includes login information into an MPP database. It is important to keep these separate from the notebook so that login information is not present inside of the notebook.

- `mpp_plotting.py`: This file includes all function definitions for the backend plotting functions.

- `sql_functions.py`: This file defines utility functions for interacting with the cluster (e.g., getting the table or column names).
