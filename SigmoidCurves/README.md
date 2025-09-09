# Requirements:

Python 3.13.5

scipy           1.16.1

pandas          2.3.2

matplotlib      3.10.6

numpy           2.3.2


# OS

Should work on any OS that has python and pip, tested on Debain 13.

# Scripts

Fitting curves has two steps; 1) data filtering, 2) Curve Fitting.

## Data filtering

### Data structure 

The script expects the following structure:

```
...
├── Line;X < --- folder
│   └── Size.csv < --- file
├── Line;Y
│   └── Size.csv
...


```

And a csv structure like:

```
Date,ID 1,ID 2,ID 3,ID 4,ID 5,ID 6
2022-04-05,NA,0.062,NA,NA,NA,NA
2022-04-06,NA,0.037,NA,NA,NA,NA
2022-04-08,NA,0.074,NA,NA,NA,NA
...
```

### To run:

Adjust line ` path = ... ` to the path where your data is located

Then

`python3 001_cleanData.py`

## Sigmoid fitting

The output ` Merged_Cleaned_outlier_data.csv ` of ` 002_sigmoidFit.py ` is the input file for y.py

To run:

`python 002_sigmoidFit.py`
