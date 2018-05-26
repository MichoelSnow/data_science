# Data Science


This is a hodgepdoge of topics I find interesting, challenging, or just can never remember in data science.

Almost everything is written as jupyter notebooks, in the notebooks directory.

The loose notebooks are topics not big enough for their own directories, yet.  The two areas of focus so far are

- [Random Forests](notebooks/random_forest)
- [PyTorch](notebooks/pytorch)

## Using the code

In addition to the packages listed in the requirements file, the notebooks will reference code in different directories in the [src](src) folder. The simplest way to add that code to your python environment is to create a symbolic link from the folder to your site packages directory.

For example to utilize the code in the data_sci directory in your conda environment

- *nix
```bash
ln -s ~/git/data_science/src/data_sci /directory/to/env/site-packages
```
- Windows
```
mklink /d c:\git\data_science\src\data_sci e:\directory\to\env\site-packages
```


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
