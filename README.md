# FinanceOps

[Original repository on GitHub](https://github.com/Hvass-Labs/FinanceOps)

Original author is [Magnus Erik Hvass Pedersen](http://www.hvass-labs.org)


## Introduction

This is a small collection of research papers on long-term investing.
They are written as Python Notebooks so they can be easily modified
and run again.


### Videos

There is a [YouTube video](https://www.youtube.com/playlist?list=PL9Hr9sNUjfsmlHaWuVxIA0pKL1yjryR0Z) for each research paper.


## Papers

1. Forecasting Long-Term Stock Returns ([Notebook](https://github.com/Hvass-Labs/FinanceOps/blob/master/01_Forecasting_Long-Term_Stock_Returns.ipynb)) ([Google Colab](https://colab.research.google.com/github/Hvass-Labs/FinanceOps/blob/master/01_Forecasting_Long-Term_Stock_Returns.ipynb))
2. Comparing Stock Indices ([Notebook](https://github.com/Hvass-Labs/FinanceOps/blob/master/02_Comparing_Stock_Indices.ipynb)) ([Google Colab](https://colab.research.google.com/github/Hvass-Labs/FinanceOps/blob/master/02_Comparing_Stock_Indices.ipynb))

## Downloading

The Python Notebooks use source-code located in different files to allow for easy re-use
across multiple Notebooks. It is therefore recommended that you download the whole
repository from GitHub, instead of just downloading the individual Python Notebooks.

### Git

The easiest way to download and install this is by using git from the command-line:

    git clone https://github.com/Hvass-Labs/FinanceOps.git

This creates the directory `FinanceOps` and downloads all the files to it.

This also makes it easy to update the files, simply by executing this command inside that directory:

    git pull

### Zip-File

You can also [download](https://github.com/Hvass-Labs/FinanceOps/archive/master.zip)
the contents of the GitHub repository as a Zip-file and extract it manually.


## How To Run

If you want to edit and run the Notebooks on your own computer, then it is
suggested that you use the [Anaconda](https://www.anaconda.com/download)
distribution of **Python 3.6** (or later) because it has all the required packages
already installed. Once you have installed Anaconda, you run the following
command from the `FinanceOps` directory to view and edit the Notebooks:

    jupyter notebook

If you want to edit the other source-code then you may use the free version of [PyCharm](https://www.jetbrains.com/pycharm/).


### Run in Google Colab

If you do not want to install anything on your own computer, then the Notebooks
can be viewed, edited and run entirely on the internet by using
[Google Colab](https://colab.research.google.com).
You can click the "Google Colab"-link next to the research papers listed above.
You can view the Notebook on Colab but in order to run it you need to login using
your Google account.
Then you need to execute the following commands at the top of the Notebook,
which clones FinanceOps to your work-directory on Colab.

    import os
    work_dir = "/content/FinanceOps/"
    if os.getcwd() != work_dir:
        !git clone https://github.com/Hvass-Labs/FinanceOps.git
    os.chdir(work_dir)

## Data Sources

- Price data from [Yahoo Finance](https://finance.yahoo.com/quote/%5EGSPC/history?p=%5EGSPC).
- Financial data for individual stocks collected manually by the author from the 10-K Forms filed with the [U.S. SEC](http://www.sec.gov/cgi-bin/browse-edgar?company=&match=&CIK=jnj&filenum=&State=&Country=&SIC=&owner=exclude&Find=Find+Companies&action=getcompany).
- Newer S&P 500 data from the [S&P Earnings & Estimates Report](http://www.spindices.com/documents/additional-material/sp-500-eps-est.xlsx) and older data from the research staff at S&P and Compustat (some older data is approximated by their research staff).
- U.S. Government Bond yield for 1-year constant maturity. From the [U.S. Federal Reserve](https://www.federalreserve.gov/datadownload/Choose.aspx?rel=H15).
- The inflation index is: All Items Consumer Price Index for All Urban Consumers (CPI-U), U.S. City Average. From the [US Department of Labor, Bureau of Labor Statistics](http://www.bls.gov/cpi/data.htm).


## License (MIT)

These Python Notebooks and source-code are published under the [MIT License](https://github.com/Hvass-Labs/FinanceOps/blob/master/LICENSE)
which allows very broad use for both academic and commercial purposes.

You are very welcome to modify and use the source-code in your own project.
Please keep a link to the [original repository](https://github.com/Hvass-Labs/FinanceOps).

The financial data is **not** covered by the MIT license and may have limitations on commercial redistribution, etc.
