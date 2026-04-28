# beyond-correlation
By Benjamin Avanzi, Guillaume Boglioni Beaulieu, Pierre Lafaye de Micheaux, Ho Ming Lee, Bernard Wong, Rui Zhou.

## Introduction
This repository contains the code for the paper *Beyond pairwise correlation: capturing nonlinear and higher-order dependence with distance statistics* submitted to the [All Actuaries Summit 2026](https://www.actuaries.asn.au/all-actuaries-summit-2026). This paper aims to introduce distance-based dependence statistics in order to test or model the dependence structure between random vairables and/or random vectors, as a complementary tool of the correlation coefficient. In this paper, we applied these statistics with different real world and synthetic datasets for illustrations. The link of the paper will be provided once it is finalised and uploaded to arXiv.

The code for visualising the motivating examples, together with illustrations of the computation of the distance-based dependence statistics, is available [here](https://agi-lab.github.io/beyond-correlation/).

## Datasets
We also provide the datasets used for illustration in this repository. The datasets can be found in the [`data`](data) folder.

## Running the demostration locally
For those who are interested in running the code themselves, they can download the file [`r_illust`](r_illust.qmd) and [`jdcov`](jdcov.qmd). Here, the file [`r_illust`](r_illust.qmd) is coded under `R` which contains the illustrations for the [Hellinger correlation](https://www.tandfonline.com/doi/epdf/10.1080/01621459.2020.1791132), [distance covariance](https://projecteuclid.org/journals/annals-of-applied-statistics/volume-3/issue-4/Brownian-distance-covariance/10.1214/09-AOAS312.full), and the [auto-distance correlation function](https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1467-9892.2011.00780.x). 

For the [joint distance covariance](https://www.tandfonline.com/doi/full/10.1080/01621459.2018.1513364), one can run the file [`jdcov`](jdcov.qmd) and it is in Python. The code was executed in Quarto with Python 3.14.3. Required packages are listed in [requirements.txt](/python/requirements.txt) and can be installed with
```bash
pip install -r requirements.txt
```
