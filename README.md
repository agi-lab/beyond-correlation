# beyond-correlation
By Benjamin Avanzi, Guillaume Boglioni Beaulieu, Pierre Lafaye de Micheaux, Ho Ming Lee, Bernard Wong, and Rui Zhou.

## Introduction
This repository contains the code for the paper *Beyond pairwise correlation: capturing nonlinear and higher-order dependence with distance statistics*, submitted to the [2026 All Actuaries Summit](https://www.actuaries.asn.au/all-actuaries-summit-2026). The paper aims to introduce distance-based dependence statistics for testing and modelling dependence structures between random variables and/or random vectors, as complementary tools to the correlation coefficient. We illustrate these statistics using a range of real-world and synthetic datasets. A link to the paper will be provided once it is finalised and uploaded to arXiv.

The code for visualising the motivating examples, together with illustrations of the computation of the distance-based dependence statistics, is available [here](https://agi-lab.github.io/beyond-correlation/).

## Datasets
We also provide the datasets used for illustration in this repository. They can be found in the [`data`](data) folder.

The dataset used in this paper are:
- [World demographics data](/data/worlddemographics.csv) (CIA, 2020): This dataset contains the birth and death rates from different countries in the first trimester of 2020. It can be loaded in `R` using the `HellCor` package via:
  ```bash
  library(HellCor)
  data("worlddemographics")
  ```
- [pg15training](/data/pg15training.rda) (Dutang & Charpentier, 2026): This dataset contains 100,000 third-party liability policies for a private motor insurance product in France. It is one of the datasets in the `R` package `CASdatasets`.
  ```bash
  library(CASdatasets)
  data("pg15training")
  ```
- [S&P-500 data](/data/sp500_monthly_log_return.csv): This dataset contains the monthly stock returns downloaded from Yahoo Finance over the period 1926-01-01 to 2026-03-20. It was downloaded using the following code in `R`:
  ```bash
  getSymbols("^GSPC", src = "yahoo", from = "1926-01-01")
  price <- Cl(GSPC)
  ret <- monthlyReturn(price, type = "log")
  ```

## Running the demonstration locally
For those interested in running the code themselves, the main files are [`r_illust.qmd`](r_illust.qmd) and [`jdcov.qmd`](jdcov.qmd). The file [`r_illust.qmd`](r_illust.qmd) is written in `R` and contains the illustrations for the [Hellinger correlation](https://www.tandfonline.com/doi/epdf/10.1080/01621459.2020.1791132), [distance covariance](https://projecteuclid.org/journals/annals-of-applied-statistics/volume-3/issue-4/Brownian-distance-covariance/10.1214/09-AOAS312.full), and the [auto-distance correlation function](https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1467-9892.2011.00780.x).

For the [joint distance covariance](https://www.tandfonline.com/doi/full/10.1080/01621459.2018.1513364), you can run [`jdcov.qmd`](jdcov.qmd), which is written in Python. The code was executed in Quarto with Python 3.14.3. Required packages are listed in [`requirements.txt`](/python/requirements.txt) and can be installed with
```bash
pip install -r requirements.txt
```
