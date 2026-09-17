# Data provenance

The two processed files are derived from the FINSABER S&P 500 price release:

- Source: <https://huggingface.co/datasets/finsaber-team/FINSABER-reproduce>
- Source license: Apache-2.0
- Coverage used here: 29 June 2010 to 28 June 2024, 3,524 trading dates
- Dataset 1: TSLA, NFLX, AMZN, MSFT, JNJ
- Dataset 2: MSFT, AMZN, JNJ, XOM, CAT

Regenerate the subsets after downloading the source CSV:

```bash
bash scripts/download_data.sh
python prepare_data.py --csv data/sp500_prices.csv --output data/portfolio_5stocks.pkl  --tickers TSLA NFLX AMZN MSFT JNJ --start 2010-06-29 --end 2024-06-28
python prepare_data.py --csv data/sp500_prices.csv --output data/portfolio_5stocks2.pkl --tickers MSFT AMZN JNJ XOM CAT  --start 2010-06-29 --end 2024-06-28
```

Integrity hashes for the files packaged with this release are recorded in `SHA256SUMS`.

