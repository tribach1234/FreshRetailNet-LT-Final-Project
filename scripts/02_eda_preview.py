"""Quick preview charts from Step-1 daily aggregate files; run only after Step 1."""
import csv
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / 'reports'
FIGURES = REPORTS / 'figures'


def load_daily(which):
    with (REPORTS / f'daily_{which}.csv').open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    rows.sort(key=lambda x: x['dt'])
    return rows


def main():
    FIGURES.mkdir(exist_ok=True, parents=True)
    rows = load_daily('train')
    dates = [datetime.strptime(r['dt'], '%Y-%m-%d') for r in rows]
    sales = [float(r['total_normalized_sales']) for r in rows]
    rates = [float(r['stockout_rate']) * 100 for r in rows]
    for values, label, filename in [
        (sales, 'Total normalized sales', 'daily_sales.png'),
        (rates, 'Share of rows with stockout (%)', 'daily_stockout.png'),
    ]:
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(dates, values, linewidth=1.25)
        ax.set(xlabel='Date', ylabel=label, title=label + ' - train')
        ax.grid(alpha=0.25)
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(FIGURES / filename, dpi=160)
        plt.close(fig)
    print('EDA preview charts:', FIGURES)


if __name__ == '__main__':
    main()
