"""
main.py
-------

Usage:
    python main.py                                          # autonomous
    python main.py --asset_class crypto                    # asset-scoped
    python main.py --asset_class equities --theme "AI"     # theme-guided
    python main.py --asset_class mixed --tickers NVDA BTC  # ticker-first
"""

import argparse
from dotenv import load_dotenv
from crew import QuantCrew

load_dotenv()


def main():
    parser = argparse.ArgumentParser(description="Quant Idea Pipeline")
    parser.add_argument(
        "--asset_class",
        type=str,
        choices=["equities", "crypto", "macro", "mixed"],
        help="Asset class scope (default: auto-detect)",
    )
    parser.add_argument(
        "--tickers",
        nargs="+",
        help="Tickers to analyse (e.g. --tickers NVDA BTC SOL)",
    )
    parser.add_argument(
        "--theme",
        type=str,
        help="Theme or sector to explore (e.g. --theme 'AI infrastructure')",
    )
    args = parser.parse_args()

    crew = QuantCrew()
    crew.run(
        asset_class=args.asset_class,
        tickers=args.tickers,
        theme=args.theme,
    )


if __name__ == "__main__":
    main()