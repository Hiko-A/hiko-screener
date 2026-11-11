"""
ヒコ 観察リストB用スクリーナー
ミネルヴィニのトレンドテンプレートを参考にした8条件+足切りで、
S&P500とS&P中型株400の構成銘柄を毎週チェックし、結果をCSVで保存する。

条件(すべて日次の終値ベース。配当・分割調整後)
 1. 株価が150日・200日移動平均線より上
 2. 150日線が200日線より上
 3. 200日線が1か月以上上向き(今日の200日線 > 21営業日前の200日線)
 4. 50日線が150日線・200日線より上
 5. 株価が50日線より上
 6. 株価が52週安値から30%以上上
 7. 株価が52週高値から25%以内
 8. 過去12か月(252営業日)の騰落率がSPYを上回る(RSレーティングの代用)
足切り: 時価総額20億ドル以上、50日平均出来高50万株以上
並べ順: 過去12か月の騰落率が高い順(1位が観察Bの候補)

※過去データの機械的な絞り込みであり、売買の推奨ではありません。
"""

import io
import os
import sys
import time
from datetime import datetime, timezone

import pandas as pd
import requests
import yfinance as yf

MIN_MARKET_CAP = 2_000_000_000   # 時価総額の足切り(ドル)
MIN_AVG_VOLUME = 500_000         # 50日平均出来高の足切り(株)
BATCH = 100                      # 一度に取得する銘柄数
OUT_DIR = "results"

UA = {"User-Agent": "Mozilla/5.0 (hiko-screener; GitHub Actions)"}
WIKI = {
    "S&P500": "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
    "S&P400": "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies",
}


def get_universe():
    """S&P500とS&P400の構成銘柄をWikipediaから取得する。"""
    tickers = {}
    for name, url in WIKI.items():
        try:
            html = requests.get(url, headers=UA, timeout=30).text
            tables = pd.read_html(io.StringIO(html))
            table = next(t for t in tables if "Symbol" in t.columns)
            for sym in table["Symbol"].astype(str):
                tickers[sym.strip().replace(".", "-")] = name
            print(f"{name}: {len(table)}銘柄")
        except Exception as e:  # 片方が取れなくても続ける
            print(f"{name}の取得に失敗: {e}", file=sys.stderr)
    if not tickers:
        sys.exit("銘柄リストを取得できませんでした")
    return tickers


def download(tickers):
    """日次株価(2年分)をまとめて取得する。"""
    frames = []
    for i in range(0, len(tickers), BATCH):
        chunk = tickers[i:i + BATCH]
        for attempt in range(3):
            try:
                df = yf.download(chunk, period="2y", interval="1d",
                                 auto_adjust=True, group_by="ticker",
                                 threads=True, progress=False)
                frames.append(df)
                break
            except Exception as e:
                print(f"取得失敗({attempt + 1}回目): {e}", file=sys.stderr)
                time.sleep(10)
        time.sleep(2)
    return pd.concat(frames, axis=1) if frames else pd.DataFrame()


def evaluate(df, spy_ret):
    """1銘柄分のデータ(Close/High/Low/Volume)から各条件を判定する。"""
    df = df.dropna(subset=["Close"])
    if len(df) < 274:  # 200日線+21日分と252日分に足りない
        return None
    close, high, low, vol = df["Close"], df["High"], df["Low"], df["Volume"]
    sma50 = close.rolling(50).mean()
    sma150 = close.rolling(150).mean()
    sma200 = close.rolling(200).mean()
    c = close.iloc[-1]
    s50, s150, s200 = sma50.iloc[-1], sma150.iloc[-1], sma200.iloc[-1]
    s200_past = sma200.iloc[-22]
    low52 = low.iloc[-252:].min()
    high52 = high.iloc[-252:].max()
    ret12 = (c / close.iloc[-253] - 1) * 100
    avg_vol = vol.iloc[-50:].mean()

    row = {
        "date": df.index[-1].strftime("%Y-%m-%d"),
        "close": round(c, 2),
        "sma50": round(s50, 2),
        "sma150": round(s150, 2),
        "sma200": round(s200, 2),
        "sma200_1m_ago": round(s200_past, 2),
        "low_52w": round(low52, 2),
        "high_52w": round(high52, 2),
        "above_low_pct": round((c / low52 - 1) * 100, 1),
        "below_high_pct": round((1 - c / high52) * 100, 1),
        "ret_12m_pct": round(ret12, 1),
        "spy_ret_12m_pct": round(spy_ret, 1),
        "avg_vol_50d": int(avg_vol),
        "c1": bool(c > s150 and c > s200),
        "c2": bool(s150 > s200),
        "c3": bool(s200 > s200_past),
        "c4": bool(s50 > s150 and s50 > s200),
        "c5": bool(c > s50),
        "c6": bool(c >= low52 * 1.30),
        "c7": bool(c >= high52 * 0.75),
        "c8": bool(ret12 > spy_ret),
        "vol_ok": bool(avg_vol >= MIN_AVG_VOLUME),
    }
    row["passed_8"] = all(row[f"c{k}"] for k in range(1, 9))
    return row


def pick(data, sym):
    try:
        return data[sym][["Close", "High", "Low", "Volume"]]
    except Exception:
        return None


def market_cap(sym):
    for attempt in range(3):
        try:
            return yf.Ticker(sym).fast_info["market_cap"]
        except Exception:
            time.sleep(3)
    return None


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    universe = get_universe()
    symbols = sorted(universe)
    data = download(symbols + ["SPY"])

    spy = pick(data, "SPY")
    if spy is None or spy["Close"].dropna().shape[0] < 253:
        sys.exit("SPYのデータを取得できませんでした")
    spy_close = spy["Close"].dropna()
    spy_ret = (spy_close.iloc[-1] / spy_close.iloc[-253] - 1) * 100

    rows, missing = [], []
    for sym in symbols:
        d = pick(data, sym)
        r = evaluate(d, spy_ret) if d is not None else None
        if r is None:
            missing.append(sym)
            continue
        r = {"ticker": sym, "index": universe[sym], **r}
        rows.append(r)

    all_df = pd.DataFrame(rows)
    # 8条件と出来高を満たした銘柄だけ時価総額を確認する
    cand = all_df[all_df["passed_8"] & all_df["vol_ok"]].copy()
    cand["market_cap_b"] = [
        round((market_cap(s) or 0) / 1e9, 1) for s in cand["ticker"]
    ]
    cand["cap_ok"] = cand["market_cap_b"] >= MIN_MARKET_CAP / 1e9
    passed = (cand[cand["cap_ok"]]
              .sort_values("ret_12m_pct", ascending=False)
              .reset_index(drop=True))
    passed.insert(0, "rank", passed.index + 1)

    asof = all_df["date"].max() if len(all_df) else datetime.now(timezone.utc).strftime("%Y-%m-%d")
    passed.to_csv(f"{OUT_DIR}/passed_{asof}.csv", index=False)
    passed.to_csv(f"{OUT_DIR}/latest.csv", index=False)
    all_df.to_csv(f"{OUT_DIR}/all_latest.csv", index=False)

    lines = [
        f"# スクリーニング結果({asof}終値)",
        "",
        f"- 対象: S&P500+S&P400 {len(symbols)}銘柄(データ取得できたのは{len(all_df)}銘柄、取得できなかったのは{len(missing)}銘柄)",
        f"- SPYの過去12か月騰落率: {spy_ret:+.1f}%",
        f"- 8条件をすべて満たした銘柄: {int(all_df['passed_8'].sum())}",
        f"- さらに足切り(時価総額{MIN_MARKET_CAP / 1e8:.0f}億ドル以上、50日平均出来高{MIN_AVG_VOLUME:,}株以上)も満たした銘柄: {len(passed)}",
        "",
        "## 上位10銘柄(過去12か月騰落率の高い順。1位が観察Bの候補)",
        "",
        "| 順位 | 銘柄 | 終値 | 12か月騰落率 | 52週高値から | 52週安値から | 時価総額(億ドル) |",
        "|---|---|---|---|---|---|---|",
    ]
    for _, r in passed.head(10).iterrows():
        lines.append(
            f"| {r['rank']} | {r['ticker']} | {r['close']} | {r['ret_12m_pct']:+.1f}% | "
            f"−{r['below_high_pct']}% | +{r['above_low_pct']}% | {r['market_cap_b'] * 10:.0f} |"
        )
    lines += ["", "※ミネルヴィニのトレンドテンプレートを参考にした条件による機械的な絞り込みです。売買の推奨ではありません。"]
    if missing:
        lines += ["", "取得できなかった銘柄: " + ", ".join(missing[:50]) + (" ほか" if len(missing) > 50 else "")]
    with open(f"{OUT_DIR}/summary.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
