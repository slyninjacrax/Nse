from datetime import datetime, timezone, timedelta

import pandas as pd
import requests
import streamlit as st

try:
    from streamlit_autorefresh import st_autorefresh
    AUTO_REFRESH_AVAILABLE = True
except ImportError:
    AUTO_REFRESH_AVAILABLE = False


# -----------------------
# Helpers
# -----------------------
def pcr_value(put_oi, call_oi):
    if not call_oi or call_oi == 0:
        return None
    return float(put_oi) / float(call_oi)


def sentiment_from_pcr(p):
    if p is None:
        return "N/A"
    if p > 1.5:
        return "BULLISH"
    elif p < 1.0:
        return "BEARISH"
    elif 1.0 <= p <= 1.2:
        return "SIDEWAYS"
    else:
        return "MILD BULLISH"


def get_iv_status(sentiment, iv):
    if not iv or iv == 0:
        return "N/A"

    if sentiment in ["BULLISH", "MILD BULLISH"]:
        if iv > 21:
            return "HIGH"
        elif iv < 19:
            return "LOW"
        else:
            return "NORMAL"

    elif sentiment == "BEARISH":
        if iv > 38:
            return "HIGH"
        elif iv < 21:
            return "LOW"
        else:
            return "NORMAL"

    else:
        if iv > 20:
            return "HIGH"
        elif iv < 12:
            return "LOW"
        else:
            return "NORMAL"


def to_lakh(x):
    try:
        return float(x) / 100000.0
    except Exception:
        return 0.0


def get_ist_time():
    return datetime.now(timezone.utc) + timedelta(hours=5, minutes=30)


def get_day_greeting_and_symbol():
    hour = get_ist_time().hour
    if 5 <= hour < 12:
        return "Good morning", "🌤️"
    elif 12 <= hour < 17:
        return "Good afternoon", "☀️"
    elif 17 <= hour < 21:
        return "Good evening", "🌆"
    else:
        return "Good night", "🌙"


def extract_cookie(raw_text: str) -> str:
    if not raw_text:
        return ""
    text = raw_text.strip()
    if text.lower().startswith("cookie:"):
        return text.split(":", 1)[1].strip()
    return text


def build_headers(cookie: str) -> dict:
    return {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/133.0.0.0 Safari/537.36"
        ),
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-IN,en;q=0.9,en-US;q=0.8",
        "Referer": "https://www.nseindia.com/option-chain",
        "Origin": "https://www.nseindia.com",
        "Connection": "keep-alive",
        "Cookie": cookie,
    }


def infer_symbol_from_url(url: str) -> str:
    if "symbol=" in url:
        try:
            return url.split("symbol=", 1)[1].split("&", 1)[0]
        except Exception:
            return "OPTION"
    return "OPTION"


# -----------------------
# Data Processing
# -----------------------
def process_chain_data(j):
    records = j.get("records", {})
    data = records.get("data", [])

    if not data:
        data = j.get("filtered", {}).get("data", [])
    if not data and isinstance(j.get("data"), list):
        data = j.get("data")

    expiries = records.get("expiryDates", [])
    if not expiries:
        expiries = sorted(list(set([r.get("expiryDate") for r in data if r.get("expiryDate")])))

    underlying = records.get("underlyingValue", 0)
    if not underlying:
        for r in data:
            ce = r.get("CE", {})
            if ce.get("underlyingValue"):
                underlying = ce.get("underlyingValue")
                break

    return data, expiries, underlying


def build_tables(data, target_expiry, underlying, window_size=8):
    rows = [r for r in data if r.get("expiryDate") == target_expiry]
    if not rows:
        rows = data

    strikes = sorted(list(set([r.get("strikePrice") for r in rows if r.get("strikePrice")])))
    cols = ["STRIKE", "LAST", "OPEN_INT", "CHG_IN_OI", "IV"]
    empty_df = pd.DataFrame(columns=cols + ["ODIN_%"])

    if not strikes:
        return empty_df, empty_df, 0, []

    atm_strike = min(strikes, key=lambda x: abs(x - underlying)) if underlying else strikes[len(strikes) // 2]
    atm_index = strikes.index(atm_strike)
    min_idx = max(0, atm_index - window_size)
    max_idx = min(len(strikes), atm_index + window_size + 1)
    target_strikes = strikes[min_idx:max_idx]

    call_data, put_data = [], []
    for row in rows:
        sp = row.get("strikePrice")
        if sp not in target_strikes:
            continue

        ce, pe = row.get("CE", {}), row.get("PE", {})

        call_data.append({
            "STRIKE": sp,
            "LAST": ce.get("lastPrice", 0),
            "OPEN_INT": ce.get("openInterest", 0),
            "CHG_IN_OI": ce.get("changeinOpenInterest", 0),
            "IV": ce.get("impliedVolatility", 0)
        })
        put_data.append({
            "STRIKE": sp,
            "LAST": pe.get("lastPrice", 0),
            "OPEN_INT": pe.get("openInterest", 0),
            "CHG_IN_OI": pe.get("changeinOpenInterest", 0),
            "IV": pe.get("impliedVolatility", 0)
        })

    dfc = pd.DataFrame(call_data, columns=cols).sort_values("STRIKE")
    dfp = pd.DataFrame(put_data, columns=cols).sort_values("STRIKE")

    ctot, ptot = dfc["OPEN_INT"].sum(), dfp["OPEN_INT"].sum()
    dfc["ODIN_%"] = (dfc["OPEN_INT"] / ctot * 100).fillna(0) if ctot > 0 else 0
    dfp["ODIN_%"] = (dfp["OPEN_INT"] / ptot * 100).fillna(0) if ptot > 0 else 0

    return dfc, dfp, atm_strike, target_strikes


def render_html_table(df, title, theme, atm_strike):
    head_bg = "#1b5e20" if theme == "call" else "#b71c1c"
    rows_html = ""

    for _, row in df.iterrows():
        sp = int(row["STRIKE"]) if pd.notna(row["STRIKE"]) else 0
        is_atm = (sp == atm_strike)

        bg_color, font_weight = ("#ffff00", "bold") if is_atm else ("white", "normal")
        text_color = "black" if is_atm else ("#1b5e20" if theme == "call" else "#b71c1c")

        last_val = row["LAST"] if pd.notna(row["LAST"]) else 0
        oi_val = row["OPEN_INT"] if pd.notna(row["OPEN_INT"]) else 0
        chg_val = row["CHG_IN_OI"] if pd.notna(row["CHG_IN_OI"]) else 0
        odin_val = row["ODIN_%"] if pd.notna(row["ODIN_%"]) else 0

        if chg_val < 0:
            chg_bg = "#ffebee"
            chg_text = "#b71c1c"
        elif chg_val > 0:
            chg_bg = "#e8f5e9"
            chg_text = "#1b5e20"
        else:
            chg_bg = "white"
            chg_text = "black"

        if odin_val >= 10.0:
            odin_bg = "#fff9c4"
            odin_weight = "bold"
        else:
            odin_bg = "white"
            odin_weight = "normal"

        rows_html += f"<tr style='background-color: {bg_color}; font-weight: {font_weight}; text-align: center; color: black;'>"
        rows_html += f"<td style='padding: 4px; border: 1px solid #ddd;'>{sp}</td>"
        rows_html += f"<td style='padding: 4px; border: 1px solid #ddd;'>{last_val:.2f}</td>"
        rows_html += f"<td style='padding: 4px; border: 1px solid #ddd; color: {text_color}; font-weight: bold;'>{oi_val:,.0f}</td>"
        rows_html += f"<td style='padding: 4px; border: 1px solid #ddd; background-color: {chg_bg}; color: {chg_text}; font-weight: bold;'>{chg_val:,.0f}</td>"
        rows_html += f"<td style='padding: 4px; border: 1px solid #ddd; background-color: {odin_bg}; font-weight: {odin_weight};'>{odin_val:.1f}%</td>"
        rows_html += "</tr>"

    return f"""
    <div style="border: 2px solid {head_bg}; margin-bottom: 20px;">
        <div style="background-color: {head_bg}; color: white; text-align: center; font-weight: bold; padding: 8px;">{title}</div>
        <table style="width: 100%; border-collapse: collapse; font-family: sans-serif; font-size: 14px;">
            <tr style="background-color: #f0f0f0; color: {head_bg}; font-weight: bold;">
                <th style="padding: 6px; border: 1px solid #ddd;">STRIKE</th>
                <th style="padding: 6px; border: 1px solid #ddd;">LAST</th>
                <th style="padding: 6px; border: 1px solid #ddd;">OPEN INT</th>
                <th style="padding: 6px; border: 1px solid #ddd;">CHANGE IN OI</th>
                <th style="padding: 6px; border: 1px solid #ddd;">Odin %</th>
            </tr>
            {rows_html}
        </table>
    </div>
    """


# -----------------------
# Streamlit UI
# -----------------------
st.set_page_config(page_title="NSE Option Chain Dashboard", layout="wide")

clean_cols = ["Time", "PCR", "ATM IV", "Sentiment", "Underlying"]

if "pcr_log" not in st.session_state:
    st.session_state.pcr_log = pd.DataFrame(columns=clean_cols)

st.title("NSE Option Chain Dashboard")

greeting, symbol_emoji = get_day_greeting_and_symbol()
ist_display_time = get_ist_time().strftime("%H:%M:%S")

st.markdown(
    f"""
    <div style="height: 22px;"></div>
    <div style="font-size: 44px; font-weight: 700; line-height: 1.15; color: #333;">
        {symbol_emoji} {greeting}, Dr Chopra
    </div>
    <div style="height: 16px;"></div>
    <div style="font-size: 20px; font-weight: 500; color: #555;">
        IST Time: {ist_display_time}
    </div>
    <div style="height: 22px;"></div>
    """,
    unsafe_allow_html=True
)

with st.sidebar:
    st.header("URL + Cookie")

    base_url = st.text_input(
        "Request URL",
        value="https://www.nseindia.com/api/option-chain-v3?type=Indices&symbol=NIFTY"
    )

    cookie_input = st.text_area(
        "Cookie only",
        height=220,
        placeholder="Paste only the cookie string here, or one line starting with Cookie:"
    )

    if st.button("Clear Log Manually"):
        st.session_state.pcr_log = pd.DataFrame(columns=clean_cols)
        st.rerun()

    refresh_rate = st.selectbox("Auto Refresh", ["Off", "1 min", "3 min", "5 min"], index=1)

if refresh_rate != "Off":
    if AUTO_REFRESH_AVAILABLE:
        intervals = {"1 min": 60000, "3 min": 180000, "5 min": 300000}
        refresh_count = st_autorefresh(
            interval=intervals[refresh_rate],
            key="datarefresh"
        )
        st.sidebar.caption(f"Auto refresh active · count: {refresh_count}")
    else:
        st.sidebar.error("Auto refresh package not loaded.")

cookie_value = extract_cookie(cookie_input)

if not base_url or not cookie_value:
    st.warning("Paste the Request URL and the Cookie to begin.")
    st.stop()

headers_dict = build_headers(cookie_value)
symbol_name = infer_symbol_from_url(base_url)

try:
    with st.spinner("Fetching live data..."):
        r = requests.get(base_url, headers=headers_dict, timeout=30)

        if r.status_code != 200:
            st.error(f"NSE returned HTTP {r.status_code}. Cookie is likely stale.")
            st.stop()

        try:
            raw_data = r.json()
        except Exception:
            st.error("NSE did not return valid JSON. Paste a fresh cookie.")
            st.stop()

    data, expiries, underlying = process_chain_data(raw_data)

    if not data:
        st.error("No option chain data found. Use a fresh cookie or matching request URL.")
        st.stop()

    target_expiry = st.sidebar.selectbox("Select Expiry", expiries, index=0) if expiries else "N/A"

    df_call, df_put, atm_strike, _ = build_tables(data, target_expiry, underlying, window_size=8)

    total_call_oi = df_call["OPEN_INT"].sum()
    total_put_oi = df_put["OPEN_INT"].sum()
    diff_oi = total_put_oi - total_call_oi

    window_pcr = pcr_value(total_put_oi, total_call_oi)
    sentiment = sentiment_from_pcr(window_pcr)

    atm_call_row = df_call[df_call["STRIKE"] == atm_strike]
    atm_put_row = df_put[df_put["STRIKE"] == atm_strike]

    atm_ce_iv = atm_call_row["IV"].values[0] if not atm_call_row.empty else 0
    atm_pe_iv = atm_put_row["IV"].values[0] if not atm_put_row.empty else 0
    atm_iv = (atm_ce_iv + atm_pe_iv) / 2 if atm_ce_iv > 0 and atm_pe_iv > 0 else max(atm_ce_iv, atm_pe_iv)

    iv_tag = get_iv_status(sentiment, atm_iv)

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Underlying Value", f"{underlying:.2f}" if underlying else "N/A")
    col2.metric("ATM Strike", f"{atm_strike}")
    col3.metric("Window PCR", f"{window_pcr:.2f}" if window_pcr else "N/A", sentiment)
    col4.metric("ATM Implied Volatility", f"{atm_iv:.1f}" if atm_iv else "N/A", iv_tag, delta_color="off")
    col5.metric("OI Difference", f"{diff_oi:,.0f}")

    left, right = st.columns(2)

    with left:
        st.markdown(render_html_table(df_call, f"{symbol_name} CALL OPTION", "call", atm_strike), unsafe_allow_html=True)
        st.markdown(f"**Total Call OI (Window):** {to_lakh(total_call_oi):.2f} Lakh")

    with right:
        st.markdown(render_html_table(df_put, f"{symbol_name} PUT OPTION", "put", atm_strike), unsafe_allow_html=True)
        st.markdown(f"**Total Put OI (Window):** {to_lakh(total_put_oi):.2f} Lakh")

    ist_time_str = get_ist_time().strftime("%H:%M:%S")

    if st.session_state.pcr_log.empty or st.session_state.pcr_log.iloc[-1]["Time"] != ist_time_str:
        new_row = pd.DataFrame([{
            "Time": ist_time_str,
            "PCR": round(window_pcr, 2) if window_pcr else None,
            "ATM IV": round(atm_iv, 2) if atm_iv else None,
            "Sentiment": sentiment,
            "Underlying": underlying
        }])
        st.session_state.pcr_log = pd.concat([st.session_state.pcr_log, new_row], ignore_index=True).tail(50)

    st.markdown("---")
    st.markdown("### 📊 INTRADAY MACRO TREND (IST Market Time)")

    def highlight_sentiment(val):
        color = "green" if val == "BULLISH" else "red" if val == "BEARISH" else "orange" if val == "MILD BULLISH" else "gray"
        return f"color: {color}; font-weight: bold;"

    log_col, chart_col = st.columns([1, 1])

    with log_col:
        st.dataframe(
            st.session_state.pcr_log.style.map(highlight_sentiment, subset=["Sentiment"]),
            use_container_width=True,
            hide_index=True,
            height=300
        )

    with chart_col:
        if len(st.session_state.pcr_log) > 1:
            chart_data = st.session_state.pcr_log[["Time", "PCR"]].copy().set_index("Time")
            st.line_chart(chart_data, height=300)

except requests.exceptions.RequestException as e:
    st.error(f"Network/request error: {e}")
except Exception as e:
    st.error(f"Error fetching data: {e}")
