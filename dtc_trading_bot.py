import MetaTrader5 as mt5
import pandas as pd
import time
from datetime import datetime, timedelta
import requests

# --------------------- CONFIG ---------------------
ACCOUNT = 183438613
PASSWORD = "Suha@1920"
SERVER = "Exness-MT5Real25"
SYMBOL = "XAUUSDc"
LOT_SIZE = 0.5
STOPLOSS_PERCENT = 0.25  # Initial 1R risk in percentage (e.g., 0.25%)
TP_MULTIPLIERS = [1, 2, 3, 4]  # 4 take profit levels
INITIAL_TRADE_COUNT = len(TP_MULTIPLIERS)  # Used for breakeven trigger check

CHECK_INTERVAL = 60
MIN_TRADE_GAP = 5 * 60  # Minimum 5 min gap per symbol between trade batches
MIN_VOLUME = 50
MAX_OPEN_TRADES = 8  # Max number of simultaneous trades per symbol
MAGIC = 234000

# Trailing Stop Configuration
TRAILING_STEP_MULTIPLIER = 0.5  # Trailing distance = 0.5 * Initial SL Distance

# Telegram Bot Config
TELEGRAM_TOKEN = "8408644507:AAEsbH_e73cUEfYpKZ6q0lFB0AyG5lc5VKs"
TELEGRAM_CHAT_ID = "-1002915825734"

# 12-Hour Report Configuration
REPORT_INTERVAL_HOURS = 12
last_report_time = datetime.now() - timedelta(hours=REPORT_INTERVAL_HOURS)

# Deal Tracking for Immediate Closure Alerts
last_deal_check_time = datetime.now() - timedelta(minutes=2)

# EMA Lengths
EMA_LENGTHS = [30, 35, 40, 45, 50, 60]
MTF_FRAMES = {
    "15M": mt5.TIMEFRAME_M15,
    "30M": mt5.TIMEFRAME_M30,
    "1H": mt5.TIMEFRAME_H1,
    "4H": mt5.TIMEFRAME_H4,
    "1D": mt5.TIMEFRAME_D1
}
MTF_FAST = 20
MTF_SLOW = 50
CHART_TIMEFRAME = mt5.TIMEFRAME_M1

# Symbol state - Reset on a new trade batch
symbol_state = {
    SYMBOL: {
        "entry_price": None,
        "initial_sl_distance": None,  # Distance between Entry and SL (in price points)
        "breakeven_triggered": False,
        "last_trade_time": datetime.min
    }
}
PRICE_DIGITS = 2  # Default, will be updated after MT5 connection

# --------------------- CONNECT MT5 ---------------------
if not mt5.initialize():
    print("MT5 initialize() failed")
    mt5.shutdown()
    exit()

if not mt5.login(ACCOUNT, PASSWORD, SERVER):
    print("Failed to login")
    mt5.shutdown()
    exit()

print(f"✅ Connected to MT5 account: {ACCOUNT}")
symbol_info = mt5.symbol_info(SYMBOL)
if symbol_info is None:
    print(f"❌ Could not get symbol info for {SYMBOL}")
    mt5.shutdown()
    exit()
PRICE_DIGITS = symbol_info.digits
POINT = mt5.symbol_info(SYMBOL).point  # Key variable for precision/minimum distance

print(f"🚀 Bot running on {SYMBOL}. Price digits: {PRICE_DIGITS}, Point: {POINT}")


# --------------------- HELPER FUNCTIONS ---------------------
def send_telegram(message):
    """Send message to Telegram chat"""
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
    try:
        requests.post(url, data=payload)
    except Exception as e:
        print(f"❌ Failed to send Telegram message: {e}")


def get_rates(symbol, timeframe=mt5.TIMEFRAME_M1, n=100):
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, n)
    if rates is None or len(rates) == 0:
        return pd.DataFrame()
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df


def calculate_emas(df):
    for i, length in enumerate(EMA_LENGTHS):
        df[f'EMA{i + 1}'] = df['close'].ewm(span=length, adjust=False).mean()
    return df


def detect_trend(df):
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    bullish = all(latest[f'EMA{i + 1}'] > latest[f'EMA{i + 2}'] for i in range(len(EMA_LENGTHS) - 1))
    bearish = all(latest[f'EMA{i + 1}'] < latest[f'EMA{i + 2}'] for i in range(len(EMA_LENGTHS) - 1))
    prev_bullish = all(prev[f'EMA{i + 1}'] > prev[f'EMA{i + 2}'] for i in range(len(EMA_LENGTHS) - 1))
    prev_bearish = all(prev[f'EMA{i + 1}'] < prev[f'EMA{i + 2}'] for i in range(len(EMA_LENGTHS) - 1))
    return not prev_bullish and bullish, not prev_bearish and bearish


def mtf_majority_trend(symbol):
    bullish_count, bearish_count = 0, 0
    for name, tf in MTF_FRAMES.items():
        df = get_rates(symbol, timeframe=tf, n=MTF_SLOW + 10)
        if df.empty:
            continue
        df['fast'] = df['close'].ewm(span=MTF_FAST, adjust=False).mean()
        df['slow'] = df['close'].ewm(span=MTF_SLOW, adjust=False).mean()
        if df['fast'].iloc[-1] > df['slow'].iloc[-1]:
            bullish_count += 1
        else:
            bearish_count += 1
    return "bullish" if bullish_count > bearish_count else "bearish"


def is_volume_ok(symbol):
    df = get_rates(symbol, timeframe=CHART_TIMEFRAME, n=1)
    if df.empty:
        return False
    return df['tick_volume'].iloc[-1] >= MIN_VOLUME


def get_current_trades(symbol, magic=MAGIC):
    """Returns a list of trades opened by this bot (using MAGIC)."""
    return [pos for pos in mt5.positions_get(symbol=symbol) if pos.magic == magic]


def get_all_open_positions(symbol):
    """Returns all open positions for the symbol, regardless of MAGIC."""
    return mt5.positions_get(symbol=symbol)


def open_trades(symbol, direction, entry_price):
    """Opens multiple trades with volume splitting and price rounding."""
    global symbol_state

    # Check Max Open Trades
    all_open_positions = get_all_open_positions(symbol)
    if len(all_open_positions) + INITIAL_TRADE_COUNT > MAX_OPEN_TRADES:
        print(
            f"[{symbol}] ⚠️ Max total open trades ({len(all_open_positions)}) + new batch ({INITIAL_TRADE_COUNT}) would exceed {MAX_OPEN_TRADES} — skipping")
        return

    # 1. Volume Calculation
    volume_per_trade = LOT_SIZE / INITIAL_TRADE_COUNT

    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        print(f"[{symbol}] ❌ Failed to get tick info.")
        return

    # 2. Calculate Initial SL and Distance
    if direction == "buy":
        initial_sl = entry_price * (1 - STOPLOSS_PERCENT / 100)
        price = tick.ask
    else:
        initial_sl = entry_price * (1 + STOPLOSS_PERCENT / 100)
        price = tick.bid

    initial_sl = round(initial_sl, PRICE_DIGITS)
    initial_sl_distance = abs(entry_price - initial_sl)

    # Reset state for new trade batch
    symbol_state[symbol]["entry_price"] = entry_price
    symbol_state[symbol]["initial_sl_distance"] = initial_sl_distance
    symbol_state[symbol]["breakeven_triggered"] = False

    sl_price_for_telegram = initial_sl
    tp_levels = []
    trade_results = []

    for i, tp_mult in enumerate(TP_MULTIPLIERS):
        # Calculate TP for each R-multiple
        if direction == "buy":
            tp = entry_price * (1 + STOPLOSS_PERCENT * tp_mult / 100)
        else:
            tp = entry_price * (1 - STOPLOSS_PERCENT * tp_mult / 100)

        # Round TP price to symbol's precision
        tp = round(tp, PRICE_DIGITS)

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume_per_trade,
            "type": mt5.ORDER_TYPE_BUY if direction == "buy" else mt5.ORDER_TYPE_SELL,
            "price": price,
            "sl": initial_sl,
            "tp": tp,
            "deviation": 10,
            "magic": MAGIC,
            "comment": f"DTC {direction.upper()} TP{i + 1}",
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)
        tp_levels.append(tp)

        if result.retcode == mt5.TRADE_RETCODE_DONE:
            print(f"[{symbol}] ✅ {direction.upper()} TP{i + 1} opened (Ticket: {result.order})")
            trade_results.append(True)
        else:
            print(f"[{symbol}] ❌ TP{i + 1} failed: {result.comment} (Retcode: {result.retcode})")
            trade_results.append(False)

    if any(trade_results):
        msg = (
                f"📊 *{symbol} Trade Opened*\n\n"
                f"🟢 *Direction:* {direction.upper()}\n"
                f"💰 *Entry:* {entry_price:.{PRICE_DIGITS}f}\n"
                f"🛡️ *Stop Loss:* {sl_price_for_telegram:.{PRICE_DIGITS}f}\n"
                f"🎯 *Take Profits:*\n" +
                "\n".join([f"TP{idx + 1}: {tp_levels[idx]:.{PRICE_DIGITS}f}" for idx in range(len(tp_levels))])
        )
        send_telegram(msg)


def check_and_manage_trade(symbol):
    """
    Manages Breakeven (after TP1 hit) and Trailing Stop (after breakeven).
    (FIXED TRAILING SL LOGIC)
    """
    global symbol_state, POINT
    state = symbol_state[symbol]
    entry_price = state.get("entry_price")
    initial_sl_distance = state.get("initial_sl_distance")

    positions = get_current_trades(symbol)
    remaining_count = len(positions)

    if remaining_count == 0 or entry_price is None or initial_sl_distance is None:
        # Trade closed or no active state to manage
        state["entry_price"] = None
        state["initial_sl_distance"] = None
        state["breakeven_triggered"] = False
        return

    # Check position type (assume all remaining positions are the same type)
    is_buy = positions[0].type == mt5.ORDER_TYPE_BUY
    breakeven_sl = round(entry_price, PRICE_DIGITS)  # Breakeven floor

    # Calculate a minimum distance for SL modification (e.g., 5 points) to ensure the broker accepts the change
    MIN_SL_MOVE = 5 * POINT

    # 1. Breakeven Trigger (If at least one position closed)
    if remaining_count < INITIAL_TRADE_COUNT and not state["breakeven_triggered"]:
        print(f"[{symbol}] 🎯 TP1 hit/Position closed - remaining: {remaining_count}. Moving SL to breakeven.")

        for pos in positions:
            # Only update if the current SL is WORSE than breakeven
            move_required = (is_buy and pos.sl < breakeven_sl) or \
                            (not is_buy and pos.sl > breakeven_sl)

            if move_required:
                update_sl_request(symbol, pos.ticket, breakeven_sl, pos.tp, "BREAKEVEN")

        state["breakeven_triggered"] = True
        return

    # 2. Trailing Stop Logic (runs only if breakeven has been triggered)
    if state["breakeven_triggered"]:
        tick = mt5.symbol_info_tick(symbol)
        if tick is None: return

        current_price = tick.bid if is_buy else tick.ask

        # Trailing step is a fraction of the initial R distance
        trail_step = initial_sl_distance * TRAILING_STEP_MULTIPLIER

        for pos in positions:
            current_sl = pos.sl
            potential_new_sl = current_sl  # Start with current SL

            if is_buy:
                # Calculate SL for a long trade: Current Price - Trailing Step
                potential_new_sl = round(current_price - trail_step, PRICE_DIGITS)

                # Enforce Breakeven floor: SL cannot be below the entry price (BE)
                if potential_new_sl < breakeven_sl:
                    potential_new_sl = breakeven_sl

                # Only move if the potential new SL is BETTER than the current SL by at least MIN_SL_MOVE
                if potential_new_sl > current_sl + MIN_SL_MOVE:
                    update_sl_request(symbol, pos.ticket, potential_new_sl, pos.tp, "TRAIL")

            else:  # is_sell
                # Calculate SL for a short trade: Current Price + Trailing Step
                potential_new_sl = round(current_price + trail_step, PRICE_DIGITS)

                # Enforce Breakeven floor: SL cannot be above the entry price (BE)
                if potential_new_sl > breakeven_sl:
                    potential_new_sl = breakeven_sl

                # Only move if the potential new SL is BETTER than the current SL by at least MIN_SL_MOVE
                if potential_new_sl < current_sl - MIN_SL_MOVE:
                    update_sl_request(symbol, pos.ticket, potential_new_sl, pos.tp, "TRAIL")


def update_sl_request(symbol, ticket, new_sl, current_tp, reason):
    """Sends the MT5 request to modify the Stop Loss."""
    request = {
        "action": mt5.TRADE_ACTION_SLTP,
        "symbol": symbol,
        "position": ticket,
        "sl": new_sl,
        "tp": current_tp,
        "deviation": 10,
    }
    result = mt5.order_send(request)
    if result.retcode == mt5.TRADE_RETCODE_DONE:
        print(f"[{symbol}] ✅ SL moved to {new_sl:.{PRICE_DIGITS}f} ({reason}) for ticket {ticket}")
    else:
        # NOTE: This error message is critical for debugging
        print(f"[{symbol}] ❌ Failed SL move for {ticket}: {result.comment} (Retcode: {result.retcode})")


def assign_tp_sl_for_manual_trades(symbol):
    """Assigns the standard initial SL/TP to manual trades (magic != MAGIC) with no SL/TP set."""
    positions = mt5.positions_get(symbol=symbol)
    if not positions:
        return

    for pos in positions:
        if pos.magic != MAGIC and (pos.sl == 0 or pos.tp == 0):
            price = pos.price_open
            direction = "buy" if pos.type == mt5.ORDER_TYPE_BUY else "sell"

            sl = price * (1 - STOPLOSS_PERCENT / 100) if direction == "buy" else price * (1 + STOPLOSS_PERCENT / 100)
            tp1 = price * (1 + STOPLOSS_PERCENT * 1 / 100) if direction == "buy" else price * (
                        1 - STOPLOSS_PERCENT * 1 / 100)

            # Round for precision
            sl = round(sl, PRICE_DIGITS)
            tp1 = round(tp1, PRICE_DIGITS)

            request = {
                "action": mt5.TRADE_ACTION_SLTP,
                "symbol": symbol,
                "position": pos.ticket,
                "sl": sl,
                "tp": tp1,
                "deviation": 10,
                "comment": "MANUAL-ADJUST",
            }
            result = mt5.order_send(request)
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                print(f"[{symbol}] 🛠️ Manual trade TP1/SL assigned | Ticket={pos.ticket}")
            else:
                print(f"[{symbol}] ❌ Failed to assign TP/SL for manual trade | Ticket={pos.ticket}")


def send_account_report():
    """Calculates and sends an overall account report via Telegram."""
    global last_report_time

    account_info = mt5.account_info()
    if account_info is None:
        print("❌ Failed to get account info for report.")
        return

    balance = account_info.balance
    equity = account_info.equity
    profit = account_info.profit
    margin_free = account_info.margin_free

    # Get all history orders/deals since the last report time
    end_time = datetime.now()
    history_deals = mt5.history_deals_get(last_report_time, end_time)

    net_pnl_since_last_report = sum(d.profit for d in history_deals if d.entry == mt5.DEAL_ENTRY_OUT)

    net_emoji = "🟢" if net_pnl_since_last_report >= 0 else "🔴"
    pnl_emoji = "✅" if profit >= 0 else "❌"

    # Format the report message
    report_message = (
        f"⏰ *{REPORT_INTERVAL_HOURS}-Hour Account Report: {end_time.strftime('%Y-%m-%d %H:%M')}*\n\n"
        f"**📊 Current Status**\n"
        f"• Balance: ${balance:,.2f}\n"
        f"• Equity: ${equity:,.2f}\n"
        f"• Floating PnL: {pnl_emoji} ${profit:,.2f}\n"
        f"• Free Margin: ${margin_free:,.2f}\n\n"
        f"**📈 PnL Since Last Report ({REPORT_INTERVAL_HOURS}h)**\n"
        f"• Net Profit/Loss: {net_emoji} **${net_pnl_since_last_report:,.2f}**\n"
    )

    send_telegram(report_message)
    last_report_time = end_time
    print(f"[{SYMBOL}] ✅ Sent {REPORT_INTERVAL_HOURS}-hour account report.")


def check_and_alert_closed_deals(symbol):
    """
    Checks for closed deals since the last check and sends an immediate Telegram alert.
    """
    global last_deal_check_time

    current_time = datetime.now()

    # Fetch deals closed since the last check time
    start_time_adjusted = last_deal_check_time + timedelta(seconds=1)
    deals = mt5.history_deals_get(start_time_adjusted, current_time)

    if deals is None:
        print("❌ Failed to get deal history.")
        return

    for deal in deals:
        # Filter: 1. Must be a closure (DEAL_ENTRY_OUT) 2. Must belong to this bot (MAGIC)
        if deal.entry == mt5.DEAL_ENTRY_OUT and deal.magic == MAGIC:

            pnl_emoji = "✅" if deal.profit >= 0 else "❌"
            pnl_color = "PROFIT" if deal.profit >= 0 else "LOSS"

            # Infer the reason for closure based on profit/comment (simplified/robust inference)
            comment = deal.comment.upper() if deal.comment else ""

            if deal.profit > 0 and "TP" in comment:
                reason = "Take Profit (TP)"
            elif deal.profit > 0 and ("TRAIL" in comment or "BREAKEVEN" in comment):
                reason = "Trailing Stop/Profit Lock"
            elif deal.profit <= 0 and ("SL" in comment or "BREAKEVEN" in comment):
                reason = "Stop Loss (SL) / Breakeven"
            else:
                reason = "Closed (Other)"

            msg = (
                f"{pnl_emoji} *Position Closed for {symbol}*\n\n"
                f"Closure Reason: *{reason}*\n"
                f"Ticket: `{deal.ticket}`\n"
                f"Volume: `{deal.volume} LOTS`\n"
                f"**Realized {pnl_color}: ${deal.profit:,.2f}**"
            )
            send_telegram(msg)
            print(f"[{symbol}] Alerted on closed deal {deal.ticket}: {reason} ({deal.profit:,.2f})")

    # Update the last checked time to the end of the current period
    last_deal_check_time = current_time


# --------------------- MAIN LOOP ---------------------
try:
    while True:
        now = datetime.now()
        print(f"\n=== Checking {SYMBOL} signals at {now.strftime('%H:%M:%S')} ===")

        # 1. Immediate Closure Alert Check (Must run first)
        check_and_alert_closed_deals(SYMBOL)

        # 2. 12-Hour Report Check
        if (now - last_report_time).total_seconds() >= REPORT_INTERVAL_HOURS * 3600:
            send_account_report()

        current_bot_positions = get_current_trades(SYMBOL)

        # 3. Manual Trade SL/TP Assignment
        assign_tp_sl_for_manual_trades(SYMBOL)

        # 4. Trade Management (runs only if bot trades are open)
        if current_bot_positions:
            print(f"[{SYMBOL}] {len(current_bot_positions)} trade(s) open. Running management...")
            check_and_manage_trade(SYMBOL)  # Includes Breakeven and Trailing SL

            # Skip signal check when trades are open to focus on management
            time.sleep(CHECK_INTERVAL)
            continue

        # 5. Entry Logic (runs only if no bot trades are open)

        if (now - symbol_state[SYMBOL]["last_trade_time"]).total_seconds() < MIN_TRADE_GAP:
            print(f"[{SYMBOL}] ⏳ Skipping — last trade too recent.")
            time.sleep(CHECK_INTERVAL)
            continue

        volume_check = get_rates(SYMBOL, CHART_TIMEFRAME, 1)['tick_volume'].iloc[-1]
        if volume_check < MIN_VOLUME:
            print(f"[{SYMBOL}] ⚠️ Volume ({volume_check}) too low (<{MIN_VOLUME}) — skipping trade")
            time.sleep(CHECK_INTERVAL)
            continue

        majority = mtf_majority_trend(SYMBOL)
        print(f"[{SYMBOL}] Market trend (MTF majority): {majority}")

        df = get_rates(SYMBOL, timeframe=CHART_TIMEFRAME)
        if df.empty or len(df) < max(EMA_LENGTHS) + 2:
            time.sleep(CHECK_INTERVAL)
            continue

        df = calculate_emas(df)
        long_signal, short_signal = detect_trend(df)
        latest_close = df['close'].iloc[-1]

        if long_signal and majority == "bullish":
            print(f"[{SYMBOL}] 🟢 LONG confirmed at {latest_close}")
            open_trades(SYMBOL, "buy", latest_close)
            symbol_state[SYMBOL]["last_trade_time"] = now
        elif short_signal and majority == "bearish":
            print(f"[{SYMBOL}] 🔴 SHORT confirmed at {latest_close}")
            open_trades(SYMBOL, "sell", latest_close)
            symbol_state[SYMBOL]["last_trade_time"] = now
        else:
            print(f"[{SYMBOL}] No trade signal this cycle")

        time.sleep(CHECK_INTERVAL)

except KeyboardInterrupt:
    print("\n🛑 Bot stopped manually")
    send_telegram("🛑 DTC Bot stopped manually")
finally:
    mt5.shutdown()
