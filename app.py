import math
from io import BytesIO

import pandas as pd
import pydeck as pdk
import streamlit as st
import streamlit_authenticator as stauth  # <--- tambahan

# =========================
#  DISTANCE & DATA HELPERS
# =========================


def haversine_distance_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    )
    return 2 * R * math.asin(math.sqrt(a))


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]

    col_map = {
        "shipment_code": "shipment_code",
        "shipment": "shipment_code",
        "delivery_lat": "delivery_latitude",
        "delivery_latitude": "delivery_latitude",
        "delivery_latitude_deg": "delivery_latitude",
        "delivery_lng": "delivery_longitude",
        "delivery_long": "delivery_longitude",
        "delivery_longitude": "delivery_longitude",
        "actual_lat": "dropoff_latitude",
        "actual_latitude": "dropoff_latitude",
        "actual_dropoff_lat": "dropoff_latitude",
        "dropoff_lat": "dropoff_latitude",
        "actual_lng": "dropoff_longitude",
        "actual_longitude": "dropoff_longitude",
        "actual_dropoff_long": "dropoff_longitude",
        "dropoff_lng": "dropoff_longitude",
        "dropoff_long": "dropoff_longitude",
        "number_account": "number_account",
        "no_account": "number_account",
        "account": "number_account",
        "customer_code": "number_account",
    }

    df = df.rename(columns={c: col_map[c] for c in df.columns if c in col_map})

    required = [
        "shipment_code",
        "delivery_latitude",
        "delivery_longitude",
        "dropoff_latitude",
        "dropoff_longitude",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    cols = required + (["number_account"] if "number_account" in df.columns else [])
    df = df[cols]

    for c in required[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def classify_account_type(number_account) -> str:
    if pd.isna(number_account):
        return "Corporate/Other"
    s = str(number_account)
    if s.startswith("EM.") or "@" in s:
        return "Retail"
    return "Corporate/Other"


def compute_distances(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().dropna(
        subset=[
            "delivery_latitude",
            "delivery_longitude",
            "dropoff_latitude",
            "dropoff_longitude",
        ]
    )
    df["distance_km"] = df.apply(
        lambda r: haversine_distance_km(
            r["delivery_latitude"],
            r["delivery_longitude"],
            r["dropoff_latitude"],
            r["dropoff_longitude"],
        ),
        axis=1,
    )
    df["distance_meters"] = df["distance_km"] * 1000
    return df


def to_excel_bytes(df: pd.DataFrame) -> bytes:
    out = BytesIO()
    with pd.ExcelWriter(out, engine="openpyxl") as w:
        df.to_excel(w, index=False, sheet_name="results")
    return out.getvalue()


def build_map_deck(df: pd.DataFrame, show_delivery: bool, show_dropoff: bool, show_lines: bool):
    if df.empty:
        return None

    layers = []

    delivery_df = df.rename(
        columns={"delivery_latitude": "lat", "delivery_longitude": "lon"}
    )[["shipment_code", "lat", "lon", "distance_km"]].dropna()

    dropoff_df = df.rename(
        columns={"dropoff_latitude": "lat", "dropoff_longitude": "lon"}
    )[["shipment_code", "lat", "lon", "distance_km"]].dropna()

    lines_df = df[
        [
            "shipment_code",
            "delivery_latitude",
            "delivery_longitude",
            "dropoff_latitude",
            "dropoff_longitude",
            "distance_km",
        ]
    ].dropna()
    lines_df = lines_df.rename(
        columns={
            "delivery_latitude": "from_lat",
            "delivery_longitude": "from_lon",
            "dropoff_latitude": "to_lat",
            "dropoff_longitude": "to_lon",
        }
    )

    if delivery_df.empty and dropoff_df.empty and lines_df.empty:
        return None

    pts = []
    if not delivery_df.empty:
        pts.append(delivery_df[["lat", "lon"]])
    if not dropoff_df.empty:
        pts.append(dropoff_df[["lat", "lon"]])
    all_points = pd.concat(pts, ignore_index=True)
    center_lat, center_lon = all_points["lat"].mean(), all_points["lon"].mean()

    view_state = pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=10, pitch=0)

    if show_delivery and not delivery_df.empty:
        layers.append(
            pdk.Layer(
                "ScatterplotLayer",
                data=delivery_df,
                get_position="[lon, lat]",
                get_radius=100,
                get_fill_color=[0, 0, 255, 160],
                pickable=True,
                id="delivery",
            )
        )

    if show_dropoff and not dropoff_df.empty:
        layers.append(
            pdk.Layer(
                "ScatterplotLayer",
                data=dropoff_df,
                get_position="[lon, lat]",
                get_radius=100,
                get_fill_color=[255, 0, 0, 160],
                pickable=True,
                id="dropoff",
            )
        )

    if show_lines and not lines_df.empty:
        min_d, max_d = lines_df["distance_km"].min(), lines_df["distance_km"].max()
        if max_d == min_d:
            lines_df["dist_norm"] = 1.0
        else:
            lines_df["dist_norm"] = (lines_df["distance_km"] - min_d) / (max_d - min_d)

        def color_from_norm(n):
            n = max(0.0, min(1.0, float(n)))
            r = int(100 + n * 155)
            g = int(200 - n * 150)
            return [r, g, 0, 200]

        lines_df["color"] = lines_df["dist_norm"].apply(color_from_norm)

        layers.append(
            pdk.Layer(
                "LineLayer",
                data=lines_df,
                get_source_position="[from_lon, from_lat]",
                get_target_position="[to_lon, to_lat]",
                get_width=2,
                get_color="color",
                pickable=True,
                id="lines",
            )
        )

    if not layers:
        return None

    return pdk.Deck(
        map_style="mapbox://styles/mapbox/light-v9",
        initial_view_state=view_state,
        layers=layers,
        tooltip={
            "html": "<b>Shipment:</b> {shipment_code}<br/><b>Distance (km):</b> {distance_km}",
            "style": {"backgroundColor": "steelblue", "color": "white"},
        },
    )


# =========================
#         MAIN APP
# =========================


def main():
    st.set_page_config(
        page_title="Delivery vs Dropoff Distance Validation",
        layout="wide",
    )

    # -------- LOGIN SECTION (AUTH) --------
    names = ["Admin User", "Ops User"]
    usernames = ["admin", "ops"]
    passwords = ["paxel123", "ops123"]

    hashed_passwords = stauth.Hasher().generate(passwords)


    authenticator = stauth.Authenticate(
        names,
        usernames,
        hashed_passwords,
        "cookie_name",
        "signature_key",
        cookie_expiry_days=1,
    )

    name, auth_status, username = authenticator.login("Login", "main")

    if auth_status is False:
        st.error("Incorrect username or password")
        return
    elif auth_status is None:
        st.info("Please log in to continue")
        return

    # Jika berhasil login, lanjut ke app
    st.success(f"Welcome *{name}*")
    st.title("Delivery vs Actual Dropoff Distance Validation")

    # -------- SIDEBAR FILTERS --------
    st.sidebar.header("Distance Filter")
    min_km = st.sidebar.number_input(
        "Minimum distance to show (km)",
        min_value=0.1,
        max_value=100.0,
        value=1.0,
        step=0.1,
    )

    st.sidebar.header("Map Layers")
    show_delivery = st.sidebar.checkbox("Show Delivery Points", value=True)
    show_dropoff = st.sidebar.checkbox("Show Dropoff Points", value=True)
    show_lines = st.sidebar.checkbox("Show Lines (Delivery → Dropoff)", value=True)

    # -------- FILE UPLOADER --------
    uploaded_file = st.file_uploader(
        "Upload Excel (shipment_code + delivery lat/long + dropoff lat/long + optional number_account)",
        type=["xlsx", "xls"],
    )
    if not uploaded_file:
        st.info("Please upload an Excel file to begin.")
        return

    try:
        df_raw = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Error reading Excel: {e}")
        return

    try:
        df_norm = normalize_columns(df_raw)
    except Exception as e:
        st.error(str(e))
        return

    if "number_account" in df_norm.columns:
        df_norm["account_type"] = df_norm["number_account"].apply(classify_account_type)
    else:
        df_norm["account_type"] = "Corporate/Other"

    st.sidebar.header("Account Filter")
    account_filter = st.sidebar.selectbox(
        "Filter by account type", ["All", "Retail", "Corporate/Other"], index=0
    )

    # -------- COMPUTE DISTANCES --------
    with st.spinner("Computing distances..."):
        df_dist = compute_distances(df_norm)

    mask_dist = df_dist["distance_km"] > min_km
    if account_filter == "Retail":
        mask_acc = df_dist["account_type"] == "Retail"
    elif account_filter == "Corporate/Other":
        mask_acc = df_dist["account_type"] == "Corporate/Other"
    else:
        mask_acc = True

    df_result = df_dist[mask_dist & mask_acc].sort_values("distance_km", ascending=False)

    tab_summary, tab_table, tab_map = st.tabs(["Summary", "Table", "Map"])

    # -------- SUMMARY TAB --------
    with tab_summary:
        st.subheader("Summary")
        total_rows, filtered_rows = len(df_dist), len(df_result)
        max_d = df_result["distance_km"].max() if filtered_rows else 0
        avg_d = df_result["distance_km"].mean() if filtered_rows else 0

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total valid rows", f"{total_rows:,}")
        c2.metric("Rows > threshold (after account filter)", f"{filtered_rows:,}")
        c3.metric("Max distance (km)", f"{max_d:.2f}")
        c4.metric("Average distance (km)", f"{avg_d:.2f}")

        st.markdown("#### Distance distribution (after filters)")
        if filtered_rows:
            st.dataframe(df_result["distance_km"].describe().to_frame("distance_km_stats"))
        else:
            st.info("No rows after filters.")

        st.markdown("#### Account type distribution (before distance filter)")
        st.dataframe(df_norm["account_type"].value_counts().to_frame("count"))

    # -------- TABLE TAB --------
    with tab_table:
        st.subheader(
            f"Rows where distance > {min_km:.2f} km"
            + ("" if account_filter == "All" else f" and account_type = {account_filter}")
        )
        st.write(f"Total: **{len(df_result):,} rows**")
        st.dataframe(df_result)

        if len(df_result) > 0:
            excel_all = to_excel_bytes(df_result)
            st.download_button(
                "📥 Download Results (Excel - All Columns)",
                excel_all,
                file_name="delivery_vs_dropoff_validation_all_cols.xlsx",
                mime=(
                    "application/vnd.openxmlformats-officedocument."
                    "spreadsheetml.sheet"
                ),
            )

            ops_cols = [
                "shipment_code",
                "number_account",
                "distance_meters",
            ]
            ops_cols = [c for c in ops_cols if c in df_result.columns]
            if ops_cols:
                df_ops = df_result[ops_cols].copy()
                excel_ops = to_excel_bytes(df_ops)
                st.download_button(
                    "📥 Download Results for Ops (Excel - Key Columns)",
                    excel_ops,
                    file_name="delivery_vs_dropoff_validation_ops.xlsx",
                    mime=(
                        "application/vnd.openxmlformats-officedocument."
                        "spreadsheetml.sheet"
                    ),
                )

    # -------- MAP TAB --------
    with tab_map:
        st.subheader("Map Visualization (Delivery → Dropoff)")
        st.markdown(
            "- **Blue dots** = Delivery points  \n"
            "- **Red dots** = Actual Dropoff points  \n"
            "- **Lines** = Delivery → Dropoff (color intensity by distance)"
        )
        if len(df_result) == 0:
            st.info("No data to show on the map (no rows after filters).")
        else:
            deck = build_map_deck(df_result, show_delivery, show_dropoff, show_lines)
            if deck is None:
                st.info("No valid coordinates or all layers disabled.")
            else:
                st.pydeck_chart(deck)


if __name__ == "__main__":
    main()
