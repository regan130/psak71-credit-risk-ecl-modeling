import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os

# Konfigurasi Halaman
st.set_page_config(
    page_title="PSAK 71 ECL / CKPN Risk Analytics",
    page_icon="🏦",
    layout="wide"
)

# Load Data dengan Caching agar ringan
@st.cache_data
def load_portfolio_data():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    file_path = os.path.join(base_dir, 'data', 'processed', 'final_ecl_portfolio.parquet')
    df = pd.read_parquet(file_path)
    stage_labels = {
        1: 'Stage 1 (Performing - 12M ECL)',
        2: 'Stage 2 (Underperforming - Lifetime ECL)',
        3: 'Stage 3 (Non-Performing / Default)'
    }
    df['Stage_Label'] = df['psak71_stage'].map(stage_labels)
    return df

df = load_portfolio_data()

# ================= SIDEBAR: FILTER & STRESS-TEST SIMULATOR =================
st.sidebar.header("🎛️ Filter Portofolio")
selected_grades = st.sidebar.multiselect(
    "Pilih Grade Kredit:",
    options=sorted(df['grade'].unique()),
    default=sorted(df['grade'].unique())
)

st.sidebar.markdown("---")
st.sidebar.header("⚡ Macroeconomic Stress-Testing")
st.sidebar.caption("Geser parameter makro untuk mensimulasikan dampak guncangan ekonomi terhadap kebutuhan CKPN:")

shock_interest = st.sidebar.slider("Perubahan Suku Bunga Acuan (%)", min_value=-1.0, max_value=5.0, value=0.0, step=0.25)
shock_unemp = st.sidebar.slider("Perubahan Tingkat Pengangguran (%)", min_value=-1.0, max_value=5.0, value=0.0, step=0.25)
lgd_haircut = st.sidebar.slider("Tambahan Penurunan Nilai Agunan / LGD Shock (%)", min_value=0.0, max_value=15.0, value=0.0, step=1.0)

# Filter Data
filtered_df = df[df['grade'].isin(selected_grades)].copy()

# Hitung Ulang PD & ECL Secara Real-Time Berdasarkan Input Stress-Test
# Menggunakan koefisien sensitivitas makro (+0.015 per 1% suku bunga, +0.020 per 1% pengangguran)
delta_pd_stress = (0.015 * shock_interest) + (0.020 * shock_unemp)

filtered_df['pd_stressed'] = np.where(
    filtered_df['psak71_stage'] == 3,
    1.0,
    (filtered_df['pd_weighted'] + delta_pd_stress).clip(0.001, 0.99)
)
filtered_df['lgd_stressed'] = (filtered_df['segment_lgd'] + (lgd_haircut / 100.0)).clip(0.0, 1.0)
filtered_df['ecl_stressed'] = filtered_df['pd_stressed'] * filtered_df['lgd_stressed'] * filtered_df['ead']

# ================= HEADER & KPI UTAMA =================
st.title("🏦 Credit Risk & PSAK 71 ECL (CKPN) Analytics Platform")
st.markdown("Simulasi Pemodelan **PD, LGD, EAD**, *Markov Chain Lifetime Projection*, dan *Macroeconomic Stress-Testing*.")

total_ead = filtered_df['ead'].sum()
base_ecl = filtered_df['ecl_weighted'].sum()
stressed_ecl = filtered_df['ecl_stressed'].sum()
delta_ecl = stressed_ecl - base_ecl
coverage_ratio = (stressed_ecl / total_ead * 100) if total_ead > 0 else 0

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Eksposur Kredit (EAD)", f"${total_ead:,.0f}")
col2.metric("Baseline CKPN (PSAK 71 Weighted)", f"${base_ecl:,.0f}")
col3.metric(
    "Stressed CKPN (Simulasi Aktif)",
    f"${stressed_ecl:,.0f}",
    delta=f"${delta_ecl:,.0f} ({(delta_ecl/base_ecl*100) if base_ecl>0 else 0:+.1f}%)",
    delta_color="inverse"
)
col4.metric("CKPN Coverage Ratio", f"{coverage_ratio:.2f}%")

st.markdown("---")

# ================= TABS ANALISIS =================
tab1, tab2, tab3 = st.tabs(["📊 Ringkasan PSAK 71 Staging", "📈 Analisis Segmen & Skenario", "📋 Detail Portofolio & Export"])

with tab1:
    c1, c2 = st.columns(2)
    
    stage_summary = filtered_df.groupby('Stage_Label').agg(
        Rekening=('id', 'count'),
        Total_EAD=('ead', 'sum'),
        Baseline_ECL=('ecl_weighted', 'sum'),
        Stressed_ECL=('ecl_stressed', 'sum'),
        Avg_PD=('pd_stressed', 'mean'),
        Avg_LGD=('lgd_stressed', 'mean')
    ).reset_index()

    with c1:
        st.subheader("Distribusi Eksposur (EAD) per Stage")
        fig_ead = px.pie(
            stage_summary, values='Total_EAD', names='Stage_Label',
            hole=0.45, color_discrete_sequence=px.colors.qualitative.Set2
        )
        st.plotly_chart(fig_ead, use_container_width=True)

    with c2:
        st.subheader("Perbandingan Kebutuhan Cadangan (Baseline vs Stressed)")
        fig_bar = go.Figure(data=[
            go.Bar(name='Baseline CKPN', x=stage_summary['Stage_Label'], y=stage_summary['Baseline_ECL'], marker_color='#2b5c8f'),
            go.Bar(name='Stressed CKPN', x=stage_summary['Stage_Label'], y=stage_summary['Stressed_ECL'], marker_color='#d9534f')
        ])
        fig_bar.update_layout(barmode='group', yaxis_title="Nominal CKPN ($)")
        st.plotly_chart(fig_bar, use_container_width=True)

    st.subheader("Tabel Rekapitulasi Eksekutif per Stage")
    st.dataframe(
        stage_summary.style.format({
            'Rekening': '{:,}',
            'Total_EAD': '${:,.2f}',
            'Baseline_ECL': '${:,.2f}',
            'Stressed_ECL': '${:,.2f}',
            'Avg_PD': '{:.2%}',
            'Avg_LGD': '{:.2%}'
        }),
        use_container_width=True
    )

with tab2:
    c3, c4 = st.columns(2)
    
    grade_summary = filtered_df.groupby('grade').agg(
        Total_EAD=('ead', 'sum'),
        Stressed_ECL=('ecl_stressed', 'sum'),
        Avg_PD=('pd_stressed', 'mean')
    ).reset_index()

    with c3:
        st.subheader("Profil Risiko & CKPN per Grade Kredit")
        fig_grade = px.bar(
            grade_summary, x='grade', y='Stressed_ECL',
            color='Avg_PD', color_continuous_scale='Reds',
            labels={'grade': 'Grade Kredit', 'Stressed_ECL': 'Total CKPN ($)', 'Avg_PD': 'Rata-rata PD'}
        )
        st.plotly_chart(fig_grade, use_container_width=True)

    with c4:
        st.subheader("Perbandingan 3 Skenario Makroekonomi (PSAK 71)")
        scen_df = pd.DataFrame({
            'Skenario': ['Optimistic (25%)', 'Base Case (50%)', 'Downturn / Krisis (25%)', 'Weighted PSAK 71'],
            'Total_CKPN': [
                filtered_df['ecl_optimistic'].sum(),
                filtered_df['ecl_base'].sum(),
                filtered_df['ecl_downturn'].sum(),
                filtered_df['ecl_weighted'].sum()
            ]
        })
        fig_scen = px.bar(
            scen_df, x='Skenario', y='Total_CKPN',
            color='Skenario', color_discrete_sequence=['#5cb85c', '#5bc0de', '#d9534f', '#2b5c8f']
        )
        st.plotly_chart(fig_scen, use_container_width=True)

with tab3:
    st.subheader("Sampel Data Tingkat Rekening (Top 100 Eksposur Terbesar)")
    top_accounts = filtered_df.sort_values(by='ecl_stressed', ascending=False).head(100)[
        ['id', 'grade', 'sub_grade', 'psak71_stage', 'tenor_months', 'ead', 'pd_stressed', 'lgd_stressed', 'ecl_stressed']
    ]
    st.dataframe(top_accounts, use_container_width=True)
    
    csv_export = top_accounts.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Unduh Laporan Risiko (CSV)",
        data=csv_export,
        file_name="laporan_ckpn_psak71.csv",
        mime="text/csv"
    )