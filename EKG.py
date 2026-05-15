#%%-------------------------importowanie bibliotek-----------------------------
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
import glob, os
from pathlib import Path
from scipy.signal import savgol_filter, find_peaks
import warnings
import emd
import seaborn as sns
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import neurokit2 as nk

#%%--------------------------------Lokalizacja pliku---------------------------
#streamlit run "C:\Users\emisl\OneDrive - University of Gdansk (for Students)\Pulpit\EKG.py"

# Ścieżka dostosowana
path_inp = r"C:/Users/emili/Desktop/EKG"

try:
    current_dir = os.path.dirname(os.path.abspath(_file_))
except NameError:
    current_dir = os.getcwd()

# Wejście do folderu z danymi
if os.path.exists(path_inp):
    if os.path.isdir(path_inp):
        os.chdir(path_inp)
    else:
        os.chdir(os.path.dirname(path_inp))
else:
    os.chdir(current_dir)
    
#%%-----------------------------Ustawienia kolorów-----------------------------

st.set_page_config(layout="wide", page_title="Analiza EKG")
cyan        = "#00E5FF"  # Neonowy Błękit (Cyan)
amber       = "#FFB300"  # Bursztynowy (Amber)
niebieski    = "#2C3E50"  # Ciemny stalowy niebieski
teal   = "#00BFA5"  # Ciemny morski (Teal)
bialy           = "#E0E0E0"  # Złamana biel
grafit          = "#121212"  # Głęboki grafit
ciemnoszary       = "#1E1E1E"  # Lekko jaśniejszy ciemnoszary
szary     = "#757575"  # Średni szary
węgiel        = "#212121"  # Ciemny węgiel
jasnoszary    = "#BDBDBD"  # Jasnoszary

# USTAWIENIA CSS
st.markdown(f"""
    <style>
    .stApp {{
        color: {grafit};
        font-size: 16px;
    }}

    h1, h2, h3, h4, [data-testid="stHeader"] {{
        color: {bialy} !important;
    }}

    p, .stText, [data-testid="stWidgetLabel"] {{
        color: {węgiel};
    }}

    [data-testid="stMetricValue"] {{
        color: {amber} !important;
        font-size: 18px !important;
    }}
    
    [data-testid="stMetricLabel"] p {{
        color: {szary} !important;
    }}

    .moja-ramka {{
        border-radius: 10px;
        padding: 20px;
        background-color: {amber}; 
        text-align: center;
        height: 120px;
    }}
    </style>
    """, unsafe_allow_html=True)

#%%--------------------------------Ładowanie pliku-----------------------------

@st.cache_data
def load_my_data(file_name):
    try:
        with open(file_name, 'r') as f:
            lines = f.readlines()
        
        data_list = []
        for line in lines:
            line = line.replace(',', '.')
            parts = line.split()
            try:
                if len(parts) >= 2:
                    czas = float(parts[0])
                    ecg = float(parts[1])
                    data_list.append([czas, ecg])
            except ValueError:
                continue
        return pd.DataFrame(data_list, columns=['czas', 'ecg'])
    except Exception as e:
        st.error(f"Błąd ładowania: {e}")
        return pd.DataFrame({'czas': [], 'ecg': []})

txt_files = glob.glob("*.txt")

if not txt_files:
    st.warning(f"Brak plików .txt w: {os.getcwd()}")
    st.stop()

selected_file = st.sidebar.selectbox("Wybierz plik z danymi EKG:", txt_files)
df = load_my_data(selected_file)
df = df.apply(pd.to_numeric, errors='coerce').dropna()

#%%---------------------------------Tytuł i ramka------------------------------

st.markdown(f"""
    <div class="moja-ramka">
        <h4 style="color: {bialy}; margin: 0;">Analiza HRV sygnału EKG</h4>
        <p style="color: {niebieski};">laboratorium fizyki medycznej</p>
    </div>
    """, unsafe_allow_html=True)   

st.markdown(f'<hr style="margin-top: 10px;height:5px; border:none; background-color:{amber};" />', unsafe_allow_html=True)

#%%-------------------------------------SEKCJA 1-------------------------------

col1, col2, col3 = st.columns([2, 1.5, 5])
    
with col1:
    st.dataframe(df, height=380, use_container_width=True)

with col2:
    min_czas = float(df['czas'].min()) if not df.empty else 0.0
    max_czas = float(df['czas'].max()) if not df.empty else 10.0
    domyslny_koniec = min_czas + 20.0 if max_czas > min_czas + 20.0 else max_czas
    
    zakres_czasu = st.slider("Wybierz zakres czasu do analizy [s]:", min_czas, max_czas, (min_czas, domyslny_koniec), step=0.1)
    
    df_stary = df.copy()
    df_filtered_view = df[(df['czas'] >= zakres_czasu[0]) & (df['czas'] <= zakres_czasu[1])].copy()

    # Definiujemy parametry filtra
    n_samples = len(df_filtered_view)
    if n_samples > 11:  # filtr potrzebuje minimum danych
        df_filtered_view['ecg_filtered'] = savgol_filter(df_filtered_view['ecg'], window_length=51, polyorder=3)
    else:
        df_filtered_view['ecg_filtered'] = df_filtered_view['ecg']
    # ------------------------------------------------

    ile_zostalo = len(df_filtered_view)
    ile_wycieto = len(df_stary) - ile_zostalo
    
    # 1. Tworzenie wykresu
    fig_pie = px.pie(
        names=["Fragment do analizy", "Pozostała część"], 
        values=[ile_zostalo, ile_wycieto],
        hole=0.4,  
        color_discrete_sequence=[cyan, niebieski],
        category_orders={"names": ["Fragment do analizy", "Pozostała część"]}
    )

    # 2. Ustawienia legendy
    fig_pie.update_layout(
        height=280,                
        margin=dict(l=20, r=20, t=20, b=50),
        showlegend=True,           
        legend=dict(
            orientation="h",      
            yanchor="top",
            y=-0.1,                
            xanchor="center",
            x=0.5
        )
    )
    st.plotly_chart(fig_pie, use_container_width=True)
    
with col3:
    with st.container(border=True):
        fig = go.Figure()

        # 1. Dodajemy Sygnał - Całość 
        fig.add_trace(go.Scatter(
            x=df_stary['czas'], 
            y=df_stary['ecg'], 
            name='Pozostała część',
            line=dict(color=jasnoszary, width=1)
        ))

        # 2. Dodajemy Sygnał - Wybrany fragment
        fig.add_trace(go.Scatter(
            x=df_filtered_view['czas'], 
            y=df_filtered_view['ecg'], 
            name='Fragment do analizy',
            line=dict(color=cyan, width=2)
        ))

        # 3. Stylizacja wykresu
        fig.update_layout(
            height=350,
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            # Ustawienia legendy
            legend=dict(
                orientation="h",      # Legenda w poziomie
                yanchor="bottom",
                y=1.02,               # Nad wykresem
                xanchor="left",       # Do lewej krawędzi
                x=0
            ),
            xaxis=dict(
                title="Czas [s]",
                range=[zakres_czasu[0], zakres_czasu[1]]
            ),
            yaxis_title="Amplituda [mV]"
        )

        st.plotly_chart(fig, use_container_width=True)
    
#%%-----------------------------------SEKCJA 2: HRV & HISTOGRAM---------------------------------

# Obliczenia pików
peaks, _ = find_peaks(df_filtered_view['ecg_filtered'], distance=500, height=0.25)
if len(peaks) > 1:
    rr_intervals = np.diff(df_filtered_view['czas'].iloc[peaks].values) * 1000
else:
    rr_intervals = []

# GŁÓWNY PODZIAŁ SEKCJI
col_hrv, col_hist = st.columns([ 4 , 4.5 ])


with col_hrv:
    st.markdown(f"""<p style="font-size: 18px; font-weight: bold; color: {amber};">Identyfikacja załamków R i tworzenie szeregu RR</p>""", unsafe_allow_html=True)
    st.markdown(f"""<hr style="margin-top: -10px; height:5px; border:none; background-color:{amber};" />""", unsafe_allow_html=True)
    
    c_left, c_right = st.columns([1.5, 3.5])

    with c_left:
        # OKNO 1: (identyfikacja pików)
        st.markdown(f'<p style="font-size: 14px; color: {amber}; text-align: center;">Identyfikacja załamków R</p>', unsafe_allow_html=True)
        
        t_start = df_filtered_view['czas'].iloc[0]
        df_zoom = df_filtered_view[df_filtered_view['czas'] <= t_start + 10]
        peaks_zoom, _ = find_peaks(df_zoom['ecg_filtered'], distance=400, height=0.25)
        
        fig_zoom = go.Figure()
        fig_zoom.add_trace(go.Scatter(x=df_zoom['czas'], y=df_zoom['ecg_filtered'], line=dict(color=cyan, width=2)))
        fig_zoom.add_trace(go.Scatter(x=df_zoom['czas'].iloc[peaks_zoom], y=df_zoom['ecg_filtered'].iloc[peaks_zoom], mode='markers', marker=dict(color=amber, size=10)))
        fig_zoom.update_layout(height=200, margin=dict(l=0,r=0,t=0,b=0), showlegend=False, xaxis_visible=False, yaxis_visible=False)
        st.plotly_chart(fig_zoom, use_container_width=True)

    with c_right:
        # OKNO 2: Wykres szeregu RR (Tachogram)
        st.markdown(f'<p style="font-size: 14px; color: {amber}; text-align: center;">Szereg RR (ms)</p>', unsafe_allow_html=True)
        
        if len(rr_intervals) > 0:
            # Wykres punktowy odstępów RR w czasie
            fig_rr = go.Figure()
            fig_rr.add_trace(go.Scatter(
                x=df_filtered_view['czas'].iloc[peaks[1:]], # czas wystąpienia uderzenia
                y=rr_intervals, 
                mode='lines+markers',
                line=dict(color=amber, width=1),
                marker=dict(size=4)
            ))
            fig_rr.update_layout(height=200, margin=dict(l=0,r=0,t=10,b=0), xaxis_title="Czas [s]", yaxis_title="RR [ms]")
            st.plotly_chart(fig_rr, use_container_width=True)

with col_hist:
    st.markdown(f'<p style="font-size: 18px; font-weight: bold; color:{amber};">Histogram</p>', unsafe_allow_html=True)
    st.markdown(f"""<hr style="margin-top: -10px; height:5px; border:none; background-color:{amber};" />""", unsafe_allow_html=True)
    # Histogram
    if len(rr_intervals) > 0:
        fig_hist = px.histogram(x=rr_intervals, nbins=20, color_discrete_sequence=[amber], labels={'x': 'Odstęp RR [ms]'})
        fig_hist.update_layout(height=230, margin=dict(l=0,r=0,t=0,b=0),yaxis_title="Liczba zliczeń")
        st.plotly_chart(fig_hist, use_container_width=True)

from PyEMD import EMD

#%%--------------------------------SEKCJA 3: DEKOMPOZYCJA EMD--------------------------

st.markdown(f"""<p style="font-size: 18px; font-weight: bold; color: {amber};">Empiryczna Dekompozycja Modalna (EMD)</p>""", unsafe_allow_html=True)
st.markdown(f"""<hr style="margin-top: -10px; height:5px; border:none; background-color:{amber};" />""", unsafe_allow_html=True)

if not df_filtered_view.empty:
    
    # POBIERANIE DANYCH Z SYNCHRONIZACJĄ:
    data_to_emd = df_filtered_view['ecg'].values #możliwoć zamianysygnału: ecg-surowy sygnał, ecg_filtered-po użyciu filtra
    time_to_emd = df_filtered_view['czas'].values
    
    # Informacja o długości analizowanego fragmentu
    czas_trwania = time_to_emd[-1] - time_to_emd[0]
    st.info(f"Analizowany fragment: {czas_trwania:.2f} s ({len(data_to_emd)} próbek)")

    # Uruchomienie EMD
    with st.spinner('Trwa dekompozycja...'):
        emd = EMD()
        imfs = emd(data_to_emd)
        n_imfs = imfs.shape[0]
        
# 1. Obliczenia i przygotowanie wierszy imfs
display_imfs = n_imfs  
rows_count = display_imfs + 1
fig_emd = make_subplots(
        rows=rows_count, cols=1, 
        shared_xaxes=True,
        vertical_spacing=0.02
    )

    # 1.Dodajemy SUMMED IMFs na samej górze
samples_x = np.arange(len(data_to_emd))
    
# 2. SUMMED IMFs
fig_emd.add_trace(
        go.Scatter(x=samples_x, y=data_to_emd, line=dict(color=węgiel, width=1.5)),
        row=1, col=1
    )

    # Dodajemy składowe IMF w pętli
kolory = [cyan, amber, teal, "#FF69B4", "#DA70D6", "#FF1493", niebieski]
    
for i in range(display_imfs):
        fig_emd.add_trace(
            go.Scatter(
                x=samples_x, 
                y=imfs[i], 
                line=dict(color=kolory[i % len(kolory)], width=2)
            ),
            row=i + 2, col=1
        )
    

# 3. Legenda i skala
names = ["Summed<br>IMFs"] + [f"IMF-{i+1}" for i in range(display_imfs)]
    
for i, name in enumerate(names):
        curr_row = i + 1
        
        # wysokoć
        current_data = data_to_emd if curr_row == 1 else imfs[i-1]
            
        y_max = np.max(np.abs(current_data)) * 1.1
        
        # Ustawienie osi Y
        fig_emd.update_yaxes(
            range=[-y_max, y_max], 
            row=curr_row, col=1,
            showline=True, linewidth=1.5, linecolor=grafit,
            gridcolor='lightgrey'
        )
        
        # DODANIE skali po lewej stronie
        fig_emd.add_annotation(
            dict(
                text=f"<b>{name}</b>", 
                x=-0.12,
                y=0.5,
                xref="paper",
                yref=f"y{curr_row if curr_row > 1 else ''} domain",
                showarrow=False,
                textangle=0,
                xanchor='right',
                font=dict(size=12, color=amber)
            )
        )

    # 5. Stylizacja końcowa
fig_emd.update_xaxes(
        showline=True, 
        linewidth=1.5, 
        linecolor=grafit,  # Kolor linii osi
        title_text="<b>Time (samples)</b>", 
        title_font=dict(color=amber, size=14), # Kolor napisu "Time (samples)"
        tickfont=dict(color=teal), # Kolor numerków (0, 5k, 10k...)
        row=rows_count, 
        col=1
    )

fig_emd.update_layout(
        height=150 * rows_count,
        margin=dict(l=250, r=30, t=30, b=80), 
        plot_bgcolor='white',   # Białe tło pod wykresem
        paper_bgcolor='white',  # Białe tło całej karty
        showlegend=False
    )

st.plotly_chart(fig_emd, use_container_width=True)

#%% ------------- SEKCJA 4: PROSTOWANIE SYGNAŁU (ZADANIE 2 i 3) ---------------
st.markdown(f"""<p style="font-size: 18px; font-weight: bold; color: {amber};">Usuwanie modulacji oddechowej (Prostowanie EKG)</p>""", unsafe_allow_html=True)
st.markdown(f"""<hr style="margin-top: -10px; height:5px; border:none; background-color:{amber};" />""", unsafe_allow_html=True)
    
if n_imfs >= 2:
        # 1. Definiujemy oddech
        oddech_drift = imfs[-1] + imfs[-2]
        
        # 2. PROSTOWANIE: Odejmowanie dryftu od oryginalnego sygnału
        ecg_wyprostowane = data_to_emd - oddech_drift

        # 3. Wykres porównawczy
        fig_clean = go.Figure()

        # Sygnał oryginalny (szary)
        fig_clean.add_trace(go.Scatter(
            x=time_to_emd, y=data_to_emd, 
            name="Sygnał surowy", 
            line=dict(color=jasnoszary, width=1)
        ))

        # Sygnał wyprostowany (fioletowy)
        fig_clean.add_trace(go.Scatter(
            x=time_to_emd, y=ecg_wyprostowane, 
            name="EKG po usunięciu modulacji", 
            line=dict(color=amber, width=1.5)
        ))

        fig_clean.update_layout(
            height=400,
            xaxis_title="Czas [s]",
            yaxis_title="Amplituda [mV]",
            plot_bgcolor='white',
            paper_bgcolor='white',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_clean, use_container_width=True)

        # 4. EKSPORT (Zadanie 3 )
        st.success("Zadanie 3: Sygnał wyprostowany gotowy do eksportu.")
        
        # Przygotowanie danych do pliku tekstowego
        df_out = pd.DataFrame({
            'czas[s]': time_to_emd, 
            'ECG_clean': ecg_wyprostowane
        })
        
        # Generowanie pliku do pobrania (format TXT)
        csv_txt = df_out.to_csv(index=False, sep='\t')
        
        st.download_button(
            label="Pobierz wyprostowane EKG (.txt)",
            data=csv_txt,
            file_name="EKG_wyprostowane_output.txt",
            mime="text/plain"
        )
# --- SEKCJA 6: ANALIZA CZĘSTOTLIWOŚCIOWA (FFT) ---
st.markdown(f"""<p style="font-size: 18px; font-weight: bold; color: {amber};">Analiza Widmowa (FFT) - Przed i Po</p>""", unsafe_allow_html=True)
st.markdown(f"""<hr style="margin-top: -10px; height:5px; border:none; background-color:{amber};" />""", unsafe_allow_html=True)

   # 1. Obliczenia FFT
n_fft = len(data_to_emd)
freqs = np.fft.fftfreq(n_fft, d=1/1000) # 1000=fs
    
fft_raw = np.abs(np.fft.fft(data_to_emd) / n_fft)
fft_clean = np.abs(np.fft.fft(ecg_wyprostowane) / n_fft)

pos_mask = freqs > 0
f_plot = freqs[pos_mask]
fft_raw_plot = fft_raw[pos_mask]
fft_clean_plot = fft_clean[pos_mask]

    # 2. Wykres widma
fig_fft = go.Figure()

fig_fft.add_trace(go.Scatter(
        x=f_plot, y=fft_raw_plot, 
        name="Widmo sygnału surowego", 
        line=dict(color=teal, width=2)
    ))

fig_fft.add_trace(go.Scatter(
        x=f_plot, y=fft_clean_plot, 
        name="Widmo po usunięciu oddechu", 
        line=dict(color=cyan, width=1.5)
    ))

fig_fft.update_layout(
        height=400,
        xaxis_title="Częstotliwość [Hz]",
        yaxis_title="Amplituda",
        xaxis_range=[0, 15], # zakres 0-15 Hz
        plot_bgcolor='white',
        paper_bgcolor='white',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

st.plotly_chart(fig_fft, use_container_width=True)
