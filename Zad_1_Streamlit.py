#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 25.11.2025 edergergfer
#pomoc: https://www.kaggle.com/code/stetelepta/exploring-heart-rate-variability-using-python

#%%------------------------------------BIBLIOTEKI------------------------------

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
import glob, os
from pathlib import Path
from scipy.signal import savgol_filter
from scipy.signal import find_peaks
import warnings
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import neurokit2 as nk
import emd
import scipy.ndimage as ndimage
import gdown
#%%--------------------------------Lokalizacja pliku---------------------------

path_inp='C:/Users/emili/Desktop/EKG'

#%%--------------------------------Ustawienia wstępne--------------------------

st.set_page_config(layout="wide")

#---------------------------Definicje Kolorów do których potem się odwołujemy

bialy          ="#ffffff"
bialo_szary    ="#aeaeae"
rozowy         ="#ff29a7"
niebieski      ="#0092ff"
zielony        ="#c8ff4a"
czerwony       ="#ff1100"
lekki_szary    ="#363636"
lekki_czerwony ="#e74c3c"
mocny_szary    ="#999999"

#---------------------------Ustawienia kolorów

st.markdown(f"""
    <style>
    /* 1. Zmiana koloru głównego tekstu w aplikacji */
    .stApp {{
        color: {bialy};
    }}

    /* 2. Zmiana koloru dla wszystkich nagłówków (h1, h2, h3) */
    h1, h2, h3, [data-testid="stHeader"] {{
        color: {bialy} !important;
    }}

    /* 3. Zmiana koloru zwykłego tekstu (paragrafy, opisy) */
    p, .stText, [data-testid="stWidgetLabel"] {{
        color: {bialy};
        font-size: 16px;
    }}

    /* 4. Metryki - Wartość (liczba) */
    [data-testid="stMetricValue"] {{
        font-size: 18px !important;
        color: {bialy} !important; 
    }}
    
    /* 5. Metryki - Etykieta (opis nad liczbą) */
    [data-testid="stMetricLabel"] p {{
        color: {mocny_szary} !important;
    }}
    </style>
    """, unsafe_allow_html=True)
    
#%%--------------------------------Ładowanie pliku-----------------------------

@st.cache_data
@st.cache_data(ttl=1, show_spinner=False)
@st.cache_data(show_spinner=False)
def load_my_data():
    filename_1 = "ecg.txt"
    file_id1 = '1h51q75CzyWE75pPT0uI6uXoZzdVmFJkJ'
  
    # Funkcja pomocnicza do pobierania
    def download_file(file_id, output_name):
        url = f'https://drive.google.com/uc?id={file_id}'
        # Jeśli plik istnieje, usuwamy go, aby gdown pobrał nową wersję
        if os.path.exists(output_name):
            os.remove(output_name)
        
        try:
            # gdown czasami potrzebuje parametru fuzzy=True dla GDrive
            gdown.download(url, output_name, quiet=False, fuzzy=True)
        except Exception as e:
            st.error(f"Błąd pobierania {output_name}: {e}")

    # Pobieranie plików
    with st.spinner('Synchronizacja z Google Drive...'):
        download_file(file_id1, filename_1)

    # Wczytywanie
    data = pd.read_csv(filename_1, 
                       sep='\t', 
                       decimal=',', 
                       names=['czas', 'oddech', 'ecg'], # Definiujemy 3 kolumny
                       header=None) 
    return data

# Wywołanie danych
df = load_my_data()

df = load_my_data()
df.columns = ['czas','oddech','ecg']
df['czas'] = df['czas'].astype(str).str.replace(',', '.', regex=False)

# duration = 200
sampling_rate = 1000
# ecg_signal = nk.ecg_simulate(duration=duration, sampling_rate=sampling_rate, heart_rate=80)

# # 2. Tworzenie osi czasu
# czas = np.linspace(0, duration, duration * sampling_rate)

# df = pd.DataFrame({
#     'czas': czas,
#     'ecg': ecg_signal
# })



#---------------------------Usuwamy wszystko to co nie jest liczbą

df = df.apply(pd.to_numeric, errors='coerce')
df = df.dropna()

#%%---------------------------------Tytuł i ramka------------------------------

st.markdown(f"""
    <style>
    .moja-ramka {{
        
        border-radius: 10px;
        padding: 20px;
        background-color: {lekki_szary};
        text-align: center;
        height: 120px;
    }}
    .moja-ramka h4 {{
        color: {lekki_czerwony};
        margin: 0;
    }}
    </style>
    
    <div class="moja-ramka">
        <h4>Analiza HRV sygnału EKG</h4>
        <p style="color: {bialy};">Zaawansowane laboratorium fizyki medycznej</p>
    </div>
    """, unsafe_allow_html=True)   

#---------------------------Pozioma linia biała

st.markdown(f"""
    <hr style="margin-top: 10px;height:5px; border:none; color:{lekki_szary}; background-color:#444444;" />
""", unsafe_allow_html=True)

#%%-----------------------------SEKCJA 1 - ZAKRES SYGNAŁU----------------------

col1, col2, col3 = st.columns([2.0,1.5,5])
    
with col1:
    
#---------------------------Kolumna 1 
        
    st.dataframe(df,height=265, use_container_width=True)

with col2:

    #-----------------------Kolumna 2
            
        # Wyświetlamy w MB jeśli plik jest duży, inaczej w KB
    min_czas = float(df['czas'].min())
    max_czas = float(df['czas'].max())
    zakres_czasu = st.slider("Wybierz zakres czasu do analizy [s]:",min_value=min_czas,max_value=max_czas,value=(min_czas, max_czas),step=0.1)
    df_stary=df.copy()
    df = df[(df['czas'] >= zakres_czasu[0]) & (df['czas'] <= zakres_czasu[1])].copy()
    ile_zostalo = len(df)
    ile_wycieto = len(df_stary) - ile_zostalo
    dane_pie = {
    "Status": ["Fragment do analizy", "Pozostała część"],
    "Liczba próbek": [ile_zostalo, ile_wycieto]}

# 3. Tworzenie wykresu Plotly Express
    fig_pie = px.pie(
    dane_pie, 
    values='Liczba próbek', 
    names='Status',
    hole=0.4,  # Robimy z tego "Donut chart", wygląda nowocześniej
    color_discrete_sequence=['#e74c3c', '#7e7e7e'] # Zielony i ciemnoszary
)

# 4. Stylizacja
    fig_pie.update_layout(
    height=200,
    margin=dict(l=20, r=20, t=0, b=20),
    showlegend=True,
    legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
)
    st.plotly_chart(fig_pie, use_container_width=True)
    
with col3:
        
    fig = go.Figure()

        # 2. Dodajemy Sygnał Surowy (niebieski, cieńszy)
    fig.add_trace(go.Scatter(
            x=df_stary['czas'], 
            y=df_stary['ecg'], 
            mode='lines',
            name='Pozostała częsc',
            line=dict(color=bialo_szary, width=2)
        ))

        # 3. Dodajemy Sygnał Przefiltrowany (czerwony, grubszy)
    fig.add_trace(go.Scatter(
            x=df['czas'], 
            y=df['ecg'], 
            mode='lines',
            name='Fragment do analizy',
            line=dict(color=lekki_czerwony, width=3) # Wyrazisty czerwony
        ))

        # 4. Stylizacja wykresu
    fig.update_layout(
            height=230,
            margin=dict(l=0, r=0, t=10, b=0),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            legend=dict(
                orientation="h",      # Legenda w poziomie
                yanchor="bottom",
                y=1.02,               # Nad wykresem
                xanchor="left",       # Zakotwiczenie do lewej
                x=0                   # Pozycja na osi X (0 = start od lewej)
            ),
            xaxis_title="Czas [s]",
            yaxis_title="Amplituda [mV]"
        )

    with st.container(border=True):
        st.plotly_chart(fig, use_container_width=True)
    # Dodajemy grubszą linię dla oddzielenia (tę, którą robiliśmy wcześniej)


    
st.markdown("""
    <hr style="margin-top: -10px;height:5px; border:none; color:#444444; background-color:#444444;" />
""", unsafe_allow_html=True)


#%%-----------------------------SEKCJA 2 - FILTRY------------------------------
col1, col2 = st.columns([2, 7])

with col1:
# ----------------KOUMNA 1: Suwaki i parametry filtra
    st.markdown("###### Filtrowanie")
    st.markdown(f"""
            <div style="background-color: {lekki_czerwony}; 
                border-radius: 10px; 
                padding: 40px;
                margin-bottom: -420px; /* Trik, żeby 'podłożyć' tło pod wykres */
                height: 270px;
                border: 0px solid rgba(100,100,100,1);
            ">
            </div>
        """, unsafe_allow_html=True)
        
    #-----------------------------------dajemy 3 kolumny zeby wycentrować suwaki
    lewy, srodek, prawy = st.columns([0.1, 0.8, 0.1])
    with srodek:
        window_length = st.slider("Długoć okna filtra:", min_value=1,max_value=102,value=43, step=1)
        polyorder = st.slider("Stopień wielomianu:", min_value=1,max_value=6,value=2, step=1)
    
    
    
df['ecg'].astype(float)
surowy_wektor = df['ecg'].values
df['ecg_filtrowany'] = savgol_filter(surowy_wektor, window_length, polyorder)


with col2:
    # 1. Tworzymy pustą figurę
    fig = go.Figure()

    # 2. Dodajemy Sygnał Surowy (niebieski, cieńszy)
    fig.add_trace(go.Scatter(
        x=df['czas'], 
        y=df['ecg'], 
        mode='lines',
        name='Surowy',
        line=dict(color='rgba(52, 152, 219, 0.5)', width=1) # Przezroczysty niebieski
    ))

    # 3. Dodajemy Sygnał Przefiltrowany (czerwony, grubszy)
    fig.add_trace(go.Scatter(
        x=df['czas'], 
        y=df['ecg_filtrowany'], 
        mode='lines',
        name='Savgol Filter',
        line=dict(color=lekki_czerwony, width=3) # Wyrazisty czerwony
    ))

    # 4. Stylizacja wykresu
    fig.update_layout(
        height=232,
        margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        legend=dict(
            orientation="h",      # Legenda w poziomie
            yanchor="bottom",
            y=1.02,               # Nad wykresem
            xanchor="left",       # Zakotwiczenie do lewej
            x=0                   # Pozycja na osi X (0 = start od lewej)
        ),
        xaxis_title="Czas [s]",
        yaxis_title="Amplituda [mV]"
    )

    st.markdown("###### Filtracja sygnału")
    with st.container(border=True):
        st.plotly_chart(fig, use_container_width=True)
    
#%%-----------------------------SEKCJA 3 - Załamki R---------------------------
st.markdown("""
    <hr style="margin-top: 10px;height:5px; border:none; color:#444444; background-color:#444444;" />
""", unsafe_allow_html=True)
            
col1, col2 = st.columns([ 4 ,4.5 ])

with col1:    # Wyciągamy wartości .values, aby uniknąć błędów z indeksami
    st.markdown('<p style="margin-top: px; font-size: 18px; font-weight: bold; color: #0092ff;"> Identyfikacja załamków R i tworzenie szeregu RR</p>', unsafe_allow_html=True)

    col_left, col_right = st.columns([ 1 ,4 ])

    with col_left:  
        st.markdown(...) # Twoje tło div pozostaje bez zmian
        
        # Zamiast kolumn, po prostu używamy kontenera dla suwaków
        with st.container():
            st.markdown('<div style="margin-top: 20px;"></div>', unsafe_allow_html=True)
            threshold_rr = st.slider("Próg dla pików R:", min_value=0.0, max_value=2.0, value=0.11, step=0.01)
            distance_rr = st.slider("Dystans między RR:", min_value=0.0, max_value=2000.0, value=450.0, step=10.0)


    

        sygnal = df['ecg_filtrowany'].values
    
        peaks, _ = find_peaks(sygnal, distance=distance_rr, height=threshold_rr)
    with col_right:    
        fig = go.Figure()

        fig.add_trace(go.Scatter(x=df['czas'], y=df['ecg_filtrowany'], mode='lines',name='Sygnał EKG',
        line=dict(color='#C3E5FF', width=1.5)))

    # 4. Dodajemy wykryte szczyty (Czerwone kropki)
        fig.add_trace(go.Scatter(
        x=df['czas'].iloc[peaks],
        y=df['ecg_filtrowany'].iloc[peaks],
        mode='markers',
        name='Piki R',
        marker=dict(
            color=niebieski, 
            size=8, 
            symbol='circle',
            line=dict(color='white', width=1) # Biała obwódka dla lepszego kontrastu
        )
    ))
        fig.add_hline(y=threshold_rr, line_dash="dash", line_color="rgba(255,255,255,0.3)", 
                  annotation_text="Aktualny próg")

    # 5. Stylizacja (Legenda po lewej, brak zbędnych marginesów)
        fig.update_layout(
        height=200,
        margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0),
        xaxis_title="Czas [s]",
        yaxis_title="Amplituda [mV]")

        with st.container(border=True):
            st.plotly_chart(fig, use_container_width=True)





    czasy_pikow = df['czas'].iloc[peaks].values
    odstepy_rr = np.diff(czasy_pikow)
    
    df_rr = pd.DataFrame({
    '#': range(1, len(odstepy_rr) + 1),
    'rr_ms': odstepy_rr * 1000,  # Konwersja na milisekundy (standard w medycynie)
    'rr_s': odstepy_rr           # Odstępy w sekundach
})

    rr_intervals = df_rr['rr_ms'].values
    
    
    st.markdown(f"""
                <div style="background-color: {lekki_szary}; 
                    border-radius: 10px; 
                    padding: 40px;
                    margin-bottom: -1820px; /* Trik, żeby 'podłożyć' tło pod wykres */
                    height: 300px;
                    border: 0px solid rgba(100,100,100,1);
                ">
                </div>
            """, unsafe_allow_html=True)  
    
    
    lewy, srodek, prawy = st.columns([0.02, 0.9, 0.02])

    with srodek:
        fig = go.Figure()

        fig.add_trace(go.Scatter(
    x=df['czas'].iloc[peaks],
    y=df_rr['rr_ms'].values,
    mode='lines+markers',
    name='Odstępy RR',
    line=dict(color=bialy, width=2), 
    marker=dict(size=6, color=niebieski, symbol='circle') # punkty
))

# 3. Stylizacja wykresu (Layout)
        fig.update_layout(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    xaxis_title="Czas badania [s]",
    yaxis_title="Odstęp RR [ms]",
    template="plotly_dark", 
    hovermode="x unified",
    margin=dict(l=30, r=20, t=30, b=90),
    height=300
)

# 4. Wyświetlenie w Streamlit
        st.plotly_chart(fig, use_container_width=True)
    

with col2:
    st.markdown(f'<p style="margin-top: 0px; font-size: 18px; font-weight: bold; color:#0092ff;">Histogram</p>', unsafe_allow_html=True)

    st.markdown(f"""
                <div style="background-color: {lekki_szary}; 
                    border-radius: 10px; 
                    padding: 40px;
                    margin-bottom: -1820px; /* Trik, żeby 'podłożyć' tło pod wykres */
                    height: 550px;
                    border: 0px solid rgba(100,100,100,1);
                ">
                </div>
            """, unsafe_allow_html=True)  
    
    lewy, srodek, prawy = st.columns([0.02, 0.9, 0.02])  
    
    with srodek:
    
        col_rr1, col_rr2 = st.columns([1., 1.8])

        with col_rr1:
            st.dataframe(df_rr, height=310, use_container_width=True)

        
        
        with col_rr2:
    # Podstawowe statystyki

            histogram_bins = st.slider('Histogram', min_value=20,max_value=300,value=180, step=1)

            fig_hist = px.histogram(
    df_rr, 
    x="rr_ms", 
    nbins=histogram_bins, # Możesz zwiększyć tę liczbę dla większej precyzji
    labels={'rr_ms': 'Odstęp RR [ms]'},
    color_discrete_sequence=[niebieski], # Czerwony pasujący do pików
    marginal="rug" # Dodaje małe kreski na dole pokazujące konkretne uderzenia
)

# 2. Stylizacja (podciągamy do góry i dopasowujemy do dashboardu)
            fig_hist.update_layout(
    height=250,
    margin=dict(l=0, r=0, t=0, b=0),
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    xaxis_title="Czas trwania  [ms]",
    yaxis_title="Częstość",
    bargap=0.1 # Odstęp między słupkami
)

# 3. Wyświetlenie
            with st.container(border=True):
                st.plotly_chart(fig_hist, use_container_width=True)
    
        srednie_rr = df_rr['rr_ms'].mean()
        sdnn = df_rr['rr_ms'].std()
        max_rr=df_rr['rr_ms'].max()
        min_rr=df_rr['rr_ms'].min()
        liczba_R=df_rr.shape[0]
        
        st.markdown(f"""
        <hr style="margin-top: 10px;height:5px; border:none; color: {niebieski}; background-color:{niebieski};" />
    """, unsafe_allow_html=True)

    
        cola, colb,colc,cold,cole = st.columns([1,1,1,1,2])
        with cola:
            st.metric("Średnie RR", f"{srednie_rr:.0f} ms")
        with colb: 
            st.metric("Std RR", f"{sdnn:.0f} ms")
        with colc:
            st.metric("Max RR", f"{max_rr:.0f} ms")        
        with cold: 
            st.metric("Min RR", f"{min_rr:.0f} ms")
        with cole:
            st.metric("Liczba zidentyfikowanych załamków R", f"{liczba_R:.0f}")
    
        st.markdown(f"""
            <hr style="margin-top: 10px;height:5px; border:none; color: {niebieski}; background-color:{niebieski};" />
        """, unsafe_allow_html=True)

#%%-----------------------------SEKCJA 3 - Segmentacja zespołu QRS-------------

st.markdown(f'<p style="margin-top: 0px; font-size: 18px; font-weight: bold; color:{zielony};">Segmentacja zespołu QRS</p>', unsafe_allow_html=True)

st.markdown(f"""
            <hr style="margin-top: 10px;height:5px; border:none; color: {niebieski}; background-color:{zielony};" />
        """, unsafe_allow_html=True)
col1, col2 = st.columns([4, 4.5])

with col1:
    
    col_left, col_right = st.columns([1, 1])

    with col_left:
           
        window = st.slider("Próg dla pików R:", min_value=100,max_value=1000,value=250, step=10)


    # Jeśli częstotliwość to np. 500Hz, window=125 da nam 250ms przed i po załamku R
        ecg_signal = df['ecg_filtrowany'].values
        qrs_dict = {} # Słownik do szybkiego budowania DataFrame

    # 2. Ekstrakcja segmentów
        for i, r in enumerate(peaks):
            # Sprawdzamy granice sygnału
            if r > window and r + window < len(ecg_signal):
                segment = ecg_signal[r - window : r + window].copy()
            
            # Opcjonalna korekcja linii izoelektrycznej (odejmujemy średnią)
                segment = segment - np.mean(segment)
            
            # Zapisujemy do słownika: klucz to nazwa kolumny, wartość to segment
                qrs_dict[f'QRS_{i+1:02d}'] = segment

    # 3. Tworzenie DataFrame
    # Indeks ustawiamy od -window do +window, żeby środek (R) był w zerze
        df_qrs = pd.DataFrame(qrs_dict, index=range(-window, window))
        
        
    with col_right:
            
        idx_segmentu = st.slider("Wybierz numer zespołu QRS:", 0, len(peaks)-1, 2)

    # 2. Pobranie lokalizacji wybranego załamka R
    r_center = peaks[idx_segmentu]
    start_idx = max(0, r_center - window)
    stop_idx = min(len(df), r_center + window)

    # 3. Tworzenie wykresu
    fig_seg = go.Figure()

    # Ścieżka całego sygnału (szary, żeby nie odciągał uwagi)
    fig_seg.add_trace(go.Scatter(
        x=df['czas'], 
        y=df['ecg_filtrowany'],
        mode='lines',
        line=dict(color='rgba(150, 150, 150, 0.5)', width=1),
        name='Pełny sygnał'
    ))

    # Podświetlony segment (wybrany kolor, np. Twój zielony)
    fig_seg.add_trace(go.Scatter(
        x=df['czas'].iloc[start_idx:stop_idx],
        y=df['ecg_filtrowany'].iloc[start_idx:stop_idx],
        mode='lines',
        line=dict(color=zielony, width=3),
        name='Wybrany segment'
    ))

    # 4. Dodanie prostokąta podświetlającego (Shape)
    fig_seg.add_vrect(
        x0=df['czas'].iloc[start_idx], 
        x1=df['czas'].iloc[stop_idx],
        fillcolor=zielony, opacity=0.2, # Twój żółty jako tło
        layer="below", line_width=0,
    )

    # 5. Ustawienia wyglądu
    fig_seg.update_layout(
        title=f"Podgląd segmentu nr {idx_segmentu + 1} (R w {df['czas'].iloc[r_center]:.2f} s)",
        xaxis_title="Czas [s]",
        yaxis_title="Amplituda",
        template="plotly_dark",
        height=300,
        legend=dict(
            orientation="h",      # Legenda w poziomie
            yanchor="bottom",
            y=1.02,               # Nad wykresem
            xanchor="left",       # Zakotwiczenie do lewej
            x=0                   # Pozycja na osi X (0 = start od lewej)
        ),

    )

    st.plotly_chart(fig_seg, use_container_width=True)    

    
with col2:

    
    lewy, prawy = st.columns([1, 1])  

    with lewy:

        df_qrs['SREDNI_QRS'] = df_qrs.mean(axis=1)
        y_min_sredni = df_qrs['SREDNI_QRS'].min()
        y_max_sredni = df_qrs['SREDNI_QRS'].max()
        margines = (y_max_sredni - y_min_sredni) * 0.15

# 2. Tworzymy bazowy wykres Plotly Express
        fig_qrs = px.line(df_qrs, 
                  labels={'index': 'Próbki względem R', 'value': 'Amplituda'},
                  title="Nałożone segmenty QRS z uśrednionym profilem")

# 3. Ustawiamy WSZYSTKIE linie na szaro i lekko przezroczyste
        fig_qrs.update_traces(line=dict(width=1, color='rgba(150, 150, 150, 0.3)'), opacity=0.4)
        fig_qrs.update_layout(
    yaxis=dict(
        range=[y_min_sredni - margines, y_max_sredni + margines],
        fixedrange=False # Pozwalamy użytkownikowi na ręczny zoom, ale startujemy z "uśrednionej" skali
    ),
    template="plotly_dark",
    uirevision='constant'
)

# 4. Wyróżniamy tylko linię średnią (nadajemy jej Twój zielony i większą grubość)
        fig_qrs.for_each_trace(
    lambda trace: trace.update(line=dict(color=zielony, width=4), opacity=1) 
    if trace.name == 'SREDNI_QRS' else ()
)

# 5. Opcjonalnie: Przesuń średni profil na sam wierzch (żeby nie był przykryty szarymi)
        fig_qrs.data = [t for t in fig_qrs.data if t.name != 'SREDNI_QRS'] + \
               [t for t in fig_qrs.data if t.name == 'SREDNI_QRS']

# Wyświetlenie w Streamlit
        st.plotly_chart(fig_qrs, use_container_width=True)
        
        
        
    with prawy:
        wybrana_kolumna = f'QRS_{idx_segmentu + 1:02d}'
        # 2. Wyciągamy dane dla tej kolumny
        y_values = df_qrs[wybrana_kolumna].values
        x_values = df_qrs.index  # To są nasze próbki od -window do +window

# 3. Tworzymy wykres pojedynczego zespołu
        fig_single = go.Figure()

        fig_single.add_trace(go.Scatter(
    x=list(x_values), 
    y=list(y_values),
    mode='lines',
    line=dict(color=zielony, width=4), # Twój zielony, gruby profil
    name=wybrana_kolumna,
    fill='tozeroy', # Opcjonalnie: wypełnienie pod wykresem dla lepszego efektu
    fillcolor='rgba(46, 204, 113, 0.2)'
))

# 4. Stylizacja "laboratoryjna"
        fig_single.update_layout(
    title=f"Analiza morfologii: {wybrana_kolumna}",
    xaxis_title="Próbki względem załamka R [n]",
    yaxis_title="Amplituda [mV]",
    template="plotly_dark",
    height=400,
    showlegend=False,
    # Dodajemy linię zero (izoelektryczną) dla orientacji
    shapes=[dict(
        type='line', yref='y', y0=0, y1=0, xref='x', x0=x_values.min(), x1=x_values.max(),
        line=dict(color="white", width=1, dash="dot")
    )]
)

# 5. Wyświetlenie w Streamlit
        st.plotly_chart(fig_single, use_container_width=True)
        
        
        
        
        
# 1. Przygotowanie danych do macierzy (Z)
# Transponujemy, aby oś X była czasem wewnątrz QRS, a oś Y numerem uderzenia
# Usuwamy kolumnę 'SREDNI_QRS', jeśli ją wcześniej dodałeś, żeby nie psuła wykresu
z_data = df_qrs.drop(columns=['SREDNI_QRS'], errors='ignore').values.T 

# 2. Tworzenie osi
x_axis = df_qrs.index  # Próbki względem R (-150 do 150)
y_axis = np.arange(z_data.shape[0])  # Kolejne numery uderzeń

# 3. Tworzenie wykresu Surface (Powierzchnia)
fig_3d = go.Figure(data=[go.Surface(
    z=z_data, 
    x=x_axis, 
    y=y_axis,
    colorscale='Rainbow', # Możesz zmienić na 'Greens', jeśli wolisz swój styl
    colorbar=dict(title="Amplituda"),
    opacity=0.9
)])

# 4. Stylizacja widoku
fig_3d.update_layout(
    title='Trójwymiarowa segmentacja zespołów QRS',
    scene=dict(
        xaxis_title='Czas wewnątrz QRS [n]',
        yaxis_title='Numer uderzenia',
        zaxis_title='Amplituda',
        # Ustawienie początkowego kąta patrzenia
        camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
    ),
    template="plotly_dark",
    margin=dict(l=0, r=0, b=0, t=40),
    height=700
)

# -------------------------------------------------------------
# Wyświetlenie tabeli z danymi QRS (bez obciążającego stylowania)
st.dataframe(df_qrs, use_container_width=True)
# -------------------------------------------------------------


#%%------------------------------------EMD-------------------------------------

y = df['ecg'].to_numpy() # Wyciągamy czysty wektor liczb
plt.plot(y)
imf = emd.sift.sift(y)
emd.plotting.plot_imfs(imf[:, :])


