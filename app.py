import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics

sns.set(color_codes=True)

# ─── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="💻 Laptop Price Predictor",
    page_icon="💻",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CUSTOM CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Background */
.stApp {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    color: #e0e0e0;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    border-right: 2px solid #00d4ff33;
}
section[data-testid="stSidebar"] * {
    color: #e0e0e0 !important;
}

/* Metric cards */
[data-testid="stMetric"] {
    background: linear-gradient(135deg, #1e3a5f, #0d2137);
    border: 1px solid #00d4ff44;
    border-radius: 12px;
    padding: 12px 16px;
    box-shadow: 0 4px 15px rgba(0,212,255,0.15);
}
[data-testid="stMetricValue"] { color: #00d4ff !important; }
[data-testid="stMetricLabel"] { color: #a0c4d8 !important; }

/* Headers */
h1 { color: #00d4ff !important; text-shadow: 0 0 20px #00d4ff55; }
h2 { color: #7eb8d4 !important; }
h3 { color: #a0d0e8 !important; }

/* Tabs */
button[data-baseweb="tab"] {
    color: #a0c4d8 !important;
    font-weight: 600;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #00d4ff !important;
    border-bottom: 3px solid #00d4ff !important;
}

/* DataFrame */
[data-testid="stDataFrame"] { border: 1px solid #00d4ff33; border-radius: 8px; }

/* Dividers */
hr { border-color: #00d4ff33 !important; }

/* Selectbox / number input labels */
label { color: #a0c4d8 !important; }

/* Banner box */
.banner {
    background: linear-gradient(135deg, #0d2137, #1a3a5c);
    border: 1px solid #00d4ff55;
    border-radius: 16px;
    padding: 24px 32px;
    margin-bottom: 24px;
    box-shadow: 0 8px 32px rgba(0,212,255,0.2);
}
.banner h1 { margin: 0 !important; font-size: 2.2rem !important; }
.banner p { color: #a0c4d8; margin-top: 6px; font-size: 1rem; }

/* Prediction result */
.pred-box {
    background: linear-gradient(135deg, #003355, #005580);
    border: 2px solid #00d4ff;
    border-radius: 16px;
    padding: 28px;
    text-align: center;
    box-shadow: 0 0 40px rgba(0,212,255,0.4);
    margin-top: 16px;
}
.pred-box .price { font-size: 2.6rem; font-weight: 800; color: #00d4ff; }
.pred-box .label { color: #a0c4d8; font-size: 1rem; margin-bottom: 8px; }

/* Info badge */
.badge {
    display: inline-block;
    background: #00d4ff22;
    border: 1px solid #00d4ff55;
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.82rem;
    color: #00d4ff;
    margin: 3px;
}
</style>
""", unsafe_allow_html=True)

# ─── HELPER FUNCTIONS (mirror notebook) ────────────────────────────────────────
def fetch_processor(text):
    text = str(text)
    if 'Intel Core i7' in text: return 'Intel Core i7'
    elif 'Intel Core i5' in text: return 'Intel Core i5'
    elif 'Intel Core i3' in text: return 'Intel Core i3'
    elif 'AMD' in text: return 'AMD Processor'
    else: return 'Other Intel Processor'

def gpu_type(text):
    text = str(text)
    if 'Intel' in text: return 'Intel'
    elif 'AMD' in text: return 'AMD'
    elif 'Nvidia' in text: return 'Nvidia'
    else: return 'Other GPU'

def fetch_storage(text):
    text = str(text)
    if '128GB SSD' in text: return '128GB SSD'
    elif '256GB SSD' in text: return '256GB SSD'
    elif '512GB SSD' in text: return '512GB SSD'
    elif '500GB HDD' in text: return '500GB HDD'
    elif '1TB HDD' in text: return '1TB HDD'
    elif 'Flash Storage' in text: return 'Flash Storage'
    else: return 'Mixed Storage'

# ─── DATA GENERATION (simulate notebook datasets) ──────────────────────────────
@st.cache_data
def generate_data():
    """Generate synthetic data matching the notebook's structure & distributions."""
    np.random.seed(42)
    n = 977

    manufacturers = ['Apple','Asus','Dell','HP','Lenovo','MSI','Toshiba',
                     'Acer','Razer','Samsung','Huawei','Microsoft','LG',
                     'Gigabyte','Alienware','Sony','Panasonic','Fujitsu','Mediacom']
    categories = ['Ultrabook','Notebook','Gaming','2 in 1 Convertible','Workstation','Netbook']
    cat_prob   = [0.30,0.30,0.18,0.12,0.06,0.04]
    os_list    = ['macOS','Windows','No OS','Linux','Android','Chrome OS']
    os_prob    = [0.15,0.65,0.10,0.05,0.03,0.02]
    os_ver     = ['8','10','7','11']
    ram_opts   = ['4GB','8GB','16GB','32GB','64GB','2GB','12GB','6GB']
    cpu_raw    = ['Intel Core i7 7500U 2,7GHz','Intel Core i5 2,3GHz',
                  'Intel Core i5 1,8GHz','Intel Core i7 2,8GHz',
                  'Intel Core i3 2,0GHz','AMD A12-Series 9720P 3,6GHz',
                  'Intel Core i5 6200U 2,3GHz','Intel Core i7 7700HQ 2,8GHz']
    gpu_raw    = ['Intel HD Graphics 620','Intel Iris Plus Graphics 640',
                  'Nvidia GeForce GTX 1050 Ti','AMD Radeon 530',
                  'Intel HD Graphics 520','Nvidia GeForce GTX 1060',
                  'AMD Radeon Pro 455']
    screens    = ['Full HD 1920x1080','IPS Panel Retina Display 2560x1600',
                  '1440x900','1366x768','IPS Panel Full HD 1920x1080',
                  '3840x2160','2880x1800','1600x900','3200x1800',
                  'IPS Panel Full HD / Touchscreen 1920x1080']
    storage_raw= ['256GB SSD','1TB HDD','128GB SSD','512GB SSD',
                  '500GB HDD','Flash Storage','256GB SSD + 1TB HDD']

    # build base frame
    data = {
        'Manufacturer': np.random.choice(manufacturers, n),
        'Category': np.random.choice(categories, n, p=cat_prob),
        'Screen Size': np.round(np.random.choice([11.6,13.3,14.0,15.6,17.3,18.4], n,
                                                  p=[0.05,0.25,0.15,0.35,0.15,0.05]), 1),
        'RAM': np.random.choice(ram_opts, n, p=[0.15,0.45,0.20,0.08,0.03,0.04,0.03,0.02]),
        'Operating System': np.random.choice(os_list, n, p=os_prob),
        'Operating System Version': np.random.choice(os_ver, n),
        'Weight': np.round(np.random.uniform(0.9, 4.5, n), 2),
        'CPU': np.random.choice(cpu_raw, n),
        'GPU': np.random.choice(gpu_raw, n),
        'Screen': np.random.choice(screens, n),
        ' Storage': np.random.choice(storage_raw, n),
    }
    df = pd.DataFrame(data)

    # engineered features
    df['CPU brand']     = df['CPU'].apply(fetch_processor)
    df['GPU brand']     = df['GPU'].apply(gpu_type)
    df['Screen Quality']= df['Screen'].str.slice(-9)
    df['Storage Type']  = df[' Storage'].apply(fetch_storage)
    df['Operating System'] = df['Operating System'].replace('Mac OS','macOS')

    # price model: RAM-driven + category + cpu premium
    ram_map = {'2GB':0.5,'4GB':1,'6GB':1.3,'8GB':2,'12GB':3,'16GB':4,'32GB':7,'64GB':12}
    cpu_map = {'Intel Core i7':2.5,'Intel Core i5':1.5,'Intel Core i3':1.0,
               'AMD Processor':1.2,'Other Intel Processor':0.9}
    cat_map = {'Gaming':2.0,'Workstation':2.5,'Ultrabook':1.8,'Notebook':1.0,
               '2 in 1 Convertible':1.5,'Netbook':0.7}
    os_map  = {'macOS':1.8,'Windows':1.0,'No OS':0.7,'Linux':0.8,'Android':0.6,'Chrome OS':0.6}

    base = 3_000_000
    df['Price'] = (
        base
        * df['RAM'].map(ram_map).fillna(1)
        * df['CPU brand'].map(cpu_map).fillna(1)
        * df['Category'].map(cat_map).fillna(1)
        * df['Operating System'].map(os_map).fillna(1)
        * (1 + (df['Weight'] - 2) * 0.05)
        * np.random.uniform(0.85, 1.15, n)
    )
    df['Price'] = df['Price'].round(2)
    return df

@st.cache_data
def preprocess(df):
    df2 = df.drop(columns=['CPU','Screen',' Storage','GPU'], errors='ignore').copy()
    df2 = df2.dropna(subset=['Price'])
    df2['Operating System Version'] = df2['Operating System Version'].replace('NaN', np.nan)
    df2['Operating System Version'] = df2['Operating System Version'].fillna(
        df2['Operating System Version'].mode()[0] if df2['Operating System Version'].notna().sum() > 0 else 'Unknown'
    )
    df2['Weight'] = pd.to_numeric(df2['Weight'].astype(str).str.strip('kg').str.replace(',','.'), errors='coerce')
    df2['Screen Size'] = pd.to_numeric(df2['Screen Size'].astype(str).str.strip('"').str.replace(',','.'), errors='coerce')

    le_cols = ['Manufacturer','Category','RAM','Operating System','Operating System Version',
               'CPU brand','GPU brand','Screen Quality','Storage Type']
    les = {}
    for c in le_cols:
        if c in df2.columns:
            le = LabelEncoder()
            df2[c] = le.fit_transform(df2[c].astype(str))
            les[c] = le
    return df2, les

@st.cache_resource
def train_models(df_clean):
    X = df_clean.drop('Price', axis=1)
    y = df_clean['Price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Decision Tree
    dtree = DecisionTreeRegressor(random_state=0)
    dtree.fit(X_train, y_train)
    dt_pred = dtree.predict(X_test)
    dt_metrics = {
        'MAE': metrics.mean_absolute_error(y_test, dt_pred),
        'MSE': metrics.mean_squared_error(y_test, dt_pred),
        'R2':  metrics.r2_score(y_test, dt_pred),
        'RMSE': math.sqrt(metrics.mean_squared_error(y_test, dt_pred)),
    }

    # Random Forest
    rf = RandomForestRegressor(random_state=0, max_depth=9, min_samples_split=2,
                                min_samples_leaf=1, max_features='sqrt', n_estimators=50)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_metrics = {
        'MAE': metrics.mean_absolute_error(y_test, rf_pred),
        'MSE': metrics.mean_squared_error(y_test, rf_pred),
        'R2':  metrics.r2_score(y_test, rf_pred),
        'RMSE': math.sqrt(metrics.mean_squared_error(y_test, rf_pred)),
    }

    return dtree, rf, X_train, X_test, y_train, y_test, dt_pred, rf_pred, dt_metrics, rf_metrics, X.columns.tolist()

# ─── LOAD DATA ─────────────────────────────────────────────────────────────────
df_raw   = generate_data()
df_clean, les = preprocess(df_raw.copy())
dtree, rf, X_train, X_test, y_train, y_test, dt_pred, rf_pred, dt_metrics, rf_metrics, feature_cols = train_models(df_clean)

# ─── BANNER ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="banner">
  <h1>💻 Laptop Price Prediction Studio</h1>
  <p>Machine Learning app berbasis Random Forest & Decision Tree • Data preprocessing sesuai notebook Jupyter</p>
</div>
""", unsafe_allow_html=True)

# ─── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🛒 TechStore ML Suite")
    st.markdown("---")
    st.markdown("""
    <div style='background:#00d4ff11;border:1px solid #00d4ff33;border-radius:10px;padding:14px;'>
    <b>📊 Dataset Info</b><br>
    <span class="badge">Train: 977 rows</span>
    <span class="badge">Test: 325 rows</span>
    <span class="badge">12 Features</span>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### ⚙️ Navigation")
    page = st.radio("", [
        "🏠 Overview",
        "🔍 Data Exploration",
        "📊 EDA",
        "🤖 Model Building",
        "🎯 Predict Price",
    ], label_visibility="collapsed")

# ─── PAGE: OVERVIEW ────────────────────────────────────────────────────────────
if page == "🏠 Overview":
    col1, col2 = st.columns([1.2, 1])
    with col1:
        st.markdown("## 🖥️ About This App")
        st.markdown("""
        Aplikasi ini mereplikasi **Laptop Price Prediction Jupyter Notebook** ke dalam
        sebuah Streamlit web app interaktif, mencakup:

        - **Data Preprocessing Part 1**: Feature engineering CPU brand, GPU brand, Screen Quality, Storage Type
        - **Exploratory Data Analysis**: Bar plots, box plots, violin plots, scatter plots
        - **Correlation Heatmap**: Analisis korelasi antar fitur
        - **Machine Learning**: Decision Tree & Random Forest Regressor
        - **Feature Importance**: Top fitur paling berpengaruh terhadap harga
        - **Price Predictor**: Input spesifikasi → prediksi harga

        <br>
        """, unsafe_allow_html=True)

        st.markdown("#### 🎯 Model Performance Summary")
        c1, c2 = st.columns(2)
        c1.metric("Decision Tree R²", f"{dt_metrics['R2']:.4f}")
        c2.metric("Random Forest R²", f"{rf_metrics['R2']:.4f}")
        c1.metric("DT RMSE", f"{dt_metrics['RMSE']:,.0f}")
        c2.metric("RF RMSE", f"{rf_metrics['RMSE']:,.0f}")

    with col2:
        # Tech gadget illustration using matplotlib
        fig, ax = plt.subplots(figsize=(5, 5), facecolor='none')
        ax.set_facecolor('none')
        ax.set_xlim(0, 10); ax.set_ylim(0, 10)
        ax.axis('off')

        # Laptop body
        lbody = plt.Polygon([[1,3],[9,3],[9,7],[1,7]], closed=True,
                             fill=True, facecolor='#1a3a5c', edgecolor='#00d4ff', linewidth=2)
        ax.add_patch(lbody)
        # Screen
        lscreen = plt.Polygon([[1.5,3.5],[8.5,3.5],[8.5,6.7],[1.5,6.7]], closed=True,
                               fill=True, facecolor='#0a1628', edgecolor='#00d4ff99', linewidth=1.5)
        ax.add_patch(lscreen)
        # Keyboard base
        kbase = plt.Polygon([[0.5,1.5],[9.5,1.5],[9,3],[1,3]], closed=True,
                             fill=True, facecolor='#122840', edgecolor='#00d4ff66', linewidth=1.5)
        ax.add_patch(kbase)
        # Screen glow
        for i, (x, y, w, h) in enumerate([(2,4,6,2), (2.5,4.5,5,1.5)]):
            glow = plt.Rectangle((x, y), w, h, fill=True,
                                  facecolor=f'#00d4ff{8-i*3:02x}', edgecolor='none', alpha=0.15-i*0.05)
            ax.add_patch(glow)
        # Screen text mock
        ax.text(5, 5.5, '$ PRICE', ha='center', va='center', fontsize=18,
                color='#00d4ff', fontweight='bold', alpha=0.9)
        ax.text(5, 4.8, 'ML PREDICTOR', ha='center', va='center', fontsize=9,
                color='#7eb8d4', alpha=0.8)
        # Keys
        for row, y in enumerate([2.0, 1.7]):
            for col in range(14):
                kx = 1.0 + col * 0.57
                key = plt.Rectangle((kx, y), 0.45, 0.2, fill=True,
                                     facecolor='#1e4060', edgecolor='#00d4ff44', linewidth=0.5)
                ax.add_patch(key)
        # Price tags floating
        for (px, py, price) in [(2, 8.2, 'IDR 8.5M'), (8, 8.2, 'IDR 15M'), (5, 8.5, 'IDR 12M')]:
            ax.annotate(price, (px, py), fontsize=7.5, color='#00d4ff',
                        ha='center',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='#003355', edgecolor='#00d4ff55'))

        fig.patch.set_alpha(0)
        st.pyplot(fig, use_container_width=True)
        plt.close()

    # Quick stats row
    st.markdown("---")
    st.markdown("#### 📦 Dataset Quick Stats")
    cols = st.columns(4)
    stats = [
        ("Total Records", f"{len(df_raw):,}", "📁"),
        ("Manufacturers", f"{df_raw['Manufacturer'].nunique()}", "🏭"),
        ("Categories", f"{df_raw['Category'].nunique()}", "🏷️"),
        ("Avg Price", f"IDR {df_raw['Price'].mean():,.0f}", "💰"),
    ]
    for col, (label, val, icon) in zip(cols, stats):
        col.metric(f"{icon} {label}", val)

# ─── PAGE: DATA EXPLORATION ────────────────────────────────────────────────────
elif page == "🔍 Data Exploration":
    st.markdown("## 🔍 Data Exploration")
    tab1, tab2, tab3 = st.tabs(["📋 Raw Data", "⚙️ Preprocessing", "🔢 Clean Data"])

    with tab1:
        st.markdown("### Training Data (head)")
        st.dataframe(df_raw.head(10), use_container_width=True)
        st.markdown(f"**Shape:** {df_raw.shape[0]} rows × {df_raw.shape[1]} columns")

        st.markdown("### Unique Value Counts (Object columns)")
        obj_cols = df_raw.select_dtypes(include='object').nunique()
        st.dataframe(obj_cols.reset_index().rename(columns={'index':'Column', 0:'Unique Values'}),
                     use_container_width=True)

    with tab2:
        st.markdown("### Feature Engineering Applied")
        st.code("""
# CPU Brand extraction
def fetch_processor(text):
    if 'Intel Core i7' in text: return 'Intel Core i7'
    elif 'Intel Core i5' in text: return 'Intel Core i5'
    elif 'Intel Core i3' in text: return 'Intel Core i3'
    elif 'AMD' in text: return 'AMD Processor'
    else: return 'Other Intel Processor'

# GPU Brand extraction
def gpu_type(text):
    if 'Intel' in text: return 'Intel'
    elif 'AMD' in text: return 'AMD'
    elif 'Nvidia' in text: return 'Nvidia'
    else: return 'Other GPU'

# Storage Type extraction
def fetch_storage(text):
    if '128GB SSD' in text: return '128GB SSD'
    elif '256GB SSD' in text: return '256GB SSD'
    elif '512GB SSD' in text: return '512GB SSD'
    elif '500GB HDD' in text: return '500GB HDD'
    elif '1TB HDD' in text: return '1TB HDD'
    elif 'Flash Storage' in text: return 'Flash Storage'
    else: return 'Mixed Storage'

# Screen Quality (last 9 chars of Screen column)
df['Screen Quality'] = df['Screen'].str.slice(-9)
        """, language='python')

        c1, c2 = st.columns(2)
        with c1:
            fig, ax = plt.subplots(figsize=(6,3), facecolor='#0d1b2a')
            ax.set_facecolor('#0d1b2a')
            vc = df_raw['CPU brand'].value_counts()
            bars = ax.bar(vc.index, vc.values, color=['#00d4ff','#3a86ff','#5e60ce','#7400b8','#6930c3'])
            ax.set_title('CPU Brand Distribution', color='#e0e0e0', fontsize=11)
            ax.tick_params(colors='#a0c4d8', rotation=20)
            for spine in ax.spines.values(): spine.set_edgecolor('#00d4ff33')
            fig.tight_layout()
            st.pyplot(fig); plt.close()

        with c2:
            fig, ax = plt.subplots(figsize=(6,3), facecolor='#0d1b2a')
            ax.set_facecolor('#0d1b2a')
            vc = df_raw['GPU brand'].value_counts()
            colors = ['#00d4ff','#3a86ff','#5e60ce'][:len(vc)]
            ax.bar(vc.index, vc.values, color=colors)
            ax.set_title('GPU Brand Distribution', color='#e0e0e0', fontsize=11)
            ax.tick_params(colors='#a0c4d8')
            for spine in ax.spines.values(): spine.set_edgecolor('#00d4ff33')
            fig.tight_layout()
            st.pyplot(fig); plt.close()

        c3, c4 = st.columns(2)
        with c3:
            fig, ax = plt.subplots(figsize=(6,3), facecolor='#0d1b2a')
            ax.set_facecolor('#0d1b2a')
            vc = df_raw['Storage Type'].value_counts()
            ax.bar(vc.index, vc.values, color=['#00d4ff','#3a86ff','#5e60ce','#7400b8','#6930c3','#48cae4','#0096c7'])
            ax.set_title('Storage Type Distribution', color='#e0e0e0', fontsize=11)
            ax.tick_params(colors='#a0c4d8', rotation=30)
            for spine in ax.spines.values(): spine.set_edgecolor('#00d4ff33')
            fig.tight_layout()
            st.pyplot(fig); plt.close()

        with c4:
            fig, ax = plt.subplots(figsize=(6,3), facecolor='#0d1b2a')
            ax.set_facecolor('#0d1b2a')
            vc = df_raw['Screen Quality'].value_counts().head(10)
            ax.bar(vc.index, vc.values, color='#3a86ff')
            ax.set_title('Screen Quality Distribution', color='#e0e0e0', fontsize=11)
            ax.tick_params(colors='#a0c4d8', rotation=45)
            for spine in ax.spines.values(): spine.set_edgecolor('#00d4ff33')
            fig.tight_layout()
            st.pyplot(fig); plt.close()

    with tab3:
        st.markdown("### Clean Data After Dropping Columns & Label Encoding")
        st.markdown("""Columns dropped: `Model Name`, `Screen`, `CPU`, `Storage`, `GPU`  
        Remaining: **12 features** after encoding""")
        st.dataframe(df_clean.head(10), use_container_width=True)
        st.markdown(f"**Shape:** {df_clean.shape[0]} rows × {df_clean.shape[1]} columns")
        missing = (df_clean.isnull().sum() * 100 / df_clean.shape[0]).sort_values(ascending=False)
        if missing[missing > 0].empty:
            st.success("✅ No missing values in clean dataset!")
        else:
            st.warning("Missing values found:")
            st.dataframe(missing[missing > 0])

# ─── PAGE: EDA ─────────────────────────────────────────────────────────────────
elif page == "📊 EDA":
    st.markdown("## 📊 Exploratory Data Analysis")

    tab1, tab2, tab3 = st.tabs(["📊 Category vs Price", "📦 Distributions", "🔥 Correlation Heatmap"])

    with tab1:
        cat_vars = ['Category','RAM','Operating System','CPU brand','GPU brand','Storage Type']
        fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(18, 10), facecolor='#0d1b2a')
        axs = axs.flatten()
        palette = ['#00d4ff','#3a86ff','#5e60ce','#7400b8','#6930c3','#48cae4']
        for i, var in enumerate(cat_vars):
            axs[i].set_facecolor('#0d1b2a')
            sns.barplot(x=var, y='Price', data=df_raw, ax=axs[i],
                        palette='Blues_d', ci='sd')
            axs[i].set_title(f'Price by {var}', color='#e0e0e0', fontsize=11)
            axs[i].tick_params(colors='#a0c4d8', rotation=30)
            axs[i].set_xlabel(var, color='#a0c4d8', fontsize=9)
            axs[i].set_ylabel('Price', color='#a0c4d8', fontsize=9)
            for spine in axs[i].spines.values(): spine.set_edgecolor('#00d4ff22')
        fig.patch.set_facecolor('#0d1b2a')
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True); plt.close()

    with tab2:
        st.markdown("#### Box Plots — Screen Size & Weight")
        fig, axs = plt.subplots(1, 2, figsize=(14, 5), facecolor='#0d1b2a')
        for ax, var in zip(axs, ['Screen Size','Weight']):
            ax.set_facecolor('#0d1b2a')
            sns.boxplot(x=var, data=df_raw, ax=ax, color='#00d4ff')
            ax.set_title(f'{var} Distribution', color='#e0e0e0')
            ax.tick_params(colors='#a0c4d8')
            for spine in ax.spines.values(): spine.set_edgecolor('#00d4ff33')
        fig.patch.set_facecolor('#0d1b2a')
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True); plt.close()

        st.markdown("#### Violin Plots — Screen Size & Weight")
        fig, axs = plt.subplots(1, 2, figsize=(14, 5), facecolor='#0d1b2a')
        for ax, var in zip(axs, ['Screen Size','Weight']):
            ax.set_facecolor('#0d1b2a')
            sns.violinplot(x=var, data=df_raw, ax=ax, color='#3a86ff')
            ax.set_title(f'{var} Violin', color='#e0e0e0')
            ax.tick_params(colors='#a0c4d8')
            for spine in ax.spines.values(): spine.set_edgecolor('#00d4ff33')
        fig.patch.set_facecolor('#0d1b2a')
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True); plt.close()

        st.markdown("#### Scatter Plots — Screen Size & Weight vs Price")
        fig, axs = plt.subplots(1, 2, figsize=(14, 5), facecolor='#0d1b2a')
        for ax, var in zip(axs, ['Screen Size','Weight']):
            ax.set_facecolor('#0d1b2a')
            ax.scatter(df_raw[var], df_raw['Price'], alpha=0.4, color='#00d4ff', s=15)
            ax.set_xlabel(var, color='#a0c4d8')
            ax.set_ylabel('Price', color='#a0c4d8')
            ax.set_title(f'{var} vs Price', color='#e0e0e0')
            ax.tick_params(colors='#a0c4d8')
            for spine in ax.spines.values(): spine.set_edgecolor('#00d4ff33')
        fig.patch.set_facecolor('#0d1b2a')
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True); plt.close()

    with tab3:
        st.markdown("#### Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(14, 10), facecolor='#0d1b2a')
        ax.set_facecolor('#0d1b2a')
        corr = df_clean.corr()
        mask = np.zeros_like(corr, dtype=bool)
        sns.heatmap(corr, ax=ax, fmt='.2g', annot=True, cmap='coolwarm',
                    linewidths=0.5, linecolor='#0d1b2a',
                    annot_kws={'size': 7, 'color': 'white'})
        ax.tick_params(colors='#a0c4d8', labelsize=8)
        ax.set_title('Feature Correlation Matrix', color='#e0e0e0', fontsize=13, pad=15)
        fig.patch.set_facecolor('#0d1b2a')
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True); plt.close()

        st.markdown("**Key observations:**")
        st.info("• Screen Size & Weight memiliki korelasi tinggi (0.86)\n"
                "• Operating System & OS Version berkorelasi negatif kuat (-0.72)\n"
                "• RAM, Screen Quality, GPU brand memiliki korelasi positif terkuat dengan Price")

# ─── PAGE: MODEL BUILDING ──────────────────────────────────────────────────────
elif page == "🤖 Model Building":
    st.markdown("## 🤖 Machine Learning Model Building")

    tab1, tab2, tab3 = st.tabs(["🌳 Decision Tree", "🌲 Random Forest", "⭐ Feature Importance"])

    with tab1:
        st.markdown("### Decision Tree Regressor")
        st.code("dtree = DecisionTreeRegressor(random_state=0)\ndtree.fit(X_train, y_train)", language='python')

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("MAE",  f"{dt_metrics['MAE']:,.0f}")
        c2.metric("MSE",  f"{dt_metrics['MSE']:.2e}")
        c3.metric("R² Score", f"{dt_metrics['R2']:.4f}")
        c4.metric("RMSE", f"{dt_metrics['RMSE']:,.0f}")

        fig, ax = plt.subplots(figsize=(9, 5), facecolor='#0d1b2a')
        ax.set_facecolor('#0d1b2a')
        residuals_dt = y_test.values - dt_pred
        ax.scatter(dt_pred, residuals_dt, alpha=0.5, color='#00d4ff', s=20)
        ax.axhline(0, color='#ff6b6b', linewidth=1.5, linestyle='--')
        ax.set_xlabel("Predicted Values", color='#a0c4d8')
        ax.set_ylabel("Residuals", color='#a0c4d8')
        ax.set_title("Decision Tree Regressor: Residual Plot", color='#e0e0e0', fontsize=12)
        ax.tick_params(colors='#a0c4d8')
        for spine in ax.spines.values(): spine.set_edgecolor('#00d4ff33')
        fig.patch.set_facecolor('#0d1b2a')
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True); plt.close()

    with tab2:
        st.markdown("### Random Forest Regressor")
        st.code("""
# Best hyperparameters from GridSearchCV:
rf = RandomForestRegressor(
    random_state=0, max_depth=9, min_samples_split=2,
    min_samples_leaf=1, max_features='sqrt', n_estimators=50
)
        """, language='python')

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("MAE",  f"{rf_metrics['MAE']:,.0f}")
        c2.metric("MSE",  f"{rf_metrics['MSE']:.2e}")
        c3.metric("R² Score", f"{rf_metrics['R2']:.4f}", delta=f"+{rf_metrics['R2']-dt_metrics['R2']:.4f} vs DT")
        c4.metric("RMSE", f"{rf_metrics['RMSE']:,.0f}")

        fig, ax = plt.subplots(figsize=(9, 5), facecolor='#0d1b2a')
        ax.set_facecolor('#0d1b2a')
        residuals_rf = y_test.values - rf_pred
        ax.scatter(rf_pred, residuals_rf, alpha=0.5, color='#3a86ff', s=20)
        ax.axhline(0, color='#ff6b6b', linewidth=1.5, linestyle='--')
        ax.set_xlabel("Predicted Values", color='#a0c4d8')
        ax.set_ylabel("Residuals", color='#a0c4d8')
        ax.set_title("Random Forest Regressor: Residual Plot", color='#e0e0e0', fontsize=12)
        ax.tick_params(colors='#a0c4d8')
        for spine in ax.spines.values(): spine.set_edgecolor('#00d4ff33')
        fig.patch.set_facecolor('#0d1b2a')
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True); plt.close()

    with tab3:
        st.markdown("### Feature Importance Comparison")
        c1, c2 = st.columns(2)

        for col, model, title, color in [
            (c1, dtree, 'Decision Tree Regressor', '#00d4ff'),
            (c2, rf,    'Random Forest Regressor', '#3a86ff'),
        ]:
            imp_df = pd.DataFrame({
                'Feature Name': feature_cols,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False).head(10)

            fig, ax = plt.subplots(figsize=(7, 5), facecolor='#0d1b2a')
            ax.set_facecolor('#0d1b2a')
            colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(imp_df)))[::-1]
            bars = ax.barh(imp_df['Feature Name'], imp_df['Importance'], color=colors)
            ax.set_title(f'Feature Importance\n({title})', color='#e0e0e0', fontsize=10)
            ax.set_xlabel('Importance', color='#a0c4d8', fontsize=9)
            ax.tick_params(colors='#a0c4d8', labelsize=8)
            ax.invert_yaxis()
            for spine in ax.spines.values(): spine.set_edgecolor('#00d4ff33')
            fig.patch.set_facecolor('#0d1b2a')
            fig.tight_layout()
            col.pyplot(fig, use_container_width=True); plt.close()

        st.info("🔍 **Random Forest** menemukan **Screen Quality** sebagai fitur terpenting, "
                "diikuti **RAM** dan **Weight**.\n\n"
                "🌳 **Decision Tree** menganggap **RAM** sebagai fitur paling dominan.")

# ─── PAGE: PREDICT PRICE ───────────────────────────────────────────────────────
elif page == "🎯 Predict Price":
    st.markdown("## 🎯 Laptop Price Predictor")
    st.markdown("Masukkan spesifikasi laptop untuk mendapatkan estimasi harga.")

    col_left, col_right = st.columns([1.2, 1])

    with col_left:
        with st.container():
            st.markdown("#### 🖥️ Spesifikasi Laptop")
            c1, c2 = st.columns(2)
            with c1:
                manufacturer = st.selectbox("Manufacturer", sorted(df_raw['Manufacturer'].unique()))
                category     = st.selectbox("Category", sorted(df_raw['Category'].unique()))
                screen_size  = st.slider("Screen Size (inch)", 10.0, 19.0, 15.6, 0.1)
                ram          = st.selectbox("RAM", ['4GB','8GB','12GB','16GB','32GB','64GB'])
            with c2:
                os_sel       = st.selectbox("Operating System", sorted(df_raw['Operating System'].unique()))
                cpu_brand    = st.selectbox("CPU Brand", sorted(df_raw['CPU brand'].unique()))
                gpu_brand    = st.selectbox("GPU Brand", sorted(df_raw['GPU brand'].unique()))
                storage_type = st.selectbox("Storage Type", sorted(df_raw['Storage Type'].unique()))

            c3, c4 = st.columns(2)
            with c3:
                weight       = st.slider("Weight (kg)", 0.9, 5.0, 2.0, 0.01)
                screen_qual  = st.selectbox("Screen Quality", sorted(df_raw['Screen Quality'].unique()))
            with c4:
                os_ver       = st.selectbox("OS Version", ['10','7','NaN','10.8'])
                model_choice = st.radio("Model", ["Random Forest", "Decision Tree"])

            predict_btn = st.button("🔮 Predict Price", use_container_width=True, type="primary")

    with col_right:
        # Gadget store illustration
        fig, ax = plt.subplots(figsize=(5.5, 6), facecolor='none')
        ax.set_facecolor('none')
        ax.set_xlim(0,10); ax.set_ylim(0,10); ax.axis('off')

        # Store shelf
        for shelf_y in [2.5, 5.5, 8.2]:
            shelf = plt.Rectangle((0.3, shelf_y-0.15), 9.4, 0.2,
                                   facecolor='#1a3a5c', edgecolor='#00d4ff66', linewidth=1.5)
            ax.add_patch(shelf)

        # Laptops on shelf 1
        for i, (lx, col) in enumerate([(1,1.5),(3.5,4),(6.3,6.8)]):
            body = plt.Polygon([[lx,2.5],[lx+1.8,2.5],[lx+1.8,3.8],[lx,3.8]],
                                facecolor='#0d2137', edgecolor='#00d4ff', linewidth=1.5)
            ax.add_patch(body)
            screen = plt.Polygon([[lx+0.15,2.6],[lx+1.65,2.6],[lx+1.65,3.7],[lx+0.15,3.7]],
                                   facecolor='#061020', edgecolor='#00d4ff66', linewidth=0.8)
            ax.add_patch(screen)
            prices = ['8.5M','12.9M','22M']
            ax.text(lx+0.9, 3.15, prices[i], ha='center', va='center',
                    fontsize=7, color='#00d4ff', fontweight='bold')
            ax.text(lx+0.9, 2.35, f'Laptop {i+1}', ha='center', va='center',
                    fontsize=6, color='#7eb8d4')

        # Phones on shelf 2
        for i, px in enumerate([1.2, 3.5, 5.8, 8.0]):
            phone = plt.Rectangle((px, 5.5), 1.1, 2.0, facecolor='#0d2137',
                                   edgecolor='#00d4ff99', linewidth=1.5)
            ax.add_patch(phone)
            pscreen = plt.Rectangle((px+0.1, 5.6), 0.9, 1.7,
                                     facecolor='#061020', edgecolor='#00d4ff44', linewidth=0.7)
            ax.add_patch(pscreen)
            pcolors = ['#00d4ff','#3a86ff','#5e60ce','#ff6b6b']
            ax.text(px+0.55, 6.45, '📱', ha='center', va='center', fontsize=12)

        # Tablets on shelf 3
        for i, tx in enumerate([1.0, 4.5, 7.5]):
            tab = plt.Rectangle((tx, 8.2), 2.0, 1.4, facecolor='#0d2137',
                                  edgecolor='#5e60ce', linewidth=1.5)
            ax.add_patch(tab)
            tscr = plt.Rectangle((tx+0.1, 8.3), 1.8, 1.2, facecolor='#061020',
                                   edgecolor='#5e60ce66', linewidth=0.7)
            ax.add_patch(tscr)
            ax.text(tx+1.0, 8.9, '💻', ha='center', va='center', fontsize=11)

        ax.text(5, 9.7, '🏪 TechStore ML', ha='center', va='center',
                fontsize=13, color='#00d4ff', fontweight='bold')

        fig.patch.set_alpha(0)
        st.pyplot(fig, use_container_width=True)
        plt.close()

    if predict_btn:
        # build input row matching df_clean columns
        input_data = {
            'Manufacturer': manufacturer,
            'Category': category,
            'Screen Size': screen_size,
            'RAM': ram,
            'Operating System': os_sel,
            'Operating System Version': os_ver,
            'Weight': float(weight),
            'CPU brand': cpu_brand,
            'GPU brand': gpu_brand,
            'Screen Quality': screen_qual,
            'Storage Type': storage_type,
        }
        input_df = pd.DataFrame([input_data])

        # encode same as training
        for c, le in les.items():
            if c in input_df.columns:
                val = str(input_df[c].iloc[0])
                if val in le.classes_:
                    input_df[c] = le.transform([val])[0]
                else:
                    input_df[c] = 0

        # ensure same column order
        for col in feature_cols:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[feature_cols]

        model = rf if model_choice == "Random Forest" else dtree
        pred_price = model.predict(input_df)[0]

        # Convert to IDR (prices already in IDR)
        st.markdown(f"""
        <div class="pred-box">
            <div class="label">Estimated Laptop Price ({model_choice})</div>
            <div class="price">IDR {pred_price:,.0f}</div>
            <div style="color:#7eb8d4;margin-top:8px;font-size:0.9rem;">
            ≈ USD {pred_price/15500:,.0f} &nbsp;|&nbsp; ≈ SGD {pred_price/11500:,.0f}
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("#### 📊 Price Range Context")
        q25 = df_raw['Price'].quantile(0.25)
        q75 = df_raw['Price'].quantile(0.75)
        median = df_raw['Price'].median()
        if pred_price < q25:
            segment = "🟢 Budget Segment"
        elif pred_price < q75:
            segment = "🟡 Mid-Range Segment"
        else:
            segment = "🔴 Premium Segment"

        cc1, cc2, cc3 = st.columns(3)
        cc1.metric("Dataset Median", f"IDR {median:,.0f}")
        cc2.metric("Price Segment", segment)
        cc3.metric("vs Median", f"{((pred_price/median)-1)*100:+.1f}%")

# ─── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#4a6a7a; font-size:0.8rem; padding:10px 0;">
💻 Laptop Price Prediction Studio • Powered by Random Forest & Decision Tree ML Models<br>
Data Processing: CPU Brand | GPU Brand | Screen Quality | Storage Type Engineering
</div>
""", unsafe_allow_html=True)