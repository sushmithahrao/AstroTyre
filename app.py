import streamlit as st
import pickle
import numpy as np

st.set_page_config(page_title="AstroTyre", layout="wide", initial_sidebar_state="collapsed")

@st.cache_resource
def load_models():
    with open("xgb_model.pkl", "rb") as f:
        xgb_model = pickle.load(f)
    with open("linear_model.pkl", "rb") as f:
        linear_model = pickle.load(f)
    return xgb_model, linear_model

xgb_model, linear_model = load_models()

st.markdown(
    """
    <style>
    /* Main page styling */
    [data-testid="stAppViewContainer"] {
      background: #f5f7fa;
    }
    
    .page-header {
      background: linear-gradient(135deg, #2f80ed 0%, #8338ec 100%);
      border-radius: 0px;
      padding: 50px 30px;
      color: white;
      text-align: center;
      box-shadow: 0 10px 30px rgba(47, 128, 237, 0.2);
      margin: -70px -70px 40px -70px;
      margin-bottom: 40px;
    }
    .page-header {
      background: linear-gradient(135deg, #2f80ed 0%, #8338ec 100%);
      border-radius: 28px;
      padding: 56px 34px 40px;
      color: white;
      text-align: center;
      box-shadow: 0 18px 42px rgba(47, 128, 237, 0.22);
      margin: -70px -70px 10px -70px;
      margin-bottom: 24px;
      position: relative;
      overflow: hidden;
    }
    .page-header::before {
      content: "";
      position: absolute;
      top: -20%;
      left: -20%;
      width: 160%;
      height: 180%;
      background: radial-gradient(circle at top left, rgba(255,255,255,0.18), transparent 35%),
                  radial-gradient(circle at bottom right, rgba(255,255,255,0.12), transparent 25%);
      pointer-events: none;
    }
    .page-header h1 {
      margin: 0 0 10px 0;
      font-size: 3.6rem;
      font-weight: 900;
      letter-spacing: -1px;
      text-shadow: 0 12px 30px rgba(0, 0, 0, 0.16);
    }
    .page-header p {
      margin: 0;
      font-size: 1.15rem;
      opacity: 0.92;
      font-weight: 500;
      color: rgba(255, 255, 255, 0.92);
    }
    .page-footer {
      padding: 10px 16px;
      margin: 32px auto 24px auto;
      text-align: center;
      color: #0f172a;
      background: rgba(255, 255, 255, 0.90);
      border-radius: 999px;
      border: 1px solid rgba(148, 163, 184, 0.25);
      backdrop-filter: blur(8px);
      max-width: 620px;
      width: fit-content;
      font-size: 0.92rem;
      font-weight: 600;
      letter-spacing: 0.02em;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 8px;
      min-height: 38px;
    }
    .page-footer span {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      color: inherit;
    }
    
    /* Section cards */
    .section-card-geo {
      background: linear-gradient(135deg, #d4e4f7 0%, #c8dff5 100%);
      border-radius: 16px;
      padding: 24px 22px;
      box-shadow: 0 4px 15px rgba(79, 139, 212, 0.1);
      margin-bottom: 20px;
      border: none;
    }
    
    .section-card-load {
      background: linear-gradient(135deg, #d4f1d4 0%, #c8e6c8 100%);
      border-radius: 16px;
      padding: 24px 22px;
      box-shadow: 0 4px 15px rgba(79, 179, 79, 0.1);
      margin-bottom: 20px;
      border: none;
    }
    
    .section-card-mat {
      background: linear-gradient(135deg, #fce4d6 0%, #fad6c8 100%);
      border-radius: 16px;
      padding: 24px 22px;
      box-shadow: 0 4px 15px rgba(242, 140, 89, 0.1);
      margin-bottom: 20px;
      border: none;
    }
    
    .section-header {
      font-size: 1.1rem;
      font-weight: 700;
      margin-bottom: 16px;
      margin-top: 0;
      color: #1e293b;
    }

    /* 1. Target the actual layout row container enclosing the button */
    .element-container:has(div[data-testid="stButton"]) {
        display: flex !important;
        justify-content: center !important;
        width: 100% !important;
    }

    /* 2. Target the button's wrapper block */
    div[data-testid="stButton"] {
        display: flex !important;
        justify-content: center !important;
        width: 100% !important;
    }
    
    /* 3. Your custom button styling */
    .stButton > button {
        background: linear-gradient(135deg, #2f80ed 0%, #8338ec 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 24px !important;
        padding: 10px 40px !important;
        font-size: 0.95rem !important;
        font-weight: 800 !important;
        box-shadow: 0 8px 20px rgba(131, 56, 236, 0.35) !important;
        transition: all 0.3s ease !important;
        letter-spacing: 0.5px !important;
        
        /* Force block display and margins to clear out Streamlit floats */
        display: block !important;
        margin: 0 auto !important;
        width: auto !important;
        max-width: 300px !important;
    }

    /* 4. Your custom hover styling */
    .stButton > button:hover {
        box-shadow: 0 12px 30px rgba(131, 56, 236, 0.45) !important;
        transform: translateY(-2px) !important;
        background: linear-gradient(135deg, #1e5cc8 0%, #7a2fd4 100%) !important;
        color: white !important;
    }
    
    /* Results table styling */
    .result-table {
      width: auto;
      margin: 24px auto;
      border-collapse: collapse;
      box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
      border-radius: 8px;
      overflow: hidden;
      animation: slideIn 0.5s ease-out;
    }
    
    .result-table td {
      padding: 12px 18px;
      border: none;
      font-weight: 700;
      font-size: 0.95rem;
    }
    
    .result-table td:first-child {
      background: linear-gradient(135deg, #4caf50 0%, #45a049 100%);
      color: white;
      width: 45%;
      text-align: left;
      letter-spacing: 0.5px;
    }
    
    .result-table td:last-child {
      background: linear-gradient(135deg, #66bb6a 0%, #5da65f 100%);
      color: white;
      text-align: left;
      font-weight: 700;
      font-size: 1.0rem;
    }
    
    /* Input styling */
    .stNumberInput > label {
      font-weight: 900 !important;
      color: #0f172a !important;
      font-size: 1.05rem !important;
      margin-top: 0 !important;
      margin-bottom: 8px !important;
      letter-spacing: 0.3px !important;
      display: block !important;
    }
    
    .stNumberInput input {
      border-radius: 10px;
      border: 2px solid #cbd5e1;
      padding: 12px 14px;
      font-size: 0.98rem;
      transition: all 0.3s ease;
    }
    
    .stNumberInput input:focus {
      border-color: #2f80ed;
      box-shadow: 0 0 0 3px rgba(47, 128, 237, 0.1);
    }
    </style>
    <div class="page-header">
      <h1>AstroTyre</h1>
      <p>Rolling Resistance Prediction System</p>
    </div>
    <div class='page-footer'>
      <span>✨ One Interface, All Critical Inputs, Instant RRC Prediction</span>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("### 📋 Enter Tyre Parameters")
col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    st.markdown('<div class="section-card-geo"><div class="section-header">📏 Geometry Parameters</div></div>', unsafe_allow_html=True)
    Tyre_width_mm = st.number_input("🛞 Tyre Width (mm)", min_value=100.0, max_value=500.0, value=205.0, step=1.0, format="%.1f")
    Aspect_ratio = st.number_input("📐 Aspect Ratio (%)", min_value=20.0, max_value=100.0, value=55.0, step=1.0, format="%.1f")
    Rim_dia_inch = st.number_input("⭕ Rim Diameter (inch)", min_value=10.0, max_value=26.0, value=16.0, step=0.5, format="%.1f")

with col2:
    st.markdown('<div class="section-card-load"><div class="section-header">⚡ Load & Operating Conditions</div></div>', unsafe_allow_html=True)
    Tyre_Mass_kg = st.number_input("⚖️ Tyre Mass (kg)", min_value=1.0, max_value=50.0, value=8.2, step=0.1, format="%.1f")
    Vertical_Load_kg = st.number_input("⬇️ Vertical Load (kg)", min_value=50.0, max_value=2000.0, value=450.0, step=5.0, format="%.1f")
    Inflation_pressure_psi = st.number_input("💨 Inflation Pressure (psi)", min_value=10.0, max_value=80.0, value=32.0, step=0.5, format="%.1f")

with col3:
    st.markdown('<div class="section-card-mat"><div class="section-header">🧪 Material Properties</div></div>', unsafe_allow_html=True)
    Hardness = st.number_input("🔨 Hardness (Shore A)", min_value=20.0, max_value=100.0, value=65.0, step=1.0, format="%.1f")
    tan_delta_70C = st.number_input("🌡️ Tan Delta at 70°C", min_value=0.01, max_value=0.50, value=0.08, step=0.01, format="%.3f")
    Silica_pct = st.number_input("🧪 Silica (%)", min_value=0.0, max_value=100.0, value=25.0, step=1.0, format="%.1f")

Vertical_Load = Vertical_Load_kg * 9.81
Inflation_pressure_kPa = Inflation_pressure_psi * 6.89476

st.markdown("<br>", unsafe_allow_html=True)
predict_button = st.button("🚀 Predict RRC", key="predict")
st.markdown("<br>", unsafe_allow_html=True)

if predict_button:
    load_pressure_ratio = Vertical_Load / Inflation_pressure_kPa if Inflation_pressure_kPa != 0 else 0
    tan_delta_load = tan_delta_70C * Vertical_Load
    rubber_volume_proxy = Tyre_width_mm * Aspect_ratio
    contact_stress_proxy = Vertical_Load / Tyre_width_mm if Tyre_width_mm != 0 else 0
    pressure_load_interaction = Inflation_pressure_kPa * Vertical_Load
    width_aspect = Tyre_width_mm * Aspect_ratio
    rim_aspect_ratio = Rim_dia_inch / Aspect_ratio if Aspect_ratio != 0 else 0

    if Tyre_width_mm <= 279.4:
        input_features = np.array([[
            tan_delta_70C,
            Tyre_width_mm,
            Aspect_ratio,
            Rim_dia_inch,
            Vertical_Load,
            Inflation_pressure_kPa,
            contact_stress_proxy,
            load_pressure_ratio,
            Tyre_Mass_kg,
            Silica_pct,
            Hardness,
            tan_delta_load,
            width_aspect,
            rim_aspect_ratio,
            rubber_volume_proxy,
            pressure_load_interaction
        ]])
        
        prediction = xgb_model.predict(input_features)
        model_used = "XGBoost (Narrow Tyre Model)"
    else:
        input_features = np.array([[
            Vertical_Load,
            Inflation_pressure_kPa,
            Aspect_ratio,
            Rim_dia_inch,
            Tyre_Mass_kg,
            Hardness,
            tan_delta_70C,
            Silica_pct,
        ]])
        prediction = linear_model.predict(input_features)
        model_used = "Linear Regression (Wide Tyre Model)"

    st.markdown(
        f"""
        <div style='text-align: center;'>
        <table class="result-table">
            <tr>
                <td>📊 Predicted RRC</td>
                <td>{prediction[0]:.4f}</td>
            </tr>
            <tr>
                <td>🤖 Model Used</td>
                <td>{model_used}</td>
            </tr>
        </table>
        </div>
        """,
        unsafe_allow_html=True,
    )


