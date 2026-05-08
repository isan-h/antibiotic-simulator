import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.optimize import linprog
import time
import os
import tempfile

# Try importing vedo
VEDO_AVAILABLE = False
try:
    from vedo import Plotter, Points, Volume, Mesh
    VEDO_AVAILABLE = True
except ImportError:
    pass

st.set_page_config(page_title="Digital Antibiotic Simulator", page_icon="⚕️", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&display=swap');
    .main, .stApp { background: linear-gradient(135deg, #050505 0%, #0a0a1a 50%, #0d1117 100%); color: #e0e0e0; }
    h1, h2, h3 { font-family: 'Orbitron', sans-serif !important; color: #00d4ff !important; text-shadow: 0 0 20px rgba(0, 212, 255, 0.5); }
    .stSlider > div > div > div { background: linear-gradient(90deg, #00d4ff, #7b2cbf) !important; }
    .metric-card { background: rgba(255,255,255,0.03); border: 1px solid rgba(0, 212, 255, 0.2); border-radius: 12px; padding: 15px; backdrop-filter: blur(10px); }
    .warning-pill { display: inline-block; padding: 6px 14px; border-radius: 20px; font-weight: bold; font-size: 0.85em; }
    .safe-pill { background: rgba(0, 255, 136, 0.15); color: #00ff88; border: 1px solid #00ff88; }
    .danger-pill { background: rgba(255, 51, 102, 0.15); color: #ff3366; border: 1px solid #ff3366; animation: blink 1s infinite; }
    @keyframes blink { 0%, 100% { opacity: 1; } 50% { opacity: 0.4; } }
    .stButton>button { background: linear-gradient(135deg, #00d4ff, #7b2cbf); color: white; border: none; border-radius: 25px; padding: 12px 30px; font-family: 'Orbitron', sans-serif; font-weight: 700; text-transform: uppercase; letter-spacing: 2px; box-shadow: 0 4px 15px rgba(0, 212, 255, 0.4); }
    .sidebar .sidebar-content { background: rgba(5, 5, 5, 0.95); border-right: 1px solid rgba(0, 212, 255, 0.15); }
</style>
""", unsafe_allow_html=True)

def solve_optimization(x_user=None, y_user=None, z_user=None, toxicity_threshold=120, efficacy_req=70):
    c = [4, 6, 3]
    A_ub = [[-2, -3, -1], [3, 2, 0], [0, -1, -1]]
    b_ub = [-efficacy_req, toxicity_threshold, -10]
    bounds = [(15, 40), (2, 6), (5, 14)]
    if x_user: bounds[0] = (x_user, x_user)
    if y_user: bounds[1] = (y_user, y_user)
    if z_user: bounds[2] = (z_user, z_user)
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    if result.success:
        return {'x': result.x[0], 'y': result.x[1], 'z': result.x[2], 'Z': result.fun,
                'efficacy': 2*result.x[0] + 3*result.x[1] + result.x[2],
                'toxicity': 3*result.x[0] + 2*result.x[1], 'status': 'Optimal'}
    return None

# ==================== REAL MESH LOADING (FIXED) ====================

def load_real_anatomy(dose, freq, duration, toxicity, efficacy):
    """Load REAL .stl anatomical meshes — FINAL FIXED VERSION"""
    if not VEDO_AVAILABLE:
        return None
    
    try:
        concentration = dose / 40
        models_dir = "models"
        
        required_files = ['torso.stl', 'heart.stl', 'lungs.stl', 'liver.stl', 'kidneys.stl']
        missing = [f for f in required_files if not os.path.exists(os.path.join(models_dir, f))]
        
        if missing:
            st.error(f" Missing model files: {missing}")
            return None
        
        # ==================== LOAD & ANALYZE ====================
        
        def load_and_analyze(filename):
            m = Mesh(os.path.join(models_dir, filename))
            b = m.bounds()
            return {
                'mesh': m,
                'cx': (b[0]+b[1])/2,
                'cy': (b[2]+b[3])/2,
                'cz': (b[4]+b[5])/2,
                'height': b[3]-b[2],
                'width': b[1]-b[0],
                'depth': b[5]-b[4]
            }
        
        torso_data = load_and_analyze("torso.stl")
        heart_data = load_and_analyze("heart.stl")
        lungs_data = load_and_analyze("lungs.stl")
        liver_data = load_and_analyze("liver.stl")
        kidneys_data = load_and_analyze("kidneys.stl")
        
        # ==================== SCALE TORSO ====================
        
        target_height = 10.0
        torso_scale = target_height / torso_data['height']
        
        torso = torso_data['mesh']
        torso.scale(torso_scale)
        torso.shift(-torso_data['cx']*torso_scale, -torso_data['cy']*torso_scale, -torso_data['cz']*torso_scale)
        
        tb = torso.bounds()
        torso_ymin, torso_ymax = tb[2], tb[3]
        torso_height = torso_ymax - torso_ymin
        torso_cx = (tb[0] + tb[1]) / 2
        torso_cz = (tb[4] + tb[5]) / 2
        
        # ==================== SCALE & POSITION ORGANS ====================
        
        def prepare_organ(data, color, opacity_val, target_height_ratio, y_offset_ratio, x_offset=0, z_offset=0, extra_scale=1.0):
            m = data['mesh']
            m.color(color)
            m.opacity(opacity_val)
            m.lighting("glossy")
            
            target_organ_height = torso_height * target_height_ratio
            organ_scale = (target_organ_height / data['height']) * extra_scale
            
            m.scale(organ_scale)
            
            target_y = torso_ymin + (torso_height * y_offset_ratio)
            
            m.shift(
                (torso_cx + x_offset) - (data['cx'] * organ_scale),
                target_y - (data['cy'] * organ_scale),
                (torso_cz + z_offset) - (data['cz'] * organ_scale)
            )
            return m
        
        # Position each organ
        lungs = prepare_organ(lungs_data, "steelblue", 0.75, 0.45, 0.70, extra_scale=1.0)
        heart = prepare_organ(heart_data, "darkred", 0.9, 0.30, 0.58, x_offset=-0.2, extra_scale=0.85)
        liver = prepare_organ(liver_data, "orangered", 0.8, 0.35, 0.42, x_offset=0.5, extra_scale=1.1)
        kidneys = prepare_organ(kidneys_data, "purple", 0.75, 0.25, 0.28, z_offset=-0.3, extra_scale=0.9)
        
        # Style torso
        torso.color("lightblue")
        torso.opacity(0.2)
        torso.lighting("plastic")
        
        # ==================== PARTICLES (NO VOLUME FOG) ====================
        
        np.random.seed(42)
        n = 80
        px = np.random.uniform(tb[0]*0.6, tb[1]*0.6, n)
        py = np.random.uniform(tb[2]*0.5, tb[3]*0.7, n)
        pz = np.random.uniform(tb[4]*0.6, tb[5]*0.6, n)
        
        if toxicity > 100:
            particles = Points(np.column_stack([px, py, pz]), r=8, c="red", alpha=0.9)
        else:
            if concentration < 0.33:
                particles = Points(np.column_stack([px, py, pz]), r=5, c="cyan", alpha=0.85)
            elif concentration < 0.66:
                particles = Points(np.column_stack([px, py, pz]), r=5, c="yellow", alpha=0.85)
            else:
                particles = Points(np.column_stack([px, py, pz]), r=7, c="orange", alpha=0.9)
        
        particles.render_points_as_spheres(True)
        
        # ==================== RENDER (NO VOLUME, NO LIGHT CLASS) ====================
        
        plt = Plotter(bg="black", bg2="navy", size=(1000, 700), offscreen=True)
        plt += torso
        plt += lungs
        plt += heart
        plt += liver
        plt += kidneys
        plt += particles
        
        # Simple camera
        plt.camera.SetPosition(0, (torso_ymin + torso_ymax)/2, 15)
        plt.camera.SetFocalPoint(0, (torso_ymin + torso_ymax)/2, 0)
        plt.camera.SetViewUp(0, 1, 0)
        
        # Screenshot
        plt.show(interactive=False)
        
        tmp_img = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        plt.screenshot(tmp_img.name)
        plt.close()
        return tmp_img.name
        
    except Exception as e:
        st.error(f"3D rendering error: {str(e)}")
        st.info("Falling back to Plotly visualization...")
        return None

# ==================== PLOTLY FALLBACK ====================

def create_plotly_fallback(dose, freq, duration, toxicity, efficacy):
    fig = go.Figure()
    concentration = dose / 40
    
    if toxicity > 100:
        body_color = 'rgba(255, 50, 50, 0.45)'
    elif concentration < 0.33:
        body_color = 'rgba(0, 150, 255, 0.3)'
    elif concentration < 0.66:
        body_color = 'rgba(255, 200, 50, 0.4)'
    else:
        body_color = 'rgba(255, 100, 50, 0.5)'
    
    u = np.linspace(0, 2*np.pi, 60)
    v = np.linspace(0, np.pi, 60)
    U, V = np.meshgrid(u, v)
    a = 3.2 + 0.8 * np.sin(V) ** 2
    b = 5.0
    c = 2.0 + 0.3 * np.cos(2 * V)
    tx = a * np.cos(U) * np.sin(V)
    ty = b * np.cos(V)
    tz = c * np.sin(U) * np.sin(V)
    
    fig.add_trace(go.Surface(x=tx, y=ty, z=tz, colorscale=[[0, body_color], [1, body_color]], showscale=False, opacity=0.2))
    
    hu = np.linspace(0, 2*np.pi, 40)
    hv = np.linspace(0, np.pi, 40)
    HU, HV = np.meshgrid(hu, hv)
    hx = 0.8 * (16 * np.sin(HU)**3) / 16
    hy = 0.8 * (13 * np.cos(HU) - 5 * np.cos(2*HU) - 2 * np.cos(3*HU) - np.cos(4*HU)) / 16 + 1.5
    hz = 0.6 * np.sin(HV) * np.cos(HU)
    fig.add_trace(go.Surface(x=hx, y=hy, z=hz, colorscale=[[0, 'rgba(220,20,60,0.7)'], [1, 'rgba(220,20,60,0.7)']], showscale=False, opacity=0.7))
    
    for side in [1, -1]:
        lu = np.linspace(0, 2*np.pi, 30)
        lv = np.linspace(0, np.pi, 30)
        LU, LV = np.meshgrid(lu, lv)
        la = 1.0 + 0.3 * np.sin(3*LV)
        lx = side * (1.8 + la * np.cos(LU) * np.sin(LV))
        ly = 2.5 + 1.4 * np.cos(LV)
        lz = 0.8 * np.sin(LU) * np.sin(LV)
        fig.add_trace(go.Surface(x=lx, y=ly, z=lz, colorscale=[[0, 'rgba(100,149,237,0.5)'], [1, 'rgba(100,149,237,0.5)']], showscale=False, opacity=0.5))
    
    lvu = np.linspace(0, 2*np.pi, 30)
    lvv = np.linspace(0, np.pi, 30)
    LVU, LVV = np.meshgrid(lvu, lvv)
    lva = 1.6 + 0.4 * np.cos(LVV)
    lvx = 1.0 + lva * np.cos(LVU) * np.sin(LVV)
    lvy = -0.5 + 1.0 * np.cos(LVV)
    lvz = 1.2 * np.sin(LVU) * np.sin(LVV)
    fig.add_trace(go.Surface(x=lvx, y=lvy, z=lvz, colorscale=[[0, 'rgba(255,140,0,0.5)'], [1, 'rgba(255,140,0,0.5)']], showscale=False, opacity=0.5))
    
    for side in [1, -1]:
        ku = np.linspace(0, 2*np.pi, 25)
        kv = np.linspace(0, np.pi, 25)
        KU, KV = np.meshgrid(ku, kv)
        kr = 0.5 + 0.15 * np.cos(2*KU)
        kx = side * (2.0 + kr * np.cos(KU) * np.sin(KV))
        ky = -2.5 + 0.6 * np.cos(KV)
        kz = 0.4 * np.sin(KU) * np.sin(KV)
        fig.add_trace(go.Surface(x=kx, y=ky, z=kz, colorscale=[[0, 'rgba(138,43,226,0.5)'], [1, 'rgba(138,43,226,0.5)']], showscale=False, opacity=0.5))
    
    np.random.seed(42)
    t = np.linspace(0, 4*np.pi, 50)
    px = 2.5 * np.cos(t) + np.random.normal(0, 0.3, 50)
    py = 3 * np.sin(t * 0.7) + np.random.normal(0, 0.5, 50)
    pz = 1.5 * np.sin(t * 1.3) + np.random.normal(0, 0.2, 50)
    
    if toxicity > 100:
        pc = [f'rgba(255,50,50,{0.5 + 0.4*np.sin(i)})' for i in range(50)]
        ps = 8
    else:
        alpha = 0.5
        if concentration < 0.33:
            pc = [f'rgba(0,212,255,{alpha})'] * 50
        elif concentration < 0.66:
            pc = [f'rgba(255,215,0,{alpha})'] * 50
        else:
            pc = [f'rgba(255,100,50,{alpha})'] * 50
        ps = 6
    
    fig.add_trace(go.Scatter3d(x=px, y=py, z=pz, mode='markers', marker=dict(size=ps, color=pc, line=dict(width=0)), showlegend=False))
    
    if toxicity > 100:
        for pos in [[-0.5, 1.5, 0], [1.8, 2.5, 0], [-1.8, 2.5, 0]]:
            fig.add_trace(go.Scatter3d(x=[pos[0]], y=[pos[1]], z=[pos[2]], mode='markers', marker=dict(size=25, color='rgba(255,0,0,0.5)', symbol='diamond'), showlegend=False))
    
    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False, range=[-5, 5]),
            yaxis=dict(visible=False, range=[-6, 6]),
            zaxis=dict(visible=False, range=[-3, 3]),
            bgcolor='rgba(0,0,0,0)',
            camera=dict(eye=dict(x=0, y=0, z=2.5))
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend=False,
        margin=dict(l=0, r=0, t=30, b=0),
        height=550
    )
    return fig

# ==================== FEASIBLE REGION ====================

def create_nebula_feasible_region():
    x_range = np.linspace(15, 40, 25)
    y_range = np.linspace(2, 6, 25)
    z_range = np.linspace(5, 14, 25)
    feasible, objectives = [], []
    
    for x in x_range:
        for y in y_range:
            for z in z_range:
                if (2*x + 3*y + z >= 70 and 3*x + 2*y <= 120 and y + z >= 10):
                    feasible.append([x, y, z])
                    objectives.append(4*x + 6*y + 3*z)
    
    feasible = np.array(feasible)
    objectives = np.array(objectives)
    min_idx = np.argmin(objectives)
    opt_x, opt_y, opt_z = feasible[min_idx]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=feasible[:, 0], y=feasible[:, 1], z=feasible[:, 2], mode='markers',
        marker=dict(size=4, color=objectives, colorscale='Viridis', opacity=0.6, line=dict(width=0),
                   colorbar=dict(title='Z Value', tickfont=dict(color='white'), thickness=15, len=0.5)), name='Nebula Cloud'))
    
    fig.add_trace(go.Scatter3d(x=[opt_x], y=[opt_y], z=[opt_z], mode='markers',
        marker=dict(size=20, color='#ffd700', symbol='diamond', line=dict(color='#ff8c00', width=3), opacity=0.95),
        name=f' OPTIMAL ({opt_x:.1f}, {opt_y:.1f}, {opt_z:.1f})'))
    
    xx, yy = np.meshgrid(np.linspace(15, 40, 15), np.linspace(2, 6, 15))
    zz = np.clip(70 - 2*xx - 3*yy, 5, 14)
    fig.add_trace(go.Surface(x=xx, y=yy, z=zz, colorscale=[[0, 'rgba(0,255,136,0.12)'], [1, 'rgba(0,255,136,0.12)']], showscale=False, name='🟢 Efficacy'))
    
    zz_tox = np.linspace(5, 14, 15)
    XX, ZZ = np.meshgrid(np.linspace(15, 40, 15), zz_tox)
    YY = np.clip((120 - 3*XX) / 2, 2, 6)
    fig.add_trace(go.Surface(x=XX, y=YY, z=ZZ, colorscale=[[0, 'rgba(255,51,102,0.12)'], [1, 'rgba(255,51,102,0.12)']], showscale=False, name='Toxicity'))
    
    xx_fd = np.linspace(15, 40, 15)
    zz_fd = np.linspace(5, 14, 15)
    XX_fd, ZZ_fd = np.meshgrid(xx_fd, zz_fd)
    YY_fd = np.clip(10 - ZZ_fd, 2, 6)
    fig.add_trace(go.Surface(x=XX_fd, y=YY_fd, z=ZZ_fd, colorscale=[[0, 'rgba(0,150,255,0.12)'], [1, 'rgba(0,150,255,0.12)']], showscale=False, name='Continuity'))
    
    fig.update_layout(
        scene=dict(xaxis_title=dict(text='Dose (x)', font=dict(color='white')), yaxis_title=dict(text='Frequency (y)', font=dict(color='white')), zaxis_title=dict(text='Duration (z)', font=dict(color='white')),
            bgcolor='rgba(5,5,15,0.9)', xaxis=dict(gridcolor='rgba(0,212,255,0.15)', tickfont=dict(color='white'), showbackground=False),
            yaxis=dict(gridcolor='rgba(0,212,255,0.15)', tickfont=dict(color='white'), showbackground=False),
            zaxis=dict(gridcolor='rgba(0,212,255,0.15)', tickfont=dict(color='white'), showbackground=False), camera=dict(eye=dict(x=2, y=2, z=1.5))),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=600, showlegend=True,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, font=dict(color='white', size=12), bgcolor='rgba(0,0,0,0.5)'),
        margin=dict(l=0, r=0, t=30, b=0)
    )
    return fig

# ==================== GAUGES & CURVES ====================

def create_gauge(value, title, max_val, color):
    fig = go.Figure(go.Indicator(mode="gauge+number", value=value, domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 14, 'color': 'white', 'family': 'Orbitron'}},
        gauge={'axis': {'range': [0, max_val], 'tickcolor': color, 'tickfont': {'color': 'white'}}, 'bar': {'color': color, 'thickness': 0.7},
            'bgcolor': 'rgba(0,0,0,0.3)', 'borderwidth': 2, 'bordercolor': color,
            'steps': [{'range': [0, max_val*0.33], 'color': 'rgba(0,255,136,0.15)'}, {'range': [max_val*0.33, max_val*0.66], 'color': 'rgba(255,215,0,0.15)'}, {'range': [max_val*0.66, max_val], 'color': 'rgba(255,51,102,0.15)'}],
            'threshold': {'line': {'color': 'red', 'width': 3}, 'thickness': 0.8, 'value': max_val*0.8}}))
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font={'color': "white", 'family': "Orbitron"}, height=220, margin=dict(l=20, r=20, t=50, b=20))
    return fig

def create_concentration_curve(dose, freq, duration):
    time_points = np.linspace(0, duration, 100)
    concentration = [max(0, (dose * freq / 10) * np.exp(-0.15 * t) * (1 + 0.3 * np.sin(freq * t))) for t in time_points]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time_points, y=concentration, fill='tozeroy', fillcolor='rgba(0, 212, 255, 0.25)', line=dict(color='#00d4ff', width=3), name='Concentration'))
    fig.add_hline(y=25, line_dash="dash", line_color="#ff3366", annotation_text="Toxic")
    fig.add_hrect(y0=10, y1=25, fillcolor="rgba(0, 255, 136, 0.08)", line_width=0)
    
    fig.update_layout(title=dict(text='Drug Concentration Over Time', font=dict(color='white', family='Orbitron', size=16)), xaxis_title='Time (days)', yaxis_title='Concentration (mg/L)',
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0.2)', font=dict(color='white'),
        xaxis=dict(gridcolor='rgba(255,255,255,0.1)'), yaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
        height=280, margin=dict(l=50, r=50, t=60, b=50))
    return fig

# ==================== MAIN ====================

def main():
    st.markdown("""
    <div style="text-align: center; padding: 10px 0 20px 0;">
        <h1 style="font-size: 2.5em; margin-bottom: 5px;">Digital Antibiotic Simulator</h1>
        <p style="font-family: 'Orbitron', sans-serif; font-size: 1.1em; color: #7b2cbf; letter-spacing: 4px;">ANTIBIOTIC OPTIMIZATION SYSTEM </p>
        <div style="height: 2px; background: linear-gradient(90deg, transparent, #00d4ff, #7b2cbf, transparent); margin: 15px 0;"></div>
    </div>
    """, unsafe_allow_html=True)
    
    with st.sidebar:
        st.markdown("<h2 style='color: #00d4ff; text-align: center;'> CONTROL PANEL</h2>", unsafe_allow_html=True)
        st.markdown("---")
        
        dose = st.slider(" Dose (x)", 15.0, 40.0, 23.5, 0.5)
        frequency = st.slider("Frequency (y)", 2.0, 6.0, 6.0, 0.5)
        duration = st.slider(" Duration (z)", 5.0, 14.0, 5.0, 0.5)
        
        st.markdown("---")
        st.markdown("###  Constraints")
        toxicity_threshold = st.slider(" Toxicity Threshold", 80.0, 150.0, 120.0, 5.0)
        efficacy_requirement = st.slider(" Efficacy Requirement", 50.0, 100.0, 70.0, 5.0)
        
        st.markdown("---")
        if st.button("AUTO-OPTIMIZE"):
            with st.spinner("Computing..."):
                time.sleep(1)
                opt = solve_optimization(toxicity_threshold=toxicity_threshold, efficacy_req=efficacy_requirement)
                if opt:
                    st.session_state['optimal'] = opt
                    st.success(f"Z* = {opt['Z']:.1f}")
                    st.info(f"x={opt['x']:.1f}, y={opt['y']:.1f}, z={opt['z']:.1f}")
        
        current_efficacy = 2*dose + 3*frequency + duration
        current_toxicity = 3*dose + 2*frequency
        
        st.markdown("---")
        st.markdown("### Status")
        if current_toxicity <= toxicity_threshold and current_efficacy >= efficacy_requirement:
            st.markdown('<div class="warning-pill safe-pill">SAFE ZONE</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="warning-pill danger-pill">TOXIC WARNING</div>', unsafe_allow_html=True)
    
    current_efficacy = 2*dose + 3*frequency + duration
    current_toxicity = 3*dose + 2*frequency
    opt_score = 4*dose + 6*frequency + 3*duration
    
    st.markdown("<h3 style='text-align: center; color: #00d4ff; margin-bottom: 10px;'>LIVE OPTIMIZATION METRICS</h3>", unsafe_allow_html=True)
    
    g1, g2, g3 = st.columns(3)
    with g1: st.plotly_chart(create_gauge(current_efficacy, "EFFICACY", 150, '#00ff88'), use_container_width=True)
    with g2: st.plotly_chart(create_gauge(current_toxicity, "TOXICITY", 150, '#ff3366'), use_container_width=True)
    with g3: st.plotly_chart(create_gauge(opt_score, "OPT. SCORE", 300, '#ffd700'), use_container_width=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <h2 style='text-align: center; color: #00d4ff; text-shadow: 0 0 30px rgba(0,212,255,0.5);'> 3D PATIENT SIMULATION</h2>
    <p style='text-align: center; color: #7b2cbf; font-family: Orbitron; letter-spacing: 3px;'>REAL ANATOMICAL MESH VISUALIZATION</p>
    """, unsafe_allow_html=True)
    
    body_col, conc_col = st.columns([2, 1])
    
    with body_col:
        if VEDO_AVAILABLE:
            result = load_real_anatomy(dose, frequency, duration, current_toxicity, current_efficacy)
            if result and isinstance(result, str):
                if result.endswith('.html'):
                    st.components.v1.html(result, height=600, scrolling=False)
                elif result.endswith('.png'):
                    st.image(result, use_container_width=True)
            else:
                st.plotly_chart(create_plotly_fallback(dose, frequency, duration, current_toxicity, current_efficacy), use_container_width=True)
        else:
            st.info(" Install `vedo` for real 3D: `pip install vedo vtk`")
            st.plotly_chart(create_plotly_fallback(dose, frequency, duration, current_toxicity, current_efficacy), use_container_width=True)
    
    with conc_col:
        st.markdown(f"""
        <div class="metric-card" style="margin-bottom: 15px;">
            <h4 style="color: #00d4ff; margin: 0;"> Constraint Values</h4>
            <p style="margin: 5px 0;"><strong style="color: #00ff88;">Efficacy:</strong> {current_efficacy:.1f} / {efficacy_requirement:.1f}</p>
            <p style="margin: 5px 0;"><strong style="color: #ff3366;">Toxicity:</strong> {current_toxicity:.1f} / {toxicity_threshold:.1f}</p>
            <p style="margin: 5px 0;"><strong style="color: #ffd700;">Freq+Dur:</strong> {frequency + duration:.1f} / 10.0</p>
        </div>
        """, unsafe_allow_html=True)
        
        if 'optimal' in st.session_state:
            opt = st.session_state['optimal']
            st.markdown(f"""
            <div class="metric-card" style="border: 1px solid #ffd700;">
                <h4 style="color: #ffd700; margin: 0;">OPTIMAL SOLUTION</h4>
                <p style="margin: 5px 0; font-family: monospace;">Z* = {opt['Z']:.2f}</p>
                <p style="margin: 5px 0; font-family: monospace;">x* = {opt['x']:.2f}</p>
                <p style="margin: 5px 0; font-family: monospace;">y* = {opt['y']:.2f}</p>
                <p style="margin: 5px 0; font-family: monospace;">z* = {opt['z']:.2f}</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.plotly_chart(create_concentration_curve(dose, frequency, duration), use_container_width=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <h2 style='text-align: center; color: #7b2cbf; text-shadow: 0 0 30px rgba(123,44,191,0.5);'>FEASIBLE REGION — 3D NEBULA VISUALIZATION</h2>
    <p style='text-align: center; color: #888;'>Glowing nebula cloud | Pulsing optimal star | Laser constraint planes</p>
    """, unsafe_allow_html=True)
    
    st.plotly_chart(create_nebula_feasible_region(), use_container_width=True)
    
    st.markdown("""
    <div style="text-align: center; margin-top: 40px; padding: 20px; border-top: 1px solid rgba(0,212,255,0.15);">
        <p style="color: #444; font-family: 'Orbitron'; letter-spacing: 3px; font-size: 0.9em;">DIGITAL ANTIBIOTIC SIMULATOR </p>
        <p style="color: #333; font-size: 0.8em;">Minimizing bacterial resistance through linear programming optimization</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()