import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma
from scipy.optimize import minimize



### REGGE MODEL ###

# Parámetros del modelo de Regge (Mathieu et al. 2015)
# Trayectorias (Ecuación 23)
alpha_V0 = 0.44      # Intercepto vectorial
alpha_V_prime = 0.93 # Pendiente vectorial (GeV^-2)
alpha_A0 = -0.22     # Intercepto axial
alpha_A_prime = 1.08 # Pendiente axial (GeV^-2)
alpha_c0 = 0.9       # Intercepto del corte
alpha_c_prime = 0.2  # Pendiente del corte (GeV^-2)

# Acoplamientos (Ecuación 24)
g1 = -0.14    # GeV^-2 (acoplamiento vectorial helicidad-flip)
g2 = -9.74    # GeV^-4 (acoplamiento axial)
g4 = 3.98     # GeV^-2 (acoplamiento vectorial helicidad-nonflip)
g1c = 0.24    # GeV^-2 (corte vectorial helicidad-flip)
g4c = -0.13   # GeV^-2 (corte vectorial helicidad-nonflip)

# Escala de energía
s0 = 1.0    # GeV^2 (parámetro de escala)

def alpha_V(t):
    """Trayectoria para intercambios vectoriales (ω y ρ)"""
    return alpha_V0 + alpha_V_prime * t

def alpha_A(t):
    """Trayectoria para intercambios axiales (b y h)"""
    return alpha_A0 + alpha_A_prime * t

def alpha_c(t):
    """Trayectoria para el corte Regge-Pomerón"""
    return alpha_c0 + alpha_c_prime * t

def regge_amplitude(s, t, alpha_traj):
    """
    Amplitud de Regge completa (Ecuación 11 del artículo)
    
    Args:
        s: Mandelstam s (GeV^2)
        t: Mandelstam t (GeV^2)
        alpha_traj: Función de trayectoria (alpha_V, alpha_A o alpha_c)
    
    Returns:
        Amplitud compleja de Regge
    """
    # Término gamma para eliminar polos no físicos
    
    gamma_term = np.pi / gamma(alpha_traj(t))
    # Factor de firma (1 - e^{-iπα})/(2 sin πα)  
    alpha = alpha_traj(t) + 1e-6  
    signature = (1 - np.exp(-1j * np.pi * alpha)) / (2 * np.sin(np.pi * alpha))
    
    # Dependencia energética (s/s0)^{α(t)-1}
    energy_dep = (s / s0)**(alpha - 1)
    
    return gamma_term * signature * energy_dep

def regge_cut(s, t, alpha_traj):
    """
    Corte Regge-Pomerón (Ecuación 12 del artículo)
    Similar a la amplitud de Regge pero con factor logarítmico adicional
    """
    log_term = 1.0 / np.log(s / s0)
    return log_term * regge_amplitude(s, t, alpha_traj)
    
def F1(s, t):
    """Amplitud invariante F1 (Ecuación 15, contribución vectorial + corte)"""
    RV = regge_amplitude(s, t, alpha_V)
    Rc = regge_cut(s, t, alpha_c)
    return (-g1*t + 2*mb*g4)*RV + (-g1c*t + 2*mb*g4c)*Rc

def F2(s, t):
    """Amplitud invariante F2 (Ecuación 15, contribución axial)"""
    RA = regge_amplitude(s, t, alpha_A)
    return g2 * t * RA

def F3(s, t):
    """Amplitud invariante F3 (Ecuación 15, contribución vectorial + corte)"""
    RV = regge_amplitude(s, t, alpha_V)
    Rc = regge_cut(s, t, alpha_c)
    return (2*mb*g1 - g4)*t*RV + (2*mb*g1c - g4c)*t*Rc

def F4(s, t):
    """Amplitud invariante F4 (despreciada en el modelo)"""
    return 0.0 + 0.0j


################################################################

# --- PARÁMETROS FÍSICOS ---
ma = 0.0       # masa del fotón (GeV)
mb = 0.938     # masa del protón (GeV)
m1 = 0.135     # masa del pi0 (GeV)
m2 = 0.938     # masa del protón (GeV)
sqrt_s = 2.5   # energía total en el CM (GeV)
s = sqrt_s ** 2
P_gamma = 1.0  # polarización del haz (puede variar entre 0 y 1)

# Modelo simplificado para Σ(s, t)
def sigma_beam_asymmetry(s, t):
    return 0.5 * np.exp(-0.5 * (t + 0.5)**2)

# Modelo simplificado para dσ/dt
#def dsigma_dt(s, t):
    #return np.exp(-5 * (t + 0.5)**2) #ejemplo de prueba
    
def dsigma_dt(s, t, M = mb):
    return 1 / 32*np.pi * ( ( abs(F3(s,t))**2 - t*abs(F1(s,t))**2) / ( 4 * M**2 - t ) + abs(F2(s,t))**2 - t*abs(F4(s,t))**2 )

# Intensidad total del evento
def intensity(s, t, phi):
    return dsigma_dt(s, t) * (1 - P_gamma * sigma_beam_asymmetry(s, t) * np.cos(2 * phi))

# Momento en el CM
def momentum_mag(m1, m2, s):
    E1 = (s + m1**2 - m2**2) / (2 * np.sqrt(s))
    return np.sqrt(E1**2 - m1**2)

# Cálculo de Mandelstam t
def mandelstam_t(pa, p1):
    diff = p1 - pa
    return diff[0]**2 - np.sum(diff[1:]**2)

# Cálculo de beta del boost del CM respecto al laboratorio
def compute_beta_cm(ma, mb, s):
    Ea = (s + ma**2 - mb**2) / (2 * np.sqrt(s))
    pa = np.sqrt(Ea**2 - ma**2)
    beta = pa / (Ea + mb)
    return beta, Ea, pa

# Generador de un evento candidato en el CM
def generate_candidate_event():
    E1 = (s + m1**2 - m2**2) / (2 * np.sqrt(s))
    p = np.sqrt(E1**2 - m1**2)
    
    cos_theta = 2 * np.random.rand() - 1
    phi = 2 * np.pi * np.random.rand()
    theta = np.arccos(cos_theta)
    
    p_vec = p * np.array([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta)
    ])
    
    p1 = np.array([E1, *p_vec])
    return p1, phi

# --- BUCLE DE ACEPTACIÓN-RECHAZO ---
beta, Ea, pa_mag = compute_beta_cm(ma, mb, s)
pa_4vec = np.array([Ea, 0, 0, pa_mag])  # 4-vector del fotón incidente en CM
n_events = 10000                       # Número deseado de eventos aceptados
accepted_phis = []
accepted_costhetas = []

# Estimación del máximo de la intensidad para normalizar el criterio de aceptación
phi_test = np.linspace(0, 2*np.pi, 100)
t_test = np.linspace(-2, 0, 100)
phi_grid, t_grid = np.meshgrid(phi_test, t_test)
intensity_vals = intensity(s, t_grid, phi_grid)
I_max = np.max(intensity_vals)

while len(accepted_phis) < n_events:
    p1, phi = generate_candidate_event()
    t = mandelstam_t(pa_4vec, p1)
    I_evt = intensity(s, t, phi)
    r = np.random.rand() * I_max
    if r < I_evt:
        cos_theta = p1[3] / np.linalg.norm(p1[1:])
        accepted_phis.append(phi)
        accepted_costhetas.append(cos_theta)

# --- HISTOGRAMAS ---
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

axs[0].hist(accepted_costhetas, bins=50, color='skyblue', edgecolor='black', density=True)
axs[0].set_xlabel(r'$\cos\theta$')
axs[0].set_ylabel('Distribución normalizada')
axs[0].set_title(r'Distribución aceptada en $\cos\theta$ (CM)')

axs[1].hist(accepted_phis, bins=50, color='salmon', edgecolor='black', density=True)
axs[1].set_xlabel(r'$\phi$')
axs[1].set_ylabel('Distribución normalizada')
axs[1].set_title(r'Distribución aceptada en $\phi$ (CM)')

plt.tight_layout()
plt.savefig('Angular Distributions in CM frame.png')

### Lab Frame ###

# --- BOOST DEL CM AL LABORATORIO ---
def compute_beta_gamma(ma, mb, s):
    Ea = (s + ma**2 - mb**2) / (2 * np.sqrt(s))
    pa = np.sqrt(Ea**2 - ma**2)
    beta = pa / (Ea + mb)
    gamma = 1 / np.sqrt(1 - beta**2)
    return beta, gamma

def boost_z(p, beta, gamma):
    """Aplica un boost en z a un 4-vector"""
    p0, px, py, pz = p
    pz_prime = gamma * (pz + beta * p0)
    E_prime = gamma * (p0 + beta * pz)
    return np.array([E_prime, px, py, pz_prime])

# --- GENERADOR DE EVENTOS ISOTRÓPICOS EN EL CM ---
def generate_candidate_event():
    E1 = (s + m1**2 - m2**2) / (2 * np.sqrt(s))  # energía de la partícula 1
    p = np.sqrt(E1**2 - m1**2)                   # módulo del momento
    
    cos_theta = 2 * np.random.rand() - 1
    phi = 2 * np.pi * np.random.rand()
    theta = np.arccos(cos_theta)
    
    # Momento en coordenadas cartesianas
    p_vec = p * np.array([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta)
    ])
    
    # 4-vector (E, px, py, pz)
    p1 = np.array([E1, *p_vec])
    return p1

# --- BOOST DE EVENTOS Y CÁLCULO DEL ÁNGULO EN EL LABORATORIO ---
beta, gamma = compute_beta_gamma(ma, mb, s)
n_events = 10000
theta_lab = []

for _ in range(n_events):
    p1_cm = generate_candidate_event()
    p1_lab = boost_z(p1_cm, beta, gamma)
    px, py, pz = p1_lab[1:]
    theta = np.arccos(pz / np.linalg.norm([px, py, pz]))
    theta_lab.append(np.cos(theta))

# --- HISTOGRAMA DE DISTRIBUCIÓN EN EL LABORATORIO ---
plt.figure(figsize=(7,5))
plt.hist(theta_lab, bins=50, color='green', edgecolor='black', density=True)
plt.xlabel(r'$\cos\theta_{\mathrm{lab}}$')
plt.ylabel('Distribución normalizada')
plt.title('Distribución angular en el sistema laboratorio')
plt.grid(True)
plt.tight_layout()
plt.savefig('Angular distribution in laboratory frame.png')