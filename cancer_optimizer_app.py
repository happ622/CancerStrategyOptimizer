
import streamlit as st
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

st.title("Cancer Strategy Optimizer")

# 初期条件入力
X0 = st.slider("Initial deviation X₀", 0.0, 1.0, 0.1)
X0_dot = st.slider("Initial velocity X₀'", -1.0, 1.0, 0.0)
a, b, c = 1.0, 0.8, 0.2

q = st.slider("q (treatment strength)", 0.5, 2.0, 1.0)
p = st.slider("p (precision)", 0.5, 2.0, 1.0)
h = st.slider("h (human body factor)", 0.5, 2.0, 1.0)
e = st.slider("e (environment)", 0.5, 1.5, 1.0)
z = st.slider("z (external stress)", 0.2, 1.0, 0.5)

def rhs(t, y):
    X, X_dot = y
    forcing = (q * p / h) - e * z
    X_ddot = (b * X_dot - c * X + forcing) / a
    return [X_dot, X_ddot]

sol = solve_ivp(rhs, [0, 30], [X0, X0_dot], t_eval=np.linspace(0, 30, 300))
st.line_chart(sol.y[0])
