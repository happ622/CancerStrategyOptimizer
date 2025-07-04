
import streamlit as st
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

st.set_page_config(page_title="Cancer Strategy Optimizer", layout="centered")
st.title("🧬 Cancer Strategy Optimizer（がん治療戦略シミュレーター）")

st.markdown("本ツールは、がんの進行と治療効果の数理モデルに基づくシミュレーションです。")

st.header("🔧 パラメータ設定")

X0 = st.slider("がんの初期進行度（X₀）", 0.0, 1.0, 0.1)
X0_dot = st.slider("進行速度（X₀'）", -1.0, 1.0, 0.0)

q = st.slider("治療の強さ（q）", 0.5, 2.0, 1.0)
p = st.slider("治療の精密さ（p）", 0.5, 2.0, 1.0)
h = st.slider("体の応答性（h）", 0.5, 2.0, 1.0)
e = st.slider("環境因子（e）", 0.5, 1.5, 1.0)
z = st.slider("ストレスレベル（z）", 0.2, 1.0, 0.5)

a, b, c = 1.0, 0.8, 0.2

def rhs(t, y):
    X, X_dot = y
    forcing = (q * p / h) - e * z
    X_ddot = (b * X_dot - c * X + forcing) / a
    return [X_dot, X_ddot]

sol = solve_ivp(rhs, [0, 30], [X0, X0_dot], t_eval=np.linspace(0, 30, 300))

st.header("📈 がん進行度の時間変化")
fig, ax = plt.subplots()
ax.plot(sol.t, sol.y[0], label="がん進行度 X(t)", color="crimson")
ax.set_xlabel("時間 t")
ax.set_ylabel("進行度 X")
ax.legend()
ax.grid(True)
st.pyplot(fig)
