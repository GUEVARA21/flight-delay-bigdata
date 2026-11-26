"""Flight Delay Intelligence V6 — Databricks Apps / Streamlit.

Analytics, geographic intelligence and illustrative route cost scenarios.

Read-only analytics app backed by Unity Catalog / Databricks SQL.
Financial outputs are explicitly scenario-based, NOT audited airline costs.
"""
from __future__ import annotations

import os
import re
from datetime import datetime

import numpy as np
import pandas as pd
import pydeck as pdk
import airportsdata
import html
import math
import json
import streamlit.components.v1 as components
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from databricks import sql
from databricks.sdk.core import Config

st.set_page_config(
    page_title="Flight Delay Intelligence",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Flight Delay Intelligence V6 · operational analysis and Monte Carlo ─────────────
# The original Databricks SQL + OAuth connection below is intentionally unchanged.
NAVY = "#24262B"
TEAL = "#327B7D"            # secondary analytics accent, not neon
ORANGE = "#D98255"          # restrained warm flight/financial accent
MUTED = "#777982"

st.markdown(r"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;450;500;600;700&family=Manrope:wght@400;500;600;650;700;800&display=swap');
:root {
  --ink:#292A2C; --sub:#747780; --muted:#8C8D92; --amber:#CC754D;
  --sea:#366E73; --paper:#F7F5F1; --rule:rgba(44,43,40,.11);
  --glass:rgba(255,255,255,.72);
}
html, body {font-family:'DM Sans',system-ui,sans-serif !important;}
[data-testid="stAppViewContainer"], .stApp {
  color:var(--ink) !important;
  background:
   radial-gradient(ellipse 43rem 36rem at 93% 12%,rgba(220,144,97,.16),transparent 74%),
   radial-gradient(ellipse 40rem 28rem at 56% 78%,rgba(231,185,148,.10),transparent 78%),
   linear-gradient(145deg,#FBF9F5 0%,#F5F2EE 48%,#F8F5F1 100%) !important;
}
[data-testid="stHeader"] {background:rgba(248,246,241,.65)!important;backdrop-filter:blur(18px);}
[data-testid="stAppViewContainer"] > .main {background:transparent;}
.block-container{max-width:1440px;padding-top:2.6rem;padding-bottom:3.8rem;padding-left:3rem;padding-right:3rem;}
*, *:before, *:after {box-sizing:border-box;}
h1,h2,h3,h4 {font-family:'Manrope',system-ui,sans-serif!important;letter-spacing:-.038em;}
h1,h2,h3,h4,p,li{color:var(--ink);}
h2{font-size:1.38rem!important;font-weight:750!important;margin-top:1.1rem!important;}
h3{font-size:1.10rem!important;font-weight:700!important;}
[data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li{line-height:1.64;}
[data-testid="stCaptionContainer"] p{color:var(--sub)!important;font-size:.81rem;}

/* Dark graphite navigation, confident and understated. */
[data-testid="stSidebar"] {background:linear-gradient(165deg,#26282D,#1C2025) !important;border-right:1px solid rgba(255,255,255,.10);}
[data-testid="stSidebar"] > div:first-child{background:transparent;}
[data-testid="stSidebar"] .stMarkdown,
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] .stMarkdown strong,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p
{color:#E8E7E4!important;opacity:1!important;}
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p{color:#A9ABB0!important;}
[data-testid="stSidebar"] [data-testid="stRadio"] label p,
[data-testid="stSidebar"] [data-testid="stRadio"] label span{color:#E5E4E2!important;}
[data-testid="stSidebar"] [role="radiogroup"] {gap:.38rem;}
[data-testid="stSidebar"] [role="radiogroup"] label {
  border:1px solid transparent;border-radius:13px;padding:.32rem .45rem;
  transition:background .15s ease,border .15s ease;
}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked){background:rgba(255,255,255,.11);border-color:rgba(255,255,255,.13);}
[data-testid="stSidebar"] input[type="radio"]{accent-color:#E6A27D!important;}
[data-testid="stSidebar"] hr{border-color:rgba(255,255,255,.14)!important;}
[data-testid="stSidebar"] [data-baseweb="select"] > div {
 background:rgba(255,255,255,.95)!important;border:1px solid #EEE8DF!important;border-radius:10px!important;
}
[data-testid="stSidebar"] [data-baseweb="select"] *{color:#282A30!important;}
[data-testid="stSidebar"] [data-testid="stButton"] button{
 background:#E2AC89!important;color:#222326!important;border-radius:12px!important;
 border:1px solid rgba(255,255,255,.25)!important;font-weight:650!important;
}
[data-testid="stSidebar"] [data-testid="stButton"] button p{color:#222326!important;}
[data-testid="stSidebar"] [data-testid="stButton"] button:hover{background:#F0C6AA!important;}
.brand-lockup{padding:10px 2px 6px;}
.brand-mark{display:flex;align-items:center;gap:10px;color:#F5F1EA;font-family:'Manrope',sans-serif;font-size:16px;font-weight:750;letter-spacing:-.045em;}
.brand-symbol{width:30px;height:30px;border:1px solid rgba(229,163,115,.5);border-radius:11px;display:grid;place-items:center;color:#E9B38E;font-size:18px;background:rgba(215,126,69,.12);}
.brand-caption{font-size:10px;color:#9DA4AC;letter-spacing:.20em;font-weight:650;margin-top:9px;}
.side-divider{border-top:1px solid rgba(255,255,255,.12);margin:20px 0 14px;}
.side-status{border:1px solid rgba(255,255,255,.13);background:rgba(255,255,255,.055);border-radius:13px;padding:12px 13px;margin:8px 0 15px;}
.side-status b{color:#F1EDE8;font-size:12px;font-weight:650;}
.side-status small{display:block;color:#A9B0B4;margin-top:4px;font-size:10px;letter-spacing:.05em;}
.side-dot{display:inline-block;width:7px;height:7px;border-radius:100%;background:#7BC5B5;margin-right:8px;box-shadow:0 0 8px rgba(123,197,181,.32);}

/* Warm translucent top panel with disciplined geometry, not decorative noise. */
.hero{
 position:relative;overflow:hidden;
 background:linear-gradient(115deg,rgba(255,255,255,.86),rgba(255,255,255,.52)),
  radial-gradient(ellipse at 91% 68%,rgba(232,152,103,.18),transparent 54%);
 border:1px solid rgba(255,255,255,.95);
 box-shadow:0 24px 65px rgba(75,58,45,.070),0 2px 5px rgba(75,58,45,.036);
 backdrop-filter:blur(18px);-webkit-backdrop-filter:blur(18px);
 border-radius:29px;padding:37px 44px 36px;margin:0 0 25px;
}
.hero:after{content:'';position:absolute;width:340px;height:340px;right:-85px;top:-195px;
 border-radius:50%;background:radial-gradient(circle,rgba(227,154,103,.18),transparent 72%);pointer-events:none;}
.hero-layout{display:grid;grid-template-columns:1fr auto;align-items:end;gap:22px;position:relative;z-index:1;}
.hero-eyebrow{display:flex;align-items:center;gap:9px;color:#8B654C;text-transform:uppercase;letter-spacing:.16em;font-size:10px;font-weight:750;}
.hero-eyebrow:before{content:'';width:19px;height:1px;background:#D48C65;}
.hero-title{font-family:'Manrope',sans-serif;color:#23252A!important;font-weight:750!important;letter-spacing:-.06em;font-size:clamp(28px,3.4vw,49px)!important;line-height:1.13;margin:17px 0 13px!important;}
.hero-subtitle{color:#747378;font-size:14px;line-height:1.75;max-width:750px;}
.hero-stamp{min-width:148px;text-align:right;color:#7D7E83;font-size:10px;letter-spacing:.12em;text-transform:uppercase;border-left:1px solid rgba(32,32,32,.12);padding-left:22px;padding-bottom:3px;}
.hero-stamp b{display:block;font-size:24px;color:#B5714B;letter-spacing:-.055em;font-family:'Manrope',sans-serif;margin-bottom:2px;}

/* Truly readable frosted KPI cards, with a warm glint rather than heavy neon. */
.glass-kpi-grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:15px;margin:4px 0 33px;}
.glass-kpi{
 position:relative;min-width:0;overflow:hidden;min-height:137px;
 padding:22px 22px 19px;
 background:linear-gradient(139deg,rgba(255,255,255,.91),rgba(255,255,255,.60));
 border:1px solid rgba(255,255,255,.97);
 border-radius:24px;box-shadow:0 13px 35px rgba(80,66,56,.055),0 1px 4px rgba(80,66,56,.03);
 backdrop-filter:blur(16px);-webkit-backdrop-filter:blur(16px);
}
.glass-kpi:after{content:'';position:absolute;width:128px;height:128px;border-radius:50%;
 right:-61px;bottom:-83px;background:radial-gradient(circle,rgba(220,154,115,.17),transparent 75%);pointer-events:none;}
.kpi-index{font:650 10px 'Manrope',sans-serif;letter-spacing:.12em;color:#C18E71;float:right;}
.kpi-label{font-size:11px;font-weight:650;color:#77777A;letter-spacing:.03em;}
.kpi-value{font:750 clamp(23px,2vw,35px)/1.28 'Manrope',sans-serif;letter-spacing:-.065em;color:#27292D;margin:16px 0 6px;white-space:nowrap;}
.kpi-foot{color:#9A9794;font-size:10px;letter-spacing:.075em;text-transform:uppercase;}

/* Native Streamlit components on the warm canvas. */
[data-testid="stMetric"]{
 background:linear-gradient(145deg,rgba(255,255,255,.87),rgba(255,255,255,.65));
 border:1px solid rgba(255,255,255,.95);padding:18px 20px;border-radius:21px;
 box-shadow:0 9px 26px rgba(65,55,48,.045);
}
[data-testid="stMetricLabel"] *,[data-testid="stMetricLabel"]{color:#77777D!important;opacity:1!important;font-size:12px!important;}
[data-testid="stMetricValue"] *,[data-testid="stMetricValue"]{color:#28292E!important;opacity:1!important;font-family:'Manrope',sans-serif;font-weight:750!important;letter-spacing:-.055em;}
[data-testid="stMetricDelta"] *{color:#366E73!important;}
[data-testid="stPlotlyChart"]{
 background:rgba(255,255,255,.63);border:1px solid rgba(255,255,255,.94);
 border-radius:25px;padding:18px 14px 7px;margin:8px 0 17px;
 box-shadow:0 12px 36px rgba(87,72,60,.045);backdrop-filter:blur(10px);
}
[data-testid="stDataFrame"]{border:1px solid rgba(117,110,103,.12)!important;border-radius:16px!important;overflow:hidden;}
[data-testid="stAlert"]{border-radius:14px!important;border-width:1px!important;}
[data-testid="stTabs"] [role="tablist"]{gap:9px;border-bottom:1px solid rgba(38,38,40,.10);}
[data-testid="stTabs"] button[role="tab"]{font-family:'Manrope',sans-serif;font-size:12px;font-weight:650;color:#72727A;background:transparent;border-radius:11px 11px 0 0;padding:11px 17px;}
[data-testid="stTabs"] button[role="tab"][aria-selected="true"]{color:#A45C3C;background:rgba(221,143,99,.1);border-bottom-color:#C97C53;}
[data-testid="stNumberInput"] input, [data-testid="stSelectbox"] input{
  color:#292B30!important;background:#FFFDFB!important;
}
[data-testid="stNumberInput"] [data-baseweb="input"],
[data-testid="stSelectbox"] [data-baseweb="select"] >div {background:rgba(255,255,255,.82)!important;border-radius:11px!important;}
[data-testid="stWidgetLabel"] p{font-size:12px;color:#5E6067!important;font-weight:600;}
[data-testid="stSlider"] [role="slider"]{background:#C97A52!important;}
[data-testid="stSegmentedControl"] button{border-color:rgba(45,42,38,.12)!important;}
[data-testid="stMarkdownContainer"] h2,[data-testid="stMarkdownContainer"] h3,[data-testid="stMarkdownContainer"] h4{color:#303138!important;}
.section-label{font-size:11px;letter-spacing:.155em;color:#8C654E;text-transform:uppercase;font-weight:750;margin:27px 0 12px;display:flex;align-items:center;gap:12px;}
.section-label:after{content:'';flex:1;height:1px;background:rgba(67,62,58,.12);}
.disclaimer{background:rgba(254,249,242,.77);border:1px solid rgba(189,147,111,.20);padding:17px 19px;border-radius:17px;color:#6C6967;font-size:12px;line-height:1.75;}
.insight-strip{display:flex;flex-wrap:wrap;gap:10px;margin:-11px 0 23px;}
.insight-chip{font-size:10px;font-weight:650;color:#726963;padding:8px 11px;border-radius:100px;background:rgba(255,255,255,.7);border:1px solid rgba(109,91,73,.10);}
.insight-chip b{color:#B97852;}
.footer{font-size:11px;color:#85817F;margin-top:47px;padding-top:18px;border-top:1px solid rgba(52,46,42,.12);line-height:1.7;}
@media(max-width:1100px){.glass-kpi-grid{grid-template-columns:repeat(2,minmax(0,1fr));}.block-container{padding-left:1.8rem;padding-right:1.8rem;}}
@media(max-width:690px){.glass-kpi-grid{grid-template-columns:1fr;}.hero{padding:25px 23px;border-radius:22px;}.hero-layout{grid-template-columns:1fr;}.hero-stamp{display:none;}.block-container{padding-left:1.1rem;padding-right:1.1rem;}}

/* V4 — Airport intelligence / deeper visual hierarchy */
body {font-variant-numeric:tabular-nums;}
.block-container {max-width:1510px;padding-top:2rem;}
[data-testid="stAppViewContainer"] .main .block-container > div {gap:.62rem;}
.hero {border-radius:26px; padding:44px 47px 41px;
  background:radial-gradient(ellipse at 93% 28%,rgba(216,137,87,.26),transparent 40%),
  linear-gradient(121deg,rgba(255,255,255,.92) 0%,rgba(253,248,241,.8) 72%,rgba(255,255,255,.65) 100%);
  box-shadow:0 20px 60px rgba(99,70,47,.075),inset 0 1px 0 rgba(255,255,255,.95);
  border:1px solid rgba(255,255,255,.85);
}
.hero-title {font-size:clamp(31px,3.1vw,52px)!important;font-weight:800!important;}
.hero-stamp{background:rgba(255,255,255,.52);padding:17px 20px;border:1px solid rgba(113,88,66,.13);border-radius:19px;min-width:164px;text-align:left;box-shadow:0 5px 19px rgba(73,59,49,.035)}
.hero-stamp b{font-size:30px;}
.glass-kpi {min-height:146px;padding:23px 24px;background:linear-gradient(145deg,rgba(255,255,255,.94),rgba(249,244,238,.67));border:1px solid rgba(255,255,255,.96);
  box-shadow:0 15px 40px rgba(65,52,43,.066),inset 0 1px 0 rgba(255,255,255,1);
}
.kpi-value{font-size:clamp(24px,2.05vw,37px);}
[data-testid="stPlotlyChart"], [data-testid="stDeckGlJsonChart"] {
  border-radius:22px;border:1px solid rgba(79,74,69,.10);background:linear-gradient(135deg,rgba(255,255,255,.77),rgba(255,255,255,.47));
  box-shadow:0 12px 36px rgba(65,52,43,.055);
}
[data-testid="stSidebar"]{min-width:270px;}
[data-testid="stSidebar"] [role="radiogroup"] label {padding:8px 10px!important;margin-bottom:1px;}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {border-left:2px solid #DB986C!important;}
[data-testid="stSelectbox"], [data-testid="stSlider"], [data-testid="stNumberInput"]{margin-bottom:5px;}
[data-testid="stExpander"] {border-radius:17px;border-color:rgba(82,69,57,.14)!important;background:rgba(255,255,255,.43);}
[data-testid="stAlert"] {background:rgba(250,246,236,.73)!important;}
[data-testid="stDataFrame"]{box-shadow:0 8px 24px rgba(65,52,43,.035)!important;}
.report-rule{border:0;border-top:1px solid rgba(62,55,50,.13);margin:20px 0 19px;}
.report-index{color:#B88264;font-size:11px;letter-spacing:.17em;font-weight:800;}
.report-lead{font-size:13px;line-height:1.72;color:#6D7276;max-width:900px;}
.method-grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin:11px 0 22px;}
.method-step{padding:20px 17px 18px;border:1px solid rgba(94,76,58,.12);background:rgba(255,255,255,.64);border-radius:18px;min-height:139px;box-shadow:0 5px 20px rgba(74,56,42,.035);}
.method-step .num{font-size:11px;color:#B57C57;letter-spacing:.12em;font-weight:800;}
.method-step .name{font-family:'Manrope',sans-serif;font-size:15px;color:#292C30;font-weight:750;margin:12px 0 6px;}
.method-step .detail{font-size:11px;color:#73767A;line-height:1.55;}
.stat-rail{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px;margin:6px 0 20px;}
.stat-rail .stat{border-left:2px solid #D48D65;padding:8px 17px;background:rgba(255,255,255,.4);border-radius:0 10px 10px 0;}
.stat-rail .stat b{font-family:'Manrope',sans-serif;color:#24282B;font-weight:800;font-size:22px;display:block;}
.stat-rail .stat span{color:#77797C;font-size:11px;}
.analysis-note{border:1px solid rgba(104,91,79,.13);border-left:3px solid #D58A61;background:linear-gradient(110deg,rgba(255,253,250,.78),rgba(255,250,242,.63));padding:20px 23px;border-radius:15px;font-size:13px;color:#5D6266;line-height:1.74;margin:12px 0 20px;}
.analysis-note b{color:#25282C;}
.map-legend{padding:12px 0;color:#77797C;font-size:11px;letter-spacing:.025em;}
.map-legend .dot{width:10px;height:10px;display:inline-block;border-radius:50%;margin-right:6px;vertical-align:middle;}
.source-line{font-size:11px;color:#8B8581;border-top:1px solid rgba(58,54,51,.10);padding-top:14px;margin-top:12px;}
@media(max-width:1100px){.method-grid{grid-template-columns:repeat(2,minmax(0,1fr));}.stat-rail{grid-template-columns:1fr 1fr;}}
@media(max-width:700px){.method-grid{grid-template-columns:1fr;}.stat-rail{grid-template-columns:1fr;}}
/* ===== V5: premium product craft, numerical density and richer depth ===== */
[data-testid="stAppViewContainer"],.stApp{
 background:radial-gradient(ellipse 60rem 40rem at 95% 4%,rgba(228,145,91,.17),transparent 60%),
 radial-gradient(ellipse 56rem 42rem at 5% 95%,rgba(80,133,133,.07),transparent 70%),#F5F2EC!important;
}
.block-container{max-width:1580px!important;padding:2.2rem 2.4rem 4.5rem!important;}
.hero{background:linear-gradient(116deg,rgba(255,255,255,.96) 0%,rgba(255,252,247,.84) 65%,rgba(255,245,232,.85) 100%)!important;
 border:1px solid rgba(211,199,186,.58)!important;min-height:215px;
 box-shadow:0 29px 65px rgba(86,66,45,.095),inset 0 1px 1px #fff!important;}
.hero:before{content:'';position:absolute;right:4%;top:8%;width:235px;height:235px;
 border:1px solid rgba(211,145,100,.13);border-radius:100%;box-shadow:0 0 0 35px rgba(211,145,100,.042),0 0 0 75px rgba(211,145,100,.022)}
.hero-title{font-size:clamp(38px,4vw,62px)!important;letter-spacing:-.065em!important;}
.glass-kpi{border:1px solid #E7DFD5!important;box-shadow:0 15px 42px rgba(79,63,47,.075),0 2px 2px rgba(79,63,47,.035)!important;
 transition:transform .23s ease,box-shadow .23s ease;min-height:161px;}
.glass-kpi:hover{transform:translateY(-3px);box-shadow:0 23px 45px rgba(79,63,47,.10)!important;}
.glass-kpi:first-child{border-top:3px solid #386F71!important;}
.glass-kpi:nth-child(2){border-top:3px solid #D38B60!important;}
.glass-kpi:nth-child(3){border-top:3px solid #B86B4E!important;}
.glass-kpi:nth-child(4){border-top:3px solid #5B7774!important;}
.kpi-value{font-size:clamp(26px,2.25vw,41px)!important;}
.kpi-foot{color:#7E7C77!important;font-weight:600;letter-spacing:.06em!important;}
[data-testid="stPlotlyChart"]{border-radius:24px!important;padding:22px 18px 18px!important;
 background:linear-gradient(126deg,rgba(255,255,255,.94),rgba(255,253,249,.81))!important;
 border:1px solid #E5DED6!important;box-shadow:0 20px 40px rgba(88,69,51,.063)!important;}
[data-testid="stDeckGlJsonChart"]{background:#F6F6F1!important;border:1px solid #DED8CE!important;border-radius:22px!important;}
[data-testid="stDataFrame"]{border:1px solid #E0D7CD!important;box-shadow:0 12px 35px rgba(69,57,48,.06)!important;}
[data-testid="stTabs"] button[role="tab"]{border-radius:11px 11px 0 0!important;}
[data-testid="stSidebar"]{background:linear-gradient(155deg,#20252A,#151A20)!important;}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked){background:linear-gradient(115deg,rgba(217,140,89,.20),rgba(255,255,255,.06))!important;border-left:3px solid #E7A279!important;}
[data-testid="stSidebar"] [data-testid="stButton"] button{box-shadow:0 6px 20px rgba(23,17,11,.16)!important;}
[data-testid="stSidebar"] [data-testid="stButton"] button p{color:#24252A!important;}
.section-label{font-size:12px!important;margin:37px 0 17px!important;color:#9A6345!important;letter-spacing:.14em!important;}
.page-kicker{display:flex;gap:11px;align-items:center;color:#987B63;font-size:11px;font-weight:800;letter-spacing:.12em;text-transform:uppercase;margin:0 0 10px;}
.section-title{font:780 27px/1.2 'Manrope',sans-serif;color:#282B30;margin:4px 0 8px;letter-spacing:-.045em;}
.section-desc{font-size:13px;color:#75767A;margin-bottom:16px;line-height:1.6;max-width:830px;}
.signal-box{padding:19px 21px;background:linear-gradient(95deg,rgba(240,249,246,.94),rgba(251,254,252,.77));border:1px solid rgba(73,119,111,.20);border-left:4px solid #347C7D;border-radius:15px;margin:15px 0 25px;box-shadow:0 10px 25px rgba(58,92,83,.04);color:#3C5F5F;font-size:13px;line-height:1.65;}
.signal-box b{color:#235557;}
.signal-box.amber{border-left-color:#D18553;background:linear-gradient(95deg,#FFF8F0,#FFFDFC);color:#8A624A;}
.signal-box.amber b{color:#8E5137;}
.report-rail{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px;margin:9px 0 20px;}
.report-stat{padding:15px 19px;border:1px solid rgba(136,114,94,.18);background:rgba(255,255,255,.77);border-radius:17px;}
.report-stat .v{font:760 26px/1.3 'Manrope',sans-serif;letter-spacing:-.05em;color:#2B3337;}
.report-stat .l{font-size:11px;color:#73777A;margin-top:3px;line-height:1.5;}
.dashboard-table-intro{font-size:12px;color:#797774;line-height:1.5;margin:10px 0 9px;}
[data-testid="stHorizontalBlock"]{align-items:stretch;}
.route-note{border-radius:14px;padding:14px 18px;background:rgba(255,255,255,.73);border:1px solid #E2D8CD;color:#766A62;font-size:12px;line-height:1.6;}
@media(max-width:1050px){.block-container{padding:1.6rem 1.35rem 3rem!important}.report-rail{grid-template-columns:1fr}}

/* V6: native controls are aligned with the editorial design system. */
[data-testid="stVerticalBlockBorderWrapper"] > div {border-radius:22px!important;}
[data-testid="stNumberInput"] input,[data-testid="stSelectbox"] [data-baseweb="select"]>div {border-radius:12px!important;}
[data-testid="stButton"] button[kind="primary"] {background:#2F7775!important;color:white!important;border:0!important;border-radius:14px!important;padding:.85rem 1rem!important;font-weight:750!important;box-shadow:0 12px 28px rgba(41,99,92,.17)!important;}
[data-testid="stButton"] button[kind="primary"] p {color:white!important;}
[data-testid="stButton"] button[kind="primary"]:hover {background:#235F5D!important;}
[data-testid="stSlider"] [role="slider"] {background:#C67754!important;}
[data-testid="stDownloadButton"] button {border-radius:12px!important;}

</style>
""", unsafe_allow_html=True)

CATALOG = "workspace"
SCHEMA = "gold_flights"
FLIGHTS = f"{CATALOG}.{SCHEMA}.flights_features"
TABLES = {
    "model_metrics": f"{CATALOG}.{SCHEMA}.part3_model_metrics",
    "model_bootstrap": f"{CATALOG}.{SCHEMA}.part3_bootstrap_ci",
    "model_alerts": f"{CATALOG}.{SCHEMA}.part3_alert_capacity",
    "rotation_transitions": f"{CATALOG}.{SCHEMA}.part3_rotation_transitions",
    "rotation_severity": f"{CATALOG}.{SCHEMA}.part3_rotation_severity",
    "rotation_gap": f"{CATALOG}.{SCHEMA}.part3_rotation_gap_sensitivity",
    "rotation_network": f"{CATALOG}.{SCHEMA}.part3_rotation_network",
    "financial_overview": f"{CATALOG}.{SCHEMA}.part3_financial_overview",
    "financial_dimensions": f"{CATALOG}.{SCHEMA}.part3_financial_by_dimension",
    "financial_bands": f"{CATALOG}.{SCHEMA}.part3_financial_delay_bands",
    "financial_scenarios": f"{CATALOG}.{SCHEMA}.part3_financial_scenarios",
    "financial_assumptions": f"{CATALOG}.{SCHEMA}.part3_financial_assumptions",
    "benchmark": f"{CATALOG}.{SCHEMA}.part3_benchmark_results",
}


def usd(value: float, decimals: int = 1) -> str:
    return f"USD {float(value) / 1_000_000:,.{decimals}f} M"


def fmt_int(v) -> str:
    return f"{int(v):,}"


def plot_style(fig, height=345):
    """Editorial, softly framed Plotly style to match the Aero Glass UI."""
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans", color="#676A70", size=12),
        margin=dict(l=16, r=17, t=30, b=16),
        legend=dict(orientation="h", y=-0.19, title=None, font=dict(size=11)),
        xaxis=dict(gridcolor="rgba(45,42,39,.09)", zeroline=False,
                   linecolor="rgba(45,42,39,.09)", title=None),
        yaxis=dict(gridcolor="rgba(45,42,39,.09)", zeroline=False,
                   linecolor="rgba(45,42,39,.09)", title=None),
        hoverlabel=dict(bgcolor="#FFFFFF", font_color="#292A2E",
                        bordercolor="rgba(45,42,39,.13)"),
        hovermode="closest",
    )
    return fig


def warehouse_id() -> str:
    return os.getenv("DATABRICKS_WAREHOUSE_ID", "").strip()


def query(sql_text: str, wid: str) -> pd.DataFrame:
    """Read-only query using the CURRENT user's Databricks permissions.

    OAuth on-behalf-of-user requires the `sql` scope in Databricks Apps.
    The token is scoped to this app request and never stored in source code.
    """
    if not wid:
        raise RuntimeError("No se configuró DATABRICKS_WAREHOUSE_ID")
    user_token = st.context.headers.get("x-forwarded-access-token")
    if not user_token:
        raise RuntimeError(
            "No hay token de usuario. En Databricks Apps: Settings → User authorization "
            "→ Add scope → sql → Save; vuelve a autorizar y abre la aplicación."
        )
    return _query_as_user(sql_text, wid, user_token)


@st.cache_data(ttl=300, show_spinner=False)
def _query_as_user(sql_text: str, wid: str, user_token: str) -> pd.DataFrame:
    """Cache is keyed by user's token, preventing cross-user data-cache reuse."""
    cfg = Config()
    hostname = cfg.host.removeprefix("https://").removeprefix("http://").rstrip("/")
    connection = sql.connect(
        server_hostname=hostname,
        http_path=f"/sql/1.0/warehouses/{wid}",
        access_token=user_token,
        _use_arrow_native_complex_types=False,
    )
    try:
        with connection.cursor() as cur:
            cur.execute(sql_text)
            try:
                return cur.fetchall_arrow().to_pandas()
            except Exception:
                rows = cur.fetchall()
                return pd.DataFrame(rows, columns=[col[0] for col in cur.description])
    finally:
        connection.close()


def fetch_or_warn(sql_text, wid, label="consulta") -> pd.DataFrame:
    try:
        return query(sql_text, wid)
    except Exception as exc:
        st.warning(f"No se pudo cargar {label}. Verifica permisos de Unity Catalog y SQL Warehouse.")
        with st.expander("Detalles técnicos"):
            st.code(str(exc))
        return pd.DataFrame()


def small_table(table_key: str, wid: str, limit: int = 1000) -> pd.DataFrame:
    # Table name is always taken from our fixed whitelist, never from user input.
    return fetch_or_warn(f"SELECT * FROM {TABLES[table_key]} LIMIT {int(limit)}", wid, table_key)


def safe_dimension(value: str, allowed: set[str]) -> str:
    # Only filters sourced from allowed existing dimension values may enter SQL.
    value = str(value)
    if value not in allowed or not re.fullmatch(r"[A-Za-z0-9_-]{1,12}", value):
        raise ValueError("Valor de filtro no permitido")
    return value


def source_filters(wid):
    options = fetch_or_warn(
        f"SELECT DISTINCT year, airline, origin FROM {FLIGHTS} WHERE year IS NOT NULL",
        wid, "opciones de filtros",
    )
    if options.empty:
        return [2024, 2025], [], []
    years = sorted(int(x) for x in options["year"].dropna().unique())
    airlines = sorted(str(x) for x in options["airline"].dropna().unique())
    airports = sorted(str(x) for x in options["origin"].dropna().unique())
    return years, airlines, airports


def where_clause(year, airline, origin, years, airlines, origins):
    parts = []
    if year != "Todos":
        assert int(year) in years
        parts.append(f"year = {int(year)}")
    if airline != "Todas":
        parts.append(f"airline = '{safe_dimension(airline, set(airlines))}'")
    if origin != "Todos":
        parts.append(f"origin = '{safe_dimension(origin, set(origins))}'")
    return "WHERE " + " AND ".join(parts) if parts else ""


def kpi_sql(where):
    return f"""
      SELECT COUNT(*) AS flights,
             SUM(CASE WHEN dep_delay >= 15 THEN 1 ELSE 0 END) AS late_flights,
             SUM(CASE WHEN dep_delay >= 15 THEN dep_delay ELSE 0 END) AS late_minutes,
             SUM(CASE WHEN dep_delay >= 15 THEN LEAST(dep_delay,300) ELSE 0 END) AS capped_minutes,
             ROUND(AVG(CASE WHEN dep_delay >= 15 THEN 1.0 ELSE 0.0 END)*100,2) AS delay_rate
      FROM {FLIGHTS} {where}
    """


def grouped_sql(dim: str, where: str, limit: int = 20):
    if dim not in {"airline", "origin", "dest", "year", "month"}:
        raise ValueError("Dimensión inválida")
    return f"""
      SELECT {dim} AS dimension, COUNT(*) AS flights,
             SUM(CASE WHEN dep_delay>=15 THEN 1 ELSE 0 END) AS late_flights,
             ROUND(AVG(CASE WHEN dep_delay>=15 THEN 1.0 ELSE 0.0 END)*100,2) AS delay_rate,
             SUM(CASE WHEN dep_delay>=15 THEN LEAST(dep_delay,300) ELSE 0 END) AS capped_minutes
      FROM {FLIGHTS} {where}
      GROUP BY {dim} ORDER BY capped_minutes DESC LIMIT {int(limit)}
    """


def metric_row(kpi):
    """Render first-class, high-contrast, responsive KPI tiles from live SQL data."""
    flights = int(kpi.get("flights", 0) or 0)
    late = int(kpi.get("late_flights", 0) or 0)
    minutes = float(kpi.get("capped_minutes", 0) or 0)
    delay_rate = float(kpi.get("delay_rate", 0) or 0)
    tiles = [
        ("Vuelos analizados", fmt_int(flights), "FUENTE: DATABRICKS GOLD"),
        ("Vuelos con retraso ≥15 min", fmt_int(late), "SALIDAS OBSERVADAS"),
        ("Tasa de retraso", f"{delay_rate:.2f}%", "PORCENTAJE OBSERVADO"),
        ("Minutos de retraso", f"{minutes / 1_000_000:,.2f} M", "TOPE 300 MIN / VUELO"),
    ]
    cards = "".join(
        f'<div class="glass-kpi"><span class="kpi-index">0{i}</span>'
        f'<div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{number}</div>'
        f'<div class="kpi-foot">{foot}</div></div>'
        for i, (label, number, foot) in enumerate(tiles, start=1)
    )
    st.markdown(f'<div class="glass-kpi-grid">{cards}</div>', unsafe_allow_html=True)


def intro(title, subtitle, eyebrow):
    import html
    # HTML-escape all text before placing it in the visual-only header.
    title, subtitle, eyebrow = map(html.escape, (title, subtitle, eyebrow))
    st.markdown(
        f'''<div class="hero"><div class="hero-layout"><div>
          <div class="hero-eyebrow">{eyebrow}</div>
          <h1 class="hero-title">{title}</h1>
          <div class="hero-subtitle">{subtitle}</div>
        </div><div class="hero-stamp"><b>FD / 01</b>Selected U.S. origins<br>2024 — 2025</div>
        </div></div>''', unsafe_allow_html=True,
    )




@st.cache_resource(show_spinner=False)
def airport_reference():
    """Coordinates bundled with airportsdata (not generated or geocoded by an LLM)."""
    return airportsdata.load("IATA")


def airport_agg_sql(where):
    return f"""
       SELECT origin,
              COUNT(*) AS flights,
              SUM(CASE WHEN dep_delay >= 15 THEN 1 ELSE 0 END) AS late_flights,
              ROUND(100.0 * AVG(CASE WHEN dep_delay>=15 THEN 1.0 ELSE 0.0 END),2) AS delay_rate,
              SUM(CASE WHEN dep_delay >= 15 THEN LEAST(dep_delay, 300) ELSE 0 END) AS capped_minutes,
              SUM(CASE WHEN dep_delay >= 180 THEN 1 ELSE 0 END) AS severe_flights
       FROM {FLIGHTS} {where}
       GROUP BY origin
       ORDER BY capped_minutes DESC
    """


def add_airport_geo(agg):
    """Merge one US airport metadata record per IATA code, never fabricate coords."""
    if agg.empty:
        return pd.DataFrame()
    db = airport_reference()
    rows = []
    for rec in agg.to_dict("records"):
        code = str(rec.get("origin", "")).strip().upper()
        geo = db.get(code)
        if not geo or geo.get("country") != "US":
            continue
        try:
            lat, lon = float(geo["lat"]), float(geo["lon"])
        except (TypeError, ValueError, KeyError):
            continue
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            continue
        rec = dict(rec)
        rec.update(lat=lat, lon=lon,
                   airport_name=geo.get("name", code),
                   city=geo.get("city", ""),
                   state=geo.get("subd", ""))
        rows.append(rec)
    return pd.DataFrame(rows)


def _page_airport_map_v4(wid, where):
    intro("Airport Network", "Un mapa operativo de la red: dónde se concentran los vuelos, la frecuencia de retrasos y los minutos de exposición histórica.", "02 / Airport intelligence")
    agg = fetch_or_warn(airport_agg_sql(where), wid, "red de aeropuertos")
    if agg.empty:
        return
    geo = add_airport_geo(agg)
    if geo.empty:
        st.warning("No hay coordenadas verificadas para los aeropuertos seleccionados.")
        st.dataframe(agg, use_container_width=True, hide_index=True)
        return
    missing = sorted(set(agg["origin"].astype(str)) - set(geo["origin"].astype(str)))
    if missing:
        st.caption("Sin coordenadas de referencia: " + ", ".join(missing))
    metric_labels = {
        "Minutos de retraso (tope 300)": "capped_minutes",
        "Tasa de retraso": "delay_rate",
        "Volumen de vuelos": "flights",
        "Vuelos con retraso ≥180 min": "severe_flights",
    }
    a,b,c,d = st.columns([1.55,1.0,1.0,.95])
    with a:
        metric_name = st.selectbox("Dimensionar puntos por", list(metric_labels), key="map_metric")
    with b:
        top_mode = st.selectbox("Puntos visibles", ["Toda la red", "Top por métrica"], key="map_mode")
    with c:
        min_flights = st.number_input("Mínimo de vuelos", min_value=1, max_value=1_000_000, value=1000, step=1000, key="map_min_flights")
    with d:
        cost_rate = st.selectbox("USD/min (referencia)", [11.0865, 22.173, 44.346], index=1, key="map_cost_rate")
    dff = geo.loc[geo.flights >= min_flights].copy()
    if dff.empty:
        st.info("Los filtros no dejan aeropuertos visibles. Reduce el mínimo de vuelos.")
        return
    metric = metric_labels[metric_name]
    dff = dff.sort_values(metric, ascending=False).reset_index(drop=True)
    if top_mode == "Top por métrica" and len(dff) > 1:
        top_n = st.slider("Aeropuertos destacados", 1, len(dff), min(12, len(dff)), key="map_topn")
        shown = dff.head(top_n).copy()
    else:
        shown = dff.copy()
    shown["rank"] = range(1, len(shown) + 1)
    mmax = max(float(shown[metric].max()), 1.0)
    shown["radius_m"] = 10000 + 53000*np.sqrt(np.maximum(shown[metric].astype(float),0)/mmax)
    # Color communicates rate, independent of selected radius metric.
    rate = shown["delay_rate"].astype(float)
    shown["fill_color"] = rate.apply(lambda v:
        [198,94,58,215] if v >= 27 else ([219,145,88,204] if v >= 23 else [55,119,119,207]))
    shown["ring_color"] = [[55,63,66,190]]*len(shown)
    shown["tooltip_flights"] = shown["flights"].map(lambda v: f"{int(v):,}")
    shown["tooltip_minutes"] = shown["capped_minutes"].map(lambda v: f"{float(v):,.0f}")
    shown["tooltip_usd"] = (shown["capped_minutes"] * cost_rate).map(lambda v: f"USD {v/1e6:,.1f} M")
    shown["tooltip_rate"] = shown["delay_rate"].map(lambda v: f"{v:.2f}%")
    row_a, row_b, row_c = st.columns(3)
    row_a.metric("Aeropuertos visibles", f"{len(shown)}")
    row_b.metric("Vuelos en selección", f"{int(shown.flights.sum()):,}")
    row_c.metric("Minutos acumulados", f"{shown.capped_minutes.sum()/1e6:,.2f} M")
    layer = pdk.Layer(
        "ScatterplotLayer", id="airport-exposure", data=shown,
        get_position="[lon, lat]", get_fill_color="fill_color",
        get_line_color=[255,255,255,235], get_line_width=1.7,
        get_radius="radius_m", radius_min_pixels=7, radius_max_pixels=43,
        stroked=True, filled=True, pickable=True, opacity=0.86,
    )
    best = shown.head(min(4,len(shown))).copy()
    rings = pdk.Layer(
        "ScatterplotLayer", id="airport-top-markers", data=best,
        get_position="[lon, lat]", get_radius="radius_m * 1.18",
        get_fill_color=[0,0,0,0], get_line_color=[190,109,68,235],
        get_line_width=3, line_width_min_pixels=2,
        stroked=True, filled=False, pickable=False,
    )
    labels = pdk.Layer(
        "TextLayer", id="airport-codes", data=shown.head(min(20,len(shown))),
        get_position="[lon, lat]", get_text="origin", get_size=13,
        get_color=[44,52,57,235], get_pixel_offset=[0,-24],
        get_alignment_baseline="bottom", pickable=False,
    )
    deck = pdk.Deck(
        map_style=None,  # Streamlit/Carto style, no Mapbox secret required.
        initial_view_state=pdk.ViewState(latitude=38.6, longitude=-98.5, zoom=3.65, pitch=0),
        layers=[layer,rings,labels],
        tooltip={
            "html": "<div style='font-family:Arial,sans-serif;min-width:240px;color:#202429'>"
                    "<b style='font-size:15px'>{origin} · {city}</b><br/>"
                    "<span style='font-size:11px'>{airport_name}</span><hr/>"
                    "Vuelos: <b>{tooltip_flights}</b><br/>"
                    "Retrasos ≥15 min: <b>{tooltip_rate}</b><br/>"
                    "Minutos limitados: <b>{tooltip_minutes}</b><br/>"
                    "Exposición ilustrativa: <b>{tooltip_usd}</b>"
                    "</div>",
            "style": {"backgroundColor":"rgba(255,255,255,.96)","borderRadius":"13px", "border":"1px solid #ddd6cd", "padding":"11px", "color":"#202429"}
        },
    )
    st.pydeck_chart(deck, use_container_width=True, height=530)
    st.markdown("""<div class="map-legend">
       <span class="dot" style="background:#377777"></span> Tasa menor de 23% &nbsp; &nbsp;
       <span class="dot" style="background:#DB9158"></span> 23–26.99% &nbsp; &nbsp;
       <span class="dot" style="background:#C65E3A"></span> 27% o más &nbsp; &nbsp;
       ◌ Contorno ámbar: primeros 4 del ranking. El tamaño refleja la métrica elegida.
       </div>""",unsafe_allow_html=True)
    st.markdown("<div class='section-label'>Ranking geográfico y comparación</div>",unsafe_allow_html=True)
    a,b = st.columns([1.15,1])
    with a:
        comp = shown.sort_values(metric,ascending=False).head(12).sort_values(metric,ascending=True)
        fig=px.bar(comp,x=metric,y="origin",orientation="h", color="delay_rate",color_continuous_scale=["#377777","#E6A078","#B85F40"],
                   hover_data=["airport_name","flights","delay_rate","capped_minutes"])
        fig.update_layout(coloraxis_showscale=False)
        fig.update_traces(text=[number_compact(v, "pct" if metric=="delay_rate" else "minutes" if metric=="capped_minutes" else "number") for v in comp[metric]], textposition="outside", cliponaxis=False)
        fig.update_xaxes(range=[0,max(float(comp[metric].max())*1.3,1)])
        st.plotly_chart(better_plot(fig,425,padding_right=40),use_container_width=True)
    with b:
        grid_cols = ["origin","city","flights","late_flights","delay_rate","severe_flights","capped_minutes"]
        table_view(shown[grid_cols],rename={
            "origin":"IATA","city":"Ciudad","flights":"Vuelos",
            "late_flights":"Retrasos ≥15","delay_rate":"Tasa (%)",
            "severe_flights":"180+ min","capped_minutes":"Minutos (300)"
        },formats={"Vuelos":"%,.0f","Retrasos ≥15":"%,.0f","Tasa (%)":"%.2f%%","180+ min":"%,.0f","Minutos (300)":"%,.0f"},
        emphasize="Minutos (300)",height=420)
    st.markdown("<div class='section-label'>Detalle de un aeropuerto</div>",unsafe_allow_html=True)
    selected_airport=st.selectbox("Explorar aeropuerto",sorted(shown["origin"].astype(str).unique()),key="map_detail_airport")
    code=safe_dimension(selected_airport,set(shown["origin"].astype(str)))
    where_detail=(where+" AND " if where else "WHERE ")+f"origin = '{code}'"
    detail=fetch_or_warn(f"""
        SELECT date_format(flight_date,'yyyy-MM') AS period,
               COUNT(*) AS flights,
               ROUND(100.0 * AVG(CASE WHEN dep_delay>=15 THEN 1.0 ELSE 0.0 END),2) AS delay_rate,
               SUM(CASE WHEN dep_delay>=15 THEN LEAST(dep_delay,300) ELSE 0 END) AS capped_minutes
        FROM {FLIGHTS} {where_detail}
        GROUP BY date_format(flight_date,'yyyy-MM')
        ORDER BY period
    """,wid,"tendencia del aeropuerto")
    if not detail.empty:
        left,right=st.columns(2)
        with left:
            fig=px.line(detail,x="period",y="delay_rate",markers=True,color_discrete_sequence=[TEAL])
            fig.update_traces(line_width=3,marker_size=6)
            fig.update_layout(yaxis_title="Tasa de retraso (%)")
            fig.update_traces(text=[f"{v:.1f}%" for v in detail.delay_rate],textposition="top center")
            st.plotly_chart(better_plot(fig,350),use_container_width=True)
        with right:
            fig=px.bar(detail,x="period",y="capped_minutes",color_discrete_sequence=[ORANGE])
            fig.update_layout(yaxis_title="Minutos limitados a 300")
            fig.update_traces(text=[number_compact(v, "minutes") for v in detail.capped_minutes],textposition="outside",cliponaxis=False)
            fig.update_yaxes(range=[0,max(1,float(detail.capped_minutes.max())*1.25)])
            st.plotly_chart(better_plot(fig,350),use_container_width=True)
    st.caption("Ubicaciones: catálogo de aeropuertos airportsdata por código IATA; métricas SQL en vivo desde Gold. El mapa se filtra por año, aerolínea y origen usando el menú lateral. Los USD son exposición hipotética y no costos contables auditados.")




def model_matrix(tn, fp, fn, tp, title):
    matrix = [[tn, fp], [fn, tp]]
    fig=go.Figure(go.Heatmap(
        z=matrix, x=["Pred. puntual","Pred. retraso"],
        y=["Real puntual","Real retraso"],
        colorscale=[[0,"#F4F0EB"],[.5,"#99B6B1"],[1,"#2E7378"]],
        text=[[f"{tn:,}", f"{fp:,}"],[f"{fn:,}", f"{tp:,}"]],
        texttemplate="%{text}", textfont={"size":17},
        showscale=False, hovertemplate="%{y} / %{x}<br>%{z:,} vuelos<extra></extra>"
    ))
    fig.update_layout(title=title,xaxis_side="bottom")
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(plot_style(fig,325),use_container_width=True)













# =================== V5 VISUAL AND DECISION MODULES ===================
# These functions extend V4 without changing the already working Databricks OAuth path.


def number_compact(v, unit="number"):
    v = float(v or 0)
    if unit == "pct":
        return f"{v:,.2f}%"
    if unit == "usd":
        return f"${v/1e6:,.1f} M" if abs(v) >= 1e6 else f"${v:,.0f}"
    if unit == "minutes":
        return f"{v/1e6:,.2f} M" if abs(v) >= 1e6 else f"{v:,.0f}"
    if abs(v) >= 1e6: return f"{v/1e6:,.2f} M"
    if abs(v) >= 1e3: return f"{v/1e3:,.1f} K"
    return f"{v:,.0f}"


def story_header(eyebrow, title, text):
    st.markdown(f'<div class="page-kicker">{html.escape(eyebrow)}</div>'
                f'<div class="section-title">{html.escape(title)}</div>'
                f'<div class="section-desc">{html.escape(text)}</div>',
                unsafe_allow_html=True)


def insight(message, kind="teal"):
    st.markdown(f'<div class="signal-box {"amber" if kind=="amber" else ""}">'
                + html.escape(message) + '</div>', unsafe_allow_html=True)


def report_stats(metrics):
    """Three key numeric facts, never invented deltas."""
    items = "".join(f'<div class="report-stat"><div class="v">{html.escape(str(val))}</div>'
                    f'<div class="l">{html.escape(str(label))}</div></div>'
                    for val, label in metrics)
    st.markdown(f'<div class="report-rail">{items}</div>', unsafe_allow_html=True)


def better_plot(fig, height=370, *, percent_axis=None, padding_right=26):
    fig = plot_style(fig, height)
    fig.update_layout(
        height=height, margin=dict(l=24, r=padding_right, t=39, b=32),
        font=dict(family="DM Sans, sans-serif", color="#687179", size=12),
        hoverlabel=dict(bgcolor="#20282D", font=dict(color="#FFFFFF", size=12), bordercolor="#20282D"),
        xaxis=dict(showline=False, gridcolor="rgba(47,66,68,.09)", tickfont=dict(size=11)),
        yaxis=dict(showline=False, gridcolor="rgba(47,66,68,.09)", tickfont=dict(size=11)),
        legend=dict(orientation="h", y=-.18, x=0, font=dict(size=11)),
        hovermode="closest"
    )
    if percent_axis == "y": fig.update_yaxes(ticksuffix="%")
    if percent_axis == "x": fig.update_xaxes(ticksuffix="%")
    return fig


def executive_bars(df, category, field, *, title="", unit="number", color=TEAL, height=380, top=10):
    d = df.nlargest(top, field).sort_values(field).copy()
    if d.empty: return
    value_labels = [number_compact(v, unit) for v in d[field]]
    colors = [color] * len(d)
    colors[-1] = ORANGE if color == TEAL else "#2F7376"
    fig = go.Figure(go.Bar(
        x=d[field], y=d[category].astype(str), orientation="h", marker_color=colors,
        marker_line_width=0, text=value_labels, textposition="outside",
        textfont=dict(size=11, color="#343C42"), cliponaxis=False,
        customdata=np.stack([d[field].to_numpy()], axis=1),
        hovertemplate="%{y}<br><b>%{x:,.2f}</b><extra></extra>"
    ))
    max_v=max(float(d[field].max()),.001)
    fig.update_xaxes(range=[0,max_v*1.25], showgrid=True, tickformat="~s", zeroline=False)
    fig.update_yaxes(showgrid=False)
    fig.update_layout(title=dict(text=title,font=dict(size=13,color="#596165")), showlegend=False, bargap=.34)
    st.plotly_chart(better_plot(fig,height,padding_right=40),use_container_width=True)


def table_view(df, *, rename=None, formats=None, emphasize=None, height=370):
    """Readable native table with formatted numbers and a useful progress bar."""
    if df.empty:
        st.info("No hay registros para esta selección.")
        return
    d=df.copy()
    if rename: d=d.rename(columns=rename)
    numeric_columns={}
    for col in d.columns:
        if not pd.api.types.is_numeric_dtype(d[col]): continue
        nfmt=(formats or {}).get(col,"%.0f")
        if emphasize==col and d[col].notna().any():
            v_max=max(float(d[col].max()),1e-9)
            numeric_columns[col]=st.column_config.ProgressColumn(
                col,help="Intensidad relativa dentro de la tabla",format=nfmt,min_value=0,max_value=v_max
            )
        else:
            numeric_columns[col]=st.column_config.NumberColumn(col,format=nfmt)
    st.dataframe(d, hide_index=True, use_container_width=True,
                 column_config=numeric_columns, height=height)


def rate_heatmap(heat, *, title="", minimum=50):
    if heat.empty: return
    h=heat.loc[heat.flights>=minimum].copy()
    if h.empty:
        st.info("Volumen insuficiente para calcular el patrón horario.")
        return
    piv=h.pivot(index="day_of_week",columns="departure_hour",values="delay_rate")
    dates=[str(i) for i in piv.index]
    fig=go.Figure(go.Heatmap(z=piv.values, x=piv.columns.tolist(),y=dates,
        colorscale=[[0,"#EAF0ED"],[.48,"#8DBFBA"],[.76,"#D5A273"],[1,"#BC6948"]],
        zmin=0,zmax=max(35,float(np.nanmax(piv.values))),
        colorbar=dict(title="%",thickness=11,len=.83),
        hovertemplate="Día codificado %{y} · Hora %{x}:00<br><b>%{z:.2f}%</b><extra></extra>",
        text=np.where(np.isfinite(piv.values), np.char.add(np.char.mod('%.0f',np.nan_to_num(piv.values)), '%'),""),
        texttemplate="%{text}",textfont=dict(size=9,color="#26383B")
    ))
    fig.update_layout(title=title)
    fig.update_xaxes(dtick=2, showgrid=False, title="Hora programada")
    fig.update_yaxes(title="Día de semana (código de origen)",showgrid=False,autorange="reversed")
    st.plotly_chart(better_plot(fig,470),use_container_width=True)


def page_overview(wid, where):
    intro("Flight Delay Intelligence",
          "Inteligencia de operaciones aéreas. Del comportamiento histórico a decisiones fundamentadas en riesgo, propagación y exposición económica.",
          "NETWORK INTELLIGENCE / USA / 2024–2025")
    k=fetch_or_warn(kpi_sql(where),wid,"indicadores de la red")
    if k.empty: return
    row=k.iloc[0]
    metric_row(row)
    report_stats([
        ("30","Aeropuertos de origen en la muestra Gold completa; el filtro puede reducirlos"),
        ("2024–2025","Cobertura histórica, no seguimiento de vuelos en tiempo real"),
        ("300 min","Límite aplicado por vuelo al estimar exposición de retrasos ≥15 min"),
    ])
    story_header("01 / Performance timeline","Evolución de la puntualidad",
                 "Cada punto corresponde a un mes observado. Mostramos valores concretos, el promedio y los extremos del periodo seleccionado.")
    m=fetch_or_warn(f"""SELECT date_format(flight_date,'yyyy-MM') AS period,
      COUNT(*) AS flights,
      SUM(CASE WHEN dep_delay>=15 THEN 1 ELSE 0 END) AS late_flights,
      ROUND(100.0*AVG(CASE WHEN dep_delay>=15 THEN 1.0 ELSE 0.0 END),2) AS delay_rate,
      SUM(CASE WHEN dep_delay>=15 THEN LEAST(dep_delay,300) ELSE 0 END) AS capped_minutes
      FROM {FLIGHTS} {where}
      GROUP BY date_format(flight_date,'yyyy-MM') ORDER BY period""",wid,"tendencia histórica")
    if not m.empty:
        m=m.sort_values("period").reset_index(drop=True)
        mean=float(m.delay_rate.mean()); peak=m.loc[m.delay_rate.idxmax()]; low=m.loc[m.delay_rate.idxmin()]
        last=m.iloc[-1]
        report_stats([(f"{mean:.2f}%","Promedio mensual del periodo filtrado"),
                      (f"{peak.delay_rate:.2f}%",f"Mes de mayor tasa · {peak.period}"),
                      (f"{last.delay_rate:.2f}%",f"Último periodo · {last.period}")])
        fig=go.Figure()
        fig.add_trace(go.Scatter(x=m.period,y=m.delay_rate,mode="lines+markers",name="Retrasos ≥15 min",
            line=dict(color=TEAL,width=4,shape="spline",smoothing=.45),
            marker=dict(size=7,color="#F7F5F0",line=dict(color=TEAL,width=2.3)),
            fill="tozeroy",fillcolor="rgba(45,122,122,.06)",
            customdata=m[["flights","late_flights"]].to_numpy(),
            hovertemplate="<b>%{x}</b><br>Tasa: %{y:.2f}%<br>Vuelos: %{customdata[0]:,}"+
                          "<br>Retrasados: %{customdata[1]:,}<extra></extra>"))
        fig.add_hline(y=mean,line_dash="dash",line_color="#B6A493",line_width=1.2,
            annotation_text=f"Media mensual {mean:.2f}%",annotation_position="top left",
            annotation_font=dict(color="#8F7967",size=11))
        fig.add_annotation(x=peak.period,y=peak.delay_rate,showarrow=True,ay=-35,
            text=f"Pico {peak.delay_rate:.2f}%",font=dict(color="#AE6C4A",size=12),
            arrowcolor="#AE6C4A",bgcolor="rgba(255,251,247,.93)")
        fig.add_annotation(x=low.period,y=low.delay_rate,showarrow=True,ay=35,
            text=f"Mín. {low.delay_rate:.2f}%",font=dict(color="#357979",size=12),
            arrowcolor="#357979",bgcolor="rgba(255,251,247,.93)")
        fig.update_yaxes(ticksuffix="%",rangemode="tozero",range=[0,max(36,float(m.delay_rate.max())*1.15)])
        fig.update_xaxes(tickangle=-35,showgrid=False)
        st.plotly_chart(better_plot(fig,445,percent_axis="y"),use_container_width=True)
        insight(f"El mes de mayor frecuencia de retrasos en la selección fue {peak.period} ({peak.delay_rate:.2f}%). "
                "Esta tasa describe salidas históricas; no es una predicción para los vuelos de hoy.")
    story_header("02 / Exposure drivers","Dónde se acumulan los minutos",
                 "El ranking considera volumen y duración de retrasos significativos con límite de 300 minutos por vuelo.")
    left,right=st.columns(2,gap="large")
    with left:
        a=fetch_or_warn(grouped_sql("origin",where,10),wid,"ranking de aeropuertos")
        if not a.empty:
            executive_bars(a,"dimension","capped_minutes",title="Aeropuertos de origen · minutos acumulados",unit="minutes",height=400)
            st.caption(f"Mayor exposición en la selección: {a.iloc[0].dimension} ({number_compact(a.iloc[0].capped_minutes,'minutes')} min).")
    with right:
        b=fetch_or_warn(grouped_sql("airline",where,10),wid,"ranking de aerolíneas")
        if not b.empty:
            executive_bars(b,"dimension","capped_minutes",title="Aerolíneas · minutos acumulados",unit="minutes",color=ORANGE,height=400)
            st.caption(f"Mayor exposición en la selección: {b.iloc[0].dimension} ({number_compact(b.iloc[0].capped_minutes,'minutes')} min).")
    story_header("03 / Network detail","Tabla de seguimiento histórico",
                 "Comparación descargable de los principales aeropuertos. Consulta los valores completos, no solo las barras.")
    if not a.empty:
        view=a.copy();view["minutes_per_flight"]=view.capped_minutes/view.flights.clip(lower=1)
        table_view(view[["dimension","flights","late_flights","delay_rate","capped_minutes","minutes_per_flight"]],
            rename={"dimension":"Aeropuerto","flights":"Vuelos","late_flights":"Retrasados","delay_rate":"Tasa (%)",
                    "capped_minutes":"Minutos (300)","minutes_per_flight":"Min/vuelo"},
            formats={"Vuelos":"%,.0f","Retrasados":"%,.0f","Tasa (%)":"%.2f%%","Minutos (300)":"%,.0f","Min/vuelo":"%.2f"},
            emphasize="Minutos (300)",height=330)
    st.caption("Fuentes: Databricks Gold · Histórico 2024–2025 · 30 aeropuertos de origen. Los filtros se aplican a estas visualizaciones.")


def page_operations(wid,where):
    intro("Operational Analytics","Patrones de severidad, concentración y horario; una lectura cuantitativa de dónde se acumula el retraso.","03 / Flight operations")
    summary=fetch_or_warn(f"""SELECT COUNT(*) AS flights,
      SUM(CASE WHEN dep_delay>=15 THEN 1 ELSE 0 END) AS late,
      SUM(CASE WHEN dep_delay>=180 THEN 1 ELSE 0 END) AS severe,
      ROUND(100.0*AVG(CASE WHEN dep_delay>=15 THEN 1.0 ELSE 0.0 END),2) AS late_rate,
      SUM(CASE WHEN dep_delay>=15 THEN LEAST(dep_delay,300) ELSE 0 END) AS minutes
      FROM {FLIGHTS} {where}""",wid,"resumen operativo")
    if not summary.empty:
        r=summary.iloc[0]
        report_stats([(f"{int(r.late):,}","Vuelos con demora ≥15 minutos"),
                      (f"{int(r.severe):,}","Vuelos con demora ≥180 minutos"),
                      (f"{float(r.late_rate):.2f}%","Tasa sobre todos los vuelos filtrados")])
    story_header("01 / Distribution","¿Los retrasos son leves o graves?","Bandas disjuntas para evitar contar un mismo vuelo dos veces.")
    bands=fetch_or_warn(f"""SELECT CASE WHEN dep_delay < 15 THEN '00_<15'
       WHEN dep_delay < 30 THEN '01_15–29' WHEN dep_delay < 60 THEN '02_30–59'
       WHEN dep_delay < 180 THEN '03_60–179' WHEN dep_delay <= 300 THEN '04_180–300'
       ELSE '05_>300' END AS delay_band,
       COUNT(*) AS flights,
       SUM(CASE WHEN dep_delay>=15 THEN LEAST(dep_delay,300) ELSE 0 END) AS capped_minutes
       FROM {FLIGHTS} {where} GROUP BY 1 ORDER BY 1""",wid,"bandas de retraso")
    if not bands.empty:
        d=bands.copy();d["band"]=d.delay_band.astype(str).str.split("_",n=1).str[1]
        d=d.loc[d.band!="<15"]
        a,b=st.columns([1.15,1],gap="large")
        with a:
            fig=go.Figure(go.Bar(x=d.band,y=d.flights,marker_color=["#5C9894","#46918D","#D1A16D","#C87A53","#B34E3E"],
                text=[number_compact(x) for x in d.flights],textposition="outside",cliponaxis=False,
                customdata=d.capped_minutes,
                hovertemplate="Duración: %{x}<br>Vuelos: %{y:,}<br>Minutos limitados: %{customdata:,}<extra></extra>"))
            fig.update_yaxes(tickformat="~s",range=[0,float(d.flights.max())*1.25])
            st.plotly_chart(better_plot(fig,390),use_container_width=True)
        with b:
            dd=d.copy();dd["pct_minutes"]=100*dd.capped_minutes/dd.capped_minutes.sum() if dd.capped_minutes.sum() else 0
            table_view(dd[["band","flights","capped_minutes","pct_minutes"]],
                rename={"band":"Retraso","flights":"Vuelos","capped_minutes":"Minutos (300)","pct_minutes":"Participación (%)"},
                formats={"Vuelos":"%,.0f","Minutos (300)":"%,.0f","Participación (%)":"%.2f%%"},
                emphasize="Participación (%)",height=350)
    story_header("02 / Time patterns","¿En qué horarios aumenta el riesgo?",
                 "Cada celda muestra la tasa de retraso ≥15 min. Consulta los valores al pasar el cursor; mínimo de 50 vuelos por celda.")
    heat=fetch_or_warn(f"""SELECT day_of_week, departure_hour, COUNT(*) AS flights,
        ROUND(100.0*AVG(CASE WHEN dep_delay>=15 THEN 1.0 ELSE 0.0 END),2) AS delay_rate
        FROM {FLIGHTS} {where} GROUP BY 1,2""",wid,"heatmap horario")
    rate_heatmap(heat,minimum=50)
    story_header("03 / Segments","Ranking de exposición operativa","Filtra grupos con suficiente volumen para evitar comparar muestras pequeñas.")
    dim_name=st.segmented_control("Comparar",["Aeropuerto", "Aerolínea", "Destino"],default="Aeropuerto",key="ops_dim_v5")
    dimension={"Aeropuerto":"origin","Aerolínea":"airline","Destino":"dest"}[dim_name]
    min_flights=st.number_input("Volumen mínimo por grupo",min_value=1,max_value=1000000,value=1000,step=500,key="ops_min_v5")
    df=fetch_or_warn(grouped_sql(dimension,where,100),wid,"ranking por segmento")
    if not df.empty:
        df=df.loc[df.flights>=min_flights].copy()
        if not df.empty:
            left,right=st.columns(2,gap="large")
            with left: executive_bars(df,"dimension","delay_rate",unit="pct",title="Tasa de retraso (%)",height=440,top=12)
            with right: executive_bars(df,"dimension","capped_minutes",unit="minutes",color=ORANGE,title="Minutos acumulados",height=440,top=12)
            table_view(df.sort_values("capped_minutes",ascending=False).head(30),
                rename={"dimension":"Segmento","flights":"Vuelos","late_flights":"Retrasos ≥15","delay_rate":"Tasa (%)","capped_minutes":"Minutos (300)"},
                formats={"Vuelos":"%,.0f","Retrasos ≥15":"%,.0f","Tasa (%)":"%.2f%%","Minutos (300)":"%,.0f"},
                emphasize="Minutos (300)",height=380)
    insight("Un segmento con tasa alta no necesariamente concentra más minutos: el volumen y la gravedad también importan. Estos rankings muestran asociación, no causalidad.")


def page_propagation(wid):
    intro("Delay Propagation","Qué revela la secuencia observada de aeronaves sobre el riesgo del siguiente vuelo.","05 / Tail-number rotations")
    report_stats([("4,422,412","Pares plausibles con coincidencia de matrícula, lugar y horarios"),
                  ("3.32×","Riesgo relativo descriptivo tras un vuelo anterior retrasado"),
                  ("+33.61 pp","Diferencia entre 48.11% y 14.50%")])
    story_header("01 / Main finding","Una asociación clara, no un efecto causal",
                 "Probabilidad observada de que el vuelo siguiente tenga demora ≥15 min.")
    fig=go.Figure(go.Bar(x=["Anterior puntual", "Anterior retrasado"],y=[14.50,48.11],
         marker_color=[TEAL,ORANGE],text=["14.50%","48.11%"],textposition="outside",
         textfont=dict(size=20,color="#333D41"),cliponaxis=False,
         hovertemplate="%{x}<br>Riesgo siguiente: %{y:.2f}%<extra></extra>"))
    fig.update_yaxes(range=[0,58],ticksuffix="%",title="Probabilidad del siguiente retraso")
    fig.update_layout(showlegend=False,bargap=.52)
    st.plotly_chart(better_plot(fig,365),use_container_width=True)
    insight("Cuando el tramo observado anterior se retrasó, el siguiente también se retrasó 48.11% de las veces, frente al 14.50% tras un tramo puntual. La diferencia no demuestra causalidad.")
    tabs=st.tabs(["Transiciones", "Severidad previa", "Tiempo entre salidas", "Red de conexiones"])
    with tabs[0]:
        df=small_table("rotation_transitions",wid,20)
        required={"prev_delayed_15","delayed_15","flights"}
        if required.issubset(df.columns):
            d=df.copy()
            d["Estado previo"]=d.prev_delayed_15.map({0:"Anterior puntual",1:"Anterior retrasado"})
            d["Estado actual"]=d.delayed_15.map({0:"Siguiente puntual",1:"Siguiente retrasado"})
            d["Proporción dentro del estado previo (%)"]=d.flights/d.groupby("prev_delayed_15").flights.transform("sum")*100
            table_view(d[["Estado previo","Estado actual","flights","Proporción dentro del estado previo (%)"]],
                rename={"flights":"Vuelos"},formats={"Vuelos":"%,.0f","Proporción dentro del estado previo (%)":"%.2f%%"},
                emphasize="Proporción dentro del estado previo (%)",height=240)
    with tabs[1]:
        df=small_table("rotation_severity",wid,50)
        if {"previous_delay_band","n","following_delay_rate"}.issubset(df.columns):
            labels={"early":"Antes de hora","0_to_14":"0–14 min","15_to_59":"15–59 min","60_plus":"60+ min"}
            df["grupo"]=df.previous_delay_band.map(labels).fillna(df.previous_delay_band)
            df["rate_pct"]=df.following_delay_rate*100
            dd=df.set_index("grupo").reindex(["Antes de hora","0–14 min","15–59 min","60+ min"]).dropna(subset=["n"]).reset_index()
            fig=go.Figure(go.Bar(x=dd.grupo,y=dd.rate_pct,marker_color=[TEAL,"#77A6A1",ORANGE,"#BD6746"],
                text=[f"{v:.2f}%" for v in dd.rate_pct],textposition="outside",cliponaxis=False,
                hovertemplate="%{x}: %{y:.2f}%<extra></extra>"))
            fig.update_yaxes(range=[0,max(70,dd.rate_pct.max()*1.18)],ticksuffix="%")
            st.plotly_chart(better_plot(fig,360),use_container_width=True)
            table_view(dd[["grupo","n","rate_pct"]],rename={"grupo":"Retraso anterior","n":"Pares","rate_pct":"Siguiente retraso (%)"},
                formats={"Pares":"%,.0f","Siguiente retraso (%)":"%.2f%%"},height=235)
    with tabs[2]:
        df=small_table("rotation_gap",wid,50)
        if {"gap_band","prev_delayed_15","n","following_delay_rate"}.issubset(df.columns):
            dd=df.copy();dd["grupo"]=dd.prev_delayed_15.map({0:"Anterior puntual",1:"Anterior retrasado"});dd["tasa"]=dd.following_delay_rate*100
            fig=px.bar(dd,x="gap_band",y="tasa",color="grupo",barmode="group",text="tasa",
                color_discrete_map={"Anterior puntual":TEAL,"Anterior retrasado":ORANGE},
                labels={"gap_band":"Intervalo programado entre salidas","tasa":"Tasa siguiente (%)","grupo":"Antecedente"})
            fig.update_traces(texttemplate="%{text:.1f}%",textposition="outside",cliponaxis=False)
            fig.update_yaxes(ticksuffix="%",range=[0,max(80,float(dd.tasa.max())*1.19)])
            st.plotly_chart(better_plot(fig,370),use_container_width=True)
            table_view(dd[["gap_band","grupo","n","tasa"]],
                rename={"gap_band":"Intervalo","grupo":"Estado previo","n":"Pares","tasa":"Tasa siguiente (%)"},
                formats={"Pares":"%,.0f","Tasa siguiente (%)":"%.2f%%"},height=285)
    with tabs[3]:
        df=small_table("rotation_network",wid,500)
        if {"feeder_origin","next_departure_airport","paired_legs","following_delay_rate"}.issubset(df.columns):
            dd=df.sort_values("paired_legs",ascending=False).head(12).copy()
            dd["ruta"]=dd.feeder_origin.astype(str)+" → "+dd.next_departure_airport.astype(str)
            executive_bars(dd,"ruta","paired_legs",title="Conexiones observadas con más pares",height=410,top=12)
            dd["following_delay_rate"]=dd.following_delay_rate*100
            table_view(dd[["ruta","paired_legs","following_delay_rate"]],rename={"ruta":"Conexión","paired_legs":"Pares","following_delay_rate":"Retraso siguiente (%)"},
                formats={"Pares":"%,.0f","Retraso siguiente (%)":"%.2f%%"},height=330)
    st.caption("Fuente: Notebook 06, 4.42M pares. Los intervalos son entre salidas programadas (no tiempo real de turnaround). Gold no cubre todas las rutas aéreas.")


def page_models(wid):
    intro("Predictive Models",
          "Dos preguntas distintas: ¿el vuelo saldrá con 15+ minutos de retraso? ¿Podemos anticipar cuánto durará si ocurre?",
          "04 / Classification, severity & model reliability")
    story_header("01 / Methodology","Cómo construimos los modelos",
                 "El proceso mantiene una separación cronológica para reducir filtración de información futura hacia la evaluación.")
    st.markdown('''<div class="method-grid">
      <div class="method-step"><div class="num">01 · TRAIN</div><div class="name">179,909 vuelos</div><div class="detail">Feb 2024 – may 2025 · entrenar XGBoost</div></div>
      <div class="method-step"><div class="num">02 · CALIBRATION</div><div class="name">23,384 vuelos</div><div class="detail">Jun – jul 2025 · calibración isotónica</div></div>
      <div class="method-step"><div class="num">03 · THRESHOLD</div><div class="name">11,565 vuelos</div><div class="detail">Ago 2025 · umbrales de alerta</div></div>
      <div class="method-step"><div class="num">04 · FINAL TEST</div><div class="name">44,687 vuelos</div><div class="detail">Sep – dic 2025 · prueba temporal</div></div>
    </div>''',unsafe_allow_html=True)
    l,r=st.columns(2,gap="large")
    with l:
        st.markdown("**XGBoost Baseline · variables operativas**")
        st.write("Aerolínea, origen, destino, mes, día, hora programada, fin de semana, hora pico y temporada. Clasificador de retrasos ≥15 min; umbral 0.17.")
    with r:
        st.markdown("**XGBoost Weather · análisis retrospectivo**")
        st.write("Mismo modelo más variables meteorológicas observadas: temperatura, viento, precipitación, humedad, nubosidad y presión. Umbral 0.18. El clima histórico no está validado como pronóstico disponible antes del vuelo.")
    story_header("02 / Classifier","Qué tan bien identifica vuelos retrasados",
                 "En un conjunto de 44,687 vuelos, únicamente 9,742 terminaron con retraso ≥15 min. No reducimos el resultado a una sola cifra de accuracy.")
    perf=pd.DataFrame([
        {"Modelo":"Baseline","ROC-AUC":.67582,"PR-AUC":.35124,"Precision":.3294,"Recall":.5811,"F1":.4205,"Brier":.1625,"Accuracy":(23421+5661)/44687},
        {"Modelo":"Weather","ROC-AUC":.69169,"PR-AUC":.38082,"Precision":.3553,"Recall":.5519,"F1":.4323,"Brier":.1595,"Accuracy":(25187+5377)/44687},
    ])
    report_stats([("0.3808","PR-AUC con clima histórico (mejor ranking retrospectivo)"),
                  ("58.11%","Recall del Baseline (mayor detección de retrasos)"),
                  ("35.53%","Precision del modelo Weather (alertas correctas)")])
    a,b=st.columns([1.24,1],gap="large")
    with a:
        labels=["PR-AUC","ROC-AUC","Precision","Recall","F1","Accuracy","Brier"]
        metric=st.selectbox("Métrica comparada",labels,key="v5_class_metric")
        show_pct=metric in ["Precision","Recall","Accuracy"]
        d=perf.copy();d["metric"]=(d[metric]*100 if show_pct else d[metric])
        fig=go.Figure(go.Bar(x=d.Model,y=d.metric,marker_color=[TEAL,ORANGE],
            text=[f"{v:.2f}%" if show_pct else f"{v:.4f}" for v in d.metric],
            textposition="outside",textfont=dict(size=17,color="#343D3F"),cliponaxis=False))
        fig.update_yaxes(range=[0,max(float(d.metric.max())*1.25,1)],ticksuffix="%" if show_pct else "")
        fig.update_layout(showlegend=False,bargap=.48)
        st.plotly_chart(better_plot(fig,395),use_container_width=True)
    with b:
        st.markdown("**Resultados completos · test**")
        d=perf.copy()
        for col in ["Precision","Recall","Accuracy"]:d[col]*=100
        table_view(d,formats={"ROC-AUC":"%.4f","PR-AUC":"%.4f","Precision":"%.2f%%",
            "Recall":"%.2f%%","F1":"%.4f","Brier":"%.4f","Accuracy":"%.2f%%"},height=300)
        st.caption("En el test, 21.8% de vuelos fueron retrasados. Predecir siempre 'puntual' daría 78.2% de accuracy, pero detectaría cero retrasos.")
    st.markdown("**Matriz de confusión · vuelos en test**")
    t1,t2=st.tabs(["Baseline · umbral 0.17", "Weather · umbral 0.18"])
    with t1:model_matrix(23421,11524,4081,5661,"Baseline · salida ≥15 min")
    with t2:model_matrix(25187,9758,4365,5377,"Weather · evaluación retrospectiva")
    insight("El Weather mejora PR-AUC de 0.3512 a 0.3808, pero reduce el recall de 58.11% a 55.19%. Más exactitud global no significa automáticamente mejores decisiones: depende del costo de falsas alertas y retrasos no detectados.")
    story_header("03 / Regression","Por qué todavía no estimamos bien la duración",
                 "Notebook 09: XGBoost, Random Forest, CatBoost y ensamble. Se entrenaron para estimar minutos de demora condicionados a retraso ≥15 min, con tope de 300.")
    regress=pd.DataFrame([
        {"Modelo":"Mediana constante","MAE":40.1574,"RMSE":67.1334,"R2":-.1454},
        {"Modelo":"XGBoost optimizado","MAE":44.8744,"RMSE":62.4005,"R2":.0104},
        {"Modelo":"Random Forest","MAE":45.2035,"RMSE":62.4448,"R2":.0090},
        {"Modelo":"CatBoost","MAE":44.9508,"RMSE":62.0668,"R2":.0210},
        {"Modelo":"Ensemble","MAE":44.9111,"RMSE":62.1631,"R2":.0179},
    ])
    report_stats([("62.07 min","Menor RMSE en validación · CatBoost"),
                  ("0.021","Mayor R² en validación · CatBoost"),
                  ("40.16 min","Menor MAE en validación · mediana constante")])
    left,right=st.columns([1.15,1],gap="large")
    with left:
        fig=go.Figure(go.Bar(x=regress.Model,y=regress.RMSE,
            marker_color=["#B6ABA3",TEAL,"#93A5A3",ORANGE,"#7A8E8C"],
            text=[f"{v:.2f}" for v in regress.RMSE],textposition="outside",cliponaxis=False))
        fig.update_yaxes(range=[0,80],title="RMSE · minutos")
        fig.update_xaxes(tickangle=-25,showgrid=False)
        st.plotly_chart(better_plot(fig,385),use_container_width=True)
    with right:
        table_view(regress,formats={"MAE":"%.2f","RMSE":"%.2f","R2":"%.4f"},height=312)
        st.caption("La mediana tiene menos error absoluto que CatBoost; CatBoost reduce mejor los errores cuadrados. Ninguno explica bien la variación individual de duración.")
    st.markdown("**Duración real frente a duración estimada · CatBoost**")
    dd=pd.DataFrame({"Grupo":["15–29","30–59","60–179","180–300*"],
                     "Real":[21.1,41.61,96.75,246.05],
                     "Estimado":[66.98,68.16,70.91,73.58]})
    fig=go.Figure()
    for name,color in [("Real",TEAL),("Estimado",ORANGE)]:
        fig.add_trace(go.Bar(x=dd.Grupo,y=dd[name],name=name,marker_color=color,
            text=[f"{v:.1f}" for v in dd[name]],textposition="outside",cliponaxis=False))
    fig.update_layout(barmode="group",bargap=.35)
    fig.update_yaxes(range=[0,285],title="Minutos de retraso")
    st.plotly_chart(better_plot(fig,385),use_container_width=True)
    st.caption("* El grupo 180–300 incluye valores reales superiores al tope, limitados a 300 min. Esta comparación es sobre vuelos que efectivamente tuvieron retraso ≥15 min.")
    story_header("04 / Limitations","Qué limita la precisión y qué necesitamos mejorar",
                 "Resultados que podemos defender sin presentar nuestras limitaciones como si fueran defectos ocultos.")
    st.markdown('''<div class="method-grid">
      <div class="method-step"><div class="num">01 · INFO OPERATIVA</div><div class="name">Variables insuficientes</div><div class="detail">No contamos con mantenimiento, tripulación, restricciones ATC ni estado operacional previo completo.</div></div>
      <div class="method-step"><div class="num">02 · CLIMA</div><div class="name">Disponibilidad temporal</div><div class="detail">Las observaciones históricas no equivalen a un pronóstico disponible al emitir una alerta.</div></div>
      <div class="method-step"><div class="num">03 · EVENTOS EXTREMOS</div><div class="name">Poca señal de severidad</div><div class="detail">Los modelos tienden a aproximarse a un valor intermedio; subestiman los retrasos de 180+ min.</div></div>
      <div class="method-step"><div class="num">04 · ALCANCE</div><div class="name">Muestra temporal</div><div class="detail">30 orígenes y un periodo histórico; necesitamos una evaluación independiente para extrapolar.</div></div>
    </div>''',unsafe_allow_html=True)
    insight("La prioridad metodológica pendiente es integrar historial operativo y congestión conocida antes de la salida, no seguir cambiando algoritmos con las mismas variables.","amber")
    st.markdown("**Fuentes verificables de los resultados del Notebook 05**")
    t1,t2,t3=st.tabs(["Métricas Gold", "Capacidad de alertas", "Bootstrap de modelos"])
    for tab,key in zip([t1,t2,t3],["model_metrics","model_alerts","model_bootstrap"]):
        with tab:
            df=small_table(key,wid,60)
            if not df.empty:table_view(df,height=350)
    st.caption("Modelos de clasificación: Notebook 05. Regresión y ensamble: experimentos del Notebook 09. Las gráficas del Notebook 09 son resultados documentados; no se declaran modelos de producción.")


def page_airport_map(wid,where):
    # Reuse V4's live Pydeck map + add more analytical context around it.
    _page_airport_map_v4(wid,where)


def page_financial(wid,where):
    intro("Financial Exposure","Evaluación transparente de exposición económica bajo tarifas por minuto supuestas; no son gastos contables comprobados.","06 / Financial sensitivity")
    summary=fetch_or_warn(kpi_sql(where),wid,"base histórica de exposición")
    if summary.empty:return
    observed=float(summary.iloc[0].capped_minutes or 0)
    a,b,c=st.columns([1.25,1,.9],gap="medium")
    with a:
        choice=st.selectbox("Escenario de USD por minuto",["Central · 22.173", "Bajo · 11.0865", "Alto · 44.346", "Personalizado"],key="finance_rate_v5")
        vals={"Central · 22.173":22.173,"Bajo · 11.0865":11.0865,"Alto · 44.346":44.346}
        rate=vals.get(choice,22.173)
        if choice=="Personalizado":rate=st.number_input("Tarifa supuesta (USD/min)",min_value=0.0,max_value=1000.0,value=22.173,step=.5,format="%.3f",key="fin_custom_v5")
    with b: reduction=st.slider("Reducción hipotética de minutos",0,30,10,5,key="fin_reduction_v5")
    with c: costs=st.number_input("Costo hipotético de intervención (USD)",0.0,100000000.0,0.0,step=10000.0,format="%.0f",key="fin_cost_v5")
    gross=observed*rate
    difference=gross*reduction/100
    net=difference-costs
    report_stats([(usd(gross),"Exposición económica histórica · proxy"),
                  (usd(difference),f"Minutos hipotéticamente reducidos {reduction}% × tarifa"),
                  (usd(net),"Balance neto simulado después de costos ingresados")])
    insight("Estos escenarios no prueban que los minutos puedan evitarse ni que el monto sea pérdida monetaria observada. La tarifa aplicada es uniforme; no se deben atribuir componentes de tripulación, pasajeros o combustible que el dataset no mide.","amber")
    story_header("01 / Severity contribution","¿Qué tipo de retraso acumula más exposición?",
                 "El costo proxy es proporcional a los minutos limitados a 300 por vuelo. Los grupos son mutuamente excluyentes.")
    bands=fetch_or_warn(f"""SELECT CASE WHEN dep_delay<15 THEN '00_<15' WHEN dep_delay<30 THEN '01_15–29'
      WHEN dep_delay<60 THEN '02_30–59' WHEN dep_delay<180 THEN '03_60–179'
      WHEN dep_delay<=300 THEN '04_180–300' ELSE '05_>300' END AS delay_band,
      COUNT(*) AS flights,
      SUM(CASE WHEN dep_delay>=15 THEN LEAST(dep_delay,300) ELSE 0 END) AS capped_minutes
      FROM {FLIGHTS} {where} GROUP BY 1 ORDER BY 1""",wid,"severidad financiera")
    if not bands.empty:
        bands=bands[bands.delay_band!='00_<15'].copy()
        bands["grupo"]=bands.delay_band.str.split("_",n=1).str[1]
        bands["proxy_usd"]=bands.capped_minutes*rate
        bands["part_pct"]=100*bands.capped_minutes/bands.capped_minutes.sum() if bands.capped_minutes.sum() else 0
        left,right=st.columns([1.25,1],gap="large")
        with left:
            fig=go.Figure(go.Bar(x=bands.grupo,y=bands.proxy_usd,marker_color=[TEAL,"#78AAA4","#D5A679",ORANGE,"#B96246"],
                text=[number_compact(v,"usd") for v in bands.proxy_usd],textposition="outside",cliponaxis=False,
                hovertemplate="%{x}<br>USD %{y:,.0f}<extra></extra>"))
            fig.update_yaxes(range=[0,max(1,float(bands.proxy_usd.max())*1.25)],tickformat="~s")
            st.plotly_chart(better_plot(fig,400),use_container_width=True)
        with right:
            table_view(bands[["grupo","flights","capped_minutes","part_pct","proxy_usd"]],
                 rename={"grupo":"Duración","flights":"Vuelos","capped_minutes":"Minutos (300)","part_pct":"Participación (%)","proxy_usd":"Proxy USD"},
                 formats={"Vuelos":"%,.0f","Minutos (300)":"%,.0f","Participación (%)":"%.2f%%","Proxy USD":"$%,.0f"},
                 emphasize="Participación (%)",height=355)
    story_header("02 / Exposure concentrations","Aerolíneas y aeropuertos que acumulan más minutos",
                 "Ranking sobre la selección y bajo una tarifa uniforme. No es comparación de eficiencia financiera por aerolínea.")
    a,b=st.columns(2,gap="large")
    with a:
        df=fetch_or_warn(grouped_sql("airline",where,10),wid,"exposición por aerolínea")
        if not df.empty:
            df["proxy_usd"]=df.capped_minutes*rate
            executive_bars(df,"dimension","proxy_usd",color=TEAL,unit="usd",height=400,title="Exposición estimada por aerolínea")
    with b:
        df=fetch_or_warn(grouped_sql("origin",where,10),wid,"exposición por origen")
        if not df.empty:
            df["proxy_usd"]=df.capped_minutes*rate
            executive_bars(df,"dimension","proxy_usd",color=ORANGE,unit="usd",height=400,title="Exposición estimada por aeropuerto")
    story_header("03 / Source traceability","Escenarios y supuestos guardados en Gold",
                 "Estos son resultados persistidos por el Notebook 08. Pueden corresponder a la población completa y no necesariamente al filtro actual.")
    t1,t2=st.tabs(["Escenarios Notebook 08","Supuestos Notebook 08"])
    for tab,key,limit in [(t1,"financial_scenarios",30),(t2,"financial_assumptions",30)]:
        with tab:
            df=small_table(key,wid,limit)
            if not df.empty:table_view(df,height=350)
    st.caption("Solo calcula escenarios. El costo proxy USD/min no proviene de gastos por pasajero o aerolínea individual. Método: Notebook 08; consulta de minutos: Gold en vivo.")


def page_priorities(wid,where):
    intro("Operational Priorities","Donde se concentran más minutos observados y dónde vale la pena investigar primero.","08 / Historical concentration")
    airports=fetch_or_warn(airport_agg_sql(where),wid,"ranking aeroportuario")
    totals=fetch_or_warn(kpi_sql(where),wid,"tamaño de la red")
    if airports.empty or totals.empty:return
    airports["minutes_per_flight"]=airports.capped_minutes/airports.flights.clip(lower=1)
    selection={"Minutos acumulados":"capped_minutes","Tasa de retraso":"delay_rate","Minutos por vuelo":"minutes_per_flight"}
    l,m,r=st.columns([1.5,1,1],gap="medium")
    with l: criterio=st.selectbox("Criterio de priorización",list(selection),key="prior_metric_v5")
    with m: floor=st.number_input("Mínimo vuelos",min_value=1,max_value=1_000_000,value=1000,step=500,key="prior_min_v5")
    with r: n=st.selectbox("Top aeropuertos",[5,10,15,20,30],key="prior_top_v5")
    elig=airports[airports.flights>=floor].sort_values(selection[criterio],ascending=False).copy()
    if elig.empty:
        st.info("Ningún aeropuerto cumple el volumen mínimo.");return
    ranked=elig.head(n).copy()
    total_minutes=float(totals.iloc[0].capped_minutes or 0)
    total_flights=float(totals.iloc[0].flights or 0)
    report_stats([(f"{len(ranked)} / {len(airports)}","Aeropuertos en la selección actual"),
                  (f"{100*ranked.capped_minutes.sum()/total_minutes:.1f}%" if total_minutes else "0%","Participación de minutos observados"),
                  (f"{100*ranked.flights.sum()/total_flights:.1f}%" if total_flights else "0%","Participación de vuelos observados")])
    story_header("01 / Ranking","Priorización explicable, no una puntuación arbitraria",
                 "Ordenamos por un indicador observado en lugar de sumar métricas redundantes con pesos subjetivos.")
    left,right=st.columns(2,gap="large")
    with left:
        unit="pct" if selection[criterio]=="delay_rate" else "minutes" if selection[criterio]=="capped_minutes" else "number"
        executive_bars(ranked,"origin",selection[criterio],title=criterio,unit=unit,color=ORANGE,height=440,top=len(ranked))
    with right:
        fig=px.scatter(elig,x="flights",y="delay_rate",size="capped_minutes",color="minutes_per_flight",
            hover_name="origin",size_max=37,
            color_continuous_scale=["#3B8482","#E5AC79","#B65C40"],
            labels={"flights":"Volumen de vuelos","delay_rate":"Tasa de retraso (%)","minutes_per_flight":"Minutos por vuelo"})
        fig.update_traces(marker_line=dict(width=1,color="white"))
        fig.update_yaxes(ticksuffix="%")
        fig.update_xaxes(tickformat="~s")
        st.plotly_chart(better_plot(fig,440),use_container_width=True)
    st.markdown("**Detalle ordenable · aeropuertos seleccionados**")
    table_view(ranked[["origin","flights","late_flights","delay_rate","severe_flights","capped_minutes","minutes_per_flight"]],
        rename={"origin":"IATA","flights":"Vuelos","late_flights":"Retrasos ≥15","delay_rate":"Tasa (%)",
                "severe_flights":"Severos ≥180","capped_minutes":"Minutos (300)","minutes_per_flight":"Min/vuelo"},
        formats={"Vuelos":"%,.0f","Retrasos ≥15":"%,.0f","Tasa (%)":"%.2f%%",
                 "Severos ≥180":"%,.0f","Minutos (300)":"%,.0f","Min/vuelo":"%.2f"},
        emphasize="Minutos (300)",height=420)
    insight("La selección no permite afirmar cuánto retraso puede evitarse ni el retorno de una inversión. Es una clasificación para investigaciones operativas con un criterio observado.")
    with st.expander("Evidencia retrospectiva de ranking · Notebook 09"):
        comparison=pd.DataFrame({"Capacidad":["5%","10%","20%"],"Solo probabilidad":[62710,124153,225475],
                                 "Probabilidad × duración":[68952,127229,228254],"Diferencia":[6242,3076,2779]})
        table_view(comparison,formats={"Solo probabilidad":"%,.0f","Probabilidad × duración":"%,.0f","Diferencia":"%+,.0f"},height=225)
        st.caption("Al 10%, el intervalo bootstrap de ventaja incluye cero; los resultados NO demuestran ganancias económicas o mejora estable fuera de muestra.")


def page_reliability(wid):
    intro("Data Reliability","Calidad de datos, cobertura meteorológica y rendimiento de consultas sobre los activos Gold.","09 / Engineering provenance")
    report_stats([("6 / 6","Controles de Notebook 07 aprobados"),
                  ("99.9747%","Cobertura meteorológica en la población Gold"),
                  ("0.597 → 0.413 s","Mediana de consulta, directo vs agregado")])
    insight("La comparación de rendimiento contrasta una agregación en vivo frente a una tabla preagregada (cinco repeticiones por método). El costo de materialización de 3.74 s no está incluido en los tiempos de consulta; no es un benchmark normalizado entre motores.")
    story_header("01 / Audit status","Controles que respaldan el dataset",
                 "Validados en Notebook 07: conteos Silver/Gold, etiquetas, cobertura, campos críticos y unicidad de clave de clima.")
    checklist=pd.DataFrame([
        ("Consistencia Silver ↔ Gold operacional","PASS"),
        ("Gold meteorológico ≤ población operacional","PASS"),
        ("Cobertura meteorológica ≥95% (umbral del proyecto)","PASS"),
        ("Variable objetivo delay_15 consistente","PASS"),
        ("Campos esenciales no nulos","PASS"),
        ("Sin duplicados en clave aeropuerto-hora meteorológica","PASS"),
    ],columns=["Control","Resultado"])
    st.dataframe(checklist,hide_index=True,use_container_width=True,height=255)
    story_header("02 / Query benchmark","Latencia de consulta medida",
                 "Las cifras son solo del entorno en que se ejecutó la prueba.")
    benchmark=pd.DataFrame({"Estrategia":["Agregado directo","Delta preagregada"],"Mediana (s)":[.597,.413],"Promedio (s)":[.608,.397],"Muestras":[5,5]})
    fig=go.Figure(go.Bar(x=benchmark.Estrategia,y=benchmark["Mediana (s)"],
        marker_color=[TEAL,ORANGE],text=["0.597 s","0.413 s"],textposition="outside",cliponaxis=False))
    fig.update_yaxes(range=[0,.75],ticksuffix=" s")
    st.plotly_chart(better_plot(fig,300),use_container_width=True)
    st.markdown("**Tabla Gold de benchmark (consulta en vivo)**")
    table=small_table("benchmark",wid,80)
    if not table.empty:table_view(table,height=290)
    story_header("03 / Limitations","Qué falta verificar para una versión operativa",
                 "La calidad estructural es buena, pero no elimina limitaciones de representatividad o disponibilidad temporal.")
    st.markdown('''<div class="analysis-note"><b>Limitaciones documentadas</b><br/>
        El historial Gold cubre 30 aeropuertos de origen, no toda la red. Falta documentar exhaustivamente la reducción desde ~27.66M vuelos BTS originales a 8.65M Gold.
        La meteorología observada todavía no se valida como forecast disponible antes de la salida. La identidad física del vuelo y la cadena completa de aeronaves también requieren trazabilidad adicional.
        Los costos financieros son escenarios, no pérdidas auditadas.</div>''',unsafe_allow_html=True)
    st.caption("Fuentes: notebooks 05–09 y tablas Gold. La app no realiza operaciones de escritura sobre Unity Catalog.")


def route_catalog_sql_cache(wid):
    # Existing query() is already scoped/cached per authenticated user.
    return query(f"""SELECT origin,dest,COUNT(*) AS flights
       FROM {FLIGHTS} WHERE origin IS NOT NULL AND dest IS NOT NULL
       GROUP BY origin,dest HAVING COUNT(*)>=20 ORDER BY origin,dest""",wid)


def geo_label(code):
    geo=airport_reference().get(str(code).upper(),{})
    city=str(geo.get("city","")).strip()
    return f"{code} · {city}" if city else str(code)


# ═════════════════════════════════════════════════════════════════════════════
# FLIGHT DELAY INTELLIGENCE V6 · NETWORK SCENARIO STUDIO
# Read-only Monte Carlo; empirical Gold departure delays + documented assumptions.
# No Mapbox, CARTO, Google Maps, MapTiler, API keys or flight tracking.
# ═════════════════════════════════════════════════════════════════════════════

MAP_GEOMETRY_B85 = "c-qZ<%aUcgZRNjcoPC0Cko_wP4cu*y9bswa@V}c_zykKkDplpJdnJ345lC|L@Co9v0RHEn{{O%K%U}QYr~musKmYmvROzqsvE)_OUp|zquO%OAz0Lr=j?>j>DX)3y`C66r$548A(q$?=#Aq}>U-FkrzR@UcuX0fPkgiSXG9O=$*RF4$kJs9)--goTt$Ooc+qX0OW9#R)SkCLzkMrsn=`Q-y=T(}Qt)bL(YR>ze&B3iX<WjA(mGdF5*;qC%)mU?DS;Oz?udhoTQnOu3NXgPRjcPAxm+IZA+sb|wqx!4s7-Trrl<LQ+sXZGkTTRDc)VQBq>R_h#)bDE1sj<SnZlNvOEvz~*b2@cb{C-Gc>36c7JX6VivTCiqdA7XP`KXN}mRASWoEjgszlKp;M|{b)oV-(f$qMq+SFcJQ^yrLNzm>TM&w9<<r~BKsNy`2G`?2`gl-5q(iMK6X?kwHb?xfk4EmK#b6`x?GMadDxZ7Mx?R3A!<kgn&W-F?4EpZon?G_QYG?xT(9-6ikmcT4-{=LOPt@FnjyZ0ptR1l+mR&dHNja@*H;uJ2M^+K=a*-g<w0`!ZLZ!urvxZ>(B}&c&s@bC<q5rr)twe!r)FKcYR+=4JBd?)JXDv)!(}Sz8C!7kj_E(k1AzKH6PYRxcG(6NBkpi|2~oqq!ZuY52LaV>F`lIiy!tTD<)`s~z1PEI&)>wK&?l>vgtB78|Va^V|A~!HN#H-_AR|U$cMo!F_VKz~vE__TD=^$r|rC$DMoe!7I6y^ub-apxF1#rQ7=Gn{#)kr8F(~!CqRrmLG3#X!E0RsW9swqdw@nwBt^8mbZ_woNsA+ql3=!16o+VtJ1!eZuy;jr90`{pK*`TX8$h>yL6rWzFzts`pNQDl|DH=N9h^!aZp~OetR$0>YJ<M&68n+7Nu8A*8$D%yRP1)b9D1#7UwWJnuh8UvvEg1<6!6>A4;>isaMIRd#uK8N^^Fm-dvX+K`)S#<YjP|_MLla!Osn+*X5O$T=DhgIrB0$mNqmlB<E3bPtq0iaawff`gLXbW#X4^ES94rr$R4(rF(oX8+6pa4Dr&=vh;kRe8l%?Zb$X<G4J9F$-R~K(yhDlk~8SLtF`z<^{zD6W$Sf2?_vh6^V}&hTv`O>lk~lV=HSd7y*h4jx87XNclqP&F;z$Jl3TqsmJfXH3{L5(^osh&%A<I^7cVq-cKqD!T`+(ByesWLujNDO9_?+>?<Kmc+vIK*?@m>QH}85I*|6|;vm6#{@8gJf$Ols{mnE$$Eit-|?ouex_O9e{=^3AcWN$ipdsoOm2M3zL!EU+sc5|(6d7Vl(lDe-bywuA%`z^j##wL9kl2#w2PeXk#w{3nnnO(=J<<23xvXb}mGTy&m*1nJSs1S7Pv%AVWly(Mu>5gI_-Nxh`@^h=c3P*oAbI0e-kd!l{+^>f*^cZRVM*1F%OZzbzkgZm*jLUfo=uPsj)haxNS=>lpdGFFDd?|f*{<^2!ww5yHgZe^xTKObx-PuuJP~QCawl3~1m&5Xd-}CF|mJdvqZ|%!kFRjCGq$kRKuPN(o@AYs0b^5Hm%TSDVE(KPL?+?rMs2m=R$?0|utE~EH|4a1m-VZvy%JR$5S8&1m`**Wmda`(;*pJFRlNH@@Mlx|b>SpuiI+)}pPga(Kv*6P|Rb#?g^6nqKA0{`KZmXuWA1$4CV+@n@?T$9mNFRJ(?dR9!w-fbAPOvQ(fiHcG4|(}zh~QH`8MisKZhgCx)X=z}`vtLWhKxE@l-EB&PMkAexOF8E%#70;$97X=1b@K7M)^JuwxGXe-*5K4qItg?7ZfS|p2ZE@%tw0H1>3}>Ak_`A?frgOs)1mecxzMZ>#=F9FSacP%kKrD+sJ#jP``5U+Ww_GnTpPARl2ZuMmmo{!=O1<!PkB$b!?{3wkSVJODwxgD8!S^Te)A=w<Yae+(!#SQs~FG^LJBT(p4K(^OvgIoV?}AyGy`p>ussZehyY`0sOqH1gIayrd4fzJvQYqTf%b_U)ouQ?z0Lyz90C0#!BB!t+t66ohZrXaRe{Tjj>e3yM69C+OG7D?jDpof7dhAKcsM5>m~xtjjEeS=^pguz(R(s_h17l-1oQlQn`=yaEsDpV0WIhy5Ck40;L3hR)%s=?#)3x(B=nekiL&qnnT~5dv>qw-1gp`?dVvYo9bQou})llcMB*fLFz&ru6u_nvS=!SSR<IJkc#xKIGmX)v|TAl;;L)Vc_mb6m44T={?NNX+Y4D+Q0$*qZ{DxAtC(E{KNnJ!+6_9C-i^Tys&x(PTo380+hkLv-L`8abkw1>osk#8nGoK41?(yfWJP?_`pjS24#A1?f=%z8R~p~@+^9XED=lGO5d408uKeb9UReF=C6}gg1458);>phos!!TFI;O@qU1_R0ENa9DyjIEMr2iPkegw>lwyk3$6<T371}4raFS&M1x_+vJX-j8eLASG|OD@SNJ`>Bk?XJ3o1Bcv{@>_`6sicN|Y2EMGN89(+B{#OIw>#ASwI~l>ve~9<Uin@tN^bvIoyX%B?)@Rb4wmRK&R+AD29ByRj&s=<HoG;{^eWZ8Je;(>>ocq;RLMcZmfjXTShw`KQ}@w(R*&IuqnPopFrdxketSXs6xs|Y#~#K%>>6R|968tx>0r5Pyt~hnT~WJ2Zaj82Z&f=TcJwyHl3**hbb%cFv|@Le1gn<InJ49v2M5Q8Gu8H^WcAWr)6RT{CpbmkDHV)RN<(9Z(b<-ELwLI6;hY~MyEk=kiB-;O#T{rT=6<WZWYF?haOHF#!m(Zr=bOiFkj^8}ut1#~u3>j}9-j&Uy1T%xEu(UpIPY9FX9N1JQpW6Wq26S7nZ$oqr@8VV+JaSI+`BskioxV&Jtmm0JOwJ8@r*#FGAZ=7RH2ax_pXZYwBYDgt}6D~3f%#06mt0*G}FeN(4pA+tQy(Ug06aK^$mxa&<jl5pe`^!RNB{7jjGPEWJ5`-&Jg54$Ez;&OIQ#1t*XZ(D+{Dt6A(o3yVE(lEGQ()8G^uUvFQ)kFU;Cv8DvJ2XhmG{v&9lSrZkb?(g6X|gcyPJt3v8zX~)W1Wn6?R;hKk{tA?s#la)5LQE5{)oQh*xT?bH4%|n5^C-#K4sHYi1t2ni+|JDwS3`+7tSHxIZpcUIXgB!h()6|^m<es_Nr1)mDsZbU_-^hjCR-7tkEU-eokaPZ(YwT2~7HsU3x1frgP#KIrgAt$?tSso1SDyGGGX6DP{jfs|Sz0H<^txrDNb%)T$vahc*P5I)u0eAH3r#`vPQb9PWB0;^!J4S4>u-m0KiU<pK!eSRB7mxPIC#d+R8@mf#x}J!)l(UR!U~IzQ#z^3cZ}8@O4{Ku@p_WSE{F;$#fq1z#<~2o+)}eVHTI&Tvf0f{FHr^3S}vyC`X(vp#dOhC=NhW6?}~eHpsAw&x-QT}O%$AFZljv3wjz9LJXA|zeG^u0$&s`!+7+kDgFj6e162RTXr_2WHpQ;74&Nfm^T1VXr>aHBER4e|>rNe<nBw-x#viIyI=m^>PP_9(Z5fr#XsmO;l{D_G6A05K3$;^?U7q-z=B7WaBCM`nxWInxUe;xrASX-FQsJiu9M*KsX$I8lvi^kAbW-$(vIbK8)ZW352X{EI*?Z;V7JecBs&F{?w!qX3$`X3H!ys>0t?IB!RZ@4)sELUO{xYROfl`HSI2w)@?hdj#d3WC>7mqu-v{^XdbSOOi`;8JfL`-X0aJ`P{N|Rp4Rmsh5d}nb`Bw6raj_H!!f}wQ`zp|B_O4_>X=&F*F4C@NLgEELuOn1(%+BNF>mlHoaoWgGwG8J^UbKrN24_cZR+#A_@EUSLHd|7p+Cik>z@cNt1ZZ{lHgH3HL#_*k}vSLFiS6NqMxVi-uR0reCOwD;(YFlyP^xslD&~}|D$qP?^w>hg8Sfxhw^^oGFb}V)8O)WJ`i0{4Xb1E<v4*D#V+)CP93GCj3ro($lUtN086(l;#{cuymlirCH5Zo?W+MT?2M5pv6F!6SAE&Y<}Kos{D2=2VjJ5Qh{b;{+p^q%QU+SwAAch!4HndC0QC0%NQfZVrBaso@ax*`w1T}qmvL)69h-VLQDe+NBoY-q!W>B7x&k7r@K_xN|BAw+Eso$|iV4B5U;#nMt?=~l`m6xgXQ$zKtCc}U)c?v(bG>`JbfwTV)3QmmBUR+RQ0GT)^J2Y%0;T246fs~L1ZPC6@a2KyAMP@}$L36b`$MkAa-t>{Wl777ZRYe_3T^a?|POMBOpl-qDoU&*Ox0TTr_3~X>UlI%*_QYoec<RPe}z<0hWkuQ)eyj;Ff>)xt<G-~?WKmi0DXx*-E^G$tGF#v~p9BfCOwx2!mJI6PoYMxt5duXi<<Z%*|$u1jQM+=9Oy0A3I?d@cvHZ0b#uv96Bsce<S;#W_>HZ7cTh_tthn_Bl}^NoI<OBSU}yT8=^w1p0?$6iYZb-IQx#i@#h9Hs)_pgJHY)hA3$p%Fom6NKQX6~UV7)I0I*qMl~op7UP1sQ{u=of_Ye9FOXQ2Pg*`Un+jcPTGNIS)H)QOAp{AZZ3k)NnNb8ij2gH94uF(n0myvke#f9<!Q8{X`!K77C!|D=8&A&xP^8no%89J{n5)>$@uolzs)5b{T+aB3hhOAqH1&)P@N>zuCAo3#!c0eN+nOtI{0>Q`4pk_G}UZH{g`-fJM5B+HU<GED7Nn9SoQd&w=w#9{D!>bUA`JtpRIR9-89bZ7YaLfa={2X3mcG7hvdD~>*P*WGa!Ib1Z$P1<RMtBpN~r8zM6eLDlc(0zidTvnS!zb#5u<Ll&_pb!$JQz?0h2ZfVdM_g2^)lMrht7Ixt!f1u_KXIV}SIjp#c=<_%);#s}@4f#IWq55(JID@TEAlULwY%CB13lPD7^Yu7ft=CoD@FA!0LTP4+if7`tKb?GYgb>NGt!!!gg!pco>rHCdur3S35%`JffAlD@e1(JIm*2*qH`Kk{1P*i09pyOFo;z_6?z;Cpy3vPsy0@Iaas@nu$p2tr}ie3uvq9lfqP_$~Ox+sU~Msf&;Srha_aRJX}gwo&bY9&iI7mT9DsF@ecri+=dYh~r(bG@j5ZmfiaLXE+6YQ<f5=%%<0_~vm@9N<Pf#|yrSP3g2~hp2PFN%=vbAoTGMaxl|_XdO~5Z(Pa_+64caZUAyuu#3{FX5j$fJgAo$1|8F_p!zC#oJ66>d*u_Amlu0w*dFcvo7g`qv_o}(g4|x(;AR{wov3WfFwKzEeWKN~Uel_nrZ?C;OSBC8X!k-jHqyd2i#rru(FFcgRkxbsgER9nKB_s2PlB(Jqh0W9n&7P3DSqZSWp2zBqXT(ZYMo%=0YKSKglfq9K^E9(Cx>qHL`%g4=1ePFQ_3P?O?nKSGOB9ruwTbCnX!9wCXRW9Sb6RAj9#*U^u@cSuyr?fqCLp&PElVwvpL7_fN!h5w!Dw#OzBYyC&A$08(dpEnv!sg`La_Bm}ZI;mvnS0;7};BJd&;<7%9?^F`=1-r)U~P{80+;55F`APpp<XM0~IXm>0U0bkD12qH>DO$@Dl&K#M%~8pSyMa;E=tg1-qZppz~fC*enuk~3~@;y8yBm3z<a2ENO6A0S(<mhirqh5@fHPU1w}%DvdpQp|X%AUTOB6fSIoM@qP#)YaTvN|^Ap2$7*`N5Z!zOe%Qod;mG`rFY8p{^=Y%N{!8$B}u|`1k|Ji1Jk=wLHElIN$SxaF7v5zOQ}lh)l_QIKsavUBAe9hJ&bm1#BG;~j5;LW>y~te2sl4*54<Yjbg`j}ddbxzbG*>%x`K^KRIU<QKgiCz-Pqu9YJl&#v%s9VG~fu#U=j2HWp>eeBDTdR<qK<R(MQqUUF=tO1D9XUpDdtZQD<)0#RR%b_tWA1AMP%k)TjrX>V=x`vXgbFjh}Qis=^JYrTB2nH@Zfu^^4kvxltAk=mLPI9uEdt*y#|e$>wD}si6vms|`rkqPTKp(9#ocVIdOwEh88VZc(U7-YHjzL8nap@WO8Vu9mviHgO6IqZW!%R9^+q26q<H)w}41Q~{y+aaN}nPD4ad=%#M$5>7*VQQAHzlbIe^T~3PIc5rHTsMAl1+gWJ!+QolU=t{d6ixAK+kmAHG4R>h-1C6D*mxF|NOsyTdhv>GN=->#a7U~LRVjJ3^kp-M;$4;vlH|rYXTJ&ixFuAqT1g2bWbJL=Sq&7$wX5pC5Pr<b%9EA;Bu29@B)>2{JgTg`4w@Sig1h^_;$cJt!Y@SWY*ckqx+1k)7LXoze+-Y$9KbnG5OK4j#QF&1%0T2UQx;ZB^SNoy$++a7-(t;U|crC^081#AJI9E#VWh9V5FJ_<)8b$Pi+xpP)hCaEY5UTm24|u`CvBH-|tj1tv!)k=RzW`>?X&4qTq(ic!sTw{!2nj;7_8BQcKym^Ug+_Zc#Ht;bnMuV!T8bv%toV{v?+J3#On)9CkbVm}rqdDU=pK1%x$s-#ufr*WB+!IkTLLbkklwL9QruIa9YUJzWAmLYMmi|A^CXWq0ON9**%5$?JEpx(3W~V(Aak+|RGsN3RQD!Fs?$QNDIkatlEVuM(Uz0caVPay4BS|z@$1!P(klHc=YjHDhC5y_dJc|JosI-dMB#Dhk|MNu4zzhz?k#0=ZMY})QQO4u%6RNGsv5v3CNL7!@_sfS&~nuNvYWkgI}HG#J69dfWGpZJCaQ$Rcn%7M$3cIU(FHhgD5bBAf{Fz%*6z3)!-m#nAUuydt3oMsJYuhe=1`#AT0)exzHuIhKdO<)Zn=QL6)^1?<tB?dKC!|`1+#?FYAXS2mI!}}n?*5qdNESLL?Bs1@s7YTTtp1jg2P4clg@IxNn611N`2CZEE7UrG?Gm00_rRmT!SYf&Fts)1BIu9A!@GS4y}Qj)8hV$h`)v(bF6zA4&7u#V9|wCPJe=)DIaNsp~1L3w5gbq8Ic4Uao7T2rDR2M4NM+{A`gKn6<DW$k{6h0sn}@WO;no^F!+FWAx6%?kBc-nnJGfk!m5f3><2E`Lvh|*qe2-Sj?+OWvLIgTP`cj;@0v7^bhIn<+gCI-=`|h}`{i{t)+*Gv<D^||z$U@fOJDqk4>^o7>Rmj(YZ0rWr1ue~Vg^R(=ztuehkStSlG@3nk8i>;G;2D+w?&W9V_sKAi%s~W#-Wa64q9I0S{Yp&EVV$jN&O0ig|?yrhq>E8;J>9$Pt`b3o4Qtp+o?K;DWuwb5PR3cfKcHNkJ14rcQk7&N1gPb;VP+0YALl{TlKJrtoHcUg7C=RC*m=cq*i*j-mUB=9Snd#Sil;vyEqZWerHn)wOwjvWR-|3T}(oA`lhH_5x?74x;`7thGLfak_fTOM_LH<70!f4hg#F|Xs97Qa}yM`m5sth51N3a>|fvjaDN)&Y!$f7QuP)zH=qyr=^7<^!~isg444StTH2<!xe~$}eTi@w3q~H2whgt>R?SJXNrd=Z9d9MzS=Hq&(=+Q%9aYW$fqug-RaZ!-ybr(0q8x61lS!3(!@J@3dW|5iJfH%^ZlxnD>Yl)?St(I#w$6>D_W!&nv){c|85RN=6n!GhVA*&mhO4nX29PshH*tf7N*bdCfk>1H>d@FyTn9U1J#jtw-P6EDp>af{u4!!Z5h;6TjRBRg%7yU*H!aFbT;w1bcMJ(cGUPEki{2ktLxtwg18;5GM-D<3X-heX6PSJo4egqBm2Od2-AP?5+pMV)y_EF*zo48EPy#o$%*?GYVX_%?fOn9QU7=$F(hRVa9CRqpit&&-n5=N6HPofYYoeu7ALAKxn?7bH5<sLiXq$Pr(>LweBI-^+t4jIQ$(PA;%w?b!=<edqor?-vQCJiv?8eCqfH-a>C&)nz^@dzKsqUYX(&#qu@}P^qc)XDCTaL#?N`xoHUcg?$vgBS;F^l;&G)V%s!wf0z>Vm?+uL_Yn3~&1N+jdbYPIvy|&aG^&MX<ubE{Nm?a-QE8lR^1CQYC2TLekNIn@sGWO3~ZrYNpDs0p0@OgVoVmiNp&Snn#0HGxA1gRYl;|r6aLI!Ed)nJ#gBiwF(|C^N~9sCf>FhtxDV=2X-X83Qw`ZL3>m+=`;=&gj06_D=HD(geEAHRfQW5oM*pTvbG<6F<Ka~QO%Y`vO&H-Z-wq;2QEt@2D~&k1AF9GF++A|7+gVMv-axv=K}4(<BJNg$-*$VV&>i+DK_At(-AYF$63ig&xSNk>)2VYX$U?39HaAc1Nc#^YWJi<a*JU0gV6b!p-P}V$g9EoJ~OL=g|SAMtHKaa&e)_*;WYyLA$dur1?VZ!Jv(PB<SS<DcMzp@M;d_()kuaH)HM7ZpwNWwKn;a?nC~9#`cZ))Dj7p5e$pMzfZJ5&V|};r`qY40vUgq3_I?!M?}}RnZh5UPu|1yl{R7BD)P@;M&yrK;aaX+c+_?@St&{5<zn455tikvB=o>OgMD+&cOy)BW#arLPSVp+XF&iBbBr8MB4q&!n9eaLiOu)wN(Vq$ftGTZQ%8D>mfgspYC8EVUa=h7j*%E;jMWwe8fmKqXCr`FZWId2;5Pm37-jiM>g=scq`XM0vgEl4Drbb}3VT;y6`qE9Udm|Yaj9BUiKo+s9N)i~|pn3+fn?GxdX6k0;F`mHjH3E$*VwHiRX)I@Tn|UA&24{XjZK;`wXaEaJk>UV@+UPpyK&bi^D^QVXb}~k;vH9)M@O%Zj4f+p5m@wW$6#zw}B^F1?LkR(Tln@crF&UHH-Kb1dkE)DB?QclBs$_Md%<NNQ&<@Czn6pUfq6`J=<)Fm_y_Hc28=8_TobvctqER<O1;<X=5ISUF^V$j)eCKE&QB35FGQ+{)&^P+;SVP#6bYgZMnK2XYbEU9BtEJ`a8X-8se-y}L6)OQ+rhKZ5dblNiBSK+>lLbz8-7XAsa~yI@=Zyl7BLx?Jr;h|u4T=gI0K36xdEXP`LE=y}?w@8ra8kZ$opW^uAk@eaRAC{N%(-KNrckpmGypXfIE@u`lbXmwFje}PDYuf98bF{#$|Ex=J0EY@U2BEa&jLnikA}FJmTV<ABwd;)5A35M;Hg?D8Mmk1H!+$ilE6?_<TGi_&A2Bn-<Y0|e+xpJ_IVgZmut@uU=b=;Q))wCl!R1gF;t#*p_*-*QF$r_S`v*D39O_gL`P*zR1ceGe;A9ER<kqR^7^$7uRU^qX=mFb6j@N8X2ioVz!fqo5J5e9Cx{CyA+E<{Q2kL}(eiLrrDmkHLt&uj)b7?Z1)eeo+G96@8HZ=b*wE?~dfzgA-Jq}tFtNh)N$K$jpQ%}&IHS)V+HJg%QHR}7xwK^v%q=}g-?IrT-4-yz0OhK&HiOa3g0LM2+~VX-;;s$as&b!G8Ut6IpP7I%iH%c0DgtIIJ5gXdv|XuvLR?M=uk1ffSvvzZ40=CC!V)VceeT314h9bi(8fxh#6@Z#2;I3A%wzM?-}BM3ARE{qQZajBAPNBr;YGLPy%)rXr>9#%af_}vRk-OzLe?vxy#^@?JaQdwtnKGe_o$*01G;zH-x)ZwG`Zqx8GjePJix^zqf&ayvgk6cTex3^aTkJ%Vi2xTa7~Z#m~~nw1a2F>wj&$@Q8a@Q9R;B=9u<(k<2-tJ&W#bNybMO-JOprQxUu>pBh`vLYrQzn?U{55jzxDibl46fWFK@-<c4QiT9GKN8}cmLYY-5^nTP&7a(J}qDMGe9uZTqHX=000A4x{0g-AyhKLnB{0tb0CEKD_$%}k_7Igxi`gDpndj)CSpaSwopD2~soH6J-q4!GRRpSOtG%p%erT=m&~UoneI4=&#<2TbN1>zLsdNx=-xmgSG{Z<BxefB*g;|N3wL@mEWrdH`9gH0K;ZTq<3h2bynn2~jB}1_Q(gg!o3l`GK6?)*Bg?yE@*;*4~iccg#lm#RwyLkGrJ8)7@KK&5);ixc88@wO3y3YN)B-qnWir0U<r^(V||`Ucz}d^lH%ng-23~7ODFRlHx=F2Ha_gP!MPN2tJvlEI;OW4;*mF?6Q1r8GNb9;llL=vD(4t6JT^|nbUIGDZ8+n5{Yx_NFh}X;2}K1pjRoUZ1|2U8L@(d0HrS!>aL_&=>^`%I#%8FxRGnGJ|w>3Qg>{<2C{`f*K0`DXSy?tRE58&GYn)xoEPr34nLYB^LO)`cJd0Z$QD^b9!4<T+6n5?T#~fkR>b9vKy@h<@Gz7u`U*^aa<CZC*MmX1f`k{!c^_Q7D|~zt05kwf_e%^Lhp`5*eS9z^RFGm|C#8@<vVTh;lYLNsD&75-my8${#4F$?u=PN^5n_L9WbBhn8ldF`DfXCXry_BeD_qFCW6|YIyJNYm7BkchM75ttAy*Jz&MZDK??%EQBahneXxZnIdbSYLtEj$p7^XUn&K398qzYpSXQ31}M+S%P=Ay*<tQ+x=sU*e7T$`a#yb;Szhvo)Y_GDO@Co-lDqiee2NUBgFS&WodBCt^4Q6Vr&Jgsv7JBQwS$GBTP<&IpjY+D|K4+8}A_#jbS{GR0;lejV_jKky{m^<WPq@`E-BkM8luSx*qMr>~dpwOzz3#pBFd&_jZX$tPRJw{K#%r~TCwbo;<j|cb^D)4s|>m%Q_nT6=u_1|J}*YgVD;S)I6Bmzs!3FxBEGL=9wwxh%P(+Tx4hi)Mj#-gKntUAm{Cwe}?Bg1Ne{LVDuKKX)22^$iDxSHt4cjp0@Q-#{e4Z`JgQ_}}Is$~>FiprSp<~<Eap2ltO@2(y!GZltx?tVWj5mrei%g-$bHtjq`ocmun;;(=Em%sg+ixCUH^VtGyW?2L(h2$!!W2o;4$c9O{Rz(t&O5Bv?M5z>_n|;eX8wDimp46@k8K@c)v46;(G6VW&sZLw2aAY<GehqAsaX<T!*VNEi!BiKZ0NRvO<vi-zSSci}$U(|d{n^cc@$L*&B7WcaD=W8?IU1L_<s$D1lGy>NY&#+8?^gcQ)Y}hz(~VD^{4|751AH5PhbB4#7Wp=DpQiQGq<>nHZx-x*HQ$!{^FDarBX1k%)Astb`95vUPaF5sw$43o_ZtLa&7;bI%#Hwfp%1MhANHoW+!1*1R@{Tc=ZuUzFpTw{-yTT2os##0v|AX|+n<z$DA)hhz51fyw11=Egt#8bbUt4ZoZN!&W<XfqsPw1384%=YK$tWdls5~43KEsMX*w>|hoN*^N~6pui1bH<XyBwQqJZEPgc3O-5pz2arLz~08`+_hulBG^_U0MKPGnME?O}-R9Zr3R*&$W&dLxG;`u=g<2Y6xsGkRYTv(lO4RLP;6Je<AdGsWCF2Qa}N`P5*-%JZOBawQKV*HtI&ahd*54mHRxlb={RyhOa0kNLQ`o)W)UR_X}h)z?iPs3=H`HDeXX)Zw3336+sVDz5H0d>UOmGOB22OmNpw<0=V6g2W%5O*3!KFkctsppWMa%6d4Hf9KcbeA4+rkDB0)?~s#=J`|7mq~P{0^`FrAf!Cr%UUG#KSayLydg9p#3AnB%IatiyT;OMK^8xP_%d)Yfz46hADgyXn1F$#P-j>7(qW$xvVef3nazUdu=rAX1o(@6`4j`?KU}HCNFrUW3qZE-l0O*H$G8cmReTCj>>6ts=m{*utU({@^RAKtgd-qnv86;xhaZ=HZFCR^9uHsP?{xT-WQ4j^O%PsNNupl-%;|>CDlqUw5r04XTMLp9&XvR3a5i-b8YCaiaxQ52ai-X*D4fl=P#o%=hv<Q@jM5XE>>2OEVB{SdvXHW$|_;CN`^C`%E)B{Ci=6zYueH~`^|KXqCcwA-_I>L)*7+R6*6B&j=m$3{(BT|YsEsZ=c70HTL^3kRISI9a+1qyV|WYtnkp)Q}qihq0`p&x5rf97WFAP3zo_Zas`N^u~LQ99X30MZ{1KON*yYJn9V4^s28c~dmqT(vXJ!X~}VOciR@#~lf3px*XGi<z=Zof0O*O|XF3pQ&LU*o7-e!S4{D0+Z8>JL71=vz21i4@);ziZ{C;&6-m4OU1Z<FhwPk^?srd1MmyA4KYUWhh`s!@iKDSD73$15*Yuu{ShBeJc3hRD|Lgx^hO7m!a=48DCc-&5w1vXR`ws?)bVsQiMLyTjmLdg0K49C-_4U5Xn^ek-kTiOpw+TI-zt7_RHW~zfx~QuS(mW>-(QQ&IZ6x4{aqq24{wWl3vhY3f2(vy<vYfq;9aB^kw>NbZOq$y_@h(#`v>3ma87i5PpjF1xS+PF0L=*U<d7VeHir(KjRPh#;Wu@~)C(Aazn~&`A&N}FA}x1XOw74C!>Sm}%AE>GymshCxRcWTDV4oEahw|wap*gpw1(X2l2*cslSpjfKk=9F<kZOo&=ZMkh>x0@vSpq(BWj4{3o?78uTyh=Bc4tDpNyYha6Z0*XrYG+z<P{EF?Ts%HWMgVJ2=R9ZU$U4V}0pn7??nZMpfrk@e1JWpxzrJ1uYdPnf+c*A3`E{cnU;CQpbeBI9k)9E8?i&XeR6vCf^oMF8J$nNn2j7^@`;5mKDy?u?17s;ynpKs@n)yuf{U99;xv0h?UB{5{rgBP7%qX#ORq^Pck`yA|vS)8oF$JD#cRvo%BZ+<e_dOqv;^5w;<boJfK&gTIU;aWw<iX6D4j5_eKmE$;Svn7mX8Zkg?1cfI$(}IB{%eDpAPY*;>pP-Wh8np8Eq{Sl@2jv4R-PNT9W3N<W82xA9sJ<aA@!A7|aBiaZsq(4-0kq&LzsH2zZcy*$BKBwbKn<{-33(_&Y8jSym45int!;drdruJ0%d<`5Vu*sAcjP2>A}->$EItDri(Lb)gtZah4B9y!r;dP5-JC*ZGu4$DK}ZrvZrb5$`P&D-g~jsJZ2i%%^+evtBRX|xt>Bo|=L4i$xw`i#^DZia~>$2M!k!%CUx0k|bo_e1PjrR&Br$XCgHf(w<ykz9wzQd!&ZhIgiqav{hCy#w_1gX%kzbGi}p5kP_j0;6T<Qv)o?Pe;XL5Dx_hPkN2e+$(I_4;#iG+}>xGI>_15cXyCKo7m~!SUve?NFoQkgJ?oE*fe_R!CEXFF$AZBki!&Y)a*dVqtYf=m1wTvVN;2Nv3n#m1%kH|4}q+MJuMc$rLLKw#v_=nq=p8FrML78wKou-;ZGjQoB8C|M%W!dlg>37G7J_*II5X1?An_1f*hlplUPW>9!LY7<|=2nqNSEd|0?bhCY{@`W!I8Va3E(QfnxUHLuIOCa}*$Kp7|ufmL7bhJ$JIH?(GQ_`BAz0dBS0h^Xls3gief9n>I<^vN_h7fuBqiA^-)Q%v7|IlQ7h*<1w#|cuRe%Hf=r?T0o)kX=3xugiJOn^#?J!mLD_HMj|02@g)V>Cz)y&W%1g=uuPHMr&ooqjT;1yDpiIhZCnj_@yCtCntf&x-n+1!X3I4Eg5yiPQ2j*IkV*cicJRaua5g*b$b(6PkP-+tTm7Z#$8=~0oH$vN>rX(s+>mC1%ZdA2<ybKGML>y~THazSe^~x(@H;>EsWXqR1|Tc>Plcgo=DQvXUIdp8V9M+O!rC;w?;o|6ew+u0R#^z8@!#`1nyCqSg(I$8O+tJ%XamRC8K%DRE=w<Aiov6#4~&UIjP(Fe0I~;QE7Ls9op!<$kJ>L)SF8|N5#bXHFp^N}G$<QSqH>B3(sb4ce`xuzIH2I?0ru{u;@KXs``6<DVZ0s^Gw|cm-#5GV_DjIdgDlKgHIhOW3|^~(oWTPz_J43oZq!O1KHPEEo)FE&7|RN$U`KZUKjeGMPs3NB>}8k8bc-mqOgRgv86G2#S!ht@`OuDC%R7UVksvs-`Q;9=JU)kLE%B_bBk!Xv&l{*9@okB0Vbg2-RgGk`ypYwvasp>AKms3v4UXl>9hsU(qf|pwP>3(xa&Xwthai?qHPQ~KCpIxrnvfy#7Yw^gBd1bT0bPiBffs!5=qgh4j>sa0Q4?8pdfdqiS0{4Re$O-W*}?kB`<lMlfag6-7f)-1gVh8)Jk{CK9{M1$@a1deW||&8k)G(90U3}YYmW;;hGk)H%sF9>(x7nZGg+ZH10rjH@-t_KpTO$Ci$~3w4|V1axzY}F5a(Zs#yJ*87;oMo?6Ag*c+2xhEIt}1)6+^st_4KY&B!|sFxes47g}rmh020dU)1)bnz4{r4^7}17&Jt>_jj=NL>70NMBqWtOzrWcp&N-R%NtB0EfJ0t>7V;&DayreFbW>`r!;4L+!A$o@_js}QzP#^o)NB*Gk<%Ik8sd^#~8g}jHdmjqInwMlt%Tol1X0P)cGoq(u%<u5!wGh!vEqI8+ojIfTw{r=ub#2e5ETMgE*+%zgGO?#*ZZqXKFLm`(eBmz8tr30>hmKb8clouSJ?x2F7fp&wnB(w*}{%{^Xi_Wx6&!hCT70c));%vgt95Ocj~|AWgwgC<na)8!zoVDRZm*LgnxJhJM93f&4FEW=zF8aM4TuTKUE^2K{8NezU7HH~!DcflILLICQ-1mLbsk;O2590JTxEjOdI?7+9V|vNcC3^XCs@8YIGsu~P<xKm;vBV8uk>3WdIfH)5(g2A5De7~lAS!tZVlsK_pnIhaC?WEAIK^mpzzk0t@v_L4@WVDpT9p$Tp2bx<QiDv&vk#|9HSj(4U%7>qUTNxC?=2wqTg=Z=J~b><rnZ@q+BxX!RX9314z5VV6Cfru8y@d{^2#NjtM5jyej#oiIB;ZE(aR1UsF!q#lHwk?-WktKg5@?&*-C81?POcSYor6Z@g$#jz8O!F|Wuso=l>&4_}u%EPeq-?$$%Zo*KGwRY}qj{4c6-;2UrnOrrOatx3!}jyGa0&L45e51#(BXv|qYPDK)~6GArCtKQ2I{v~-H>TQyL?kMp{#cXf+Gg7YZ}gk-j8_a0&m&|Yh7Y~t+Re*q`#w2j;fEx_cR$b)5ULlUn7(zJWm%WSp#)^B1fhxdH&<u>K5E-<WCIJR4FutR6L$>#zS-h{4~-Hmt_V+Uabe{n|RN4pUh`*nm=sDpoQl@zEl0t7{y)S2P;jBn9RTy9(Pr8`sRH2ACXXOJRc=_(@>NCZsll-%Kw4@TnRSO^dKuk8juE3V;1DW_9Afto~yRlHwI9^k9#8jt9(QX<88i;+0&I*(A5%eJ&#16{IFCx+()D(M%MS8iANwEdj?~?a`3Pth}#hwZ<kdQSB?Bwpnjn;18O>%vgsX6nDI8v$RdrX%$R0`g0#pl*8FuC-cuTX;4+Wq$b4Q}n))Xq%=Pl^j|@m~At!7H@syFg?;MvgCM~h1;GFZN#-zNx##8OZ6mG=lRpS|nA(8O%JuvCikteDyzcD$Xnud_+FE*?UHA9ot+A<&a0|GiLjn$&JMSF~68*C)9t{UuP#LMUcDMTKULE7VxKz7B>)qT<OvB4DO=GV%ROb`KguKmk+0(O2<tl3-<>eo)DQ67P*x5)p<ZWy)u7b+87%j+z`P?3le4+-YuNcEjOZ+v=yZKp*0k8HfckrYtM^0rZbT)NSd%D1`gP4L<imtE|kx@WxeGNtTkW%)$*NAty^iNj$Av2y->0R6MIPrvo(3Kgv%*>FBP>8A$?mQ>eGKOg&z1GqkUJTYibOZ0feyO`y#yKE@{<C^FZ5;Pnzof5R_EG@HPAiGvP!Dh**cAyzHQUh0pB{_qAgxhi5;;poeb!&MH_XE5K_|*WC>?_5;ZK#5vuhH@W0;8C5bOWJBc=d0qttkQV8W+<{Z6F$Y>IU}g8`<PS_+!K@+M{vIeyxh1DGTxS!mLdS0AoDj)d88-+WBKgk|`g+m1f#o3e3#)1V+hT^*Vug>7GoMs^zyP3&X;;R8$RCvY5LIZd9Sg;_C3V=l=Z03APO1QomL{pYb1aS^|NcZt#M+nS{8YD`C5)y?<u7$S7&b)*t5UW`gWs<u!l*r~1{$<_o8Y+~4WBpSWW?g_7y-%*kVO*y_(r-Ibd|@1C?NfC}?E(hmWNO@&u%9r+mMnr={ds`Vi`GA{zx`Qh;KoIC9a#rR>Vfqov3$yzpt^*vQ_TbAM2P?lvl8RApX@wX4F+vmPkjz>OfWHj4gxDwgh7O0VCf@fY2RTCbWmrwJ<5{)ao<Kuxk??C;87en(>Ft9|Bj=MgbV1yIZ=lX?Af{aL!9~!q#m55unYy*`^dk@0r)eKNW@NcjKkCqQI)Dqj{8wK&aiL+ujO=`PH7ZlzQdj~Kdl!meCG2hsBb0p~0WXd?R&|m7zRK5D_VSD1#pP5wban3dKt=Lx5nY=9704GUfVfhLao+O&N;lqUQ5<Tn_gWWI~*yF1D4V88FatFiS|2Rn#*b9dDO-p>`=K2ZAbiK*ohG(vs6VMF8na4IC4IMNZj?^B<n`ooJG8J4yxM+~F+v6G7#9D787I}{37&vdljt#%#2O~w&|5x$eW=R*4Fv9^Ku3}b9N+T7Y3p4dK(^TK#5rqnS-A;b39G**6=>lFMuu_f`#4xzBGWB0=k88Lz2TZ1{GcwG>`~VLxjQPJG>D|x6f?y$@;p_d`*Jk_<=B>Yu(|sPN2m!?_Qr-Xy6ebuvv^GZOGU{;UpD&h&07TYQVep^zOT}UTlwi4(5qP*vrdd?unZzLi2>Zjb4tTd3q!WuTkYX#7cOHO@9^jOL+grw)f8tA}-#A4Jbl{b@)sr7>WI|~rrJ)I}bw-M#qxaz9+Zmb=FwF9TS1oFQy)6C2es~T$!{y`|D<P4`w_|3~Jg|#dA{VbC`0jVg76MxmT?!pBPjh@1RG5ayFO+^ZEx<oqOK?b;ejCL@y@qCJY1@k!8Mo9Thf(b3pOkxQmB2v4&ZU=QVp-<XC~U$$_v?>Kg>L0KGLv6T-<jUTPb_9VPk-Q#3iokca8g7oZ`o#&-O})|pV$mb`EQDc!@u!_arr)v3CHF7S~=b}`rs4j#B%IJG#8UU=yx!`=S=ye;U9~^C);A^cB-2B%C>CNG{*x=V;ii8(|0ZhdwsEn9#w<9lZSRwUVA*fqp3ae#TiW-$v12P9GVfD_=x3*qi37q5xl75jR7H^r9J>Acowa1zQeIF+YdLvPs;W6oE9RrX0-Z9Z^|<-j=KYgC%GlRZXVd~_w@P5D7mMTo&nNKNljVtoQLhr4RP#OR{^Zkr39-VrBoGL0S73I00M_=scY8l_nLdl`}5d}#Y^tjBOnyi)4<y^fs!R}TaQiDls%Ss1ItPB41;TJs;&j+zVEM<A4|&>cc@yx$B4ibr2T~{uVprvk?FR6=jV?Uf%WmASZ`0M)NI=E8vS}2b2QDS#SuI*l2!@H+Fo=y`5t*!?c{;%0>AI^*dw%yM-G5yR)<K!x*TOfk(dTt+ftfSGEc4zHn=%b*gYK14}rjKV*QiD`bVGJV<DqwI`0bZ-|E-OJ&|vZ2WP&Sd4aDv65}B`|B3HK+^#Ao-%IL=&-+=qKc-WNWP9ji%#>uPuM_XKLt6=#&`-+0b)3!f+?hWsSJ_kmNLu+qM$L!zOlt_lZ`IlvEUe(z2@p18`y4uZzE+OZscK6ekJkfB*yesxti}k~=1*S8Lvm;Rq!`hTV`MT(aA9tGOW|Jmp2Xe`1}gPSpTiLS4kdiVf_*~PB<kiX-&UkO=S}&ow(yg3XN`QDP>q1Y5kZs=HB6g%&+)W9VzA_E<v(q|d5jtTTEN*$%4f5N!$r+v=B!y?D*j2^)t_f8fxZBS@HuloD&KfJW9&#HwfteY0Qy0#D`70e!Kz>SnAg{cXu$5|m&)G+s*g`NJc=xaEIjd)@J8JFZ)u^mh^`eIZkhnhG#HMxQ~Xi7cD`6{X33g2c<a86LJyU<eKoslhF>lJVf){n8`oO;*UEK)6CY>(*2Gc#xHLmo8P1k*x(Qcb4<Fn&35dRjYX(V%vopI<I--*rkwf4B&P`nbuZbe*^`u}nlfwIXc!>B(>R2--bn-tRkxN<1)0g;(!aF>F`OAO(H(+IdH2"


def _map_lines():
    """Static, bundled coastline/country/state linework (no network requests)."""
    import base64
    import zlib
    return json.loads(zlib.decompress(base64.b85decode(MAP_GEOMETRY_B85)).decode("utf-8"))


def _merc_lat(lat):
    return math.degrees(math.log(math.tan(math.pi / 4 + math.radians(max(-78, min(78, float(lat)))) / 2)))


def _map_projector(points, width=1120, height=385):
    """True Mercator projection of supplied IATA coordinates, fitted to route."""
    lons = [float(p["lon"]) for p in points]
    ys = [_merc_lat(p["lat"]) for p in points]
    cx, cy = (min(lons) + max(lons)) / 2, (min(ys) + max(ys)) / 2
    span_x = max((max(lons) - min(lons)) * 1.60, 7.5)
    span_y = max((max(ys) - min(ys)) * 1.70, span_x * (height / width) * 1.17)
    span_x = max(span_x, span_y * width / height * .92)

    def project(lon, lat):
        x = (float(lon) - cx) / span_x * width + width / 2
        y = (cy - _merc_lat(lat)) / span_y * height + height / 2
        return x, y

    return project, (cx, cy, span_x, span_y)


def _offline_svg_route(airports, width=1120, height=385):
    """Geographic vector illustration; exact airport coordinates, no tile/API calls."""
    project, limits = _map_projector(airports, width, height)
    cx, cy, sx, sy = limits
    svg = [
        '<rect width="1120" height="385" fill="#F4F5F2"/>',
        '<defs><linearGradient id="mapwash" x1="0" y1="0" x2="1" y2="1"><stop stop-color="#E8EFEC"/><stop offset="1" stop-color="#F8F3EB"/></linearGradient>',
        '<linearGradient id="routeink" x1="0" y1="0" x2="1" y2="0"><stop stop-color="#277C79"/><stop offset="1" stop-color="#CF855E"/></linearGradient>',
        '<filter id="planesheen"><feDropShadow dx="0" dy="3" stdDeviation="3" flood-color="#253737" flood-opacity="0.23"/></filter>',
        '<clipPath id="mapclip"><rect x="0" y="0" width="1120" height="385"/></clipPath></defs>',
        '<rect width="1120" height="385" fill="url(#mapwash)"/>',
        '<g clip-path="url(#mapclip)">',
    ]
    # Geography from bundled vector boundaries; no external images or watermarks.
    raw = _map_lines()
    for typ, color, line_width, alpha in [
        ("coast", "#6D9C96", 1.8, .66),
        ("borders", "#A8B9AF", 1.25, .64),
        ("states", "#B7C7BF", .75, .61),
    ]:
        pieces = []
        for segment in raw[typ]:
            if all(not ((cx - sx * .72) <= lon <= (cx + sx * .72) and
                        (cy - sy * .72) <= _merc_lat(lat) <= (cy + sy * .72))
                   for lon, lat in segment):
                continue
            coords = [project(lon, lat) for lon, lat in segment]
            if len(coords) > 1:
                pieces.append("M " + " L ".join(f"{x:.1f} {y:.1f}" for x, y in coords))
        if pieces:
            svg.append(f'<path d="{" ".join(pieces)}" fill="none" stroke="{color}" stroke-width="{line_width}" opacity="{alpha}"/>')
    # An atlas-style graticule is more legible than empty canvas when zoomed in.
    for lon in range(-170, -49, 5):
        x, _ = project(lon, 30)
        if 0 <= x <= width:
            svg.append(f'<path d="M {x:.1f} 0 V {height}" stroke="#DBE4DC" opacity=".42" stroke-width=".7"/>')
    for lat in range(10, 76, 5):
        _, y = project(-100, lat)
        if 0 <= y <= height:
            svg.append(f'<path d="M 0 {y:.1f} H {width}" stroke="#DBE4DC" opacity=".38" stroke-width=".7"/>')
    coordinates = [project(p["lon"], p["lat"]) for p in airports]
    segments = []
    for (x1, y1), (x2, y2) in zip(coordinates, coordinates[1:]):
        dx, dy = x2 - x1, y2 - y1
        offset = -min(45.0, max(12.0, math.hypot(dx, dy) * .055))
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2 + offset
        segments.append(f"M {x1:.1f},{y1:.1f} Q {mx:.1f},{my:.1f} {x2:.1f},{y2:.1f}")
    full_path = " ".join(segments)
    svg.append(f'<path id="flightpath" d="{full_path}" fill="none" stroke="#FFFFFF" stroke-width="9" stroke-linecap="round" opacity=".85"/>')
    svg.append(f'<path class="animated-route" d="{full_path}" fill="none" stroke="url(#routeink)" stroke-width="4" stroke-linecap="round" pathLength="100"/>')
    for k, ((x, y), place) in enumerate(zip(coordinates, airports)):
        name = html.escape(str(place["code"]))
        label = "ORIGEN" if k == 0 else ("DESTINO" if k == len(airports) - 1 else f"ESCALA {k}")
        svg.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="9" fill="white" stroke="#297A76" stroke-width="3"/>')
        svg.append(f'<rect x="{x - 31:.1f}" y="{y - 49:.1f}" width="62" height="30" rx="9" fill="#FFFFFF" fill-opacity=".92" stroke="#CFD9D2" stroke-width="1"/>')
        svg.append(f'<text x="{x:.1f}" y="{y - 26:.1f}" text-anchor="middle" fill="#223137" font-size="21" font-weight="800">{name}</text>')
        svg.append(f'<text x="{x:.1f}" y="{y + 30:.1f}" text-anchor="middle" fill="#6B8084" font-size="10" font-weight="800" letter-spacing="1.5">{label}</text>')
    svg.append('<g filter="url(#planesheen)" fill="#CC7855" stroke="#FFF" stroke-width="1.4"><path d="M 19 0 L -17 -10 L -11 -1 L -23 7 L -23 12 L 0 5 L -11 16 L -5 17 Z"/>')
    svg.append('<animateMotion dur="4.1s" begin=".3s" fill="freeze" rotate="auto" path="' + full_path + '" /></g>')
    svg.append('</g>')
    return "".join(svg)


def _route_map_html(airports, result, layout):
    """Self-contained browser animation; results appear after plane arrives."""
    from html import escape
    from math import ceil
    names = " · ".join(escape(str(p["code"])) for p in airports)
    svg = _offline_svg_route(airports)
    result_cards = [
        ("DEMORA ESPERADA", f'{result["mean_delay"]:.1f} min', "contra llegada programada"),
        ("ESCENARIO P90", f'{result["p90_delay"]:.0f} min', "9 de cada 10 escenarios ≤ este valor"),
        ("RIESGO DE CONEXIÓN", f'{100 * result["miss_any"]:.1f}%', "al menos una conexión perdida"),
        ("COSTO GENERALIZADO", f'${result["generalized"]:,.0f}', "MXN por pasajero · supuesto"),
    ]
    cells = "".join(
        '<div class="scenario-stat"><span>' + escape(n) + '</span><b>' + escape(v) + '</b><small>' + escape(s) + '</small></div>'
        for n, v, s in result_cards
    )
    return f"""<!DOCTYPE html><html lang="es"><head><meta charset="utf-8"/><style>
    *{{box-sizing:border-box}}body{{margin:0;font-family:Inter,Arial,sans-serif;background:transparent;color:#253339}}
    .travel{{border:1px solid #DFE1D9;border-radius:28px;overflow:hidden;background:linear-gradient(135deg,#FBFBF8,#F4F0E8);box-shadow:0 21px 45px rgba(58,60,54,.06)}}
    .top{{padding:17px 26px;display:flex;justify-content:space-between;gap:12px;align-items:center;border-bottom:1px solid #E6E3DD;background:#FFFDFC;}}
    .title{{font-size:12px;font-weight:800;letter-spacing:1.4px;color:#496C70}}.route{{font-size:12px;color:#B36F49;font-weight:800;letter-spacing:.04em}}
    .view{{width:100%;height:385px;display:block}}
    .animated-route{{stroke-dasharray:100;stroke-dashoffset:100;animation:draw 4.1s cubic-bezier(.24,.66,.3,1) .3s forwards}}
    .post{{padding:22px 23px 24px;opacity:0;transform:translateY(14px);animation:reveal .55s ease 4.5s forwards}}
    .stats{{display:grid;grid-template-columns:repeat(4,1fr);gap:11px}}
    .scenario-stat{{background:white;border-radius:18px;border:1px solid #E5E2DC;padding:16px 17px;min-width:0;display:flex;flex-direction:column;gap:9px}}
    .scenario-stat span{{font-size:10px;color:#80898C;letter-spacing:1.05px;font-weight:700}}
    .scenario-stat b{{font-size:clamp(16px,2.1vw,25px);font-weight:800;color:#273439;white-space:nowrap;letter-spacing:-.8px}}
    .scenario-stat small{{font-size:10px;color:#8A8F90;line-height:1.3}}
    .meta{{font-size:11px;color:#77858A;margin-top:13px}}
    @keyframes draw{{to{{stroke-dashoffset:0}}}}@keyframes reveal{{to{{opacity:1;transform:translateY(0)}}}}
    @media(max-width:760px){{.stats{{grid-template-columns:repeat(2,1fr)}}.view{{height:300px}}.scenario-stat b{{font-size:18px}}}}
    @media(prefers-reduced-motion:reduce){{.animated-route,.post{{animation:none;stroke-dashoffset:0;opacity:1;transform:none}}}}
    </style></head><body><div class="travel"><div class="top"><div class="title">NETWORK SCENARIO · MONTE CARLO</div><div class="route">{names}</div></div>
    <svg class="view" viewBox="0 0 1120 385" preserveAspectRatio="xMidYMid meet" role="img" aria-label="Itinerario animado: {names}">{svg}</svg>
    <div class="post"><div class="stats">{cells}</div><div class="meta">Mapa vectorial local. El avión representa una animación, no un vuelo en tiempo real. Cálculos basados en demoras históricas y parámetros editados.</div></div></div></body></html>"""


def empirical_delay_samples(wid, origin, dest, airline, month, hour, min_usable=40, limit=1800):
    """Historical departures for a route, with explicit progressive fallback.

    No black-box model inference and no synthetic flight rows; each draw is drawn
    from observed dep_delay in 2024–25, capped at 300 min for stability.
    """
    a = safe_dimension(origin, {origin})
    b = safe_dimension(dest, {dest})
    al = safe_dimension(airline, {airline}) if airline != "Todas" else None
    root = f"origin='{a}' AND dest='{b}' AND dep_delay IS NOT NULL"
    conditions = [
        ("Ruta + aerolínea + mes + hora", root + (f" AND airline='{al}'" if al else "") + f" AND month={int(month)} AND departure_hour={int(hour)}"),
        ("Ruta + aerolínea + mes", root + (f" AND airline='{al}'" if al else "") + f" AND month={int(month)}"),
        ("Ruta + aerolínea", root + (f" AND airline='{al}'" if al else "")),
        ("Ruta completa", root),
    ]
    # Remove duplicate SQL conditions when airline = Todas.
    used = set()
    last = None
    for label, condition in conditions:
        if condition in used:
            continue
        used.add(condition)
        sql_text = f"""SELECT dep_delay FROM {FLIGHTS} WHERE {condition}
            ORDER BY xxhash64(COALESCE(CAST(flight_date AS STRING),''),
                               COALESCE(CAST(tail_number AS STRING),''),
                               COALESCE(CAST(scheduled_departure_utc AS STRING),''))
            LIMIT {int(limit)}"""
        frame = query(sql_text, wid)
        values = pd.to_numeric(frame["dep_delay"], errors="coerce").dropna().to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if len(values) >= min_usable or label == "Ruta completa":
            if not len(values):
                raise ValueError(f"La ruta {origin}–{dest} no tiene duraciones de retraso utilizables")
            return np.clip(values, 0.0, 300.0), label
        last = label
    raise ValueError(f"No hay observaciones utilizables para {origin}–{dest} ({last})")


def monte_carlo_itinerary(legs, delay_pools, simulations=6000, seed=2026,
                           min_connections=None, rebook_wait=180,
                           dependence=0.0, fare=3500.0, value_hour=200.0,
                           meal=200.0, transfer=150.0, lodging=0.0,
                           fee_rebook=0.0):
    """Simulate passenger arrival and missed onward connections.

    legs: [{flight_minutes:..., layover_after:...}, ...]
    pools: observed positive departure delays, one array per leg.
    Arrivals use scheduled flight duration as entered by the user; no observed
    arrival-delay information or real seat/rebooking availability is assumed.
    """
    if not 1 <= len(legs) <= 4 or len(legs) != len(delay_pools):
        raise ValueError("El itinerario debe contener entre 1 y 4 tramos con datos para cada uno")
    if not 100 <= int(simulations) <= 50000:
        raise ValueError("Número de simulaciones fuera de rango")
    n = int(simulations)
    k = len(legs)
    if min_connections is None:
        min_connections = [45] * (k - 1)
    if len(min_connections) != k - 1:
        raise ValueError("Se requiere un tiempo mínimo por cada escala")
    if not 0 <= dependence <= 1:
        raise ValueError("La dependencia debe estar entre 0 y 1")
    if any(float(x) < 0 for x in [rebook_wait, fare, value_hour, meal, transfer, lodging, fee_rebook]):
        raise ValueError("Los costos y el tiempo de reacomodo no pueden ser negativos")
    sched_dep = np.zeros(k, dtype=float)
    for j in range(k):
        duration = float(legs[j]["flight_minutes"])
        if not 20 <= duration <= 1200:
            raise ValueError("Duración de vuelo programada fuera de rango")
        if j < k - 1:
            layover = float(legs[j]["layover_after"])
            if layover < float(min_connections[j]):
                raise ValueError("La escala programada es menor que el mínimo de conexión")
            sched_dep[j + 1] = sched_dep[j] + duration + layover
    planned_arrival = sched_dep[-1] + float(legs[-1]["flight_minutes"])
    rng = np.random.default_rng(int(seed))
    common_quantile = rng.uniform(0, 1, size=n)
    observed = np.zeros((n, k), dtype=float)
    for j, source in enumerate(delay_pools):
        pool = np.sort(np.asarray(source, dtype=float))
        pool = pool[np.isfinite(pool)]
        if not len(pool):
            raise ValueError(f"No hay observaciones de demoras en el tramo {j+1}")
        if (pool < 0).any():
            raise ValueError("Los retrasos de entrada deben ser no negativos")
        independent_u = rng.uniform(0, 1, size=n)
        choose_shared = rng.random(n) < float(dependence)
        u = np.where(choose_shared, common_quantile, independent_u)
        indices = np.minimum((u * len(pool)).astype(int), len(pool) - 1)
        observed[:, j] = pool[indices]
    miss_counts = np.zeros(n, dtype=np.int32)
    prev_arrival = sched_dep[0] + observed[:, 0] + float(legs[0]["flight_minutes"])
    for j in range(1, k):
        scheduled_next_departure = sched_dep[j] + observed[:, j]
        connection_deadline = prev_arrival + float(min_connections[j - 1])
        missed = connection_deadline > scheduled_next_departure
        miss_counts += missed.astype(np.int32)
        # Illustrative replacement: after the earliest eligible time, wait
        # the user-specified rebooking interval. Not an airline schedule.
        departure = np.where(
            missed,
            np.maximum(scheduled_next_departure, connection_deadline) + float(rebook_wait),
            scheduled_next_departure,
        )
        prev_arrival = departure + float(legs[j]["flight_minutes"])
    total_delay = np.maximum(0.0, prev_arrival - planned_arrival)
    miss_any = miss_counts > 0
    meal_events = total_delay >= 60
    transfer_events = miss_any
    hotel_events = total_delay >= 360
    outlays = (float(fare) + meal_events * float(meal) +
               transfer_events * float(transfer) + hotel_events * float(lodging) +
               miss_counts * float(fee_rebook))
    time_costs = total_delay / 60 * float(value_hour)
    result = {
        "n": n,
        "mean_delay": float(np.mean(total_delay)),
        "median_delay": float(np.median(total_delay)),
        "p90_delay": float(np.quantile(total_delay, .90)),
        "p95_delay": float(np.quantile(total_delay, .95)),
        "p_arrival_late15": float(np.mean(total_delay >= 15)),
        "p_arrival_late60": float(np.mean(total_delay >= 60)),
        "p_arrival_late180": float(np.mean(total_delay >= 180)),
        "miss_any": float(np.mean(miss_any)),
        "mean_misses": float(np.mean(miss_counts)),
        "mean_outlay": float(np.mean(outlays)),
        "mean_time_cost": float(np.mean(time_costs)),
        "generalized": float(np.mean(outlays + time_costs)),
        "planned_minutes": float(planned_arrival),
        "total_delays": total_delay,
        "misses": miss_counts,
    }
    return result


def network_simulator_page(wid, years, airlines):
    intro(
        "Network Scenario Studio",
        "Evalúa itinerarios con conexiones, incertidumbre histórica de salidas, alternativas de reacomodo y exposición económica por pasajero.",
        "07 / Network decision support · Monte Carlo",
    )
    st.markdown("""<div class="analysis-note"><b>Alcance del análisis</b><br>
    Los retrasos de salida provienen de vuelos históricos reales (2024–2025). Los tiempos entre vuelos,
    disponibilidad de conexión, espera de reacomodo y gastos son parámetros del escenario. No se usan
    llegadas reales ni vuelos actuales; esto es una simulación de sensibilidad, no una predicción validada
    de retraso de llegada ni de conexiones perdidas.</div>""", unsafe_allow_html=True)
    story_header("01 / Network selection", "Traza el itinerario", "Elige rutas existentes y enlaza hasta cuatro tramos. Las escalas deben tener servicio de salida en nuestro subconjunto histórico.")
    try:
        catalog = route_catalog_sql_cache(wid)
    except Exception as exc:
        st.error("No fue posible consultar las rutas históricas.")
        with st.expander("Detalle técnico"): st.code(str(exc))
        return
    if catalog.empty:
        st.warning("No hay rutas disponibles para simular.")
        return
    geo = airport_reference()
    catalog = catalog[
        catalog.origin.isin(geo) & catalog.dest.isin(geo)
    ].copy()
    catalog["origin"] = catalog.origin.astype(str)
    catalog["dest"] = catalog.dest.astype(str)
    cat_n = {(str(row.origin), str(row.dest)): int(row.flights) for row in catalog.itertuples()}
    adjacency = {}
    for a, b in cat_n:
        adjacency.setdefault(a, set()).add(b)
    if not adjacency:
        st.warning("El catálogo no tiene rutas con coordenadas IATA válidas.")
        return
    n_legs = st.segmented_control("Número de tramos", options=[1, 2, 3, 4], default=2,
                                  format_func=lambda x: "Directo" if x == 1 else f"{x-1} escala" + ("s" if x > 2 else ""),
                                  key="v6_legs")
    n_legs = int(n_legs or 2)

    def can_finish(start, depth, visited):
        if depth <= 0:
            return True
        return any(nxt not in visited and can_finish(nxt, depth - 1, visited | {nxt})
                   for nxt in adjacency.get(start, []))

    first_choices = sorted(x for x in adjacency if can_finish(x, n_legs, {x}))
    if not first_choices:
        st.info("No encontramos una cadena de rutas históricas suficientemente conectada para esa cantidad de tramos.")
        return
    start = st.selectbox("Aeropuerto de salida", first_choices, format_func=geo_label,
                         index=first_choices.index("ATL") if "ATL" in first_choices else 0, key="v6_initial")
    segments = []
    visited = {start}
    current = start
    for j in range(n_legs):
        possible = sorted(v for v in adjacency.get(current, [])
                          if v not in visited and can_finish(v, n_legs - j - 1, visited | {v}))
        if not possible:
            st.warning(f"No hay conexiones históricas posteriores a {current}. Elige otro aeropuerto anterior.")
            return
        with st.container(border=True):
            st.markdown(f"**TRAMO {j+1}** &nbsp; · &nbsp; {current}")
            c_dest, c_airline, c_flight = st.columns([1.5, 1.25, 1], gap="medium")
            with c_dest:
                dest = st.selectbox(f"Destino del tramo {j+1}", possible,
                                    format_func=geo_label, key=f"v6_dest_{j}")
            n_history = cat_n.get((current, dest), 0)
            try:
                al_frame = query(f"SELECT DISTINCT airline FROM {FLIGHTS} WHERE origin='{safe_dimension(current,{current})}' AND dest='{safe_dimension(dest,{dest})}' ORDER BY airline", wid)
                candidate_airlines = sorted(v for v in al_frame.airline.dropna().astype(str).tolist() if v in airlines)
            except Exception:
                candidate_airlines = []
            with c_airline:
                al = st.selectbox(f"Aerolínea del tramo {j+1}", ["Todas"] + candidate_airlines,
                                  key=f"v6_airline_{j}")
            with c_flight:
                flight_minutes = st.number_input(f"Vuelo programado (min) · {j+1}", min_value=20,
                                                  max_value=1200, value=120, step=15, key=f"v6_duration_{j}")
            c_time, c_n = st.columns([1, 2])
            with c_time:
                hour = st.selectbox(f"Hora de salida de referencia · {j+1}", list(range(24)),
                                    index=min(23, 8+3*j), format_func=lambda h:f"{h:02d}:00",key=f"v6_hour_{j}")
            with c_n:
                st.caption(f"{n_history:,} vuelos históricos de {current} → {dest} en Gold, antes de filtros de aerolínea/temporada")
            layover, minimum = None, None
            if j < n_legs - 1:
                lcol, mcol = st.columns(2)
                with lcol:
                    layover = st.number_input(f"Escala programada en {dest} (min)",
                                              min_value=30,max_value=720,value=90,step=15,key=f"v6_layover_{j}")
                with mcol:
                    minimum = st.number_input(f"Conexión mínima requerida (min) · {dest}",
                                              min_value=15,max_value=360,value=45,step=5,key=f"v6_minconn_{j}")
            segments.append({"origin":current,"dest":dest,"airline":al,
                             "hour":int(hour),"flight_minutes":int(flight_minutes),
                             "layover_after":float(layover or 0),"min_connection":int(minimum or 0),
                             "historical_count":n_history})
        visited.add(dest)
        current = dest
    if any(s["layover_after"] < s["min_connection"] for s in segments[:-1]):
        st.error("El tiempo programado de una escala no puede ser inferior al mínimo de conexión. Ajusta los valores.")
        return
    story_header("02 / Empirical conditions", "Condiciones y costos", "Las probabilidades se obtienen de demoras reales de salida; los gastos de viaje y reglas de reacomodo los defines tú.")
    a, b, c, d = st.columns(4)
    with a: month = st.selectbox("Mes de referencia histórico", list(range(1, 13)), index=5, key="v6_month")
    with b: trials = st.select_slider("Escenarios Monte Carlo", options=[1000, 3000, 6000, 12000, 20000], value=6000, key="v6_trials")
    with c: rebook_wait = st.number_input("Espera de reacomodo (min)", min_value=0, max_value=1440, value=180, step=30, key="v6_rebook")
    with d: dependence = st.select_slider("Dependencia hipotética entre tramos", options=[0.0,.25,.50,.75,1.0],value=0.0,key="v6_dep")
    st.caption("Dependencia 0: muestreos independientes. Valores superiores: algunos tramos comparten percentiles de demora, como análisis de sensibilidad; no es una correlación calibrada con el Notebook 06.")
    st.markdown("**Parámetros económicos · MXN por pasajero**")
    p1, p2, p3 = st.columns(3)
    with p1: fare = st.number_input("Boleto total del itinerario", min_value=0, max_value=200000, value=3500, step=250, key="v6_fare")
    with p2: value_hour = st.number_input("Valor de 1 hora de tiempo", min_value=0, max_value=20000, value=200, step=50, key="v6_value_hour")
    with p3: meal = st.number_input("Comida extra si atraso ≥60 min", min_value=0, max_value=20000, value=200, step=50, key="v6_meal")
    p4, p5, p6 = st.columns(3)
    with p4: transfer = st.number_input("Transporte extra si pierde conexión", min_value=0, max_value=20000, value=150, step=50, key="v6_transfer")
    with p5: hotel = st.number_input("Hospedaje si demora ≥6 h", min_value=0, max_value=50000, value=0, step=100, key="v6_hotel")
    with p6: fee = st.number_input("Cargo de reacomodo supuesto por conexión", min_value=0, max_value=20000, value=0, step=50, key="v6_fee")
    st.caption("El valor del tiempo es costo de oportunidad, no desembolso. Los gastos por reacomodo no implican que la aerolínea realmente los cobraría; son supuestos editables.")
    # Input fingerprint: never show the output of a previous scenario after inputs change.
    signature = json.dumps({
        "segments": segments, "month": month, "trials": trials,
        "rebook_wait": rebook_wait, "dependence": dependence, "fare": fare,
        "value_hour": value_hour, "meal": meal, "transfer": transfer,
        "hotel": hotel, "fee": fee,
    }, sort_keys=True)
    submitted = st.button("Ejecutar análisis de itinerario", type="primary", use_container_width=True, key="v6_run")
    if submitted:
        pools = []
        details = []
        try:
            with st.spinner("Consultando históricos de los tramos y calculando escenarios..."):
                for leg in segments:
                    pool, level = empirical_delay_samples(wid, leg["origin"], leg["dest"],
                                                          leg["airline"], month, leg["hour"])
                    pools.append(pool)
                    details.append({
                        "Ruta":f'{leg["origin"]} → {leg["dest"]}',
                        "Aerolínea":leg["airline"],
                        "Observaciones muestreadas":len(pool),
                        "Criterio utilizado":level,
                        "Retrasos ≥15 min (%)":100*float(np.mean(pool>=15)),
                        "Duración positiva media (min)":float(np.mean(pool)),
                    })
                result = monte_carlo_itinerary(
                    segments, pools, simulations=int(trials), seed=2026,
                    min_connections=[s["min_connection"] for s in segments[:-1]],
                    rebook_wait=rebook_wait, dependence=float(dependence),
                    fare=fare, value_hour=value_hour, meal=meal, transfer=transfer,
                    lodging=hotel, fee_rebook=fee,
                )
                st.session_state.v6_last_result = {
                    "signature":signature,
                    "segments":segments,"details":details,"result":result,
                    "month":int(month),"params":{"rebook_wait":int(rebook_wait),"dependence":float(dependence),
                    "fare":fare,"value_hour":value_hour,"meal":meal,"transfer":transfer,"hotel":hotel,"fee":fee},
                }
        except Exception as exc:
            st.error("No se pudo completar la simulación. Comprueba las rutas y filtros elegidos.")
            with st.expander("Detalle técnico"): st.code(str(exc))
            st.session_state.pop("v6_last_result",None)
    rec = st.session_state.get("v6_last_result")
    if not rec or rec.get("signature") != signature:
        st.info("Define los tramos y pulsa **Ejecutar análisis de itinerario** para calcular y visualizar el escenario actual. Si cambias un parámetro, vuelve a ejecutar el cálculo.")
        return
    result = rec["result"]
    steps = rec["segments"]
    airport_codes = [steps[0]["origin"]] + [leg["dest"] for leg in steps]
    places = []
    for code in airport_codes:
        info = geo.get(code)
        if info is None:
            st.warning(f"Sin coordenadas verificadas para {code}.")
            return
        places.append({"code":code,"lat":float(info["lat"]),"lon":float(info["lon"])})
    story_header("03 / Animated itinerary", "Trayecto y escenarios resultantes", "El avión recorre la ruta planificada. Al llegar, se muestran los resultados agregados de Monte Carlo.")
    components.html(_route_map_html(places,result,steps),height=665,scrolling=False)
    st.markdown("**Distribución de demora estimada sobre la llegada programada**")
    x = result["total_delays"]
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=x, nbinsx=35, marker_color="#417B79", opacity=.90,
                               hovertemplate="Demora: %{x:.0f} min<br>Simulaciones: %{y:,}<extra></extra>"))
    for value, label, col in [(result["median_delay"],"P50", TEAL),
                              (result["p90_delay"],"P90", ORANGE)]:
        fig.add_vline(x=value,line_width=2,line_dash="dash",line_color=col,
                      annotation_text=f"{label} · {value:.0f} min",annotation_position="top right")
    fig.update_layout(showlegend=False,bargap=.04)
    fig.update_xaxes(title_text="Minutos de demora relativa al horario planificado")
    fig.update_yaxes(title_text="Escenarios")
    st.plotly_chart(better_plot(fig,365),use_container_width=True)
    stat_a, stat_b, stat_c, stat_d = st.columns(4)
    with stat_a: st.metric("Llegadas con 15+ min",f'{100*result["p_arrival_late15"]:.1f}%')
    with stat_b: st.metric("Llegadas con 60+ min",f'{100*result["p_arrival_late60"]:.1f}%')
    with stat_c: st.metric("Llegadas con 180+ min",f'{100*result["p_arrival_late180"]:.1f}%')
    with stat_d: st.metric("Conexiones perdidas (al menos una)",f'{100*result["miss_any"]:.1f}%')
    story_header("04 / Economic breakdown", "Costo esperado por pasajero", "Separa desembolsos simulados y el valor del tiempo perdido. Ninguna cifra es una tarifa consultada.")
    costs = pd.DataFrame([
        ("Boleto del itinerario (supuesto)", float(rec["params"]["fare"])),
        ("Gastos contingentes esperados", result["mean_outlay"] - float(rec["params"]["fare"])),
        ("Valor económico esperado del tiempo", result["mean_time_cost"]),
        ("Costo total generalizado", result["generalized"]),
    ],columns=["Concepto","Costo medio (MXN)"])
    table_view(costs,formats={"Costo medio (MXN)":"$%,.2f"},height=208)
    st.caption("Comida: si demora ≥60 min; transporte: si pierde conexión; alojamiento: si demora ≥360 min; cargo de reacomodo: por conexión perdida. El costo del tiempo no es desembolso.")
    story_header("05 / Auditability", "Datos que utilizó la simulación", "Muestra observada por ruta, con relajación documentada de filtros cuando faltan vuelos suficientes.")
    sources = pd.DataFrame(rec["details"])
    table_view(sources, formats={"Observaciones muestreadas":"%,.0f",
                                 "Retrasos ≥15 min (%)":"%.2f%%",
                                 "Duración positiva media (min)":"%.2f"},height=260)
    st.caption(f"{result['n']:,} escenarios, semilla fija 2026, fuente Gold 2024–2025. Muestreo histórico por tramo limitado a 300 min. No hay datos de llegada, ventanas de conexiones reales, tarifas ni disponibilidad real de vuelos de reemplazo.")
    exported = pd.DataFrame({
        "simulated_arrival_delay_min":result["total_delays"],
        "missed_connections":result["misses"],
    })
    st.download_button("Exportar resultados Monte Carlo (CSV)",
                       data=exported.to_csv(index=False).encode("utf-8"),
                       file_name="flight_delay_monte_carlo_scenarios.csv",
                       mime="text/csv",use_container_width=True)



with st.sidebar:
    st.markdown(
        '<div class="brand-lockup">'
        '<div class="brand-mark"><span class="brand-symbol">✈</span> FLIGHT / DELAY</div>'
        '<div class="brand-caption">AVIATION DECISION INTELLIGENCE &nbsp; / &nbsp; 2024—25</div>'
        '</div><div class="side-divider"></div>',
        unsafe_allow_html=True,
    )
    choice = st.radio(
        "Secciones",
        ["Overview", "Airport Map", "Operations", "Predictive Models",
         "Propagation", "Financial Impact", "Network Simulator", "Operational Priorities", "Data Reliability"],
        label_visibility="collapsed",
    )
    st.markdown('<div class="side-divider"></div>', unsafe_allow_html=True)
    wid = warehouse_id()
    if not wid:
        st.error("SQL Warehouse sin configurar")
        st.caption("Conecta sql-warehouse en Databricks Apps → Settings → Resources.")
    else:
        st.markdown(
            '<div class="side-status"><b><span class="side-dot"></span>Conexión SQL configurada</b>'
            '<small>UNITY CATALOG · CONSULTAS HISTÓRICAS</small></div>',
            unsafe_allow_html=True,
        )
    if st.button("↻  Actualizar consultas", use_container_width=True):
        _query_as_user.clear()
        st.rerun()

if not wid:
    intro("Conexión pendiente", "La aplicación está instalada, pero requiere vincular un SQL Warehouse para consultar las tablas Gold en vivo.", "Configuration · Databricks Apps")
    st.markdown("**Configuración necesaria**")
    st.code("Settings → Resources → Add resource → SQL warehouse → Can use\nResource key: sql-warehouse",language=None)
    st.code("env:\n  - name: DATABRICKS_WAREHOUSE_ID\n    valueFrom: sql-warehouse",language="yaml")
    st.info("No coloques tokens ni contraseñas en el código. La conexión usa credenciales administradas por Databricks Apps.")
    st.stop()

years, airlines, origins=source_filters(wid)
with st.sidebar:
    st.markdown("**FILTROS DE RED**")
    year=st.selectbox("Año",["Todos"]+years,index=0)
    airline=st.selectbox("Aerolínea",["Todas"]+airlines,index=0)
    origin=st.selectbox("Aeropuerto de origen",["Todos"]+origins,index=0)
where=where_clause(year,airline,origin,years,airlines,origins)

if choice=="Overview": page_overview(wid,where)
elif choice=="Airport Map": page_airport_map(wid,where)
elif choice=="Operations": page_operations(wid,where)
elif choice=="Predictive Models": page_models(wid)
elif choice=="Propagation": page_propagation(wid)
elif choice=="Financial Impact": page_financial(wid,where)
elif choice=="Network Simulator": network_simulator_page(wid,years,airlines)
elif choice=="Operational Priorities": page_priorities(wid,where)
else: page_reliability(wid)

st.markdown("<div class='footer'>Flight Delay Intelligence · Historical flight operations · 2024–2025 · Selected U.S. origins · Financial figures are scenario estimates · Databricks Gold</div>",unsafe_allow_html=True)
