from pathlib import Path
import os
import streamlit as st
from observatorio_athena.common import Config
from observatorio_athena.aws import Athena
from observatorio_athena.ui import render_app
ROOT=Path(__file__).resolve().parent
config=Config.read(os.environ.get('OBS_CONFIG',str(ROOT/'config/ada.json')))
@st.cache_resource
def query_client(settings):
 return Athena(settings)
render_app(config,query_client(config))
