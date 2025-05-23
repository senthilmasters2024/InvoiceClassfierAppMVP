@echo off
SET OPENAI_API_KEY=sk-your-api-key-here
CD /D %~dp0
streamlit run app\main.py
PAUSE
