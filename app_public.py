"""
F1 2026 Race Predictor: public dashboard, deployed to Streamlit Community Cloud.
Read-only. Same view as app.py, drawn by dashboard.py.

    streamlit run app_public.py
"""

import dashboard

dashboard.render(local=False)
