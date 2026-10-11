"""
F1 2026 Race Predictor: local dashboard.
Same view as the public dashboard, drawn by dashboard.py, plus a Run
Prediction panel in the sidebar.

    streamlit run app.py
"""

import dashboard

dashboard.render(local=True)
