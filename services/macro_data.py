import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def get_mock_macro_features(dates):
    np.random.seed(42)

    # Ensure it's a proper list of timestamps
    if not isinstance(dates, (pd.Index, list, np.ndarray)):
        try:
            dates = list(dates)
        except Exception as e:
            print(f"[Error] Invalid dates passed to get_mock_macro_features: {e}")
            dates = pd.date_range(start='2020-01-01', periods=100)  # fallback

    dates = pd.Index(dates)

    return pd.DataFrame({
        'GDP_Growth': np.random.normal(2, 0.5, len(dates)),
        'Inflation': np.random.normal(2.5, 0.2, len(dates)),
        'Interest_Rate': np.random.normal(1.5, 0.3, len(dates))
    }, index=dates)
