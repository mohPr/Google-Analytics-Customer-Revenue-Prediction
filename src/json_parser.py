import json
import pandas as pd

def parse_json_col(series, col_name):
    """
    Parses a single JSON string column and returns a flat DataFrame.
    
    Args:
        series (pd.Series): The column containing JSON strings.
        col_name (str): The name of the column (used as prefix for new columns).
        
    Returns:
        pd.DataFrame: A flattened DataFrame with prefixed column names.
    """
    def safe_parse(x):
        try:
            return json.loads(x) if isinstance(x, str) else {}
        except Exception:
            return {}

    parsed = series.apply(safe_parse)
    flat   = pd.json_normalize(parsed.tolist())
    flat.columns = [f'{col_name}.{c}' for c in flat.columns]
    flat.index   = series.index
    return flat
