import numpy as np
import pandas as pd

def safe_mode(series):
    """Returns the most frequent value in a series. Returns NaN if empty."""
    mode = series.mode()
    return mode.iloc[0] if len(mode) > 0 else np.nan

def build_rfm_features(df):
    """
    RFM — Recency, Frequency, Monetary
    The three pillars of customer value analysis.
    """
    reference_date = df['date'].max()

    rfm = df.groupby('fullVisitorId').agg(
        recency_days        = ('date', lambda x: (reference_date - x.max()).days),
        session_count       = ('date', 'count'),
        unique_days_visited = ('date', 'nunique'),
        total_revenue       = ('revenue', 'sum'),
        tenure_days         = ('date', lambda x: (x.max() - x.min()).days),
        first_visit_date    = ('date', 'min'),
        last_visit_date     = ('date', 'max'),
    ).reset_index()

    rfm['log_total_revenue']   = np.log1p(rfm['total_revenue'])
    rfm['avg_sessions_per_day'] = rfm['session_count'] / (rfm['tenure_days'] + 1)

    return rfm

def build_behavioral_features(df):
    """
    Session behavior aggregations per user.
    """
    cols = {}

    if 'totals.pageviews' in df.columns:
        cols['pageviews_sum']  = ('totals.pageviews', 'sum')
        cols['pageviews_mean'] = ('totals.pageviews', 'mean')
        cols['pageviews_max']  = ('totals.pageviews', 'max')

    if 'totals.hits' in df.columns:
        cols['hits_sum']  = ('totals.hits', 'sum')
        cols['hits_mean'] = ('totals.hits', 'mean')

    if 'totals.timeOnSite' in df.columns:
        cols['time_on_site_sum']  = ('totals.timeOnSite', 'sum')
        cols['time_on_site_mean'] = ('totals.timeOnSite', 'mean')

    if 'totals.bounces' in df.columns:
        cols['bounce_count'] = ('totals.bounces', 'sum')

    if 'hits_max_ecommerce_action' in df.columns:
        cols['funnel_max_action']       = ('hits_max_ecommerce_action', 'max')
        cols['add_to_cart_sessions']    = ('hits_max_ecommerce_action', lambda x: (x >= 3).sum())

    agg = df.groupby('fullVisitorId').agg(**cols).reset_index()
    
    # Ratios
    session_count = df.groupby('fullVisitorId').size().rename('_session_count')
    agg = agg.merge(session_count, on='fullVisitorId', how='left')
    
    if 'bounce_count' in agg.columns:
        agg['bounce_rate'] = agg['bounce_count'] / agg['_session_count']
    
    return agg.drop(columns=['_session_count'])

def build_categorical_features(df):
    """
    Most-used categorical attributes per user.
    """
    cat_cols = {
        'device.deviceCategory'    : 'most_used_device',
        'channelGrouping'          : 'most_used_channel',
        'geoNetwork.country'       : 'most_used_country',
        'device.browser'           : 'most_used_browser',
        'device.operatingSystem'   : 'most_used_os',
    }

    agg_dict = {alias: (col, safe_mode) for col, alias in cat_cols.items() if col in df.columns}
    cat = df.groupby('fullVisitorId').agg(**agg_dict).reset_index()

    return cat
