import pandas as pd
import numpy as np
import re

def extract_hits_signal(chunk):
    """
    Extracts high-value signal from the heavy 'hits' column without fully
    parsing the massive nested JSON dictionaries.

    Features created:
      hits_interaction_count      : total number of hit interactions in session
      hits_max_ecommerce_action   : highest funnel step reached
                                    (0=Unknown, 1=Click, 2=View, 3=Add to Cart,
                                     5=Checkout, 6=Purchase)
    """
    if 'hits' not in chunk.columns:
        return chunk

    # Count hit interactions
    chunk['hits_interaction_count'] = chunk['hits'].apply(
        lambda x: x.count("'hitNumber':") if isinstance(x, str) else 0
    )

    # Extract maximum eCommerceAction type reached using regex for speed
    def get_max_action_type(hits_str):
        if not isinstance(hits_str, str):
            return 0
        actions = re.findall(r"'action_type': '(\d+)'", hits_str)
        if not actions:
            return 0
        return max(int(a) for a in actions)

    chunk['hits_max_ecommerce_action'] = chunk['hits'].apply(get_max_action_type)

    # Drop the heavy column now that we've extracted what we need
    return chunk.drop(columns=['hits'])

def engineer_temporal_features(chunk):
    """
    Extracts micro-temporal patterns from the visitStartTime UNIX timestamp.
    """
    if 'visitStartTime' not in chunk.columns:
        return chunk

    visit_dt = pd.to_datetime(chunk['visitStartTime'], unit='s')

    chunk['visit_hour']         = visit_dt.dt.hour
    chunk['visit_weekday']      = visit_dt.dt.dayofweek
    chunk['is_weekend']         = visit_dt.dt.dayofweek.isin([5, 6]).astype(int)
    chunk['is_business_hours']  = visit_dt.dt.hour.between(9, 17).astype(int)

    return chunk

def smart_impute(chunk):
    """
    Conditional missing-value fills based on session behaviour signals.
    """
    # bounces NaN ≡ did not bounce
    if 'totals.bounces' in chunk.columns:
        chunk['totals.bounces'] = chunk['totals.bounces'].fillna(0)

    # pageviews: fallback chain
    if 'totals.pageviews' in chunk.columns:
        if 'totals.hits' in chunk.columns:
            chunk['totals.pageviews'] = (
                chunk['totals.pageviews']
                .fillna(chunk['totals.hits'])
                .fillna(1)
            )
        else:
            chunk['totals.pageviews'] = chunk['totals.pageviews'].fillna(1)

    # timeOnSite: conditional on bounce status
    if 'totals.timeOnSite' in chunk.columns and 'totals.bounces' in chunk.columns:
        bounced_mask = chunk['totals.bounces'] == 1
        chunk.loc[bounced_mask & chunk['totals.timeOnSite'].isna(),
                  'totals.timeOnSite'] = 0
        # Non-bounced sessions with missing time → use approximate median (~100 s)
        chunk['totals.timeOnSite'] = chunk['totals.timeOnSite'].fillna(100)
    elif 'totals.timeOnSite' in chunk.columns:
        chunk['totals.timeOnSite'] = chunk['totals.timeOnSite'].fillna(0)

    # Remaining categorical fills
    fill_unknown = [
        'trafficSource.keyword', 'trafficSource.referralPath',
        'trafficSource.adContent',
        'trafficSource.adwordsClickInfo.adNetworkType',
        'trafficSource.adwordsClickInfo.slot'
    ]
    for col in fill_unknown:
        if col in chunk.columns:
            chunk[col] = chunk[col].fillna('(not set)')

    # Other numeric fills
    fill_zero = ['totals.newVisits', 'totals.transactions',
                 'totals.sessionQualityDim']
    for col in fill_zero:
        if col in chunk.columns:
            chunk[col] = chunk[col].fillna(0)

    return chunk
