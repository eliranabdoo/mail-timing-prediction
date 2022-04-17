from typing import Tuple

import pandas as pd

DELIVERED = "delivered"
SUCCESS_COL, PREV_EMAILS_COL, LABEL_COL = "success", "prev_emails", "label"
EMAIL_ID_COL, CONTACT_ID_COL, CAMPAIGN_ID_COL, COMPANY_ID_COL = (
    "email_id",
    "contact_id",
    "campaign_id",
    "company_id",
)
TIMESTAMP_COL, EVENT_COL = "timestamp", "event"
POSITION_COL, FUNCTION_COL, SENIORITY_COL = "position", "function", "seniority"
STATE_COL, COUNTRY_COL, SIZE_COL, INDUSTRY_COL = (
    "state",
    "country",
    "size",
    "linkedin_industry",
)
HOUR_COL = "hour"


def create_success_df(engagements_df, companies_df, contacts_df) -> pd.DataFrame:
    index_columns = [EMAIL_ID_COL, CONTACT_ID_COL, CAMPAIGN_ID_COL]
    engagements_df = engagements_df.sort_values(TIMESTAMP_COL)

    engaged_prospects = (
        engagements_df[engagements_df[EVENT_COL] != DELIVERED]
        .set_index(index_columns)
        .index.drop_duplicates()
    )

    df = (
        engagements_df[engagements_df[EVENT_COL] == DELIVERED]
        .merge(contacts_df, on=CONTACT_ID_COL, how="left")
        .merge(companies_df, on=COMPANY_ID_COL, how="left")
    )
    df = df.set_index(index_columns)
    df[SUCCESS_COL] = df.index.isin(engaged_prospects)
    df[PREV_EMAILS_COL] = df.groupby(index_columns).size() - 1
    df.timestamp = df.timestamp.apply(lambda t: pd.Timestamp(t))
    df = df.reset_index(drop=True)
    df = df.drop([EVENT_COL, COMPANY_ID_COL], axis=1)
    return df


def create_train_data(
    engagements_df, companies_df, contacts_df
) -> Tuple[pd.DataFrame, pd.Series]:
    success_df = create_success_df(engagements_df, companies_df, contacts_df)
    x = success_df.drop([SUCCESS_COL, TIMESTAMP_COL], axis=1)
    x[HOUR_COL] = success_df[TIMESTAMP_COL].apply(lambda t: str(t.hour))
    y = success_df[SUCCESS_COL].replace({True: 1, False: 0})
    return x, y
