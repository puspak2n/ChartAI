# db_connect.py

import streamlit as st
import pandas as pd
from connectors.aws_rds_connector import connect_aws_rds
# from connectors.snowflake_connector import connect_snowflake
# from connectors.teradata_connector import connect_teradata
# from connectors.greenplum_connector import connect_greenplum

def connect_to_database_ui():
    st.subheader("🌐 Connect to Database")

    db_options = ["AWS RDS (Postgres/MySQL)"]
    db_choice = st.selectbox("Select Database Type:", db_options)

    with st.form("db_connection_form"):
        host = st.text_input("Host")
        port = st.text_input("Port", value="5432")
        user = st.text_input("User")
        password = st.text_input("Password", type="password")
        database = st.text_input("Database")
        custom_query = st.text_area("Custom Query (e.g., SELECT * FROM my_table LIMIT 100)")

        submit = st.form_submit_button("Connect and Load Data")

    if submit:
        if db_choice == "AWS RDS (Postgres/MySQL)":
            df = connect_aws_rds(host, port, user, password, database, custom_query)
            return df
        # Future: add Snowflake, Teradata, Greenplum connectors here
        else:
            st.error("Database type not yet supported.")

    return None
