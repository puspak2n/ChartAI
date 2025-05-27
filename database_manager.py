# database_manager.py

import streamlit as st
import pandas as pd
from connectors.aws_rds_connector import connect_aws_rds
#from connectors.snowflake_connector import connect_snowflake
from connectors.teradata_connector import connect_teradata
from connectors.greenplum_connector import connect_greenplum

def connect_to_database(db_type):
    st.sidebar.subheader(f"🔗 Enter {db_type} Credentials")
    host = st.text_input("Host", key=f"{db_type}_host")
    user = st.text_input("User", key=f"{db_type}_user")
    password = st.text_input("Password", type="password", key=f"{db_type}_password")
    database = st.text_input("Database", key=f"{db_type}_database")


    option = st.sidebar.radio("Fetch Mode", ["Select Table", "Custom Query"])
    if option == "Select Table":
        table_name = st.sidebar.text_input("Table Name")
        custom_query = f"SELECT * FROM {table_name}" if table_name else None
    else:
        custom_query = st.sidebar.text_area("Custom SQL Query")

    if st.sidebar.button("Connect Now"):
        try:
            if db_type == "AWS RDS":
                df = connect_aws_rds(host, port, user, password, database, custom_query)
           # elif db_type == "Snowflake":
            #    df = connect_snowflake(user, password, account=host, database=database, query=custom_query)
            elif db_type == "Teradata":
                df = connect_teradata(host, user, password, database, custom_query)
            elif db_type == "Greenplum":
                df = connect_greenplum(host, port, user, password, database, custom_query)
            else:
                st.error("Unsupported database selected.")
                return False, None

            st.success(f"Connected successfully to {db_type}!")
            return True, df

        except Exception as e:
            st.error(f"Connection failed: {str(e)}")
            return False, None

    return False, None
