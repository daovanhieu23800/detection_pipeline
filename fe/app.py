import streamlit as st
import numpy as np
import cv2
import base64
import requests
import pandas as pd
# Define API endpoints
IMAGE_API_URL = "http://flask-be:5000/detection"
TABLE_API_URL = "http://flask-be:5000/record"


def send_image_to_api(image_file):
    files = {"image": (image_file.name, image_file, image_file.type)}
    response = requests.post(IMAGE_API_URL, files=files)
    return response


def render_image_detection():
    st.title("Image Detection API Client")

    image_file = st.file_uploader(
        "Choose an image...", type=["jpg", "jpeg", "png"])

    if image_file is not None:

        # if st.button("Send Image to API"):
        #     with st.spinner("Sending image to the API..."):
        #         try:
        response = send_image_to_api(image_file)
        if response.status_code == 200:
            result = response.json()
            image_bytes = np.frombuffer(base64.b64decode(
                result['visualize_image']), dtype=np.uint8)
# Decode the image from the NumPy array.
            image_cv = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
            image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
            st.image(
                image_rgb, caption=f"Visualized Image, There are {result['n_person']} people")
            st.success("Image processed successfully!")
        else:
            st.error(
                f"API request failed with status code {response.status_code}")


def render_table_viewer():
    page_number = 1
    st.title("Table Viewer")
    input_page = st.number_input("Go to Page:", min_value=1)
    if st.button("Go"):
        page_number = input_page
    try:
        response = requests.get(TABLE_API_URL, params={
                                "page_number": page_number})
        if response.status_code == 200:
            data = response.json()
            df = pd.DataFrame(data)
            search_query = st.text_input("Search table", "")
            if search_query:
                df = df[df.apply(lambda row: row.astype(str).str.contains(
                    search_query, case=False, na=False).any(), axis=1)]
            st.dataframe(df)
        else:
            st.error(
                f"Failed to fetch table data. Status code: {response.status_code}")

    except Exception as e:
        st.error(f"Error occurred while fetching table data: {e}")


page = st.sidebar.selectbox("Select Page", ["Image Detection", "Table Viewer"])

if page == "Image Detection":
    render_image_detection()
elif page == "Table Viewer":
    render_table_viewer()
