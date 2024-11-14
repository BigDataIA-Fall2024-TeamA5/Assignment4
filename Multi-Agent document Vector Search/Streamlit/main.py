import streamlit as st
import requests
import snowflake.connector
from dotenv import load_dotenv
import os
from PIL import Image
import io
import base64

# Load environment variables
load_dotenv()

# FastAPI endpoint URLs for user login and PDF list retrieval
FASTAPI_URL = os.getenv("FASTAPI_URL", "http://127.0.0.1:8000")
REGISTER_URL = f"{FASTAPI_URL}/auth/register"
LOGIN_URL = f"{FASTAPI_URL}/auth/login"

# Set up Streamlit page configuration with a wide layout
st.set_page_config(page_title="PDF Text Extraction Application", layout="wide")

# Initialize session state variables
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
if 'access_token' not in st.session_state:
    st.session_state['access_token'] = None
if 'pdf_data' not in st.session_state:
    st.session_state['pdf_data'] = []
if 'selected_pdf' not in st.session_state:
    st.session_state['selected_pdf'] = None
if 'view_mode' not in st.session_state:
    st.session_state['view_mode'] = 'list'  # default view is list
if 'page' not in st.session_state:
    st.session_state['page'] = 'main'

# Snowflake connection setup
def create_snowflake_connection():
    try:
        conn = snowflake.connector.connect(
            user=os.getenv("SNOWFLAKE_USER"),
            password=os.getenv("SNOWFLAKE_PASSWORD"),
            account=os.getenv("SNOWFLAKE_ACCOUNT"),
            warehouse=os.getenv("SNOWFLAKE_WAREHOUSE"),
            database=os.getenv("SNOWFLAKE_DATABASE"),
            schema=os.getenv("SNOWFLAKE_SCHEMA")
        )
        return conn
    except Exception as e:
        st.error(f"Error connecting to Snowflake: {e}")
        return None

# Function to fetch PDFs and their corresponding image links from Snowflake
def fetch_pdf_data_from_snowflake():
    conn = create_snowflake_connection()
    if not conn:
        return []
    
    cursor = conn.cursor()

    # Fetch title, image link, and PDF link from the Snowflake PUBLICATIONS table
    query = "SELECT title, brief_summary, image_link, pdf_link FROM PUBLIC.PUBLICATIONS"
    cursor.execute(query)
    result = cursor.fetchall()
    
    cursor.close()
    conn.close()

    return result  # Returns a list of tuples (title, brief_summary, image_link, pdf_link)

# Function to resize and get image as base64
def get_resized_image_base64(image_url, width=300):
    try:
        response = requests.get(image_url)
        img = Image.open(io.BytesIO(response.content))
        new_height = int(width * img.height / img.width)
        resized_img = img.resize((width, new_height))
        buffered = io.BytesIO()
        resized_img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()
    except Exception as e:
        st.error(f"Error loading image: {e}")
        return None

# Function to display PDF details on a new page
def show_pdf_details():
    pdf_data = st.session_state['selected_pdf']
    if pdf_data:
        pdf_name, brief_summary, image_link, pdf_link = pdf_data

        # Display the PDF title
        st.header(f"Details for {pdf_name}")

        # Display the resized image
        if image_link:
            img_base64 = get_resized_image_base64(image_link)
            if img_base64:
                st.image(f"data:image/png;base64,{img_base64}", caption=pdf_name, use_container_width=False)

        # Display the summary
        st.subheader("Extracted Summary")
        st.write(brief_summary)

        # Provide a link to open the PDF
        st.markdown(f"[Open PDF File]({pdf_link})", unsafe_allow_html=True)

        if st.button("Back to Main Page"):
            st.session_state['page'] = 'main'
            st.rerun()
    else:
        st.error("No PDF selected. Please go back and select a PDF.")
        if st.button("Back to Main Page"):
            st.session_state['page'] = 'main'
            st.rerun()

# Main Application
def main_app():
    # Custom CSS for orange buttons and hover effect
    st.markdown("""
        <style>
        .stButton button {
            background-color: orange;
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 5px;
            font-size: 14px;
            font-weight: bold;
        }
        .centered-title {
            font-size: 40px;
            font-weight: bold;
            text-align: center;
            border-bottom: 2px solid black;
            padding-bottom: 10px;
            margin-bottom: 30px;
        }
        .pdf-container {
            position: relative;
            display: inline-block;
            overflow: hidden;
        }
        .pdf-container img {
            transition: transform 0.3s ease-in-out;
        }
        .pdf-container:hover img {
            transform: scale(1.1);
        }
        .pdf-container .hover-text {
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background-color: rgba(0, 0, 0, 0.7);
            color: white;
            padding: 10px;
            opacity: 0;
            transition: opacity 0.3s ease-in-out;
            overflow: auto;
        }
        .pdf-container:hover .hover-text {
            opacity: 1;
        }
        </style>
    """, unsafe_allow_html=True)

    # Logout button on the upper-left corner
    if st.sidebar.button("Logout", help="Logout"):
        logout()

    st.markdown("<h1 class='centered-title'>Multi-Agent Document Vector Search Application</h1>", unsafe_allow_html=True)

    # View mode selector
    view_mode = st.radio("Select view mode", ["List View", "Grid View"], index=0 if st.session_state['view_mode'] == 'list' else 1)

    # Update session state based on view mode
    st.session_state['view_mode'] = 'list' if view_mode == "List View" else 'grid'

    # Fetch PDF data from Snowflake if not already fetched
    if not st.session_state['pdf_data']:
        st.session_state['pdf_data'] = fetch_pdf_data_from_snowflake()

    # Display PDFs based on selected view mode
    if st.session_state['view_mode'] == 'list':
        display_pdfs_list_view()
    else:
        display_pdfs_grid_view()

# Function to display PDFs in list view
def display_pdfs_list_view():
    st.subheader("PDF Files (List View)")
    for i, pdf_data in enumerate(st.session_state['pdf_data']):
        pdf_name, brief_summary, image_link, pdf_link = pdf_data
        
        # Create columns for image and title
        col1, col2 = st.columns([1, 3])
        
        with col1:
            img_base64 = get_resized_image_base64(image_link, width=150)
            if img_base64:
                st.markdown(f"""
                    <div class="pdf-container">
                        <img src="data:image/png;base64,{img_base64}" width="150">
                        <div class="hover-text">{brief_summary}</div>
                    </div>
                """, unsafe_allow_html=True)
        
        with col2:
            # Align the button vertically with the image
            st.markdown("<div style='display:flex;align-items:center;height:100%;'>", unsafe_allow_html=True)
            if st.button(f"{pdf_name}", key=f"list_{i}"):
                st.session_state['selected_pdf'] = pdf_data
                st.session_state['page'] = 'details'
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

# Function to display PDFs in grid view
def display_pdfs_grid_view():
    st.subheader("PDF Files (Grid View)")
    cols = st.columns(3)
    
    for i, pdf_data in enumerate(st.session_state['pdf_data']):
        pdf_name, brief_summary, image_link, pdf_link = pdf_data
        
        with cols[i % 3]:
            img_base64 = get_resized_image_base64(image_link, width=250)
            
            if img_base64:
                st.markdown(f"""
                    <div class="pdf-container">
                        <img src="data:image/png;base64,{img_base64}" width="250">
                        <div class="hover-text">{brief_summary}</div>
                    </div>
                """, unsafe_allow_html=True)
            
            # Center the button below the image
            st.markdown("<div style='text-align:center;'>", unsafe_allow_html=True)
            if st.button(f"{pdf_name}", key=f"grid_{i}"):
                st.session_state['selected_pdf'] = pdf_data
                st.session_state['page'] = 'details'
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

# Logout function
def logout():
    st.session_state['logged_in'] = False
    st.session_state['access_token'] = None
    st.session_state['page'] = 'main'
    st.rerun()

# Login Page
def login_page():
    st.header("Login / Signup")
    
    option = st.selectbox("Select Login or Signup", ("Login", "Signup"))
    
    if option == "Login":
        st.subheader("Login")
        
        username = st.text_input("Username")
        
        password = st.text_input("Password", type="password")
        
        if st.button("Login"):
            login(username, password)

    elif option == "Signup":
        
        st.subheader("Signup")
        
        username = st.text_input("Username")
        
        email = st.text_input("Email")
        
        password = st.text_input("Password", type="password")
        
        if st.button("Signup"):
            signup(username, email, password)

# Signup function
def signup(username, email, password):
    response = requests.post(REGISTER_URL, json={
        "username": username,
        "email": email,
        "password": password
    })
    if response.status_code == 200:
        st.success("Account created successfully! Please login.")
    else:
        st.error(f"Signup failed: {response.json().get('detail', 'Unknown error occurred')}")

# Login function 
def login(username, password):
    
   response = requests.post(LOGIN_URL, json={
      "username": username,
      "password": password 
   })
   
   if response.status_code == 200:
       token_data = response.json()
       st.session_state['access_token'] = token_data['access_token']
       st.session_state['logged_in'] = True 
       st.success("Logged in successfully!")
       st.rerun()
   else:
       st.error("Invalid username or password. Please try again.")

# Main Interface depending on login state 
if st.session_state['logged_in']:
   if st.session_state['page'] == 'details':
       show_pdf_details()
   else:
       main_app()
else:
   login_page()