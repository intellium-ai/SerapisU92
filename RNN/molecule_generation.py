import streamlit as st
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from launcher_of_clm import valid_generate
from launcher_of_sm import score
import os
import json

# load credentials and api keys
with open('Streamlit_app/env.json','r+') as f:
    env_vars = json.load(f)
credentials = env_vars['credentials']



def add_logo(logo_path, width, height):
    """Read and return a resized logo"""
    logo = Image.open(logo_path)
    modified_logo = logo.resize((width, height))
    return modified_logo

# set session states so they persist
if not hasattr(st.session_state, 'logged_in'):
    st.session_state.logged_in = False

# Set page title
st.set_page_config(page_title="BoostPrep", page_icon=Image.open('Streamlit_app/Logomark-RGB-vector-purple.ico'), layout='wide')

# Hide 'made with streamlit' footer - make it say 'Intellium AI Ltd'
hide_footer_style = """
<style>footer {
	
	visibility: hidden;
	
	}
footer:after {
	content:'Intellium AI Ltd'; 
	visibility: visible;
	display: block;
	position: relative;
	#background-color: red;
	padding: 5px;
	top: 2px;
}
</style>   
"""
st.markdown(hide_footer_style, unsafe_allow_html=True)

def main():
    # login screen
    login_left, login_mid, login_right = st.columns([1, 1, 1], gap="small")
    image = login_mid.image(add_logo(logo_path="Streamlit_app/Horizontal-RGB-200-purple.png", height=43, width=200))
    login_container = login_mid.empty()

    if not st.session_state.logged_in:
    
        with login_container.form("login"):
            
            username = st.text_input("Username:", placeholder="Enter user name...")
            password = st.text_input("Password:", placeholder="Enter password...", type="password")
            login_button = st.form_submit_button("Login")

            if login_button:
                if username in credentials:
                    if credentials[username] == password:
                        st.session_state.logged_in = True
                    else:
                        st.warning('Wrong Password!', icon="⚠️")
                else:
                    st.warning('Wrong username!', icon="⚠️")

    # app screen
    if st.session_state.logged_in:

        # clear login screen
        image.empty()
        login_container.empty()
        main_left, main_mid, main_right = st.columns([1, 3, 1], gap="small")
        with main_mid:
            st.image(add_logo(logo_path="Streamlit_app/Horizontal-RGB-200-purple.png", height=43, width=200))
            csv_source()

def generate_scored_mol(number_of_molecules: int):
    valid_generate(number_of_molecules, 0, 'Streamlit_app/Dm.csv', 'Streamlit_app/ft_pretrained_100k.pth', 'Streamlit_app/generated.csv', None)
    score('Streamlit_app/Dm.csv', 'Streamlit_app/generated.csv', 'Streamlit_app/reg_50_pretrained.pth', f'Streamlit_app/generated_scored_{number_of_molecules}.csv', 0, 0, 11)
    return f'Streamlit_app/generated_scored_{number_of_molecules}.csv'

def upload_csv():
    # Display file uploader widget
    csv_file = st.file_uploader("Upload a CSV file", type=["csv"])

    # Check if file is uploaded or not
    if csv_file is not None:
        # Read the dataset from the uploaded file
        df = pd.read_csv(csv_file)
        await_input(df)
        csv_file = None
    else:
        if 'df' in st.session_state:
            del st.session_state.df
            del st.session_state.old_df

# Create a function to upload CSV file
def generate_csv():
    num_mol = st.number_input("Enter the number of molecules to be generated:")
    if num_mol:
        st.write("Generating molecules...")
        csv_file = generate_scored_mol(num_mol)
        if csv_file is not None:
            df = pd.read_csv(csv_file)
            await_input(df)
            csv_file = None
        else:
            if 'df' in st.session_state:
                del st.session_state.df
                del st.session_state.old_df

def csv_source():
    data_source = st.sidebar.selectbox("Choose Data Source", ("Upload CSV File", "Generate molecules"))
    if data_source == 'Upload CSV File':
        upload_csv()
    else:
        generate_csv()

def plot_graph(sa_threshold, dv_threshold,df):
    # Create a Matplotlib figure
    fig, ax = plt.subplots()

    highlight = df[df.D > dv_threshold][df.SA < sa_threshold]
    
    # Plot the data points
    ax.scatter(df['D'], df['SA'], marker='.', alpha=0.8)
    ax.scatter(highlight['D'], highlight['SA'], marker='.', alpha=1, c='gold')
    
    ax.set_xlabel('Detonation Velocity')
    ax.set_ylabel('SA Score')

    # Display the Matplotlib figure in Streamlit
    st.pyplot(fig)

def await_input(df):
    if not hasattr(st.session_state, 'container'):
        st.session_state.container = st.container()
    if not hasattr(st.session_state, 'text_input'):
        st.session_state.text_input = None
    if not hasattr(st.session_state, 'old_df'):
        st.session_state.old_df = df.copy()
        #print(st.session_state.old_df)
        st.session_state.container = st.session_state.old_df
    if not hasattr(st.session_state, 'df'):
        st.session_state.df = st.session_state.old_df.copy()
    if not hasattr(st.session_state, 'reset'):
        st.session_state.reset = None
    with st.form(key='thresholds'):
        st.session_state.dv_threshold = st.number_input('Please enter the lowest allowed detonation velocity threshold:')
        st.session_state.sa_threshold = st.number_input('Please enter the highest allowed SA score threshold:')
        submit_button = st.form_submit_button('Submit')
    # Check if both thresholds have been submitted
    if submit_button:
        st.session_state.submitted = True
        submitted()
        st.session_state.container = st.session_state.df
    st.write(st.session_state.container)
    st.session_state.reset = st.button('Reset Transforms', on_click=reset)
def reset():
    st.session_state.df = pd.DataFrame()
    st.session_state.df = st.session_state.old_df.copy()
    st.session_state.container = st.container()
    st.session_state.container = st.session_state.df
def submitted():
    dv_threshold = st.session_state.dv_threshold
    sa_threshold = st.session_state.sa_threshold
    st.session_state.df = st.session_state.df[(st.session_state.df['D'] > dv_threshold) & (st.session_state.df['SA'] < sa_threshold)]
    plot_graph(sa_threshold = sa_threshold,dv_threshold = dv_threshold,df = st.session_state.df)
if __name__ == '__main__':
    main()