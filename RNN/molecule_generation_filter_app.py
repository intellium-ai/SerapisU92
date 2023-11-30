import streamlit as st
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from launcher_of_clm import valid_generate,generate_novel_smiles
from launcher_of_sm import score
import json
import os
import rdkit
import base64
from rdkit import Chem
from rdkit.Chem import Draw

# load credentials and api keys
with open("streamlit_utils/env.json", "r+") as f:
    env_vars = json.load(f)
credentials = env_vars["credentials"]


def add_logo(logo_path, width, height):
    """Read and return a resized logo"""
    logo = Image.open(logo_path)
    modified_logo = logo.resize((width, height))
    return modified_logo


# set session states so they persist
if not hasattr(st.session_state, "logged_in"):
    st.session_state.logged_in = False
if not hasattr(st.session_state, "df"):
    st.session_state.df = None
if not hasattr(st.session_state, "filtered_df"):
    st.session_state.filtered_df = None

# Set page title
st.set_page_config(
    page_title="Energetic Materials",
    page_icon=Image.open("streamlit_utils/Logomark-RGB-vector-purple.ico"),
    layout="wide",
)

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
    image = login_mid.image(
        add_logo(
            logo_path="streamlit_utils/Horizontal-RGB-200-purple.png",
            height=43,
            width=200,
        )
    )
    login_container = login_mid.empty()

    if not st.session_state.logged_in:
        with login_container.form("login"):
            username = st.text_input("Username:", placeholder="Enter user name...")
            password = st.text_input(
                "Password:", placeholder="Enter password...", type="password"
            )
            login_button = st.form_submit_button("Login")

            if login_button:
                if username in credentials:
                    if credentials[username] == password:
                        st.session_state.logged_in = True
                    else:
                        st.warning("Wrong Password!", icon="⚠️")
                else:
                    st.warning("Wrong username!", icon="⚠️")

    # app screen
    if st.session_state.logged_in:
        with st.sidebar:
            # get data by uploading or generating
            mode = st.selectbox(
                "Data Selection:",
                ["Use Existing Data", "Generate New Molecules"],
                index=0,
            )

            if mode == "Use Existing Data":
                csv_file = st.selectbox(
                    "Select File:",
                    sorted(os.listdir("streamlit_utils/generated_scored_data")),
                    index=0,
                )

                st.session_state.df = pd.read_csv(
                    "streamlit_utils/generated_scored_data/" + csv_file
                )

            elif mode == "Generate New Molecules":
                num_molecules = st.number_input(
                    "Number of novel molecules to generate:", step=1, min_value=2
                )

                if st.button("Generate molecules"):
                    with st.spinner("Please wait, molecules are being generated. This may take some time if the number of desired novel molecules is large."):
                        csv_file = generate_scored_mol(num_molecules)

                    st.session_state.df = pd.read_csv(csv_file)
        download_csv('streamlit_utils/generated_scored_data/')
        # clear login screen
        image.empty()
        login_container.empty()
        main_left, main_mid, main_right = st.columns([1, 100, 1], gap="small")
        with main_mid:
            st.image(
                add_logo(
                    logo_path="streamlit_utils/Horizontal-RGB-200-purple.png",
                    height=43,
                    width=200,
                )
            )

            if st.session_state.df is not None:
                mid_left, mid_right = st.columns([1, 1], gap="medium")

                with mid_left:
                    # filter
                    dv_threshold = st.slider(
                        "Detonation Velocity (D)",
                        st.session_state.df["D"].min(),
                        st.session_state.df["D"].max(),
                        (
                            st.session_state.df["D"].min(),
                            st.session_state.df["D"].max(),
                        ),
                        key="jj",
                    )
                    sa_threshold = st.slider(
                        "SA Score (SA)",
                        st.session_state.df["SA"].min(),
                        st.session_state.df["SA"].max(),
                        (
                            st.session_state.df["SA"].min(),
                            st.session_state.df["SA"].max(),
                        ),
                        key="jlj",
                    )

                    st.session_state.filtered_df = st.session_state.df[
                        (st.session_state.df["D"] >= dv_threshold[0])
                        & (st.session_state.df["D"] <= dv_threshold[1])
                        & (st.session_state.df["SA"] >= sa_threshold[0])
                        & (st.session_state.df["SA"] <= sa_threshold[1])
                    ]

                    st.dataframe(
                        st.session_state.filtered_df, column_order=["D", "SA", "smiles"]
                    )

                with mid_right:
                    st.write(
                        len(st.session_state.filtered_df),
                        "out of",
                        len(st.session_state.df),
                        "datapoints selected",
                    )
                    plot_graph()

                    # show molecule
                    molecule_id = st.selectbox(
                        "Select Molecule Index to Visualise:",
                        list(st.session_state.filtered_df.index),
                        index=0,
                    )

                    if molecule_id is not None:
                        smiles = st.session_state.filtered_df["smiles"][molecule_id]
                        mol = Chem.MolFromSmiles(smiles)
                        img = Draw.MolToImage(mol)
                        st.image(img, width=200, caption=smiles)


def generate_scored_mol(num_molecules):
    generate_novel_smiles(
        num_molecules,
        0,
        "streamlit_utils/data/Dm.csv",
        "streamlit_utils/models/ft.pth",
        "streamlit_utils/data/generated_tmp.csv",
        None,
    )
    score(
        "streamlit_utils/data/Dm.csv",
        "streamlit_utils/data/generated_tmp.csv",
        "streamlit_utils/models/reg_50_pretrained.pth",
        f"streamlit_utils/generated_scored_data/generated_scored_{num_molecules}.csv",
        0,
        0,
        11,
    )
    return f"streamlit_utils/generated_scored_data/generated_scored_{num_molecules}.csv"


def plot_graph():
    # Create a Matplotlib figure
    fig, ax = plt.subplots()

    # Plot the data points
    ax.scatter(
        st.session_state.df["D"],
        st.session_state.df["SA"],
        marker=".",
        alpha=0.2,
        c="blue",
    )
    ax.scatter(
        st.session_state.filtered_df["D"],
        st.session_state.filtered_df["SA"],
        marker=".",
        alpha=1,
        c="blue",
    )

    blx = st.session_state.filtered_df["D"].min()
    bly = st.session_state.filtered_df["SA"].min()

    width = st.session_state.filtered_df["D"].max() - blx
    height = st.session_state.filtered_df["SA"].max() - bly

    ax.add_patch(
        Rectangle(
            xy=(blx, bly),
            width=width,
            height=height,
            linewidth=1,
            color="blue",
            fill=True,
            alpha=0.2,
        )
    )

    ax.set_xlabel("Detonation Velocity")
    ax.set_ylabel("SA Score")

    # Display the Matplotlib figure in Streamlit
    st.pyplot(fig)

def download_csv(csv_directory):

    def extract_data(file_path):
        with open(file_path, 'rb') as file:
            data = file.read()
        return data
    csv_files = [file for file in os.listdir(csv_directory) if file.endswith(".csv")]
    # Display a selectbox for file selection
    selected_file = st.selectbox("Select a CSV file to download", csv_files)
    #Add a download button
    print(os.path.join(csv_directory,selected_file))
    st.download_button(label = "Download CSV",data = extract_data(os.path.join(csv_directory,selected_file)), file_name=selected_file)

if __name__ == "__main__":
    main()
