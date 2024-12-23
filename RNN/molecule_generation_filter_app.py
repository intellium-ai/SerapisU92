import streamlit as st
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from launcher_of_clm import generate_novel_smiles
from launcher_of_sm import score
import json
import os
from rdkit import Chem
from rdkit.Chem import Draw
from pathlib import Path

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
if not hasattr(st.session_state, "file_name"):
    st.session_state.csv_filename = None

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
    if not st.session_state.logged_in:
        login_left, login_mid, login_right = st.columns([1, 1, 1], gap="small")

        image = login_mid.image(
            add_logo(
                logo_path="streamlit_utils/Horizontal-RGB-200-purple.png",
                height=43,
                width=200,
            )
        )

        login_container = login_mid.empty()

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
        # try clearing login screen
        try:
            image.empty()
            login_container.empty()
        except:
            pass

        # sidebar
        with st.sidebar:
            # Select Data Source
            with st.expander("Data", expanded=True):
                mode = st.radio(
                    "Dataset source",
                    [
                        "Generate novel molecules",
                        "Use pre-generated molecules",
                    ],
                    index=0,
                    label_visibility="collapsed",
                )

                if mode == "Use pre-generated molecules":
                    st.session_state.csv_filename = st.selectbox(
                        "Selected source file:",
                        sorted(os.listdir("streamlit_utils/generated_scored_data")),
                        index=None,
                    )

                elif mode == "Generate novel molecules":
                    num_molecules = st.number_input(
                        "Molecule Number:", step=100, min_value=100, max_value=5000
                    )

                    st.session_state.csv_filename = (
                        f"generated_scored_{num_molecules}.csv"
                    )

                    # generate button

                    if st.button(
                        "Regenerate molecules"
                        if st.session_state.csv_filename
                        in os.listdir("streamlit_utils/generated_scored_data")
                        else "Generate molecules"
                    ):
                        with st.spinner(
                            "Generation may take some time if the number of desired novel molecules is large. Please wait..."
                        ):
                            generate_scored_mol(num_molecules)

            if st.session_state.csv_filename in os.listdir(
                "streamlit_utils/generated_scored_data"
            ):
                st.session_state.df = pd.read_csv(
                    "streamlit_utils/generated_scored_data/"
                    + st.session_state.csv_filename
                )

                # Apply filters
                with st.expander("Filters", expanded=True):
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
                        "Synthetic Availability Score (SA)",
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

                    filepath = Path("streamlit_utils/data/generated_filtered_tmp.csv")
                    filepath.parent.mkdir(parents=True, exist_ok=True)
                    st.session_state.filtered_df.to_csv(filepath)

        # logo
        image = st.image(
            add_logo(
                logo_path="streamlit_utils/Horizontal-RGB-200-purple.png",
                height=43,
                width=200,
            )
        )

        # title
        st.title("Novel Energetic Molecule Generation")

        st.write(
            "Energetic materials are generated using the approach outlined in the following paper:",
            "https://pubs.acs.org/doi/pdf/10.1021/acs.jcim.2c00997.",
        )

        # main
        if st.session_state.csv_filename in os.listdir(
            "streamlit_utils/generated_scored_data"
        ):
            st.write(
                len(st.session_state.filtered_df),
                "/",
                len(st.session_state.df),
                "datapoints selected from file: ",
                st.session_state.csv_filename,
            )

            with st.container():
                mid_left, mid_center, mid_right = st.columns([2, 2, 1], gap="medium")

                # data table
                with mid_left:
                    st.subheader("Data")

                    st.dataframe(
                        st.session_state.filtered_df,
                        column_order=["D", "SA", "smiles"],
                        height=300,
                    )
                    with open(
                        "streamlit_utils/data/generated_filtered_tmp.csv",
                        "rb",
                    ) as file:
                        data = file.read()

                        st.download_button(
                            label="Download Dataset",
                            data=data,
                            file_name="energetic_molecules.csv",
                        )

                # scatter
                with mid_center:
                    st.subheader("Scatter")
                    plot_graph(sa_threshold, dv_threshold)

                # draw molecule
                with mid_right:
                    st.subheader("Draw Molecule")
                    molecule_id = st.selectbox(
                        "Molecule index:",
                        list(st.session_state.filtered_df.index),
                        index=0,
                    )

                    if molecule_id is not None:
                        smiles = st.session_state.filtered_df["smiles"][molecule_id]
                        mol = Chem.MolFromSmiles(smiles)
                        img = Draw.MolToImage(mol)
                        st.image(img, width=300, caption=smiles, use_column_width=True)


def generate_scored_mol(num_molecules):
    generate_novel_smiles(
        num_novel=num_molecules,
        idx=0,
        data_path="streamlit_utils/data/Dm.csv",
        model_path="streamlit_utils/models/ft_pretrained_100k.pth",
        saving_path="streamlit_utils/data/generated_tmp.csv",
        tokens=None,
    )
    score(
        train_data_path="streamlit_utils/data/Dm.csv",
        data_path="streamlit_utils/data/generated_tmp.csv",
        model_path="streamlit_utils/models/reg_50_pretrained.pth",
        saving_path=f"streamlit_utils/generated_scored_data/generated_scored_{num_molecules}.csv",
        SMILE_index_1=0,
        SMILE_index_2=0,
        target_index=11,
    )


def plot_graph(sa_threshold, dv_threshold):
    # Create a Matplotlib figure
    fig, ax = plt.subplots()

    # Plot the data points
    ax.scatter(
        st.session_state.df["D"],
        st.session_state.df["SA"],
        marker=".",
        alpha=0.2,
        c="blue",
        edgecolors="none",
    )
    ax.scatter(
        st.session_state.filtered_df["D"],
        st.session_state.filtered_df["SA"],
        marker=".",
        alpha=0.8,
        c="blue",
        edgecolors="none",
    )

    blx = dv_threshold[0]
    bly = sa_threshold[0]
    width = dv_threshold[1] - blx
    height = sa_threshold[1] - bly

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

    ax.set_xlabel("Detonation Velocity (D)")
    ax.set_ylabel("Synthetic Availability Score (SA)")

    # Display the Matplotlib figure in Streamlit
    st.pyplot(fig)


if __name__ == "__main__":
    main()
