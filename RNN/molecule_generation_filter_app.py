import streamlit as st
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from launcher_of_clm import valid_generate
from launcher_of_sm import score
import json

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
if not hasattr(st.session_state, "reset"):
    st.session_state.reset = None
if not hasattr(st.session_state, "num_molecules"):
    st.session_state.num_molecules = None


# Set page title
st.set_page_config(
    page_title="BoostPrep",
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
            data_source = st.selectbox(
                "Choose Data Source", ("Upload CSV File", "Generate molecules")
            )
            if data_source == "Upload CSV File":
                # Display file uploader widget
                csv_file = st.file_uploader("Upload a CSV file", type=["csv"])
                if csv_file is not None:
                    df = pd.read_csv(csv_file)
                    st.session_state.df = df
            else:
                st.session_state.num_molecules = st.number_input(
                    "Enter the number of molecules to be generated:"
                )

                st.button("Generate molecules", on_click=generate_scored_mol)

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
                        "Filter Detonation Velocity",
                        st.session_state.df["D"].min(),
                        st.session_state.df["D"].max(),
                        (
                            st.session_state.df["D"].min(),
                            st.session_state.df["D"].max(),
                        ),
                        key="jj",
                    )
                    sa_threshold = st.slider(
                        "Filter SA Score",
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

                    st.write(st.session_state.filtered_df)

                with mid_right:
                    plot_graph()

                    st.session_state.reset = st.button(
                        "Reset Transforms", on_click=reset
                    )


def generate_scored_mol():
    valid_generate(
        st.session_state.num_molecules,
        0,
        "streamlit_utils/data/Dm.csv",
        "streamlit_utils/models/ft_pretrained_100k.pth",
        "streamlit_utils/data/generated.csv",
        None,
    )
    score(
        "streamlit_utils/data/Dm.csv",
        "streamlit_utils/data/generated.csv",
        "streamlit_utils/models/reg_50_pretrained.pth",
        f"streamlit_utils/data/generated_scored_{st.session_state.num_molecules}.csv",
        0,
        0,
        11,
    )
    return f"streamlit_utils/data/generated_scored_{st.session_state.num_molecules}.csv"


def plot_graph():
    # Create a Matplotlib figure
    fig, ax = plt.subplots()

    # highlight = df[df.D > dv_threshold][df.SA < sa_threshold]

    # Plot the data points
    ax.scatter(
        st.session_state.df["D"], st.session_state.df["SA"], marker=".", alpha=0.8
    )
    ax.scatter(
        st.session_state.filtered_df["D"],
        st.session_state.filtered_df["SA"],
        marker=".",
        alpha=1,
        c="gold",
    )

    ax.set_xlabel("Detonation Velocity")
    ax.set_ylabel("SA Score")

    # Display the Matplotlib figure in Streamlit
    st.pyplot(fig)


def reset():
    st.session_state.df = pd.DataFrame()
    st.session_state.df = st.session_state.old_df.copy()
    st.session_state.container = st.container()
    st.session_state.container = st.session_state.df


if __name__ == "__main__":
    main()
