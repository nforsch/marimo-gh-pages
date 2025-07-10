import marimo

__generated_with = "0.14.10"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(
        rf"""
    # Load data
    - need to stop rest of code from running until data_path is set
    """
    )
    return


@app.cell
def _(find_dotenv, load_dotenv, mo, os):
    load_dotenv(find_dotenv(usecwd=True))

    data_path = None
    if not os.environ.get("DATA_PATH"):
        text = mo.ui.text(
            placeholder="/my/data/path", label="Enter data path:", full_width=True
        )
    else:
        data_path = os.environ["DATA_PATH"]
        text = mo.ui.text(
            placeholder=data_path,
            label="Enter data path:",
            full_width=True,
            value=data_path,
        )
    form = text.form(show_clear_button=True)
    return data_path, form


@app.cell
def _(data_path, form, mo):
    mo.vstack([form]) if data_path is None else mo.md("DATA_PATH found in .env file.")
    return


@app.cell
def _(data_path, form, mo):
    mo.stop(form.value is None and data_path is None, mo.md("**Fill in your data path in the form above to continue.**"))
    _message = None
    if data_path is not None:
        DATA_PATH = data_path
        _message = mo.md(f"Data path: {data_path}")
    elif data_path is None:
        DATA_PATH = form.value
        _message = mo.md(f"Data path: {form.value}")
    _message
    return (DATA_PATH,)


@app.cell
def _(mo):
    mo.md(r"""## Interactive dataframe""")
    return


@app.cell
def _(DATA_PATH, mo, pd):
    df = pd.read_pickle(DATA_PATH+"/df_ntnu.pkl")
    transformed_df = mo.ui.dataframe(df)
    transformed_df
    return (df,)


@app.cell
def _(df):
    # df_next = transformed_df.value
    df_next = df.copy()
    df_next = df_next[["curling_min_deg", "curling_difference_deg", "annular_opening_difference_mm", "annular_opening_ddt_max_mm/ms", "curling_angle_ddt_min_deg/ms", "posterior_tortuosity", "anterior_tortuosity", "basal_tortuosity", "posterior_transverse_displacement_mm", "anterior_transverse_displacement_mm", "basal_transverse_displacement_mm", "posterior_longitudinal_displacement_mm", "anterior_longitudinal_displacement_mm", "basal_longitudinal_displacement_mm", "MAD 0= no 1= yes", "Curling motion 0=no 1=yes", "Tracking 0=good 1=poor", "BSA", "Gender", "age_inclusion", "Weight", "Height", "BMI", "syst_BP", "diast_BP", "VT_or_ACA"]]
    df_next["Curling motion 0=no 1=yes"] = df_next["Curling motion 0=no 1=yes"].astype("bool", errors="raise")
    df_next["MAD 0= no 1= yes"] = df_next["MAD 0= no 1= yes"].astype("bool", errors="raise")
    df_next["Tracking 0=good 1=poor"] = df_next["Tracking 0=good 1=poor"].astype("bool", errors="raise")
    df_next["VT_or_ACA"] = df_next["VT_or_ACA"].astype("bool", errors="raise")
    df_next.index = df_next.index.astype(int)
    df_next
    return (df_next,)


@app.cell
def _(df_next):
    index_dtype = df_next.index.dtype
    print(f"The data type of the index is: {index_dtype}")
    return


@app.cell
def _(df_next, mo):
    var_A = mo.ui.dropdown(options=df_next.columns, searchable=True)
    first_row = mo.hstack([mo.md("Choose variable X:"), var_A], justify="start")
    var_B = mo.ui.dropdown(options=df_next.columns, searchable=True)
    second_row = mo.hstack([mo.md("Choose variable Y:"), var_B], justify="start")
    category = mo.ui.dropdown(options=[col for col,ty in zip(df_next.columns, df_next.dtypes) if ty=='bool'], value=None, searchable=True)
    third_row = mo.hstack([mo.md("(Optional) choose categorical variable:"), category], justify="start")
    button_on = mo.ui.run_button(label="Generate plot")
    button_off = mo.ui.run_button(label="Generate plot", disabled=True)

    plot_variables = mo.vstack([first_row, second_row, third_row], align="stretch")
    plot_variables
    return category, var_A, var_B


@app.cell
def _():
    # button_on if any(elem is not None for elem in [var_A.value, var_B.value]) else button_off
    return


@app.cell
def _(category, df_next, mo, px, var_A, var_B):

    # mo.stop(not button_on.value, mo.md("**Select variables for plotting and click `Generate plot`**"))
    mo.stop(all(elem is None for elem in [var_A.value, var_B.value]), mo.md("**Select variables for plotting**"))
    fig_data = px.scatter(
        df_next,
        x=var_A.value,
        y=var_B.value,
        color=category.value,
        marginal_x="violin",
        marginal_y="violin"
    )
    mo.ui.plotly(fig_data)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ---
    ## View CMR data
    """
    )
    return


@app.cell
def _(DATA_PATH, Path, df_next, mo):
    # cmr_video_path = Path(DATA_PATH + "MAD/data_processed/MAD_OUS_vids/")
    cmr_video_dir = mo.ui.file_browser(Path(DATA_PATH), selection_mode="directory", multiple=False)
    select_patient = mo.ui.dropdown(options=df_next.index)
    mo.vstack([cmr_video_dir, select_patient])
    return cmr_video_dir, select_patient


@app.cell
def _(cmr_video_dir, mo, select_patient):
    mo.stop(cmr_video_dir.value is None, "Nope")
    list_of_patients_from_dir = []
    for item in cmr_video_dir.path().iterdir():
        if item.is_dir():
            list_of_patients_from_dir.append(item.name)
    list_of_patients_from_dir

    if str(select_patient.value) in list_of_patients_from_dir:
        video_path = cmr_video_dir.path() / str(select_patient.value) / "cine/4ch"
        _video_list = list(video_path.glob("*.mp4"))
        if not _video_list:
            cmr4ch_video = mo.md(f"Folder of patient {select_patient.value} does not contain 4ch cine video (.mp4)")
        else:
            cmr4ch_video = mo.video(src=_video_list[0],
                                             muted=True,
                                             autoplay=True,
                                             loop=True,
                                             height=500,width=500
                                            )
        # cmr4ch_video = list(video_path.glob("*.gif"))[0]
    else:
        cmr4ch_video = mo.md(f"Folder of patient {select_patient.value} does not contain 4ch cine video (.mp4)")
    cmr4ch_video
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ---
    ## Control LV ES shapes
    """
    )
    return


@app.cell
def _(DATA_PATH, Path, np):
    points_path = Path(DATA_PATH + "controls/Aligned_models/")
    XYZ_control = []
    for _xyz_ii in sorted(points_path.glob(pattern="*.txt")):
        XYZ_control.append(np.loadtxt(_xyz_ii, delimiter=","))
    XYZ_control = np.array(XYZ_control)
    XYZ_control = np.reshape(XYZ_control, (XYZ_control.shape[0], XYZ_control.shape[1]*3), order="C")
    print(XYZ_control[1, :4])

    return (XYZ_control,)


@app.cell
def _(mo):
    mo.md(
        r"""
    ---
    ## Demo shape data
    """
    )
    return


@app.cell
def _(np):
    n_points = 200
    n_samples = 100
    XYZ_demo = []
    for _n in range(n_samples):
        _x = np.cos(np.random.standard_normal(n_points) - np.random.rand(1))
        _y = np.cos(np.random.standard_normal(n_points) - np.random.rand(1))
        _z = np.cos(np.random.standard_normal(n_points) - np.random.rand(1))
        _xyz = np.stack([_x, _y, _z]).flatten("F")
        XYZ_demo.append(_xyz)
    XYZ_demo = np.array(XYZ_demo)
    return XYZ_demo, n_points


@app.cell
def _(XYZ_demo, mo):
    mo.md(rf"""`Generated demo shape data with dimensions {XYZ_demo.shape}`""")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    Shape data has dimensions $N_{samples}$ x $3N_{points}$, where points are ordered as $$[x_1, y_1, z_1, x_2, y_2, z_2, ..., x_{200}, y_{200}, z_{200}]$$
    A 1D-array of points can be reshaped into a ($N_{points}$ x $3$) array with the following:
    ```python
    xyz_2d = xyz_1d.reshape((n_points, 3), order="C")
    ```
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ---
    # Run PCA on data
    """
    )
    return


@app.cell
def _(XYZ_control, XYZ_demo, mo):
    choice_of_XYZ = mo.ui.dropdown(
        options={"Control data": XYZ_control, "Demo data": XYZ_demo},
        value="Control data"
    )
    choice_of_XYZ
    return (choice_of_XYZ,)


@app.cell
def _(PCA, choice_of_XYZ, mo, np):
    mo.stop(choice_of_XYZ.value is None)
    XYZ = choice_of_XYZ.value
    pca = PCA()
    pca.fit(XYZ)
    pca_scores = pca.transform(XYZ) / np.sqrt(pca.explained_variance_)
    return XYZ, pca


@app.cell
def _(mo, np, pca, pd, plt, sns):
    # Get explained variance ratio
    explained_variance_ratio = pca.explained_variance_ratio_[:10]

    # Create a DataFrame for plotting
    explained_variance_df = pd.DataFrame(
        {
            "Principal Component": [
                f"PC{i+1}" for i in range(len(explained_variance_ratio))
            ],
            "Explained Variance Ratio": explained_variance_ratio,
        }
    )

    cumulative_explained_variance = np.cumsum(explained_variance_ratio)
    cumulative_df = pd.DataFrame(
        {
            "Principal Component": [
                f"PC{i+1}" for i in range(len(cumulative_explained_variance))
            ],
            "Cumulative Explained Variance": cumulative_explained_variance,
        }
    )

    # Plot using Seaborn's barplot
    plt.figure(figsize=(8, 5))
    sns.barplot(
        x="Principal Component",
        y="Explained Variance Ratio",
        data=explained_variance_df,
        label="Explained variance ratio",
    )
    sns.lineplot(
        x="Principal Component",
        y="Cumulative Explained Variance",
        data=cumulative_df,
        marker="o",
        color="red",
        label="Cumulative explained variance",
    )
    plt.xlabel("Principal Component")
    plt.ylabel("Explained Variance Ratio")
    plt.ylim(0, 1)  # Explained variance ratio is between 0 and 1
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # Plot cumulative explained variance

    mo.mpl.interactive(plt.gcf())
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ---
    # Visualize results
    """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""## Plot individual modes""")
    return


@app.cell
def _(mo, pca):
    mode_ed_dropdown = mo.ui.dropdown(
        options=[ii for ii in range(1, pca.n_components_ + 1)], label="Mode", value=1
    )
    mode_es_dropdown = mo.ui.dropdown(
        options=[ii for ii in range(1, pca.n_components_ + 1)], label="Mode", value=1
    )
    score_ed_slider = mo.ui.slider(start=-3, stop=3, step=0.5, label="Score", value=0)
    score_es_slider = mo.ui.slider(start=-3, stop=3, step=0.5, label="Score", value=0)
    return mode_ed_dropdown, mode_es_dropdown, score_ed_slider, score_es_slider


@app.cell
def _(
    mode_ed_dropdown,
    mode_es_dropdown,
    np,
    pca,
    score_ed_slider,
    score_es_slider,
):
    xyz_mu = pca.mean_.reshape((-1, 3), order="C")
    _m_ed = mode_ed_dropdown.value - 1
    _m_es = mode_es_dropdown.value - 1

    ed_scores = np.zeros(pca.n_components_)
    ed_scores[_m_ed] = score_ed_slider.value * np.sqrt(pca.explained_variance_[_m_ed])
    xyz_ed = pca.inverse_transform(ed_scores).reshape((-1, 3), order="C")[:801]

    es_scores = np.zeros(pca.n_components_)
    es_scores[_m_es] = score_es_slider.value * np.sqrt(pca.explained_variance_[_m_es])
    xyz_es = pca.inverse_transform(es_scores).reshape((-1, 3), order="C")
    return xyz_ed, xyz_es, xyz_mu


@app.cell
def _(mo, np, px, xyz_ed, xyz_es, xyz_mu):
    fig_ed = px.scatter_3d(
        x=xyz_ed[:, 0],
        y=xyz_ed[:, 1],
        z=xyz_ed[:, 2],
        range_x=[-1.5*np.abs(xyz_mu[:,0].min()) + np.mean(xyz_mu[:, 0]), 1.5*np.abs(xyz_mu[:,0].max()) + np.mean(xyz_mu[:, 0])],
        range_y=[-1.5*np.abs(xyz_mu[:,1].min()) + np.mean(xyz_mu[:, 1]), 1.5*np.abs(xyz_mu[:,1].max()) + np.mean(xyz_mu[:, 1])],
        range_z=[-1.5*np.abs(xyz_mu[:,2].min()) + np.mean(xyz_mu[:, 2]), 1.5*np.abs(xyz_mu[:,2].max()) + np.mean(xyz_mu[:, 2])],
        template="simple_white",
    )
    fig_ed.update_layout(
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="manual",
            aspectratio=dict(x=1, y=1, z=1)
        ),
        autosize=False,
        dragmode="turntable",
        width=900,
        height=900,
        scene_camera_projection_type='orthographic'
    )

    fig_es = px.scatter_3d(
        x=xyz_es[:, 0],
        y=xyz_es[:, 1],
        z=xyz_es[:, 2],
        range_x=[-1 + np.mean(xyz_mu[:, 0]), 1 + np.mean(xyz_mu[:, 0])],
        range_y=[-1 + np.mean(xyz_mu[:, 1]), 1 + np.mean(xyz_mu[:, 1])],
        range_z=[-1 + np.mean(xyz_mu[:, 2]), 1 + np.mean(xyz_mu[:, 2])],
        template="simple_white",
    )
    fig_es.update_layout(
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="manual",
            aspectratio=dict(x=1, y=1, z=1),
        ),
        autosize=False,
        dragmode="turntable",
        width=900,
        height=900,
        scene_camera_projection_type='orthographic',
    )
    plot_ed = mo.ui.plotly(fig_ed)
    plot_es = mo.ui.plotly(fig_es)
    return plot_ed, plot_es


@app.cell
def _(
    mo,
    mode_ed_dropdown,
    mode_es_dropdown,
    plot_ed,
    plot_es,
    score_ed_slider,
    score_es_slider,
):
    tab1 = mo.vstack(
        [
            mo.hstack(
                [mode_ed_dropdown]
            ),
            mo.hstack(
                [score_ed_slider, mo.md(f"Selected score: {score_ed_slider.value}")], justify="start"
            ),
            mo.hstack([plot_ed, plot_ed.value]),
        ],
        justify="center",
    )

    tab2 = mo.vstack(
        [
            mo.hstack(
                [mode_es_dropdown]
            ),
            mo.hstack(
                [score_es_slider, mo.md(f"Selected score: {score_es_slider.value}")], justify="start"
            ),
            mo.hstack([plot_es, plot_es.value]),
        ],
        justify="center",
    )

    tabs = mo.ui.tabs({"ED shape": tab1, "ES shape": tab2})

    tabs
    return


@app.cell
def _(mo):
    mo.md(r"""## Plot individual patients""")
    return


@app.cell
def _(mo):
    button_surface = mo.ui.run_button()
    button_surface
    return (button_surface,)


@app.cell
def _(XYZ, button_surface, mo, pv):
    mo.stop(not button_surface.value, "Click the button to continue")
    # button_surface.value = None
    cloud = pv.PolyData(XYZ[0].reshape((-1,3))[:801][::3])

    # Perform 3D Delaunay triangulation and extract the surface
    mesh = cloud.delaunay_3d().extract_surface()

    # Plot the resulting mesh
    plotter = pv.Plotter()
    plotter.add_mesh(mesh, show_edges=True, color='lightblue')
    plotter.add_points(cloud, color='red', point_size=5)

    # surf_plot = plotter.export_html(None)
    mo.md(f"{mo.as_html(plotter.export_html("test.html"))}")
    return


@app.cell
def _():
    # TODO: setup patient shape import, include list of scores 1:N
    return


@app.cell
def _(PCA, n_points, np):
    def get_pca_scores(xyz: np.ndarray, pca: PCA, n_components: int = 10):
        if xyz.ndim < 2:
            scores = pca.transform(xyz.reshape(1, -1)) / np.sqrt(pca.explained_variance_)
            return scores[0, :n_components]
        else:
            scores = pca.transform(xyz) / np.sqrt(pca.explained_variance_)
            return scores[:, :n_components]

    def get_xyz_from_scores(scores: np.ndarray | list, pca: PCA):
        if scores.ndim < 2:
            xyz = pca.inverse_transform(scores).reshape((n_points, 3), order="F")
            return xyz
        else:
            xyz = np.array([_xyz.reshape((n_points, 3), order="F") for _xyz in pca.inverse_transform(scores)])
            return xyz
    return


@app.cell
def _():
    import marimo as mo
    import os
    from pathlib import Path
    import matplotlib.pyplot as plt
    import numpy as np
    from dotenv import load_dotenv, find_dotenv
    from sklearn.decomposition import PCA
    import seaborn as sns
    import pandas as pd
    import plotly.express as px
    import pyvista as pv

    return (
        PCA,
        Path,
        find_dotenv,
        load_dotenv,
        mo,
        np,
        os,
        pd,
        plt,
        pv,
        px,
        sns,
    )


if __name__ == "__main__":
    app.run()
