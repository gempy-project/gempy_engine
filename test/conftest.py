import pandas as pd
import pytest


@pytest.fixture(scope='package')
def moureze():
    # %%
    # Loading surface points from repository:
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #
    # With pandas we can do it directly from the web and with the right args
    # we can directly tidy the data in gempy style:
    #

    # %%
    Moureze_points = pd.read_csv(
        'https://raw.githubusercontent.com/Loop3D/ImplicitBenchmark/master/Moureze/Moureze_Points.csv', sep=';',
        names=['X', 'Y', 'Z', 'G_x', 'G_y', 'G_z', '_'], header=0, )
    Sections_EW = pd.read_csv(
        'https://raw.githubusercontent.com/Loop3D/ImplicitBenchmark/master/Moureze/Sections_EW.csv',
        sep=';',
        names=['X', 'Y', 'Z', 'ID', '_'], header=1).dropna()
    Sections_NS = pd.read_csv(
        'https://raw.githubusercontent.com/Loop3D/ImplicitBenchmark/master/Moureze/Sections_NS.csv',
        sep=';',
        names=['X', 'Y', 'Z', 'ID', '_'], header=1).dropna()

    # %%
    # Extracting the orientatins:
    #

    # %%
    mask_surfpoints = Moureze_points['G_x'] < -9999
    sp = Moureze_points[mask_surfpoints]
    orientations = Moureze_points[~mask_surfpoints]
    return sp, orientations