from pathlib import Path
import os
import subprocess
import sys

import autolens as al
import autolens.plot as aplt



workspace_path = Path(os.getcwd())

dataset_type = "imaging"

config_path = workspace_path / "config"
dataset_path = workspace_path / "dataset" / dataset_type
output_path = workspace_path / "output"

dataset_name = "euclid"

dataset_pixel_scale_dict = {
    "euclid": 0.1,
    "hst": 0.05,
    "jwst": 0.03,
    "ao": 0.02,
    "ao_high": 0.01,
}

dataset_shape_native_dict = {
    "euclid": (100, 100),
    "hst": (200, 200),
    "jwst": (300, 300),
    "ao": (500, 500),
    "ao_high": (1000, 1000),
}

pixel_scales = dataset_pixel_scale_dict[dataset_name]
shape_native = dataset_shape_native_dict[dataset_name]

psf = al.Kernel2D.from_gaussian(
    shape_native=(21, 21), sigma=pixel_scales, pixel_scales=pixel_scales, normalize=True
)

grid = al.Grid2D.uniform(
    shape_native=shape_native,
    pixel_scales=pixel_scales,
)

over_sample_size = al.util.over_sample.over_sample_size_via_radial_bins_from(
    grid=grid,
    sub_size_list=[32, 8, 2],
    radial_list=[0.3, 0.6],
    centre_list=[(0.0, 0.0)],
)

grid = grid.apply_over_sampling(over_sample_size=over_sample_size)


"""
To simulate the dataset dataset we first create a simulator, which defines the exposure time, background sky,
noise levels and psf of the dataset that is simulated.
"""
simulator = al.SimulatorImaging(
    exposure_time=300.0, psf=psf, background_sky_level=0.1, add_poisson_noise_to_data=True
)

source_galaxy = al.Galaxy(
    redshift=1.0,
    light=al.lp.SersicCore(
        centre=(0.01, 0.01),
        ell_comps=(0.096225, -0.055555),
        intensity=1.5,
        effective_radius=0.3,
        sersic_index=2.5,
    ),
)

dataset_path = dataset_path / dataset_name

"""
Setup the lens galaxy's mass (SIE+Shear), subhalo (NFW) and source galaxy light (elliptical Sersic) for this 
simulated lens.
"""
lens_galaxy = al.Galaxy(
    redshift=0.5,
    bulge=al.lp.Sersic(
        centre=(0.0, 0.0),
        ell_comps=(0.0, -0.05),
        intensity=0.35,
        effective_radius=0.7,
        sersic_index=3.0,
    ),
    disk=al.lp.Sersic(
        centre=(0.0, 0.0),
        ell_comps=(0.02, -0.05),
        intensity=0.35,
        effective_radius=4.5,
        sersic_index=1.2,
    ),
    mass=al.mp.PowerLaw(
        centre=(0.0, 0.0),
        einstein_radius=2.0,
        ell_comps=(0.02, -0.02),
        slope=2.1,
    ),
    shear=al.mp.ExternalShear(gamma_1=0.0, gamma_2=0.05),
)

"""
Use these galaxies to setup a tracer, which will generate the image for the simulated dataset dataset.
"""
tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy])

"""
Output the tracer's image, this is the image we simulate.
"""
mat_plot_2d = aplt.MatPlot2D(output=aplt.Output(path=dataset_path, format=["png", "fits"]))

tracer_plotter = aplt.TracerPlotter(
    tracer=tracer, grid=grid, mat_plot_2d=mat_plot_2d
)
tracer_plotter.figures_2d(image=True)

"""
We can now pass this simulator a tracer, which creates the ray-traced image plotted above and simulates it as an
dataset dataset.
"""
dataset = simulator.via_tracer_from(tracer=tracer, grid=grid)

"""
Plot the simulated dataset dataset before we output it to fits.
"""
mat_plot_2d = aplt.MatPlot2D(output=aplt.Output(path=dataset_path, format="png"))
imaging_plotter = aplt.ImagingPlotter(dataset=dataset, mat_plot_2d=mat_plot_2d)
imaging_plotter.subplot_dataset()

"""
Output dataset to the `dataset_path` as .fits files
"""
dataset.output_to_fits(
    data_path=dataset_path / "data.fits",
    psf_path=dataset_path / "psf.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    overwrite=True,
)

"""
Pickle the `Tracer` in the dataset folder, ensuring the true `Tracer` is safely stored and available if we need to 
check how the dataset was simulated in the future. 

This will also be accessible via the `Aggregator` if a model-fit is performed using the dataset.
"""
al.output_to_json(
    obj=tracer,
    file_path=dataset_path / "tracer.json"
)

"""
Produce the no lens light data.
"""
lens_image = lens_galaxy.padded_image_2d_from(
    grid=grid, psf_shape_2d=psf.shape_native,
)

lens_image = psf.convolved_image_from(image=lens_image, blurring_image=None)

lens_image = lens_image.trimmed_after_convolution_from(kernel_shape=psf.shape_native)

data_no_lens = dataset.data - lens_image

plotter = aplt.Array2DPlotter(
    array=data_no_lens, mat_plot_2d=mat_plot_2d
)
plotter.set_filename("data_no_lens")
plotter.figure_2d()

data_no_lens.output_to_fits(file_path=dataset_path / "data_no_lens.fits", overwrite=True)

print(f"Simulated Dataset {dataset_path}")

solver = al.PointSolver.for_grid(
    grid=grid, pixel_scale_precision=0.001, magnification_threshold=0.1
)

positions = solver.solve(
    tracer=tracer, source_plane_coordinate=source_galaxy.light.centre
)

al.output_to_json(
    file_path=dataset_path / "positions.json",
    obj=positions,
)