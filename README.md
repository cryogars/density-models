

<h1 align="center">
Snow Density Estimation using Machine Learning</h1>
<p align="center">This is the codebase for "A Machine Learning Model for Estimating Snow Density and Snow Water Equivalent from Snow Depth and Seasonal Snow Climate Classes."
</p>

<details close> 
  <summary><h2>Overview</summary>

This study proposes a machine learning model for estimating snowpack bulk density ($\rho_s$) from snow depth ($HS$) and other variables that can be measured or derived from the date and location of $HS$ measurements. This repository contains:

<ul>

<li>Source code for our paper (<em>DOI</em> will be shared after paper acceptance).</li>

<li>Instructions for setup and usage.</li>

</ul>
</details>

<details close> 
  <summary><h2>Dataset</summary>

The dataset used in this study comes from thre sources:

- SNOTEL Dataset - was downloaded using [metloom](https://metloom.readthedocs.io/en/latest/usage.html#snotel).
- [Global Seasonal Snow Classification](https://nsidc.org/data/nsidc-0768/versions/1).
- [Maine Snow Survey Data](https://mgs-maine.opendata.arcgis.com/datasets/maine-snow-survey-data/explore).
</details>

<details close> 
  <summary><h2>Software & Hardware List</summary>

| Software used | Version  | Hardware specifications  | OS required |
|:---:  |:---:  |:---:  |:---:  |
| Python | 3.11.5 | The codes in this repository should work on any recent PC/Laptop | Linux (any), MacOS, Windows|
</details>


<details close> 
  <summary><h2>Installation and Setup</summary>

This project uses *Conda* for environment management. However, you can use any environment management tool of your choice. For example, you can manage Python versions with [pyenv](https://github.com/pyenv/pyenv) and create a virtual environment using [venv](https://docs.python.org/3/library/venv.html). Go to **Step 3** if you wish not to use *Conda*.

### 1. Install Conda

If you don’t have Conda installed, download **[Anaconda or MiniConda](https://www.anaconda.com/download/success)**. See [Installing Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) to know which installer is right for you.

### 2. Create and Activate the Conda Environment

```bash
conda env create --file environment.yml
conda activate ml_density_env 
```

### 3. Installation (Non-Conda Users)

For those not using Conda, you can set up a virtual environment with [venv](https://docs.python.org/3/library/venv.html):

```bash
python -m venv ml_density_env
source ml_density_env/bin/activate # activate the virtual environment
```

Note: the Python version will be the default version on your PC. You can manage Python version using [pyenv](https://github.com/pyenv/pyenv).

### 4. Create and Navigate to Directory

```bash
mkdir ml_density
cd ml_density
```

### 5. Clone This Repository

```bash
git clone https://github.com/cryogars/density-models.git
cd density-models
```

### 6. Verify Installation

Ensure everything is set up correctly:
```bash
python --version  # Should return 3.11.5
pip list  # Displays installed packages
```

### 7. Run Tests (Optional)

To verify the models are working correctly, you can run the tests:

```bash
pytest # Run all tests
```

### 8. Install Source Code

```bash
pip install .
```

If you wish to modify the source code, install in development mode:

```bash
pip install -e .
```

**Note**: This project uses `conda` to manage only manage the Python version and install Jupyter. All package dependencies are installed via `pip`. We found this approach avoids dependency conflicts that sometimes occur when mixing conda-installed and pip-installed packages in the same environment.
</details>

<details close> 
  <summary><h2>Directory Setup</summary>

Create the data folder and download:

1. SNOTEL Data: [link](https://drive.google.com/file/d/1tcMnNPq_SYLGoLEY-FeBJVaJf1qqtntZ/view?usp=sharing). 
2. Global Seasonal Snow Classification on NSIDC: [NSIDC link](https://nsidc.org/data/nsidc-0768/versions/1). For this project, download `SnowClass_NA_300m_10.0arcsec_2021_v01.0.nc`.
3. Main Snow Survey Data: [link](https://mgs-maine.opendata.arcgis.com/datasets/maine-snow-survey-data/explore).

</details>

<details close> 
  <summary><h2>Deactivate and/or Remove Environment</summary>

After running the experiments, you can deactivate the conda environment by running the command below:

```bash
conda deactivate
```

To completely remove the environment, run:
```bash
conda env remove --name ml_density_env
```
</details>

<details close> 
  <summary><h2>Acknowledgments</summary>

The authors would like to thank:

1. USDA NRCS for providing the SNOTEL data
2. [M3Works](https://m3works.io/) for their [metloom](https://metloom.readthedocs.io/en/latest/usage.html#snotel) package, which we used to download the SNOTEL data.
3. Maine Geological Survey and the United States Geological Survey for providing the [Maine Snow Survey data](https://mgs-maine.opendata.arcgis.com/datasets/maine-snow-survey-data/explore).
4. The creators of the [srtm.py Python package](https://github.com/tkrajina/srtm.py?tab=readme-ov-file) for their open-source tool, which we used to obtain the SRTM elevation data.
5. U.S. Army CRREL for the funding (BAA W913E520C0017).

</details>

<details close> 
  <summary><h2>Contact</summary>

For any questions or issues, please open an **issue** or reach out to **ibrahimolalekana@u.boisestate.edu**.

</details>