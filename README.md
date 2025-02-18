

<h1 align="center">
Snow Density Estimation using Machine Learning</h1>
<p align="center">This is the codebase for "A Machine Learning Model for Estimating Snow Water Equivalent from Snow Depth and Seasonal Snow Climate Class."
</p>

<details open> 
  <summary><h2>Overview</summary>

This study proposes a machine learning model for estimating snowpack bulk density ($\rho_s$) from snow depth ($HS$) and other variables that can be measured or derived from the date and location of $HS$ measurements.

<ul>

<li>Source code for our paper (<em>DOI</em> will be shared after paper acceptance).</li>

<li>A reproducible Conda environment.</li>

<li>Instructions for setup and usage.</li>

</ul>
</details>

<details open> 
  <summary><h2>Dataset</summary>

The dataset used in this study comes from two sources:

- SNOTEL Dataset - was downloaded using [metloom](https://metloom.readthedocs.io/en/latest/usage.html#snotel).
- [Global Seasonal Snow Classification](https://nsidc.org/data/nsidc-0768/versions/1).
</details>

<details open> 
  <summary><h2>Software & Hardware List</summary>

| Software used | Version  | Hardware specifications  | OS required |
|:---:  |:---:  |:---:  |:---:  |
| Python | 3.11.5 | The codes in this repository should work on any recent PC/Laptop | Linux (any), MacOS, Windows|
</details>


<details open> 
  <summary><h2>Installation and Setup</summary>

This project uses *Conda* for environment management.

### 1️⃣ Install Conda
If you don’t have Conda installed, download **[Anaconda or MiniConda](https://www.anaconda.com/download/success)**. See [Installing Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) to know which installer is right for you.

### 2️⃣ Create and Navigaate to Directory

```bash
mkdir ml_density
cd ml_density
```

### 3️⃣ Clone This Repository
```bash
git https://github.com/cryogars/density-models.git
cd density-models
```

### 4️⃣ Create and Activate the Conda Environment
Run the following commands to create a reproducible Conda environment:
```bash
conda env create --file environment.yml
conda activate ml_density_env 
```

### 5️⃣ Verify Installation
Ensure everything is set up correctly:
```bash
python --version  # Should return 3.11.5
conda list  # Displays installed packages
```

### 5️⃣ Install Source Code

```bash
pip install .
```

If you wish to modify the source code, install in development mode:

```bash
pip install -e .
```
</details>

<details open> 
  <summary><h2>Directory Setup</summary>

Create the data folder and dowload:

1. SNOTEL Data: [link](https://drive.google.com/file/d/1tcMnNPq_SYLGoLEY-FeBJVaJf1qqtntZ/view?usp=sharing). 
2. Global Seasonal Snow Classification on NSIDC: [NSIDC link](https://nsidc.org/data/nsidc-0768/versions/1). For this project, download `SnowClass_NA_300m_10.0arcsec_2021_v01.0.nc`.

</details>

<details open> 
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

<details open> 
  <summary><h2>Acknowledgments</summary>

The authors would like to thank:

1. USDA NRCS for providing the SNOTEL data
2. [M3Works](https://m3works.io/) for their [metloom](https://metloom.readthedocs.io/en/latest/usage.html#snotel) package, which we used to download the SNOTEL data.
3. Maine Geological Survey and the United States Geological Survey for providing the [Maine Snow Survey data](https://mgs-maine.opendata.arcgis.com/datasets/maine-snow-survey-data/explore).
4. The creators of the [srtm.py Python package](https://github.com/tkrajina/srtm.py?tab=readme-ov-file) for their open-source tool, which we used to obtain the SRTM elevation data.

</details>

<details open> 
  <summary><h2>Contact</summary>

For any questions or issues, please open an **issue** or reach out to **ibrahimolalekana@u.boisestate.edu**.

</details>