
"""
Created on Tue Jan  17 11:15 2024

This script contains codes to compute snow water equivalent (SWE) using four statistical models:

    1. Sturm et al. (2010) - DOI: https://doi.org/10.1175/2010JHM1202.1
    2. Jonas et al. (2009) - DOI: https://doi.org/10.1016/j.jhydrol.2009.09.021
    3. Pistochi A. (2016) - DOI: https://doi.org/10.1016/j.ejrh.2016.03.004
    4. Hill et al. (2019) - DOI: https://doi.org/10.5194/tc-13-1767-2019

Author: Ibrahim Alabi
Email: ibrahimolalekana@u.boisestate.edu
Institution: Boise State University
"""

import warnings
import argparse
import numpy as np
import pandas as pd
from typing import Dict, Optional, Any
from ..utils._jonas_model_constants import  MONTH_MAPPING
from ..utils._sturm_model_constants import VALID_SNOW_CLASSES
from ..density import (
    SturmDensity, 
    JonasDensity, 
    PistochiDensity
)


class HillSWE:
    def __init__(self):
        """Initialize the HillSWE class."""
        pass

    def swe_acc_and_abl(self, pptwt: float, TD: float, DOY: int, h: float) -> Dict[str, float]:
        """
        Calculate accumulated and ablated snow water equivalent using Hill et al. (2019).

        Parameters:
        ===========
            * pptwt (float): Winter precipitation in mm.
            * TD (float): Temperature difference degree celcius.
            * DOY (int): Day of the year (with October 1 as the origin).
            * h (float): Snow depth in mm.

        Returns:
        ========
            * Dict[str, float]: A dictionary with keys 'swe_acc' and 'swe_abl' representing
                              accumulated (mm) and ablated (mm) snow water equivalent, respectively.
        """

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", RuntimeWarning)
                A, a1, a2, a3, a4 = 0.0533, 0.9480, 0.1701, -0.1314, 0.2922
                B, b1, b2, b3, b4 = 0.0481, 1.0395, 0.1699, -0.0461, 0.1804

                swe_acc = (A * h ** a1) * (pptwt ** a2) * (TD ** a3) * (DOY ** a4)
                swe_abl = (B * h ** b1) * (pptwt ** b2) * (TD ** b3) * (DOY ** b4)

                return {'swe_acc': swe_acc, 'swe_abl': swe_abl}
            
        except RuntimeWarning as e:
            return {'swe_acc': np.nan, 'swe_abl': np.nan}
        
        except Exception as e:
            return {'swe_acc': np.nan, 'swe_abl': np.nan}
        

    def SWE_Hill(self, swe_acc: float, swe_abl: float, DOY: int, DOY_: int = 180) -> float:
        """
        Compute the snow water equivalent on a particular day using Hill et al. (2019)'s model.

        Parameters:
        ===========
            * swe_acc (float): Accumulated snow water equivalent (mm).
            * swe_abl (float): Ablated snow water equivalent (mm).
            * DOY (int): Day of the year (with October 1 as the origin).
            * DOY_ (int): Day of peak SWE, default is 180.

        Returns:
        ========
            * float: Computed snow water equivalent for the given day (mm).
        """
        first = swe_acc * 0.5 * (1 - np.tanh(0.01 * (DOY - DOY_)))
        second = swe_abl * 0.5 * (1 + np.tanh(0.01 * (DOY - DOY_)))
        return first + second
    

    def compute_swe(self, pptwt: float, TD: float, DOY: int, snow_depth: float) -> Optional[float]:
        """
        Compute the snow water equivalent (SWE) based on precipitation weight, temperature difference, day of the year, and height or depth parameter.

        Parameters:
        ===========
            * pptwt (float): Winter precipitation in mm.
            * TD (float): Temperature difference degree celcius.
            * DOY (int): Day of the year (with October 1 as the origin).
            * snow_depth (float): Snow depth in mm.

        Returns:
        ========
            * Optional[float]: The computed snow water equivalent (cm) as a float, or None if any input is NaN.

        Raises:
        =======
            ValueError: If any of the inputs are not valid (e.g., NaN values).
        """
        # Check if any input is missing or not a number, return None if so
        if pd.isna(pptwt) or pd.isna(TD) or pd.isna(DOY) or pd.isna(snow_depth):
            return None  

        # Calculate accumulated and ablated SWE using provided formulas
        swe_preds = self.swe_acc_and_abl(pptwt, TD, DOY, snow_depth)

        # Calculate final SWE using the Hill model
        swe = self.SWE_Hill(swe_preds['swe_acc'], swe_preds['swe_abl'], DOY)

        return swe / 10  # Adjusted to convert to cm (orginally in mm)


class SWE_Models(HillSWE):
    def __init__(self, algorithm: str = 'default', **kwargs: Any):
        """
        Initialize the SWE model with a specified algorithm and additional keyword arguments.

        Parameters:
        ===========
            * algorithm (str): The name of the algorithm to use for SWE calculation.
            * kwargs (Any): Additional parameters specific to each algorithm.
        """
        super().__init__()
        self.algorithm = algorithm
        self.kwargs = kwargs

    def calculate_swe(self) -> float:
        """
        Calculate the snow water equivalent (SWE) based on the chosen algorithm and parameters.

        Returns:
        ========
            * float: The calculated snow water equivalent.

        Raises:
        =======
            * ValueError: If an unsupported algorithm is specified.
        """
        if self.algorithm.lower() == 'default':
            depth = self.kwargs.get('snow_depth', np.nan)
            density = self.kwargs.get('snow_density', np.nan)
            return self.default_SWE(depth, density)
        
        elif self.algorithm.lower() == 'hill':
            return self.compute_swe(**self.kwargs)
        
        elif self.algorithm.lower() == 'sturm':
            depth = self.kwargs.get('snow_depth', np.nan)
            DOY = self.kwargs.get('DOY', np.nan)
            snow_class = self.kwargs.get('snow_class', np.nan)
            density = SturmDensity().compute_density(snow_depth=depth, DOY=DOY, snow_class=snow_class)
            return self.default_SWE(depth, density)

        elif self.algorithm.lower() == 'jonas':
            depth = self.kwargs.get('snow_depth', np.nan)
            month = self.kwargs.get('month', np.nan)
            elevation = self.kwargs.get('elevation', np.nan)
            density = JonasDensity().compute_density(snow_depth=depth, month=month, elevation=elevation)
            return self.default_SWE(depth*100, density)

        elif self.algorithm.lower() == 'pistochi':
            density = PistochiDensity().compute_density(DOY=self.kwargs.get('DOY', np.nan))
            depth = self.kwargs.get('snow_depth', np.nan)
            return self.default_SWE(depth, density)
        
        else:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")

    def default_SWE(self, snow_depth: float, snow_density: float) -> float:
        """
        Calculate snow water equivalent using depth and density - the default algorithm.

        Parameters:
        ===========
            * depth (float): The depth of the snow in cm.
            * density (float): The density of the snow in g/cm^3.

        Returns:
        ========
            * float: The calculated snow water equivalent.
        """
        return snow_depth * snow_density

def main():

    help_snow_class="The snow class. See Matthew Sturm and Glen E. Liston 2001."

    parser = argparse.ArgumentParser(description='Calculate Snow Water Equivalent (cm) using various statistical models.')

    # Define a common argument for algorithm
    parser.add_argument('-a', '--algorithm', type=str, default='default', help='Algorithm to use for SWE calculation')

    # Parse known args first to determine the algorithm
    args, remaining_argv = parser.parse_known_args()

    # Depending on the algorithm, add specific arguments
    if args.algorithm.lower() == 'default':
        parser.add_argument('-d', '--snow_depth', type=float, help='Snow depth in cm')
        parser.add_argument('-ds', '--snow_density', type=float, help='Snow density in g/cm^3')
    
    elif args.algorithm.lower() == 'sturm':
        parser.add_argument('--snow_depth', type=float, required=True, help='Snow depth in meters.')
        parser.add_argument('--DOY', type=int, required=True, help='Day of the year.')
        parser.add_argument('--snow_class', type=str, required=True, choices=VALID_SNOW_CLASSES, help=f'{help_snow_class}')

    elif args.algorithm.lower() == 'jonas':
        parser.add_argument('--snow_depth', type=float, required=True, help='Snow depth in meters.')
        parser.add_argument('--month', type=str, required=True, help=f"Month. Must be one of {', '.join(MONTH_MAPPING.keys())}.")
        parser.add_argument('--elevation', type=float, required=True, help='Elevation in meters.')
        
    elif args.algorithm.lower() == 'pistochi':
        parser.add_argument('--DOY', type=int, required=True, help='Day of the year.')
    
    elif args.algorithm.lower() == 'hill':
        parser.add_argument('--pptwt', type=float, help='Winter precipitation (mm)')
        parser.add_argument('--TD', type=float, help='Temperature difference')
        parser.add_argument('--DOY', type=int, help='Day of the Year')
        parser.add_argument('--snow_depth', type=float, help='Snow depth (mm)')
    
    else:
        parser.print_help()

    # Re-parse args with new arguments specific to the chosen algorithm
    args = parser.parse_args(remaining_argv)

    # Dictionary to collect all arguments
    kwargs = {key: value for key, value in vars(args).items() if key not in ['algorithm']}

    # Use parsed arguments to create an instance of SWE_Models and calculate SWE
    try:
        swe_model = SWE_Models(algorithm=args.algorithm, **kwargs)
        result = swe_model.calculate_swe()
        print(result)
    
    except ValueError as e:
        print(e)
        parser.print_help()

if __name__ == "__main__":
    main()