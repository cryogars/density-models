# -*- coding: utf-8 -*-
"""
Created on Tue Jan  10 17:08 2024

This script contains functions for pulling SNOTEL data from National Resources Conservation Service (NRCS) using the metloom Python API.

Author: Ibrahim Alabi
Email: ibrahimolalekana@u.boisestate.edu
Institution: Boise State University
"""

import os
import logging
import argparse
import pandas as pd
from datetime import datetime
from typing import List, Tuple
from metloom.pointdata import SnotelPointData

"""
Configure logging
"""

# Create a log directory and file
parent_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
log_dir = os.path.join(parent_directory, "log")
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "data_download.log")

# Empty the log file
with open(log_file, "w"):
    pass  # Opening in 'w' mode and closing will clear the file

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
    level=logging.INFO,
    handlers=[logging.FileHandler(log_file)],
)

# List of allowed variables to pull
ALLOWED_VARIABLES = [
    SnotelPointData.ALLOWED_VARIABLES.SWE,
    SnotelPointData.ALLOWED_VARIABLES.TEMPMIN,
    SnotelPointData.ALLOWED_VARIABLES.TEMPMAX,
    SnotelPointData.ALLOWED_VARIABLES.TEMPAVG,
    SnotelPointData.ALLOWED_VARIABLES.SNOWDEPTH,
    SnotelPointData.ALLOWED_VARIABLES.PRECIPITATION,
]


class SnotelData:
    """
    A class for pulling SNOTEL data from NRCS using the metloom API.
    """

    def __init__(
        self,
        start_date: Tuple[int, int, int],
        end_date: Tuple[int, int, int],
        triplets: List[str],
        station_name: List[str],
        elevation: List[int],
        snow_class: List[str],
    ):
        """
        Initialize the SnotelData object.

        Arguments:
        ==========
            * start_date (Tuple[int, int, int]): The start date for data pulling (year, month, day).
            * end_date (Tuple[int, int, int]): The end date for data pulling (year, month, day).
            * triplets (List[str]): A list of site triples.
            * station_name (List[str]): A list of station names.
            * elevation (List[int]): A list of elevations for each station.
            * snow_class (List[str]): A list of snow classes for each station.
        """

        self.start_date = start_date
        self.end_date = end_date
        self.triplets = triplets
        self.station_name = station_name
        self.elevation = elevation
        self.snow_class = snow_class

    def grab_daily_data(self):
        """
        Pull the daily data for the specified stations and date range.

        Arguments:
        ==========
            * n_jobs (int): The number of workers to use. Use -1 for all available CPUs.

        Returns:
        ========
            * final_df (pd.DataFrame): The final DataFrame containing the pulled data. The function returns None if no data is found.
        """

        # Check if the number of stations, station names, elevations, and snow classes are the same
        if (
            len(self.triplets) != len(self.station_name)
            or len(self.triplets) != len(self.elevation)
            or len(self.triplets) != len(self.snow_class)
        ):
            logging.error(
                "The number of stations, station names, elevations, and snow classes must be the same!"
            )
            return None

        logging.info("Pulling data...")

        results = [
            self.fetch_data_for_station(triplet, name, elev, s_class)
            for triplet, name, elev, s_class in zip(
                self.triplets, self.station_name, self.elevation, self.snow_class
            )
        ]

        # TODO: debug multiple workers (returns errors)

        # if n_jobs == -1:
        #     n_jobs = os.cpu_count()  # Use all available CPUs

        # with ThreadPoolExecutor(max_workers=n_jobs) as executor:

        #     futures = [
        #         executor.submit(self.fetch_data_for_station, triplet, name, elev, s_class)
        #         for triplet, name, elev, s_class in zip(self.triplets, self.station_name, self.elevation, self.snow_class)
        #     ]

        #     results = [future.result() for future in as_completed(futures)]

        data_frames = [result for result in results if result is not None]

        if not data_frames:
            logging.error("No data found for any of the stations!")
            return None
        else:
            final_df = pd.concat(data_frames, ignore_index=False)
            logging.info("Data successfully pulled!")
            return final_df

    def fetch_data_for_station(self, triplet, name, elev, s_class):
        """
        Fetch the data for a given station.

        Arguments:
        ==========
            * triplet (str): The site triplet.
            * name (str): The station name.
            * elev (int): The station elevation.
            * s_class (str): The snow class.

        Returns:
        ========
            * data (pd.DataFrame): The DataFrame containing the pulled data. The function returns None if no data is found.
        """

        try:
            snotel_point = SnotelPointData(station_id=triplet, name=name)
            data = snotel_point.get_daily_data(
                start_date=datetime(*self.start_date),
                end_date=datetime(*self.end_date),
                variables=ALLOWED_VARIABLES,
            )
            if data is not None:
                data["Elevation"] = elev
                data["Station Name"] = name
                data["Snow_Class"] = s_class
                logging.info(f"Data successfully pulled for {name}!")
                return data
            else:
                logging.warning(f"No data found for {name}!")
                return None

        except Exception as e:
            logging.error(
                f"Failed to pull data for {name}! \nError: {e}", exc_info=True
            )
            return None


def main() -> None:

    parser = argparse.ArgumentParser(
        description="Pull SNOTEL data from NRCS using the metloom API for a particular station."
    )
    parser.add_argument(
        "--start_date",
        type=int,
        nargs=3,
        help="The start date for data pulling (year, month, day)",
    )
    parser.add_argument(
        "--end_date",
        type=int,
        nargs=3,
        help="The end date for data pulling (year, month, day)",
    )
    parser.add_argument(
        "--triplets", type=str, nargs="+", help="A list of site triplets"
    )
    parser.add_argument(
        "--station_name", type=str, nargs="+", help="A list of station names"
    )
    parser.add_argument(
        "--elevation",
        type=float,
        nargs="+",
        help="A list of elevations for each station",
    )
    parser.add_argument(
        "--snow_class",
        type=str,
        nargs="+",
        help="A list of snow classes for each station",
    )

    args = parser.parse_args()

    # Initialize the SnotelData with provided command line arguments
    snotel_data = SnotelData(
        start_date=tuple(args.start_date),
        end_date=tuple(args.end_date),
        triplets=args.triplets,
        station_name=args.station_name,
        elevation=args.elevation,
        snow_class=args.snow_class,
    )

    try:
        # Fetch the data
        final_df = snotel_data.grab_daily_data()

        if final_df is not None:
            # Construct the filename
            filename = f"{parent_directory}/snotel_data.csv"
            final_df.to_csv(filename)
            logging.info(f"Data successfully pulled! Check {filename} for the data.")

        else:
            logging.warning(
                "Failed to pull data for the stations, resulting in an empty dataset."
            )

    except Exception as e:
        logging.error(f"An error occurred during data fetching: {e}")


if __name__ == "__main__":
    main()
