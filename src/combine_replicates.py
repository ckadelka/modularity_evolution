# -*- coding: utf-8 -*-
"""
Created on Mon Jan 26 11:07:39 2026

@author: faith
"""

import os
import pandas as pd
import glob
import numpy as np
from pathlib import Path
import sys
from globals import *











def combine_data(signature):
    base_path = Path(GLOBAL_MASTER_DIR) / "data" / signature
    
    if not base_path.exists():
        print(f"Error: Path {base_path} does not exist.")
        return

    # 2. Find all job folders (job_0, job_1, etc.)
    job_folders = sorted(list(base_path.glob("job_*")))
    
    if not job_folders:
        print("No job folders found! Did the simulation run?")
        return

    print(f"Found {len(job_folders)} job folders.")

    # 3. Get the list of expected files from the first job folder
    # We assume all jobs produce the same filenames
    reference_folder = job_folders[0]
    expected_files = list(reference_folder.glob("*.pkl"))
    
    print(f"Found {len(expected_files)} unique parameter files per job.")
    print("Starting merge process...\n")

    # 4. Loop through each parameter file (e.g., alpha_0_strength_1.pkl)
    for file_path in expected_files:
        filename = file_path.name
        print(f"Processing {filename}...")
        
        combined_data = []
        missing_count = 0

        # Collect this specific file from ALL job folders
        for job_dir in job_folders:
            target_file = job_dir / filename
            
            if target_file.exists():
                try:
                    # Load the pickle file
                    df = pd.read_pickle(target_file)
                    print(f"  Loaded {target_file} with {len(df)} rows. ")
                    
                    # If it's a list, convert to DataFrame (optional, depends on your utils)
                    if isinstance(df, list):
                        df = pd.DataFrame(df)
                        
                    combined_data.append(df)
                    print(f"  -> Appended data from {target_file}.")
                except Exception as e:
                    print(f"  Error reading {target_file}: {e}")
            else:
                missing_count += 1

        # 5. Merge and Save
        if combined_data:
            print(f"  Merging data from {len(combined_data)} files...")
            print(f" Type of combined data is list correct? {isinstance(combined_data, list)}")
            first_item = combined_data[0]
            print(f" First item type: {type(first_item)}")

            # If each pickle is a tuple/list of matrices/arrays, merge element-wise
            if isinstance(first_item, (list, tuple)):
                num_elements = len(first_item)
                print(f"  Detected tuple/list with {num_elements} elements. Merging element-wise...")
                merged_elements = []

                for i in range(num_elements):
                    parts = [item[i] for item in combined_data]

                    # If parts are DataFrame/Series -> use pd.concat
                    if all(isinstance(p, (pd.DataFrame, pd.Series)) for p in parts):
                        try:
                            merged_i = pd.concat(parts, ignore_index=True)
                        except Exception:
                            merged_i = pd.DataFrame(parts)
                    else:
                        # Try numpy concatenate for arrays/lists
                        try:
                            arrays = [np.asarray(p) for p in parts]
                            merged_i = np.concatenate(arrays)
                        except Exception:
                            # Fallback: keep as list-of-items
                            merged_i = parts

                    merged_elements.append(merged_i)

                # Save merged tuple/list
                output_path = base_path / f"{filename}"
                pd.to_pickle(tuple(merged_elements), output_path)

                # Print summary of merged element sizes
                for idx, el in enumerate(merged_elements):
                    try:
                        size = len(el)
                    except Exception:
                        size = getattr(getattr(el, 'shape', None), '__str__', lambda: 'unknown')()
                    print(f"  -> Element {idx}: merged size {size}")

                if missing_count > 0:
                    print(f"  -> Warning: {missing_count} jobs were missing this file.")
            else:
                # Standard case: items are DataFrames/Series that can be concatenated directly
                try:
                    final_df = pd.concat(combined_data, ignore_index=True)
                except Exception as e:
                    print(f"  Error concatenating dataframes: {e}")
                    final_df = None

                if final_df is not None:
                    output_path = base_path / f"{filename}"
                    final_df.to_pickle(output_path)
                    print(f"  -> Saved merged file with {len(final_df)} total rows.")
                    if missing_count > 0:
                        print(f"  -> Warning: {missing_count} jobs were missing this file.")
                else:
                    print(f"  -> Failed to concatenate {filename} into a single DataFrame.")
        else:
            print(f"  -> Failed to combine {filename} (no data found).")

    print("\n------------------------------------------------")
    print("Done! All combined files are in:")
    print(f"{base_path}")
    print("------------------------------------------------")

if __name__ == "__main__":
    #also allow passing signature from command line
    TARGET_SIGNATURE = "b7edb4804b" 

    if len(sys.argv) > 1:
        TARGET_SIGNATURE = sys.argv[1]
    combine_data(TARGET_SIGNATURE)


    