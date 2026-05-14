import pandas as pd

def build_behavior_table(metadata_df, base_path, loader, parser):
    """
    Builds the behavioral flat table from metadata rows where added == 0.
    
    Parameters
    ----------
    metadata_df : DataFrame
        Filtered metadata (only rows with added == 0).
    base_path : str
        Path to the BehData folder.
    loader : function
        Function that returns valid files for a given session folder.
    parser : function
        Function that extracts Red/Green from a TXT file.

    Returns
    -------
    DataFrame
        Long-format behavioral table.
    """
    rows = []

    for _, meta in metadata_df.iterrows():
        run_number = 1
        folder = meta["FolderFile"]
        files = loader(base_path, folder)

        for f in files:
            parsed = parser(f)

            rows.append({
                "SubID": meta["SUB_ID"],
                "Red": parsed["Red"],
                "Green": parsed["Green"],
                "RunNumber": run_number,
                "session": meta["Session"],
                "PartType": meta["PartType"],
                "Date": meta["Date"],
                "FolderOrg": meta["FolderFile"]
            })

            run_number += 1

    return pd.DataFrame(rows)
