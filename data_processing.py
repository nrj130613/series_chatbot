import pandas as pd

def preprocess_dataframe(df):
    """Cleans, normalizes, verbalizes, and combines DataFrame data into a single string per series."""
    
    # Normalize spaces
    df = df.map(lambda cell: " ".join(cell.split()) if isinstance(cell, str) else cell)
    
    # Remove unnecessary columns
    df.drop(['web-scraper-order', 'pagina', 'web-scraper-start-url', 'list-series'], axis='columns', inplace=True)
    
    # Rename columns
    df.columns = ["Source", "Title", "Length", "Genre", "EP Meta", "Actors", "Director", "Tag", "Description"]
    
    # Verbalize data
    verbalization_map = {
        "Title": "ซีรีส์เรื่อง ",
        "Length": "เป็น",
        "Genre": "ประเภท",
        "Actors": "นักแสดงนำ ได้แก่ ",
        "Director": "กำกับโดย ",
        "Tag": "Tag: ",
        "Description": "เรื่องย่อคือ "
    }
    
    for column, text in verbalization_map.items():
        df[column] = df[column].apply(lambda x: f"{text}{x}")
    
    # Sort columns
    df = df[["Title", "Length", "Genre", "EP Meta", "Actors", "Director", "Description", "Tag", "Source"]]
    
    # Combine data into a single string per series
    combined_data = pd.DataFrame()
    combined_data["series_id"] = ["series_" + str(i) for i in range(1, len(df) + 1)]
    combined_data["data"] = df.apply(lambda row: " ".join(map(str, row)), axis=1)
    combined_data["source"] = df["Source"]
    
    return combined_data

df = pd.read_csv('combined_data.csv')
preprocess_dataframe(df)
