# Process data set 
import pandas as pd 

# loading raw data 
raw_sf = pd.read_csv("./data/raw/sf_data.csv", sep=';')
raw_districts = pd.read_csv("./data/raw/sf_districts.csv", sep=';')

joined_df = (raw_sf.set_index('id')
    .join(raw_districts.set_index('id'), how = 'inner')
    .reset_index()
    .assign(date=lambda x: pd.to_datetime(x.date),
            day=lambda x: x.date.dt.day,
            month=lambda x: x.date.dt.month, 
            year=lambda x: x.date.dt.year, 
            hour=lambda x: [int(x.time[i][0:2]) for i in range(len(x.time))],
            is_weekend = lambda x: [x.weekday[i] in ["friday", "saturday","sunday"] for i in range(x.shape[0])], 
            is_violent = lambda x: x.label == 'violent')
    .drop(columns = ['description', 'longitude', 'latitude'])        
    .query("date < '2018-05-15'")
    )    

joined_df.to_csv("data/processed/sf_data_processed.csv", index = False)


# subsetting to a smaller dataset from 2013 to 2018 to do modelling on 
model_data_set = (joined_df
    .query("date >= '2013-01-01'")[["id", "weekday", "district", "day", "month", "year", "hour", "is_violent"]]
)

model_data_set.to_csv("data/processed/sf_data_processed_2013_2018.csv", index = False)
