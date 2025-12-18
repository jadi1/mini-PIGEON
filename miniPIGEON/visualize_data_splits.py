import folium
import random
import pandas as pd


# creates an interactive map with data points for each of train/val/test splits
def plot_data_folium(df, map_center=(0,0), zoom_start=2):
    
    m = folium.Map(location=map_center, zoom_start=zoom_start)
    
    color = "#{:06x}".format(random.randint(0, 0xFFFFFF))

    for _, row in df.iterrows():
        folium.CircleMarker(
            location=(row["lat"], row["lng"]),
            radius=2,
            color=color,
            fill=True,
            fill_opacity=0.6,
        ).add_to(m)
        
    return m

# load data
df =pd.read_csv("data/metadata.csv")

train_df = df[df['selection'] == 'train']
val_df = df[df['selection'] == 'val']
test_df = df[df['selection'] == 'test']

splits = {'train': train_df,'val':val_df,'test':test_df}
for split, df in splits.items():
    m = plot_data_folium(df, map_center=(20,0))
    m.save(f"{split}_dataset_map.html")
