import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv("Atlantic_South_Korea_data.csv", encoding="utf-8")
# print(df.head())
# print(df.info())
# print(df.shape)





#1. Data Validation____________________________________________________________

# converting Date from Str to date
df["date"]=pd.to_datetime(df["date"])
print(df.info())
print(df.head())

# Count entries for each date
daily_count = df.groupby("date").size()
 
# Find dates that do not have 50 entries
incorrect_dates = daily_count[daily_count != 50]

print("Total dates:", len(daily_count))
print("Dates with incorrect entries:", incorrect_dates) # 2025-03-01    100

day_data = df[df["date"] == pd.Timestamp("2025-03-01")]
print(day_data["position"].value_counts().sort_index()) 

#1.1 removing duplicates 
df = df.drop_duplicates(
    subset=["date", "position"],
    keep="first"    
)

#1.2 normalize song-artist identifiers.
df["song_id"] = (
    df["song"].str.strip().str.lower()
    + " | "
    + df["artist"].str.strip().str.lower()
)

#1.3 convert duration in minutes
df["Duration_min"] = df["duration_ms"] / 60000
#print(df[["song", "duration_ms", "Duration_min"]].head())

#1.4 validationg popularity score
# Check missing values
print(df["popularity"].isna().sum())

# Check minimum and maximum values
print(df["popularity"].min())
print(df["popularity"].max())

print(df["popularity"].dtype)


#number of days song present on chart
song_days_count=(df.groupby("song_id").size().sort_values(ascending=False))
print(song_days_count.head())

#2. chart Re-entry Detection________________________________________________________
#entry_date of each song
entry_dates=(df.groupby(["song_id"]))["date"].min().reset_index(name="entry_dates")
#print(entry_dates.head())

#2.1 exit_date of each song
exit_dates=(df.groupby(["song_id"]))["date"].max().reset_index(name="exit_dates")
#print(exit_dates.head())

sorted_df=df.sort_values(["song_id","date"])
#print(sorted_df.head())

#number of absent dates
sorted_df["previous_date"]=(df.groupby(["song_id"])["date"].shift(1))

sorted_df["days_absent"] = (
    sorted_df["date"] - sorted_df["previous_date"]
).dt.days - 1

#print(sorted_df.head(50))

#2.2 re-entry or not (1,0)
sorted_df["re_entry"]=((sorted_df["previous_date"].notna()) & (sorted_df["days_absent"])>=1).astype(int)
# print(sorted_df.head())     

# 2.3 counting re-entries
total_re_entry=sorted_df["re_entry"].groupby(sorted_df["song_id"]).sum().sort_values(ascending=False)
print(total_re_entry)

#dataframe with re-entry rows only
reentries = sorted_df[sorted_df["re_entry"] == 1].copy()
reentries=reentries.sort_values(["song_id","date"])

reentries["previous_reentry_date"] = (
    reentries.groupby(["song_id"])["date"].shift(1)
)
  
#2.4 gap between re-entries
reentries["gap_between_reentries"] = (
    reentries["date"] - reentries["previous_reentry_date"]
).dt.days





#3 Momentum Spike measurement.....................................

sorted_df["previous_popularity"] = (
    sorted_df.groupby(["song_id"])["popularity"].shift(1)
)
#3.1 popularity change rate
sorted_df["popularity_change_rate"] = (
    (sorted_df["popularity"] - sorted_df["previous_popularity"])
    / sorted_df["previous_popularity"].where(sorted_df["previous_popularity"] != 0)
) * 100


#3.2 Rank Jump Magnitude   
sorted_df["previous_position"] = (
    sorted_df.groupby(["song_id"])["position"].shift(1)
)

sorted_df["rank_jump"] = (sorted_df["previous_position"] - sorted_df["position"])
print(sorted_df[["previous_position","position","rank_jump"]].head(50))

#peak Rank achieved after re-entry....................

#gap between appearance
sorted_df["date_gap"] = (
    sorted_df["date"] - sorted_df["previous_date"]
).dt.days

#3.3 peak rank achieved after re-entry
#identifying each re-entry and giving it a new run id
sorted_df["new_run"] = (
    sorted_df["previous_date"].isna() | (sorted_df["date_gap"] > 1)
)   

sorted_df["run_id"] = (
    sorted_df.groupby("song_id")["new_run"].cumsum()
)
sorted_df["peak_rank"] = (
    sorted_df.groupby(["song_id", "run_id"])["position"]
    .transform("min")
)

#3.4  moment intensity score

# Normalize popularity change rate to a 0–1 scale
pop_min = sorted_df["popularity_change_rate"].min()
pop_max = sorted_df["popularity_change_rate"].max()

sorted_df["normalized_popularity"] = (
    (sorted_df["popularity_change_rate"] - pop_min)
    / (pop_max - pop_min)
)

# Normalize rank improvement to a 0–1 scale
rank_min = sorted_df["rank_jump"].min()
rank_max = sorted_df["rank_jump"].max()

sorted_df["normalized_rank_jump"] = (
    (sorted_df["rank_jump"] - rank_min)
    / (rank_max - rank_min)
)

sorted_df["momentum_intensity_score"] = (
    0.5 * sorted_df["normalized_popularity"]
    + 0.5 * sorted_df["normalized_rank_jump"]
)
#print(sorted_df.info())


# 4. Momentum Sustainability Analysis
#4.1  Days retained after comeback

sorted_df["days_retained"] = (
    sorted_df.groupby(["song_id", "run_id"])["date"]
      .transform("max")
    - sorted_df.groupby(["song_id", "run_id"])["date"]
      .transform("min")
).dt.days + 1

print(sorted_df[["days_retained","date","song_id"]].head(50))
# 4.2 Rank decay speed post-peak

sorted_df["peak_date"] = sorted_df["date"].where(
    sorted_df["position"] == sorted_df["peak_rank"]
) #peak date of every peal rank after entry and re-entry

sorted_df["peak_date"] = sorted_df.groupby(
    ["song_id", "run_id"]
)["peak_date"].transform("min")  #peak date vale to every row 

sorted_df["days_after_peak"] = (
    sorted_df["date"] - sorted_df["peak_date"]
).dt.days

sorted_df["rank_decay_speed"] = (
    (sorted_df["position"] - sorted_df["peak_rank"])
    / sorted_df["days_after_peak"].replace(0, np.nan) #Replacing zero with NaN to prevent invalid calculation.
)
print(sorted_df[["song_id","rank_decay_speed"]].head(50))


# 4.3 Stability vs Volatility After Surge

# Keep only observations after the peak
sorted_df["post_peak_position"] = sorted_df["position"].where(
    sorted_df["days_after_peak"] > 0
)


# Calculate rank volatility for each song appearance
sorted_df["post_peak_volatility"] = (
    sorted_df.groupby(["song_id", "run_id"])["post_peak_position"]
    .transform("std")
)


sorted_df["stability_category"] = np.select(
    [
        sorted_df["post_peak_volatility"].isna(),
        sorted_df["post_peak_volatility"] <= 5,
        sorted_df["post_peak_volatility"] > 5
    ],
    [
        "Insufficient Data",
        "Stable",
        "Volatile"
    ],
    default="Insufficient Data"
)


# print(sorted_df[[
#     "song", "run_id", "post_peak_volatility",
#     "stability_category"
# ]].head(50))

#5.content attribute and momemtum analysis________________________________________

#5.1 singles v/s Albums
#  Identify songs that returned after disappearing
comebacks = sorted_df[sorted_df["run_id"] > 1].copy()

# Measure the strongest momentum score for each comeback
comeback_strength = (
    comebacks.groupby(["song_id", "run_id"])
    .agg(
        album_type=("album_type", "first"),
        comeback_strength=("momentum_intensity_score", "max")
    )
    .reset_index()
)

# Compare Single vs Album comeback strength
result = (
    comeback_strength.groupby("album_type")
    .agg(
        total_comebacks=("comeback_strength", "count"),
        average_comeback_strength=("comeback_strength", "mean")
    )
    .reset_index()
)

print(result)

# 5.2 Album Size vs Resurgence Intensity

comeback_summary = (
    comebacks.groupby(["song_id", "run_id"])
    .agg(
        total_tracks=("total_tracks", "first"),
        resurgence_intensity=("momentum_intensity_score", "max")
    )
    .reset_index()
)

correlation = comeback_summary[
    ["total_tracks", "resurgence_intensity"]
].corr().iloc[0, 1]

print("correlation between album size and resurgency ",correlation)

#5.3  Song Duration vs Spike Magnitude

correlation_duration_spike = sorted_df[
    ["duration_ms", "rank_jump"]   
].corr().iloc[0, 1]

print("Correlation between song duration and rank jump:",correlation_duration_spike )

#5.4 Explicit vs clean content momentum comparison
comeback_summary_type = (
    comebacks.groupby(["song_id", "run_id"])
    .agg(
        is_explicit=("is_explicit", "first"),
        comeback_strength=("momentum_intensity_score", "max")
    )
    .reset_index()
)

result_type = (
    comeback_summary_type.groupby("is_explicit")
    .agg(
        total_comebacks=("comeback_strength", "count"),
        average_momentum=("comeback_strength", "mean"),
        median_momentum=("comeback_strength", "median")
    )
    .reset_index()
)

result_type["content_type"] = result_type["is_explicit"].map({
    True: "Explicit",
    False: "Clean"
})

print(result_type[
    ["content_type", "total_comebacks",
     "average_momentum", "median_momentum"]
])

# 6. Fandom Engagement Proxy Analysis
#6.1 re-entry Frequency 
reentry_frequency = (
    sorted_df.groupby(["song_id", "song", "artist"])["re_entry"]
    .sum()
    .reset_index(name="reentry_frequency")
)

# Normalize Re-entry Frequency (0 to 1)

min_val = reentry_frequency["reentry_frequency"].min()
max_val = reentry_frequency["reentry_frequency"].max()

reentry_frequency["normalized_reentry_frequency"] = (
    (reentry_frequency["reentry_frequency"] - min_val)
    / (max_val - min_val)
    if max_val != min_val else 0.5
)

print(reentry_frequency.sort_values(
       "reentry_frequency", ascending=False
      ).head(50))

#6.2 Popularity spike sharpness
comebacks = sorted_df[sorted_df["re_entry"] == 1].copy()

spike_sharpness = (
    comebacks.groupby(["song_id", "song", "artist"])
    ["popularity_change_rate"]
    .max()
    .reset_index(name="spike_sharpness")
)

# Normalize Popularity Spike Sharpness (0 to 1)

min_val = spike_sharpness["spike_sharpness"].min()
max_val = spike_sharpness["spike_sharpness"].max()

spike_sharpness["normalized_spike_sharpness"] = (
    (spike_sharpness["spike_sharpness"] - min_val)
    / (max_val - min_val)
    if max_val != min_val else 0.5
)


print(spike_sharpness.sort_values(
    "spike_sharpness", ascending=False
).head(10))


# 6.3 Rank recovery speed to estimate fandom-driven engagement intensity
rank_score = sorted_df[sorted_df["re_entry"] == 1].copy()

rank_score["rank_recovery_speed"] = (
    rank_score["rank_jump"] / rank_score["date_gap"]
)

print(
    rank_score[
        ["song", "artist", "date", "rank_jump",
         "date_gap", "rank_recovery_speed"]
    ].head(50)
)
# Normalize Rank Recovery Speed (0 to 1)

min_val = rank_score["rank_recovery_speed"].min()
max_val = rank_score["rank_recovery_speed"].max()

rank_score["normalized_rank_recovery_speed"] = (
    (rank_score["rank_recovery_speed"] - min_val)
    / (max_val - min_val)
    if max_val != min_val else 0.5
)

#Combine all three normalized factors
fandom_score = (
    reentry_frequency
    .merge(spike_sharpness, on=["song_id", "song", "artist"])
    .merge(rank_score, on=["song_id", "song", "artist"])
)

#Calculate final score
fandom_score["fandom_engagement_proxy"] = (
    0.40 * fandom_score["normalized_reentry_frequency"]
    + 0.30 * fandom_score["normalized_spike_sharpness"]
    + 0.30 * fandom_score["normalized_rank_recovery_speed"]
) * 100

# Display top 50 songs
print(
    fandom_score.sort_values(
        "fandom_engagement_proxy", ascending=False
    )[
        ["song", "artist", "fandom_engagement_proxy"]
    ].head(50)
)

#marking entries as first entry and comeback
# Default category
sorted_df["entry_type"] = "Continuing"

# Mark the first-ever appearance of each song
first_entry_indices = sorted_df.groupby("song_id")["date"].idxmin()

sorted_df.loc[first_entry_indices, "entry_type"] = "First Entry"

# Mark comeback dates
sorted_df.loc[
    sorted_df["re_entry"] == 1,
    "entry_type"
] = "Comeback"


print(sorted_df[["song_id","entry_type"]].head(50))




# converting processed data into CSV files

sorted_df.to_csv(
    "Atlantic_South_Korea_data_processed.csv",
    index=False,
    encoding="utf-8-sig",
    date_format="%Y-%m-%d"
)

#re-entry frequency Table
reentry_frequency.to_csv(
    "Reentry_Frequency.csv",
    index=False,
    encoding="utf-8-sig"
)


# # Select required columns
fandom_events = fandom_score[
    [
        "song_id",
        "song",
        "artist",
        "reentry_frequency",
        "normalized_reentry_frequency",
        "spike_sharpness",
        "normalized_spike_sharpness",
        "rank_recovery_speed",
        "normalized_rank_recovery_speed",
        "fandom_engagement_proxy"
    ]
].copy()

# Sort by Fandom Engagement Proxy Score
fandom_events = fandom_events.sort_values(
    "fandom_engagement_proxy",
    ascending=False
)


# Save event-level CSV
fandom_events.to_csv(
    "Fandom_Engagement_Proxy_Top50.csv",
    index=False,
    encoding="utf-8-sig"
)

print("CSV saved successfully!")



