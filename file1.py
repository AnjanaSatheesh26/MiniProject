import pandas as pd

df1 = pd.read_csv("dataset_general.csv")
df2 = pd.read_csv("dataset_youtube.csv")

df1["platform"] = "news_blog_social"
df = pd.concat([df1, df2], ignore_index=True)
df = df.drop_duplicates(subset=["title"])
df.to_csv("clickbait_multiplatform.csv", index=False)
