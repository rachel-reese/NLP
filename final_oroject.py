import pandas as pd
import os
from openai import OpenAI

dataset = pd.read_csv('final_dataset.csv')
results_df = pd.DataFrame(columns=['tweet', 'bias'])

client = OpenAI()

print("\n\ngpt-4\n")

for tweet in dataset["tweet"]:
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system",
                "content": f"Please assess the bias of the following tweet: {tweet}. if the tweet is biased towards the right, print the letter R. if the tweet is biased towards the left, print the letter L. if the tweet is unbiased, print the letter N."}
        ]
    )

    row = {"tweet": tweet, "bias": response.choices[0].message.content}
    row = pd.DataFrame([row])
    results_df = pd.concat([results_df, row])

results_df.to_csv('results.csv')
