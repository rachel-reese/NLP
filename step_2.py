import nltk
import re
import pandas as pd

# step 2
# tokenize tweets
nltk.download('punkt')
all_tweets = []
prev_line = ""
append_to_df = pd.read_csv('dataset.csv', index_col=False)
with open('2nd_presidential_debate_2020/debate_61.txt', 'r') as f:
    lines = f.readlines()

    for line in lines:
        prev_timestamp = re.search("0000 2020$", prev_line)
        if bool(prev_timestamp) and bool(re.search("[a-zA-Z]+", line)):
            all_tweets.append(line)
        prev_line = line

f.close()

token = pd.DataFrame(columns=["tweet", "tokens", "length"])
for i in range(len(all_tweets)):
    tokens = nltk.word_tokenize(all_tweets[i])
    row = {"tweet": all_tweets[i], "tokens": tokens, "length": len(tokens)}
    row = pd.DataFrame(row)
    token = pd.concat([token, row])

append_to_df = pd.concat([append_to_df, token])
append_to_df.to_csv('dataset.csv')

tweets = append_to_df['tweet']
tweets.drop_duplicates(inplace=True)
tweets.reset_index()
tweets.to_csv('final_dataset.csv')
