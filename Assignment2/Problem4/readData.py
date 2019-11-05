import json

# read data
with open("./train_data.json", 'r') as fin:
    contents = json.load(fin)

print("Total number of summaries in dictionary "+str(len(contents)))
print("Dictionary's key is the file names of a summary and value is another dictionary containing all three summary quality scores.")

# iterate over the dictionary
for k, v in contents.items():
    print(k)
    print(v)
    break
