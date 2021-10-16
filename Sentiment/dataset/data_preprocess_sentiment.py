'''
convert json.gz file to dataframe
http://jmcauley.ucsd.edu/data/amazon/links.html
'''
file_paths = ['loc_Clothing_Shoes_and_Jewelry.json.gz',
              'loc_Toys_and_Games.json.gz',
              'loc_Cell_Phones_and_Accessories.json.gz',
              'globF_Grocery_and_Gourmet_Food.json.gz'
              ]


print(f"\nwe will process the following files...\n")
print(*file_paths, sep="\n")
print()

def process_data(file_path):
  def parse(path):
    g = gzip.open(path, 'r')
    for l in g:
      yield json.loads(l)
  #
  import pandas as pd
  import gzip
  import json
  #
  def parse(path):
    g = gzip.open(path, 'rb')
    for l in g:
      yield json.loads(l)
  #
  def getDF(path):
    i = 0
    df = {}
    for d in parse(path):
      df[i] = d
      i += 1
    return pd.DataFrame.from_dict(df, orient='index')
  #
  df = getDF(file_path)
  df = df.dropna()
  #
  df_to_save = pd.DataFrame()
  df_to_save['labels'] = df['overall']
  df_to_save['text'] = df['reviewText']
  df_to_save.to_csv(file_path.replace('.json.gz','.csv'))


for dat_path in file_paths:
  print(f"processing file: {dat_path}")
  process_data(dat_path)

print("done...")