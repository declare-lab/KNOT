import pandas as pd
import logging

from sklearn.model_selection import train_test_split

import sklearn
import torch

f_names = ['loc_Clothing_Shoes_and_Jewelry.csv',
            'loc_Toys_and_Games.csv',
            'loc_Cell_Phones_and_Accessories.csv',
            'globP_set_reviews_Food.csv',
            'globF_Grocery_and_Gourmet_Food.csv'
            ]

results = pd.DataFrame()

for f_name in f_names:
    print(f"\nWorking on {f_name}")
    df = pd.read_csv(f_name).dropna()

    df['labels'] = df['labels'].apply(int)-1

    print("Class proportion")
    for lab in df['labels'].unique():
        print(f"Label: {lab} has fraction {(df.labels == lab).sum()/len(df)}")

    print(f'total number of samples {len(df)}')


    #train/val text list
    df_texts = df['text'].values.tolist()
    df_labels = df['labels'].values.tolist()


    train_texts, val_texts, train_labels, val_labels = train_test_split(df_texts, df_labels,
                                                                        random_state=2021,
                                                                        test_size=.3,
                                                                        stratify=df_labels)

    val_texts, test_texts, val_labels, test_labels = train_test_split(val_texts, val_labels,
                                                                        random_state=2021,
                                                                        test_size=.66,
                                                                        stratify=val_labels)

    '''
    save train-test split
    '''
    train_test_split_dir = "../train_test_split"
    import pickle as pk
    pk.dump({
            'train_labels':train_labels, 
            'train_texts':train_texts, 
            'val_labels':val_labels, 
            'val_texts':val_texts, 
            'test_labels': test_labels, 
            'test_texts': test_texts
            }
            , open(f"{train_test_split_dir}/"+f_name.replace('csv','pkl'), 'wb'))


print(f"saved all files to the directory: {train_test_split_dir}")
