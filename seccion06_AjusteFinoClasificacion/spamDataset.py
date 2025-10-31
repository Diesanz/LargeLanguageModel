import pandas as pd
import torch
from torch.utils.data import Dataset

def create_balanced_dataset(df):
    num_spam = df[df["Label"] == "spam"].shape[0] #estancias de spam
    ham_subset = df[df["Label"] == "ham"].sample(num_spam, random_state=123) #muestreo aleatorio de instancias "no spam" para que coicidan con el n.instancias de spam
    balanced_df = pd.concat([ham_subset, df[df["Label"] == "spam"]]) #combinar el subconjunto con el "spam"
    return balanced_df

def random_split(df, train_frac, val_frac):
    df = df.sample(frac=1, random_state=123) #barajar todo el df
    df = df.reset_index(drop=True)

    train_end = int(len(df) * train_frac ) #calcular los índices
    train_df = df[:train_end] #dividir el df

    val_end = train_end + int(len(df) * val_frac)
    val_df = df[train_end:val_end]

    #Se  supone  que  el  tamaño  de  la  prueba  es  0,2
    test_df = df[val_end:]
    return train_df, val_df, test_df

class SpamDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=None, pad_token_id=50256):
        self.data = pd.read_csv(csv_file)
        self.encoded_texts = [tokenizer.encode(text) for text in self.data["Text"]] #pre-tokenizar el texto

        if max_length is None:
            self.max_length = self._longest_encoded_length()    #truncar secuencias si son más largas que max_lenght
        else:
            self.max_length = max_length
                                                                
            self.encoded_texts = [  
                encoded_text[:self.max_length]
                for encoded_text in self.encoded_texts
            ]

        self.encoded_texts = [text + [pad_token_id] * (self.max_length - len(text)) for text in self.encoded_texts]
    
    def __getitem__(self, index):
        encoded = self.encoded_texts[index]
        label = self.data.loc[index]["Label"]
        return (torch.tensor(encoded), torch.tensor(label))

    def __len__(self):
        return len(self.data)

    def _longest_encoded_length(self):
        maxq = 0
        for text in self.encoded_texts:
            len_1 = len(text)
            if len_1 > maxq:
                maxq = len_1
        return maxq