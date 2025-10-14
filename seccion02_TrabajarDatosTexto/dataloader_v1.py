from torch.utils.data import Dataset, DataLoader
import torch
import tiktoken

class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_lenght, stride):
        self.input_ids = []
        self.labels_ids = []

        tokens_ids = tokenizer.encode(txt) #Tokenzar el texto completo

        for i in range(0, len(tokens_ids) - max_lenght, stride): #Uso de una  ventana  deslizante  para  dividir  el  libro  en  secuencias  superpuestas  de  longitud  máxima
            self.input_ids.append(torch.tensor(tokens_ids[i:i+max_lenght]))
            self.labels_ids.append(torch.tensor(tokens_ids[i+1:i+max_lenght+1]))

    def __len__(self): #Devuelve  el  número  total  de  filas  en  el  conjunto  de  datos
        return len(self.labels_ids)
    
    def __getitem__(self, index): #Devuelve  una  sola  fila  del  conjunto  de  datos
        return self.input_ids[index], self.labels_ids[index]
def create_dataloader_v1(txt, batch_size=4, max_length=256,
        stride=128, shuffle=True, drop_last=True, num_workers=0):
    tokenizer = tiktoken.get_encoding("gpt2")         #inicializar toenizer          
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)    #Crear dataset
    dataloader = DataLoader( 
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,                                  
        num_workers=num_workers                                   
    )
    return dataloader