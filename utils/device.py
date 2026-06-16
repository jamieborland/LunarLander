import torch

def get_device(): 
    device = torch.device("cpu")   
    #if torch.backends.mps.is_available():
        #device = torch.device("mps")
    #elif torch.cuda.is_available():
     #   device = torch.device("cuda")
    #else:
     #   device = torch.device("cpu")
    print(f"Using device: {device}")
    return device
