# PYTORCH
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import sys
sys.path.append("..")
from main.get_note import get_notes
from main.create_model import MusicDataset,create_network
import os
script_directory = os.path.dirname(os.path.abspath(__file__))

def train(model,epoch, dataloader_notes, dataloader_offsets, dataloader_durations,save_times):
    """ train the neural network """
   
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters())
    model.to(device)
    
    for epoch in range(epoch):
        running_loss = 0.0
        model.train()

        for (notes, notes_output), (offsets, offsets_output), (durations, durations_output) in zip(dataloader_notes, dataloader_offsets, dataloader_durations):
            notes = notes.to(device)
            offsets = offsets.to(device)
            durations = durations.to(device)
            notes_output = notes_output.squeeze(dim=1).to(device)
            offsets_output = offsets_output.squeeze(dim=1).to(device)
            durations_output = durations_output.squeeze(dim=1).to(device)
            notes_batch_output, offsets_batch_output, durations_batch_output = model(notes, offsets, durations)
            
            notes_loss = criterion(notes_batch_output,notes_output)
            offsets_loss = criterion(offsets_batch_output,offsets_output)
            durations_loss = criterion(durations_batch_output,durations_output)

            loss = notes_loss + offsets_loss + durations_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch+1} Loss: {running_loss / len(dataloader_notes)}")
        if (epoch+1) % save_times == 0:
            model_path = os.path.join(script_directory, f'../checkpoints/model-{epoch+1}.pt')
            torch.save(model.state_dict(), model_path)
            print(f"Checkpoint saved to {model_path}")
def train_network(epoch,batch,preprocess,save_times,dataset = "classical-piano-type0"):
    """ Train a Neural Network to generate music """
    notes, offsets, durations = get_notes(preprocess,dataset)

    n_vocab_notes = len(set(notes))
    n_vocab_offsets = len(set(offsets))
    n_vocab_durations = len(set(durations))

    dataset_notes = MusicDataset(notes, n_vocab_notes)
    dataset_offsets = MusicDataset(offsets, n_vocab_offsets)
    dataset_durations = MusicDataset(durations, n_vocab_durations)

    dataloader_notes = DataLoader(dataset_notes, batch_size=batch, shuffle=True)
    dataloader_offsets = DataLoader(dataset_offsets, batch_size=batch, shuffle=True)
    dataloader_durations = DataLoader(dataset_durations, batch_size=batch, shuffle=True)

    model = create_network(n_vocab_notes, n_vocab_offsets, n_vocab_durations)
    print("Start training")
    train(model,epoch, dataloader_notes, dataloader_offsets, dataloader_durations,save_times)

if __name__ == '__main__':
    # 創建解析器對象
    parser = argparse.ArgumentParser()

    # 添加命令行參數及其縮寫
    parser.add_argument('-e', '--epochs', type=int, default=50, help='設定 epochs 的數量')
    parser.add_argument('-b', '--batch', type=int, default=512, help='設定 batch 的大小')
    parser.add_argument('-p', '--preprocess', type=int, default=1, help='設定預處理選項')
    parser.add_argument('-st', '--save_times', type=int, default=10, help='設定每隔多少 epochs 儲存一次模型')
    parser.add_argument('-d', '--dataset', type=str, default="classical-piano-type0", help='設定資料集')
    
    # 解析命令行參數
    args = parser.parse_args()

    # 調用 train_network 函數，並將解析後的參數傳入
    train_network(args.epochs, args.batch, args.preprocess, args.save_times, args.dataset)
