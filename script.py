from main.generate import generate
from main.train import train_network
import os

if __name__ == '__main__':
    os.chdir(os.path.dirname(__file__))

    print("Welcome to the Music Generator!")
    print("Please choose an option:")

    print("1. Train a neural network")
    print("2. Generate music")
    print("3. Exit")
    
    while True:
        op = int(input())

        if op == 1:
            print("Please enter the number of epochs:")
            print("Enter 0 to use the default value (50)")
            epoch = int(input())
            epoch = 50 if epoch == 0 else epoch
            print("Please enter the batch size:")
            print("Enter 0 to use the default value (512)")
            batch = int(input())
            batch = 512 if batch == 0 else batch
            print("Please enter the number of times to save the model:")
            print("Enter 0 to use the default value (10)")
            save_times = int(input())
            save_times = 10 if save_times == 0 else save_times
            print("Do you want to parse the dataset?")
            print("If you have not parsed the dataset or you want to parse your own dataset, please enter y or Y")
            print("If you have already parsed the dataset or you want to use the default dataset, please enter n or N")
            while True:
                preprocess = input()
                if preprocess == "y" or preprocess == "Y":
                    preprocess = 1
                    break
                elif preprocess == "n" or preprocess == "N":
                    preprocess = 0
                    break
                else:
                    print("Please enter y or n.")
            
            if preprocess == 1:
                print("Please enter the dataset name:")
                print("Enter 0 to use the default dataset (classical-piano-type0)")
                dataset = input()
                dataset = "classical-piano-type0" if dataset == "0" else dataset
                train_network(epoch,batch,preprocess,save_times,dataset)
            else:
                train_network(epoch,batch,preprocess,save_times)
            print ("The neural network has been trained successfully!")
            print("Please choose an option:")
            print("1. Train a neural network")
            print("2. Generate music")
            print("3. Exit")
        elif op == 2:
            print("Please enter the model name:")
            print("Enter 0 to use the default model (pretrained_model.pt)")
            input_model = input()
            input_model = "pretrained_model.pt" if input_model == "0" else input_model
            print("Please enter the length of the music:")
            print("Enter 0 to use the default length (300)")
            length = int(input())
            length = 300 if length == 0 else length
            print("Please enter the output file name:")
            print("Enter 0 to use the default name (output.mid)")
            output_file = input()
            output_file = "output.mid" if output_file == "0" else output_file
            generate(input_model,length,output_file)
            print("The music has been generated successfully!")
            print("Please check the output file in the output folder.")
            print("Please choose an option:")
            print("1. Train a neural network")
            print("2. Generate music")
            print("3. Exit")
        elif op == 3:
            break
        else:
            print("Please enter 1, 2 or 3.")
        
          
    
