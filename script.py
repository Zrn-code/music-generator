from main.generate import *
from main.train import *
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

            parser = argparse.ArgumentParser()

            # 添加命令行參數及其縮寫
            parser.add_argument('-e', '--epochs', type=int, default=50, help='設定 epochs 的數量')
            parser.add_argument('-b', '--batch', type=int, default=512, help='設定 batch 的大小')
            parser.add_argument('-p', '--preprocess', type=int, default=1, help='設定預處理選項')
            parser.add_argument('-st', '--save_times', type=int, default=10, help='設定每隔多少 epochs 儲存一次模型')

            # 解析命令行參數
            args = parser.parse_args()  

            train_network(args.epochs, args.batch, args.preprocess, args.save_times)
        elif op == 2:

            # 創建解析器對象
            parser = argparse.ArgumentParser()

            # 添加命令行參數及其縮寫
            parser.add_argument('-m', '--model_name', type=str, default='model-10.pt', help='設定模型名稱')
            parser.add_argument('-l', '--length', type=int, default=400, help='設定生成的長度')
            parser.add_argument('-n', '--mid_name', type=str, default='output.mid', help='設定生成的中間文件名稱')

            # 解析命令行參數
            args = parser.parse_args()

            # 調用 generate 函數，並將解析後的參數傳入
            generate(args.model_name, args.length, args.mid_name)
            generate(args.model_name, args.length, args.output_name)
        elif op == 3:
            break
        else:
            print("syntax Error")
        
          
    
