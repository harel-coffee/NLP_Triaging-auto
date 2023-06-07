"""
Example file for experiments
"""
import rf as randomforest
import knnEA as ea
import nn as neuralnetwork

if __name__ == '__main__':
    print('Choose the example you want to run. Type "help" for available examples.')
    while True:
        command = str(input("Waiting for input: "))
        if command == 'help':
            print('Type "ea" for running an evolutionary algorithm example')
            print('Type "rf" for running a random forest example')
            print('Type "nn" for running a neural network hypertuning example')
            print('Type "exit" for terminating the program')
        elif command == 'ea':
            ea.example(1)
            print('Finished!')
        elif command == 'rf':
            randomforest.example(1)
            print('Finished!')
        elif command == 'nn':
            neuralnetwork.example()
            print('Finished!')
        elif command == 'exit':
            exit()
        else:
            print('Command not understood. Please use "help" for available commands.')
