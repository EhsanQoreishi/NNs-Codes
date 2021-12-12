import numpy as np
import matplotlib.pyplot as plt

schemes = [
    '1-Combination Output', '2-dx output', '3-Frequency indicator data',
          '4-Integration', '5-LBP', '6-SSF output', '7-Summation of Columns'
          , '9-Voltage Interval output'
        ]

network = 'Fully Connected'
for sc in schemes:
    loss = np.load('Results/MLP/'+sc+'/loss.npy')
    val_loss = np.load('Results/MLP/'+sc+'/val_loss.npy')


    plt.plot(loss, '-b', label='mse-train')
    plt.plot(val_loss, 'orange', label='mse-val')
    plt.xlabel('epochs')
    plt.ylabel('mse')
    plt.title(network + ' && ' + sc[2:])
    plt.legend()
    plt.savefig('Results/MSE/mse curve-Fully/mse_curve-'+sc[2:7]+'.png')
    plt.show()