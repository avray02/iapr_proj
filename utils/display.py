import numpy as np 
import matplotlib.pyplot as plt 

def plot_images(images, coins=[], est_coins=[], n_cols=-1, x_size=3, y_size=3, cmap='gray'):  
    estimation = False
    legend = False

    # Sanity check
    if len(images) == 0:
        raise ValueError('images must not be empty')
    
    if len(coins) > 0:
        if len(images) != len(coins):
            raise ValueError('images and coins must have the same length') 
        legend = True
    else:
        coins = np.zeros((len(images), 16), dtype=int)
    
    if len(est_coins) > 0:
        if len(images) != len(est_coins):
            raise ValueError('images and est_coins must have the same length')
        estimation = True

    if n_cols == -1:
        n_cols = np.min([len(images), 6])

    labels = ['5CHF', '2CHF', '1CHF', '0.5CHF', '0.2CHF', '0.1CHF', '0.05CHF',
       '2EUR', '1EUR', '0.5EUR', '0.2EUR', '0.1EUR', '0.05EUR', '0.02EUR',
       '0.01EUR', 'OOD']
    
    n_images = len(images)
    n_rows = int(np.ceil(n_images/n_cols))

    _, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * x_size, n_rows * y_size))

    for i, (image, coin) in enumerate(zip(images, coins)):
        row = i // n_cols
        col = i % n_cols
        if n_rows > 1:
            ax = axs[row, col]
        else:
            ax = axs[col]

        ax.imshow(image, cmap=cmap)
        ax.axis('off')

        if not legend:
            continue

        error = False
        text = ''
        for j in range(len(labels)):
            if estimation and ((coins[i][j] != 0 or est_coins[i][j] != 0) and est_coins[i][j] != coins[i][j]):  
                error = True
                text += f'({est_coins[i][j]}){coins[i][j]}x{labels[j]}\n' 
            elif coins[i][j] != 0:
                text += f'{coins[i][j]}x{labels[j]}\n'


        ax.text(0, 0, text, color='white', fontsize=10, va='top', ha='left', bbox=dict(facecolor='black', alpha=0.2))

    plt.show()

    