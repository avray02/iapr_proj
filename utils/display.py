import numpy as np 
import matplotlib.pyplot as plt 
import cv2

def plot_images(images, coins=[], est_coins=[], coins_coord=[], types=[], n_cols=-1, x_size=3, y_size=3, ratio=1, cmap='gray'):  
    # images = images.copy()
    # coins = coins.copy()
    # est_coins = est_coins.copy()
    # coins_coord = coins_coord.copy()

    estimation = False
    legend = False
    localization = False
    title = False

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

    if len(coins_coord) > 0:
        if len(images) != len(coins_coord):
            raise ValueError('images and coins_coord must have the same length')
        localization = True

    if len(types) > 0:
        if len(images) != len(types):
            raise ValueError('images and types must have the same length')
        title = True

    if n_cols == -1:
        n_cols = np.min([len(images), 6])

    labels = ['5CHF', '2CHF', '1CHF', '0.5CHF', '0.2CHF', '0.1CHF', '0.05CHF',
       '2EUR', '1EUR', '0.5EUR', '0.2EUR', '0.1EUR', '0.05EUR', '0.02EUR',
       '0.01EUR', 'OOD']
    
    n_images = len(images)
    n_rows = int(np.ceil(n_images/n_cols))

    _, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * x_size, n_rows * y_size))

    for i, (image, coin) in enumerate(zip(images, coins)):
        N,M = image.shape[:2]
        # img = cv2.resize(image, (int(M*ratio), int(N*ratio))).copy()

        row = i // n_cols
        col = i % n_cols
        if n_rows > 1 and n_cols > 1:
            ax = axs[row, col]
        elif n_rows > 1:
            ax = axs[row]
        elif n_cols > 1:
            ax = axs[col]
        else:
            ax = axs

        ax.imshow(image, cmap=cmap)
        ax.axis('off')

        if localization:
            for circle in coins_coord[i]:
                x, y, radius = circle

                center = (int(x), int(y))
                circle = plt.Circle(center, radius, color='r', fill=False)  # Create a circle object
                ax.add_artist(circle)  # Add the circle to the plot

        if legend:
            error = False
            text = ''
            for j in range(len(labels)):
                if estimation and ((coins[i][j] != 0 or est_coins[i][j] != 0) and est_coins[i][j] != coins[i][j]):  
                    error = True
                    text += f'({est_coins[i][j]}){coins[i][j]}x{labels[j]}\n' 
                elif coins[i][j] != 0:
                    text += f'{coins[i][j]}x{labels[j]}\n'       


            ax.text(0, 0, text, color='white', fontsize=10, va='top', ha='left', bbox=dict(facecolor='black', alpha=0.2))

        if title:
            ax.set_title(f'Type: {types[i]}')

    plt.show()

def plot_coins(coins, labels=[], predicted_labels=[], x_size=3, y_size=3):
    title = False
    prediction = False

    label_list = ['5CHF', '2CHF', '1CHF', '0.5CHF', '0.2CHF', '0.1CHF', '0.05CHF',
       '2EUR', '1EUR', '0.5EUR', '0.2EUR', '0.1EUR', '0.05EUR', '0.02EUR',
       '0.01EUR', 'OOD']

    if len(labels) > 0:
        if len(labels) != len(coins):
            raise ValueError('coins and labels must have the same length') 
        title = True
        label_type = type(labels[0])

        if len(predicted_labels) > 0:
            if len(predicted_labels) != len(coins):
                raise ValueError('coins and predicted_labels must have the same length') 
            prediction = True

    

    n = len(coins)

    n_rows = ((n-1)//6)+1
    n_cols = min(n,6)

    fig,axs = plt.subplots(n_rows,n_cols,figsize=(n_cols * x_size, n_rows * y_size))

    for i in range(n):
        row = i//n_cols
        col = i%n_cols
        if n_rows>1 and n_cols>1:
            ax = axs[row,col]
        elif n_rows>1:
            ax = axs[row]
        elif n_cols>1:
            ax = axs[col]
        else:
            ax = axs

        ax.imshow(coins[i], cmap='gray')
        ax.axis('off')

        if title:
            if label_type == int:
                if prediction:
                    if labels[i] == predicted_labels[i]:
                        ax.set_title(f'{label_list[labels[i]]}', color='green')
                    else:
                        ax.set_title(f'{label_list[predicted_labels[i]]} ({label_list[labels[i]]})', color='red')
            else:
                if prediction:
                    if labels[i] == predicted_labels[i]:
                        ax.set_title(f'{labels[i]}', color='green')
                    else:
                        ax.set_title(f'{predicted_labels[i]} ({labels[i]})', color='red')
                else:
                    ax.set_title(f'{labels[i]}')
        



        

    plt.show()

    