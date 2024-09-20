
import numpy as np
import cv2

### Strength: Fast, efficient and allows you to partially ignore annotations in the image
### Weakness: Smoothes signals and crushes peaks
def lazy_extraction(image_bin):
    # We define a starting pixel which corresponds to our anchor point - the extraction will start from this point
    # We look for all the lit pixels in the first column and average over them
    first_pixel_position = []
    for j in range(len(image_bin)):
        if image_bin[j,0] == 255:
            first_pixel_position.append(j)
    try:
        anchor = int(np.mean(first_pixel_position))
        signal = [anchor]
    except ValueError:
        anchor = int(len(image_bin)/2)
        signal = [anchor]

    # We then go through the image column by column, looking for the lit pixel closest to the anchor pixel.
    for i in range(1,len(image_bin[0])):
        # If we can stay at the same level as the anchor pixel, we do so
        if image_bin[anchor,i] == 255:
            signal.append(anchor)
        else:
            # Otherwise we look for the nearest lit pixel at the top and bottom, and as soon as we find one we stop and store it.
            # We search within a window of 1000 pixels to avoid searching too far.
            try:
                for j in range(1000):
                    if image_bin[anchor+j,i] == 255:
                        signal.append(anchor+j)
                        anchor = anchor+j
                        break
                    elif image_bin[anchor-j,i] == 255:
                        signal.append(anchor-j)
                        anchor = anchor-j
                        break
            except IndexError:
                signal.append(anchor)
    return signal


### Strengths: Fast, allows partial account to be taken of noise in the signal, low peak crushing
### Weakness: Also extracts pixels that are not part of the signal
def full_extraction(image_bin):
    # We look at all the columns in the image and average the position of the lit pixels
    extraction = np.array([sum(i for i, valeur in enumerate(ligne) if valeur == 255) / (ligne.count(255)+0.01) for ligne in image_bin.T.tolist()])
    return extraction



### Strength: Extracts only the signal, even if the labels are present.
### Weakness: Slower than other methods
def fragmented_extraction(image_bin):
    # Look at all the columns in the image and store the lit pixels. 
    # if there's a gap between two lit pixels, we store them in a new list
    signal = []
    for i in range(len(image_bin[0])):
        matrix = []
        sub_list = []
        begin = 0
        positions = np.where(image_bin[:,i] == 255)[0]
        if len(positions) == 0:
            it = 0
        else:
            it = positions[0]
        for j in (positions):
            if it == j:
                sub_list.append(j)
                it+=1
            elif it != j:
                matrix.append(sub_list)
                sub_list = []
                sub_list.append(j)
                it = j+1
        matrix.append(sub_list)
        # If there are several lists for a column, that means there are several groups of pixels.
        # The first group of pixels are always the letters and the last group is supposed to correspond to our signal.
        try:
            if len(matrix) > 1:
                signal.append(np.mean(matrix[-1]))
            elif len(matrix) == 1:
                signal.append(np.mean(matrix[0]))
            else:
                signal.append(signal[-1])
        except Exception as e:
            signal.append(len(image_bin)/2)
    return signal



