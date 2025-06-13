from numpy import *
import matplotlib.pyplot as plt
from matplotlib.widgets import RangeSlider
import cupy as cp


fullField = load("ETOT1.npy")

ampImage = abs(fullField)
angleImage = angle(fullField)

ampBins = linspace(0, max(ampImage), 100)
angleBins = linspace(-pi, pi, 50)

fig, axs = plt.subplots(1, 3, figsize=(10, 5))
img = axs[0].imshow(ampImage)
sliderX_ax = fig.add_axes([0.20, 0.1, 0.60, 0.03])
sliderY_ax = fig.add_axes([0.20, 0.15, 0.60, 0.03])
sliderX = RangeSlider(sliderX_ax, "X-Threshold", 0, ampImage.shape[0])
sliderY = RangeSlider(sliderY_ax, "Y-Threshold", 0, ampImage.shape[1])
Xlower_limit_line = axs[0].axvline(sliderX.val[0], color='k')
Xupper_limit_line = axs[0].axvline(sliderX.val[1], color='k')

Ylower_limit_line = axs[0].axhline(sliderY.val[0], color='k')
Yupper_limit_line = axs[0].axhline(sliderY.val[1], color='k')

fig.subplots_adjust(bottom=0.25)

axs[1].hist(ampImage.flatten(), bins=ampBins)
axs[2].hist(angleImage.flatten(), bins=angleBins)

axs[2].set_xlim(-pi, pi)


def update(val):

    Xlower_limit_line.set_xdata([sliderX.val[0], sliderX.val[0]])
    Xupper_limit_line.set_xdata([sliderX.val[1], sliderX.val[1]])
    xLower = int(sliderX.val[0])
    xUpper = int(sliderX.val[1])

    Ylower_limit_line.set_ydata([sliderY.val[0], sliderY.val[0]])
    Yupper_limit_line.set_ydata([sliderY.val[1], sliderY.val[1]])
    yLower = int(sliderY.val[0])
    yUpper = int(sliderY.val[1])

    axs[1].cla()
    ampBins =  int(len(ampImage[yLower:yUpper, xLower:xUpper].flatten()) / sqrt(200*200))
    axs[1].hist(ampImage[yLower:yUpper, xLower:xUpper].flatten(), density=True, bins=ampBins)

    axs[2].cla()
    axs[2].set_xlim(-pi, pi)
    angleBins =  int(len(angleImage[yLower:yUpper, xLower:xUpper].flatten()) / sqrt(200*200))
    axs[2].hist(angleImage[yLower:yUpper, xLower:xUpper].flatten(), density=True, bins=angleBins)


    axs[1].relim()
    axs[1].autoscale_view()
    axs[2].relim()
    axs[2].autoscale_view()


sliderX.on_changed(update)
sliderY.on_changed(update)
plt.show()



