import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

rse = pd.read_csv('./log/run-.-tag-rse.csv', sep=',', header=0)
ssim = pd.read_csv('./log/run-.-tag-ssim.csv', sep=',', header=0)
print(rse, ssim)
rse_step0 = rse['Step'].to_numpy()[:350]
rse_step1 = rse['Step'].to_numpy()[380:720]
rse_step2 = rse['Step'].to_numpy()[780:]
rse_step = np.concatenate([rse_step0, rse_step1, rse_step2], axis=0)
rse_value0 = rse['Value'].to_numpy()[:350]
rse_value1 = rse['Value'].to_numpy()[380:720]
rse_value2 = rse['Value'].to_numpy()[780:]
rse_value = np.concatenate([rse_value0, rse_value1, rse_value2], axis=0)
ssim_step0 = ssim['Step'].to_numpy()[:350]
ssim_step1 = ssim['Step'].to_numpy()[380:720]
ssim_step2 = ssim['Step'].to_numpy()[780:]
ssim_step = np.concatenate([ssim_step0, ssim_step1, ssim_step2], axis=0)
ssim_value0 = ssim['Value'].to_numpy()[:350]
ssim_value1 = ssim['Value'].to_numpy()[380:720]
ssim_value2 = ssim['Value'].to_numpy()[780:]
ssim_value = np.concatenate([ssim_value0, ssim_value1, ssim_value2], axis=0)
fig, ax = plt.subplots()
ax.set_ylim((0., 1.))
ax.set_xlabel('Epochs')
ax.set_ylabel('Value')
if rse_step.shape == rse_value.shape:
  ax.plot(rse_step, rse_value, color='r', label='rmse')  
else:
  print(rse_step.shape, rse_value.shape)
  
if ssim_step.shape == ssim_step.shape:
  ax.plot(ssim_step, ssim_value, color='g', label='ssim')
else:
  print(ssim_step.shape, ssim_value.shape)
  
fig.legend()
plt.show()