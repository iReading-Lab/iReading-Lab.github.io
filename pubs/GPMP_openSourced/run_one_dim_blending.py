# plot blended GP_MP
import numpy as np
from algorithm.GP_MP import GpMp
from algorithm.GP_MP import BlendedGpMp
import matplotlib.pyplot as plt
np.random.seed(3)
font_size = 25
font1 = {'family' : 'Times new Roman',
'weight' : 'normal',
'size'   : font_size,
}

size_set = 10
x1 = np.random.uniform(0, 1, size_set)
y1 = np.sin(2 * np.pi * x1) + np.cos(2 * np.pi * x1) + np.random.normal(0, 0.8, size_set)
size_via_points = 2
x1_ = np.random.uniform(0.5, 1.0, size_via_points)
y1_ = np.sin(2 * np.pi * x1_) + np.cos(2 * np.pi * x1_)
gp_mp1 = GpMp(x1.reshape(-1, 1), y1, x1_.reshape(-1, 1), y1_, observation_noise=0.4)
print('GP_MP1 training...')
gp_mp1.train()

x2 = np.random.uniform(0, 1, size_set)
y2 = 2 * np.sin(2 * np.pi * x2) + 1 * np.cos(2 * np.pi * x2) + np.random.normal(0, 0.8, size_set)
size_via_points = 2
x2_ = np.random.uniform(0, 0.5, size_via_points)
y2_ = 2 * np.sin(2 * np.pi * x2_) + 1 * np.cos(2 * np.pi * x2_)
gp_mp2 = GpMp(x2.reshape(-1, 1), y2, x2_.reshape(-1, 1), y2_, observation_noise=0.4)
print('GP_MP2 training...')
gp_mp2.train()

blended_gpmp = BlendedGpMp([gp_mp1, gp_mp2])

test_x = np.arange(0, 1, 0.01)
mean1, var1 = gp_mp1.predict_determined_input(test_x.reshape(-1, 1))
mean1 = mean1.reshape(-1)
var1 = var1.reshape(-1)

mean2, var2 = gp_mp2.predict_determined_input(test_x.reshape(-1, 1))
mean2 = mean2.reshape(-1)
var2 = var2.reshape(-1)

alpha_list = (np.tanh((test_x - 0.5) * 5) + 1.0) / 2
alpha_list = np.vstack((alpha_list, 1 - alpha_list))
# alpha_list = np.vstack((np.ones(np.shape(test_x)[0]), np.ones(np.shape(test_x)[0])))
mean_blended, var_blended = blended_gpmp.predict_determined_input(test_x.reshape(-1, 1), alpha_list)

plt.figure(figsize=(16, 8), dpi=100)
plt.subplots_adjust(left=0.05, right=0.99, wspace=0.8, hspace=0.8, bottom=0.1, top=0.99)
plt1 = plt.subplot2grid((8, 8), (0, 0), rowspan=4, colspan=4)
size = 30
plt1.scatter(x1, y1, c='grey', marker='x', s=np.ones(size_set) * size)
plt1.scatter(x1_, y1_, c='grey', marker='o', s=np.ones(size_via_points) * size)
plt1.scatter(x2, y2, c='blue', marker='x', s=np.ones(size_set) * size)
plt1.scatter(x2_, y2_, c='blue', marker='o', s=np.ones(size_via_points) * size)

linewidth = 3
alpha = 0.3
plt1.plot(test_x, mean1, c='grey', label='GP-MP1', linewidth=linewidth)
plt1.fill_between(test_x, mean1 - 2 * var1, mean1 + 2 * var1, color='grey', alpha=alpha)

plt1.plot(test_x, mean2, c='blue', label='GP-MP2', linewidth=linewidth)
plt1.fill_between(test_x, mean2 - 2 * var2, mean2 + 2 * var2, color='blue', alpha=alpha)

plt1.plot(test_x, mean_blended, c='red', label='Blended GP-MP')
plt1.fill_between(test_x, mean_blended - 2 * var_blended, mean_blended + 2 * var_blended, color='red', alpha=0.5)

plt1.legend(loc='upper right', prop=font1, frameon=False, handlelength=1, ncol=2, columnspacing=1)
plt1.tick_params(labelsize=font_size)


plt2 = plt.subplot2grid((8, 8), (4, 0), rowspan=4, colspan=4)
length = np.shape(test_x)[0]
plt2.plot(test_x, alpha_list[0, :], c='grey', label='$\\alpha_1$', linewidth=linewidth)
plt2.plot(test_x, alpha_list[1, :], c='blue', label='$\\alpha_2$', linewidth=linewidth)
plt2.legend(loc='center right', prop=font1, frameon=False, handlelength=1, ncol=2, columnspacing=1)
plt2.tick_params(labelsize=font_size)
plt2.set_xlabel('(a): Blending case 1', fontsize=font_size, fontname='Times New Roman')


alpha_list = np.vstack((np.ones(np.shape(test_x)[0]), np.ones(np.shape(test_x)[0])))
mean_blended, var_blended = blended_gpmp.predict_determined_input(test_x.reshape(-1, 1), alpha_list)
plt3 = plt.subplot2grid((8, 8), (0, 4), rowspan=4, colspan=4)
size = 30
plt3.scatter(x1, y1, c='grey', marker='x', s=np.ones(size_set) * size)
plt3.scatter(x1_, y1_, c='grey', marker='o', s=np.ones(size_via_points) * size)
plt3.scatter(x2, y2, c='blue', marker='x', s=np.ones(size_set) * size)
plt3.scatter(x2_, y2_, c='blue', marker='o', s=np.ones(size_via_points) * size)
plt3.plot(test_x, mean1, c='grey', label='GP-MP1', linewidth=linewidth)
plt3.fill_between(test_x, mean1 - 2 * var1, mean1 + 2 * var1, color='grey', alpha=alpha)
plt3.plot(test_x, mean2, c='blue', label='GP-MP2', linewidth=linewidth)
plt3.fill_between(test_x, mean2 - 2 * var2, mean2 + 2 * var2, color='blue', alpha=alpha)
plt3.plot(test_x, mean_blended, c='red', label='Blended GP-MP')
plt3.fill_between(test_x, mean_blended - 2 * var_blended, mean_blended + 2 * var_blended, color='red', alpha=0.5)
plt3.legend(loc='upper right', prop=font1, frameon=False, handlelength=1, ncol=2, columnspacing=1)
plt3.tick_params(labelsize=font_size)

plt4 = plt.subplot2grid((8, 8), (4, 4), rowspan=4, colspan=4)
length = np.shape(test_x)[0]
plt4.plot(test_x, alpha_list[0, :], c='grey', label='$\\alpha_1$', linewidth=linewidth)
plt4.plot(test_x, alpha_list[1, :], c='blue', label='$\\alpha_2$', linewidth=linewidth)
plt4.legend(loc='upper right', prop=font1, frameon=False, handlelength=1, ncol=2, columnspacing=1)
plt4.set_xlabel('(b): Blending case 2', fontsize=font_size, fontname='Times New Roman')
plt4.tick_params(labelsize=font_size)

plt.show()