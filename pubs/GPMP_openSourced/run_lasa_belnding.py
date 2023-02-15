import matplotlib.pyplot as plt
import pyLasaDataset as lasa
from algorithm.GP_MP import GpMp
import numpy as np
from algorithm.GP_MP import BlendedGpMp

np.random.seed(3)
font_size = 18
font1 = {'family' : 'Times new Roman',
'weight' : 'normal',
'size'   : font_size,
}
# using multi_models 3, or you will change the loading-data code
data = lasa.DataSet.Multi_Models_3
dt = data.dt
print(dt)
demos = data.demos
gap = 30

# --------------------------------------------- loading and training the model1 data------------------------------------
# loading demo 0 data
demo_0 = demos[0]
pos0 = demo_0.pos[:, 0::gap]  # np.ndarray, shape: (2,1000/gap)
vel0 = demo_0.vel[:, 0::gap]  # np.ndarray, shape: (2,1000/gap)
acc0 = demo_0.acc[:, 0::gap]  # np.ndarray, shape: (2,1000/gap)
t0 = demo_0.t[:, 0::gap]  # np.ndarray, shape: (1,1000/gap)
X_0 = t0.T
Y_0 = pos0.T

# loading demo 1 data
demo_1 = demos[1]
pos1 = demo_1.pos[:, 0::gap]
vel1 = demo_1.vel[:, 0::gap]
acc1 = demo_1.acc[:, 0::gap]
t1 = demo_0.t[:, 0::gap]
X_1 = t1.T
Y_1 = pos1.T

# loading demo 2 data
demo_2 = demos[2]
pos2 = demo_2.pos[:, 0::gap]
vel2 = demo_2.vel[:, 0::gap]
acc2 = demo_2.acc[:, 0::gap]
t2 = demo_2.t[:, 0::gap]
X_2 = t2.T
Y_2 = pos2.T

# constructing training set
X = np.vstack((X_0, X_1, X_2))  # np.ndarray, shape: (3 * 1000/gap, 1)
Y = np.vstack((Y_0, Y_1, Y_2))  # np.ndarray, shape: (3 * 1000/gap, 2)
target_t = ((demo_0.t + demo_1.t + demo_2.t) / 3)[0, -1]
target_position = ((demo_0.pos + demo_1.pos + demo_2.pos) / 3)[:, -1]

via_point0_t = ((t0 + t1 + t2) / 3)[0, 0]
via_point0_position = ((pos0 + pos1 + pos2) / 3)[:, 0]
middle = np.shape(pos0)[1] * 2 // 4
via_point1_t = ((t0 + t1 + t2) / 3)[0, middle]
via_point1_position = ((pos0 + pos1 + pos2) / 3)[:, middle]

X_ = np.array([via_point0_t, via_point1_t, target_t]).reshape(-1, 1)
Y_ = np.array([via_point0_position, via_point1_position, target_position])
via_point1_t_model1 = via_point1_t
# predicting for dim0
observation_noise = 1.0
gp_mp1_dim0 = GpMp(X, Y[:, 0], X_, Y_[:, 0], observation_noise=observation_noise)
gp_mp1_dim0.train()
test_x = np.arange(0.0, target_t, dt)
gp_mp1_predict_y_dim0, gp_mp1_predict_var_dim0 = gp_mp1_dim0.predict_determined_input(test_x.reshape(-1, 1))
gp_mp1_predict_y_dim0 = gp_mp1_predict_y_dim0.reshape(-1)
gp_mp1_predict_var_dim0 = gp_mp1_predict_var_dim0.reshape(-1)

# predicting for dim1
gp_mp1_dim1 = GpMp(X, Y[:, 1], X_, Y_[:, 1], observation_noise=observation_noise)
gp_mp1_dim1.train()
test_x = np.arange(0.0, target_t, dt)
gp_mp1_predict_y_dim1, gp_mp1_predict_var_dim1 = gp_mp1_dim1.predict_determined_input(test_x.reshape(-1, 1))
gp_mp1_predict_y_dim1 = gp_mp1_predict_y_dim1.reshape(-1)
gp_mp1_predict_var_dim1 = gp_mp1_predict_var_dim1.reshape(-1)

# --------------------------------------------- loading and training the model2 data------------------------------------
# loading demo 4 data
demo_4 = demos[4]
pos4 = demo_4.pos[:, 0::gap]  # np.ndarray, shape: (2,1000/gap)
vel4 = demo_4.vel[:, 0::gap]  # np.ndarray, shape: (2,1000/gap)
acc4 = demo_4.acc[:, 0::gap]  # np.ndarray, shape: (2,1000/gap)
t4 = demo_4.t[:, 0::gap]  # np.ndarray, shape: (1,1000/gap)
X_4 = t4.T
Y_4 = pos4.T

# loading demo 5 data
demo_5 = demos[5]
pos5 = demo_5.pos[:, 0::gap]
vel5 = demo_5.vel[:, 0::gap]
acc5 = demo_5.acc[:, 0::gap]
t5 = demo_5.t[:, 0::gap]
X_5 = t5.T
Y_5 = pos5.T

# loading demo 6 data
demo_6 = demos[6]
pos6 = demo_6.pos[:, 0::gap]
vel6 = demo_6.vel[:, 0::gap]
acc6 = demo_6.acc[:, 0::gap]
t6 = demo_6.t[:, 0::gap]
X_6 = t6.T
Y_6 = pos6.T

# constructing training set
X_model2 = np.vstack((X_4, X_5, X_6))  # np.ndarray, shape: (3 * 1000/gap, 1)
Y_model2 = np.vstack((Y_4, Y_5, Y_6))  # np.ndarray, shape: (3 * 1000/gap, 2)
target_t = ((demo_4.t + demo_5.t + demo_6.t) / 3)[0, -1]
target_position = ((demo_4.pos + demo_5.pos + demo_6.pos) / 3)[:, -1]

via_point0_t = ((t4 + t5 + t6) / 3)[0, 0]
via_point0_position = ((pos4 + pos5 + pos6) / 3)[:, 0]
middle = np.shape(pos4)[1] * 2 // 4
via_point1_t = ((t4 + t5 + t6) / 3)[0, middle]
via_point1_position = ((pos4 + pos5 + pos6) / 3)[:, middle]


X_model2_ = np.array([via_point0_t, target_t]).reshape(-1, 1)
Y_model2_ = np.array([via_point0_position, target_position])
# predicting for dim0
gp_mp2_dim0 = GpMp(X_model2, Y_model2[:, 0], X_model2_, Y_model2_[:, 0], observation_noise=observation_noise)
gp_mp2_dim0.train()
test_x_model2 = np.arange(0.0, target_t, dt)
gp_mp2_predict_y_dim0, gp_mp2_predict_var_dim0 = gp_mp2_dim0.predict_determined_input(test_x_model2.reshape(-1, 1))
gp_mp2_predict_y_dim0 = gp_mp2_predict_y_dim0.reshape(-1)
gp_mp2_predict_var_dim0 = gp_mp2_predict_var_dim0.reshape(-1)

# predicting for dim1
gp_mp2_dim1 = GpMp(X_model2, Y_model2[:, 1], X_model2_, Y_model2_[:, 1], observation_noise=observation_noise)
gp_mp2_dim1.train()
test_x_model2 = np.arange(0.0, target_t, dt)
gp_mp2_predict_y_dim1, gp_mp2_predict_var_dim1 = gp_mp2_dim1.predict_determined_input(test_x_model2.reshape(-1, 1))
gp_mp2_predict_y_dim1 = gp_mp2_predict_y_dim1.reshape(-1)
gp_mp2_predict_var_dim1 = gp_mp2_predict_var_dim1.reshape(-1)

#  -------------------------------------------------------blending-----------------------------------------------------
blended_gpmp_dim0 = BlendedGpMp([gp_mp1_dim0, gp_mp2_dim0])
blended_gpmp_dim1 = BlendedGpMp([gp_mp1_dim1, gp_mp2_dim1])
test_x_model2_length = np.shape(test_x_model2)[0]
index = 0
for i in range(test_x_model2_length):
    if test_x_model2[i] > via_point1_t_model1:
        index = i
        break
alpha_model1 = np.ones(test_x_model2_length)
model12model2_length = test_x_model2_length - 1 - index
mu_dim0_list = np.empty(test_x_model2_length)
var_dim0_list = np.empty(test_x_model2_length)
mu_dim1_list = np.empty(test_x_model2_length)
var_dim1_list = np.empty(test_x_model2_length)
alpha_list = np.empty(test_x_model2_length)
for i in range(test_x_model2_length):
    if i <= index:
        alpha = 1
    else:
        alpha = 1 - np.tanh((i - index - 1) / model12model2_length * 5)
    alpha_list[i] = alpha
    mu_dim0, var_dim0 = blended_gpmp_dim0.predict_single_determined_input(test_x_model2[i], np.array([alpha, 1 - alpha]))
    mu_dim1, var_dim1 = blended_gpmp_dim1.predict_single_determined_input(test_x_model2[i], np.array([alpha, 1 - alpha]))
    mu_dim0_list[i] = mu_dim0
    var_dim0_list[i] = var_dim0
    mu_dim1_list[i] = mu_dim1
    var_dim1_list[i] = var_dim1

#  -------------------------------------------------------plotting-----------------------------------------------------
alpha = 0.3
variance_factor = 5
plt.figure(figsize=(16, 6), dpi=100)
plt.subplots_adjust(left=0.06, right=0.99, wspace=7.8, hspace=10.8, bottom=0.17, top=0.99)
plt1 = plt.subplot2grid((8, 16), (0, 0), rowspan=8, colspan=8)
plt1.plot(gp_mp1_predict_y_dim0, gp_mp1_predict_y_dim1, ls='-', c='blue', linewidth=2, label='$p_{gpmp1}$')
plt1.plot(gp_mp2_predict_y_dim0, gp_mp2_predict_y_dim1, ls='-', c='red', linewidth=2, label='$p_{gpmp2}$')
plt1.plot(mu_dim0_list, mu_dim1_list, ls='-', c='grey', linewidth=2, label='$p_{blended}$')

plt1.scatter(Y_[:, 0], Y_[:, 1], s=200, c='blue', marker='x')
plt1.scatter(Y[:, 0], Y[:, 1], s=10, c='blue', marker='o', alpha=alpha)
plt1.scatter(Y_model2_[:, 0], Y_model2_[:, 1], s=200, c='red', marker='x')
plt1.scatter(Y_model2[:, 0], Y_model2[:, 1], s=10, c='red', marker='o', alpha=alpha)

plt1.legend(loc='upper left', prop=font1, frameon=False, handlelength=1, ncol=3, columnspacing=1)
plt1.tick_params(labelsize=font_size)
plt1.set_xlabel('$x$/mm\n(a)', fontname='Times New Roman', fontsize=font_size)
plt1.set_ylabel('$y$/mm', fontname='Times New Roman', fontsize=font_size)

plt2 = plt.subplot2grid((8, 16), (0, 8), rowspan=3, colspan=8)
plt2.plot(test_x, gp_mp1_predict_y_dim0, c='blue', linewidth=2, label='$x_{gpmp1}$')
plt2.fill_between(test_x, gp_mp1_predict_y_dim0 - variance_factor * np.sqrt(gp_mp1_predict_var_dim0), gp_mp1_predict_y_dim0 + variance_factor * np.sqrt(gp_mp1_predict_var_dim0), color='blue', alpha=alpha)
plt2.scatter(X_[:, 0], Y_[:, 0], s=200, c='blue', marker='x')
plt2.scatter(X[:, 0], Y[:, 0], s=10, c='blue', marker='o', alpha=alpha)
plt2.plot(test_x_model2, gp_mp2_predict_y_dim0, c='red', linewidth=2, label='$x_{gpmp2}$')
plt2.fill_between(test_x_model2, gp_mp2_predict_y_dim0 - variance_factor * np.sqrt(gp_mp2_predict_var_dim0), gp_mp2_predict_y_dim0 + variance_factor * np.sqrt(gp_mp2_predict_var_dim0), color='red', alpha=alpha)
plt2.scatter(X_model2_[:, 0], Y_model2_[:, 0], s=200, c='red', marker='x')
plt2.scatter(X_model2[:, 0], Y_model2[:, 0], s=10, c='red', marker='o', alpha=alpha)
plt2.plot(test_x_model2, mu_dim0_list, c='grey', linewidth=2, label='$x_{blended}$')
plt2.fill_between(test_x_model2, mu_dim0_list - variance_factor * np.sqrt(var_dim0_list), mu_dim0_list + variance_factor * np.sqrt(var_dim0_list), color='grey', alpha=alpha)
plt2.legend(loc='upper left', prop=font1, frameon=False, handlelength=1, ncol=3, columnspacing=1)
plt2.tick_params(labelsize=font_size)
plt2.set_xlabel('(b)', fontname='Times New Roman', fontsize=font_size)
plt2.set_ylabel('$x$/mm', fontname='Times New Roman', fontsize=font_size)

plt3 = plt.subplot2grid((8, 16), (3, 8), rowspan=3, colspan=8)
plt3.plot(test_x, gp_mp1_predict_y_dim1, c='blue', linewidth=2, label='$y_{gpmp1}$')
plt3.fill_between(test_x, gp_mp1_predict_y_dim1 - variance_factor * np.sqrt(gp_mp1_predict_var_dim1), gp_mp1_predict_y_dim1 + variance_factor * np.sqrt(gp_mp1_predict_var_dim1), color='blue', alpha=alpha)
plt3.scatter(X_[:, 0], Y_[:, 1], s=200, c='blue', marker='x')
plt3.scatter(X[:, 0], Y[:, 1], s=10, c='blue', marker='o', alpha=alpha)
plt3.plot(test_x_model2, gp_mp2_predict_y_dim1, c='red', linewidth=2, label='$y_{gpmp2}$')
plt3.fill_between(test_x_model2, gp_mp2_predict_y_dim1 - variance_factor * np.sqrt(gp_mp2_predict_var_dim1), gp_mp2_predict_y_dim1 + variance_factor * np.sqrt(gp_mp2_predict_var_dim1), color='red', alpha=alpha)
plt3.scatter(X_model2_[:, 0], Y_model2_[:, 1], s=200, c='red', marker='x')
plt3.scatter(X_model2[:, 0], Y_model2[:, 1], s=10, c='red', marker='o', alpha=alpha)
plt3.plot(test_x_model2, mu_dim1_list, c='grey', linewidth=2, label='$y_{blended}$')
plt3.fill_between(test_x_model2, mu_dim1_list - variance_factor * np.sqrt(var_dim1_list), mu_dim1_list + variance_factor * np.sqrt(var_dim1_list), color='grey', alpha=alpha)
plt3.legend(loc='upper left', prop=font1, frameon=False, handlelength=1, ncol=3, columnspacing=1)
plt3.tick_params(labelsize=font_size)
plt3.set_xlabel('(c)', fontname='Times New Roman', fontsize=font_size)
plt3.set_ylabel('$y$/mm', fontname='Times New Roman', fontsize=font_size)

plt4 = plt.subplot2grid((8, 16), (6, 8), rowspan=2, colspan=8)
plt4.plot(test_x_model2, alpha_list, linewidth=2, label='$\\alpha_1$', c='blue')
plt4.plot(test_x_model2, 1 - alpha_list, linewidth=2, label='$\\alpha_2$', c='red')
plt4.legend(loc='upper left', prop=font1, frameon=False, handlelength=1, ncol=2, columnspacing=1)
plt4.tick_params(labelsize=font_size)
plt4.set_ylabel('$\\alpha$', fontname='Times New Roman', fontsize=font_size)
plt4.set_xlabel('time/s\n(d)', fontname='Times New Roman', fontsize=font_size)
plt.show()

gp_mp1_predict_y_dim0_training_data, _ = gp_mp1_dim0.predict_determined_input(X.reshape(-1, 1))
gp_mp1_predict_y_dim1_training_data, _ = gp_mp1_dim1.predict_determined_input(X.reshape(-1, 1))
gp_mp2_predict_y_dim0_training_data, _ = gp_mp2_dim0.predict_determined_input(X_model2.reshape(-1, 1))
gp_mp2_predict_y_dim1_training_data, _ = gp_mp2_dim1.predict_determined_input(X_model2.reshape(-1, 1))
error_model1 = np.sqrt(np.average((gp_mp1_predict_y_dim0_training_data.reshape(-1) - Y[:, 0]) ** 2 + (gp_mp1_predict_y_dim1_training_data.reshape(-1) - Y[:, 1]) ** 2))
error_model2 = np.sqrt(np.average((gp_mp2_predict_y_dim0_training_data.reshape(-1) - Y_model2[:, 0]) ** 2 + (gp_mp2_predict_y_dim1_training_data.reshape(-1) - Y_model2[:, 1]) ** 2))


