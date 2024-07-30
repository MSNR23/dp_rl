import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import csv
import time
import os

# 定数

ma = 70 # 全体質量  

g = 9.80  # 重力加速度 

# リンクの長さ
l1 = 1.0
l2 = 1.0

# 重心までの長さ
lg1 = l1 / 2
lg2 = l2 / 2

# 質点の質量
m1 = 0.016 * ma
m2 = 0.028 * ma + 7.0
# m3 = 7.0 # 投擲物の質量

# 慣性モーメント
I1 = m1 * l1**2 / 12
I2 = (m2 - 0.19) * l2**2 / 12 + 0.19 * (l2 / 2)**2

# 粘性係数
b1 = 0.01
b2 = 0.01

# 初期条件
q10 =  -np.pi / 8
q20 = 10 * np.pi / 180
# q20 = 1 * np.pi / 6
q1_dot0 = 0.0
q2_dot0 = 0.0

dt = 0.001

# CSVファイルの保存先ディレクトリ
save_dir = r'exp5'

# ディレクトリが存在しない場合は作成
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# 運動方程式
def update_world(q1, q2, q1_dot, q2_dot, tau, action):
    # 行動に基づくトルク[Nm]を設定
    tau = np.zeros((2, 1))
    if action == 0:
        tau = np.array([[20.0], [0.0]])
    elif action == 1:
        tau = np.array([[-20.0], [0.0]])
    elif action == 2:
        tau = np.array([[0.0], [20.0]])
    elif action == 3:
        tau = np.array([[0.0], [-20.0]])
    elif action == 4:
        tau = np.array([[20.0], [20.0]])
    elif action == 5:
        tau = np.array([[-20.0], [-20.0]])
    elif action == 6:
        tau = np.array([[20.0], [-20.0]])
    elif action == 7:
        tau = np.array([[-20.0], [20.0]])
    elif action == 8:
        tau = np.array([[0.0], [0.0]])

    # リンク2が可動範囲の限界に達した場合の外力
    if q2 <= 0:
        tau[1, 0] += 30.0  # 0度のとき、正の方向に5N
    elif q2 >= np.radians(150):
        tau[1, 0] += -30.0  # 145度のとき、負の方向に5N

    # 質量行列
    M_11 = m1*lg1**2 + I1 + m2*(l1**2 + lg2**2 + 2*l1*lg2*np.cos(q2)) + I2
    M_12 = m2 * (lg2**2 + l1 * lg2*np.cos(q2)) + I2
    M_21 = m2 * (lg2**2 + l1*lg2 * np.cos(q2)) + I2
    M_22 = m2 * lg2**2 + I2

    M = np.array([[M_11, M_12],
                  [M_21, M_22]])

    # コリオリ行列
    C_11 = -m2 * l1 * lg2 * np.sin(q2) * q2_dot * (2 * q1_dot + q2_dot)
    C_21 = m2 * l1 * lg2 * np.sin(q2) * q1_dot**2
    C = np.array([[C_11], [C_21]])

    # 重力ベクトル
    G_11 = m1 * g * lg1 * np.cos(q1) + m2 * g * (l1 * np.cos(q1) + lg2 * np.cos(q1 + q2))
    G_21 = m2 * g * lg2 * np.cos(q1 + q2)
    G = np.array([[G_11], [G_21]])

    # 粘性
    B_11 = b1 * q1_dot
    B_21 = b2 * q2_dot
    B = np.array([[B_11], [B_21]])

    # 逆行列
    M_inv = np.linalg.inv(M)

    q_ddot = M_inv.dot(tau - C - G - B)


    return np.array([q1_dot, q2_dot, q_ddot[0, 0], q_ddot[1, 0]])

# Runge-Kutta法
def runge_kutta(t, q1, q2, q1_dot, q2_dot, action, dt):
    tau = np.zeros((2, 1))

    k1 = dt * update_world(q1, q2, q1_dot, q2_dot, tau, action)
    k2 = dt * update_world(q1 + 0.5 * k1[0], q2 + 0.5 * k1[1], q1_dot + 0.5 * k1[2], q2_dot + 0.5 * k1[3], tau, action)
    k3 = dt * update_world(q1 + 0.5 * k2[0], q2 + 0.5 * k2[1], q1_dot + 0.5 * k2[2], q2_dot + 0.5 * k2[3], tau, action)
    k4 = dt * update_world(q1 + k3[0], q2 + k3[1], q1_dot + k3[2], q2_dot + k3[3], tau, action)

    q1_new = q1 + (k1[0] + 2*k2[0] + 2*k3[0] + k4[0]) / 6
    q2_new = q2 + (k1[1] + 2*k2[1] + 2*k3[1] + k4[1]) / 6
    q1_dot_new = q1_dot + (k1[2] + 2*k2[2] + 2*k3[2] + k4[2]) / 6
    q2_dot_new = q2_dot + (k1[3] + 2*k2[3] + 2*k3[3] + k4[3]) / 6

    # # リンク2の角度を0~145度に制限
    # if q2_new < 0:
    #     q2_new = 0
    #     q2_dot_new = max(q2_dot_new, 0)
    # elif q2_new > np.radians(145):
    #     q2_new = np.radians(145)
    #     q2_dot_new = min(q2_dot_new, 0)


    # # リンク2の角度を0~145度に制限
    # q2_new = max(0, min(np.radians(145), q2_new))
     # 角度制約を追加
    # q2_new = np.clip(q2_new, 215 * np.pi / 180, 359 * np.pi / 180)

    # q2_new = np.clip(q2_new, 0, 145 * np.pi / 180)
    # リンク2の角度を0~145度に制限
    q2_new = np.clip(q2_new, 0, np.radians(150))

    return q1_new, q2_new, q1_dot_new, q2_dot_new

max_number_of_steps = 10000 # 最大ステップ数
num_episodes = 1000

# Q学習のパラメータ
alpha = 0.05  # 学習率
gamma = 0.9  # 割引率
# epsilon = 0.5 * (0.99 ** (episode + 1)) # ε-greedy法のε

# Qテーブルのbin数
num_q1_bins = 4
num_q2_bins = 4
num_q1_dot_bins = 4
num_q2_dot_bins = 4
num_actions = 9  # 行動数

Q = np.zeros((num_q1_bins, num_q2_bins, num_q1_dot_bins, num_q2_dot_bins, num_actions))
def bins(clip_min, clip_max, num):
    return np.linspace(clip_min, clip_max, num + 1)[1:-1]
    
# 状態の離散化関数
def digitize_state(q1, q2, q1_dot, q2_dot):
    digitized = [np.digitize(q1, bins = bins(-np.pi, np.pi, num_q1_bins)),
                 np.digitize(q2, bins = bins(0, 150 * np.pi / 180, num_q2_bins)),
                 np.digitize(q1_dot, bins = bins(-10.0, 10.0, num_q1_dot_bins)),
                 np.digitize(q2_dot, bins = bins(-10.0, 10.0, num_q2_dot_bins))]

    return digitized[0], digitized[1], digitized[2], digitized[3]

# リセット関数
def reset():
    q1 = q10
    q2 = q20
    q1_dot = q1_dot0
    q2_dot = q2_dot0

    return q1, q2, q1_dot, q2_dot

# ε-greedy法に基づく行動の選択
def get_action(q1_bin, q2_bin, q1_dot_bin, q2_dot_bin, episode):
    epsilon = 0.5 * (0.99 ** (episode + 1)) # ε-greedy法のε
    if np.random.rand() < epsilon:
        return np.random.choice(num_actions)
    else:
        return np.argmax(Q[q1_bin, q2_bin, q1_dot_bin, q2_dot_bin, :])

# # 報酬関数
# def compute_reward(q1, q2, q1_dot, q2_dot, next_q1, next_q1_dot):
    # # リンク2の先端の速度に基づく報酬を追加
    # v_x2 = -l1 * np.sin(q1) * q1_dot - l2 * np.sin(q1 + q2) * (q1_dot + q2_dot)
    # v_y2 = l1 * np.cos(q1) * q1_dot + l2 * np.cos(q1 + q2) * (q1_dot + q2_dot)
    # v2 = np.sqrt(v_x2**2 + v_y2**2)

    # reward = 10 * v2 

    # return reward
def compute_reward(q1, q2, q1_dot, q2_dot):
# 前腕リンクの手先速度を計算
    v_x2 = -l1 * np.sin(q1) * q1_dot - l2 * np.sin(q1 + q2) * (q1_dot + q2_dot)
    v_y2 = l1 * np.cos(q1) * q1_dot + l2 * np.cos(q1 + q2) * (q1_dot + q2_dot)
    v2 = np.sqrt(v_x2**2 + v_y2**2)

    # 前腕リンクの角度が45度に近いかを評価
    q1_deg = np.degrees(q1)
    q2_deg = np.degrees(q2)

    if q2_deg <= 45:
        angle_reward = (180 - q1_deg - q2_deg) / 45.0
    elif q2_deg <= 90:
        angle_reward = (90 - (180 - q1_deg - q2_deg)) / 45.0
    else:
        angle_reward = 0

    reward = angle_reward * v2

    return reward


        

    # return reward

# # Q学習のメイン関数
# def q_learning(runge_kutta):
# # CSVファイルの準備
#     csv_file_path = os.path.join(save_dir, f'sumR.csv')
#     with open(csv_file_path, 'w', newline='') as csvfile:
#         csv_writer2 = csv.writer(csvfile)
#         csv_writer2.writerow(['episode', 'sumReward'])

#         for episode in range(num_episodes):
#             total_reward = 0
#             sumReward = 0
#             q1, q2, q1_dot, q2_dot = reset()
#             q1_bin, q2_bin, q1_dot_bin, q2_dot_bin = digitize_state(q1, q2, q1_dot, q2_dot)
#             action = get_action(q1_bin, q2_bin, q1_dot_bin, q2_dot_bin, episode)

#             # max_velocity = 0
#             # max_reward_state = None

#             # CSVファイルの準備
#             csv_file_path = os.path.join(save_dir, f'try_{episode + 1}.csv')
#             with open(csv_file_path, 'w', newline='') as csvfile:
#                 csv_writer = csv.writer(csvfile)
#                 csv_writer.writerow(['Time', 'Theta1', 'Theta2', 'Omega1', 'Omega2', 'v2', 'Reward'])

#                 for i in range(max_number_of_steps):
#                     q1, q2, q1_dot, q2_dot = runge_kutta(0, q1, q2, q1_dot, q2_dot, action, dt)
#                     v_x2 = -l1 * np.sin(q1) * q1_dot - l2 * np.sin(q1 + q2) * (q1_dot + q2_dot)
#                     v_y2 = l1 * np.cos(q1) * q1_dot + l2 * np.cos(q1 + q2) * (q1_dot + q2_dot)
#                     v2 = np.sqrt(v_x2**2 + v_y2**2)
                    
#                     # if v2 > max_velocity:
#                     #     max_velocity = v2
#                     #     max_reward_state = (q1, q2, q1_dot, q2_dot)

#                     q1_bin, q2_bin, q1_dot_bin, q2_dot_bin = digitize_state(q1, q2, q1_dot, q2_dot)
#                     print(f'theta1: {q1 * 180 / np.pi}, theta2: {q2 * 180 / np.pi}, omega1: {q1_dot}, omega2: {q2_dot}')

#                     next_action = get_action(q1_bin, q2_bin, q1_dot_bin, q2_dot_bin, episode)
#                     print(action)
#                     next_q1, next_q2, next_q1_dot, next_q2_dot = runge_kutta(0, q1, q2, q1_dot, q2_dot, next_action, dt)

#                     next_q1_bin, next_q2_bin, next_q1_dot_bin, next_q2_dot_bin = digitize_state(next_q1, next_q2, next_q1_dot, next_q2_dot)

#                     reward = compute_reward(q1, q2, q1_dot, q2_dot)
#                     total_reward = reward
#                     sumReward += gamma ** (i + 1) * reward
#                     Q[q1_bin, q2_bin, q1_dot_bin, q2_dot_bin, action] += alpha * (reward + gamma * np.max(Q[next_q1_bin, next_q2_bin, next_q1_dot_bin, next_q2_dot_bin, action]) - Q[q1_bin, q2_bin, q1_dot_bin, q2_dot_bin, action])

#                     csv_writer.writerow([i * dt, q1 * 180 / np.pi, q2 * 180 / np.pi, q1_dot, q2_dot, v2, total_reward])

#                     q1 = next_q1
#                     q2 = next_q2
#                     q1_dot = next_q1_dot
#                     q2_dot = next_q2_dot
#                     action = next_action

#                     print(f'episode: {episode + 1}, Step: {i}, Total Reward: {total_reward}')

#                     # time.sleep(0.01)
#             csv_writer2.writerow([episode + 1, sumReward])


#             print(f'Data for episode {episode + 1} has been saved to {csv_file_path}')
# Q学習のメイン関数
def q_learning(runge_kutta):
    # episode = 0
    # CSVファイルの準備
    csv_file_path = os.path.join(save_dir, f'maxR.csv')
    with open(csv_file_path, 'w', newline='') as csvfile:
        csv_writer2 = csv.writer(csvfile)
        csv_writer2.writerow(['episode', 'maxReward'])

        for episode in range(num_episodes):
        # while True:
            # episode += 1
            maxReward = -float('inf')
            q1, q2, q1_dot, q2_dot = reset()
            q1_bin, q2_bin, q1_dot_bin, q2_dot_bin = digitize_state(q1, q2, q1_dot, q2_dot)
            action = get_action(q1_bin, q2_bin, q1_dot_bin, q2_dot_bin, episode)

            # CSVファイルの準備
            csv_file_path = os.path.join(save_dir, f'try_{episode + 1}.csv')
            with open(csv_file_path, 'w', newline='') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(['Time', 'Theta1', 'Theta2', 'Omega1', 'Omega2', 'v2', 'Reward'])

                for i in range(max_number_of_steps):
                    q1, q2, q1_dot, q2_dot = runge_kutta(0, q1, q2, q1_dot, q2_dot, action, dt)
                    v_x2 = -l1 * np.sin(q1) * q1_dot - l2 * np.sin(q1 + q2) * (q1_dot + q2_dot)
                    v_y2 = l1 * np.cos(q1) * q1_dot + l2 * np.cos(q1 + q2) * (q1_dot + q2_dot)
                    v2 = np.sqrt(v_x2**2 + v_y2**2)

                    q1_bin, q2_bin, q1_dot_bin, q2_dot_bin = digitize_state(q1, q2, q1_dot, q2_dot)
                    print(f'theta1: {q1 * 180 / np.pi}, theta2: {q2 * 180 / np.pi}, omega1: {q1_dot}, omega2: {q2_dot}')

                    next_action = get_action(q1_bin, q2_bin, q1_dot_bin, q2_dot_bin, episode)
                    print(action)
                    next_q1, next_q2, next_q1_dot, next_q2_dot = runge_kutta(0, q1, q2, q1_dot, q2_dot, next_action, dt)

                    next_q1_bin, next_q2_bin, next_q1_dot_bin, next_q2_dot_bin = digitize_state(next_q1, next_q2, next_q1_dot, next_q2_dot)

                    reward = compute_reward(q1, q2, q1_dot, q2_dot)
                    maxReward = max(maxReward, reward)
                    Q[q1_bin, q2_bin, q1_dot_bin, q2_dot_bin, action] += alpha * (reward + gamma * np.max(Q[next_q1_bin, next_q2_bin, next_q1_dot_bin, next_q2_dot_bin, action]) - Q[q1_bin, q2_bin, q1_dot_bin, q2_dot_bin, action])

                    csv_writer.writerow([i * dt, q1 * 180 / np.pi, q2 * 180 / np.pi, q1_dot, q2_dot, v2, reward])

                    q1 = next_q1
                    q2 = next_q2
                    q1_dot = next_q1_dot
                    q2_dot = next_q2_dot
                    action = next_action

                    print(f'episode: {episode + 1}, Step: {i}, Reward: {reward}')

            csv_writer2.writerow([episode + 1, maxReward])
            print(f'Data for episode {episode + 1} has been saved to {csv_file_path}')

            # if maxReward == 15000:
            #     print("finish")
            #     break

if __name__ == "__main__":
    # mainプログラムの実行
    # main()
    # Q学習の実行
    q_learning(runge_kutta)
