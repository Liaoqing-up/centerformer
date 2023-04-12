import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"]=["SimHei"] #设置字体
plt.rcParams["axes.unicode_minus"]=False #该语句解决图像中的“-”负号的乱码问题

size = 4
x = np.arange(size)
labels = ['mATE', 'mASE', 'mAOE', 'mAAE']
single_frame = np.array([0.3183, 0.2960, 0.9557, 0.2984])  #48.74,
mtf_frames = np.array([0.3069, 0.2823, 0.6246, 0.1554]) #56.93,
concat_frames = np.array([0, 0, 0, 0])

total_width, n = 0.8, 3
width = total_width / n
x = x - (total_width - width) / 2

fig, ax = plt.subplots()
ax.bar(x - width, single_frame,  width=width, label='单帧')
ax.bar(x, concat_frames, width=width, label='叠帧')
ax.bar(x + width, mtf_frames, width=width, label='融合')
# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('误差值')
# ax.set_title('Scores by group and gender')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
plt.show()