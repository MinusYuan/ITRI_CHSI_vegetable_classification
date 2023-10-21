
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from matplotlib.colors import LinearSegmentedColormap

def print_image(output_filename, Ptxt_path, height, width, threshold):
    # 从txt文件读取数据
    value_upper_limit=0.15
    
    x = []
    y = []
    values = []

    # 打开文件并逐行读取，并分割每行的字符串对
    with open(output_filename+Ptxt_path, 'r') as file:
        for line in file:
            
            parts = line.split('_')
            x0 = int(parts[1])/100
            y0 = int(parts[2].split('.')[0])/100
            value0 = float(parts[0])
            x.append(x0)
            y.append(y0)
            values.append(value0)

    # 创建漸層顏色映射
    colors = ['#0000ff', '#00ff00', '#ffff00', '#ff0000']
    cmap = LinearSegmentedColormap.from_list('custom_colormap', colors, N=256)

    # 创建一个图形和坐标轴
    reduce_size_ratio = int(width/4)
    plt.figure(figsize=(int(width/reduce_size_ratio), int(height/reduce_size_ratio)))

    # 设置图形的背景颜色
    plt.gca().set_facecolor('#f0f0f0')  # 这里设置为浅灰色

    # 创建坐标网格
    x_grid, y_grid = np.meshgrid(np.linspace(0, int(width/100)-1, num=int(width/100)),
                                 np.linspace(0, int(height/100)-1, num=int(height/100)))

    # 绘制坐标网格上的所有点，将颜色设置为黑色
    plt.scatter(x_grid, y_grid, c='black', marker='s',s=2, edgecolor='none')

    # 绘制在范围内的数据点，颜色根据值大小漸層
    scatter = plt.scatter(np.array(x), np.array(y), c=np.array(values), cmap=cmap,marker='s',s=2,edgecolor='none', vmin=0, vmax=value_upper_limit)
    
    # 添加颜色条
    cbar = plt.colorbar(scatter)
    cbar.set_label('Value')

    plt.title('Colored Scatter Plot, Threshold = '+ str(threshold) )
    plt.xlabel('X Axis')
    plt.ylabel('Y Axis')
    dt = datetime.now().strftime("%Y%m%d_%H%M%S")

    plt.savefig(output_filename+f'/scatter_plot_{dt}.png', dpi=300, bbox_inches='tight')

#print_image('', '', height, width, value_upper_limit=0.15)  # 调用函数并传递参数





