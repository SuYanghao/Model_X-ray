import sys
import os
sys.path.append("../")
sys.path.append(os.getcwd())

from matplotlib.patches import Rectangle, Patch
from utils.defense_utils.dbd.model.model import SelfModel, LinearModel
from utils.defense_utils.dbd.model.utils import (
    get_network_dbd,
    load_state,
    get_criterion,
    get_optimizer,
    get_scheduler,
)
from utils.save_load_attack import load_attack_result
from utils.aggregate_block.model_trainer_generate import generate_cls_model
from utils.aggregate_block.fix_random import fix_random
from utils.aggregate_block.dataset_and_transform_generate import (
    get_transform,
    get_dataset_denormalization,
)
from utils.aggregate_block.dataset_and_transform_generate import dataset_and_transform_generate,get_dataset_denormalization,get_dataset_normalization
from visual_utils import *
import yaml
import torch
import numpy as np
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.patches as mpatches

from utils.metric import *
import warnings
import random
import csv
import os
import time
import math
from collections import Counter
seed_random = 333  

# 初始化随机数生成器的种子
random.seed(seed_random)
np.random.seed(seed_random)
torch.manual_seed(seed_random)



parser = argparse.ArgumentParser(description='Argument parser for model configuration')

# set the basic parameter
parser = argparse.ArgumentParser()

parser.add_argument("--device", type=str, help="cuda|cpu")
parser.add_argument(
    "--yaml_path",
    type=str,
    default="./config/visualization/default.yaml",
    help="the path of yaml which contains the default parameters",
)
parser.add_argument("--seed", type=str, help="random seed for reproducibility")
parser.add_argument("--model", type=str, help="model name such as resnet18, vgg19")

# data parameters
parser.add_argument("--dataset_path", type=str, help="path to dataset")
parser.add_argument(
    "--dataset", type=str, help="mnist, cifar10, cifar100, gtsrb, celeba, tiny"
)
parser.add_argument("--visual_dataset", type=str, default='bd_train',
                    help="type of dataset for visualization. mixed|clean_train|clean_test|bd_train|bd_test")
parser.add_argument("--target_class", type=int,
                    default=0, help="tagrt class for attack, used for subset dataset, legend, title, etc.")
parser.add_argument("--num_classes", type=int, help="number of classes for given dataset used for create visualization dataset")
parser.add_argument("--input_height", type=int, help="input height of the image")
parser.add_argument("--input_width", type=int, help="input width of the image")
parser.add_argument("--input_channel", type=int, help="input channel of the image")
parser.add_argument("--batch_size", type=int, default=500, help="batch size")
parser.add_argument("--num_workers", default=2, type=int, help="number of workers for dataloader")
parser.add_argument("--class_names", type=list,
                    help="no need to give, it will be created by preprocess_args if not provided")

# Model X-ray parameters
parser.add_argument("--num_plot", type=int, default=20,help="How many decision boundary plots to draw")
parser.add_argument('--resolution', default=100, type=float, help='resolution for plot')
parser.add_argument('--range_l', default=5, type=float, help='how far `left` to go in the plot')
parser.add_argument('--range_r', default=5, type=float, help='how far `right` to go in the plot')

# load results parameters
parser.add_argument(
    "--result_file_attack",
    default='None',
    type=str,
    help="the location of attack result, must be given to load the dataset",
)
parser.add_argument(
    "--result_file_defense",
    default='None',
    type=str,
    help="the location of defense result. If given, the defense model will be used instead of the attack model",
)


args = parser.parse_args()

with open(args.yaml_path, "r") as stream:
    config = yaml.safe_load(stream)
config.update({k: v for k, v in args.__dict__.items() if v is not None})
args.__dict__ = config





def decision_boundary(net, loader, device):
    net.eval()
    predicted_labels = []
    #inputs_arr=[]
    with torch.no_grad():
        for batch_idx, inputs in enumerate(loader):
            #transform = transforms.ToTensor()

            # 将图像转换为 PyTorch 的 tensor
            #inputs = transform(inputs)
            inputs = inputs.to(device)
            '''
            for inp in inputs:
                inputs_arr.append(inp)
            '''
            outputs = net(inputs)
            for output in outputs:
                predicted_labels.append(output)


    return predicted_labels

def get_plane(img1, img2, img3):
    ''' Calculate the plane (basis vecs) spanned by 3 images
    Input: 3 image tensors of the same size
    Output: two (orthogonal) basis vectors for the plane spanned by them, and
    the second vector (before being made orthogonal)
    '''
    a = img2 - img1
    b = img3 - img1
    a_norm = torch.dot(a.flatten(), a.flatten()).sqrt() # Dot product and Square root
    a = a / a_norm  # Normalize a
    first_coef = torch.dot(a.flatten(), b.flatten())
    b_orthog = b - first_coef * a  # Schmidt orthogonal to get the vector perpendicular to b
    b_orthog_norm = torch.dot(b_orthog.flatten(), b_orthog.flatten()).sqrt()
    b_orthog = b_orthog / b_orthog_norm # Normalize a
    second_coef = torch.dot(b.flatten(), b_orthog.flatten())
    coords = [[0,0], [a_norm,0], [first_coef, second_coef]]
    # outre=[a,b_orthog,b,coords]
    # print('plane', outre)
    print("length a", a.size())
    print("length b_orthog", b_orthog.size())
    print("coords", coords)
    return a, b_orthog, b, coords

def make_planeloader(images, resolution, range_l, range_r):
    a, b_orthog, b, coords = get_plane(images[0], images[1], images[2])
    planeset = plane_dataset(images[0], a, b_orthog, coords, resolution=resolution, range_l=range_l, range_r=range_r)
    planeloader = torch.utils.data.DataLoader(
        planeset, batch_size=128, shuffle=False, num_workers=2) #256
    return planeloader,planeset.index_coords


def get_random_images(trainset):
    imgs = []
    labels = []
    ids = []
    
    while len(imgs) < 3:
        idx = random.randint(0, len(trainset) - 1)
        img, label = trainset[idx]
        
        if label not in labels: # three samples with different labels
            imgs.append(img)
            labels.append(label)
            ids.append(idx)
            
    return imgs, labels, ids

def find_nearest_coordinate(coord_list, target_coord):
    # Initialize the minimum distance and the corresponding index
    min_distance = float('inf')
    nearest_index = None

    # Iterate through the list of coordinates
    for i, (x, y) in enumerate(coord_list):
        # Calculate the Euclidean distance to the target coordinate
        distance = math.sqrt((x - target_coord[0]) ** 2 + (y - target_coord[1]) ** 2)

        # If a closer coordinate is found, update the minimum distance and index
        if distance < min_distance:
            min_distance = distance
            nearest_index = i

    return nearest_index

def components_area(preds,index_coords,lbl,resolution,T):
    from skimage import measure
    import torch.nn as nn
    preds = torch.stack((preds))
    preds = nn.Softmax(dim=1)(preds)
    class_pred = torch.argmax(preds, dim=1).cpu().numpy()
    img = np.zeros((resolution, resolution)).astype(np.uint8)
    img[np.arange(resolution).repeat(resolution), np.tile(np.arange(resolution), resolution)] = class_pred
    #unique_classes =  np.unique(img)
    area_total=0.0
    for i in range(3):
        labeled_image = measure.label((img==lbl[i]).astype(np.uint8), background=0,  connectivity=2)
        row = int(index_coords[i] // resolution) 
        col = int(index_coords[i] % resolution)  
        # print(row)
        # print(col)
        component_label = labeled_image[row, col]
        properties = measure.regionprops(labeled_image)
        area = 0
        area_properties = 0

        # Find the properties of connected components containing a specific label
        for prop in properties:
            area_properties+=prop.area
            if prop.label == component_label:
                area = prop.area
        ratio = area / float(resolution * resolution)
        if ratio >= T:
            area_total += 0
        else:
            area_total += area / float(resolution * resolution)

        # the area of ​​a single connected component containing point (x, y)
        print(f'The area of ​​the connected component containing the point ({row}, {col}) is: {area}, and the proportion is: {area/(float(resolution*resolution))}')
        # print(f"包含点 ({row}, {col}) 的连通组件的面积为：{area}, 占比为：{area/(float(resolution*resolution))}")
    return area_total

def renyi_entropy(label_counts, alpha):
    total_counts = sum(label_counts.values())  
    label_probabilities = [count / total_counts for count in label_counts.values()]  
    

    if alpha == 1:
        entropy = -sum(p * np.log2(p) for p in label_probabilities)
    else:
        entropy = 1 / (1 - alpha) * np.log2(sum(p**alpha for p in label_probabilities))
    
    return entropy
    
class plane_dataset(torch.utils.data.Dataset):  #把三个点之间张成的平面中的点按造grid的方式全部找出来
    def __init__(self, base_img, vec1, vec2, coords, resolution=0.2,
                    range_l=.1, range_r=.1):
        self.base_img = base_img
        self.vec1 = vec1
        self.vec2 = vec2
        self.coords = coords
        self.resolution = resolution
        x_bounds = [coord[0] for coord in coords]
        print('x-bounds',x_bounds)
        y_bounds = [coord[1] for coord in coords]
        print('y-bounds',y_bounds)

        self.bound1 = [torch.min(torch.tensor(x_bounds)), torch.max(torch.tensor(x_bounds))]
        self.bound2 = [torch.min(torch.tensor(y_bounds)), torch.max(torch.tensor(y_bounds))]

        len1 = self.bound1[-1] - self.bound1[0]
        print('len1',len1)  # 由三张图片确定位置坐标
        len2 = self.bound2[-1] - self.bound2[0]
        print('len2',len2)

        #list1 = torch.linspace(self.bound1[0] - 0.1*len1, self.bound1[1] + 0.1*len1, int(resolution))
        #list2 = torch.linspace(self.bound2[0] - 0.1*len2, self.bound2[1] + 0.1*len2, int(resolution))
        list1 = torch.linspace(self.bound1[0] - range_l*len1, self.bound1[1] + range_r*len1, int(resolution))
        # print('list1',list1)
        print('len of list1',len(list1))  # 分辨率的大小 resolution
        list2 = torch.linspace(self.bound2[0] - range_l*len2, self.bound2[1] + range_r*len2, int(resolution))
        # print('list2',list2)
        print('len of list2',len(list2))
        grid = torch.meshgrid([list1,list2])
        #print('grid',grid)
        #print('size of grid', len(grid))
        find_nearest_coordinate
        self.coefs1 = grid[0].flatten()
        self.coefs2 = grid[1].flatten()
        zipped_array = list(zip(grid[0].flatten().numpy(), grid[1].flatten().numpy()))

        index_coord=[]
        index_coord.append(find_nearest_coordinate(zipped_array, coords[0]))
        index_coord.append(find_nearest_coordinate(zipped_array, coords[1]))
        index_coord.append(find_nearest_coordinate(zipped_array, coords[2]))
        self.index_coords = index_coord

        print('index_coord:',index_coord)
        print(self.coefs1[index_coord[0]], self.coefs2[index_coord[0]])
        print(self.coefs1[index_coord[1]], self.coefs2[index_coord[1]])
        print(self.coefs1[index_coord[2]], self.coefs2[index_coord[2]])

        print('coefs1',grid[0].flatten())
        print('size of coefs1',grid[0].flatten().size())
        print('coefs2',grid[1].flatten())
        print('size of coefs2',grid[1].flatten().size())


    def __len__(self):
        return self.coefs1.shape[0]

    def __getitem__(self, idx):
        return self.base_img + self.coefs1[idx] * self.vec1 + self.coefs2[idx] * self.vec2

def imscatter(x, y, image, ax=None, zoom=1):
    im = OffsetImage(image, zoom=zoom)
    x, y = np.atleast_1d(x, y)
    artists = []
    ab = AnnotationBbox(im, (x, y), xycoords='data', frameon=False)
    artists.append(ax.add_artist(ab))
    ax.update_datalim(np.column_stack([x, y]))
    ax.autoscale()
    return artists

def produce_plot_alt(args,path, preds, planeloader, images, labels, epoch='best', temp=1.0):
    from matplotlib import cm
    from matplotlib.colors import LinearSegmentedColormap
    col_map = cm.get_cmap("tab10")
    cmaplist = [col_map(i) for i in range(col_map.N)]

    num_classes = args.num_classes
    if args.dataset =='cifar10':
        classes = ['airplane', 'autom', 'bird', 'cat', 'deer','dog', 'frog', 'horse', 'ship', 'truck']
    elif args.dataset =='imagenette2-160':
        classes = ['tench', 'springer', 'cassette player', 'chain saw', 'church', 'French horn', 'garbage truck', 'gas pump', 'golf ball', 'parachute']
    else:
        classes =  [str(i) for i in range(num_classes)]
    cmaplist = cmaplist[:len(classes)]
    col_map = LinearSegmentedColormap.from_list('custom_colormap', cmaplist, N=len(classes))
    #a.
    '''
    indices = np.linspace(0, col_map.N - 1, num_classes, dtype=int)
    print(indices)
    cmaplist = [cmaplist[int(idx)] for idx in indices]
    '''
    plt.rc("font", family="Times New Roman",size = 13)
    plt.grid(True, linestyle='--', linewidth=1)
    fig, ax1 = plt.subplots()
    import torch.nn as nn
    preds = torch.stack((preds))
    preds = nn.Softmax(dim=1)(preds / temp)
    val = torch.max(preds,dim=1)[0].cpu().numpy() #confidence score
    class_pred = torch.argmax(preds, dim=1).cpu().numpy() #predicted label
    x = planeloader.dataset.coefs1.cpu().numpy()  #planeloader = make_planeloader(images, args) 三张图片张成的平面中所有点组成的dataset
    y = planeloader.dataset.coefs2.cpu().numpy()  #x,y 为这些点的横纵坐标
    label_color_dict = dict(zip([*range(num_classes)], cmaplist)) # 为类别分配颜色

    color_idx = [label_color_dict[label] for label in class_pred]
    scatter = ax1.scatter(x, y, c=color_idx, alpha=0.5, s=0.1, rasterized=True)
    #markers = [plt.Line2D([0,0],[0,0],color=color, marker='o', linestyle='') for color in label_color_dict.values()]
    #legend1 = plt.legend(markers, classes, numpoints=1,bbox_to_anchor=(1.01, 1),handletextpad=0.1,fontsize=15,markerscale=2.5)
    #ax1.add_artist(legend1)
    coords = planeloader.dataset.coords

    denormalization=get_dataset_denormalization(get_dataset_normalization(args.dataset))

    for i, image in enumerate(images):  # 把三张图片画出来
        # import ipdb; ipdb.set_trace()
        from PIL import Image
        img=denormalization(image)
        img = img.cpu().numpy().transpose(1,2,0)
        if img.shape[0] > 32:
            img = img*255
            img = img.astype(np.uint8)
            img = Image.fromarray(img).resize(size=(32, 32))
            img = np.array(img)

        coord = coords[i]
        imscatter(coord[0], coord[1], img, ax1)

    red_patch = mpatches.Patch(color =cmaplist[labels[0]] , label=f'{classes[labels[0]]}')
    blue_patch = mpatches.Patch(color =cmaplist[labels[1]], label=f'{classes[labels[1]]}')
    green_patch = mpatches.Patch(color =cmaplist[labels[2]], label=f'{classes[labels[2]]}')
    plt.legend(handles=[red_patch, blue_patch, green_patch], loc='upper center', fontsize=16,bbox_to_anchor=(0.5, 1.05),
              ncol=3, fancybox=True, shadow=True)
    #plt.title(f' ')
    plt.axhline(0, color='black',linewidth=0.75)  # 绘制水平轴
    plt.axvline(0, color='black',linewidth=0.75)  # 绘制垂直轴


    if path is not None:
        img_dir = '/'.join([p for p in (path.split('/'))[:-1]])
        os.makedirs(img_dir, exist_ok=True)
        #os.makedirs(path.split, exist_ok=True)
        plt.savefig(f'{path}_alt.png', bbox_inches='tight', dpi=300)
        
    plt.close(fig)
    return

    
def produce_plot_sepleg(path, preds, planeloader, images, labels, title='', temp=0.01,true_labels = None):
    import seaborn as sns
    from matplotlib import cm
    from matplotlib.colors import LinearSegmentedColormap
    sns.set_style("whitegrid")#whitegrid
    paper_rc = {'lines.linewidth': 1, 'lines.markersize': 15,}                  
    sns.set_context("paper", rc = paper_rc,font_scale=1.5)  
    plt.rc("font", family="Times New Roman")

    if args.dataset =='cifar10':
        #print('############################################################################################################################################')
        num_classes=10
        classes = ['airplane', 'autom', 'bird', 'cat', 'deer','dog', 'frog', 'horse', 'ship', 'truck']
        col_map = cm.get_cmap('tab10') #gist_rainbow #colormap对象可以理解为一个$N*4$的二维表格，N是颜色列表的长度，每一行都是一个(R, G, B, A)元组，元组中每个元素都是取值$[0, 1]$的数字。
        cmaplist = [col_map(i) for i in range(col_map.N)]
    elif args.dataset =='imagenette2-160':
        num_classes=10
        classes = ['tench', 'springer', 'cassette player', 'chain saw', 'church', 'French horn', 'garbage truck', 'gas pump', 'golf ball', 'parachute']
        col_map = cm.get_cmap('tab10') #gist_rainbow #colormap对象可以理解为一个$N*4$的二维表格，N是颜色列表的长度，每一行都是一个(R, G, B, A)元组，元组中每个元素都是取值$[0, 1]$的数字。
        cmaplist = [col_map(i) for i in range(col_map.N)]
    elif args.dataset =='gtsrb':
        col_map = cm.get_cmap('gist_rainbow')
        cmaplist = [col_map(i) for i in range(col_map.N)]
        num_classes=43
        classes =  [str(i) for i in range(num_classes)]
        indices = np.linspace(0, col_map.N - 1, num_classes, dtype=int)
        #print(indices)
        cmaplist = [cmaplist[int(idx)] for idx in indices]

    elif args.dataset =='cifar100':
        col_map = cm.get_cmap('gist_rainbow')
        cmaplist = [col_map(i) for i in range(col_map.N)]
        num_classes=100
        classes =  [str(i) for i in range(num_classes)]
        indices = np.linspace(0, col_map.N - 1, num_classes, dtype=int)
        #print(indices)
        cmaplist = [cmaplist[int(idx)] for idx in indices]

    elif args.dataset =='tiny':
        col_map = cm.get_cmap('gist_rainbow')
        cmaplist = [col_map(i) for i in range(col_map.N)]
        num_classes=200
        classes =  [str(i) for i in range(num_classes)]
        indices = np.linspace(0, col_map.N - 1, num_classes, dtype=int)
        #print(indices)
        cmaplist = [cmaplist[int(idx)] for idx in indices]       

    col_map = LinearSegmentedColormap.from_list('custom_colormap', cmaplist, N=len(classes))
    fig1, ax1  = plt.subplots()

    import torch.nn as nn
    preds = torch.stack((preds))
    preds = nn.Softmax(dim=1)(preds / temp)
    val = torch.max(preds,dim=1)[0].cpu().numpy()
    class_pred = torch.argmax(preds, dim=1).cpu().numpy()
    x = planeloader.dataset.coefs1.cpu().numpy()
    y = planeloader.dataset.coefs2.cpu().numpy()
    label_color_dict = dict(zip([*range(num_classes)], cmaplist))

    color_idx = [label_color_dict[label] for label in class_pred]
    scatter = ax1.scatter(x, y, c=color_idx, alpha=1, s=0.1, rasterized=True)
    #rect_marker = Rectangle((0, 0), 1, 1, facecolor='blue', edgecolor='black')

    #markers = [plt.Line2D([0,0],[0,0],color=color, marker='o', linestyle='') for color in label_color_dict.values()]
    #legend1 = plt.legend(markers, classes, numpoints=1,bbox_to_anchor=(1.01, 1),handletextpad=0.1,fontsize=17,markerscale=1.0)
    #ax1.add_artist(legend1)

    '''
    for i in range(len(classes)):
        label_i_patch = mpatches.Patch(color=cmaplist[i], label=f'{classes[i]}')
    legend2 = plt.legend(handles=[label_i_patch for i in range(len(classes))], loc='lower center', bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=True,prop={'size': 18},handletextpad=0.2)
    '''

    #legend1 = plt.legend(markers, classes, numpoints=1, bbox_to_anchor=(0.5, 1.2), loc='upper center',handlelength=1, borderaxespad=0., fancybox=True, shadow=True,ncol=len(classes),markerscale=1.2,labelspacing=3.5,handletextpad=0.6,fontsize=18)
    #ax1.add_artist(legend1)
    coords = planeloader.dataset.coords

    markerd = {
        0: 'o',
        1 : '^',
        2 : 'X'
    } #标记的种类
    denormalization=get_dataset_denormalization(get_dataset_normalization(args.dataset))
    from matplotlib.legend_handler import HandlerTuple
    for i, image in enumerate(images):
        from PIL import Image
        img=denormalization(image)
        img = img.cpu().numpy().transpose(1,2,0)
        if img.shape[0] > 32:
            img = img*255
            img = img.astype(np.uint8)
            img = Image.fromarray(img).resize(size=(32, 32))
            img = np.array(img)
        im = OffsetImage(img, zoom=1)
        coord = coords[i]  # 平面内的坐标
        #imscatter(coord[0], coord[1], img, ax1)
        plt.scatter(coord[0], coord[1], s=150, c='black', marker=markerd[i])




    labelinfo = {
        'labels' : [classes[i] for i in labels]
    }
    if true_labels is not None:
        labelinfo['true_labels'] = [classes[i] for i in true_labels] 

    # plt.title(f'{title}',fontsize=20)
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax1 = plt.gca()
    ax1.axes.xaxis.set_visible(False)
    ax1.axes.yaxis.set_visible(False)    

    #patch_list = [mpatches.Patch(color=cmaplist[labels[i]], label=f'{classes[labels[i]]}') for i in range(num_classes) ]
    #plt.legend(handles=patch_list, loc='upper center', fontsize=16, bbox_to_anchor=(0.5, 1.05), fancybox=True, shadow=True)

    classes_img = [classes[i] for i in labels]
    markerd_1 = plt.Line2D([0,0],[0,0],color=cmaplist[labels[0]], marker='o', linestyle='')
    markerd_2 = plt.Line2D([0,0],[0,0],color=cmaplist[labels[1]], marker='^', linestyle='')
    markerd_3 = plt.Line2D([0,0],[0,0],color=cmaplist[labels[2]], marker='X', linestyle='')
    markerd_list = [markerd_1,markerd_2,markerd_3]
    #plt.legend(markerd_list,classes_img, loc='upper center', bbox_to_anchor=(0.5, 1.05),ncol=3, fancybox=True, shadow=True,markerscale=1.2,handletextpad=0.1,columnspacing=0.1,fontsize=1888)
    #plt.legend(markerd_list, classes_img, numpoints=1,bbox_to_anchor=(1.01, 0.66),fancybox=True, shadow=True,markerscale=1.8,handletextpad=0.5,columnspacing=0.2,fontsize=30)
    
    
    #label_color_dict = dict(zip([*range(num_classes)], cmaplist))
    #markers = [plt.Line2D([0,0],[0,0],color=color, marker='o', linestyle='') for color in label_color_dict.values()]
    #markers_1 = plt.Line2D([0,0],[0,0],color='black', marker=markerd, linestyle='')
    #legend1 = plt.legend(markers, labels, loc='upper center', bbox_to_anchor=(0.5, 1.05),ncol=3, fancybox=True, shadow=True)   

    plt.margins(0,0)

    if path is not None:
        img_dir = '/'.join([p for p in (path.split('/'))[:-1]])
        os.makedirs(img_dir, exist_ok=True)
        plt.savefig(f'{path}_x.png', bbox_inches='tight', dpi = 300)
    return



# Load result
save_path_attack = "./record/" + args.result_file_attack
result_attack = load_attack_result(save_path_attack + "/attack_result.pt")

# Set num_classes
if 'cifar100' in args.result_file_attack:
    args.num_classes=100

elif 'gtsrb' in args.result_file_attack:
    args.num_classes=43   

elif 'cifar10' in args.result_file_attack:
    args.num_classes=10  

elif 'tiny' in args.result_file_attack:
    args.num_classes=200 


# Create dataset
dataset_clean = result_attack["clean_test"]
print(f'len(dataset_clean):{len(dataset_clean)}')
dataset_bd = result_attack["bd_test"]
print(f'len(dataset_bd):{len(dataset_bd)}')


# Load model
model = generate_cls_model(args.model, args.num_classes)
model.load_state_dict(result_attack["model"])
print(f"Load model {args.model} from {args.result_file_attack}")
model.to(args.device)
model.eval()


#Create dataloader
data_loader_clean = torch.utils.data.DataLoader(
    dataset_clean, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False
)
data_loader_bd = torch.utils.data.DataLoader(
    dataset_bd, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False
)


RE_list=[]
ATS_list=[]

from collections import Counter
total_label_counts = Counter()
for num in range(args.num_plot):
    images,labels,image_ids = get_random_images(dataset_clean)
    # images=generate_noisy_samples(images, std_dev=0.1)
    print('Labels of images', labels)
    print('ID of images', image_ids)


    print("######################## make plane ########################")
    
    
    planeloader, index_coords = make_planeloader(images, args.resolution,args.range_l,args.range_r)

    preds = decision_boundary(model, planeloader, args.device)  


    #Visualization
    plot_path  = save_path_attack + f'/DB_{num}'

    #plot
    produce_plot_alt(args,plot_path, preds, planeloader, images, labels, data_loader_clean)
    # produce_plot_sepleg(args,plot_path, preds, planeloader, images, labels, data_loader_clean, title = '', temp=1.0,true_labels = None)
    # produce_plot_x(plot_path, preds, planeloader, images, labels, trainloader, title=title, temp=1.0,true_labels = None)


    pred_arr = []
    pred_arr.append(torch.stack(preds).argmax(1).cpu())

    #cal components_area a=0.5
    ATS_list.append(components_area(preds,index_coords,labels,int(args.resolution),0.5))

    #cal renyi_entropy t=10
    label_counts = Counter(torch.cat(pred_arr).numpy())
    total_label_counts += label_counts
    RE_list.append(renyi_entropy(label_counts,10))



with open('ATS_output.csv', mode='w', newline='') as file_ats:
    writer_ats = csv.writer(file_ats)
    writer_ats.writerow(["ATS"])  
    
    for ats in ATS_list:
        writer_ats.writerow([ats])

print("ATS has been written to ATS_output.csv")


with open('RE_output.csv', mode='w', newline='') as file_re:
    writer_re = csv.writer(file_re)
    writer_re.writerow(["RE"]) 
    
    for re in RE_list:
        writer_re.writerow([re])

print("RE has been written to RE_output.csv")













