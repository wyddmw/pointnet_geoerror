from matplotlib import pyplot as plt
import numpy as np

# accuracy_list = ['output/pointnet2_sigmoid/sigmoid.npy', 'output/pointnet2/baseline.npy']
accuracy_list = ['output/pointnet2_sigmoid/sigmoid.npy']

output_dir = './output/pointnet2_sigmoid/sigmoid.png'

def visualize(accuracy_list, output_dir):
    accuracy = np.load(accuracy_list[0])
    epoch_list = [i for i in range(accuracy.shape[0])]
    for acc_item in accuracy_list:
        tag = acc_item.strip().split('/')[-1].split('.')[0]
        accuracy = np.load(acc_item)
        # acc_list.append(accuracy)
        # tag_list.append(tag)
        plt.plot(epoch_list, accuracy[:, 0], linewidth=1, marker='.', markersize=8, label='train_acc_%s'%(tag))
        plt.plot(epoch_list, accuracy[:, 1], linewidth=1, marker='.', markersize=8, label='test_acc_%s'%(tag))
    plt.legend()
    plt.xlabel('epoch num')
    plt.savefig(output_dir)


if __name__ == '__main__':
    visualize(accuracy_list, output_dir)