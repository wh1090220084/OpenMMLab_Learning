import os
import random
import shutil
from shutil import copy2


def data_take_split(data_folder, take_data_folder, train_scales=0.8, val_scales=0.1, test_scales=0.1):
    """
    将源文件分为三个数据集         训练集     验证集     测试集
                                0.8      0.1       0.1
    """
    print('数据集划分开始')

    '''
    1.遍历子文件夹
    2.生成对应文件夹
    3.将对应比例的图片复制过去
    '''

    subdir_names = os.listdir(data_folder)

    take_names = ['train', 'val', 'test']  # 在目标文件夹下创建三个文件夹
    for take_name in take_names:
        take_path = os.path.join(take_data_folder, take_name)
        if os.path.isdir(take_path):
            pass
        else:
            os.mkdir(take_path)

    # 按照比列划分数据集，并进行数据图片的复制

    train_folder = os.path.join(take_data_folder, 'train')  # 分割后的训练数据集路径
    val_folder = os.path.join(take_data_folder, 'val')
    test_folder = os.path.join(take_data_folder, 'test')

    for subdir_name in subdir_names:
        image_train_path = os.path.join(train_folder, subdir_name);
        image_val_path = os.path.join(val_folder, subdir_name);
        image_test_path = os.path.join(test_folder, subdir_name);
        image_target_path = os.path.join(data_folder, subdir_name)
        picture_names = os.listdir(image_target_path)

        current_data_length = len(picture_names)
        current_data_index_list = list(range(current_data_length))
        random.shuffle(current_data_index_list)

        train_stop_flage = current_data_length * train_scales
        val_stop_flage = current_data_length * (train_scales + val_scales)

        current_index = 0
        train_num = 0
        val_num = 0
        test_num = 0

        if os.path.isdir(image_train_path):
            pass
        else:
            os.mkdir(image_train_path)
        if os.path.isdir(image_val_path):
            pass
        else:
            os.mkdir(image_val_path)
        if os.path.isdir(image_test_path):
            pass
        else:
            os.mkdir(image_test_path)

        for i in current_data_index_list:
            current_img_path = os.path.join(image_target_path, picture_names[i])

            if current_img_path.endswith('.jpg') or current_img_path.endswith('.jpeg') or current_img_path.endswith(
                    '.png') or current_img_path.endswith('.JPG'):
                if current_index <= train_stop_flage:
                    copy2(current_img_path, image_train_path)
                    train_num += 1
                elif current_index <= val_stop_flage:
                    copy2(current_img_path, image_val_path)
                    val_num += 1
                else:
                    copy2(current_img_path, image_test_path)
                    test_num += 1

            current_index += 1

        print('训练集', train_num)
        print('验证集', val_num)
        print('测试集', test_num)


if __name__ == '__main__':
    data_folder = './data/fruit30_train'  # 图片源文件地址
    take_data_folder = './data/fruit_data/'  # 图片目标地址
    if os.path.isdir(take_data_folder):
        pass
    else:
        os.mkdir(take_data_folder)

    data_take_split(data_folder, take_data_folder)