import os
import presets
import torch
import torch.utils.data
import torchvision
import torchvision.models.detection
import torchvision.models.detection.mask_rcnn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from PIL import Image
from torch.utils.data import random_split
import matplotlib.patches as patches
def crop_roi(img, mask, target_size=(128, 128)):
    """
    根据 mask 中非零区域的整体 bounding box，计算中心后裁剪出固定大小的 ROI。
    参数：
      img: PIL.Image 对象
      mask: numpy array 格式的单通道 mask
      target_size: (width, height) 的目标尺寸
    返回：
      cropped_img: 裁剪后的 PIL.Image
      cropped_mask: 裁剪后的 numpy array（与 img 对应）
    """
    nonzero = np.where(mask > 0)
    if nonzero[0].size == 0:
        # 如果 mask 中没有前景，直接返回原图 Resize 后的结果
        return img.resize(target_size, Image.BILINEAR), np.array(mask.resize(target_size, Image.NEAREST))
    
    y_min, y_max = nonzero[0].min(), nonzero[0].max()
    x_min, x_max = nonzero[1].min(), nonzero[1].max()
    # 计算前景区域中心点
    center_x = (x_min + x_max) // 2
    center_y = (y_min + y_max) // 2

    crop_w, crop_h = target_size
    half_w, half_h = crop_w // 2, crop_h // 2

    # 计算裁剪框，并确保不超出图像范围
    left = max(center_x - half_w, 0)
    upper = max(center_y - half_h, 0)
    right = left + crop_w
    lower = upper + crop_h

    orig_w, orig_h = img.size
    if right > orig_w:
        right = orig_w
        left = max(right - crop_w, 0)
    if lower > orig_h:
        lower = orig_h
        upper = max(lower - crop_h, 0)

    cropped_img = img.crop((left, upper, right, lower))
    cropped_mask = mask[upper:lower, left:right]
    return cropped_img, cropped_mask


def get_transform(train, args):

    if train:
        return presets.DetectionPresetTrain(data_augmentation=args.data_augmentation)
    elif args.weights and args.test_only:
        print("enter the weights")
        weights = torchvision.models.get_weight(args.weights)
        trans = weights.transforms()
        return lambda img, target: (trans(img), target)
    else:
        return presets.DetectionPresetEval()

def get_dataset(name, image_set, transform, data_path ,use_gray=False):
    if name == "LV":
        ds = HeartDataset(root=data_path, transforms=transform,use_gray=use_gray)
        num_classes = 2  # 假設只有一個前景類別（左心室）
    elif name == "Myo":
        ds = MYODataset(root=data_path, transforms=transform,use_gray=use_gray)
        num_classes = 2  # 假設只有一個前景類別（心肌）
    else:
        ds = RV_HeartDataset(root=data_path, transforms=transform,use_gray=use_gray)
        num_classes = 2  # 假設只有一個前景類別（右心室）
    return ds, num_classes

def load_datasets(args, use_second_dataset=False):
    # 加载第一个数据集
    dataset1, num_classes1 = get_dataset(args.dataset1, "train", get_transform(True, args), args.data_path1,use_gray=args.gray)
    dataset_test1, _ = get_dataset(args.dataset1, "val", get_transform(False, args), args.data_path1,use_gray=args.gray)
    # 数据集划分
    train_size1 = int(0.8 * len(dataset1))
    val_size1 = len(dataset1) - train_size1
    train_dataset1, val_dataset1 = random_split(dataset1, [train_size1, val_size1])

    # 初始化第二个数据集
    train_dataset2, val_dataset2, num_classes2 = None, None, None

    # 仅在需要时加载第二个数据集
    if use_second_dataset:
        if args.data_path2 and args.dataset2:  # 检查路径和数据集名称是否存在
            dataset2, num_classes2 = get_dataset(args.dataset2, "train", get_transform(True, args), args.data_path2,use_gray=args.gray)
            dataset_test2, _ = get_dataset(args.dataset2, "val", get_transform(False, args), args.data_path2,use_gray=args.gray)

            train_size2 = int(0.8 * len(dataset2))
            val_size2 = len(dataset2) - train_size2
            train_dataset2, val_dataset2 = random_split(dataset2, [train_size2, val_size2])
        else:
            print("Warning: use_second_dataset is enabled but data_path2 or dataset2 is not specified.")

    return train_dataset1, val_dataset1, train_dataset2, val_dataset2, num_classes1, num_classes2

class MYODataset(Dataset):

    def __init__(self, root, transforms=None, target_size=(128, 128),use_gray=False, min_box_size=1,debug_show=False):
        self.root = root
        self.transforms = transforms
        self.target_size = target_size
        self.use_gray = use_gray
        self.min_box_size = min_box_size  # 小於此值就視為太小而跳過
        self.debug_show = debug_show  # 是否顯示裁剪後圖像與 mask
        # 初始化資料集，例如加载图像和掩码的路径
        self.imgs = list(sorted(os.listdir(os.path.join(root, "Myo_img"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "Myo_mask"))))
        #print("[DEBUG] imgs:", self.imgs)
        #print("[DEBUG] masks:", self.masks)
        self.samples = []
        for img_name, mask_name in zip(self.imgs, self.masks):
            img_path = os.path.join(root, "Myo_img", img_name)
            mask_path = os.path.join(root, "Myo_mask", mask_name)

            # 簡單檢查檔案是否存在
            if not os.path.isfile(img_path) or not os.path.isfile(mask_path):
                continue

            # 1) 先讀取 mask
            mask_pil = Image.open(mask_path)

            # 2) 先做跟訓練時相同的 Resize (NEAREST 用於 mask)
            mask_pil = mask_pil.resize(self.target_size, Image.NEAREST)

            # 3) 檢查前景物件
            arr = np.array(mask_pil)
            unique_vals = np.unique(arr)
            if len(unique_vals) <= 1:
                # 代表只有背景(0)
                continue


             # 4) 找出前景的 bounding box => 判斷物件是否太小
            #    這裡假設一張 mask 只標示一個前景(或把多個前景視為同一類)
            #    如果其實有多個物件，則要找 max box 或逐一檢查
            nonzero = np.where(arr > 0)
            y_min, y_max = nonzero[0].min(), nonzero[0].max()
            x_min, x_max = nonzero[1].min(), nonzero[1].max()
            w = x_max - x_min
            h = y_max - y_min

            # 你可以根據需要判斷 "過小" 的門檻。例如最小邊 < 10
            if w < self.min_box_size or h < self.min_box_size:
                continue

            # 如果通過所有條件，就視為合格樣本
            self.samples.append((img_name, mask_name))
            # 如果通過篩選，就加到 self.samples
            #print(f"[INFO] 最终保留 {len(self.samples)} 个样本：{self.samples}")
        # 這樣就只留下「有物件」的樣本在 self.samples 裡
    def __len__(self):
        return len(self.samples)    

    def __getitem__(self, idx):
        img_name,mask_name = self.samples[idx]
        img_path = os.path.join(self.root, "Myo_img", img_name)
        mask_path = os.path.join(self.root, "Myo_mask", mask_name)
    
    # 使用 PIL 打開圖像
        img = Image.open(img_path)
        # 如果是單通道 => convert("L")，否則 => convert("RGB")
        if self.use_gray:
            img = img.convert("L")  # 單通道
        else:
            img = img.convert("RGB")  # 3 通道

        # 讀取 mask (單通道)
        mask = Image.open(mask_path)
        if mask.mode != "P":
            mask = mask.convert("L")
            arr_gray = np.array(mask)
            arr_bin = (arr_gray > 0).astype(np.uint8)  # 大於 0 為前景
            mask = Image.fromarray(arr_bin, mode="P")
        # 設定調色盤：索引 0 為黑色（背景），索引 1 為紅色（前景）
            palette = []
            palette.extend([0, 0, 0])    # 0 -> 黑色
            palette.extend([255, 0, 0])  # 1 -> 紅色
            for _ in range(254):
                palette.extend([0, 0, 0])
            mask.putpalette(palette)

        orig_size = img.size  # (width, height)
        target_height, target_width = self.target_size


         # 如果原图尺寸明显大于目标尺寸（比如全尺寸数据），自动进行 ROI 裁剪
        if orig_size[0] > target_width * 1.5 or orig_size[1] > target_height * 1.5:
            # 先不 Resize，直接基于原图和原 mask 裁剪
            img, mask_np = crop_roi(img, np.array(mask), target_size=self.target_size)
            #print("successful crop")
            # mask_np 已经是裁剪后的 numpy 数组
        else:
            # 如果图像尺寸已经接近目标尺寸，则直接 Resize
            img = img.resize((target_width, target_height), Image.BILINEAR)
            mask = mask.resize((target_width, target_height), Image.NEAREST)
            mask_np = np.array(mask)
    # 若需要 debug 顯示，則同時顯示裁剪後的 image 與 mask
        '''if self.debug_show:
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.imshow(img)
            ax.imshow(mask_np, cmap="jet", alpha=0.5)  # 調整 alpha 控制遮罩透明度（0 表示全透明，1 表示不透明）
            ax.set_title("ROI Image with Mask Overlay")
            ax.axis("off")
            plt.show()'''
    # 最後將 PIL 圖像轉換為張量
        #img = F_tv.to_tensor(img)  # 將 PIL 圖像轉為張量
        #img_tensor = F.to_tensor(img)  # (C, H, W)
        mask = torch.as_tensor(np.array(mask_np), dtype=torch.int64)
    

    # 打印最終張量的尺寸
        #print(f"Final img shape (Tensor): {img.shape}")  # 打印張量格式的圖像尺寸
        #print(f"Final mask shape (Tensor): {mask.shape}")  # 打印張量格式的掩碼尺寸

    # 提取目標區域
        obj_ids = torch.unique(mask)[1:]  # 去掉背景（假設背景為0）
        if len(obj_ids) == 0:
            print(f"Skipping image {idx}: No objects found in mask.")
            return self.__getitem__((idx + 1) % len(self.samples))
        masks = mask == obj_ids[:, None, None]

    # 計算每個物體的邊界框，並根據圖像的縮放比例進行調整
        boxes = []
        valid_masks = []
    
        for i in range(len(obj_ids)):
            pos = torch.where(masks[i])
            if pos[0].numel() == 0 or pos[1].numel() == 0:
                print(f"Warning: object {i} in sample {idx} has no nonzero pixels, skipping object.")
                continue
            xmin = torch.min(pos[1])
            xmax = torch.max(pos[1])
            ymin = torch.min(pos[0])
            ymax = torch.max(pos[0])
            boxes.append([xmin.item(), ymin.item(), xmax.item(), ymax.item()])
            valid_masks.append(masks[i])

        if len(boxes) == 0:
            print(f"Skipping sample {idx} ({img_name}): No valid objects after processing.")
            return self.__getitem__((idx + 1) % len(self.samples))
        if self.debug_show:
            fig, ax = plt.subplots(figsize=(8, 8))
    # 顯示 ROI 圖
            ax.imshow(img)
    # 以透明色疊上 mask
            ax.imshow(mask_np, cmap="jet", alpha=0.5)
    # 為每個 bounding box 加上矩形
            for box in boxes:
                xmin, ymin, xmax, ymax = box
                width = xmax - xmin
                height = ymax - ymin
                rect = patches.Rectangle((xmin, ymin), width, height, linewidth=2, edgecolor='green', facecolor='none')
                ax.add_patch(rect)
            ax.set_title("ROI Image with Mask & Bounding Boxes")
            ax.axis("off")
            plt.show()
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones((len(boxes),), dtype=torch.int64)  # 假設所有物體標籤為 1（心臟）
        image_id = torch.tensor([idx])
    

     # 合并所有有效的 masks
        valid_masks = torch.stack(valid_masks, dim=0)

        target = {"boxes": boxes, "labels": labels, "masks": valid_masks, "image_id": image_id}
        '''if self.use_gray:
            # convert single channel PIL => Tensor shape=(H,W)，再加 batch dim
            img_tensor = F.to_tensor(img)  # shape=(1,H,W)
        else:
            img_tensor = F.to_tensor(img)  # shape=(3,H,W)'''

        if self.transforms:
            img, target = self.transforms(img, target)

        return img, target
    

class HeartDataset(Dataset):

    def __init__(self, root, transforms=None, target_size=(128, 128),use_gray=False, min_box_size=1,debug_show=False):
        self.root = root
        self.transforms = transforms
        self.target_size = target_size
        self.use_gray = use_gray
        self.min_box_size = min_box_size  # 小於此值就視為太小而跳過
        self.debug_show = debug_show  # 是否顯示裁剪後圖像與 mask
        # 初始化資料集，例如加载图像和掩码的路径
        self.imgs = list(sorted(os.listdir(os.path.join(root, "LV_img"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "LV_mask"))))
        #print("[DEBUG] imgs:", self.imgs)
        #print("[DEBUG] masks:", self.masks)
        self.samples = []
        for img_name, mask_name in zip(self.imgs, self.masks):
            img_path = os.path.join(root, "LV_img", img_name)
            mask_path = os.path.join(root, "LV_mask", mask_name)

            # 簡單檢查檔案是否存在
            if not os.path.isfile(img_path) or not os.path.isfile(mask_path):
                continue

            # 1) 先讀取 mask
            mask_pil = Image.open(mask_path)

            # 2) 先做跟訓練時相同的 Resize (NEAREST 用於 mask)
            mask_pil = mask_pil.resize(self.target_size, Image.NEAREST)

            # 3) 檢查前景物件
            arr = np.array(mask_pil)
            unique_vals = np.unique(arr)
            if len(unique_vals) <= 1:
                # 代表只有背景(0)
                continue


             # 4) 找出前景的 bounding box => 判斷物件是否太小
            #    這裡假設一張 mask 只標示一個前景(或把多個前景視為同一類)
            #    如果其實有多個物件，則要找 max box 或逐一檢查
            nonzero = np.where(arr > 0)
            y_min, y_max = nonzero[0].min(), nonzero[0].max()
            x_min, x_max = nonzero[1].min(), nonzero[1].max()
            w = x_max - x_min
            h = y_max - y_min

            # 你可以根據需要判斷 "過小" 的門檻。例如最小邊 < 10
            if w < self.min_box_size or h < self.min_box_size:
                continue

            # 如果通過所有條件，就視為合格樣本
            self.samples.append((img_name, mask_name))
            # 如果通過篩選，就加到 self.samples
            #print(f"[INFO] 最终保留 {len(self.samples)} 个样本：{self.samples}")
        # 這樣就只留下「有物件」的樣本在 self.samples 裡
    def __len__(self):
        return len(self.samples)    

    def __getitem__(self, idx):
        img_name,mask_name = self.samples[idx]
        img_path = os.path.join(self.root, "LV_img", img_name)
        mask_path = os.path.join(self.root, "LV_mask", mask_name)
    
    # 使用 PIL 打開圖像
        img = Image.open(img_path)
        # 如果是單通道 => convert("L")，否則 => convert("RGB")
        if self.use_gray:
            img = img.convert("L")  # 單通道
        else:
            img = img.convert("RGB")  # 3 通道

        # 讀取 mask (單通道)
        mask = Image.open(mask_path)
        if mask.mode != "P":
            mask = mask.convert("L")
            arr_gray = np.array(mask)
            arr_bin = (arr_gray > 0).astype(np.uint8)  # 大於 0 為前景
            mask = Image.fromarray(arr_bin, mode="P")
        # 設定調色盤：索引 0 為黑色（背景），索引 1 為紅色（前景）
            palette = []
            palette.extend([0, 0, 0])    # 0 -> 黑色
            palette.extend([255, 0, 0])  # 1 -> 紅色
            for _ in range(254):
                palette.extend([0, 0, 0])
            mask.putpalette(palette)

        orig_size = img.size  # (width, height)
        target_height, target_width = self.target_size


         # 如果原图尺寸明显大于目标尺寸（比如全尺寸数据），自动进行 ROI 裁剪
        if orig_size[0] > target_width * 1.5 or orig_size[1] > target_height * 1.5:
            # 先不 Resize，直接基于原图和原 mask 裁剪
            img, mask_np = crop_roi(img, np.array(mask), target_size=self.target_size)
            #print("successful crop")
            # mask_np 已经是裁剪后的 numpy 数组
        else:
            # 如果图像尺寸已经接近目标尺寸，则直接 Resize
            img = img.resize((target_width, target_height), Image.BILINEAR)
            mask = mask.resize((target_width, target_height), Image.NEAREST)
            mask_np = np.array(mask)
    # 若需要 debug 顯示，則同時顯示裁剪後的 image 與 mask
        '''if self.debug_show:
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.imshow(img)
            ax.imshow(mask_np, cmap="jet", alpha=0.5)  # 調整 alpha 控制遮罩透明度（0 表示全透明，1 表示不透明）
            ax.set_title("ROI Image with Mask Overlay")
            ax.axis("off")
            plt.show()'''
    # 最後將 PIL 圖像轉換為張量
        #img = F_tv.to_tensor(img)  # 將 PIL 圖像轉為張量
        #img_tensor = F.to_tensor(img)  # (C, H, W)
        mask = torch.as_tensor(np.array(mask_np), dtype=torch.int64)
    

    # 打印最終張量的尺寸
        #print(f"Final img shape (Tensor): {img.shape}")  # 打印張量格式的圖像尺寸
        #print(f"Final mask shape (Tensor): {mask.shape}")  # 打印張量格式的掩碼尺寸

    # 提取目標區域
        obj_ids = torch.unique(mask)[1:]  # 去掉背景（假設背景為0）
        if len(obj_ids) == 0:
            print(f"Skipping image {idx}: No objects found in mask.")
            return self.__getitem__((idx + 1) % len(self.samples))
        masks = mask == obj_ids[:, None, None]

    # 計算每個物體的邊界框，並根據圖像的縮放比例進行調整
        boxes = []
        valid_masks = []
    
        for i in range(len(obj_ids)):
            pos = torch.where(masks[i])
            if pos[0].numel() == 0 or pos[1].numel() == 0:
                print(f"Warning: object {i} in sample {idx} has no nonzero pixels, skipping object.")
                continue
            xmin = torch.min(pos[1])
            xmax = torch.max(pos[1])
            ymin = torch.min(pos[0])
            ymax = torch.max(pos[0])
            boxes.append([xmin.item(), ymin.item(), xmax.item(), ymax.item()])
            valid_masks.append(masks[i])

        if len(boxes) == 0:
            print(f"Skipping sample {idx} ({img_name}): No valid objects after processing.")
            return self.__getitem__((idx + 1) % len(self.samples))
        if self.debug_show:
            fig, ax = plt.subplots(figsize=(8, 8))
    # 顯示 ROI 圖
            ax.imshow(img)
    # 以透明色疊上 mask
            ax.imshow(mask_np, cmap="jet", alpha=0.5)
    # 為每個 bounding box 加上矩形
            for box in boxes:
                xmin, ymin, xmax, ymax = box
                width = xmax - xmin
                height = ymax - ymin
                rect = patches.Rectangle((xmin, ymin), width, height, linewidth=2, edgecolor='green', facecolor='none')
                ax.add_patch(rect)
            ax.set_title("ROI Image with Mask & Bounding Boxes")
            ax.axis("off")
            plt.show()
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones((len(boxes),), dtype=torch.int64)  # 假設所有物體標籤為 1（心臟）
        image_id = torch.tensor([idx])
    

     # 合并所有有效的 masks
        valid_masks = torch.stack(valid_masks, dim=0)

        target = {"boxes": boxes, "labels": labels, "masks": valid_masks, "image_id": image_id}
        '''if self.use_gray:
            # convert single channel PIL => Tensor shape=(H,W)，再加 batch dim
            img_tensor = F.to_tensor(img)  # shape=(1,H,W)
        else:
            img_tensor = F.to_tensor(img)  # shape=(3,H,W)'''

        if self.transforms:
            img, target = self.transforms(img, target)

        return img, target

class RV_HeartDataset(Dataset):

    def __init__(self, root, transforms=None, target_size=(128, 128),use_gray=False, min_box_size=1,debug_show=False):
        self.root = root
        self.transforms = transforms
        self.target_size = target_size
        self.use_gray = use_gray
        self.min_box_size = min_box_size  # 小於此值就視為太小而跳過
        self.debug_show = debug_show  # 是否顯示裁剪後圖像與 mask
        # 初始化資料集，例如加载图像和掩码的路径
        self.imgs = list(sorted(os.listdir(os.path.join(root, "RV_img"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "RV_mask"))))
        #print("[DEBUG] imgs:", self.imgs)
        #print("[DEBUG] masks:", self.masks)
        self.samples = []
        for img_name, mask_name in zip(self.imgs, self.masks):
            img_path = os.path.join(root, "RV_img", img_name)
            mask_path = os.path.join(root, "RV_mask", mask_name)

            # 簡單檢查檔案是否存在
            if not os.path.isfile(img_path) or not os.path.isfile(mask_path):
                continue

            # 1) 先讀取 mask
            mask_pil = Image.open(mask_path)

            # 2) 先做跟訓練時相同的 Resize (NEAREST 用於 mask)
            mask_pil = mask_pil.resize(self.target_size, Image.NEAREST)

            # 3) 檢查前景物件
            arr = np.array(mask_pil)
            unique_vals = np.unique(arr)
            if len(unique_vals) <= 1:
                # 代表只有背景(0)
                continue


             # 4) 找出前景的 bounding box => 判斷物件是否太小
            #    這裡假設一張 mask 只標示一個前景(或把多個前景視為同一類)
            #    如果其實有多個物件，則要找 max box 或逐一檢查
            nonzero = np.where(arr > 0)
            y_min, y_max = nonzero[0].min(), nonzero[0].max()
            x_min, x_max = nonzero[1].min(), nonzero[1].max()
            w = x_max - x_min
            h = y_max - y_min

            # 你可以根據需要判斷 "過小" 的門檻。例如最小邊 < 10
            if w < self.min_box_size or h < self.min_box_size:
                continue

            # 如果通過所有條件，就視為合格樣本
            self.samples.append((img_name, mask_name))
            # 如果通過篩選，就加到 self.samples
            #print(f"[INFO] 最终保留 {len(self.samples)} 个样本：{self.samples}")
        # 這樣就只留下「有物件」的樣本在 self.samples 裡
    def __len__(self):
        return len(self.samples)    

    def __getitem__(self, idx):
        img_name,mask_name = self.samples[idx]
        img_path = os.path.join(self.root, "RV_img", img_name)
        mask_path = os.path.join(self.root, "RV_mask", mask_name)
    
    # 使用 PIL 打開圖像
        img = Image.open(img_path)
        # 如果是單通道 => convert("L")，否則 => convert("RGB")
        if self.use_gray:
            img = img.convert("L")  # 單通道
        else:
            img = img.convert("RGB")  # 3 通道

        # 讀取 mask (單通道)
        mask = Image.open(mask_path)
        if mask.mode != "P":
            mask = mask.convert("L")
            arr_gray = np.array(mask)
            arr_bin = (arr_gray > 0).astype(np.uint8)  # 大於 0 為前景
            mask = Image.fromarray(arr_bin, mode="P")
        # 設定調色盤：索引 0 為黑色（背景），索引 1 為紅色（前景）
            palette = []
            palette.extend([0, 0, 0])    # 0 -> 黑色
            palette.extend([255, 0, 0])  # 1 -> 紅色
            for _ in range(254):
                palette.extend([0, 0, 0])
            mask.putpalette(palette)

        orig_size = img.size  # (width, height)
        target_height, target_width = self.target_size


         # 如果原图尺寸明显大于目标尺寸（比如全尺寸数据），自动进行 ROI 裁剪
        if orig_size[0] > target_width * 1.5 or orig_size[1] > target_height * 1.5:
            # 先不 Resize，直接基于原图和原 mask 裁剪
            img, mask_np = crop_roi(img, np.array(mask), target_size=self.target_size)
            #print("successful crop")
            # mask_np 已经是裁剪后的 numpy 数组
        else:
            # 如果图像尺寸已经接近目标尺寸，则直接 Resize
            img = img.resize((target_width, target_height), Image.BILINEAR)
            mask = mask.resize((target_width, target_height), Image.NEAREST)
            mask_np = np.array(mask)
    # 若需要 debug 顯示，則同時顯示裁剪後的 image 與 mask
        '''if self.debug_show:
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.imshow(img)
            ax.imshow(mask_np, cmap="jet", alpha=0.5)  # 調整 alpha 控制遮罩透明度（0 表示全透明，1 表示不透明）
            ax.set_title("ROI Image with Mask Overlay")
            ax.axis("off")
            plt.show()'''
    # 最後將 PIL 圖像轉換為張量
        #img = F_tv.to_tensor(img)  # 將 PIL 圖像轉為張量
        #img_tensor = F.to_tensor(img)  # (C, H, W)
        mask = torch.as_tensor(np.array(mask_np), dtype=torch.int64)
    

    # 打印最終張量的尺寸
        #print(f"Final img shape (Tensor): {img.shape}")  # 打印張量格式的圖像尺寸
        #print(f"Final mask shape (Tensor): {mask.shape}")  # 打印張量格式的掩碼尺寸

    # 提取目標區域
        obj_ids = torch.unique(mask)[1:]  # 去掉背景（假設背景為0）
        if len(obj_ids) == 0:
            print(f"Skipping image {idx}: No objects found in mask.")
            return self.__getitem__((idx + 1) % len(self.samples))
        masks = mask == obj_ids[:, None, None]

    # 計算每個物體的邊界框，並根據圖像的縮放比例進行調整
        boxes = []
        valid_masks = []
    
        for i in range(len(obj_ids)):
            pos = torch.where(masks[i])
            if pos[0].numel() == 0 or pos[1].numel() == 0:
                print(f"Warning: object {i} in sample {idx} has no nonzero pixels, skipping object.")
                continue
            xmin = torch.min(pos[1])
            xmax = torch.max(pos[1])
            ymin = torch.min(pos[0])
            ymax = torch.max(pos[0])
            boxes.append([xmin.item(), ymin.item(), xmax.item(), ymax.item()])
            valid_masks.append(masks[i])

        if len(boxes) == 0:
            print(f"Skipping sample {idx} ({img_name}): No valid objects after processing.")
            return self.__getitem__((idx + 1) % len(self.samples))
        if self.debug_show:
            fig, ax = plt.subplots(figsize=(8, 8))
    # 顯示 ROI 圖
            ax.imshow(img)
    # 以透明色疊上 mask
            ax.imshow(mask_np, cmap="jet", alpha=0.5)
    # 為每個 bounding box 加上矩形
            for box in boxes:
                xmin, ymin, xmax, ymax = box
                width = xmax - xmin
                height = ymax - ymin
                rect = patches.Rectangle((xmin, ymin), width, height, linewidth=2, edgecolor='green', facecolor='none')
                ax.add_patch(rect)
            ax.set_title("ROI Image with Mask & Bounding Boxes")
            ax.axis("off")
            plt.show()
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones((len(boxes),), dtype=torch.int64)  # 假設所有物體標籤為 1（心臟）
        image_id = torch.tensor([idx])
    

     # 合并所有有效的 masks
        valid_masks = torch.stack(valid_masks, dim=0)

        target = {"boxes": boxes, "labels": labels, "masks": valid_masks, "image_id": image_id}
        '''if self.use_gray:
            # convert single channel PIL => Tensor shape=(H,W)，再加 batch dim
            img_tensor = F.to_tensor(img)  # shape=(1,H,W)
        else:
            img_tensor = F.to_tensor(img)  # shape=(3,H,W)'''

        if self.transforms:
            img, target = self.transforms(img, target)

        return img, target