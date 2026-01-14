"""A medical image analysis pipeline.

The pipeline is used for brain tissue segmentation using a 2.5D U-Net.
"""
import argparse
import datetime
import os
import sys
import timeit
import warnings

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import pymia.data.conversion as conversion
import pymia.evaluation.writer as writer

try:
    import mialab.data.structure as structure
    import mialab.utilities.file_access_utilities as futil
    import mialab.utilities.pipeline_utilities as putil
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(sys.argv[0]), '.'))
    import mialab.data.structure as structure
    import mialab.utilities.file_access_utilities as futil
    import mialab.utilities.pipeline_utilities as putil


LOADING_KEYS = [
    structure.BrainImageTypes.T1w,
    structure.BrainImageTypes.T2w,
    structure.BrainImageTypes.GroundTruth,
    structure.BrainImageTypes.BrainMask,
    structure.BrainImageTypes.RegistrationTransform
]


# -------------------------
# 2.5D DATASET
# -------------------------
class BrainSliceDataset(Dataset):
    """2.5D dataset: uses neighboring slices as channels"""

    def __init__(self, images, training=True):
        self.samples = []
        self.training = training

        for img in images:
            t1 = sitk.GetArrayFromImage(img.images[structure.BrainImageTypes.T1w])
            t2 = sitk.GetArrayFromImage(img.images[structure.BrainImageTypes.T2w])
            gt = sitk.GetArrayFromImage(img.images[structure.BrainImageTypes.GroundTruth])

            for z in range(1, t1.shape[0] - 1):
                x = np.stack([
                    t1[z - 1],
                    t1[z],
                    t1[z + 1],
                    t2[z]
                ], axis=0)

                y = gt[z]
                self.samples.append((x.astype(np.float32), y.astype(np.int64)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.from_numpy(x), torch.from_numpy(y)


# -------------------------
# SIMPLE 2.5D U-NET
# -------------------------
class UNet2D(nn.Module):
    """Minimal 2D U-Net for 2.5D input"""

    def __init__(self, in_channels=4, num_classes=6):
        super().__init__()

        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(cout, cout, 3, padding=1),
                nn.ReLU(inplace=True)
            )

        self.enc1 = block(in_channels, 32)
        self.enc2 = block(32, 64)
        self.pool = nn.MaxPool2d(2)

        self.dec1 = block(64, 32)
        self.final = nn.Conv2d(32, num_classes, 1)

    def forward(self, x):
        x1 = self.enc1(x)  # [B, 32, H, W]
        x2 = self.enc2(self.pool(x1))  # [B, 64, H/2, W/2]

        x3 = nn.functional.interpolate(
            x2,
            size=x1.shape[2:],  # 🔥 FIX
            mode='bilinear',
            align_corners=False
        )

        x4 = self.dec1(x3)
        return self.final(x4)


# -------------------------
# MAIN PIPELINE
# -------------------------
def main(result_dir: str, data_atlas_dir: str, data_train_dir: str, data_test_dir: str):
    """Brain tissue segmentation using a 2.5D U-Net."""

    # load atlas images
    putil.load_atlas_images(data_atlas_dir)

    print('-' * 5, 'Training.')

    crawler = futil.FileSystemDataCrawler(
        data_train_dir,
        LOADING_KEYS,
        futil.BrainImageFilePathGenerator(),
        futil.DataDirectoryFilter()
    )

    pre_process_params = {
        'registration_pre': True,
        'skullstrip_pre': True,
        'biasfield_pre': True,
        'normalization_pre': True,
        'training': True
    }

    images = putil.pre_process_batch(
        crawler.data, pre_process_params, multi_process=False
    )

    # Dataset & loader
    train_ds = BrainSliceDataset(images, training=True)
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet2D().to(device)

    # ---- class-weighted loss (important for small structures)
    class_weights = torch.tensor(
        [0.05, 1.0, 1.0, 5.0, 5.0, 5.0],
        dtype=torch.float32
    ).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    start_time = timeit.default_timer()

    model.train()
    for epoch in range(10):
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

        print(f'Epoch {epoch + 1}/10 done')

    print(' Time elapsed:', timeit.default_timer() - start_time, 's')

    # -------------------------
    # TESTING
    # -------------------------
    print('-' * 5, 'Testing.')

    evaluator = putil.init_evaluator()

    crawler = futil.FileSystemDataCrawler(
        data_test_dir,
        LOADING_KEYS,
        futil.BrainImageFilePathGenerator(),
        futil.DataDirectoryFilter()
    )

    pre_process_params['training'] = False
    images_test = putil.pre_process_batch(
        crawler.data, pre_process_params, multi_process=False
    )

    model.eval()
    with torch.no_grad():
        for img in images_test:
            print('-' * 10, 'Testing', img.id_)

            t1 = sitk.GetArrayFromImage(img.images[structure.BrainImageTypes.T1w])
            t2 = sitk.GetArrayFromImage(img.images[structure.BrainImageTypes.T2w])

            pred = np.zeros_like(t1, dtype=np.uint8)

            for z in range(1, t1.shape[0] - 1):
                x = np.stack([
                    t1[z - 1],
                    t1[z],
                    t1[z + 1],
                    t2[z]
                ], axis=0).astype(np.float32)

                x = torch.from_numpy(x[None]).to(device)
                out = model(x)
                pred[z] = out.argmax(1).cpu().numpy()[0]

            seg_img = conversion.NumpySimpleITKImageBridge.convert(
                pred, img.image_properties
            )

            evaluator.evaluate(
                seg_img,
                img.images[structure.BrainImageTypes.GroundTruth],
                img.id_
            )

    # write results
    t = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    result_dir = os.path.join(result_dir, t)
    os.makedirs(result_dir, exist_ok=True)

    result_file = os.path.join(result_dir, 'results.csv')
    writer.CSVWriter(result_file).write(evaluator.results)
    writer.ConsoleWriter().write(evaluator.results)


if __name__ == "__main__":
    """The program's entry point."""

    script_dir = os.path.dirname(sys.argv[0])

    parser = argparse.ArgumentParser(
        description='Medical image analysis pipeline for brain tissue segmentation'
    )

    parser.add_argument('--result_dir', type=str,
                        default=os.path.join(script_dir, './mia-result'))
    parser.add_argument('--data_atlas_dir', type=str,
                        default=os.path.join(script_dir, '../data/atlas'))
    parser.add_argument('--data_train_dir', type=str,
                        default=os.path.join(script_dir, '../data/train'))
    parser.add_argument('--data_test_dir', type=str,
                        default=os.path.join(script_dir, '../data/test'))

    args = parser.parse_args()
    main(args.result_dir, args.data_atlas_dir,
         args.data_train_dir, args.data_test_dir)
