# ContextualLoss-PyTorch

## Overview

This repository contains an op-for-op PyTorch reimplementation
of [Maintaining Natural Image Statistics with the Contextual Loss](https://arxiv.org/abs/1803.04626).

## Table of contents

- [ContextualLoss-PyTorch](#contextualloss-pytorch)
    - [Overview](#overview)
    - [Table of contents](#table-of-contents)
    - [Download weights](#download-weights)
    - [Download datasets](#download-datasets)
    - [How Test and Train](#how-test-and-train)
        - [Test SRGAN_CX](#test-srgan_cx)
        - [Train SRResNet_CX model](#train-srresnet_cx-model)
        - [Resume train SRResNet model](#resume-train-srresnet_cx-model)
        - [Train SRGAN_CX model](#train-srgan_cx-model)
        - [Resume train SRGAN_CX model](#resume-train-srgan_cx-model)
    - [Result](#result)
    - [Contributing](#contributing)
    - [Credit](#credit)
        - [Maintaining Natural Image Statistics with the Contextual Loss](#maintaining-natural-image-statistics-with-the-contextual-loss)

## Download weights

- [Google Driver](https://drive.google.com/drive/folders/17ju2HN7Y6pyPK2CC_AqnAfTOe9_3hCQ8?usp=sharing)
- [Baidu Driver](https://pan.baidu.com/s/1yNs4rqIb004-NKEdKBJtYg?pwd=llot)

## Download datasets

Contains DIV2K, DIV8K, Flickr2K, OST, T91, Set5, Set14, BSDS100 and BSDS200, etc.

- [Google Driver](https://drive.google.com/drive/folders/1A6lzGeQrFMxPqJehK9s37ce-tPDj20mD?usp=sharing)
- [Baidu Driver](https://pan.baidu.com/s/1o-8Ty_7q6DiS3ykLU09IVg?pwd=llot)

Please refer to `README.md` in the `data` directory for the method of making a dataset.

## How Test and Train

Both training and testing only need to modify the `srresnet_config.py` file and `srgan_config.py` file.

### Test SRGAN_CX

Modify the `srgan_config.py` file.

- line 32: `g_arch_name` change to `srresnet_x4`.
- line 39: `upscale_factor` change to `4`.
- line 41: `mode` change to `test`.
- line 43: `exp_name` change to `SRGAN_CX_x4-DIV2K`.
- line 96: `g_model_weights_path` change to `./results/pretrained_models/SRGAN_CX_x4-DIV2K-8c4a7569.pth.tar`.

```bash
python3 test.py
```

### Train SRResNet_CX model

Modify the `srresnet_config.py` file.

- line 32: `g_arch_name` change to `srresnet_x4`.
- line 39: `upscale_factor` change to `4`.
- line 41: `mode` change to `train`.
- line 43: `exp_name` change to `SRResNet_CX_x4-DIV2K`.

```bash
python3 train_srresnet.py
```

### Resume train SRResNet_CX model

Modify the `srresnet_config.py` file.

- line 32: `g_arch_name` change to `srresnet_x4`.
- line 39: `upscale_factor` change to `4`.
- line 41: `mode` change to `train`.
- line 43: `exp_name` change to `SRResNet_CX_x4-DIV2K`.
- line 59: `resume_g_model_weights_path` change to `./samples/SRGAN_CX_x4-DIV2K/g_epoch_xxx.pth.tar`.

```bash
python3 train_srresnet.py
```

### Train SRGAN_CX model

- line 31: `d_arch_name` change to `discriminator`.
- line 32: `g_arch_name` change to `srresnet_x4`.
- line 39: `upscale_factor` change to `4`.
- line 41: `mode` change to `train`.
- line 43: `exp_name` change to `SRGAN_CX_x4-DIV2K`.
- line 58: `pretrained_g_model_weights_path` change to `./results/SRResNet_CX_x4-DIV2K/g_last.pth.tar`.

```bash
python3 train_srgan.py
```

### Resume train SRGAN_CX model

- line 31: `d_arch_name` change to `discriminator`.
- line 32: `g_arch_name` change to `srresnet_x4`.
- line 39: `upscale_factor` change to `4`.
- line 41: `mode` change to `train`.
- line 43: `exp_name` change to `SRGAN_CX_x4-DIV2K`.
- line 61: `resume_d_model_weights_path` change to `./samples/SRGAN_CX_x4-DIV2K/d_epoch_xxx.pth.tar`.
- line 62: `resume_g_model_weights_path` change to `./samples/SRGAN_CX_x4-DIV2K/g_epoch_xxx.pth.tar`.

```bash
python3 train_srgan.py
```

## Result

Source of original paper results: [https://arxiv.org/pdf/1803.04626.pdf](https://arxiv.org/pdf/1803.04626.pdf)

In the following table, the psnr value in `()` indicates the result of the project, and `-` indicates no test.

| Set5 | Scale |  SRResNet_CX  |   SRGAN_CX    |
|:----:|:-----:|:-------------:|:-------------:|
| PSNR |   4   | -(**32.14**)  | -(**30.64**)  |
| SSIM |   4   | -(**0.8954**) | -(**0.8642**) |

| Set14 | Scale |  SRResNet_CX  |   SRGAN_CX    |
|:-----:|:-----:|:-------------:|:-------------:|
| PSNR  |   4   | -(**28.57**)  | -(**27.12**)  |
| SSIM  |   4   | -(**0.7815**) | -(**0.7321**) |

```bash
# Download `SRGAN_CX_x4-DIV2K-8c4a7569.pth.tar` weights to `./results/pretrained_models`
# More detail see `README.md<Download weights>`
python3 ./inference.py
```

Input:

<span align="center"><img width="240" height="360" src="figure/comic_lr.png"/></span>

Output:

<span align="center"><img width="240" height="360" src="figure/comic_sr.png"/></span>

```text
Build `srresnet_x4` model successfully.
Load `srresnet_x4` model weights `./results/pretrained_models/SRGAN_CX_x4-DIV2K-8c4a7569.pth.tar` successfully.
SR image save to `./figure/comic_sr.png`
```

## Contributing

If you find a bug, create a GitHub issue, or even better, submit a pull request. Similarly, if you have questions,
simply post them as GitHub issues.

I look forward to seeing what the community does with these models!

## Credit

### Maintaining Natural Image Statistics with the Contextual Loss

_Roey Mechrez, Itamar Talmi, Firas Shama, Lihi Zelnik-Manor_ <br>

**Abstract** <br>
Maintaining natural image statistics is a crucial factor in restoration and generation of realistic looking images. When
training CNNs, photorealism is usually attempted by adversarial training (GAN), that pushes the output images to lie on
the manifold of natural images. GANs are very powerful, but not perfect. They are hard to train and the results still
often suffer from artifacts. In this paper we propose a complementary approach, that could be applied with or without
GAN, whose goal is to train a feed-forward CNN to maintain natural internal statistics. We look explicitly at the
distribution of features in an image and train the network to generate images with natural feature distributions. Our
approach reduces by orders of magnitude the number of images required for training and achieves state-of-the-art results
on both single-image super-resolution, and high-resolution surface normal estimation.

[[Paper]](https://arxiv.org/pdf/1803.04626.pdf)

```bibtex
@inproceedings{mechrez2018maintaining,
  title={Maintaining natural image statistics with the contextual loss},
  author={Mechrez, Roey and Talmi, Itamar and Shama, Firas and Zelnik-Manor, Lihi},
  booktitle={Asian Conference on Computer Vision},
  pages={427--443},
  year={2018},
  organization={Springer}
}

```
