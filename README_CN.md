# StableSR - Stable Diffusion WebUI

S-Lab License 1.0 & [![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]

[English](README.md) | 中文

- StableSR 是原初由 Jianyi Wang 等人提出的具有竞争力的超分辨率方法。
- 本仓库是将 StableSR 项目迁移到 Automatic1111 WebUI 的迁移工作。

相关链接

> 点击查看高质量官方示例！

- [项目页面](https://iceclear.github.io/projects/stablesr/)
- [官方仓库](https://github.com/IceClear/StableSR)
- [arXiv 上的论文](https://arxiv.org/abs/2305.07015)

> 如果你觉得这个项目有用，请给我和 Jianyi Wang 点个赞！⭐
---
## 使用

### 1. 安装

⚪ 方法 1: URL 安装

- 打开 Automatic1111 WebUI -> 点击 "扩展" 标签页 -> 点击 "从 URL 安装" 标签页 -> 输入 https://github.com/pkuliyi2015/sd-webui-stablesr.git -> 点击 "安装"

![installation](https://github.com/pkuliyi2015/multidiffusion-img-demo/blob/master/installation.png?raw=true)

⚪ 方法 2: 进行中...

> 安装成功后，你应该能在 img2img 脚本下拉列表中看到 "StableSR"。

### 2. 下载主要组件

- 你必须使用来自 StabilityAI 的 Stable Diffusion V2.1 512 **EMA** 检查点（大约 5.21GB）
    - 你可以从 [HuggingFace](https://huggingface.co/stabilityai/stable-diffusion-2-1-base) 下载它
    - 放入 stable-diffusion-webui/models/Stable-Diffusion/
- 下载剪枝后的 StableSR 模块（大约 400MB）
    - 官方资源：进行中
    - 我的资源：<[GoogleDrive](https://drive.google.com/file/d/1tWjkZQhfj07sHDR4r9Ta5Fk4iMp1t3Qw/view?usp=sharing)> <[百度网盘-提取码aguq](https://pan.baidu.com/s/1Nq_6ciGgKnTu0W14QcKKWg?pwd=aguq)>
    - 放入 stable-diffusion-webui/extensions/sd-webui-stablesr/models/

### 3. 可选组件

- 安装 [Tiled Diffusion & VAE](https://github.com/pkuliyi2015/multidiffusion-upscaler-for-automatic1111) 扩展
    - 原始的 StableSR 对大于 512 的大图像容易出现 OOM。
    - 为了获得更好的质量和更少的 VRAM 使用，我们建议使用 Tiled Diffusion & VAE。
- 使用官方 VQGAN VAE（大约 700MB）
    - 官方资源：进行中
    - 我的资源：<[GoogleDrive](https://drive.google.com/file/d/1ARtDMia3_CbwNsGxxGcZ5UP75W4PeIEI/view?usp=share_link)> <[百度网盘-提取码83u9](https://pan.baidu.com/s/1YCYmGBethR9JZ8-eypoIiQ?pwd=83u9)>
    - 将它放在你的 stable-diffusion-webui/models/VAE 中

### 4. 扩展使用

- 在 WebUI 的顶部，选择你下载的 v2-1_512-ema-pruned 检查点。
- 切换到 img2img 标签。在页面底部找到 "脚本" 下拉列表。
    - 选择 StableSR 脚本。
    - 点击刷新按钮并选择你已下载的 StableSR 检查点。
    - 选择一个比例因子。
- 上传你的图像并开始生成（无需提示）。

### 5. 有用的提示

- 推荐使用 Euler 采样器。步数 >= 20。
- 对于输出图像大小 > 512，我们推荐使用 Tiled Diffusion & VAE，否则，图像质量可能不理想，VRAM 使用量会很大。
- 这里有一些 Tiled Diffusion 设置，可以复制论文中的官方行为。
    - 方法 = Diffusers 混合
    - 隐变量瓷砖大小 = 64，隐变量瓷砖重叠 = 32
    - 隐变量瓷砖批大小尽可能大，避免内存不足。
    - 上采样器必须为 None。
- 什么是 "纯噪声"？
    - 纯噪声指的是从完全随机的噪声张量开始，而不是从你的图像开始。**这是 StableSR 论文中的默认行为。**
    - 启用时，脚本会忽略你的去噪强度，并给你更详细的图像，但也会显著改变颜色和锐度
    - 禁用时，脚本会开始添加一些噪声到你的图像。即使你将去噪强度设为 1，结果也不会完全详细（但可能在美感上更好）。参见 [对比](https://imgsli.com/MTgwMTMx)。

### 6. 重要提醒

> 为什么我的结果和官方示例不同？

- 这不是你或我们的错。
    - 如果正确安装，这个扩展有与 StableSR 相同的 UNet 模型权重。
    - 如果你安装了可选的 VQVAE，整个模型权重将与融合权重为 0 的官方模型相同。
- 但是，你的结果将**不如**官方结果，因为：
    - 采样器差异：
        -官方仓库进行 100 或 200 步的 legacy DDPM 采样，并使用自定义的时间步调度器，采样时不使用负提示。
        - 然而，WebUI 不提供这样的采样器，必须带有负提示进行采样。**这是主要的差异。**
    - VQVAE 解码器差异：
        - 官方 VQVAE 解码器将一些编码器特征作为输入。
        - 然而，在实践中，我发现这些特征对于大图像来说非常大。 (>10G 用于 4k 图像，即使是在 float16！)
        - 因此，**我移除了 VAE 解码器中的 CFW 组件**。由于这导致了对细节的较低保真度，我将尝试将它作为一个选项添加回去。

---
## 许可

此项目在以下许可下授权：

- S-Lab License 1.0.
- [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa]，由于使用了 NVIDIA SPADE 模块。

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]
[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg

### 免责声明

- 此扩展中的所有代码仅供研究目的。
- 代码和检查点的商业用途**严格禁止**。

### 成果图像的重要通知

- 请注意，NVIDIA SPADE 模块中的 CC BY-NC-SA 4.0 许可也禁止使用成果图像进行商业用途。
- Jianyi Wang 可能会将 SPADE 模块更改为商业友好的一个，但他很忙。
- 如果你希望*加快*他为商业目的的进程，请通过电子邮件与他联系：iceclearwjy@gmail.com

## 致谢

我要感谢 Jianyi Wang 等人提出的原始 StableSR 方法。