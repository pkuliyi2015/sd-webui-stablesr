# StableSR - Stable Diffusion WebUI

Licensed under S-Lab License 1.0

[![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]

[English](README.md) | 中文

- StableSR 是由 Jianyi Wang 等人提出的强力超分辨率项目。
- 本仓库将 StableSR 项目迁移到 Automatic1111 WebUI。

相关链接

> 点击查看大量官方示例！

- [项目页面](https://iceclear.github.io/projects/stablesr/)
- [官方仓库](https://github.com/IceClear/StableSR)
- [论文](https://arxiv.org/abs/2305.07015)

> 如果你觉得这个项目有帮助，请给我和 Jianyi Wang 的仓库点个星！⭐
***

## 重要更新

- 2023.07.01: 我们偶然发现 **合适的负面提示词会大幅提高StableSR的生成质量.**
    - 我们使用了CFG Scale=7以及下面的负面提示词：3d, cartoon, anime, sketches, (worst quality:2), (low quality:2)
    - 点击[对比](https://imgsli.com/MTg5MjM1)来查看负面提示词的威力
    - 正面提示词用处不大，但你依然可以尝试下面的提示词：(masterpiece:2), (best quality:2), (realistic:2),(very clear:2)
    - 使用上面提示词，我们正努力接近闭源方法GigaGAN的放大效果（虽然还有差距）。点击[comparison2](https://imgsli.com/MTg5MzAx/)看看实力。
- 2023.06.30: 我们很高兴发布 StableSR 的新版本 SD 2.1 768！（感谢 Jianyi Wang）
    - 它产生类似的细节，但具有**更自然的边缘（更少的白边黑边）**和**更好的颜色**。
    - 它支持 768 * 768 的分辨率。
- 要使用新模型：
    - 使用 SD 2.1 768 基础模型。可以从[HuggingFace](https://huggingface.co/stabilityai/stable-diffusion-2-1)下载。
    - 下载相应的 SR 模块（约400MB）：[官方资源](https://huggingface.co/Iceclear/StableSR/blob/main/webui_768v_139.ckpt)，[我的百度网盘-提取码8ju9](https://pan.baidu.com/s/17on7GA2RLvVzdDnBwA0N0g?pwd=8ju9)
    - 现在在 Tiled Diffusion 中可以使用更大的块大小（96 * 96，与默认设置相同），速度可能会稍微更快。
    - 其他设置保持不变。
- Jianyi Wang一直在努力训练更强大的适用于 AIGC 图像的 SR 模块。这些模型都将基于 SD2.1 768 或 SDXL 进行调整。SD2.1 512版本将不再继续尝试。

***
## 功能

1. **高保真图像放大**：
    - 不修改人物脸部的同时添加非常细致的细节和纹理
    - 适合大多数图片（真实或动漫，摄影作品或AIGC，SD 1.5或Midjourney图片...）
2. **较少的显存消耗**：
    - 我移除了官方实现中显存消耗高的模块。
    - 剩下的模型比ControlNet Tile模型小得多，需要的显存也少得多。
    - 当结合Tiled Diffusion & VAE时，你可以在有限的显存（例如，<12GB）中进行4k图像放大。
    > 注意，sdp可能会不明原因炸显存。建议使用xformers。
3. **小波分解颜色修正**：
    - StableSR官方实现有明显的颜色偏移，这一问题在分块放大时更加明显。
    - 我实现了一个强大的后处理技术，有效地匹配放大图像与原图的颜色。请看[小波分解颜色修正例子](https://imgsli.com/MTgwNDg2/)。

***
## 使用

### 1. 安装

⚪ 方法 1: 官方市场

- 打开Automatic1111 WebUI -> 点击“扩展”选项卡 -> 点击“可用”选项卡 -> 找到“StableSR” -> 点击“安装”

⚪ 方法 2: URL 安装

- 打开 Automatic1111 WebUI -> 点击 "Extensions" 标签页 -> 点击 "Install from URL" 标签页 -> 输入 https://github.com/pkuliyi2015/sd-webui-stablesr.git -> 点击 "Install"

![installation](https://github.com/pkuliyi2015/multidiffusion-img-demo/blob/master/installation.png?raw=true)

### 2. 必须模型

我们目前有两个版本。它们产生的细节相似，但是768版本的边缘更自然。
#### 🆕 SD2.1 768 版本
- 您必须使用 StabilityAI 提供的 Stable Diffusion V2.1 768 **EMA** 检查点（约5.21GB）
    - 您可以从 [HuggingFace](https://huggingface.co/stabilityai/stable-diffusion-2-1) 下载它
    - 将其放入 stable-diffusion-webui/models/Stable-Diffusion/ 文件夹中

- 下载提取后的 StableSR 模块
    - [官方资源](https://huggingface.co/Iceclear/StableSR/blob/main/webui_768v_139.ckpt)
    - 将 StableSR 模块（约400MB）放入 stable-diffusion-webui/extensions/sd-webui-stablesr/models/ 文件夹中

****
#### SD2.1 512 版本（更锐利，但黑边白边现象更明显）
- 您必须使用 StabilityAI 提供的 Stable Diffusion V2.1 512 **EMA** 检查点（约5.21GB）
    - 您可以从 [HuggingFace](https://huggingface.co/stabilityai/stable-diffusion-2-1-base) 下载它
    - 将其放入 stable-diffusion-webui/models/Stable-Diffusion/ 文件夹中

- 下载提取后的 StableSR 模块
    - 官方资源：[HuggingFace](https://huggingface.co/Iceclear/StableSR/resolve/main/weibu_models.zip)（约1.2G）。请注意，这是一个包含 StableSR 模块和 VQVAE 的压缩文件。
    - 我的资源：[GoogleDrive](https://drive.google.com/file/d/1tWjkZQhfj07sHDR4r9Ta5Fk4iMp1t3Qw/view?usp=sharing) [百度网盘-提取码aguq](https://pan.baidu.com/s/1Nq_6ciGgKnTu0W14QcKKWg?pwd=aguq)
    - 将 StableSR 模块（约400MB）放入 stable-diffusion-webui/extensions/sd-webui-stablesr/models/ 文件夹中

> 虽然我们使用了 SD2.1 的检查点，但您仍然可以放大任何图片（甚至来自 SD1.5 或 NSFW）。您的图片不会被审查，输出质量也不会受到影响。

### 3. 可选组件

- 安装 [Tiled Diffusion & VAE](https://github.com/pkuliyi2015/multidiffusion-upscaler-for-automatic1111) 扩展
    - 原始的 StableSR 对大于 512 的大图像容易出现 OOM。
    - 为了获得更好的质量和更少的 VRAM 使用，我们建议使用 Tiled Diffusion & VAE。
- 使用官方 VQGAN VAE
    - 官方资源：同2中的链接
    - 我的资源：<[GoogleDrive](https://drive.google.com/file/d/1ARtDMia3_CbwNsGxxGcZ5UP75W4PeIEI/view?usp=share_link)> <[百度网盘-提取码83u9](https://pan.baidu.com/s/1YCYmGBethR9JZ8-eypoIiQ?pwd=83u9)>
    - 把VQVAE（约750MB大小）放在你的 stable-diffusion-webui/models/VAE 中

### 4. 扩展使用

- 在 WebUI 的顶部，选择你下载的 v2-1_512-ema-pruned 模型。
- 切换到 img2img 标签。在页面底部找到 "Scripts" 下拉列表。
    - 选择 StableSR 脚本。
    - 点击刷新按钮，选择你已下载的 StableSR 检查点。
    - 选择一个放大因子。
- 上传你的图像并开始生成（无需提示也能工作）。
- 推荐使用 Euler a 采样器，CFG值=7，步数 >= 20。
    - 尽管StableSR不需要提示词也能工作，我们发现负面提示词能显著增强细节。比如使用3d, cartoon, anime, sketches, (worst quality:2), (low quality:2)
    - 点击查看有/没有prompt的[对比](https://imgsli.com/MTg5MjM1)
- 如果生成图像尺寸 > 512，我们推荐使用 Tiled Diffusion & VAE，否则，图像质量可能不理想，VRAM 使用量也会很大。
- 这里是官方推荐的 Tiled Diffusion 设置。
    - 方法 = Mixture of Diffusers
    - 隐空间Tile大小 = 64，隐空间Tile重叠 = 32
    - Tile批大小尽可能大，直到差一点点就炸显存为止。
    - Upscaler**必须**选择None。
- 下图是24GB显存的推荐设置。
    - 对于4GB的设备，**只需将Tiled Diffusion Latent tile批处理大小改为1，Tiled VAE编码器Tile大小改为1024，解码器Tile大小改为128。**
    - SDP注意力优化可能会导致OOM（内存不足），因此推荐使用xformers。
    - 除非你有深入的理解，否则你**不要**改变Tiled Diffusion & Tiled VAE中的其他设置。**这些参数对于StableSR基本上是最优解。**
![推荐设置](https://github.com/pkuliyi2015/multidiffusion-img-demo/blob/master/recommended_settings_24GB.jpg?raw=true)

### 5. 参数解释

- 什么是 "Pure Noise"？
    - Pure Noise也就是纯噪声，指的是从完全随机的噪声张量开始，而不是从你的图像开始。**这是 StableSR 论文中的默认做法。**
    - 启用这个选项时，脚本会忽略你的重绘幅度设置。产出将会是更详细的图像，但也会显著改变颜色和锐度。
    - 禁用这个选项时，脚本会开始添加一些噪声到你的图像。即使你将去噪强度设为1，结果也不会那么的细节（但可能更和谐好看）。参见 [对比图](https://imgsli.com/MTgwMTMx)。
    - 如果禁用Pure Noise，推荐重绘幅度设置为1
- 什么是"颜色修正"？
    - 这是为了缓解来自StableSR和Tile处理过程中的颜色偏移问题。
    - AdaIN简单地匹配原图和结果图的颜色统计信息。这是StableSR官方算法，但常常效果不佳。
    - Wavelet将原图和结果图分解为低频和高频，然后用原图的低频信息（颜色）替换掉结果图的低频信息。该算法对于不均匀的颜色偏移非常强力。算法来自GIMP和Krita，对每张图像需要几秒钟的时间。
    - 启用颜色修正时，原图也会出现在您的预览窗口中，但不会被自动保存。

### 6. 重要问题

> 为什么我的结果和官方示例不同？

- 这不是你或我们的错。
    - 如果正确安装，这个扩展有与 StableSR 相同的 UNet 模型权重。
    - 如果你安装了可选的 VQVAE，整个模型权重将与融合权重为 0 的官方模型相同。
- 但是，你的结果将**不如**官方结果，因为：
    - 采样器差异：
        - 官方仓库进行 100 或 200 步的 legacy DDPM 采样，并使用自定义的时间步调度器，采样时不使用负提示。
        - 然而，WebUI 不提供这样的采样器，必须带有负提示进行采样。**这是主要的差异。**
    - VQVAE 解码器差异：
        - 官方 VQVAE 解码器将一些编码器特征作为输入。
        - 然而，在实践中，我发现这些特征对于大图像来说非常大。 (>10G 用于 4k 图像，即使是在 float16！)
        - 因此，**我移除了 VAE 解码器中的 CFW 组件**。由于这导致了对细节的较低保真度，我将尝试将它作为一个选项添加回去。

***
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
- 严禁贩售代码和权重

### 产出图像的重要通知

- 请注意，NVIDIA SPADE 模块中的 CC BY-NC-SA 4.0 许可也禁止把产生的图像用于商业用途。
- Jianyi Wang 可能会将 SPADE 模块更改为商业友好的一个，但他很忙。
- 如果你希望**加快**他的进度，请通过电子邮件与他联系：iceclearwjy@gmail.com

## 致谢

感谢 Jianyi Wang 等人提出的 StableSR 方法