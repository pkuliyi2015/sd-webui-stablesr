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

⚪ 方法 1: URL 安装

- 打开 Automatic1111 WebUI -> 点击 "Extensions" 标签页 -> 点击 "Install from URL" 标签页 -> 输入 https://github.com/pkuliyi2015/sd-webui-stablesr.git -> 点击 "Install"

![installation](https://github.com/pkuliyi2015/multidiffusion-img-demo/blob/master/installation.png?raw=true)

⚪ 方法 2: 施工中...

### 2. 必须模型

- 你必须使用 StabilityAI 官方的 Stable Diffusion V2.1 512 **EMA** 模型（约 5.21GB）
    - 你可以从 [HuggingFace](https://huggingface.co/stabilityai/stable-diffusion-2-1-base) 下载
    - 放入 stable-diffusion-webui/models/Stable-Diffusion/ 文件夹
    > 虽然StableSR需要一个SD2.1的模型权重，但你仍然可以放大来自SD1.5的图片。NSFW图片不会被模型扭曲，输出质量也不会受到影响。
- 下载 StableSR 模块
    - 官方资源：[HuggingFace](https://huggingface.co/Iceclear/StableSR/resolve/main/weibu_models.zip) (约1.2G)。请注意这是一个zip文件，同时包含StableSR模块和可选组件VQVAE.
    - 我的资源：<[GoogleDrive](https://drive.google.com/file/d/1tWjkZQhfj07sHDR4r9Ta5Fk4iMp1t3Qw/view?usp=sharing)> <[百度网盘-提取码aguq](https://pan.baidu.com/s/1Nq_6ciGgKnTu0W14QcKKWg?pwd=aguq)>
    - 把StableSR模块（约400M大小）放入 stable-diffusion-webui/extensions/sd-webui-stablesr/models/ 文件夹

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

### 5. 跑图提示

- 推荐使用 Euler a 采样器。步数 >= 20。
- 如果生成图像尺寸 > 512，我们推荐使用 Tiled Diffusion & VAE，否则，图像质量可能不理想，VRAM 使用量也会很大。
- 这里有一些 Tiled Diffusion 设置，可以复制官方的结果。
    - 方法 = Mixture of Diffusers
    - 隐空间Tile大小 = 64，隐空间Tile重叠 = 32
    - Tile批大小尽可能大，直到差一点点就炸显存为止。
    - Upscaler**必须**选择None。
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