/**

 * Janus-Pro: Unified Multimodal Understanding and Generation Framework
 *
 * Overview:
 * Janus-Pro is an innovative autoregressive framework that unifies multimodal understanding and generation.
 * It decouples visual encoding into separate pathways while maintaining a single transformer architecture,
 * effectively addressing the inherent conflicts between visual comprehension and image generation.
 *
 * Key Features:
 * * Unified architecture for multimodal processing.
 * * Decoupled visual encoding that enhances both understanding and generation flexibility.
 * * Leverages the SigLIP-L vision encoder for processing images at a resolution of 384x384.
 * * Utilizes a dedicated tokenizer (with a downsample rate of 16) for image generation tasks.
 *
 * Implementation Details:
 * * Built on the foundational DeepSeek-LLM models, benefiting from robust base architectures.
 * * Designed to surpass the performance of previous unified models while matching or exceeding task-specific models.
 * * Simplifies and streamlines the integration of multimodal features in a unified framework.
 *
 * Licensing and Citation:
 * * The repository is distributed under the MIT License, subject to further usage restrictions specified in the DeepSeek Model License.
 * * For academic and research purposes, a citation is provided to reference the methodology used in Janus-Pro.
 *
 * Usage:
 * Refer to the GitHub repository for quick start guides, detailed implementation steps, and further documentation.
 *
 * Contact:
 * For questions or further information, users are encouraged to open an issue or contact support via email.
 */

---
license: mit
license_name: deepseek
license_link: LICENSE
pipeline_tag: any-to-any
library_name: transformers
tags:
* multimodal
* text-to-image
* unified-model

---

## 1. Introduction

Janus-Pro is a novel autoregressive framework that unifies multimodal understanding and generation. 
It addresses the limitations of previous approaches by decoupling visual encoding into separate pathways, while still utilizing a single, unified transformer architecture for processing. The decoupling not only alleviates the conflict between the visual encoder's roles in understanding and generation, but also enhances the framework's flexibility. 
Janus-Pro surpasses previous unified model and matches or exceeds the performance of task-specific models. 
The simplicity, high flexibility, and effectiveness of Janus-Pro make it a strong candidate for next-generation unified multimodal models.

[**Github Repository**](https://github.com/deepseek-ai/Janus)

<div align="center">
<img alt="image" src="janus_pro_teaser1.png" style="width:90%;">
</div>

<div align="center">
<img alt="image" src="janus_pro_teaser2.png" style="width:90%;">
</div>


### 2. Model Summary

Janus-Pro is a unified understanding and generation MLLM, which decouples visual encoding for multimodal understanding and generation. 
Janus-Pro is constructed based on the DeepSeek-LLM-1.5b-base/DeepSeek-LLM-7b-base.

For multimodal understanding, it uses the [SigLIP-L](https://huggingface.co/timm/ViT-L-16-SigLIP-384) as the vision encoder, which supports 384 x 384 image input. For image generation, Janus-Pro uses the tokenizer from [here](https://github.com/FoundationVision/LlamaGen) with a downsample rate of 16.

## Setup Instructions

1. Clone the repository:
   git clone <https://github.com/deepseek-ai/Janus>

2. Install dependencies:
   cd Janus && pip install -r requirements.txt

3. Pull the model using Ollama:
   ollama pull &lt;model&gt;

## 3. Quick Start

Please refer to [**Github Repository**](https://github.com/deepseek-ai/Janus)


## 4. License

This code repository is licensed under [the MIT License](https://github.com/deepseek-ai/DeepSeek-LLM/blob/HEAD/LICENSE-CODE). The use of Janus-Pro models is subject to [DeepSeek Model License](https://github.com/deepseek-ai/DeepSeek-LLM/blob/HEAD/LICENSE-MODEL).

## 5. Citation

```
@article{chen2025janus,
  title={Janus-Pro: Unified Multimodal Understanding and Generation with Data and Model Scaling},
  author={Chen, Xiaokang and Wu, Zhiyu and Liu, Xingchao and Pan, Zizheng and Liu, Wen and Xie, Zhenda and Yu, Xingkai and Ruan, Chong},
  journal={arXiv preprint arXiv:2501.17811},
  year={2025}
}
```

## 6. Contact

If you have any questions, please raise an issue or contact us at [service@deepseek.com](mailto:service@deepseek.com).
