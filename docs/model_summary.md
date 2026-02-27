# Model Architecture Breakdown

| Layer (type) | Output Shape | Param # |
| :--- | :--- | :--- |
| Conv2d-1 | [-1, 64, 5, 1000] | 40,384 |
| ReLU-2 | [-1, 64, 5, 1000] | 0 |
| Conv2d-3 | [-1, 64, 5, 1000] | 36,928 |
| ReLU-4 | [-1, 64, 5, 1000] | 0 |
| ConvBlock-5 | [-1, 64, 5, 1000] | 0 |
| MaxPool2d-6 | [-1, 64, 5, 500] | 0 |
| Conv2d-7 | [-1, 128, 5, 500] | 73,856 |
| ReLU-8 | [-1, 128, 5, 500] | 0 |
| Conv2d-9 | [-1, 128, 5, 500] | 147,584 |
| ReLU-10 | [-1, 128, 5, 500] | 0 |
| ConvBlock-11 | [-1, 128, 5, 500] | 0 |
| MaxPool2d-12 | [-1, 128, 5, 250] | 0 |
| Conv2d-13 | [-1, 256, 5, 250] | 295,168 |
| ReLU-14 | [-1, 256, 5, 250] | 0 |
| Conv2d-15 | [-1, 256, 5, 250] | 590,080 |
| ReLU-16 | [-1, 256, 5, 250] | 0 |
| ConvBlock-17 | [-1, 256, 5, 250] | 0 |
| ConvTranspose2d-18 | [-1, 128, 5, 500] | 65,664 |
| Conv2d-19 | [-1, 128, 5, 500] | 295,040 |
| ReLU-20 | [-1, 128, 5, 500] | 0 |
| Conv2d-21 | [-1, 128, 5, 500] | 147,584 |
| ReLU-22 | [-1, 128, 5, 500] | 0 |
| ConvBlock-23 | [-1, 128, 5, 500] | 0 |
| ConvTranspose2d-24 | [-1, 64, 5, 1000] | 16,448 |
| Conv2d-25 | [-1, 64, 5, 1000] | 73,792 |
| ReLU-26 | [-1, 64, 5, 1000] | 0 |
| Conv2d-27 | [-1, 64, 5, 1000] | 36,928 |
| ReLU-28 | [-1, 64, 5, 1000] | 0 |
| ConvBlock-29 | [-1, 64, 5, 1000] | 0 |
| Conv2d-30 | [-1, 32, 5, 1000] | 18,464 |
| ReLU-31 | [-1, 32, 5, 1000] | 0 |
| AdaptiveAvgPool2d-32 | [-1, 32, 70, 70] | 0 |
| Conv2d-33 | [-1, 1, 70, 70] | 33 |

---

### Memory & Parameter Summary

* **Total params:** 1,837,953
* **Trainable params:** 1,837,953
* **Non-trainable params:** 0

**Resource Usage Estimate:**
* **Input size:** 1.34 MB
* **Forward/backward pass size:** 72.03 MB
* **Params size:** 7.01 MB
* **Estimated Total Size:** 80.38 MB