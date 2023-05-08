# Forward-Forward-Algorithm

This repo is the implementation of a simple unsupervised example of FF in this paper.
>[Geoffrey Hinton. The Forward-Forward Algorithm: Some Preliminary Investigations](https://arxiv.org/pdf/2212.13345.pdf)


## Negative data
<p align="left">
  <img src="negative_img/x_pos.PNG" "width="100" height="100" hspace="10"/>
  <img src="negative_img/x_pos_2.PNG" width="100" height="100" hspace="10"/>
  <img src="negative_img/mask.PNG" width="100" height="100" hspace="10"/>
  <img src="negative_img/x_neg.PNG" width="100" height="100" hspace="10"/>
</p>

I created mask by repeatedly blurring a random bit image with a filter [1/4,1/2,1/4] in both the horizontal and vertical as the paper explained. Then generated negative data by x1 * mask + x2 * (1-mask) with random x1, x2 in original mnist data.

## Architecture
writing...


