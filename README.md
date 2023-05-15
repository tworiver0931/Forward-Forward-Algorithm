# Forward-Forward-Algorithm(unsupervised)

 This repo is the unofficial implementation of a simple unsupervised example of FF in this paper.
>[Geoffrey Hinton. The Forward-Forward Algorithm: Some Preliminary Investigations](https://arxiv.org/pdf/2212.13345.pdf)


## Negative data
<table>
  <tbody>
    <tr>
      <td align="center">
        <img src="negative_img/x_pos.PNG" "width="100" height="100" alt="x1">
      </td>
      <td align="center">
        <img src="negative_img/x_pos_2.PNG" width="100" height="100" alt="x2">
      </td>
      <td align="center">
        <img src="negative_img/mask.PNG" width="100" height="100" alt="mask">
      </td>
      <td align="center">
        <img src="negative_img/x_neg.PNG" width="100" height="100" alt="x_neg">
      </td>
    </tr>
    <tr>
      <td align="center">x1</td>
      <td align="center">x2</td>
      <td align="center">mask</td>
      <td align="center">x_neg</td>
    </tr>
  </tbody>
</table>

I created mask by repeatedly blurring a random bit image with a filter [1/4,1/2,1/4] in both the horizontal and vertical as the paper explained. Then generated negative data by x1 * mask + x2 * (1-mask) with random x1, x2 in original mnist data.

## Architecture
<img src="https://github.com/tworiver0931/Forward-Forward-Unsupervised/assets/63793194/8a763a7b-dc55-4242-a580-b95ae46cfdca" width="510" height="300">

Both positive and negative data go into input of the first FF layer. After training the first layer, the layer outputs representation and it go into input of the second FF layer and so on. Each FF layer has a normalization term(to eliminate length of activity vector and pass only orientation) in front of them. After training all FF layers, representations of positive data except first layer go into softmax layer to train classification task.

## Local Receptive Field
In the paper, Hinton also trained FF with local receptive fields without weight sharing. Pytorch doesn't offer this module, so I implemented it with reference to Keras LocalConv2D. It works same as convolution, but in each operation, the kernel doesn't share weights.

                                                                                                                                                     
He used "peer normalization" in this experiment. I think it's similar to batch normalization, but still not sure what peer normalization is. So I used batch normalization instead of peer.
                                                                           
                                                                           
Test errors of my implementation didn't achieve that of the experiment in the paper. It's just for personal study.

                                                                                                       
## References
- [Geoffrey Hinton. The Forward-Forward Algorithm: Some Preliminary Investigations](https://arxiv.org/pdf/2212.13345.pdf)
- [mohammadpz/pytorch_forward_forward](https://github.com/mohammadpz/pytorch_forward_forward)

