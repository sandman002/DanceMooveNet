# DanceMooveNet
A simple dance move generator based on temporal positional embedding and a feed forward network.
The network has been trained on [AIST++](https://aistdancedb.ongaaccel.jp/) dataset using only the optimised 3d keypoints. The network takes a random vector sampled from a gaussian normal distribution of 128 dimensions. The generator and discriminators are conditioned on genre of music (60 genres).

Furthe improvements are needed. The conditioning can be done using the actual audio feature rather than the genre label. Current conditioning performs poorly. Instead of 3d joints position, working with SMPL representation for the body might be more useful for extending the animation to 3d softwares.

![](https://github.com/sandman002/DanceMooveNet/blob/main/anim/ss.gif)

<table>
  <tr>
    <td>Few training sequences</td>
    <td>Few generated sequences</td>
  </tr>
  <tr>
    <td><img src="https://github.com/sandman002/DanceMooveNet/blob/main/anim/gt.gif" width="370" /></td>
    <td><img src="https://github.com/sandman002/DanceMooveNet/blob/main/anim/pics.gif" width="370" /></td>
  </tr>
 </table>




