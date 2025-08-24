## 📊 Training Progress (30 Epochs)

<table>
  <tr>
    <th style="background-color:#4CAF50;color:white;">Epoch</th>
    <th style="background-color:#2196F3;color:white;">Train Loss</th>
    <th style="background-color:#2196F3;color:white;">Train Cosine</th>
    <th style="background-color:#FF9800;color:white;">Val Loss</th>
    <th style="background-color:#FF9800;color:white;">Val Cosine</th>
    <th style="background-color:#9C27B0;color:white;">Epoch Time (sec)</th>
  </tr>
  <tr><td>1</td><td>0.8059</td><td>0.4946</td><td>0.2741</td><td>0.6311</td><td>210.61</td></tr>
  <tr><td>2</td><td>0.3830</td><td>0.4749</td><td>0.5003</td><td>0.5155</td><td>178.21</td></tr>
  <tr><td>3</td><td>0.4467</td><td>0.4337</td><td>0.7289</td><td>0.6458</td><td>208.00</td></tr>
  <tr><td>4</td><td>0.1222</td><td>0.6766</td><td>0.7662</td><td>0.4922</td><td>178.42</td></tr>
  <tr><td>5</td><td>0.0531</td><td>0.7823</td><td>0.5324</td><td>0.6195</td><td>206.63</td></tr>
  <tr><td>6</td><td>0.0406</td><td>0.8309</td><td>0.7539</td><td>0.4991</td><td>212.91</td></tr>
  <tr><td>7</td><td>0.0470</td><td>0.8163</td><td>0.5415</td><td>0.6038</td><td>199.43</td></tr>
  <tr><td>8</td><td>0.0287</td><td>0.8788</td><td>0.3311</td><td>0.6976</td><td>196.93</td></tr>
  <tr><td>9</td><td>0.0159</td><td>0.9274</td><td>0.3485</td><td>0.6454</td><td>193.51</td></tr>
  <tr><td>10</td><td>0.0147</td><td>0.9360</td><td>0.1989</td><td style="color:green;font-weight:bold;">0.8095</td><td>162.87</td></tr>
  <tr><td>11</td><td>0.0095</td><td>0.9611</td><td>0.2963</td><td>0.7384</td><td>199.08</td></tr>
  <tr><td>12</td><td>0.0159</td><td>0.9399</td><td>0.1969</td><td>0.8137</td><td>187.23</td></tr>
  <tr><td>13</td><td>0.0120</td><td>0.9534</td><td>0.1796</td><td>0.8097</td><td>180.73</td></tr>
  <tr><td>14</td><td>0.0080</td><td>0.9692</td><td>0.0911</td><td>0.8912</td><td>187.95</td></tr>
  <tr><td>15</td><td>0.0070</td><td>0.9730</td><td>0.1092</td><td>0.8613</td><td>201.72</td></tr>
  <tr><td>16</td><td>0.0050</td><td>0.9814</td><td>0.0592</td><td>0.9212</td><td>194.71</td></tr>
  <tr><td>17</td><td>0.0063</td><td>0.9772</td><td>0.0520</td><td>0.9239</td><td>214.67</td></tr>
  <tr><td>18</td><td>0.0060</td><td>0.9785</td><td>0.0390</td><td>0.9349</td><td>201.76</td></tr>
  <tr><td>19</td><td>0.0070</td><td>0.9764</td><td>0.0332</td><td>0.9340</td><td>184.11</td></tr>
  <tr><td>20</td><td>0.0048</td><td>0.9815</td><td>0.0141</td><td>0.9687</td><td>198.04</td></tr>
  <tr><td>21</td><td>0.0045</td><td>0.9835</td><td>0.0110</td><td>0.9745</td><td>188.08</td></tr>
  <tr><td>22</td><td>0.0050</td><td>0.9825</td><td>0.0130</td><td>0.9712</td><td>186.20</td></tr>
  <tr><td>23</td><td>0.0045</td><td>0.9837</td><td>0.0128</td><td>0.9750</td><td>209.09</td></tr>
  <tr><td>24</td><td>0.0030</td><td>0.9892</td><td>0.0134</td><td>0.9760</td><td>210.60</td></tr>
  <tr><td>25</td><td>0.0037</td><td>0.9879</td><td>0.0118</td><td>0.9779</td><td>200.53</td></tr>
  <tr><td>26</td><td>0.0027</td><td>0.9909</td><td>0.0085</td><td>0.9812</td><td>193.43</td></tr>
  <tr><td>27</td><td>0.0030</td><td>0.9905</td><td>0.0062</td><td>0.9838</td><td>193.85</td></tr>
  <tr><td>28</td><td>0.0026</td><td>0.9907</td><td>0.0045</td><td>0.9868</td><td>213.01</td></tr>
  <tr><td>29</td><td>0.0026</td><td>0.9908</td><td>0.0037</td><td>0.9894</td><td>198.86</td></tr>
  <tr style="background-color:#C8E6C9;">
    <td><b>30</b></td>
    <td><b>0.0021</b></td>
    <td style="color:green;font-weight:bold;">0.9917</td>
    <td><b>0.0026</b></td>
    <td style="color:green;font-weight:bold;">0.9925</td>
    <td>180.89</td>
  </tr>
</table>

✅ Notice how training loss steadily decreases while validation cosine similarity increases towards ~**0.99**, confirming excellent convergence.
 

### Figures

(./SMSAT_Encoder_page-0001.jpg.png)

(./statistics_plots.png)

(./tsne_scatter.png)

(./centroid_scatter.png)

(./distance_statistics.png)

(./smasat-trainloss.png)

(./activations_Music.png)

(./activations_Normal(Silence).png)

(./activations_SpiritualMeditation.png)

