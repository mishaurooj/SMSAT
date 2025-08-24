## 📊 Ablation Study Results (SMSAT Encoder)

Below is the summary of experiments for the SMSAT Encoder ablation study.

<table>
  <tr>
    <th style="background-color:#4CAF50;color:white;">Experiment</th>
    <th style="background-color:#2196F3;color:white;">Train Loss (Final)</th>
    <th style="background-color:#2196F3;color:white;">Val Loss (Final)</th>
    <th style="background-color:#FF9800;color:white;">Train Cosine (Final)</th>
    <th style="background-color:#FF9800;color:white;">Val Cosine (Final)</th>
    <th style="background-color:#9C27B0;color:white;">Time (sec)</th>
    <th style="background-color:#9C27B0;color:white;">Time (min)</th>
    <th style="background-color:#E91E63;color:white;">Inter-class SM–M</th>
    <th style="background-color:#E91E63;color:white;">Inter-class SM–NS</th>
    <th style="background-color:#795548;color:white;">Intra SM Mean</th>
    <th style="background-color:#795548;color:white;">Intra M Mean</th>
    <th style="background-color:#795548;color:white;">Intra NS Mean</th>
  </tr>
  <tr>
    <td><b>no_noise</b></td>
    <td>2.24E-05</td>
    <td>4.03E-05</td>
    <td>0.9986</td>
    <td>0.9974</td>
    <td>206.21</td>
    <td>3.44</td>
    <td>0.0149</td>
    <td>0.0137</td>
    <td>0.1077</td>
    <td>0.1111</td>
    <td>0.1098</td>
  </tr>
  <tr>
    <td><b>no_time_stretch</b></td>
    <td>2.79E-05</td>
    <td>3.63E-05</td>
    <td>0.9982</td>
    <td>0.9977</td>
    <td>214.33</td>
    <td>3.57</td>
    <td>0.0202</td>
    <td>0.0237</td>
    <td>0.1597</td>
    <td>0.1597</td>
    <td>0.1582</td>
  </tr>
  <tr>
    <td><b>no_pitch_shift</b></td>
    <td>2.96E-05</td>
    <td>2.08E-05</td>
    <td>0.9981</td>
    <td>0.9987</td>
    <td>178.93</td>
    <td>2.98</td>
    <td>0.0277</td>
    <td>0.0286</td>
    <td>0.2001</td>
    <td>0.1943</td>
    <td>0.1975</td>
  </tr>
  <tr>
    <td><b>no_spec_augment</b></td>
    <td>2.90E-05</td>
    <td>1.85E-05</td>
    <td>0.9981</td>
    <td>0.9988</td>
    <td>211.64</td>
    <td>3.53</td>
    <td>0.0173</td>
    <td>0.0153</td>
    <td>0.1127</td>
    <td>0.1134</td>
    <td>0.1132</td>
  </tr>
  <tr>
    <td><b>no_crop</b></td>
    <td>3.26E-05</td>
    <td>4.89E-05</td>
    <td>0.9979</td>
    <td>0.9969</td>
    <td>500.77</td>
    <td>8.35</td>
    <td>0.0145</td>
    <td>0.0146</td>
    <td>0.1025</td>
    <td>0.1018</td>
    <td>0.1025</td>
  </tr>
  <tr>
    <td><b>encoder_baseline</b></td>
    <td>2.34E-05</td>
    <td>2.63E-05</td>
    <td>0.9985</td>
    <td>0.9983</td>
    <td>211.85</td>
    <td>3.53</td>
    <td>0.0141</td>
    <td>0.0156</td>
    <td>0.1244</td>
    <td>0.1280</td>
    <td>0.1263</td>
  </tr>
  <tr>
    <td><b>loss_cosine</b></td>
    <td style="color:red;"><b>0.00036</b></td>
    <td style="color:red;"><b>0.00022</b></td>
    <td><b>0.9996</b></td>
    <td><b>0.9998</b></td>
    <td>214.84</td>
    <td>3.58</td>
    <td style="color:orange;"><b>0.0067</b></td>
    <td style="color:orange;"><b>0.0067</b></td>
    <td>0.0471</td>
    <td>0.0496</td>
    <td>0.0481</td>
  </tr>
</table>
