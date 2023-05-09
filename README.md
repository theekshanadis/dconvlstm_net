# dconvlstm_net

This repository contains the code segments from DConv-LSTM-Net: A Novel Architecture for Single and 12-Lead ECG Anomaly Detection, currently in consideration for publication in IEEE Sensors Journal. In this paper, we present a novel deep learning architecture for single and 12-lead ECG anomaly detection, and we evaluate our modals on four publically available datasets. Alongside with specific implementation details for DConv-LSTM-Net, we also provide additional data pre-processing scripts we used and some baseline modals we implemented for benchmarking purposes. 

We evaluate our modals on the following datasets, and we perform subject-independent 10-FCV. 

1. PhysioNet-AF dataset (phy_af)
2. CPSC dataset (cpsc)
3. PTB-XL dataset (ptb_xl)
4. Georgia dataset (geor_12l)

For all our implementations we use basic/standard Pytorch (1.6.0>) commands, and other than that you don't need to install specific versions. Here, we provide our modal implementation codes, some baselines we implemented for our experiments and data pre-processing scripts we used. 

