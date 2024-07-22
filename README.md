# Detecting Energy Theft Cyberattacks in Smart Metering Infrastructure using Deep Learning Model and Rule-Based Policy
*Ashley Ajuz, MS, Computer and Electrical Engineering*

University of Pittsburgh


## About this work
The Advanced Metering Infrastructure (AMI) is an integral part of smart energy grid
development and provides a bi-directional communication system between utilities and con-
sumers. AMI offers several benefits such as automated collection of time-series energy mea-
surements, outage management, connection and disconnection requests, and system status
tracking. However, given the increased dependency on device connectivity and data-driven
decisions, AMI and smart energy systems become vulnerable to multiple security threats.
This work will focus on energy theft, which is a well-known data-falsification cyberattack
on AMI. In this attack, smart meters are compromised to report falsified energy consump-
tion data, leading to financial losses, potential disruption of power flow, and other negative
impacts on the functionality of power grid components. In this thesis, we propose an energy
theft attack detection approach utilizing deep machine learning (ML) algorithm, specifically
a long-short term memory (LSTM) model. The proposed approach combines LSTM-based
deep learning model and rule-based policy for attack detection. Unlike prior work that
mainly considers supervised machine learning models, an unsupervised learning approach is
proposed that would be more practical in real-world scenarios. The performance of both
supervised and unsupervised learning-based approaches is compared to show the effective-
ness of the proposed theft detection method under different attack models. Various tests are
conducted to evaluate the attack detection performance using a publicly available dataset
containing energy consumption data. It is shown that the proposed model achieves high
attack detection accuracy and low false alarm rates

## Things to Note
- The dataset used in this work is labeled "Tidy_LCL_Data_2Y.csv" and can be found in the *Output* Folder. This complete dataset that this work references can be found here: https://www.kaggle.com/datasets/emmanuelfwerr/london-homes-energy-data?resource=download
The dataset keeps track of the energy consumption of 5,567 randomly selected households in London from November 2011 to February 2014.

- This program runs fastest when using a GPU but can be run over CPU
