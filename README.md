# CG-KGR
This is the Tensorflow implementation for paper "Attentive Knowledge-aware Graph Convolutional Networks with Collaborative Guidance for Personalized Recommendation." [arxiv](https://arxiv.org/pdf/2109.02046.pdf). Yankai Chen, Yaming Yang, Yujing Wang, Jing Bai, Xiangchen Song, and Irwin King. 2022. 


## Environment Requirement

The code runs well under python 3.8.0. The required packages are as follows:

- Tensorflow-gpu == 1.14.0
- numpy == 1.21.1

## Datasets
**First**, please refer to [link](https://drive.google.com/file/d/1rnhNBNgiN76Gjd81vXEn34PCESrkrwQv/view?usp=sharing) to download the orginal rating data and distribute them under "/CG-KGR/data/". Then create datasplit under the ratio 6:2:2.

For example, uncomment the following line in main_movie.py:
```
data_split(args)
```
**then** uncomment the following line:

```
Exp_run(args)
```
and **Finally** run [main_xxx.py] as: 
```bash
python main_movie.py
```

You can also download our five experimental random splits for each datasets via [link](https://drive.google.com/file/d/1YlMqQl4pxnV2cwfJnsmAw_4tfmfDYNcN/view?usp=sharing). 


## Citation
If you find this paper useful for your research, please kind cite it as:

```
@inproceedings{chen2021attentive,
  title={Attentive Knowledge-aware Graph Convolutional Networks with Collaborative Guidance for Personalized Recommendation},
  author={Chen, Yankai and Yang, Yaming and Wang, Yujing and Bai, Jing and Song, Xiangchen and King, Irwin},
  booktitle={The 38th IEEE International Conference on Data Engineering},
  pages={299--311},
  publisher={{IEEE}},
  year={2022},

}
```
