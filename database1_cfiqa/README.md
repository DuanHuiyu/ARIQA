## CFIQA Database

The CFIQA database consists of 600 reference images (300 A + 300 B) and 300 distorted images. 

### Description
+ Folder naming convention

        A: reference images A
        B: reference images B
        M: distorted images (superimposed images)
        A_saliency: saliency maps of A (predicted by [UNISAL](https://github.com/rdroste/unisal))
        B_saliency: saliency maps of B (predicted by [UNISAL](https://github.com/rdroste/unisal))
        M_saliency: saliency maps of M (predicted by [UNISAL](https://github.com/rdroste/unisal))

+ File naming convention

        thresholds.csv: the mixing values used in the CFIQA database

        MOS.mat: MOS values of the CFIQA database (calculated via "../code1_subjective_data_process/process_cfiqa_subjective_data.m")
        MOSz.mat: MOSz values of the CFIQA database (calculated via "../code1_subjective_data_process/process_cfiqa_subjective_data.m")
        num_obs.mat: num_obs values of the CFIQA database (calculated via "../code1_subjective_data_process/process_cfiqa_subjective_data.m")
        SD.mat: SD values of the CFIQA database (calculated via "../code1_subjective_data_process/process_cfiqa_subjective_data.m")

+ Code naming convention

        gen_confusion.py: generate mixed images ("M") and thresholds (thresholds.csv) from folder "A" and "B"

    Example of usage:
    
        python gen_confusion.py --dataroot ./raw_images --augmented_folder setA_raw --background_folder setB_raw --outf ./test

    Note: If you have already downloaded this database, you do not need to run this code.


### Citation
If you use the CFIQA database, please consider citing:

    @article{duan2022confusing,
        title={Confusing image quality assessment: Towards better augmented reality experience},
        author={Duan, Huiyu and Min, Xiongkuo and Zhu, Yucheng and Zhai, Guangtao and Yang, Xiaokang and Le Callet, Patrick},
        journal={IEEE Transactions on Image Processing (TIP)},
        year={2022}
    }

### Contact
If you have any question, please contact huiyuduan@sjtu.edu.cn