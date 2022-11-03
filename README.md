<em> # Predictor-de-pose-humana-2D </em>
<h4 align="center">
 Proyecto terminado 🔚:
</h4>

![PYTHON](https://badgen.net/badge/python/3.7/blue?icon=github)
![TENSORFLOW](https://badgen.net/badge/TF/1.15.5/cyan?icon=github)
![PYTORCH](https://badgen.net/badge/PyTorch/1.6/orange?icon=github)

An aplication of human motion prediction using already existing models for two kinds of input (joints and skeleton structure with 2D data, and coded data representing the dynamic information of human motion on a sequence) with the porpose to low the latency 
of a human body pose corrector using the motion retargeting as reference for the performance.


- `Functionality 1`: The human pose predictor is added to the motion retargeting generating a result that respects the size of the skeleton as well as the perspective.
- `Functionality 2`: Motion sequence data is generated with a predictor within motion retargeting using the latent dynamic space representation of motion.
- `Functionality 3`: Re-training and evaluation of the best model and approach for motion retargeting
- `Functionality 4`: The effect of the number of prediction frames vs. the percentage error is displayed. 


![This is an image](https://github.com/Chr1sus/Predictor-de-pose-humana-2D/blob/master/Results/gcnmethodsv2.gif?raw=true)

![This is an image](https://github.com/Chr1sus/Predictor-de-pose-humana-2D/blob/master/Results/fourmethodsv2.gif?raw=true)

![This is an image](https://github.com/Chr1sus/Predictor-de-pose-humana-2D/blob/master/Results/FINALOBJT.png)
 

\## 📁 Access to the project

For the project, the requirements of the requirement.txt must be installed in a conda environment as following.

      python3 -m venv env

      source env/bin/activate

      pip install -r requirements.txt


Once the environment is created, the necessary data sets and repositories are installed.

-![Dataset-Human3.6M](https://github.com/kotaro-inoue/human3.6m_downloader)

-![Dataset-Panoptics-CMU](https://github.com/CMU-Perceptual-Computing-Lab/panoptic-toolbox)

-![Dataset-Penn-Action](https://github.com/dreamdragon/PennAction)

-![Motion Retargeting:](https://github.com/ChrisWu1997/2D-Motion-Retargeting)


📁 Entregables

      |-Tabla_comparativa.pdf ======:Tabla comparativa del primer objetivo específico.


📁 Modelos

      |-📁 ATTN

          |-ATTN.py

          |-seq2seq.py

      |-📁 GCN

          |-GCN_DCT.py

          |-main_gcn.py


      |-📁 Q-DRNN

          |-prediction_modelv2.py

          |-translatev2.py



      |-📁 UNET



          |-config.py

          |-seg_data_.py

          |-train_Unet.py

          |-unet.py

      |-pruebas.py (script de experimentos)


📁 Results

      |-Resultados visuales del proyecto. 


📁 ops

      |-loaddata.py (script para leer datos)

      |-train.py (script para producir experimentos con el Aberman)

      |-tils.py

      |-values_for_desnormalize

      |-vizu.py

📁 retrained_models

      |-📁 Q-DRNN_MODEL

         |-checkpoint

         |-complete.index

         |-complete.meta

         |-complete.data-00000-of-00001


      |-gcn_dct.pth

      |-gcn_dct_retrained.pth

      |-unetv1.pth

      |-bestseq2seqmodel.py

📁 tests

      |-📁 decoded_sequence (resultados del Aberman)

           |-📁 CMU-MOCAP

           |-📁 Penn-Action

      |-abermantest.py

      |-comparepredictors.py

      |-finalobj





\## 🛠️ Open and run the project

Training of the models: 

For the methods that use as input the human skeleton with its key points, you can use the models from the folder Models GCN and Q-DRNNN where you should specify the address of the input data in folders with names S1,S2,S3,S4,S5,S6,S7,S8,S9 and S11.

For methods using Motion Retargeting encoded data run train.py in the same folder as Motion Retargeting with the address of the training data defining and reporting the save address for the training and expected data.

If you want to visualize the results you only need to run comparepredictors.py and finalobj.py to observe the effects of latency vs. MSE.


Reference:

    
    
                                                                                                                                                                                                           

     
     
     

           
