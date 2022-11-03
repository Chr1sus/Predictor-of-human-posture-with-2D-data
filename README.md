<em> # Predictor-de-pose-humana-2D </em>
<h4 align="center">
 Proyecto terminado 🔚:
</h4>

![PYTHON](https://badgen.net/badge/python/3.7/blue?icon=github)
![TENSORFLOW](https://badgen.net/badge/TF/1.15.5/cyan?icon=github)
![PYTORCH](https://badgen.net/badge/PyTorch/1.6/orange?icon=github)

An aplication of human motion prediction using already existing models for two kinds of input (joints and skeleton structure with 2D data, and coded data representing the dynamic information of human motion on a sequence) with the porpose to low the latency 
of a human body pose corrector using the motion retargeting as reference for the performance.


- `Funcionalidad 1`: Se adiciona el predictor de pose humana al motion retargeting generando un resultado que respeta el tamnno del esqueleto como de la perspectiva
- `Funcionalidad 2`: Se generan datos de secuencia de movimiento con un predictor dentro del motion retargeting usando la representacion del movimiento del espacio dinamico latente
- `Funcionalidad 3`: Se reentrena y evalua el mejor modelo y acercamiento para el motion retargeting
- `Funcionalidad 4`: Se visualiza el efecto de la cantidad de cuadros de prediccion vs el porcentaje de error.


![This is an image](https://github.com/Chr1sus/Predictor-de-pose-humana-2D/blob/master/Results/gcnmethodsv2.gif?raw=true)

![This is an image](https://github.com/Chr1sus/Predictor-de-pose-humana-2D/blob/master/Results/fourmethodsv2.gif?raw=true)

![This is an image](https://github.com/Chr1sus/Predictor-de-pose-humana-2D/blob/master/Results/FINALOBJT.png)
 

\## 📁 Acceso al proyecto

Para el proyecto se debe de instalar en un ambiente de conda los requisitos del requirement.txt

      python3 -m venv env

      source env/bin/activate

      pip install -r requirements.txt


Una vez creado el ambiente se procede a instalar los conjutos de datos y repositorios necesarios.

-![Conjunto de datos-Human3.6M](https://github.com/kotaro-inoue/human3.6m_downloader)

-![Conjunto de datos-Panoptics-CMU](https://github.com/CMU-Perceptual-Computing-Lab/panoptic-toolbox)

-![Conjunto de datos-Penn-Action](https://github.com/dreamdragon/PennAction)

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





\## 🛠️ Abre y ejecuta el proyecto

Entrenamiento de los modelos: 

Para los metodos que usan de entrada el esqueleto humano con sus puntos claves, se puede utilizar los modelos de la carpeta Modelos GCN y Q-DRNN donde se debe espicificar la direccion de los datos de entrada en carpetas con nombres S1,S2,S3,S4,S5,S6,S7,S8,S9 y S11.

Para los metodos que emplean los datos codificados del Motion Retargeting se debe de correr el train.py en la misma carpeta del Motion Retargeting con la direccion de los datos de entrenamiento definiendo y reportar la direccion de guardado para los datos de entrenamiento y los esperados.

Si se desea visualizar los resultados solo es necesario correr el comparepredictors.py y el finalobj.py para observar los efectos de la latencia vs el MSE.


Reference:

    
    
                                                                                                                                                                                                           

     
     
     

           
