# Predictor-de-pose-humana-2D
Proyecto final, modelo de predicción de pose humana bidimensional

------>Entregables:
             |
             
             |------>Tabla_comparativa.pdf ======:Tabla comparativa del primer objetivo específico.


----->Modelos

         |------>ATTN
                    
                    (Modelo de atención para predicción de secuencia a secuencia adaptado de )
                    
                   |------->ATTN.py
                   
                   |------->seq2seq.py
                   
        |------>GCN
                    
                    (Modelo de GCN con DCT adaptado de)
                    
                   |------->GCN_DCT.py
                   
                   |------->main_gcn.py
                   
                   
        |------>Q-DRNN
                    
                    (Modelo de Q-DRNN adaptado de)
                    
                   |------->prediction_modelv2.py
                   
                   |------->translatev2.py
                   
                  
       
       |------>UNET
                    
                    (Modelo de U-NET adaptado de )
                    
                   |------->config.py
                   
                   |------->seg_data_.py
                   
                   |------->train_Unet.py
                   
                   |------->unet.py
                   
       |------>pruebas.py (script de experimentos)
             
         
------>Results
         
       |------>Resultados  visuales del experimento. 
       
       
------>ops
         
       |------>loaddata.py (script para leer datos)
       
       |------>train.py (script para producir experimentos con el Aberman)
       
       |------>utils.py
       
       |------>values_for_desnormalize
       
       |------>vizu.py
       
 ------>pretrained_models
         
       |------>Q-DRNN_MODEL
       
                  |------>checkpoint
                  
                  |------>complete.index
                  
                  |------>complete.meta
                  
                  |------>complete.data-00000-of-00001
                  
       
       |------>gcn_dct.pth
       
       |------>gcn_dct_retrained.pth
       
       |------>unetv1.pth
       
       |------>bestseq2seqmodel.py
       
------>tests
         
       |------>decoded_sequence (resultados del Aberman)
                  
                 |------>CMU-MOCAP
                 
                 |------>Penn-Action
       
       |------>abermantest.py
       
       |------>comparepredictors.py
       
       |------>finalobj
       
    
    
                                                                                                          
                                                                                                          

             
           
                            
                         
