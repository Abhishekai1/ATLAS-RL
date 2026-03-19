# ATLAS-RL

tokenizer_config.json: 100%
 26.0/26.0 [00:00<00:00, 1.33kB/s]
vocab.json: 
 899k/? [00:00<00:00, 27.4MB/s]
merges.txt: 
 456k/? [00:00<00:00, 15.3MB/s]
tokenizer.json: 
 1.36M/? [00:00<00:00, 46.0MB/s]
model.safetensors: 100%
 1.63G/1.63G [00:14<00:00, 91.3MB/s]
Loading weights: 100%
 515/515 [00:00<00:00, 799.07it/s, Materializing param=model.shared.weight]

[Eval] Baseline ...
  [baseline] 20/225
  [baseline] 40/225
  [baseline] 60/225
  [baseline] 80/225
  [baseline] 100/225
  [baseline] 120/225
  [baseline] 140/225
  [baseline] 160/225
  [baseline] 180/225
  [baseline] 200/225
  [baseline] 220/225

[Eval] RAG ...
  [rag] 20/225
  [rag] 40/225
  [rag] 60/225
  [rag] 80/225
  [rag] 100/225
  [rag] 120/225
  [rag] 140/225
  [rag] 160/225
  [rag] 180/225
  [rag] 200/225
  [rag] 220/225

[Train] ATLAS-RL ...
  ep=1 step=20 loss=4.2361
  ep=1 step=40 loss=4.1306
[Train] Epoch 1/8 loss=4.1306
  ep=2 step=20 loss=3.9528
  ep=2 step=40 loss=3.9604
[Train] Epoch 2/8 loss=3.9604
  ep=3 step=20 loss=3.8739
  ep=3 step=40 loss=3.8564
[Train] Epoch 3/8 loss=3.8564
  ep=4 step=20 loss=3.8166
  ep=4 step=40 loss=3.7777
[Train] Epoch 4/8 loss=3.7777
  ep=5 step=20 loss=3.6925
  ep=5 step=40 loss=3.6939
[Train] Epoch 5/8 loss=3.6939
  ep=6 step=20 loss=3.6511
  ep=6 step=40 loss=3.6205
[Train] Epoch 6/8 loss=3.6205
  ep=7 step=20 loss=3.5381
  ep=7 step=40 loss=3.5550
[Train] Epoch 7/8 loss=3.5550
  ep=8 step=20 loss=3.4967
  ep=8 step=40 loss=3.4825
[Train] Epoch 8/8 loss=3.4825

[Eval] ATLAS-RL model ...
  [rag] 20/225
  [rag] 40/225
  [rag] 60/225
  [rag] 80/225
  [rag] 100/225
  [rag] 120/225
  [rag] 140/225
  [rag] 160/225
  [rag] 180/225
  [rag] 200/225
  [rag] 220/225

==================================================================================
  MAIN RESULTS
==================================================================================
Model                 Accuracy(F1)          Hallucination         Consistency           Grounding             ATLAS Score         
----------------------------------------------------------------------------------
Baseline LLM          0.1954                0.3800                0.2891                0.1537                0.3695              
RAG                   0.1182                0.2458                0.1755                0.3816                0.4095              
ATLAS-RL              0.2192                0.1451                0.1977                0.3506                0.4057              
==================================================================================
[CSV] /content/sample_data/main_results.csv

[Ablation] ...

==================================================================================
  ABLATION STUDY
==================================================================================
Configuration         Token F1              ATLAS Score         
----------------------------------------------------------------------------------
Full ATLAS            0.3207                0.3867              
w/o KL                0.3207                0.2964              
w/o Consistency       0.3207                0.4523              
w/o Grounding         0.3207                0.4763              
==================================================================================
[CSV] /content/sample_data/ablation_results.csv

==================================================================================
  FAILURE MODE STATISTICS
==================================================================================
  retrieval_error            218  (32.3%)
  inconsistency              218  (32.3%)
  hallucination              187  (27.7%)
  none                        52  (7.7%)
==================================================================================

==================================================================================
  KEY OBSERVATIONS
==================================================================================
  1. RAG vs Baseline   ATLAS 0.4095 vs 0.3695  ↑0.0400
  2. ATLAS-RL vs RAG   ATLAS 0.4057 vs 0.4095  ↓0.0037
  3. Hallucination     Base=0.3800 RAG=0.2458 ATLAS-RL=0.1451
  4. Best  ablation    'w/o Grounding'  ATLAS=0.4763
  5. Worst ablation    'w/o KL'  ATLAS=0.2964
  6. Top failure       retrieval_error  (32.3%)

  Runtime: 61.2 min
==================================================================================

[DONE] /content/sample_data/main_results.csv
[DONE] /content/sample_data/ablation_results.csv
