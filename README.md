# BGM
Official implementation of "Forgetting the Background: A Masking Approach for Enhanced Infrared Small Target Detection" (accepted to IEEE TGRS). 
## Training

* **Install the environment according to** `requirements.txt`.

* **Run training script** :

```bash
CUDA_VISIBLE_DEVICES=0   python -u train.py --name basicunet_results  --model BasicUNet_plus   --num_workers 8 
```
