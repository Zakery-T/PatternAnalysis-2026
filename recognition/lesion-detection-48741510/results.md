No detection: 0/600

=== Localisation (top-confidence box vs mask-derived box) ===
Mean IoU, all images (misses = 0): 0.755
Mean IoU, detected only (600 images): 0.755
Images with IoU > 0.5: 89.7%
Images with IoU >= 0.8: 52.5%
Target (mean IoU >= 0.8) met: False
Mean IoU, best detection per image: 0.803
Split evaluated: test

=== Per-class localisation (grouped by TRUE class) ===
melanoma: mean IoU 0.721, IoU > 0.5: 88.0%, IoU >= 0.8: 49.6%, n=117
seborrheic keratosis: mean IoU 0.705, IoU > 0.5: 82.2%, IoU >= 0.8: 44.4%, n=90
nevus: mean IoU 0.776, IoU > 0.5: 91.9%, IoU >= 0.8: 55.2%, n=393
Ultralytics 8.4.190  Python-3.14.6 torch-2.14.1+cpu CPU (13th Gen Intel Core i7-13620H)
Model summary (fused): 73 layers, 3,133,193 parameters, 0 gradients, 8.6 GFLOPs
val: Fast image access  (ping: 0.0±0.0 ms, read: 1841.4±409.6 MB/s, size: 11532.7 KB)
val: Scanning C:\Users\zakts\CODE\PatternAnalysis-2026\recognition\lesion-detection-48741510\data\…: 100% ━━━━━━━━━━━━ 600/600 83.9Mit/s 0.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 38/38 4.5s/it 2:53
                   all        600        600      0.614      0.613      0.633      0.356
Speed: 2.0ms preprocess, 77.4ms inference, 0.0ms loss, 0.5ms postprocess per image
Results saved to C:\Users\zakts\CODE\PatternAnalysis-2026\recognition\lesion-detection-48741510\outputs\yolo_val

=== Detection metrics (Ultralytics val, test split) ===
Overall: mAP50 0.633, mAP50-95 0.356, P 0.614, R 0.613
AP at IoU 0.8 (mean over classes): 0.244
melanoma: P 0.489, R 0.504, mAP50 0.512, mAP50-95 0.256
seborrheic keratosis: P 0.547, R 0.567, mAP50 0.544, mAP50-95 0.321
nevus: P 0.805, R 0.768, mAP50 0.842, mAP50-95 0.491

=== Classification (top-confidence box per image) ===
Confusion matrix (rows = true, last column = no detection):
 [[ 64  11  42   0]
 [  4  69  17   0]
 [ 46  26 321   0]]
melanoma: precision 0.561, recall 0.547, F1 0.554
seborrheic keratosis: precision 0.651, recall 0.767, F1 0.704
nevus: precision 0.845, recall 0.817, F1 0.831
Overall accuracy: 0.757
Melanoma vs rest: sensitivity 0.547, specificity 0.896, AUC 0.805

Examples saved to C:\Users\zakts\CODE\PatternAnalysis-2026\recognition\lesion-detection-48741510\outputs/examples
