The pre-processed data are stored in h5 file with this format:
```
"xyz": point cloud data, shape = [n_ps, 3]
"cls_view": class prediction from VLLM in view-wise, shape = [n_view, n_bbox, n_pred]
"conf_view": prediction probability from VLLM in view-wise, shape = [n_view, n_pts, n_bbox, n_part]
"cls_bbox": class prediction from VLLM in bounding-box-wise, shape = [1, total_bbox]
"conf_bbox": prediction probability from VLLM in bounding-box-wise, shape = [n_view, total_bbox, n_part]
"gt_semantic_seg": GT, shape = [n_pts]
```

In short, these pre-processed store the predition output from VLLM which are already transfered to 3D space. We use [PartSLIP code](https://colin97.github.io/PartSLIP_page/) to perfrom this task.