In each shape, it contains two directories: `extracted` and `rendered_data`. The `extracted` directory contains the pre-processed data, as `preprocess.h5`, while the `rendered_data` directory provides the multi-view images. The contents of each `preprocess.h5` file are described as follows. 
```
"xyz": point cloud data, shape = [n_ps, 3]
"cls_view": class prediction from VLLM in view-wise, shape = [n_view, n_bbox, n_pred]
"conf_view": prediction probability from VLLM in view-wise, shape = [n_view, n_pts, n_bbox, n_part]
"cls_bbox": class prediction from VLLM in bounding-box-wise, shape = [1, total_bbox]
"conf_bbox": prediction probability from VLLM in bounding-box-wise, shape = [n_pts, total_bbox, n_part]
"gt_semantic_seg": GT, shape = [n_pts]
```

In short, the pre-processed data store the predition output from VLLM which are already transfered to 3D space. We use [PartSLIP code](https://colin97.github.io/PartSLIP_page/) to perfrom this task.