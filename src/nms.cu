#include "utils.cuh"

// ============================================================
// Non-Maximum Suppression (NMS) Kernel
// boxes: (num_boxes, 4) in (x1, y1, x2, y2) format
// keep: (num_boxes) output, 1 = keep, 0 = suppress
// Assumes boxes are sorted by score (descending).
// Each thread handles one box.
// Grid: ceil(num_boxes/256), Block: 256
// ============================================================

// launch: <<<(num_boxes/256, 1, 1), (256, 1, 1)>>>
__global__ void nms_kernel(const float* boxes, const float* scores, int* keep,
                           int num_boxes, float iou_threshold) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // Initialize all keep values first, then sync
  if (idx < num_boxes) keep[idx] = 1;
  __syncthreads();

  if (idx >= num_boxes) return;

  float x1 = boxes[idx * 4 + 0];
  float y1 = boxes[idx * 4 + 1];
  float x2 = boxes[idx * 4 + 2];
  float y2 = boxes[idx * 4 + 3];

  // Check against all higher-scored boxes (index < idx)
  for (int i = 0; i < idx; ++i) {
    if (keep[i] == 0) continue;  // already suppressed

    float x1_i = boxes[i * 4 + 0];
    float y1_i = boxes[i * 4 + 1];
    float x2_i = boxes[i * 4 + 2];
    float y2_i = boxes[i * 4 + 3];

    // Intersection area
    float inter_x1 = fmaxf(x1, x1_i);
    float inter_y1 = fmaxf(y1, y1_i);
    float inter_x2 = fminf(x2, x2_i);
    float inter_y2 = fminf(y2, y2_i);
    float inter_w = fmaxf(0.0f, inter_x2 - inter_x1);
    float inter_h = fmaxf(0.0f, inter_y2 - inter_y1);
    float inter_area = inter_w * inter_h;

    // IoU = intersection / union
    float area = (x2 - x1) * (y2 - y1);
    float area_i = (x2_i - x1_i) * (y2_i - y1_i);
    float iou = inter_area / (area + area_i - inter_area);

    if (iou > iou_threshold) {
      keep[idx] = 0;  // suppress this box
      return;
    }
  }
  // keep[idx] already set to 1 above
}
