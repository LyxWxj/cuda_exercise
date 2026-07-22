#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <vector>
#include <algorithm>
#include <cmath>

#include "../src/nms.cu"

// Reference NMS (CPU, sorted by score descending)
static std::vector<int> ref_nms(const std::vector<float>& boxes,
                                 const std::vector<float>& scores,
                                 int num_boxes, float iou_threshold) {
  // Sort by score descending
  std::vector<int> order(num_boxes);
  for (int i = 0; i < num_boxes; ++i) order[i] = i;
  std::sort(order.begin(), order.end(), [&](int a, int b) {
    return scores[a] > scores[b];
  });

  std::vector<float> sorted_boxes(num_boxes * 4);
  for (int i = 0; i < num_boxes; ++i) {
    sorted_boxes[i * 4 + 0] = boxes[order[i] * 4 + 0];
    sorted_boxes[i * 4 + 1] = boxes[order[i] * 4 + 1];
    sorted_boxes[i * 4 + 2] = boxes[order[i] * 4 + 2];
    sorted_boxes[i * 4 + 3] = boxes[order[i] * 4 + 3];
  }

  std::vector<int> keep(num_boxes, 1);
  for (int i = 0; i < num_boxes; ++i) {
    if (!keep[i]) continue;
    float x1 = sorted_boxes[i * 4 + 0], y1 = sorted_boxes[i * 4 + 1];
    float x2 = sorted_boxes[i * 4 + 2], y2 = sorted_boxes[i * 4 + 3];
    float area = (x2 - x1) * (y2 - y1);
    for (int j = i + 1; j < num_boxes; ++j) {
      if (!keep[j]) continue;
      float x1_j = sorted_boxes[j * 4 + 0], y1_j = sorted_boxes[j * 4 + 1];
      float x2_j = sorted_boxes[j * 4 + 2], y2_j = sorted_boxes[j * 4 + 3];
      float inter_x1 = fmaxf(x1, x1_j), inter_y1 = fmaxf(y1, y1_j);
      float inter_x2 = fminf(x2, x2_j), inter_y2 = fminf(y2, y2_j);
      float inter_w = fmaxf(0.0f, inter_x2 - inter_x1);
      float inter_h = fmaxf(0.0f, inter_y2 - inter_y1);
      float inter_area = inter_w * inter_h;
      float area_j = (x2_j - x1_j) * (y2_j - y1_j);
      float iou = inter_area / (area + area_j - inter_area);
      if (iou > iou_threshold) keep[j] = 0;
    }
  }

  std::vector<int> result;
  for (int i = 0; i < num_boxes; ++i)
    if (keep[i]) result.push_back(order[i]);
  return result;
}

TEST(NMS, Basic) {
  const int num_boxes = 10;
  const float iou_threshold = 0.5f;

  // Create some overlapping boxes
  std::vector<float> boxes = {
    10, 10, 50, 50,   // box 0
    12, 12, 52, 52,   // box 1 (high overlap with 0)
    100, 100, 150, 150, // box 2 (no overlap)
    200, 200, 250, 250, // box 3 (no overlap)
    15, 15, 55, 55,   // box 4 (high overlap with 0,1)
    300, 300, 350, 350, // box 5
    110, 110, 160, 160, // box 6 (overlap with 2)
    120, 120, 170, 170, // box 7 (overlap with 2,6)
    400, 400, 450, 450, // box 8
    410, 410, 460, 460, // box 9 (overlap with 8)
  };
  std::vector<float> scores = {0.9f, 0.8f, 0.7f, 0.6f, 0.85f,
                                0.5f, 0.65f, 0.55f, 0.4f, 0.35f};

  // Sort by score descending
  std::vector<int> order(num_boxes);
  for (int i = 0; i < num_boxes; ++i) order[i] = i;
  std::sort(order.begin(), order.end(), [&](int a, int b) {
    return scores[a] > scores[b];
  });
  std::vector<float> sorted_boxes(num_boxes * 4);
  std::vector<float> sorted_scores(num_boxes);
  for (int i = 0; i < num_boxes; ++i) {
    sorted_boxes[i * 4 + 0] = boxes[order[i] * 4 + 0];
    sorted_boxes[i * 4 + 1] = boxes[order[i] * 4 + 1];
    sorted_boxes[i * 4 + 2] = boxes[order[i] * 4 + 2];
    sorted_boxes[i * 4 + 3] = boxes[order[i] * 4 + 3];
    sorted_scores[i] = scores[order[i]];
  }

  // GPU NMS
  float *d_boxes, *d_scores;
  int *d_keep;
  cudaMalloc(&d_boxes, num_boxes * 4 * sizeof(float));
  cudaMalloc(&d_scores, num_boxes * sizeof(float));
  cudaMalloc(&d_keep, num_boxes * sizeof(int));
  cudaMemcpy(d_boxes, sorted_boxes.data(), num_boxes * 4 * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_scores, sorted_scores.data(), num_boxes * sizeof(float), cudaMemcpyHostToDevice);

  int block = 256;
  int grid = (num_boxes + block - 1) / block;
  nms_kernel<<<grid, block>>>(d_boxes, d_scores, d_keep, num_boxes, iou_threshold);

  std::vector<int> h_keep(num_boxes);
  cudaMemcpy(h_keep.data(), d_keep, num_boxes * sizeof(int), cudaMemcpyDeviceToHost);

  // Count kept boxes
  int gpu_count = 0;
  for (int i = 0; i < num_boxes; ++i) gpu_count += h_keep[i];

  // Reference
  auto ref_result = ref_nms(boxes, scores, num_boxes, iou_threshold);
  int ref_count = ref_result.size();

  EXPECT_EQ(gpu_count, ref_count);

  cudaFree(d_boxes); cudaFree(d_scores); cudaFree(d_keep);
}

TEST(NMS, NoOverlap) {
  const int num_boxes = 4;
  const float iou_threshold = 0.5f;

  // Non-overlapping boxes
  std::vector<float> boxes = {
    0, 0, 10, 10,
    20, 20, 30, 30,
    40, 40, 50, 50,
    60, 60, 70, 70,
  };
  std::vector<float> scores = {0.9f, 0.8f, 0.7f, 0.6f};

  float *d_boxes, *d_scores;
  int *d_keep;
  cudaMalloc(&d_boxes, num_boxes * 4 * sizeof(float));
  cudaMalloc(&d_scores, num_boxes * sizeof(float));
  cudaMalloc(&d_keep, num_boxes * sizeof(int));
  cudaMemcpy(d_boxes, boxes.data(), num_boxes * 4 * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_scores, scores.data(), num_boxes * sizeof(float), cudaMemcpyHostToDevice);

  nms_kernel<<<1, 256>>>(d_boxes, d_scores, d_keep, num_boxes, iou_threshold);

  std::vector<int> h_keep(num_boxes);
  cudaMemcpy(h_keep.data(), d_keep, num_boxes * sizeof(int), cudaMemcpyDeviceToHost);

  // All should be kept
  for (int i = 0; i < num_boxes; ++i)
    EXPECT_EQ(h_keep[i], 1) << "box " << i << " should be kept";

  cudaFree(d_boxes); cudaFree(d_scores); cudaFree(d_keep);
}

TEST(NMS, FullOverlap) {
  const int num_boxes = 3;
  const float iou_threshold = 0.5f;

  // Identical boxes — only first should be kept
  std::vector<float> boxes = {
    10, 10, 50, 50,
    10, 10, 50, 50,
    10, 10, 50, 50,
  };
  std::vector<float> scores = {0.9f, 0.8f, 0.7f};

  float *d_boxes, *d_scores;
  int *d_keep;
  cudaMalloc(&d_boxes, num_boxes * 4 * sizeof(float));
  cudaMalloc(&d_scores, num_boxes * sizeof(float));
  cudaMalloc(&d_keep, num_boxes * sizeof(int));
  cudaMemcpy(d_boxes, boxes.data(), num_boxes * 4 * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_scores, scores.data(), num_boxes * sizeof(float), cudaMemcpyHostToDevice);

  nms_kernel<<<1, 256>>>(d_boxes, d_scores, d_keep, num_boxes, iou_threshold);

  std::vector<int> h_keep(num_boxes);
  cudaMemcpy(h_keep.data(), d_keep, num_boxes * sizeof(int), cudaMemcpyDeviceToHost);

  EXPECT_EQ(h_keep[0], 1);   // highest score, keep
  EXPECT_EQ(h_keep[1], 0);   // suppressed
  EXPECT_EQ(h_keep[2], 0);   // suppressed

  cudaFree(d_boxes); cudaFree(d_scores); cudaFree(d_keep);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
