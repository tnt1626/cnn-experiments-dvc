#!/bin/bash

echo "🚀 Đang đưa các experiments vào DVC Queue..."

# =========================================================
# THIẾT LẬP CÁC MỐC BASELINE
# =========================================================

# 1. Baseline ban đầu: sigmoid, z-score
dvc exp run -n exp_01_baseline_sigmoid \
  -S base.base_activation="sigmoid" \
  -S improvements.use_relu=false \
  -S improvements.use_z_score=true

# 2. Mốc Baseline Mới: ReLU + z-score
# (Tất cả các exp bên dưới sẽ chỉ thêm 1 kỹ thuật từ mốc này)
dvc exp run -n exp_02_new_baseline_relu \
  -S improvements.use_relu=true \
  -S improvements.use_z_score=true

# =========================================================
# LẦN LƯỢT ÁP DỤNG CÁC KỸ THUẬT RIÊNG LẺ (ABLATION STUDY)
# =========================================================

# 3. Chỉ thêm Batch Norm
dvc exp run -n exp_03_add_batch_norm \
  -S improvements.use_relu=true \
  -S improvements.use_z_score=true \
  -S improvements.use_batch_norm=true

# 4. Chỉ thêm He Initialization
dvc exp run -n exp_04_add_he_init \
  -S improvements.use_relu=true \
  -S improvements.use_z_score=true \
  -S improvements.use_he_initialization=true

# 5. Chỉ thêm Skip Connection
dvc exp run -n exp_05_add_skip_connection \
  -S improvements.use_relu=true \
  -S improvements.use_z_score=true \
  -S improvements.use_skip_connection=true

# 6. Chỉ thêm Reduce Learning Rate (Scheduler)
dvc exp run -n exp_06_add_reduce_lr \
  -S improvements.use_relu=true \
  -S improvements.use_z_score=true \
  -S improvements.reduce_learning_rate=true

# 7. Chỉ dùng Advanced Activation
dvc exp run -n exp_07_add_adv_activation \
  -S improvements.use_relu=true \
  -S improvements.use_z_score=true \
  -S improvements.use_advanced_activation=true

echo "✅ Đã đưa 7 experiments vào Queue."
echo "🔥 Bắt đầu chạy tất cả experiments..."

# Chạy tuần tự tất cả các job trong queue
dvc queue start