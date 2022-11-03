# Copyright 2022 Dakewe Biotech Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import os

# Prepare dataset
os.system("python3 ./prepare_dataset.py --images_dir ../data/SR_RAW/alignment/train/mod24/GT      --output_dir ../data/SR_RAW/SRGAN_CoBi/train/GT --image_size 552 --step 552 --num_workers 16")
os.system("python3 ./prepare_dataset.py --images_dir ../data/SR_RAW/alignment/train/mod24/LRbicx2 --output_dir ../data/SR_RAW/SRGAN_CoBi/train/LRbicx2 --image_size 276 --step 276 --num_workers 16")
os.system("python3 ./prepare_dataset.py --images_dir ../data/SR_RAW/alignment/train/mod24/LRbicx3 --output_dir ../data/SR_RAW/SRGAN_CoBi/train/LRbicx3 --image_size 184 --step 184 --num_workers 16")
os.system("python3 ./prepare_dataset.py --images_dir ../data/SR_RAW/alignment/train/mod24/LRbicx4 --output_dir ../data/SR_RAW/SRGAN_CoBi/train/LRbicx4 --image_size 138 --step 138 --num_workers 16")
os.system("python3 ./prepare_dataset.py --images_dir ../data/SR_RAW/alignment/train/mod24/LRbicx8 --output_dir ../data/SR_RAW/SRGAN_CoBi/train/LRbicx8 --image_size 69 --step 69 --num_workers 16")

os.system("python3 ./prepare_dataset.py --images_dir ../data/SR_RAW/alignment/test/mod24/GT      --output_dir ../data/SR_RAW/SRGAN_CoBi/test/GT --image_size 552 --step 552 --num_workers 16")
os.system("python3 ./prepare_dataset.py --images_dir ../data/SR_RAW/alignment/test/mod24/LRbicx2 --output_dir ../data/SR_RAW/SRGAN_CoBi/test/LRbicx2 --image_size 276 --step 276 --num_workers 16")
os.system("python3 ./prepare_dataset.py --images_dir ../data/SR_RAW/alignment/test/mod24/LRbicx3 --output_dir ../data/SR_RAW/SRGAN_CoBi/test/LRbicx3 --image_size 184 --step 184 --num_workers 16")
os.system("python3 ./prepare_dataset.py --images_dir ../data/SR_RAW/alignment/test/mod24/LRbicx4 --output_dir ../data/SR_RAW/SRGAN_CoBi/test/LRbicx4 --image_size 138 --step 138 --num_workers 16")
os.system("python3 ./prepare_dataset.py --images_dir ../data/SR_RAW/alignment/test/mod24/LRbicx8 --output_dir ../data/SR_RAW/SRGAN_CoBi/test/LRbicx8 --image_size 69 --step 69 --num_workers 16")