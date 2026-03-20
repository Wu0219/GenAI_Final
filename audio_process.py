# preprocess.py
import os
import json
import librosa
import numpy as np
from pathlib import Path
from pydub import AudioSegment
from tqdm import tqdm
from config import TARGET_SPEC


class AudioPreprocessor:
    def __init__(self, target_spec=TARGET_SPEC):
        self.target_sr = target_spec["sample_rate"]
        self.target_duration = target_spec["duration"]
        self.target_samples = int(self.target_sr * self.target_duration)

    def preprocess_file(self, input_path, output_path):
        """
        预处理单个音频文件：
        1. 转mono
        2. 重采样到24kHz
        3. 裁剪/填充到恰好2秒
        """
        # 用librosa加载（自动处理多种格式）
        y, sr = librosa.load(input_path, sr=self.target_sr, mono=True)

        # 裁剪或填充
        if len(y) < self.target_samples:
            # 填充
            y = np.pad(y, (0, self.target_samples - len(y)), mode='constant')
        else:
            # 随机裁剪（避免总是从头开始）
            start = np.random.randint(0, len(y) - self.target_samples + 1)
            y = y[start:start + self.target_samples]

        # 保存（用librosa写wav保证兼容性）
        librosa.output.write_wav(output_path, y, self.target_sr)

        return output_path

    def preprocess_category(self, category_dir, output_dir):
        """预处理整个类别"""
        metadata_path = category_dir / "metadata.json"
        if not metadata_path.exists():
            print(f"⚠️  No metadata found in {category_dir}")
            return []

        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata_list = json.load(f)

        processed = []
        output_dir = Path(output_dir) / category_dir.name
        output_dir.mkdir(parents=True, exist_ok=True)

        for meta in tqdm(metadata_list, desc=f"Preprocessing {category_dir.name}"):
            input_path = Path(meta["local_path"])
            if not input_path.exists():
                print(f"⚠️  File not found: {input_path}")
                continue

            output_filename = f"{meta['id']}_processed.wav"
            output_path = output_dir / output_filename

            try:
                self.preprocess_file(input_path, output_path)

                # 更新元数据
                meta["processed_path"] = str(output_path)
                meta["processed_duration"] = self.target_duration
                meta["processed_sr"] = self.target_sr
                processed.append(meta)

            except Exception as e:
                print(f"✗ Failed to preprocess {input_path}: {e}")
                continue

        # 保存处理后的元数据
        with open(output_dir / "metadata_processed.json", 'w', encoding='utf-8') as f:
            json.dump(processed, f, indent=2, ensure_ascii=False)

        return processed

    def preprocess_all(self, raw_dir, processed_dir):
        """预处理所有类别"""
        raw_dir = Path(raw_dir)
        results = {}

        for category in ["speech", "solo_instrument", "complex_music", "ambient_noise"]:
            category_dir = raw_dir / category
            if not category_dir.exists():
                print(f"⚠️  Skipping {category} (directory not found)")
                continue

            processed = self.preprocess_category(category_dir, processed_dir)
            results[category] = processed
            print(f"✅ {category}: {len(processed)} files processed")

        return results


# 使用示例
if __name__ == "__main__":
    preprocessor = AudioPreprocessor()
    results = preprocessor.preprocess_all(
        raw_dir="./output/raw",
        processed_dir="./output/processed"
    )