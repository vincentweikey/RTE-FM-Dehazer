import os
import csv
import torch
from pathlib import Path
from PIL import Image
import pyiqa
import json
from datetime import datetime
from typing import Dict, List, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DehazingBenchmark:
    def __init__(self, root_dir: str, metrics: List[str] = None, device: str = 'cuda', target_size: Tuple[int, int] = (256, 256)):
        """
        Initialize the benchmark evaluator.
        
        Args:
            root_dir: Path to the root directory containing all datasets
            metrics: List of metrics to compute. Defaults to ['psnr', 'ssim', 'lpips']
            device: Device to run evaluation on ('cuda' or 'cpu')
            target_size: Target size for resizing images (width, height)
        """
        self.root_dir = Path(root_dir)
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.target_size = target_size  # Store target size for resizing
        
        # Default metrics for dehazing evaluation
        self.metrics = metrics or ['psnr', 'ssim', 'lpips']
        
        # Initialize IQA models
        self.iqa_models = {}
        self._initialize_metrics()
        
        # Results storage
        self.results = {}
        
    def _initialize_metrics(self):
        """Initialize pyiqa models for each metric."""
        available_metrics = pyiqa.list_models()
        logger.info(f"Available metrics in pyiqa: {available_metrics}")
        
        for metric in self.metrics:
            try:
                if metric in available_metrics:
                    self.iqa_models[metric] = pyiqa.create_metric(metric, device=self.device)
                    logger.info(f"Initialized metric: {metric}")
                else:
                    logger.warning(f"Metric {metric} not available in pyiqa. Skipping.")
            except Exception as e:
                logger.error(f"Failed to initialize {metric}: {e}")
    
    def _resize_image(self, img: Image.Image) -> Image.Image:
        """Resize image to target size while maintaining aspect ratio if needed."""
        if img.size == self.target_size:
            return img
        
        # Use LANCZOS for high-quality downsampling
        resized = img.resize(self.target_size, Image.LANCZOS)
        logger.debug(f"Resized image from {img.size} to {resized.size}")
        return resized
    
    def _find_datasets(self) -> List[Path]:
        """Find all dataset directories."""
        datasets = []
        for item in self.root_dir.iterdir():
            if item.is_dir() and 'hazy' in [d.name for d in item.iterdir() if d.is_dir()]:
                datasets.append(item)
        logger.info(f"Found datasets: {[d.name for d in datasets]}")
        return datasets
    
    def _find_methods(self, dataset_path: Path) -> Dict[str, Path]:
        """Find all dehazing method folders in a dataset."""
        methods = {}
        for item in dataset_path.iterdir():
            if item.is_dir() and item.name not in ['hazy', 'gt']:
                methods[item.name] = item
        return methods
    
    def _get_image_pairs(self, dataset_path: Path, method_name: str) -> List[Tuple[Path, Path, Path]]:
        """Get tuples of (hazy_path, gt_path, result_path) for evaluation."""
        hazy_dir = dataset_path / 'hazy'
        gt_dir = dataset_path / 'gt'
        result_dir = dataset_path / method_name
        
        pairs = []
        
        # Get all hazy images
        hazy_images = sorted(list(hazy_dir.glob('*.png')) + list(hazy_dir.glob('*.jpg')))
        
        for hazy_img in hazy_images:
            # Find corresponding GT image
            gt_img = gt_dir / hazy_img.name
            if not gt_img.exists():
                # Try different naming conventions
                gt_img = gt_dir / hazy_img.name.replace('_hazy', '')
                if not gt_img.exists():
                    gt_img = gt_dir / hazy_img.name.replace('hazy', 'gt')
                    if not gt_img.exists():
                        logger.warning(f"No GT found for {hazy_img.name}")
                        continue
            
            # Find corresponding result image
            result_img = result_dir / hazy_img.name
            if not result_img.exists():
                result_img = result_dir / hazy_img.name.replace('_hazy', '')
                if not result_img.exists():
                    result_img = result_dir / hazy_img.name.replace('hazy', 'dehazy')
                    if not result_img.exists():
                        logger.warning(f"No result found for {hazy_img.name} in {method_name}")
                        continue
            
            pairs.append((hazy_img, gt_img, result_img))
        
        return pairs
    
    def _evaluate_image_pair(self, result_img: Path, gt_img: Path) -> Dict[str, float]:
        """Evaluate a single image pair using all metrics with 256x256 resize."""
        results = {}
    
        try:
            # Load images
            result_pil = Image.open(result_img).convert('RGB')
            gt_pil = Image.open(gt_img).convert('RGB')
    
            # Resize to 256x256
            result_pil = self._resize_image(result_pil)
            gt_pil = self._resize_image(gt_pil)
    
            for metric_name, model in self.iqa_models.items():
                try:
                    # ✅ Handle full-reference vs. no-reference metrics properly
                    if metric_name in ['psnr', 'ssim', 'lpips']:  # Full-reference metrics
                        score = model(result_pil, gt_pil)
                    else:  # No-reference metrics (e.g., niqe, brisque)
                        score = model(result_pil)
    
                    results[metric_name] = float(score.item() if torch.is_tensor(score) else score)
    
                except Exception as e:
                    logger.error(f"Error computing {metric_name} for {result_img}: {e}")
                    results[metric_name] = None
    
        except Exception as e:
            logger.error(f"Error processing images {result_img}, {gt_img}: {e}")
            for metric_name in self.iqa_models.keys():
                results[metric_name] = None
    
        return results
    
    def run_evaluation(self):
        """Run evaluation across all datasets and methods."""
        datasets = self._find_datasets()
        
        for dataset in datasets:
            logger.info(f"Processing dataset: {dataset.name}")
            self.results[dataset.name] = {}
            
            methods = self._find_methods(dataset)
            
            for method_name, method_dir in methods.items():
                logger.info(f"  Evaluating method: {method_name}")
                self.results[dataset.name][method_name] = {}
                
                # Get image pairs
                pairs = self._get_image_pairs(dataset, method_name)
                logger.info(f"    Found {len(pairs)} image pairs")
                
                if not pairs:
                    continue
                
                # Evaluate all pairs
                all_scores = {metric: [] for metric in self.iqa_models.keys()}
                
                for hazy_img, gt_img, result_img in pairs:
                    scores = self._evaluate_image_pair(result_img, gt_img)
                    
                    for metric, score in scores.items():
                        if score is not None:
                            all_scores[metric].append(score)
                
                # Calculate averages
                for metric, scores in all_scores.items():
                    if scores:
                        self.results[dataset.name][method_name][metric] = {
                            'mean': sum(scores) / len(scores),
                            'std': (sum((s - sum(scores)/len(scores))**2 for s in scores) / len(scores))**0.5,
                            'count': len(scores)
                        }
                    else:
                        self.results[dataset.name][method_name][metric] = {
                            'mean': None,
                            'std': None,
                            'count': 0
                        }
    
    def save_results(self, output_dir: str = 'benchmark_results'):
        """Save results to various formats."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. Save detailed JSON
        json_path = output_path / f"benchmark_results_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        logger.info(f"Saved detailed results to {json_path}")
        
      
        
        # 3. Save markdown report
        md_path = output_path / f"benchmark_report_{timestamp}.md"
        
        with open(md_path, 'w') as f:
            f.write("# Dehazing Benchmark Results\n\n")
            f.write(f"**Evaluation Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**Metrics Used:** {', '.join(self.iqa_models.keys())}\n\n")
            f.write(f"**Device:** {self.device}\n\n")
            f.write(f"**Image Size:** {self.target_size[0]}×{self.target_size[1]}\n\n")
            
            for dataset_name, methods in self.results.items():
                f.write(f"## Dataset: {dataset_name}\n\n")
                f.write("| Method | " + " | ".join(self.iqa_models.keys()) + " |\n")
                f.write("|--------|" + "|".join(["--------" for _ in self.iqa_models]) + "|\n")
                
                for method_name, metrics in methods.items():
                    scores = []
                    for metric in self.iqa_models.keys():
                        score_data = metrics.get(metric, {})
                        mean = score_data.get('mean')
                        if mean is not None:
                            scores.append(f"{mean:.3f}")
                        else:
                            scores.append("N/A")
                    
                    f.write(f"| {method_name} | " + " | ".join(scores) + " |\n")
                
                f.write("\n")

          # 2. Save CSV summary
        csv_path = output_path / f"benchmark_summary_{timestamp}.csv"
        
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Header
            header = ['Dataset', 'Method'] + [f"{metric}_mean" for metric in self.iqa_models.keys()] + \
                    [f"{metric}_std" for metric in self.iqa_models.keys()]
            writer.writerow(header)
            
            # Data rows
            for dataset_name, methods in self.results.items():
                for method_name, metrics in methods.items():
                    row = [dataset_name, method_name]
                    
                    # Mean scores
                    for metric in self.iqa_models.keys():
                        score_data = metrics.get(metric, {})
                        row.append(score_data.get('mean', 'N/A'))
                    
                    # Std scores
                    for metric in self.iqa_models.keys():
                        score_data = metrics.get(metric, {})
                        row.append(score_data.get('std', 'N/A'))
                    
                    writer.writerow(row)
        
        logger.info(f"Saved summary to {csv_path}")
        
        logger.info(f"Saved markdown report to {md_path}")
    
    def print_summary(self):
        """Print a summary of results to console."""
        print("\n" + "="*80)
        print(f"BENCHMARK SUMMARY (Image Size: {self.target_size[0]}×{self.target_size[1]})")
        print("="*80)
        
        for dataset_name, methods in self.results.items():
            print(f"\nDataset: {dataset_name}")
            print("-"*60)
            print(f"{'Method':<20} " + " ".join(f"{metric:>10}" for metric in self.iqa_models.keys()))
            print("-"*60)
            
            for method_name, metrics in methods.items():
                scores = []
                for metric in self.iqa_models.keys():
                    score_data = metrics.get(metric, {})
                    mean = score_data.get('mean')
                    if mean is not None:
                        scores.append(f"{mean:>10.3f}")
                    else:
                        scores.append(f"{'N/A':>10}")
                
                print(f"{method_name:<20} " + " ".join(scores))

def main():
    # Configuration
    ROOT_DIR = "./test"  # Change this to your actual path
    METRICS = ['psnr', 'ssim', 'lpips', 'brisque']  # Metrics to evaluate
    DEVICE = 'cuda'  # or 'cpu'
    OUTPUT_DIR = "benchmark_results"
    TARGET_SIZE = (256, 256)  # **New: Set target size for resizing**
    
    # Create benchmark evaluator
    benchmark = DehazingBenchmark(
        root_dir=ROOT_DIR,
        metrics=METRICS,
        device=DEVICE,
        target_size=TARGET_SIZE  # **Pass target size**
    )
    
    # Run evaluation
    logger.info(f"Starting benchmark evaluation with {TARGET_SIZE[0]}×{TARGET_SIZE[1]} resize...")
    benchmark.run_evaluation()
    
    # Save results
    benchmark.save_results(OUTPUT_DIR)
    
    # Print summary
    benchmark.print_summary()
    
    logger.info("Benchmark evaluation completed!")

if __name__ == "__main__":
    main()
