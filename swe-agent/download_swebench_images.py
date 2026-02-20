#!/usr/bin/env python3
"""
Download all Docker images for SWE-bench datasets.
This script fetches the list of instances and pulls all required Docker images.
"""

import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Set

from datasets import load_dataset
from tqdm import tqdm


def get_docker_image_name(instance_id: str) -> str:
    """Generate Docker image name from instance ID."""
    # Docker doesn't allow double underscore, so we replace them with a magic token
    id_docker_compatible = instance_id.replace("__", "_1776_")
    return f"swebench/sweb.eval.x86_64.{id_docker_compatible}:latest".lower()


def get_swebench_images(subset: str = "lite", split: str = "test") -> Set[str]:
    """Get all unique Docker image names for a SWE-bench subset."""
    dataset_map = {
        "lite": "princeton-nlp/SWE-bench_Lite",
        "verified": "princeton-nlp/SWE-bench_Verified", 
        "full": "princeton-nlp/SWE-bench",
        "multimodal": "princeton-nlp/SWE-bench_Multimodal",
    }
    
    if subset not in dataset_map:
        raise ValueError(f"Unknown subset: {subset}. Choose from {list(dataset_map.keys())}")
    
    print(f"Loading {subset} dataset (split={split})...")
    dataset = load_dataset(dataset_map[subset], split=split)
    
    images = set()
    for instance in dataset:
        if 'image_name' in instance and instance['image_name']:
            images.add(instance['image_name'])
        else:
            # Generate default image name
            images.add(get_docker_image_name(instance['instance_id']))
    
    return images


def check_image_exists(image_name: str) -> bool:
    """Check if a Docker image exists locally."""
    try:
        result = subprocess.run(
            ["docker", "images", "-q", image_name],
            capture_output=True,
            text=True,
            check=False
        )
        return bool(result.stdout.strip())
    except Exception:
        return False


def pull_docker_image(image_name: str, force: bool = False, timeout: int = 120) -> tuple[str, bool, str]:
    """
    Pull a Docker image with timeout.
    
    Args:
        image_name: Name of the Docker image to pull
        force: Force re-download even if image exists
        timeout: Timeout in seconds (default: 120 seconds = 2 minutes)
    
    Returns:
        tuple: (image_name, success, message)
    """
    # Check if image already exists
    if not force and check_image_exists(image_name):
        return (image_name, True, "Already exists")
    
    try:
        print(f"Pulling {image_name}...")
        result = subprocess.run(
            ["docker", "pull", image_name],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=True
        )
        return (image_name, True, "Successfully pulled")
    except subprocess.TimeoutExpired:
        # Kill any hanging docker pull process
        subprocess.run(["pkill", "-f", f"docker pull {image_name}"], capture_output=True)
        return (image_name, False, f"Timeout after {timeout} seconds")
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr if e.stderr else str(e)
        return (image_name, False, f"Failed: {error_msg}")
    except Exception as e:
        return (image_name, False, f"Error: {str(e)}")


def download_images_sequential(images: List[str], force: bool = False, timeout: int = 120, max_retries: int = 3) -> None:
    """Download images sequentially with progress bar and automatic retry on timeout.
    
    Args:
        images: List of Docker image names to download
        force: Force re-download even if image exists
        timeout: Timeout in seconds for each download attempt (default: 120)
        max_retries: Maximum number of retries for timeout errors (default: 3)
    """
    failed_images = []
    skipped_images = []
    
    for image in tqdm(images, desc="Downloading images"):
        retry_count = 0
        success = False
        message = ""
        
        while retry_count < max_retries:
            image_name, success, message = pull_docker_image(image, force, timeout)
            
            # If timeout occurred, retry
            if not success and "Timeout" in message:
                retry_count += 1
                if retry_count < max_retries:
                    tqdm.write(f"⟳ {image_name}: {message}, retrying ({retry_count}/{max_retries})...")
                    continue
            
            # Exit retry loop for success or non-timeout failures
            break
        
        if not success:
            failed_images.append((image_name, message))
            tqdm.write(f"✗ {image_name}: {message}")
        elif message == "Already exists":
            skipped_images.append(image_name)
            tqdm.write(f"○ {image_name}: {message}")
        else:
            tqdm.write(f"✓ {image_name}: {message}")
    
    # Print summary
    print("\n" + "="*80)
    print("DOWNLOAD SUMMARY")
    print("="*80)
    print(f"Total images: {len(images)}")
    print(f"Downloaded: {len(images) - len(failed_images) - len(skipped_images)}")
    print(f"Skipped (already exist): {len(skipped_images)}")
    print(f"Failed: {len(failed_images)}")
    
    if failed_images:
        print("\nFailed images:")
        for image, error in failed_images:
            print(f"  - {image}: {error}")


def download_images_parallel(images: List[str], force: bool = False, max_workers: int = 4, timeout: int = 120) -> None:
    """Download images in parallel with progress bar and timeout.
    
    Args:
        images: List of Docker image names to download
        force: Force re-download even if image exists
        max_workers: Number of parallel workers
        timeout: Timeout in seconds for each download (default: 120)
    """
    failed_images = []
    skipped_images = []
    successful_images = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all download tasks with timeout
        futures = {executor.submit(pull_docker_image, img, force, timeout): img for img in images}
        
        # Process completed tasks with progress bar
        with tqdm(total=len(images), desc="Downloading images") as pbar:
            for future in as_completed(futures):
                image_name, success, message = future.result()
                
                if not success:
                    failed_images.append((image_name, message))
                    tqdm.write(f"✗ {image_name}: {message}")
                elif message == "Already exists":
                    skipped_images.append(image_name)
                    tqdm.write(f"○ {image_name}: Skipped (already exists)")
                else:
                    successful_images.append(image_name)
                    tqdm.write(f"✓ {image_name}: Downloaded")
                
                pbar.update(1)
    
    # Print summary
    print("\n" + "="*80)
    print("DOWNLOAD SUMMARY")
    print("="*80)
    print(f"Total images: {len(images)}")
    print(f"Downloaded: {len(successful_images)}")
    print(f"Skipped (already exist): {len(skipped_images)}")
    print(f"Failed: {len(failed_images)}")
    
    if failed_images:
        print("\nFailed images:")
        for image, error in failed_images:
            print(f"  - {image}: {error}")


def estimate_disk_space(images: List[str]) -> None:
    """Estimate required disk space for images."""
    # Rough estimates based on typical SWE-bench image sizes
    avg_image_size_gb = 2.5  # Average size in GB
    total_size_gb = len(images) * avg_image_size_gb
    
    print(f"\nEstimated disk space required: ~{total_size_gb:.1f} GB")
    print(f"(Based on average image size of ~{avg_image_size_gb} GB)")


def main():
    parser = argparse.ArgumentParser(
        description='Download Docker images for SWE-bench datasets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download all SWE-bench-lite images (default, sequential)
  python download_swebench_images.py
  
  # Download SWE-bench verified images (sequential)
  python download_swebench_images.py --subset verified
  
  # Download with parallel workers (use with caution if using proxy)
  python download_swebench_images.py --parallel --workers 2
  
  # Force re-download even if images exist
  python download_swebench_images.py --force
  
  # Download specific images from a file
  python download_swebench_images.py --image-list images.txt
  
  # Just list images without downloading
  python download_swebench_images.py --list-only
        """
    )
    
    parser.add_argument('--subset', default='lite',
                       choices=['lite', 'verified', 'full', 'multimodal'],
                       help='SWE-bench subset to download (default: lite)')
    
    parser.add_argument('--split', default='test',
                       choices=['test', 'dev'],
                       help='Dataset split to use (default: test)')
    
    parser.add_argument('--parallel', action='store_true',
                       help='Download images in parallel (default: sequential/single-threaded)')
    
    parser.add_argument('--workers', type=int, default=2,
                       help='Number of parallel download workers when using --parallel (default: 2)')
    
    parser.add_argument('--force', action='store_true',
                       help='Force re-download even if images already exist')
    
    parser.add_argument('--timeout', type=int, default=120,
                       help='Timeout in seconds for each image download (default: 120)')
    
    parser.add_argument('--max-retries', type=int, default=3,
                       help='Maximum number of retries for timeout errors (default: 3)')
    
    parser.add_argument('--image-list', type=Path,
                       help='Path to a file containing image names (one per line)')
    
    parser.add_argument('--list-only', action='store_true',
                       help='Only list images without downloading')
    
    parser.add_argument('--output-list', type=Path,
                       help='Save image list to a file')
    
    args = parser.parse_args()
    
    try:
        # Get list of images
        if args.image_list:
            print(f"Loading images from {args.image_list}...")
            with open(args.image_list) as f:
                images = [line.strip() for line in f if line.strip()]
            images = list(set(images))  # Remove duplicates
        else:
            print(f"Getting image list for SWE-bench-{args.subset} ({args.split} split)...")
            images = list(get_swebench_images(args.subset, args.split))
        
        images.sort()  # Sort for consistent ordering
        
        print(f"Found {len(images)} unique images")
        
        # Save image list if requested
        if args.output_list:
            with open(args.output_list, 'w') as f:
                for img in images:
                    f.write(f"{img}\n")
            print(f"Image list saved to {args.output_list}")
        
        # List only mode
        if args.list_only:
            print("\nImages:")
            for img in images:
                exists = "✓" if check_image_exists(img) else "✗"
                print(f"  {exists} {img}")
            return
        
        # Estimate disk space
        estimate_disk_space(images)
        
        # Confirm before downloading
        response = input(f"\nProceed with downloading {len(images)} images? [Y/n]: ")
        if response.lower() in ['n', 'no']:
            print("Aborted.")
            return
        
        # Download images
        print("\nStarting download...")
        print(f"Timeout per image: {args.timeout} seconds")
        print(f"Max retries on timeout: {args.max_retries}")
        
        if args.parallel:
            download_images_parallel(images, args.force, args.workers, args.timeout)
        else:
            download_images_sequential(images, args.force, args.timeout, args.max_retries)
            
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()