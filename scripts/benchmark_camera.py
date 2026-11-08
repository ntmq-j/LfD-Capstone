#!/usr/bin/env python3
"""
Benchmark RealSense camera with different configurations to find VM bottleneck
"""

import pyrealsense2 as rs
import numpy as np
import time
import sys

def benchmark_config(width, height, fps, color_fmt, depth_fmt, duration=10):
    """Test a specific configuration"""
    print(f"\nTesting: {width}x{height} @ {fps}fps")
    print(f"  Color: {color_fmt}, Depth: {depth_fmt}")

    pipeline = rs.pipeline()
    config = rs.config()

    try:
        config.enable_stream(rs.stream.color, width, height, color_fmt, fps)
        config.enable_stream(rs.stream.depth, width, height, depth_fmt, fps)

        # Start pipeline
        profile = pipeline.start(config)

        # Let it stabilize
        time.sleep(1)
        for _ in range(30):
            pipeline.wait_for_frames(timeout_ms=5000)

        # Benchmark
        frame_count = 0
        start_time = time.time()
        timeout_count = 0

        while time.time() - start_time < duration:
            try:
                frames = pipeline.wait_for_frames(timeout_ms=1000)
                color_frame = frames.get_color_frame()
                depth_frame = frames.get_depth_frame()

                if color_frame and depth_frame:
                    frame_count += 1

                    # Actually process data (simulate real usage)
                    color_data = np.asanyarray(color_frame.get_data())
                    depth_data = np.asanyarray(depth_frame.get_data())

            except RuntimeError as e:
                timeout_count += 1
                if "wait_for_frames" in str(e):
                    print(f"  [TIMEOUT] Frame timeout occurred")
                else:
                    raise

        elapsed = time.time() - start_time
        actual_fps = frame_count / elapsed

        pipeline.stop()

        # Results
        success = actual_fps > (fps * 0.8)  # Within 80% of target
        status = "✓ PASS" if success else "✗ FAIL"

        print(f"  {status} - Captured {frame_count} frames in {elapsed:.1f}s")
        print(f"  Actual FPS: {actual_fps:.1f} (target: {fps})")
        if timeout_count > 0:
            print(f"  Timeouts: {timeout_count}")

        return {
            'config': f"{width}x{height}@{fps}",
            'target_fps': fps,
            'actual_fps': actual_fps,
            'frames': frame_count,
            'timeouts': timeout_count,
            'success': success
        }

    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        try:
            pipeline.stop()
        except:
            pass
        return {
            'config': f"{width}x{height}@{fps}",
            'error': str(e),
            'success': False
        }

def main():
    print("="*70)
    print("RealSense D435i Performance Benchmark")
    print("="*70)

    # Test configurations from low to high demand
    test_configs = [
        # (width, height, fps, color_format, depth_format)
        (424, 240, 6, rs.format.bgr8, rs.format.z16),
        (424, 240, 15, rs.format.bgr8, rs.format.z16),
        (424, 240, 30, rs.format.bgr8, rs.format.z16),
        (640, 480, 6, rs.format.bgr8, rs.format.z16),
        (640, 480, 15, rs.format.bgr8, rs.format.z16),
        (640, 480, 30, rs.format.bgr8, rs.format.z16),
        (848, 480, 6, rs.format.bgr8, rs.format.z16),
        (848, 480, 15, rs.format.bgr8, rs.format.z16),
        (848, 480, 30, rs.format.bgr8, rs.format.z16),
        (1280, 720, 6, rs.format.bgr8, rs.format.z16),
        (1280, 720, 15, rs.format.bgr8, rs.format.z16),
        (1280, 720, 30, rs.format.bgr8, rs.format.z16),
        (1920, 1080, 6, rs.format.bgr8, rs.format.z16),
        (1920, 1080, 15, rs.format.bgr8, rs.format.z16),
        (1920, 1080, 30, rs.format.bgr8, rs.format.z16),
    ]

    results = []
    for config in test_configs:
        result = benchmark_config(*config, duration=10)
        results.append(result)
        time.sleep(1)  # Cool down between tests

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"{'Configuration':<20} {'Target FPS':<12} {'Actual FPS':<12} {'Status':<10}")
    print("-"*70)

    for r in results:
        if 'error' in r:
            print(f"{r['config']:<20} {'N/A':<12} {'ERROR':<12} {'✗ FAIL':<10}")
        else:
            status = "✓ PASS" if r['success'] else "✗ FAIL"
            print(f"{r['config']:<20} {r['target_fps']:<12} {r['actual_fps']:<12.1f} {status:<10}")

    # Find best working configuration
    print("\n" + "="*70)
    working_configs = [r for r in results if r.get('success', False)]
    if working_configs:
        best = max(working_configs, key=lambda x: x['actual_fps'])
        print(f"Best working config: {best['config']} @ {best['actual_fps']:.1f} FPS")
    else:
        print("No configurations passed the test!")

    print("="*70)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[INFO] Benchmark interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
