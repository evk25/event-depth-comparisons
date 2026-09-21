#!/usr/bin/env python3
"""
Event-to-Depth Inference Script
================================

This script performs depth estimation from event camera data using a trained EventDepth model.

Key Details:
- Input: Event camera data accumulated over 33ms time windows
- Output: Metric depth map (in meters, typically 0.5-13.74m range)
- Model: EventDepth with 6.79M parameters
- Input resolution: 640x360 pixels
- Output resolution: 640x360 pixels
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
from model import EventDepth


class EventDepthPredictor:
    """Depth prediction from event camera data."""

    def __init__(self, checkpoint_path=None, device='cuda'):
        """
        Initialize the depth predictor.

        Args:
            checkpoint_path: Path to trained model checkpoint
                            If None, uses checkpoint_epoch_034.pt
            device: 'cuda' or 'cpu'
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        if checkpoint_path is None:
            checkpoint_path = Path(__file__).parent / 'checkpoint_epoch_034.pt'

        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(f'Checkpoint not found: {checkpoint_path}')

        # Load model
        self.model = EventDepth(in_channels=2, variant='base', global_mode='balanced')
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

        print(f'✓ Model loaded from: {checkpoint_path}')
        print(f'✓ Device: {self.device}')
        print(f'✓ Epoch: {checkpoint["epoch"]}')

    def predict(self, events_np):
        """
        Predict depth from event data.

        Args:
            events_np: Event data as numpy array (numpy.ndarray)

        INPUT SHAPE GUIDE:
        ==================
        The input events_np should be a numpy array with shape:

        SHAPE: (2, 360, 640)  [RECOMMENDED]
               ↓    ↓    ↓
               |    |    └─→ Width: 640 pixels
               |    └────────→ Height: 360 pixels
               └─────────────→ Channels: 2 (positive and negative events)

        CHANNEL MEANING:
        ================
        Channel 0 (events_np[0, :, :]): Positive events (brightness increase)
                                        Values typically 0-25 (normalized counts per millisecond)

        Channel 1 (events_np[1, :, :]): Negative events (brightness decrease)
                                        Values typically 0-25 (normalized counts per millisecond)

        HOW THE INPUT IS CREATED (33ms Accumulation):
        =============================================
        1. Events from camera arrive with microsecond timestamps

        2. Events are grouped into 33ms time windows
           Time window: [t_end - 33ms, t_end)
           Example: If t_end = 100ms, collect all events from [67ms, 100ms)

        3. For each pixel (x, y), count events per polarity:
           - Count events with polarity > 0  → Channel 0
           - Count events with polarity ≤ 0  → Channel 1

        4. Normalize by dividing by 33ms (accumulation time):
           normalized_count = raw_count / 33.0
           This gives events per millisecond, not raw counts

        5. Z-score normalization (done inside predict method):
           normalized = (value - mean) / (std + 1e-6)
           This centers the data around 0 with unit variance

        EXAMPLE: Creating event data manually
        =====================================
        # Simulate random event accumulation
        positive_events = np.random.randn(360, 640) * 5 + 10
        negative_events = np.random.randn(360, 640) * 5 + 8
        events_data = np.stack([positive_events, negative_events], axis=0)

        # Or load from file
        events_data = np.load('events_33ms_accumulated.npy')  # Shape: (2, 360, 640)

        Returns:
            dict with keys:
                'depth': depth map as numpy array, shape (360, 640), values in meters
                'depth_min': minimum depth value (meters)
                'depth_max': maximum depth value (meters)
                'depth_mean': mean depth value (meters)
        """
        # Validate input
        if not isinstance(events_np, np.ndarray):
            raise TypeError(f'Expected numpy array, got {type(events_np)}')

        if events_np.ndim != 3:
            raise ValueError(
                f'Expected 3D array (2, 360, 640), got shape {events_np.shape}'
            )

        if events_np.shape[0] != 2:
            raise ValueError(
                f'Expected 2 channels (pos/neg events), got {events_np.shape[0]}'
            )

        if events_np.shape[1:] != (360, 640):
            raise ValueError(
                f'Expected spatial size (360, 640), got {events_np.shape[1:]}'
            )

        # Convert to tensor and add batch dimension
        # Shape: (2, 360, 640) → (1, 2, 360, 640)
        events_tensor = torch.from_numpy(events_np).float().unsqueeze(0)

        # Z-score normalization (mean=0, std=1)
        # This is critical for stable inference across different event magnitudes
        mean = events_tensor.mean()
        std = events_tensor.std() + 1e-6
        events_tensor = (events_tensor - mean) / std

        # Run inference
        with torch.no_grad():
            depth_pred = self.model(events_tensor.to(self.device))

        # Convert output to numpy and remove batch dimension
        # Shape: (1, 1, 360, 640) → (360, 640)
        depth_np = depth_pred.squeeze().cpu().numpy()

        # Compute statistics
        depth_min = float(np.min(depth_np))
        depth_max = float(np.max(depth_np))
        depth_mean = float(np.mean(depth_np))

        return {
            'depth': depth_np,
            'depth_min': depth_min,
            'depth_max': depth_max,
            'depth_mean': depth_mean,
        }

    def visualize_depth(self, depth_np, title='Depth Map', cmap='turbo'):
        """
        Display depth map with visualization.

        Args:
            depth_np: Depth map (numpy array, shape 360x640)
            title: Title for the plot
            cmap: Matplotlib colormap name
        """
        # Normalize depth to 0-1 range using 2nd-98th percentiles
        # This handles outliers while preserving mid-range detail
        depth_min = np.percentile(depth_np, 2)
        depth_max = np.percentile(depth_np, 98)
        depth_norm = np.clip((depth_np - depth_min) / (depth_max - depth_min + 1e-6), 0, 1)

        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Raw depth with colormap
        im0 = axes[0].imshow(depth_norm, cmap=cmap)
        axes[0].set_title(f'{title} (Normalized)')
        axes[0].set_xlabel('X (pixels)')
        axes[0].set_ylabel('Y (pixels)')
        plt.colorbar(im0, ax=axes[0], label='Normalized Depth')

        # Metric depth with colorbar showing actual values
        im1 = axes[1].imshow(depth_np, cmap='viridis')
        axes[1].set_title(f'{title} (Metric)')
        axes[1].set_xlabel('X (pixels)')
        axes[1].set_ylabel('Y (pixels)')
        cbar = plt.colorbar(im1, ax=axes[1], label='Depth (meters)')

        plt.tight_layout()
        return fig

    def save_depth_map(self, depth_np, output_path):
        """
        Save depth map as 16-bit PNG for lossless storage.

        Args:
            depth_np: Depth map (numpy array)
            output_path: Path to save PNG file

        Note:
            PNG stores values in [0, 65535] range.
            We scale: depth_values * 1000 to preserve millimeter precision.
            To recover: depth = png_value / 1000
        """
        import cv2

        # Scale to 16-bit range (×1000 for millimeter precision)
        depth_uint16 = np.clip(depth_np * 1000, 0, 65535).astype(np.uint16)
        cv2.imwrite(str(output_path), depth_uint16)
        print(f'✓ Saved depth map: {output_path}')
        print(f'  Range: {depth_np.min():.3f}m - {depth_np.max():.3f}m')

    def save_visualization(self, depth_np, output_path, cmap='turbo'):
        """
        Save depth visualization as RGB PNG.

        Args:
            depth_np: Depth map (numpy array)
            output_path: Path to save PNG
            cmap: Matplotlib colormap
        """
        import cv2

        # Normalize using percentiles to handle outliers
        depth_min = np.percentile(depth_np, 2)
        depth_max = np.percentile(depth_np, 98)
        depth_norm = np.clip((depth_np - depth_min) / (depth_max - depth_min + 1e-6), 0, 1)

        # Apply colormap
        colormap = cm.get_cmap(cmap)
        depth_colored = colormap(depth_norm)[:, :, :3]  # RGB only
        depth_rgb = (depth_colored * 255).astype(np.uint8)

        # OpenCV expects BGR
        depth_bgr = cv2.cvtColor(depth_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(output_path), depth_bgr)
        print(f'✓ Saved visualization: {output_path}')


def main():
    """Example usage of the depth predictor."""

    # Initialize predictor with latest checkpoint
    predictor = EventDepthPredictor(device='cuda')

    # =========================================================================
    # EXAMPLE 1: Create synthetic event data and predict
    # =========================================================================
    print('\n' + '='*70)
    print('EXAMPLE 1: Synthetic Event Data')
    print('='*70)

    # Create synthetic event data (shape: 2, 360, 640)
    # Simulate events accumulated over 33ms with realistic magnitudes
    positive_events = np.random.randn(360, 640).astype(np.float32) * 4 + 12  # Mean ~12
    negative_events = np.random.randn(360, 640).astype(np.float32) * 3 + 10  # Mean ~10
    events_synthetic = np.stack([positive_events, negative_events], axis=0)

    print(f'Created synthetic events: {events_synthetic.shape}')
    print(f'  Pos events: [{events_synthetic[0].min():.2f}, {events_synthetic[0].max():.2f}]')
    print(f'  Neg events: [{events_synthetic[1].min():.2f}, {events_synthetic[1].max():.2f}]')

    # Predict depth
    result = predictor.predict(events_synthetic)
    depth_pred = result['depth']

    print(f'\nPredicted depth: {depth_pred.shape}')
    print(f'  Min: {result["depth_min"]:.4f}m')
    print(f'  Max: {result["depth_max"]:.4f}m')
    print(f'  Mean: {result["depth_mean"]:.4f}m')

    # Visualize
    fig = predictor.visualize_depth(depth_pred, 'Synthetic Events Depth')
    plt.savefig('output_synthetic_depth_vis.png', dpi=100, bbox_inches='tight')
    print('✓ Saved visualization: output_synthetic_depth_vis.png')

    # Save as 16-bit PNG
    predictor.save_depth_map(depth_pred, 'output_synthetic_depth.png')
    predictor.save_visualization(depth_pred, 'output_synthetic_depth_color.png')

    # =========================================================================
    # EXAMPLE 2: Load events from file and predict
    # =========================================================================
    print('\n' + '='*70)
    print('EXAMPLE 2: Load Events from File')
    print('='*70)

    # Save synthetic data as example
    np.save('sample_events.npy', events_synthetic)
    print('Saved sample_events.npy (shape: 2, 360, 640)')

    # Load and predict
    events_loaded = np.load('sample_events.npy')
    result = predictor.predict(events_loaded)
    depth_pred = result['depth']

    print(f'Loaded and predicted depth from file')
    print(f'  Depth range: {result["depth_min"]:.4f}m - {result["depth_max"]:.4f}m')

    predictor.save_depth_map(depth_pred, 'output_loaded_depth.png')
    predictor.save_visualization(depth_pred, 'output_loaded_depth_color.png')

    # =========================================================================
    # EXAMPLE 3: Batch prediction
    # =========================================================================
    print('\n' + '='*70)
    print('EXAMPLE 3: Batch Prediction')
    print('='*70)

    # Create multiple event samples
    batch_size = 3
    events_batch = [
        np.random.randn(2, 360, 640).astype(np.float32) * 4 + 10
        for _ in range(batch_size)
    ]

    depths_batch = []
    for i, events in enumerate(events_batch):
        result = predictor.predict(events)
        depths_batch.append(result['depth'])
        print(f'Sample {i}: {result["depth_min"]:.4f}m - {result["depth_max"]:.4f}m')

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print('\n' + '='*70)
    print('SUMMARY')
    print('='*70)
    print(f'✓ Model inference working correctly')
    print(f'✓ Input shape: (2, 360, 640) - 2 polarity channels')
    print(f'✓ Output shape: (360, 640) - metric depth in meters')
    print(f'✓ Typical depth range: 0.5m - 13.74m')
    print(f'\nOutput files saved:')
    print(f'  - output_synthetic_depth_vis.png')
    print(f'  - output_synthetic_depth.png (16-bit encoded)')
    print(f'  - output_synthetic_depth_color.png')
    print(f'  - output_loaded_depth.png')
    print(f'  - output_loaded_depth_color.png')
    print('='*70 + '\n')


if __name__ == '__main__':
    main()
