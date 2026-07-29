import sys
import unittest
import numpy as np

# Mock spidev
from unittest.mock import MagicMock
sys.modules['spidev'] = MagicMock()

# Mock pyrealsense2
sys.modules['pyrealsense2'] = MagicMock()

from spi import pack_movement, pack_interrupt, pack_speciality, pack_grabber
from vision import get_expected_depth_map, get_obstacle_mask
from trajectory import generate_waypoints

class TestPipeline(unittest.TestCase):

    def test_spi_serialization(self):
        # 00 opcode, 5 waypoints
        # (turn, forward)
        # let's try maximum values
        wps = [(127, 127), (-128, -128), (0, 0), (64, -64), (1, -1)]
        pkt = pack_movement(wps)
        self.assertEqual(len(pkt), 11, "Movement packet should be 11 bytes")
        
        # first byte should have opcode 00 at MSB
        self.assertEqual((pkt[0] >> 6) & 0x03, 0b00, "Opcode should be 00")
        
        # Test interrupt
        pkt2 = pack_interrupt()
        self.assertEqual(len(pkt2), 1)
        self.assertEqual((pkt2[0] >> 6) & 0x03, 0b01)

        # Test speciality (red)
        pkt3 = pack_speciality(1)
        self.assertEqual(len(pkt3), 1)
        self.assertEqual((pkt3[0] >> 6) & 0x03, 0b10)
        self.assertEqual((pkt3[0] >> 4) & 0x03, 1)

        # Test grabber
        pkt4 = pack_grabber(90)
        self.assertEqual(len(pkt4), 2)
        self.assertEqual((pkt4[0] >> 6) & 0x03, 0b11)

    def test_vision_obstacle_mask(self):
        w, h = 640, 480
        tilt_deg = 25.0
        height_mm = 150.0
        
        # Expected depth map
        expected = get_expected_depth_map(w, h, tilt_deg, height_mm)
        self.assertEqual(expected.shape, (h, w))
        
        # Create a mock depth frame in uint16 (RealSense standard)
        depth_scale = 0.001
        
        # Depth frame exactly matches expected
        mock_depth = (expected / (depth_scale * 1000.0)).astype(np.uint16)
        
        # No obstacles
        obs = get_obstacle_mask(mock_depth, depth_scale, expected)
        self.assertEqual(np.count_nonzero(obs), 0)
        
        # Introduce a block (closer by 50mm) in the center
        mock_depth[200:280, 300:340] = mock_depth[200:280, 300:340] - 50
        obs = get_obstacle_mask(mock_depth, depth_scale, expected)
        self.assertTrue(np.count_nonzero(obs) > 0, "Should detect obstacle")
        
    def test_trajectory_waypoints(self):
        state = {"error": 0.0, "heading": 0.0}
        target_error = 0.5
        curve = 0.1
        target_forward = 0.5
        
        waypoints = generate_waypoints(state, target_error, curve, target_forward, dt=0.2, num=5)
        self.assertEqual(len(waypoints), 5)
        for turn, fwd in waypoints:
            self.assertTrue(-128 <= turn <= 127)
            self.assertTrue(-128 <= fwd <= 127)

if __name__ == '__main__':
    unittest.main()
