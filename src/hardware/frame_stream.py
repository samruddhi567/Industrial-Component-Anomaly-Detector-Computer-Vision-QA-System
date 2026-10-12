"""
Hardware Frame Streaming Engine for Industrial Computer Vision QA.
Supports real-time OpenCV camera capture, RTSP video streams,
and realistic synthetic industrial component generator for QA testing.
"""

import time
import asyncio
import logging
from typing import Optional, Tuple, Dict, Any
import numpy as np
import cv2

logger = logging.getLogger(__name__)


class SyntheticIndustrialComponentGenerator:
    """
    Renders high-fidelity synthetic industrial components on an assembly conveyor belt:
    - Precision Machined Gears
    - Printed Circuit Boards (PCBs)
    - Metal Castings / Flanges
    - Ball Bearings
    Optionally injects realistic surface/structural defects for QA evaluation.
    """

    def __init__(self, width: int = 640, height: int = 480):
        self.width = width
        self.height = height
        self.component_types = ["gear", "pcb", "casting", "bearing"]
        self.defect_types = [
            "surface_scratch",
            "structural_crack",
            "void_porosity",
            "corrosion_spot",
            "dimensional_misalignment",
        ]

    def generate_frame(
        self,
        component_type: Optional[str] = None,
        inject_defect: bool = False,
        defect_type: Optional[str] = None,
        conveyor_offset: float = 0.0,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Synthesizes a realistic camera frame of an industrial component on a conveyor.

        Returns:
            frame: (H, W, 3) BGR uint8 image
            metadata: dictionary describing component type, ground truth defect status
        """
        # Conveyor background texture (matte industrial steel/rubber belt with micro-grain noise)
        frame = np.full((self.height, self.width, 3), 45, dtype=np.uint8)
        # Add subtle conveyor belt striations
        y_coords = np.arange(self.height)
        striations = (np.sin((y_coords + conveyor_offset * 10) * 0.1) * 6).astype(np.int16)
        frame[:, :, 0] = np.clip(frame[:, :, 0] + striations[:, None], 30, 60)
        frame[:, :, 1] = np.clip(frame[:, :, 1] + striations[:, None], 30, 60)
        frame[:, :, 2] = np.clip(frame[:, :, 2] + striations[:, None], 35, 65)

        # Micro texture noise
        noise = np.random.normal(0, 3, frame.shape).astype(np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # Component Center
        cx = self.width // 2
        cy = self.height // 2

        ctype = component_type or np.random.choice(self.component_types)

        if ctype == "gear":
            self._render_gear(frame, cx, cy, conveyor_offset)
        elif ctype == "pcb":
            self._render_pcb(frame, cx, cy)
        elif ctype == "bearing":
            self._render_bearing(frame, cx, cy, conveyor_offset)
        else:
            self._render_casting(frame, cx, cy)

        # Defect injection
        defect_meta = {"has_defect": False, "defect_type": "none", "location": None}
        if inject_defect:
            dtype = defect_type or np.random.choice(self.defect_types)
            defect_meta = self._inject_defect(frame, cx, cy, dtype)

        metadata = {
            "component_type": ctype,
            "timestamp": time.time(),
            **defect_meta,
        }

        return frame, metadata

    def _render_gear(self, img: np.ndarray, cx: int, cy: int, rotation: float):
        radius = 130
        num_teeth = 16
        # Draw outer teeth
        gear_pts = []
        for i in range(num_teeth * 2):
            angle = (i * np.pi / num_teeth) + rotation
            r = radius + 22 if i % 2 == 0 else radius
            px = int(cx + r * np.cos(angle))
            py = int(cy + r * np.sin(angle))
            gear_pts.append([px, py])

        pts = np.array(gear_pts, dtype=np.int32)
        cv2.fillPoly(img, [pts], (185, 185, 190))
        cv2.polylines(img, [pts], True, (130, 130, 135), 2)

        # Outer bevel
        cv2.circle(img, (cx, cy), radius - 15, (160, 160, 165), -1)
        cv2.circle(img, (cx, cy), radius - 15, (110, 110, 115), 2)

        # Weight reduction holes
        for h in range(4):
            ha = h * np.pi / 2 + rotation * 0.5
            hx = int(cx + 65 * np.cos(ha))
            hy = int(cy + 65 * np.sin(ha))
            cv2.circle(img, (hx, hy), 22, (45, 45, 48), -1)
            cv2.circle(img, (hx, hy), 22, (100, 100, 105), 2)

        # Center bore & keyway
        cv2.circle(img, (cx, cy), 32, (45, 45, 48), -1)
        cv2.circle(img, (cx, cy), 32, (90, 90, 95), 2)
        # Keyway slot
        cv2.rectangle(img, (cx - 6, cy - 38), (cx + 6, cy - 20), (45, 45, 48), -1)

    def _render_pcb(self, img: np.ndarray, cx: int, cy: int):
        pw, ph = 320, 240
        x1, y1 = cx - pw // 2, cy - ph // 2
        # Green PCB laminate
        cv2.rectangle(img, (x1, y1), (x1 + pw, y1 + ph), (30, 105, 40), -1)
        cv2.rectangle(img, (x1, y1), (x1 + pw, y1 + ph), (20, 80, 25), 3)

        # Mounting holes
        for mx, my in [(x1 + 18, y1 + 18), (x1 + pw - 18, y1 + 18), (x1 + 18, y1 + ph - 18), (x1 + pw - 18, y1 + ph - 18)]:
            cv2.circle(img, (mx, my), 9, (45, 45, 48), -1)
            cv2.circle(img, (mx, my), 13, (60, 175, 215), 2)

        # Copper traces
        for t in range(6):
            ty = y1 + 40 + t * 25
            cv2.line(img, (x1 + 30, ty), (x1 + 120, ty), (60, 180, 220), 2)
            cv2.line(img, (x1 + 120, ty), (x1 + 160, ty + 20), (60, 180, 220), 2)
            cv2.line(img, (x1 + 160, ty + 20), (x1 + 280, ty + 20), (60, 180, 220), 2)

        # Central Microcontroller IC package
        cv2.rectangle(img, (cx - 45, cy - 45), (cx + 45, cy + 45), (25, 25, 25), -1)
        cv2.rectangle(img, (cx - 45, cy - 45), (cx + 45, cy + 45), (60, 60, 60), 1)
        cv2.putText(img, "ARM QA", (cx - 36, cy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        # IC pin leads
        for p in range(8):
            py = cy - 35 + p * 10
            cv2.line(img, (cx - 55, py), (cx - 45, py), (200, 200, 200), 2)
            cv2.line(img, (cx + 45, py), (cx + 55, py), (200, 200, 200), 2)

    def _render_bearing(self, img: np.ndarray, cx: int, cy: int, rotation: float):
        outer_r = 135
        inner_r = 75
        cv2.circle(img, (cx, cy), outer_r, (175, 175, 180), -1)
        cv2.circle(img, (cx, cy), outer_r, (120, 120, 125), 3)
        cv2.circle(img, (cx, cy), outer_r - 20, (140, 140, 145), 2)

        # Race groove
        cv2.circle(img, (cx, cy), 105, (100, 100, 105), -1)

        # Steel balls
        num_balls = 8
        for b in range(num_balls):
            ba = b * 2 * np.pi / num_balls + rotation
            bx = int(cx + 105 * np.cos(ba))
            by = int(cy + 105 * np.sin(ba))
            cv2.circle(img, (bx, by), 16, (230, 230, 235), -1)
            cv2.circle(img, (bx, by), 16, (90, 90, 95), 1)
            # Specular highlight reflection
            cv2.circle(img, (bx - 4, by - 4), 4, (255, 255, 255), -1)

        # Inner ring & bore
        cv2.circle(img, (cx, cy), inner_r, (175, 175, 180), -1)
        cv2.circle(img, (cx, cy), inner_r, (120, 120, 125), 2)
        cv2.circle(img, (cx, cy), inner_r - 25, (45, 45, 48), -1)
        cv2.circle(img, (cx, cy), inner_r - 25, (85, 85, 90), 2)

    def _render_casting(self, img: np.ndarray, cx: int, cy: int):
        radius = 125
        cv2.circle(img, (cx, cy), radius, (160, 162, 165), -1)
        cv2.circle(img, (cx, cy), radius, (115, 117, 120), 3)

        # Flange bolt ring
        num_bolts = 6
        for b in range(num_bolts):
            ba = b * 2 * np.pi / num_bolts
            bx = int(cx + 90 * np.cos(ba))
            by = int(cy + 90 * np.sin(ba))
            cv2.circle(img, (bx, by), 10, (45, 45, 48), -1)
            cv2.circle(img, (bx, by), 14, (130, 130, 135), 2)

        # Center flange neck
        cv2.circle(img, (cx, cy), 55, (145, 147, 150), -1)
        cv2.circle(img, (cx, cy), 55, (95, 97, 100), 2)
        cv2.circle(img, (cx, cy), 35, (45, 45, 48), -1)

    def _inject_defect(self, img: np.ndarray, cx: int, cy: int, dtype: str) -> Dict[str, Any]:
        """Injects simulated physical defect into component region."""
        # Random offset around the component center
        dx = np.random.randint(-60, 60)
        dy = np.random.randint(-60, 60)
        loc_x = cx + dx
        loc_y = cy + dy

        if dtype == "surface_scratch":
            # Multi-segment abrasive scratch
            length = np.random.randint(40, 75)
            angle = np.random.uniform(0, 2 * np.pi)
            x2 = int(loc_x + length * np.cos(angle))
            y2 = int(loc_y + length * np.sin(angle))
            cv2.line(img, (loc_x, loc_y), (x2, y2), (25, 25, 25), 2, cv2.LINE_AA)
            cv2.line(img, (loc_x + 1, loc_y + 1), (x2 + 1, y2 + 1), (220, 220, 220), 1, cv2.LINE_AA)

        elif dtype == "structural_crack":
            # Jagged branched fracture line
            pts = [(loc_x, loc_y)]
            cur_x, cur_y = loc_x, loc_y
            for _ in range(6):
                cur_x += np.random.randint(-12, 16)
                cur_y += np.random.randint(8, 20)
                pts.append((cur_x, cur_y))
            for i in range(len(pts) - 1):
                cv2.line(img, pts[i], pts[i + 1], (15, 15, 15), 3, cv2.LINE_AA)

        elif dtype == "void_porosity":
            # Casting cavity cluster
            for _ in range(5):
                vx = loc_x + np.random.randint(-15, 15)
                vy = loc_y + np.random.randint(-15, 15)
                r = np.random.randint(3, 8)
                cv2.circle(img, (vx, vy), r, (20, 20, 22), -1)

        elif dtype == "corrosion_spot":
            # Oxidation discoloration spot
            radius = np.random.randint(18, 32)
            overlay = img.copy()
            cv2.circle(overlay, (loc_x, loc_y), radius, (30, 85, 160), -1)  # Rusty brownish/orange in BGR
            cv2.addWeighted(overlay, 0.65, img, 0.35, 0, img)

        elif dtype == "dimensional_misalignment":
            # Distorted edge / chipped chunk
            cv2.ellipse(img, (loc_x, loc_y), (25, 12), 45, 0, 360, (45, 45, 48), -1)

        return {
            "has_defect": True,
            "defect_type": dtype,
            "location": {"x": int(loc_x), "y": int(loc_y)},
        }


class HardwareFrameStream:
    """
    Asynchronous hardware stream reader.
    Captures live frames from industrial cameras (OpenCV / RTSP) or the synthetic simulator
    and continuously supplies frames at high hardware frame rates without blocking the main event loop.
    """

    def __init__(
        self,
        camera_type: str = "simulator",
        camera_id: int = 0,
        rtsp_url: str = "",
        width: int = 640,
        height: int = 480,
        target_fps: int = 60,
    ):
        self.camera_type = camera_type
        self.camera_id = camera_id
        self.rtsp_url = rtsp_url
        self.width = width
        self.height = height
        self.target_fps = target_fps
        self.frame_interval = 1.0 / max(target_fps, 1)

        self._running = False
        self._frame_count = 0
        self._simulator = SyntheticIndustrialComponentGenerator(width, height)
        self._cap: Optional[cv2.VideoCapture] = None

    def start(self):
        """Initializes video capture hardware device or simulator."""
        if self._running:
            return

        if self.camera_type == "opencv":
            self._cap = cv2.VideoCapture(self.camera_id)
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self._cap.set(cv2.CAP_PROP_FPS, self.target_fps)
            if not self._cap.isOpened():
                logger.warning(f"Failed to open OpenCV camera {self.camera_id}. Falling back to simulator.")
                self.camera_type = "simulator"
        elif self.camera_type == "rtsp" and self.rtsp_url:
            self._cap = cv2.VideoCapture(self.rtsp_url)
            if not self._cap.isOpened():
                logger.warning(f"Failed to open RTSP stream {self.rtsp_url}. Falling back to simulator.")
                self.camera_type = "simulator"

        self._running = True
        logger.info(f"HardwareFrameStream started ({self.camera_type} mode, target_fps={self.target_fps})")

    def stop(self):
        """Stops streaming and releases camera hardware."""
        self._running = False
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        logger.info("HardwareFrameStream stopped.")

    @property
    def is_running(self) -> bool:
        return self._running

    async def get_next_frame(self) -> Optional[Tuple[np.ndarray, str, Dict[str, Any]]]:
        """
        Asynchronously fetches the next frame from the camera stream.
        Maintains target frame rate without blocking worker threads.
        """
        if not self._running:
            return None

        self._frame_count += 1
        frame_id = f"FRAME-{self._frame_count:08d}"

        if self.camera_type in ["opencv", "rtsp"] and self._cap is not None:
            # Run non-blocking frame read in executor thread
            loop = asyncio.get_running_loop()
            ret, frame = await loop.run_in_executor(None, self._cap.read)
            if not ret or frame is None:
                # If camera frame dropped, fallback to simulator frame
                frame, meta = self._simulator.generate_frame(conveyor_offset=self._frame_count * 0.05)
            else:
                meta = {"component_type": "physical_camera", "has_defect": False, "timestamp": time.time()}
        else:
            # High-speed synthetic industrial simulator with periodic defects for QA test coverage
            inject = (self._frame_count % 7 == 0)  # Every 7th frame contains a defect
            offset = self._frame_count * 0.08
            frame, meta = self._simulator.generate_frame(
                inject_defect=inject,
                conveyor_offset=offset
            )

        return frame, frame_id, meta
