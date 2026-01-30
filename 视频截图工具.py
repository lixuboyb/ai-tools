import os
import sys
import time
import subprocess
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import Qt, QTimer, QSize, QPoint, QThread, Signal
from PySide6.QtGui import (QAction, QImage, QPixmap, QKeySequence, QPalette, 
                           QColor, QShortcut, QPainter, QPen, QIcon)
from PySide6.QtWidgets import (
    QApplication, QFileDialog, QLabel, QMainWindow, QMessageBox,
    QPushButton, QHBoxLayout, QVBoxLayout, QWidget, QComboBox,
    QToolBar, QStatusBar, QGroupBox, QFormLayout, QSlider, QLineEdit, QSizePolicy
)


# 自定义VideoLabel：拖动时显示临时虚线框，松开后自动消失
class VideoLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent  # 关联主窗口
        self.is_selecting = False  # 是否处于选框拖动中
        self.select_start = QPoint()  # 拖动起点（绝对像素）
        self.temp_rect_abs = None  # 拖动中临时框（仅拖动时有效）
        self.select_rect_rel = None  # 最终选框（相对比例坐标，0-1范围）
        # 图像显示信息
        self.image_offset_x = 0
        self.image_offset_y = 0
        self.image_scaled_w = 0
        self.image_scaled_h = 0

    def mousePressEvent(self, event):
        # 左键按下开始选择
        if event.button() == Qt.LeftButton and self.parent.capture is not None:
            # 转换坐标到实际图像区域
            x = event.pos().x()
            y = event.pos().y()
            
            # 检查是否在图像区域内
            if (self.image_offset_x <= x < self.image_offset_x + self.image_scaled_w and
                self.image_offset_y <= y < self.image_offset_y + self.image_scaled_h):
                self.is_selecting = True
                self.select_start = event.pos()
                self.temp_rect_abs = None
                self.parent.status.showMessage("📏 拖动鼠标选择裁剪区域...")
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        # 拖动过程中显示临时虚线框（视觉反馈）
        if self.is_selecting and self.parent.current_frame_bgr is not None:
            current_pos = event.pos()
            # 计算临时框坐标（确保左上角到右下角）
            x1_abs = min(self.select_start.x(), current_pos.x())
            y1_abs = min(self.select_start.y(), current_pos.y())
            x2_abs = max(self.select_start.x(), current_pos.x())
            y2_abs = max(self.select_start.y(), current_pos.y())
            self.temp_rect_abs = (x1_abs, y1_abs, x2_abs, y2_abs)
            self.update()  # 触发重绘显示临时框
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.is_selecting:
            self.is_selecting = False
            self.temp_rect_abs = None  # 松开后立即清除临时框
            self.update()  # 触发重绘，使临时框消失
            
            if self.parent.current_frame_bgr is not None:
                # 检查起点和终点是否都在图像区域内
                start_x, start_y = self.select_start.x(), self.select_start.y()
                end_x, end_y = event.pos().x(), event.pos().y()
                
                # 检查是否在图像区域内
                in_image_area = (
                    (self.image_offset_x <= start_x < self.image_offset_x + self.image_scaled_w) and
                    (self.image_offset_y <= start_y < self.image_offset_y + self.image_scaled_h) and
                    (self.image_offset_x <= end_x < self.image_offset_x + self.image_scaled_w) and
                    (self.image_offset_y <= end_y < self.image_offset_y + self.image_scaled_h)
                )
                
                if not in_image_area:
                    self.select_rect_rel = None
                    self.parent._update_crop_res_label(0, 0)
                    self.parent.status.showMessage("❌ 选框无效，已取消选择")
                    return

                # 计算相对比例选框（基于实际图像区域）
                if self.image_scaled_w == 0 or self.image_scaled_h == 0:
                    self.select_rect_rel = None
                    self.parent._update_crop_res_label(0, 0)
                    self.parent.status.showMessage("❌ 选框无效，已取消选择")
                    return

                # 将坐标转换为相对于图像区域的坐标
                rx1 = max(0.0, min((start_x - self.image_offset_x) / self.image_scaled_w, 1.0))
                ry1 = max(0.0, min((start_y - self.image_offset_y) / self.image_scaled_h, 1.0))
                rx2 = max(0.0, min((end_x - self.image_offset_x) / self.image_scaled_w, 1.0))
                ry2 = max(0.0, min((end_y - self.image_offset_y) / self.image_scaled_h, 1.0))
                
                # 确保选框为正
                rx1, rx2 = min(rx1, rx2), max(rx1, rx2)
                ry1, ry2 = min(ry1, ry2), max(ry1, ry2)

                # 过滤过小选框
                if (rx2 - rx1) < 0.01 or (ry2 - ry1) < 0.01:
                    self.select_rect_rel = None
                    self.parent._update_crop_res_label(0, 0)
                    self.parent.status.showMessage("📏 选框过小，已取消选择")
                else:
                    self.select_rect_rel = (rx1, ry1, rx2, ry2)
                    # 计算裁剪分辨率并更新显示
                    frame_h, frame_w = self.parent.current_frame_bgr.shape[:2]
                    crop_w = int((rx2 - rx1) * frame_w)
                    crop_h = int((ry2 - ry1) * frame_h)
                    self.parent._update_crop_res_label(crop_w, crop_h)
                    self.parent.status.showMessage(
                        f"✅ 已选择裁剪区域（{crop_w}×{crop_h}）| 🚫 按Esc取消选择"
                    )
            # 刷新帧显示（应用裁剪）
            if self.parent.current_frame_bgr is not None:
                self.parent._show_frame(self.parent.current_frame_bgr)
        super().mouseReleaseEvent(event)

    def paintEvent(self, event):
        super().paintEvent(event)
        # 仅在拖动过程中显示临时红色虚线框
        if self.is_selecting and self.temp_rect_abs is not None:
            painter = QPainter(self)
            pen = QPen(QColor(255, 0, 0), 2, Qt.DashLine)  # 红色虚线
            painter.setPen(pen)
            x1, y1, x2, y2 = self.temp_rect_abs
            painter.drawRect(x1, y1, x2 - x1, y2 - y1)

    # 取消选择
    def cancel_selection(self):
        self.select_rect_rel = None
        self.temp_rect_abs = None
        self.is_selecting = False
        self.update()
        self.parent._update_crop_res_label(0, 0)


# 后台线程：异步估算视频总帧数，避免阻塞UI线程
class EstimateFramesThread(QThread):
    finished = Signal(int)

    def __init__(self, file_path: str, max_iterations: int = 20):
        super().__init__()
        self.file_path = file_path
        self.max_iterations = max_iterations

    def run(self):
        last_ok = 0
        try:
            cap = cv2.VideoCapture(self.file_path)
            if not cap.isOpened():
                try:
                    cap.release()
                except Exception:
                    pass
                self.finished.emit(0)
                return

            def can_read(index: int) -> bool:
                try:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
                    ok, _ = cap.read()
                    return bool(ok)
                except Exception:
                    return False

            low, high = 0, 1
            it = 0
            while it < self.max_iterations and can_read(high):
                low = high
                high *= 2
                it += 1

            left, right = low, high
            last_ok = low if can_read(low) else 0
            while left <= right and it < self.max_iterations + 30:
                mid = (left + right) // 2
                if can_read(mid):
                    last_ok = mid
                    left = mid + 1
                else:
                    right = mid - 1
                it += 1

            try:
                cap.release()
            except Exception:
                pass
        except Exception:
            last_ok = 0

        self.finished.emit(int(max(0, last_ok)))


class VideoPlayer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🎬 视频播放器 - 区域裁剪+帧保存与倍速")
        self.resize(1200, 800)
        self.setMinimumSize(800, 600)

        # State
        self.capture = None
        self.timer = QTimer(self)
        self.timer.setTimerType(Qt.PreciseTimer)
        self.timer.timeout.connect(self._read_frame)
        self.current_frame_bgr = None  # 原始BGR帧
        self.video_path = None
        self.fps = 30.0
        self.speed = 1.0
        self.ended = False
        self.total_frames = 0
        self.default_save_dir = str(Path.cwd() / "saved_frames")
        Path(self.default_save_dir).mkdir(parents=True, exist_ok=True)
        # 裁剪分辨率显示标签
        self.crop_res_label = None
        # 进度条相关
        self.is_seeking = False  # 是否正在拖动进度条
        self.was_playing_before_seek = False  # 拖动进度条前是否在播放
        # 记住上次打开的文件夹路径
        self.last_open_dir = None
        # 图片命名基准
        self.base_name = "frame"
        self.name_counter = 0
        # 图片命名基准输入框
        self.base_name_input = None
        # 视频名称显示标签
        self.video_name_label = None
        # 保存格式选择
        self.save_format = "png"
        self.save_format_combo = None
        # 用于可能的后台估算线程引用，避免被GC
        self._est_thread = None

        # UI
        self._init_actions()
        self._init_toolbar()
        self._init_central()
        self._init_statusbar()
        self._init_shortcuts()
        self._apply_theme()
        self._update_controls_enabled(False)
        # 初始化分辨率显示
        self._update_crop_res_label(0, 0)

    def _init_actions(self):
        self.action_open = QAction("📁 打开视频", self)
        self.action_open.triggered.connect(self.open_video)

        self.action_change_dir = QAction("📂 更改保存文件夹", self)
        self.action_change_dir.triggered.connect(self.change_save_dir)

        self.action_open_dir = QAction("📂 打开保存文件夹", self)
        self.action_open_dir.triggered.connect(self.open_save_dir)

        self.action_exit = QAction("❌ 退出", self)
        self.action_exit.triggered.connect(self.close)

        self.action_cancel_crop = QAction("🚫 取消区域选择 (Esc)", self)
        self.action_cancel_crop.triggered.connect(self._cancel_crop_selection)

    def _init_toolbar(self):
        toolbar = QToolBar("主工具栏", self)
        toolbar.setIconSize(QSize(16, 16))
        self.addToolBar(toolbar)

        toolbar.addAction(self.action_open)
        toolbar.addSeparator()
        toolbar.addAction(self.action_open_dir)
        toolbar.addSeparator()
        toolbar.addAction(self.action_exit)
        toolbar.addSeparator()
        
        # 添加弹性空间将输入框推到右侧
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        toolbar.addWidget(spacer)
        
        # 添加保存格式选择下拉框
        self.save_format_combo = QComboBox()
        self.save_format_combo.addItems(["png", "jpg", "jpeg", "bmp", "tiff"])
        self.save_format_combo.setCurrentText(self.save_format)
        self.save_format_combo.currentTextChanged.connect(self._on_save_format_changed)
        self.save_format_combo.setMinimumWidth(80)
        toolbar.addWidget(QLabel("保存格式:"))
        toolbar.addWidget(self.save_format_combo)
        
        # 添加图片命名基准输入框到右侧
        self.base_name_input = QLineEdit()
        self.base_name_input.setText(self.base_name)
        self.base_name_input.setPlaceholderText("图片命名基准")
        self.base_name_input.setMaximumWidth(120)
        self.base_name_input.textChanged.connect(self._on_base_name_changed)
        # 设置样式使文本可见
        self.base_name_input.setStyleSheet("""
            QLineEdit {
                color: #000000;
                background: #ffffff;
                border: 1px solid #cccccc;
                border-radius: 4px;
                padding: 4px;
                font-size: 14px;
            }
        """)
        toolbar.addWidget(self.base_name_input)

    def _init_central(self):
        # 视频显示区
        self.video_label = VideoLabel(self)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("""
            QLabel { 
                background: #000; 
                color: #ddd; 
                border: 2px solid #3a3a3a; 
                border-radius: 8px;
            }
        """)
        self.video_label.setMinimumSize(720, 405)

        # 控制按钮
        self.button_play = QPushButton("▶ 播放")
        self.button_play.clicked.connect(self.toggle_play)
        self.button_play.setMinimumWidth(100)

        self.button_pause = QPushButton("⏸ 暂停")
        self.button_pause.clicked.connect(self.pause)
        self.button_pause.setMinimumWidth(100)

        self.button_save = QPushButton("📷 保存当前帧 (Q)")
        self.button_save.clicked.connect(self.save_current_frame)
        self.button_save.setMinimumWidth(150)

        self.speed_combo = QComboBox()
        self.speed_combo.addItems(["0.25x", "0.5x", "0.75x", "1x", "1.25x", "1.5x", "2x", "3x", "4x"]) 
        self.speed_combo.setCurrentText("1x")
        self.speed_combo.currentTextChanged.connect(self._on_speed_changed)
        self.speed_combo.setMinimumWidth(80)

        # 视频名称显示标签
        self.video_name_label = QLabel("未加载视频")
        self.video_name_label.setStyleSheet("color: #4cd964; font-weight: bold; font-size: 14px;")
        
        # 裁剪分辨率显示标签
        self.crop_res_label = QLabel("裁剪分辨率：无")
        self.crop_res_label.setStyleSheet("color: #4cd964; font-weight: bold; font-size: 14px;")

        # 进度条
        self.progress_slider = QSlider(Qt.Horizontal)
        self.progress_slider.setRange(0, 1000)  # 使用1000作为最大值以提高精度
        self.progress_slider.setValue(0)
        self.progress_slider.sliderPressed.connect(self._on_slider_pressed)
        self.progress_slider.sliderReleased.connect(self._on_slider_released)
        self.progress_slider.sliderMoved.connect(self._on_slider_moved)
        self.progress_slider.setMinimumHeight(20)

        # 控制栏布局
        controls_top = QHBoxLayout()
        controls_top.addWidget(self.button_play)
        controls_top.addWidget(self.button_pause)
        controls_top.addSpacing(15)
        controls_top.addWidget(QLabel("播放速度:"))
        controls_top.addWidget(self.speed_combo)
        controls_top.addSpacing(25)
        controls_top.addWidget(self.crop_res_label)
        controls_top.addStretch(1)
        # 添加视频名称显示标签
        controls_top.addWidget(self.video_name_label)
        # 添加取消区域选择按钮
        self.button_cancel_crop = QPushButton("🚫 取消区域选择")
        self.button_cancel_crop.clicked.connect(self._cancel_crop_selection)
        controls_top.addWidget(self.button_cancel_crop)
        controls_top.addWidget(self.button_save)

        # 进度条布局
        progress_layout = QHBoxLayout()
        self.current_time_label = QLabel("00:00")
        self.current_time_label.setStyleSheet("font-family: monospace; font-size: 14px; color: #cfcfcf;")
        self.total_time_label = QLabel("00:00")
        self.total_time_label.setStyleSheet("font-family: monospace; font-size: 14px; color: #cfcfcf;")
        progress_layout.addWidget(self.current_time_label)
        progress_layout.addWidget(self.progress_slider)
        progress_layout.addWidget(self.total_time_label)

        # 保存设置组
        save_group = QGroupBox("保存设置")
        save_form = QFormLayout()
        self.save_dir_label = QLabel(self.default_save_dir)
        self.save_dir_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.save_dir_label.setStyleSheet("color: #cfcfcf; background: #2a2a2a; padding: 6px; border-radius: 4px;")
        self.button_change_dir = QPushButton("📁 选择保存文件夹")
        self.button_change_dir.clicked.connect(self.change_save_dir)
        save_form.addRow("当前保存目录:", self.save_dir_label)
        save_form.addRow("操作:", self.button_change_dir)
        save_group.setLayout(save_form)

        # 主布局
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.video_label, stretch=1)
        main_layout.addLayout(controls_top)
        main_layout.addLayout(progress_layout)
        main_layout.addWidget(save_group)

        central = QWidget()
        central.setLayout(main_layout)
        self.setCentralWidget(central)

    # 更新裁剪分辨率显示
    def _update_crop_res_label(self, crop_w: int, crop_h: int):
        if self.crop_res_label is None:
            return
        if crop_w <= 0 or crop_h <= 0:
            self.crop_res_label.setText("裁剪分辨率：无")
        else:
            self.crop_res_label.setText(f"裁剪分辨率：{crop_w} × {crop_h} 像素")

    # 更新视频名称显示
    def _update_video_name_label(self, video_name: str):
        if self.video_name_label is not None:
            self.video_name_label.setText(f"当前视频: {video_name}")

    # 图片命名基准改变事件
    def _on_base_name_changed(self, text):
        self.base_name = text
        self.name_counter = 0  # 重置计数器

    # 格式化时间显示
    def _format_time(self, seconds: float) -> str:
        minutes = int(seconds // 60)
        seconds = int(seconds % 60)
        return f"{minutes:02d}:{seconds:02d}"

    # 更新进度条和时间显示
    def _update_progress(self):
        if self.capture is None or self.total_frames <= 0:
            return

        try:
            current_frame = int(self.capture.get(cv2.CAP_PROP_POS_FRAMES))
            # 防止进度条更新时触发跳转
            if not self.is_seeking:
                progress_value = int((current_frame / self.total_frames) * 1000)
                self.progress_slider.setValue(progress_value)

            # 更新时间显示
            current_time = current_frame / self.fps
            total_time = self.total_frames / self.fps
            self.current_time_label.setText(self._format_time(current_time))
            self.total_time_label.setText(self._format_time(total_time))
        except Exception:
            pass

    def _init_statusbar(self):
        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status.showMessage("🎬 准备就绪 - 点击 '打开视频' 选择文件 | 拖动鼠标选择裁剪区域")

    # 图片命名基准改变事件
    def _on_base_name_changed(self, text):
        self.base_name = text
        self.name_counter = 0  # 重置计数器

    def _init_shortcuts(self):
        QShortcut(QKeySequence(Qt.Key_Q), self, activated=self.save_current_frame)
        QShortcut(QKeySequence(Qt.Key_Space), self, activated=self.toggle_play)
        QShortcut(QKeySequence(Qt.Key_Escape), self, activated=self._cancel_crop_selection)

    def _apply_theme(self):
        palette = self.palette()
        palette.setColor(QPalette.Window, QColor(30, 30, 40))  # 深蓝灰色背景
        palette.setColor(QPalette.WindowText, QColor(245, 245, 250))  # 浅灰白色文字
        palette.setColor(QPalette.Base, QColor(35, 35, 45))  # 稍微浅一点的背景
        palette.setColor(QPalette.AlternateBase, QColor(45, 45, 55))  # 交替行背景
        palette.setColor(QPalette.ToolTipBase, QColor(255, 255, 230))  # 浅黄提示背景
        palette.setColor(QPalette.ToolTipText, QColor(10, 10, 20))  # 深色提示文字
        palette.setColor(QPalette.Text, QColor(245, 245, 250))  # 浅灰白色文字
        palette.setColor(QPalette.Button, QColor(50, 50, 65))  # 按钮背景
        palette.setColor(QPalette.ButtonText, QColor(245, 245, 250))  # 按钮文字
        palette.setColor(QPalette.BrightText, QColor(255, 100, 100))  # 鲜艳文字（错误提示）
        palette.setColor(QPalette.Highlight, QColor(90, 120, 200))  # 选中高亮（蓝紫色）
        palette.setColor(QPalette.HighlightedText, QColor(255, 255, 255))  # 高亮文字
        self.setPalette(palette)
        self.setStyleSheet("""
            QWidget { 
                color: #f5f5fa; 
                font-family: "Segoe UI", "Microsoft YaHei", sans-serif;
                font-size: 14px;
            }
            
            QMainWindow { 
                background: #1e1e28;  /* 深蓝灰色背景 */
            }
            
            QToolBar { 
                background: #252535;  /* 稍浅的蓝灰色 */
                spacing: 12px; 
                border-bottom: 1px solid #3a3a50; 
                padding: 10px;
            }
            
            QToolBar QToolButton { 
                color: #e0e0f0; 
                background: #3a3a50; 
                border: 1px solid #4a4a65; 
                border-radius: 8px; 
                padding: 8px 14px; 
                font-weight: 500;
            }
            
            QToolBar QToolButton:hover { 
                background: #4a4a65; 
                border-color: #5a5a80; 
                color: #ffffff;
            }
            
            QToolBar QToolButton:pressed { 
                background: #5a6aa0;  /* 蓝紫色 */
                border-color: #5a6aa0; 
                color: #ffffff; 
            }
            
            QPushButton { 
                color: #f0f0ff; 
                background: #3a3a50; 
                border: 1px solid #4a4a65; 
                border-radius: 10px; 
                padding: 10px 18px; 
                font-weight: 500;
                font-size: 14px;
            }
            
            QPushButton:hover { 
                background: #4a4a65; 
                border-color: #5a5a80; 
                color: #ffffff;
            }
            
            QPushButton:pressed { 
                background: #5a6aa0;  /* 蓝紫色 */
                border-color: #5a6aa0; 
                color: #ffffff; 
            }
            
            QPushButton:disabled { 
                background: #252535; 
                color: #8888a0; 
                border: 1px solid #353545; 
            }
            
            QComboBox { 
                color: #f0f0ff; 
                background: #353545; 
                border: 1px solid #4a4a65; 
                border-radius: 8px; 
                padding: 8px 12px; 
                min-width: 100px; 
                font-size: 14px;
            }
            
            QComboBox:hover { 
                background: #404055; 
                border-color: #505070;
            }
            
            QComboBox QAbstractItemView { 
                background: #2a2a3a; 
                color: #e8e8f0; 
                selection-background-color: #5a6aa0; 
                border: 1px solid #4a4a65;
                border-radius: 6px;
            }
            
            QGroupBox { 
                border: 1px solid #404055; 
                border-radius: 10px; 
                margin-top: 18px; 
                background: #252535;
                padding: 12px;
            }
            
            QGroupBox::title { 
                subcontrol-origin: margin; 
                left: 15px; 
                padding: 0 10px; 
                color: #c0c0e0; 
                font-weight: 600;
                font-size: 15px;
            }
            
            QStatusBar { 
                color: #c0c0e0; 
                background: #252535;
                border-top: 1px solid #3a3a50;
                padding: 6px;
            }
            
            QSlider::groove:horizontal { 
                border: 1px solid #4a4a65; 
                height: 12px; 
                background: #353545; 
                border-radius: 6px; 
            }
            
            QSlider::handle:horizontal { 
                background: #5a6aa0;  /* 蓝紫色 */
                border: 1px solid #4a5a90; 
                width: 18px; 
                height: 18px; 
                margin: -4px 0; 
                border-radius: 9px; 
            }
            
            QSlider::sub-page:horizontal { 
                background: #5a6aa0;  /* 蓝紫色 */
                border-radius: 6px; 
            }
            
            QLabel { 
                color: #f5f5fa; 
            }
            
            QFormLayout { 
                spacing: 12px; 
            }
        """)

    # 进度条按下事件
    def _on_slider_pressed(self):
        self.is_seeking = True
        # 记录拖动前的播放状态
        self.was_playing_before_seek = self.timer.isActive()
        # 如果正在播放，则暂停播放
        if self.was_playing_before_seek:
            self.pause()

    # 进度条释放事件
    def _on_slider_released(self):
        if self.capture is None or self.total_frames <= 0:
            self.is_seeking = False
            # 恢复播放状态
            if self.was_playing_before_seek:
                self.play()
            return
            
        # 获取进度条位置并跳转到对应帧
        position = self.progress_slider.value() / 1000.0
        target_frame = int(position * self.total_frames)
        
        try:
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
            
            # 读取并显示该帧
            ok, frame = self.capture.read()
            if ok and frame is not None:
                self.current_frame_bgr = frame
                self._show_frame(frame)
                
                # 更新时间显示
                current_time = target_frame / self.fps
                self.current_time_label.setText(self._format_time(current_time))
                # 更新进度条位置（确保同步）
                if not self.is_seeking:
                    progress_value = int((target_frame / self.total_frames) * 1000)
                    self.progress_slider.setValue(progress_value)
        except Exception:
            pass
            
        self.is_seeking = False
        
        # 如果拖动前在播放，则恢复播放
        if self.was_playing_before_seek:
            self.play()

    # 进度条拖动事件
    def _on_slider_moved(self, value):
        if self.capture is None or self.total_frames <= 0:
            return
            
        # 实时更新时间显示
        position = value / 1000.0
        target_frame = int(position * self.total_frames)
        current_time = target_frame / self.fps
        self.current_time_label.setText(self._format_time(current_time))
        
        # 实时预览功能：使用更智能的预览机制
        if not hasattr(self, '_last_preview_time'):
            self._last_preview_time = 0
            
        if not hasattr(self, '_last_preview_frame'):
            self._last_preview_frame = -1
            
        current_time = time.time()
        # 只有当帧位置变化较大或时间间隔较长时才进行预览
        frame_diff = abs(target_frame - self._last_preview_frame)
        # 如果帧差异超过总帧数的1%或时间超过300ms，则进行预览
        if frame_diff > max(1, self.total_frames // 100) or current_time - self._last_preview_time > 0.3:
            try:
                self.capture.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
                
                # 读取并显示该帧
                ok, frame = self.capture.read()
                if ok and frame is not None:
                    self.current_frame_bgr = frame
                    self._show_frame(frame)
                
                self._last_preview_time = current_time
                self._last_preview_frame = target_frame
            except Exception:
                pass

    def _update_controls_enabled(self, enabled: bool):
        self.button_play.setEnabled(enabled)
        self.button_pause.setEnabled(enabled)
        self.button_save.setEnabled(enabled)
        self.speed_combo.setEnabled(enabled)
        self.action_cancel_crop.setEnabled(enabled)

    # 取消裁剪区域
    def _cancel_crop_selection(self):
        if self.video_label and hasattr(self.video_label, "cancel_selection"):
            self.video_label.cancel_selection()
        self._update_crop_res_label(0, 0)
        if self.capture is not None:
            self.status.showMessage("✅ 已取消裁剪区域，恢复完整视频播放")
            # 重新显示当前帧以应用取消裁剪
            if self.current_frame_bgr is not None:
                self._show_frame(self.current_frame_bgr)

    def open_video(self):
        # 使用上次打开的文件夹路径，如果没有则使用默认路径
        if self.last_open_dir and Path(self.last_open_dir).exists():
            default_path = self.last_open_dir
        else:
            # 默认打开路径：E:\video\2025-10
            default_path = r"E:\video\2025-10"
            if not Path(default_path).exists():
                default_path = str(Path.home())
        
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择视频文件", default_path,
            "视频文件 (*.mp4 *.avi *.mov *.mkv *.wmv *.flv);;所有文件 (*.*)"
        )
        if not file_path:
            return

        # 保存当前打开的文件夹路径
        self.last_open_dir = str(Path(file_path).parent)

        self._load_video(file_path)
        self._cancel_crop_selection()  # 重置选框

    def _load_video(self, file_path: str):
        if self.capture is not None:
            try:
                self.pause()
                self.capture.release()
            except Exception:
                pass

        self.video_path = file_path
        self.capture = cv2.VideoCapture(file_path)
        # 更新视频名称显示
        self._update_video_name_label(os.path.basename(file_path))
        if not self.capture.isOpened():
            QMessageBox.critical(self, "无法打开", f"无法打开视频:\n{file_path}")
            self.capture = None
            return

        # 读取视频FPS
        fps = self.capture.get(cv2.CAP_PROP_FPS)
        try:
            fps_val = float(fps) if fps else 0.0
        except Exception:
            fps_val = 0.0
        if fps_val <= 1.0 or fps_val > 120.0:
            fps_val = 30.0
        self.fps = fps_val

        # 读取总帧数（先尝试直接获取，如果失败则异步估算，避免阻塞UI）
        frames = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if frames <= 0 or frames > 10_000_000:
            # 不在主线程执行耗时估算，改为在后台线程中运行
            self.status.showMessage("🎬 已打开: 正在异步估算总帧数... 请稍候")
            # 先使用0作为占位，界面会及时更新为估算结果
            self.total_frames = max(0, frames)

            # 启动后台线程进行估算（使用独立 VideoCapture）
            try:
                self._est_thread = EstimateFramesThread(file_path, max_iterations=20)
                self._est_thread.finished.connect(self._on_estimation_done)
                self._est_thread.start()
            except Exception:
                # 如果线程启动失败，则退回到保守估计
                self.total_frames = max(0, frames)
        else:
            self.total_frames = max(0, frames)

        video_name = os.path.basename(file_path)
        self.status.showMessage(
            f"🎬 已打开: {video_name} | FPS: {self.fps:.2f} | 🖱️ 拖动选择裁剪区域，Esc取消"
        )
        # 更新视频名称显示
        self._update_video_name_label(video_name)

        # 更新进度条范围和时间显示
        self.progress_slider.setRange(0, 1000)
        self.progress_slider.setValue(0)
        total_time = self.total_frames / self.fps if self.fps > 0 else 0
        self.total_time_label.setText(self._format_time(total_time))
        self.current_time_label.setText("00:00")

        self._apply_speed()
        self.ended = False
        self._update_controls_enabled(True)
        self.button_play.setEnabled(True)
        self.button_pause.setEnabled(False)

    # 异步估算完成后的回调（在主线程运行）
    def _on_estimation_done(self, frames: int):
        try:
            self.total_frames = max(0, int(frames))
            total_time = self.total_frames / self.fps if self.fps > 0 else 0
            self.total_time_label.setText(self._format_time(total_time))
            video_name = os.path.basename(self.video_path) if self.video_path else ""
            self.status.showMessage(
                f"🎬 已打开: {video_name} | FPS: {self.fps:.2f} | 帧数已估算: {self.total_frames}"
            )
        except Exception:
            pass

    def _apply_speed(self):
        frame_interval_ms = 1000.0 / max(1e-3, self.fps)
        speed = max(1e-3, self.speed)
        effective_interval = frame_interval_ms / speed
        interval_ms = max(1, int(round(effective_interval)))
        self.timer.setInterval(interval_ms)

    def _on_speed_changed(self, text: str):
        try:
            value = float(text[:-1]) if text.endswith('x') else float(text)
            self.speed = value
        except ValueError:
            self.speed = 1.0
        self._apply_speed()

    def play(self):
        if self.capture is None:
            return
        if self.ended:
            try:
                self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                # 重置进度条和时间显示
                self.progress_slider.setValue(0)
                self.current_time_label.setText("00:00")
            except Exception:
                pass
            self.ended = False
        self._apply_speed()
        if not self.timer.isActive():
            self.timer.start()
        self.button_play.setEnabled(False)
        self.button_pause.setEnabled(True)

    def pause(self):
        if self.timer.isActive():
            self.timer.stop()
        self.button_play.setEnabled(True)
        self.button_pause.setEnabled(False)

    def toggle_play(self):
        if self.timer.isActive():
            self.pause()
        else:
            self.play()

    def _read_frame(self):
        if self.capture is None:
            return
        ok, frame = self.capture.read()
        if not ok or frame is None:
            self.pause()
            self.ended = True
            self.status.showMessage("✅ 播放完成 - 点击 '播放' 重新播放 | 🖱️ 拖动选择裁剪区域，Esc取消")
            return

        self.current_frame_bgr = frame
        self._show_frame(frame)
        self._update_progress()

    def _show_frame(self, frame_bgr):
        frame_h, frame_w = frame_bgr.shape[:2]
        label_w, label_h = self.video_label.size().width(), self.video_label.size().height()
        crop_rect_rel = self.video_label.select_rect_rel

        # 根据选择的区域裁剪原始帧
        if crop_rect_rel is not None:
            rx1, ry1, rx2, ry2 = crop_rect_rel
            x1_frame = int(max(0, min(rx1 * frame_w, frame_w - 1)))
            y1_frame = int(max(0, min(ry1 * frame_h, frame_h - 1)))
            x2_frame = int(max(x1_frame + 1, min(rx2 * frame_w, frame_w)))
            y2_frame = int(max(y1_frame + 1, min(ry2 * frame_h, frame_h)))
            frame_bgr_cropped = frame_bgr[y1_frame:y2_frame, x1_frame:x2_frame]
        else:
            frame_bgr_cropped = frame_bgr

        # 转换并显示帧
        frame_rgb = cv2.cvtColor(frame_bgr_cropped, cv2.COLOR_BGR2RGB)
        crop_h, crop_w = frame_rgb.shape[:2]
        bytes_per_line = 3 * crop_w
        qimg = QImage(frame_rgb.data, crop_w, crop_h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        if label_w > 0 and label_h > 0:
            pixmap = pixmap.scaled(label_w, label_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.video_label.setPixmap(pixmap)
        
        # 保存缩放后的尺寸信息供区域选择使用
        if label_w > 0 and label_h > 0:
            scaled_w = pixmap.width()
            scaled_h = pixmap.height()
            # 计算图像在标签中的偏移量（居中显示时的边距）
            self.video_label.image_offset_x = (label_w - scaled_w) // 2
            self.video_label.image_offset_y = (label_h - scaled_h) // 2
            self.video_label.image_scaled_w = scaled_w
            self.video_label.image_scaled_h = scaled_h
        else:
            self.video_label.image_offset_x = 0
            self.video_label.image_offset_y = 0
            self.video_label.image_scaled_w = label_w
            self.video_label.image_scaled_h = label_h

    def _estimate_total_frames(self, max_iterations: int = 20) -> int:
        try:
            original_pos = int(self.capture.get(cv2.CAP_PROP_POS_FRAMES) or 0)
        except Exception:
            original_pos = 0

        def can_read(index: int) -> bool:
            try:
                self.capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
                ok, _ = self.capture.read()
                return bool(ok)
            except Exception:
                return False

        low, high = 0, 1
        it = 0
        while it < max_iterations and can_read(high):
            low = high
            high *= 2
            it += 1

        left, right = low, high
        last_ok = low if can_read(low) else 0
        while left <= right and it < max_iterations + 30:
            mid = (left + right) // 2
            if can_read(mid):
                last_ok = mid
                left = mid + 1
            else:
                right = mid - 1
            it += 1

        try:
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, original_pos)
        except Exception:
            pass
        return int(max(0, last_ok))

    def change_save_dir(self):
        directory = QFileDialog.getExistingDirectory(self, "选择保存文件夹", self.save_dir_label.text())
        if directory:
            self.save_dir_label.setText(directory)
            Path(directory).mkdir(parents=True, exist_ok=True)
            self.status.showMessage(f"📁 保存目录已更新: {directory}")

    def open_save_dir(self):
        """打开当前保存文件夹"""
        save_dir = self.save_dir_label.text().strip() or self.default_save_dir
        # 确保目录存在
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        # 使用系统默认方式打开文件夹
        try:
            if sys.platform == "win32":
                os.startfile(save_dir)
            elif sys.platform == "darwin":  # macOS
                subprocess.run(["open", save_dir], check=True)
            else:  # Linux
                subprocess.run(["xdg-open", save_dir], check=True)
            self.status.showMessage(f"📂 已打开保存文件夹: {save_dir}")
        except Exception as e:
            QMessageBox.critical(self, "打开失败", f"无法打开文件夹:\n{e}")

    # 保存格式改变事件
    def _on_save_format_changed(self, text):
        self.save_format = text

    def save_current_frame(self):
        if self.current_frame_bgr is None:
            QMessageBox.information(self, "无可保存帧", "当前没有可保存的帧。")
            return

        save_dir = self.save_dir_label.text().strip() or self.default_save_dir
        Path(save_dir).mkdir(parents=True, exist_ok=True)

        # 裁剪帧（与显示区域一致）
        frame_h, frame_w = self.current_frame_bgr.shape[:2]
        crop_rect_rel = self.video_label.select_rect_rel
        if crop_rect_rel is not None:
            rx1, ry1, rx2, ry2 = crop_rect_rel
            x1_frame = int(max(0, min(rx1 * frame_w, frame_w - 1)))
            y1_frame = int(max(0, min(ry1 * frame_h, frame_h - 1)))
            x2_frame = int(max(x1_frame + 1, min(rx2 * frame_w, frame_w)))
            y2_frame = int(max(y1_frame + 1, min(ry2 * frame_h, frame_h)))
            frame_to_save = self.current_frame_bgr[y1_frame:y2_frame, x1_frame:x2_frame]
            crop_suffix = f"_crop_{x2_frame-x1_frame}x{y2_frame-y1_frame}"
        else:
            frame_to_save = self.current_frame_bgr
            crop_suffix = ""

        # 生成文件名并保存
        base_name = self.base_name
        # 使用计数器生成序号
        counter = self.name_counter
        self.name_counter += 1  # 递增计数器
        
        # 使用指定格式生成文件名
        filename = f"{base_name}_{counter:03d}.{self.save_format}"
        full_path = os.path.join(save_dir, filename)

        # 使用指定格式保存图片
        try:
            if self.save_format.lower() in ['jpg', 'jpeg']:
                # For JPG format, we need to keep the BGR format for OpenCV
                # OpenCV's imwrite expects BGR format even for JPG files
                success = cv2.imwrite(full_path, frame_to_save, [cv2.IMWRITE_JPEG_QUALITY, 95])
            else:
                # 其他格式直接保存
                success = cv2.imwrite(full_path, frame_to_save)
            
            if success:
                save_w, save_h = frame_to_save.shape[1], frame_to_save.shape[0]
                self.status.showMessage(f"✅ 已保存帧: {full_path} | 尺寸：{save_w}x{save_h}像素 | 格式：{self.save_format.upper()}")
            else:
                QMessageBox.critical(self, "保存失败", f"无法保存图片到:\n{full_path}")
        except Exception as e:
            QMessageBox.critical(self, "保存失败", f"保存失败:\n{e}")

    def closeEvent(self, event):
        try:
            if self.timer.isActive():
                self.timer.stop()
            if self.capture is not None:
                self.capture.release()
        except Exception:
            pass
        event.accept()


def main():
    app = QApplication(sys.argv)
    window = VideoPlayer()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
